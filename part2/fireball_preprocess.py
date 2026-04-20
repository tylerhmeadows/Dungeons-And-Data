"""
FIREBALL preprocessing for the D&D Part 2 (JSON -> text) task.

Reads FIREBALL's `filtered_triples.jsonl` (~150k triples) and emits our own
cleaner JSONL where each record matches the schema in schema.md.

Usage (CLI):
    python fireball_preprocess.py \\
        --input filtered_triples.jsonl \\
        --out-dir ./processed \\
        --max-records 0        # 0 = no cap

Usage (from Colab notebook):
    from fireball_preprocess import run_pipeline
    run_pipeline("/content/filtered_triples.jsonl", "/content/processed", max_records=50000)

Free-Colab friendly: streams the input line-by-line; never holds more than a
few records in memory at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

# -------------------------------------------------------------------------
# Regexes for Avrae automation_results strings
# -------------------------------------------------------------------------
# Examples the parser handles:
#   [ATK] Thoradin rolls an attack. To Hit: 18
#   [HIT] Thoradin hits Goblin Scout for 8 slashing damage.
#   [MISS] Thoradin misses Goblin Scout.
#   [SAVE] Goblin Scout rolls a DEX save. Result: 7 vs DC 15. Failed.
#   [DAM] 8 slashing damage.
# Formats vary across years of Avrae. We're permissive.

RE_TO_HIT      = re.compile(r"[Tt]o\s*[Hh]it\s*[:=]\s*(\d+)")
# Covers: "Val attacked ON2 and hit." / "Val attacked ON2 and missed."
RE_ATTACK_VS   = re.compile(r"attacked\s+[^.\n]+?\s+and\s+(hit|miss(?:ed)?)", re.IGNORECASE)
RE_DC          = re.compile(r"(?:DC|vs\s*DC|vs\.\s*DC)\s*(\d+)")
RE_AC          = re.compile(r"(?:AC|vs\s*AC|vs\.\s*AC)\s*(\d+)")
RE_DAMAGE      = re.compile(
    r"(\d+)\s+(acid|bludgeoning|cold|fire|force|lightning|necrotic|piercing|poison|psychic|radiant|slashing|thunder)",
    re.IGNORECASE,
)
# "ON2 took 17 damage." — no element specified
RE_TOOK_DMG    = re.compile(r"took\s+(\d+)\s+damage", re.IGNORECASE)
RE_HIT_LINE    = re.compile(r"\b(HIT|hits?)\b")
RE_MISS_LINE   = re.compile(r"\b(MISS|misses?|missed)\b")
RE_CRIT_LINE   = re.compile(r"\b(CRIT|critical)\b", re.IGNORECASE)
RE_KILLED      = re.compile(r"\b(dies|down|unconscious|dropped|killed)\b", re.IGNORECASE)

# HP strings FIREBALL actually contains (both v1 and v2 formats):
#   "<12/34; Bloodied>"
#   "<53/53 HP; Healthy>"
#   "<0/10 HP>"
#   "12/34"
RE_HP          = re.compile(r"<?\s*(\d+)\s*/\s*(\d+)\s*(?:HP)?\s*(?:;\s*([^>]+?))?\s*>?\s*$")

# Command parsing
# e.g. "!a longsword -t goblin", "!attack", "!cast fireball -t goblin1 -t goblin2",
#      "!check perception", "!save dex -dc 15", "!g attack sword"
RE_CMD_WORD    = re.compile(r"^!(\w+)")

ATTACK_CMDS = {"a", "attack", "ma", "multiattack", "monattack", "ra", "action"}
SPELL_CMDS  = {"cast", "c", "mcast", "moncast"}
CHECK_CMDS  = {"check", "ch", "rc"}
SAVE_CMDS   = {"save", "s", "rs"}
HEAL_CMDS   = {"heal", "g heal", "gheal"}
# Commands we want to skip entirely: initiative, status, lookup
NOOP_CMDS   = {"i", "init", "ir", "status", "g", "game", "combat", "cb", "lookup", "r", "roll"}


# -------------------------------------------------------------------------
# Narration cleaning + quality heuristics
# -------------------------------------------------------------------------
# Strip Discord/markdown noise but keep the prose.
RE_DISCORD_MD = re.compile(r"[*_`~]+")        # bold/italic/code markers
RE_URL        = re.compile(r"https?://\S+")
RE_DICE_ROLL  = re.compile(r"\b\d*d\d+(?:\s*[+\-]\s*\d+)?\b", re.IGNORECASE)  # "1d20+5"
RE_MULTISPACE = re.compile(r"\s+")

# Signals that this is out-of-character chat rather than narration.
OOC_PATTERNS = [
    re.compile(r"\bXD\b|\blol\b|\blmao\b|\brofl\b", re.IGNORECASE),
    re.compile(r"^[A-Z][a-zA-Z\s]+:\s*\*"),      # "Summer: *is 500 feet away*"
    re.compile(r"\bDM\b\s*:", re.IGNORECASE),    # "DM: rolls are ..."
    re.compile(r"@\w+"),                         # @mentions
    re.compile(r":\w+:"),                        # :emoji:
]

def clean_narration(text: str) -> str:
    """Strip common Discord artifacts but preserve the storytelling prose."""
    if not text:
        return ""
    t = RE_URL.sub("", text)
    t = RE_DISCORD_MD.sub("", t)
    t = RE_MULTISPACE.sub(" ", t).strip()
    return t


def narration_looks_ooc(text: str, actor_name: str = "", target_names: list[str] = ()) -> bool:
    """Rough heuristic — True if the text smells like OOC chat rather than narration.

    We intentionally do NOT require the actor/target name to appear in the text,
    because D&D narration commonly uses pronouns ('she fires another shot') or
    role nouns ('the guard swipes with his glaive').
    """
    if not text:
        return True
    for pat in OOC_PATTERNS:
        if pat.search(text):
            return True
    # Heavy dice-roll notation is usually mechanical chat.
    if len(RE_DICE_ROLL.findall(text)) >= 2:
        return True
    return False


# -------------------------------------------------------------------------
# Parsers
# -------------------------------------------------------------------------
def classify_action(commands_norm: list[str], automation_text: str = "") -> str:
    """Return one of attack|spell|check|save|heal|other from commands and raw Avrae output.

    Priority: explicit command verb > automation_results keywords > 'other'.
    """
    for cmd in commands_norm:
        m = RE_CMD_WORD.match(cmd.strip())
        if not m:
            continue
        word = m.group(1).lower()
        if word in ATTACK_CMDS: return "attack"
        if word in SPELL_CMDS:  return "spell"
        if word in CHECK_CMDS:  return "check"
        if word in SAVE_CMDS:   return "save"
        if word in HEAL_CMDS:   return "heal"
        # skip NOOP_CMDS and keep looking, they're bookkeeping
    # Fall back to automation text sniffing for cases like !i where the event is still a roll.
    t = (automation_text or "").lower()
    if "attacked " in t or "to hit" in t or "[atk]" in t:
        return "attack"
    if "heal" in t and "hp" in t:
        return "heal"
    if "save" in t:
        return "save"
    if "check" in t or "skill" in t:
        return "check"
    if "damage" in t or "spell" in t:
        return "spell"
    return "other"


def extract_weapon_or_spell(commands_norm: list[str], action_type: str) -> tuple[str | None, str | None]:
    """Return (weapon, spell). One is always None given action_type."""
    if not commands_norm:
        return None, None
    first = commands_norm[0].strip()
    # Strip the "!word" prefix
    body = re.sub(r"^!\S+\s*", "", first)
    # Strip trailing flags ("-t goblin -rr 2")
    body = re.split(r"\s+-\w", body)[0].strip()
    if not body:
        return None, None
    if action_type == "spell":
        return None, body
    if action_type == "attack":
        return body, None
    return None, None


def parse_automation(automation_results: list[str]) -> dict[str, Any]:
    """Extract roll total, hit/miss, crit, damage list from raw Avrae output."""
    text = "\n".join(automation_results or [])
    out: dict[str, Any] = {
        "roll_total": None,
        "dc_or_ac":   None,
        "hit":        None,
        "crit":       False,
        "damage":     [],
        "raw":        text if text else None,
    }
    if not text:
        return out

    m = RE_TO_HIT.search(text)
    if m:
        out["roll_total"] = int(m.group(1))
    m = RE_AC.search(text) or RE_DC.search(text)
    if m:
        out["dc_or_ac"] = int(m.group(1))

    # Hit/miss detection — handle multiple Avrae formats
    # Format 1: "[HIT]" / "[MISS]" tagged lines
    # Format 2: "Val attacked ON2 and hit." / "and missed."
    m_av = RE_ATTACK_VS.search(text)
    if m_av:
        verb = m_av.group(1).lower()
        out["hit"] = verb.startswith("hit")
    elif RE_HIT_LINE.search(text):
        out["hit"] = True
    elif RE_MISS_LINE.search(text):
        out["hit"] = False

    if RE_CRIT_LINE.search(text):
        out["crit"] = True

    # Typed damage first (preferred, has element type)
    typed_dmg = RE_DAMAGE.findall(text)
    if typed_dmg:
        for amt, dtype in typed_dmg:
            out["damage"].append({"amount": int(amt), "type": dtype.lower()})
    else:
        # Fallback: "X took N damage." without element type
        for amt in RE_TOOK_DMG.findall(text):
            out["damage"].append({"amount": int(amt), "type": "unspecified"})

    return out


def parse_hp(hp_str: str | None) -> tuple[int | None, int | None, list[str]]:
    """'<12/34; Bloodied>' -> (12, 34, ['Bloodied'])."""
    if not hp_str:
        return None, None, []
    m = RE_HP.search(hp_str)
    if not m:
        return None, None, []
    cur = int(m.group(1))
    mx  = int(m.group(2))
    status_raw = (m.group(3) or "").strip()
    status = [s.strip() for s in status_raw.split(",") if s.strip()] if status_raw else []
    return cur, mx, status


def _as_list(v) -> list[str]:
    """FIREBALL stores list-like fields as either list[str] or comma-joined str."""
    if v is None: return []
    if isinstance(v, list): return [x for x in v if x]
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


def normalize_actor(caster: dict | None) -> dict:
    if not caster:
        return {
            "name": "", "class": None, "race": None,
            "hp_current": None, "hp_max": None, "status": [],
        }
    cur, mx, status = parse_hp(caster.get("hp"))
    effects = _as_list(caster.get("effects"))
    status = list(dict.fromkeys([*status, *effects]))  # dedup preserving order
    return {
        "name":      caster.get("name", "") or "",
        "class":     caster.get("class") or None,
        "race":      caster.get("race") or None,
        "hp_current":cur,
        "hp_max":    mx,
        "status":    status,
    }


def normalize_target(before_combatant: dict | None, after_combatant: dict) -> dict:
    cur_b, _, _      = parse_hp((before_combatant or {}).get("hp")) if before_combatant else (None, None, [])
    cur_a, mx_a, _   = parse_hp(after_combatant.get("hp"))
    killed = bool(cur_b is not None and cur_a is not None and cur_b > 0 and cur_a <= 0)
    # AC isn't in the normalized actor state directly; often shown in description
    ac = None
    desc = after_combatant.get("description", "") or ""
    m = re.search(r"AC\s*[:=]?\s*(\d+)", desc)
    if m:
        ac = int(m.group(1))
    return {
        "name":      after_combatant.get("name", "") or "",
        "hp_before": cur_b,
        "hp_after":  cur_a,
        "ac":        ac,
        "killed":    killed,
    }


# -------------------------------------------------------------------------
# Main triple -> record
# -------------------------------------------------------------------------
def triple_to_record(triple: dict, idx: int) -> dict | None:
    """Convert one FIREBALL triple to our schema, or None if it's unusable."""
    commands = triple.get("commands_norm") or []
    after_utts = triple.get("after_utterances") or []
    narration = " ".join(u.strip() for u in after_utts if u and u.strip())
    narration = clean_narration(narration)
    if not narration:
        return None  # no gold text -> useless for Part 2

    auto = parse_automation(triple.get("automation_results") or [])
    action_type = classify_action(commands, auto.get("raw") or "")
    weapon, spell = extract_weapon_or_spell(commands, action_type)

    caster_after = triple.get("caster_after") or triple.get("current_actor") or {}
    actor = normalize_actor(caster_after)

    # targets: match before/after by name
    before_map = {c.get("name"): c for c in (triple.get("combat_state_before") or []) if c.get("name")}
    targets_after = triple.get("targets_after") or []
    targets = [normalize_target(before_map.get(t.get("name")), t) for t in targets_after]

    # If automation didn't yield a dc_or_ac but this was an attack on a target with AC,
    # backfill from the first target's AC — common in FIREBALL where Avrae formats "vs AC" inconsistently.
    if auto["dc_or_ac"] is None and action_type == "attack" and targets:
        t_ac = next((t["ac"] for t in targets if t.get("ac") is not None), None)
        if t_ac is not None:
            auto["dc_or_ac"] = t_ac

    # Context
    context = {
        "recent_utterances": (triple.get("utterance_history") or [])[-5:],
        "current_turn_actor": (triple.get("current_actor") or {}).get("name"),
    }

    # Stable turn_id
    speaker = triple.get("speaker_id", "unknown")
    seed = f"{speaker}|{idx}|{narration[:32]}"
    turn_id = hashlib.sha1(seed.encode()).hexdigest()[:12]

    return {
        "turn_id": turn_id,
        "action_type": action_type,
        "actor": actor,
        "targets": targets,
        "mechanics": {
            "roll":   {
                "kind":     action_type if action_type in ("attack","save","check","damage") else "attack",
                "total":    auto["roll_total"],
                "dc_or_ac": auto["dc_or_ac"],
                "hit":      auto["hit"],
                "crit":     auto["crit"],
            } if (auto["roll_total"] or auto["hit"] is not None) else None,
            "damage": auto["damage"],
            "spell":  spell,
            "weapon": weapon,
            "raw_results": auto["raw"],
        },
        "context": context,
        "narration": narration,
        # provenance (for split-by-combat; stripped from final record if desired)
        "_combat_id": triple.get("combat_id") or speaker,
    }


# -------------------------------------------------------------------------
# Filtering
# -------------------------------------------------------------------------
def is_good_record(r: dict,
                   min_narr_words: int = 6,
                   max_narr_words: int = 100) -> bool:
    narr = r["narration"]
    n = len(narr.split())
    if n < min_narr_words or n > max_narr_words:
        return False
    # Must have an actor name
    actor_name = r["actor"]["name"]
    if not actor_name:
        return False
    # Must have either a roll or damage or an action_type != other
    if r["action_type"] == "other" and not r["mechanics"]["damage"] and not r["mechanics"]["roll"]:
        return False
    # OOC / Discord chat filter — keep only text that plausibly narrates this action
    target_names = [t["name"] for t in r.get("targets", []) if t.get("name")]
    if narration_looks_ooc(narr, actor_name, target_names):
        return False
    return True


# -------------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------------
def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    max_records: int = 0,
    train_frac: float = 0.90,
    dev_frac:   float = 0.05,
    seed: int = 4120,
    keep_combat_id: bool = False,
) -> dict:
    """Stream through filtered_triples.jsonl, emit train/dev/test JSONL.

    Splits by combat_id so turns from the same session don't leak across splits.
    """
    input_path = Path(input_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Pass 1: stream, convert, filter, collect records
    action_counts = Counter()
    kept: list[dict] = []
    seen = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            try:
                triple = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = triple_to_record(triple, idx=seen)
            if rec is None:
                continue
            if not is_good_record(rec):
                continue
            kept.append(rec)
            action_counts[rec["action_type"]] += 1
            if max_records and len(kept) >= max_records:
                break

    # Split by combat_id
    combat_ids = sorted({r["_combat_id"] for r in kept})
    rng.shuffle(combat_ids)
    n = len(combat_ids)
    n_tr = int(n * train_frac)
    n_dv = int(n * dev_frac)
    train_ids = set(combat_ids[:n_tr])
    dev_ids   = set(combat_ids[n_tr:n_tr + n_dv])
    # remaining -> test

    splits = {"train": [], "dev": [], "test": []}
    for r in kept:
        cid = r["_combat_id"]
        if cid in train_ids:   splits["train"].append(r)
        elif cid in dev_ids:   splits["dev"].append(r)
        else:                  splits["test"].append(r)

    # Write
    for split_name, records in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                if not keep_combat_id:
                    r = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "lines_read":       seen,
        "records_kept":     len(kept),
        "split_counts":     {k: len(v) for k, v in splits.items()},
        "action_counts":    dict(action_counts),
        "unique_combats":   n,
    }
    with (out_dir / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    return stats


# -------------------------------------------------------------------------
# Linearization helpers (used by model training, not preprocessing)
# -------------------------------------------------------------------------
def linearize_for_t5(record: dict, include_history: bool = True) -> str:
    """Human-readable text prefix for T5/BART."""
    a = record["actor"]
    actor_str = a["name"]
    meta = []
    if a.get("class"): meta.append(a["class"])
    if a.get("race"):  meta.append(a["race"])
    if a.get("hp_current") is not None and a.get("hp_max") is not None:
        meta.append(f"{a['hp_current']}/{a['hp_max']} hp")
    if meta: actor_str += f" ({', '.join(meta)})"

    target_strs = []
    for t in record["targets"]:
        ts = t["name"]
        bits = []
        if t.get("hp_before") is not None and t.get("hp_after") is not None:
            bits.append(f"{t['hp_before']}→{t['hp_after']} hp")
        if t.get("ac") is not None:
            bits.append(f"ac {t['ac']}")
        if t.get("killed"):
            bits.append("killed")
        if bits: ts += f" ({', '.join(bits)})"
        target_strs.append(ts)

    mech = record["mechanics"]
    bits = [f"action: {record['action_type']}",
            f"actor: {actor_str}"]
    if target_strs:
        bits.append("target: " + "; ".join(target_strs))
    if mech.get("roll"):
        r = mech["roll"]
        rs = f"roll: {r.get('total')}"
        if r.get("dc_or_ac") is not None: rs += f" vs {r['dc_or_ac']}"
        if r.get("hit") is True:          rs += " hit"
        elif r.get("hit") is False:       rs += " miss"
        if r.get("crit"):                 rs += " CRIT"
        bits.append(rs)
    if mech.get("damage"):
        bits.append("damage: " + ", ".join(f"{d['amount']} {d['type']}" for d in mech["damage"]))
    if mech.get("spell"):   bits.append(f"spell: {mech['spell']}")
    if mech.get("weapon"):  bits.append(f"weapon: {mech['weapon']}")
    if include_history and record["context"].get("recent_utterances"):
        hist = " / ".join(record["context"]["recent_utterances"][-3:])
        bits.append(f"recent: {hist}")
    return "narrate | " + " | ".join(bits)


def linearize_for_ngram(record: dict) -> str:
    """Tag-prefixed token string for N-gram conditioning.
    Format: <ACT>...<ACTOR>...<NARR> narration_text </NARR>
    For training, append narration after <NARR>. For inference, stop at <NARR>.
    """
    a = record["actor"]
    mech = record["mechanics"]
    toks = [f"<ACT>{record['action_type']}",
            f"<ACTOR>{a['name']}"]
    for t in record["targets"]:
        toks.append(f"<TGT>{t['name']}")
        if t.get("killed"): toks.append("<KILLED>")
    if mech.get("roll"):
        r = mech["roll"]
        if r.get("total") is not None:   toks.append(f"<ROLL>{r['total']}")
        if r.get("dc_or_ac") is not None:toks.append(f"<VS>{r['dc_or_ac']}")
        if r.get("hit") is True:         toks.append("<HIT>")
        elif r.get("hit") is False:      toks.append("<MISS>")
        if r.get("crit"):                toks.append("<CRIT>")
    for d in mech.get("damage") or []:
        toks.append(f"<DMG>{d['amount']}_{d['type']}")
    if mech.get("spell"):  toks.append(f"<SPELL>{mech['spell']}")
    if mech.get("weapon"): toks.append(f"<WPN>{mech['weapon']}")
    toks.append("<NARR>")
    return " ".join(toks)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to FIREBALL filtered_triples.jsonl")
    p.add_argument("--out-dir", default="./processed")
    p.add_argument("--max-records", type=int, default=0, help="Cap total records (0 = no cap)")
    p.add_argument("--train-frac", type=float, default=0.90)
    p.add_argument("--dev-frac",   type=float, default=0.05)
    p.add_argument("--seed",       type=int,   default=4120)
    args = p.parse_args()

    stats = run_pipeline(
        input_path=args.input,
        out_dir=args.out_dir,
        max_records=args.max_records,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _main()
