# D&D Project — Shared JSON Schema (v1)

This is the contract between **Part 1 (text → JSON)** and **Part 2 (JSON → text)**.
Whatever Part 1 produces must conform to this shape so Part 2 can consume it,
and the preprocessed FIREBALL dataset will match this shape so Part 2 can train on it.

## Design principles

1. **Flat and small.** Only fields that actually show up in FIREBALL and that a D&D
   narration needs. Nothing aspirational.
2. **Optional almost everything.** Player text is messy. If a field is missing,
   generator should still produce plausible narration.
3. **One turn = one record.** A "turn" is a single mechanical action
   (one attack / one spell / one check). Multi-action turns are split.

## Top-level object

```jsonc
{
  "turn_id":        "string",            // unique id, e.g. "combat42_turn7"
  "action_type":    "attack|spell|check|save|heal|other",
  "actor":          Actor,                // who did the thing
  "targets":        [Target],             // 0+ targets (0 for self-buff / check)
  "mechanics":      Mechanics,            // the dice / numbers
  "context":        Context,              // chat history, whose turn
  "narration":      "string|null"         // GOLD output — null at inference time
}
```

## Sub-schemas

### Actor (normalized from FIREBALL `caster_after`)

```jsonc
{
  "name":      "string",         // "Bob the Bard"
  "class":     "string|null",    // "Fighter 3"  (class + level if available)
  "race":      "string|null",    // "Mountain Dwarf"
  "hp_current":"int|null",
  "hp_max":    "int|null",
  "status":    ["string"]        // ["Bloodied", "Poisoned"] — from FIREBALL effects
}
```

### Target

```jsonc
{
  "name":      "string",
  "hp_before": "int|null",
  "hp_after":  "int|null",
  "ac":        "int|null",
  "killed":    "bool"            // derived: hp_after <= 0 && hp_before > 0
}
```

### Mechanics

```jsonc
{
  "roll":      { "kind": "attack|save|check|damage", "total": "int|null", "dc_or_ac": "int|null", "hit": "bool|null", "crit": "bool" } | null,
  "damage":    [{ "amount": "int", "type": "string" }],   // e.g. [{"amount":8,"type":"slashing"}]
  "spell":     "string|null",    // "Fireball", null if attack
  "weapon":    "string|null",    // "longsword", null if spell
  "raw_results":"string|null"    // raw Avrae automation_results text (for fallback/debug)
}
```

### Context

```jsonc
{
  "recent_utterances": ["string"],   // up to 5 preceding player chat lines
  "current_turn_actor":"string|null" // who Avrae says is currently "up"
}
```

## Example (from a real FIREBALL-like triple)

```json
{
  "turn_id": "abc123_turn042",
  "action_type": "attack",
  "actor": {
    "name": "Thoradin Ironfoot",
    "class": "Fighter 4",
    "race": "Mountain Dwarf",
    "hp_current": 28,
    "hp_max": 36,
    "status": []
  },
  "targets": [{
    "name": "Goblin Scout",
    "hp_before": 11,
    "hp_after": 3,
    "ac": 13,
    "killed": false
  }],
  "mechanics": {
    "roll": {"kind": "attack", "total": 18, "dc_or_ac": 13, "hit": true, "crit": false},
    "damage": [{"amount": 8, "type": "slashing"}],
    "spell": null,
    "weapon": "longsword",
    "raw_results": "[ATK] Thoradin rolls an attack. To Hit: 18\n[HIT] Thoradin hits Goblin Scout for 8 slashing damage."
  },
  "context": {
    "recent_utterances": [
      "I step into the goblin's reach and swing my longsword low.",
      "Going for a cleave, aiming at the knees."
    ],
    "current_turn_actor": "Thoradin Ironfoot"
  },
  "narration": "Thoradin plants his boots on the cracked flagstone and drives the longsword in a tight arc. Steel bites deep into the goblin's thigh — it reels, lifeblood pouring, barely keeping its feet."
}
```

## Linearization for models

Each model in Part 2 serializes this JSON differently. **Keep the raw JSON the same;
only the linearization changes per model.**

| Model | Linearized input | Notes |
|---|---|---|
| Template baseline | (uses JSON fields directly) | no tokenization |
| N-gram LM (Week 2) | `<ACT>attack<ACTOR>Thoradin<TGT>Goblin Scout<ROLL>18<AC>13<HIT>1<DMG>8_slashing<NARR>` | special tokens as prefix, then generate after `<NARR>` |
| T5/BART seq2seq (Week 8) | `"action: attack \| actor: Thoradin (F4 dwarf, 28/36 hp) \| target: Goblin Scout (11→3/11 hp, ac 13) \| roll: 18 vs 13 hit \| damage: 8 slashing \| weapon: longsword \| recent: <utterance history>"` | natural-language-ish, fits T5's text-to-text pretraining |
| LLM few-shot (Week 9, optional) | same as T5 linearization, with 3-5 examples in prompt | |

## Train/dev/test split

Split by **combat_id** (not by turn) to avoid leakage — turns from the same
combat session stay together.

- Train: 90%
- Dev:   5%
- Test:  5%

## What Part 1 owes Part 2

Part 1's output JSON **must validate against this schema**. If Part 1 cannot
extract a field, set it to `null` / `[]` rather than omit it. Part 2 handles
missing fields.

At inference, Part 1's output will not have a `narration` field — Part 2 fills
that in.

## What Part 2 owes Part 1

Part 2 consumes whatever Part 1 produces. At training time, Part 2 uses
preprocessed FIREBALL data (same schema, with `narration` populated from
FIREBALL's `after_utterances`).
