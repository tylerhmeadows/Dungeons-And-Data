# Part 2 — JSON → DM narration

This folder contains everything Part 2 needs to preprocess the FIREBALL dataset
and train three narration models for the CS4120 final report.

## Files

| File | What it is |
|---|---|
| `schema.md` | **Shared JSON schema** — the contract between Part 1 and Part 2 |
| `fireball_preprocess.py` | Core Python module: parses FIREBALL triples into our schema, plus `linearize_for_t5` / `linearize_for_ngram` |
| `01_preprocess_fireball.ipynb` | Downloads FIREBALL, produces `train/dev/test.jsonl` |
| `02_train_ngram_lm.ipynb` | **Week-2 baseline** — trigram KN LM conditioned on tag prefix |
| `03_train_t5_seq2seq.ipynb` | **Week-8 baseline** — fine-tune `t5-small` seq2seq |
| `04_evaluate.ipynb` | Apples-to-apples BLEU / ROUGE-L / entity-faithfulness across Template / N-gram / T5 |
| `processed/` | Output of `01` — `train.jsonl` (3,533) / `dev.jsonl` (216) / `test.jsonl` (216) / `stats.json` |
| `raw/fireball_150combats_with_ids.jsonl` | 150-combat FIREBALL sample (15,583 triples, ~210 MB) used to produce `processed/` |
| `sample_data/sample_records.jsonl` | 5 processed examples from synthetic input — reference for what output looks like |

## Current dataset stats (from `processed/stats.json`)

```
lines_read     : 15,583
records_kept   :  3,965   (after quality + OOC-chat filter)
split (combat) : train 3,533 / dev 216 / test 216
unique combats : 150
actions        : attack 2,759 · spell 1,003 · save 203
```

## Colab run order

1. **`01_preprocess_fireball.ipynb`** — produces `/content/processed/{train,dev,test}.jsonl`. ~10 min on free Colab.
   *Optional for teammates:* skip this and just upload the `processed/` folder that's already in the repo.
2. **`02_train_ngram_lm.ipynb`** — trains KN-3 LM, writes `models/ngram/predictions_test.jsonl`. ~2 min, CPU is fine.
3. **`03_train_t5_seq2seq.ipynb`** — fine-tunes `t5-small`. **Runtime → Change runtime type → T4 GPU**. ~20-25 min.
4. **`04_evaluate.ipynb`** — reads both prediction files, builds comparison table + qualitative side-by-side, exports `results_comparison.{csv,md}` for the report.

## What the preprocessing pipeline does

1. Streams `filtered_triples.jsonl` line-by-line (~150k triples, but we only have 150 combats = 15k triples).
2. For each triple:
   - Classify the action by inspecting `commands_norm[0]` (with fallback to automation-result keywords for `!i`, `!status`, etc.).
   - Parse `automation_results` for roll totals, hit/miss, crit, and typed damage.
   - Parse HP strings (`"<12/34; Bloodied>"` and `"<53/53 HP; Healthy>"` both work).
   - Derive `killed` from `hp_before > 0` and `hp_after <= 0`.
   - Join `after_utterances` as the gold `narration`; strip Discord markdown/URLs.
3. Filter records that are:
   - Too short (<6 words) or too long (>100 words)
   - Missing actor name
   - `action_type == "other"` with no rolls/damage
   - OOC chat (XD / lol / @mention / `:emoji:` / heavy dice notation)
4. Split by `combat_id` (**not** by turn) — 90/5/5 — so sessions never leak across splits.

## Linearization helpers

`fireball_preprocess.py` exports two functions that turn a schema record into
model-ready strings. All three models consume the *same* preprocessed JSON.

- `linearize_for_t5(record)` → human-readable prefix for T5/BART
  ```
  narrate | action: attack | actor: Thoradin (Fighter 4, Mountain Dwarf, 28/36 hp)
  | target: Goblin Scout (11→3 hp, ac 13) | roll: 18 vs 13 hit | damage: 8 slashing | weapon: longsword
  ```
- `linearize_for_ngram(record)` → tag-prefixed tokens for N-gram conditioning
  ```
  <ACT>attack <ACTOR>Thoradin <TGT>Goblin_Scout <ROLL>18 <VS>13 <HIT> <DMG>8_slashing <WPN>longsword <NARR>
  ```

## Contract with Part 1

Part 1's text → JSON output **must conform to `schema.md`** with `narration: null`.
Part 2 inference will fill in `narration`.

If Part 1 can't extract some fields, set them to `null` / `[]` — don't omit keys.
Part 2 is robust to missing fields (template baseline degrades gracefully; T5 still generates plausible text).

## For the write-up

After running `04_evaluate.ipynb`, read `results_comparison.md` for the headline
numbers. Expected patterns to discuss:

1. **Template** wins entity faithfulness (slot-filled) but loses BLEU — no flavor text.
2. **T5** wins BLEU/ROUGE — it learns FIREBALL idiom — but hallucinates names.
3. **KN-3** trails T5 on BLEU but often produces shorter, more on-topic text than
   template baseline for actions the LM has seen enough of.

**Limitation worth stating:** FIREBALL's gold narrations are player-authored
Discord chat. ~10-15% of records have residual mechanical/OOC noise even after
filtering; this caps BLEU/ROUGE upper bounds.
