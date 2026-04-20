# Part 2 — Results So Far

Test set: 216 records (150 combats, no leakage). Ran locally on Apple M2.

## Metrics

| Model       | BLEU-4 | ROUGE-L | Actor % | Target % | Len ratio |
|-------------|-------:|--------:|--------:|---------:|----------:|
| Template    |   0.09 |    6.33 |   100.0 |     95.9 |      0.42 |
| KN-3 N-gram |   0.22 |    8.63 |     3.2 |      1.5 |      0.51 |
| T5-small    |    —   |      —  |      —  |       —  |        —  |

**KN-3 perplexity:** dev 2993, test 2416 (vocab 15,394).

**Per-action BLEU / ROUGE-L:**

| Action | n   | Template BLEU/RL | KN-3 BLEU/RL |
|--------|----:|-----------------:|-------------:|
| attack | 158 |      0.12 / 7.29 |  0.28 / 8.88 |
| spell  |  50 |      0.15 / 3.38 |  0.18 / 8.30 |
| save   |   8 |      0.01 / 5.88 |  0.14 / 5.79 |

## Takeaways

- **Template wins faithfulness** (100% / 95.9%) — slot-filling floor works as designed.
- **KN-3 wins BLEU / ROUGE** — learns FIREBALL idiom, but hallucinates names (3.2% actor mention).
- **Absolute BLEU is small** — player-authored gold narrations have low n-gram overlap ceiling.

## Next

1. Run `03_train_t5_seq2seq.ipynb` on Colab T4 (~20-25 min).
2. Copy resulting `predictions_test.jsonl` → `models/t5_small/`.
3. Re-run `04_evaluate.ipynb` locally to fill the T5 row.
4. Write up: Template vs KN-3 vs T5 — faithfulness / fluency trade-off + FIREBALL-noise limitation.

## Local-run caveat

NLTK's `KneserNeyInterpolated.score()` is O(vocab) per call; generation loop was intractable. Swapped generation (nb 02, cells 11 & 13) to raw-count MLE with stupid-backoff. KN is still used for perplexity.
