## Notes (hand-written; every number comes from the tables above or `reports/baseline_*.json`)

### Selection and determinism

- **The selected model is chosen on validation, not test.** On news, validation picks
  **logreg** (91.09 vs 90.78 macro-F1), while LinearSVC happens to score higher on test
  (91.83 vs 90.53). We do not switch to the test-best model. The news test set has 681
  examples, so one example moves accuracy by about 0.15 points, and differences of about
  1 point between the two models are within noise.
- **For news, logreg's best C is 100, the largest value in the grid,** so the optimum may lie
  above the grid. The grid was not extended after the test evaluation.
- **Determinism was checked.** Refitting LinearSVC at its best C with random_state 0, 1 and 2
  gives identical validation macro-F1 on both tasks (spread 0.0). lbfgs logistic regression
  has no randomness. Each result is therefore a single run. Variance across different data
  splits was **not measured**.
- All 16 fits converged (no ConvergenceWarning).

### Sentiment: Notr and the source shortcut

- **Notr scores (test):** logreg precision 96.73 / recall 98.63; LinearSVC precision 96.80 /
  recall 98.82.
- These high scores mainly measure **"is this Wikipedia-style text?"**, not neutrality:
  - 17,048 of the 17,091 gold-Notr test texts (99.7%) come from `wiki`, and the other 43
    come from `random`. No review or tweet source has a single gold-Notr example.
  - The highest-weighted logreg n-grams for Notr are encyclopedic register, not neutral
    opinion: `bulunmaktadır`, `bulunur`, `olmuştur`, `sahiptir`, `genellikle`, `yapmıştır`,
    `vardır`, `günümüzde`, `albüm`, `mevcuttur`, `bilinmektedir`. Our reading (an
    interpretation, not a measurement): **the baseline exploits the source shortcut.**
  - On non-wiki sources, every Notr prediction is wrong by construction, since those sources
    have no gold Notr. For logreg these false Notr predictions are 570 in total: `tweet-pn`
    192/1,096 (17.5%), `urun_yorumlari` 282/23,469 (1.2%), `HUMIR` 82/6,411 (1.3%),
    `magaza_yorumlari` 14/857 (1.6%). Tweets, which are short and written in a different
    style from reviews, get the most.
  - The dataset has no neutral *reviews*, so we **cannot** measure whether the model detects
    genuinely neutral opinions. The Notr numbers must not be quoted as "neutral sentiment
    detection".
- **The weakest class is Negative,** not Notr: recall is 77.56 for both models, and 1,149
  (logreg) / 1,148 (SVC) Negative texts are predicted Positive.
- **Some top Negative n-grams are profanity,** reflecting the tweet and review data. They are
  left unfiltered in the JSON.
