# Air-raid alert forecasting for Ukrainian oblasts

**Task.** For every oblast and every hour `t`, predict the probability that a **new** oblast-level
alert starts within the next hour, `[t, t+1h)`, using only information available at `t`.
A 6-hour horizon is kept as a weaker comparison variant.

**Why "starts", not "is active".** If an alert is already running at `t`, "there will be an alert in
the next hour" is almost always true and inflates every metric. The useful, hard part is whether a
*new* alert starts. A user-facing answer "will there be an alert" = "already running" OR our forecast.

## Run

```bash
pip install -r requirements.txt
python -m src.fetch_data      # downloads data at a fixed dataset commit -> data/raw/snapshot.json
pytest tests                  # leakage / target / merging tests (9 tests)
python -m src.run_all         # 10 experiment runs + shuffle sanity check (~3-5 min each)
python -m src.predict --at "2026-10-09 15:00"   # forecast at a past moment (UTC) + actual outcome
python -m src.show_tree       # print the real trees of a trained model
```

Reproduce the exact data: `python -m src.fetch_data --commit <hash from data/raw/snapshot.json>`.

## Project structure

| file | what it does |
|---|---|
| `src/fetch_data.py` | resolves the dataset's current commit, downloads both CSVs at that commit, stores hash + sha256 |
| `src/features.py` | reads both sources, merges raion alerts into oblast episodes, builds the hour x oblast table (22 features + target) |
| `src/train_eval.py` | time split, baselines, gradient boosting, validation and test metrics, alarms |
| `src/run_all.py` | experiment grid; chooses variants **on validation**; shuffle sanity check |
| `src/predict.py` | forecast for one moment, simulating exactly what was known at that moment |
| `src/show_tree.py` | prints the model's real decision trees |
| `tests/test_leakage.py` | proves features use only the past; checks target and merging |

## Data

[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT), times in UTC.

| | volunteer (eTryvoga) | official |
|---|---|---|
| level | oblast | oblast / raion / hromada |
| coverage (on 2026-10-12) | to 2026-10-12, no gaps > 24h | to 2026-09-07, **4-day hole 2026-08-29 .. 09-02** |
| used up to | 2026-10-12 04:59 | 2026-08-29 04:55 (`coverage_end`) |
| raw rows -> oblast episodes | 116,171 -> 116,171 (no overlaps) | 299,273 -> 80,287 (73% of rows overlap) |

Findings that differ from the dataset's documentation (checked against the data):
- `datasets/README.md` says data are updated daily; the official file was 5 weeks stale and had a 4-day hole.
  A gap > 24h with no alert anywhere in Ukraine means a collection outage, not quiet skies, so official data
  are cut at the start of the hole ("no records" != "no alerts").
- It says alerts are mostly raion-level "since December 2025". In the data the switch is gradual:
  raion rows outnumber oblast rows from **August 2025**, oblast rows are almost gone by **November 2025**.
  The "new regime" variant therefore trains from 2025-11-01.

Official raion/hromada alerts are merged into **oblast episodes** (union of overlapping intervals per
oblast), so "an alert in the oblast" means the same before and after the switch. Luhanska oblast is
excluded (permanent alert since 2022-04-04). Crimea is not in the data.

## Method

**Table.** One row per (hour `t`, oblast). Target `y = 1` if an episode starts in `[t, t+h)`.
22 features, all computed from events before `t`: Kyiv local hour and weekday; alert running now; hours
since last start / end; starts in the last 1/3/6/24 h and 7 days; share of time under alert in 6/24 h;
the same for neighbouring oblasts (hand-written adjacency map) and nationwide; share of the same hour
with a new alert over 28 days; label rate over 7 days (only labels whose window already closed).
A 30-day lookback before the first row makes sure no feature is computed from a period without data.

**Split** (anchored to the end of continuous coverage):

| part | volunteer, main run | used for |
|---|---|---|
| train | 2022-04-15 .. 2026-09-13 | learning |
| 6h gap | | label of the last train row looks ahead |
| validation | 2026-09-13 .. 09-27 (14 d) | choosing number of trees; choosing variants |
| 6h gap | | |
| test | 2026-09-27 .. 10-11 (14 d) | final score, looked at once per chosen variant |

**Model.** scikit-learn `HistGradientBoostingClassifier`, learning rate 0.05, 31 leaves,
min 200 rows per leaf, oblast as a categorical feature, number of trees from {100, 200, 400} chosen on
validation, then refit on train + validation. sklearn's default early stopping is switched **off**: it
silently held out a random 10% of rows (random, not time-based) and was found by inspecting the trees.

**Baselines.** Climatology: share of hours with a new alert for (oblast, hour of day) over the last 28 days
before test. Persistence: P(y | a start in this oblast in the last 6 h), estimated on train.

**Metrics** (test): Brier score and Brier skill vs climatology; log-loss; ROC-AUC pooled and **within each
oblast** (main metric: pooled AUC is dominated by "Kharkiv vs Zakarpattia", which climatology already
knows); PR-AUC (random level = share of positives); number of 24-hour test blocks where the model beats
climatology (blocks counted from test start, not calendar days, to avoid a 2-hour "day" at the edge).

## Leakage protection

1. `tests/test_leakage.py`: at 12 moments `t`, features from the full data must equal features computed
   the way a live system would (cut raw rows at `t`, unknown ends erased, then merge). Verified by planting
   a deliberate leak (`starts_1h` from `[t, t+1h)`): the test failed and named the affected features.
2. Gaps between train / validation / test; test never used for training or tuning.
3. **Shuffle sanity check** (`run_all`): training labels shuffled -> test within-region AUC must be ~0.5.
   Catches leaks outside the features (split, baselines, tuning) that the feature test cannot see.
   (Brier skill is not meaningful in this run: climatology is also built from shuffled labels.)
4. **Alarm**: within-region AUC > 0.9 or Brier skill > 0.5 is printed as "SUSPICIOUSLY GOOD".

## Results

Data: dataset commit recorded in `data/raw/snapshot.json`; volunteer data end 2026-10-12 04:59 UTC.
Full output of every run: `results/summary_last_run_all.csv`, per-run folders in `results/`.

### Main result: volunteer data, 1h horizon, latest 14 days (2026-09-27 23:00 .. 10-11 22:00 UTC)

Train-from date chosen on validation (2022-04-15: val log-loss 0.4400 vs 0.4425 for 2025-11-01).

| | Brier | Brier skill | log-loss | ROC-AUC pooled | **ROC-AUC within oblast** | PR-AUC | 24h blocks better than climatology |
|---|---|---|---|---|---|---|---|
| climatology (oblast x hour, 28 d) | 0.1576 | 0 | 0.5092 | 0.734 | 0.530 | 0.475 | - |
| persistence (start in last 6 h) | 0.1843 | -0.169 | 0.5593 | 0.611 | 0.515 | 0.289 | 0/14 |
| **gradient boosting** | **0.1494** | **+0.052** | **0.4571** | **0.768** | **0.637** | **0.521** | **14/14** |

Share of hours with a new alert in test: 23.9%; mean predicted probability: 23.5% (well calibrated,
although the train period had only 11.2%: features like 7-day intensity carry the current regime).

**Reading it:** the model beats the climatology baseline in every one of the 14 days, with 5% lower
Brier error. Within a single oblast it ranks an hour with a new alert above an hour without one in 64%
of pairs, vs 53% for climatology (50% = coin). A real but modest gain: see Limitations.

### Variants chosen on validation (honest test numbers)

| source | horizon | test period | train from (chosen on val) | Brier skill | AUC within: clim | AUC within: model | blocks better |
|---|---|---|---|---|---|---|---|
| volunteer | 1h | 09-27 .. 10-11 | 2022-04-15 | +0.052 | 0.530 | **0.637** | 14/14 |
| volunteer | 1h | 08-14 .. 08-28 | 2025-11-01 | +0.050 | 0.530 | **0.645** | 14/14 |
| official | 1h | 08-14 .. 08-28 | 2022-04-15 | +0.141 | 0.578 | **0.727** | 14/14 |
| volunteer | 6h | 09-27 .. 10-11 | 2022-04-15 | +0.079 | 0.567 | 0.602 | 13/14 |
| official | 6h | 08-14 .. 08-28 | 2022-04-15 | +0.083 | 0.552 | 0.603 | 13/14 |

### What the experiments show

1. **1h horizon >> 6h** in telling *when*: within-oblast AUC gain over climatology +0.10..+0.15 at 1h vs
   +0.04..+0.05 at 6h. At 6h the model mostly knows *where*, which climatology already knows.
2. **Train from 2022 vs from 2025-11: no real difference.** Validation picked 2022 on one period and
   2025-11 on the other, by margins of 0.0001..0.0025 in log-loss. One year of the current regime is as
   good as four years of history.
3. **Official data are more predictable than volunteer data on the same 14 days** (AUC within 0.727 vs
   0.632-0.645, Brier skill 0.14 vs 0.05). Hypotheses, not tested: raion alerts spread district by
   district across oblast borders, which neighbour features catch; volunteer data have imputed end times.
   Note the targets differ (merged raion episodes vs eTryvoga records), so this is not the same task.
4. **Sanity checks passed:** shuffled-label run gives test within-oblast AUC 0.522 (≈ 0.5, "ok");
   no run triggered the "suspiciously good" alarm.

## How variants were chosen (and what is honest to claim)

- **Train-from date** (2022 vs 2025-11) is chosen per source and horizon by **validation** log-loss;
  only the winner's test numbers are reported. In an earlier iteration we compared variants on the test
  period, which inflates the winner's number (the variant that got lucky on those 14 days wins).
- **Horizon and source are not chosen by metrics**: they are different tasks with different base rates.
  1h is the main task by design (at 6h, ~69% of windows contain a new alert, so the answer is mostly
  trivial); volunteer data are the main source because the task requires the most recent period and the
  official data end on 2026-08-29. The 6h and official results are reported as additional analysis.

## Limitations

- Modest improvement: the model knows "a wave is under way", but the data contain no launch information
  (aircraft take-offs, missile/drone tracking), so it cannot foresee the start of a new attack.
  Monitoring Telegram channels would add that, but would also need forecasts every few minutes,
  not hourly, to be useful (a launch report at 03:40 is invisible to a 03:00 forecast).
- One test period of 14 days: no estimate of the spread; small differences (e.g. 2022 vs 2025 start)
  are within noise. Rolling-origin evaluation over several periods would fix this.
- Only the number of trees is tuned; other hyperparameters are set by hand.
- Volunteer data: ~5% of 2026 alerts have no recorded end (dataset imputes start + 30 min).
- Official and volunteer targets differ (merged raion episodes vs eTryvoga records); their scores are
  not the same task even on the same days.
