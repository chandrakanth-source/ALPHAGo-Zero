# AlphaGo Zero — Curriculum Training & Evaluation Report (Stages 2–4)

**Board:** 12×12 Go · **Network:** `GoNetwork` (10 residual blocks, 64 channels, policy + value heads)
**Hardware:** 20-core CPU (18 self-play workers) + NVIDIA RTX 4050 Laptop GPU (training)
**Training runs:** 6–7 October 2026 · **Trainer:** `train_batched.py` (batched, resumable)

---

## 1. Summary

| Iteration | Stage | Self-play games | MCTS sims/move | Training positions | Eval result (cand–best–draw) | Win rate | Outcome |
|---|---|---|---|---|---|---|---|
| 3 | 2 – Intermediate | 400 | 35 | 60,677 | 6–14–0 | 30% | ❌ Rejected |
| 4 | 2 – Intermediate | 400 | 35 | 121,733 | 14–6–0 | 70% | ✅ **Promoted** |
| 5 | 3 – Difficult | 600 | 50 | 202,916 | 9–10–1 | 45% | ❌ Rejected |
| 6 | 3 – Difficult | 600 | 50 | 281,526 | 10–10–0 | 50% | ❌ Rejected |
| 7 | 4 – Expert | 850 | 80 | 330,296 | 11–9–0 | 55% | ✅ **Promoted** |
| 8 | 4 – Expert | 850 | 80 | — | — | — | ⏳ Pending |

**Current best model:** `models/model_iteration_7.pt` (= `models/latest_model.pt`).

Promotion rule: a candidate replaces the current best if it wins **≥ 55%** of the evaluation games. Iterations 3–7 used 20 evaluation games; from iteration 8 onwards evaluation uses **40 games**.

---

## 2. Self-Play Data (Positions Generated)

Self-play is played in batches of 25 games, saved after each batch. Each position is stored as a training example `(board planes, MCTS visit-count policy, game result)`.

| Iteration | Games | Batches | New positions | Avg positions per game | Self-play time |
|---|---|---|---|---|---|
| 3 | 400 | 16 | 60,677 | 152 | 32 min |
| 4 | 400 | 16 | 61,056 | 153 | 33 min |
| 5 | 600 | 24 | 141,860 | 236 | 106 min |
| 6 | 600 | 24 | 139,666 | 233 | 103 min |
| 7 | 850 | 34 | 190,630 | 224 | 227 min |
| **Total** | **2,850** | **114** | **593,889** | | **~8.3 h** |

Games became longer from stage 3 onwards (≈150 → ≈230 moves): the promoted model plays more contested games, and deeper search (50–80 simulations) delays passing.

### Replay window
Each iteration trains on its own games **plus the previous iteration's games** (replay window = 2):

| Iteration | Window | Positions |
|---|---|---|
| 3 | iter 3 | 60,677 |
| 4 | iter 3 + 4 | 60,677 + 61,056 = 121,733 |
| 5 | iter 4 + 5 | 61,056 + 141,860 = 202,916 |
| 6 | iter 5 + 6 | 141,860 + 139,666 = 281,526 |
| 7 | iter 6 + 7 | 139,666 + 190,630 = 330,296 |

---

## 3. Training Losses (GPU)

Loss = policy cross-entropy (vs MCTS visit distribution) + value MSE (vs game result). Optimiser: SGD, momentum 0.9, lr 0.001, L2 1e-4, batch 32.

| Iteration | Epochs | First-epoch loss (policy / value) | Final loss (policy / value) | Time per epoch |
|---|---|---|---|---|
| 3 | 4 | 5.25 (4.95 / 0.295) | 4.83 (4.78 / 0.052) | ~15 s |
| 4 | 4 | 4.83 (4.63 / 0.199) | 4.25 (4.19 / 0.053) | ~33 s |
| 5 | 5 | 3.99 (3.78 / 0.210) | 3.66 (3.62 / 0.041) | ~54 s |
| 6 | 5 | 3.58 (3.44 / 0.141) | 3.36 (3.33 / 0.035) | ~73 s |
| 7 | 6 | 3.64 (3.48 / 0.165) | 3.39 (3.35 / 0.033) | ~90 s |

The policy loss fell from **4.78 → 3.35** across the curriculum (a uniform guess over 145 moves is ≈ 4.98), showing the network learning increasingly confident move preferences.

---

## 4. Evaluation Details

Each evaluation pits the candidate against the current best model at the stage's simulation count. Colours alternate; the first 8 moves are sampled from MCTS visit counts so games differ; games are capped at 500 moves and scored by area (no komi).

| Iteration | Opponent (current best) | Candidate wins as Black | Candidate wins as White | Total |
|---|---|---|---|---|
| 3 | pre-curriculum best | 2 / 10 | 4 / 10 | 6 / 20 |
| 4 | pre-curriculum best | 7 / 10 | 7 / 10 | **14 / 20** |
| 5 | iteration 4 | 5 / 10 | 4 / 10 | 9 / 20 (+1 draw) |
| 6 | iteration 4 | 5 / 10 | 5 / 10 | 10 / 20 |
| 7 | iteration 4 | 4 / 10 | 7 / 10 | **11 / 20** |

Note: `data/evaluation_history.json` records the opponent as `latest_model.pt`; the table above names the model that `latest_model.pt` held at the time.

### Statistical reliability
With 20 games, a candidate of *equal* strength still passes the 55% bar about 41% of the time, and a *weaker* one about 25% of the time. Iterations 5–7 (45–55%) are therefore close to equal strength with the previous best. Raising evaluation to 40 games reduces those false-promotion rates to about 32% and 13%:

| True win rate | P(promoted), 20 games | P(promoted), 40 games |
|---|---|---|
| 45% | 25% | 13% |
| 50% | 41% | 32% |
| 60% | 76% | 79% |
| 65% | 88% | 93% |

---

## 5. Bugs Found and Fixed Before These Results

All results above were produced **after** the following fixes (earlier stage 2 data was archived and regenerated):

1. **MCTS backup sign bug** — tree nodes stored value from the opponent's perspective, so the search picked moves best for the opponent. A candidate with an accurate value head lost 0–20; after the fix, the same model scored 9–11.
2. **Network input mismatch** — training data used an *empty-points* third input plane, while search fed a constant *Black-to-move* plane.
3. **Double softmax in the policy loss**, **deterministic evaluation games**, **silent random-model fallback** in `load_model`, and **missing Dirichlet root noise** in self-play.

See `alphago_zero_project_report_updated.md` § 5 for details.

---

## 6. Models Included

| File | Description |
|---|---|
| `models/model_iteration_3.pt` | Stage 2 candidate (rejected, 30%) |
| `models/model_iteration_4.pt` | Stage 2 candidate (**promoted**, 70%) |
| `models/model_iteration_5.pt` | Stage 3 candidate (rejected, 45%) |
| `models/model_iteration_6.pt` | Stage 3 candidate (rejected, 50%) |
| `models/model_iteration_7.pt` | Stage 4 candidate (**promoted**, 55%) — current best |
| `models/latest_model.pt` | Current best model (identical to iteration 7) |

Self-play datasets (60–330k positions, up to ~160 MB each) are not committed because they exceed GitHub's 100 MB file limit.

---

## 7. Remaining Work
- **Stage 4, iteration 8** — 850 games at 80 simulations, 40-game evaluation (~4 h).
- **Stage 5, iterations 9–10** — 1,000 games each at 100 simulations (~7.5 h).
