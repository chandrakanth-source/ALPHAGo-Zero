# AlphaGo Zero — Final Training Report

**Board:** 12×12 Go · **Network:** `GoNetwork` (10 residual blocks, 64 channels, policy + value heads)
**Trainer:** `train_batched.py` (batched, resumable) · **Training period:** 5–10 October 2026
**Hardware:** 20-core CPU (18 self-play workers) + NVIDIA RTX 4050 Laptop GPU (training)

---

## 1. Summary

Ten candidate iterations were trained through the curriculum. Four were promoted to become the best model. The final best model is **iteration 9** (72.5% vs iteration 7).

| Iteration | Stage | Games | Sims/move | Eval games | Result (cand–best–draw) | Win rate | Outcome |
|---|---|---|---|---|---|---|---|
| 0 | baseline | — | — | — | — | — | Starting model |
| 1 | pre-curriculum | 20 | 50 | 20 | 16–4 vs iter 0 | 80% | Evaluated (see §4) |
| 2 | pre-curriculum | 20 | 50 | 20 | 13–7 vs iter 1 | 65% | Evaluated (see §4) |
| 3 | 2 – Intermediate | 400 | 35 | 20 | 6–14–0 | 30% | ❌ Rejected |
| 4 | 2 – Intermediate | 400 | 35 | 20 | 14–6–0 | 70% | ✅ **Promoted** |
| 5 | 3 – Difficult | 600 | 50 | 20 | 9–10–1 | 45% | ❌ Rejected |
| 6 | 3 – Difficult | 600 | 50 | 20 | 10–10–0 | 50% | ❌ Rejected |
| 7 | 4 – Expert | 850 | 80 | 20 | 11–9–0 | 55% | ✅ **Promoted** |
| 8 | 4 – Expert | 850 | 80 | 40 | 21–18–1 | 52.5% | ❌ Rejected |
| 9 | 5 – Super-expert | 1000 | 100 | 40 | 29–11–0 | 72.5% | ✅ **Promoted** |
| 10 | 5 – Super-expert | 1000 | 100 | 40 | 20–20–0 | 50% | ❌ Rejected |

**Current best model:** `models/model_iteration_9.pt` (= `models/latest_model.pt`).

**Promotion rule:** a candidate replaces the current best if it wins **≥ 55%** of the evaluation games. Evaluation used 20 games for iterations 3–7 and 40 games from iteration 8 onward.

---

## 2. Self-Play Data

Self-play runs in batches of 25 games, and each batch is saved before the next starts.

| Iteration | Games | Batches | New positions | Self-play time |
|---|---|---|---|---|
| 3 | 400 | 16 | 60,677 | 32 min |
| 4 | 400 | 16 | 61,056 | 33 min |
| 5 | 600 | 24 | 141,860 | 106 min |
| 6 | 600 | 24 | 139,666 | 103 min |
| 7 | 850 | 34 | 190,630 | 227 min |
| 8 | 850 | 34 | 192,125 | 277 min |
| 9 | 1000 | 40 | 223,865 | 400 min (~6.7 h) |
| 10 | 1000 | 40 | 222,656 | 289 min (~4.8 h) |

Iteration 10 batch 1 ran on Windows (10.6 min). Batches 2–40 ran on Linux, averaging about 7.2 min each. Iteration 9 ran entirely on Windows, averaging 10.0 min per batch.

### Replay window
Each iteration trains on its own games plus the previous iteration's games. The window is 2 iterations from stage 3 onward.

---

## 3. Training Losses

Loss = policy cross-entropy (against the MCTS visit distribution) + value MSE (against the game result).

| Iteration | Positions loaded | Epochs | First-epoch loss | Final loss (policy / value) | Time per epoch |
|---|---|---|---|---|---|
| 3 | 60,677 | 4 | 5.25 | 4.83 (4.78 / 0.052) | ~15 s |
| 4 | 121,733 | 4 | 4.83 | 4.25 (4.19 / 0.053) | ~33 s |
| 5 | 202,916 | 5 | 3.99 | 3.66 (3.62 / 0.041) | ~54 s |
| 6 | 281,526 | 5 | 3.58 | 3.36 (3.33 / 0.035) | ~73 s |
| 7 | 330,296 | 6 | 3.64 | 3.39 (3.35 / 0.033) | ~90 s |
| 8 | 382,755 | 6 | 3.55 | 3.34 (3.31 / 0.031) | ~118 s |
| 9 | 415,990 | 8 | 3.52 | 3.27 (3.24 / 0.031) | ~136 s (Windows) |
| 10 | 446,449 | 8 | 3.50 | 3.29 (3.26 / 0.034) | ~78 s (Linux) |

Policy loss fell from about 4.8 at iteration 3 to about 3.2–3.3 in the final iterations. A uniform guess over 145 moves would give about 4.98.

Iteration 9 and iteration 10 were trained on the same GPU, but iteration 9 ran on Windows and iteration 10 on Linux. The epoch times differ by about 1.75×, which is the same gap seen in self-play batch times.

---

## 4. Evaluation Details

Each candidate plays the current best model. Colours alternate, the first 8 moves are sampled from MCTS visit counts so games differ, games are capped at 500 moves, and scoring is by area with no komi.

| Iteration | Opponent | Candidate as Black | Candidate as White | Total |
|---|---|---|---|---|
| 3 | pre-curriculum best | 2 / 10 | 4 / 10 | 6 / 20 |
| 4 | pre-curriculum best | 7 / 10 | 7 / 10 | **14 / 20** |
| 5 | iteration 4 | 5 / 10 | 4 / 10 | 9 / 20 (+1 draw) |
| 6 | iteration 4 | 5 / 10 | 5 / 10 | 10 / 20 |
| 7 | iteration 4 | 4 / 10 | 7 / 10 | **11 / 20** |
| 8 | iteration 7 | — | — | 21 / 40 (+1 draw) |
| 9 | iteration 7 | — | — | **29 / 40** |
| 10 | iteration 9 | — | — | 20 / 40 |

Iterations 1 and 2 were evaluated in an earlier pipeline (`data/evaluation_history.json`, 20 games at 50 sims each). The per-colour split is not recorded for iterations 8–10.

### Statistical reliability
With 20 games, a candidate of equal strength passes the 55% bar about 41% of the time. With 40 games that falls to about 32%. Iterations 5, 6 and 8 (45–52.5%) are therefore close to equal strength with their opponent. Iteration 10 (50%) was rejected, and it is also close to equal strength with iteration 9.

---

## 5. Bugs Fixed Before These Results

1. **MCTS backup sign bug:** nodes stored the value from the opponent's perspective, so search picked moves that were good for the opponent. After the fix the same model went from 0–20 to 9–11.
2. **Network input mismatch:** training used an empty-points third input plane, while search used a constant Black-to-move plane.
3. **Double softmax** in the policy loss, **deterministic evaluation games**, a **silent random-model fallback** in `load_model`, and **missing Dirichlet root noise** in self-play.

See `alphago_zero_project_report_updated.md` §5 for details.

---

## 6. Models

All trained models are stored in `models/`. Files larger than 100 MB are not committed, which is why self-play datasets are excluded (see §7).

| File | Description |
|---|---|
| `model_iteration_0.pt` … `model_iteration_2.pt` | Pre-curriculum models |
| `model_iteration_3.pt` | Stage 2 candidate (rejected, 30%) |
| `model_iteration_4.pt` | Stage 2 candidate (**promoted**, 70%) |
| `model_iteration_5.pt` | Stage 3 candidate (rejected, 45%) |
| `model_iteration_6.pt` | Stage 3 candidate (rejected, 50%) |
| `model_iteration_7.pt` | Stage 4 candidate (**promoted**, 55%) |
| `model_iteration_8.pt` | Stage 5 candidate (rejected, 52.5%) |
| `model_iteration_9.pt` | Stage 5 candidate (**promoted**, 72.5%) — **current best** |
| `model_iteration_10.pt` | Stage 5 candidate (rejected, 50%) |
| `latest_model.pt` | Current best model (identical to iteration 9) |

Model files are stored with **Git LFS** (`models/*.pt`, see `.gitattributes`).

---

## 7. Data Not Committed

Self-play datasets (`data/self_play_iteration_*.pt`, 0.1–1.1 GB each) and per-batch files (`data/batches/`) are excluded by `.gitignore`. They are kept locally and are not part of the repository.
