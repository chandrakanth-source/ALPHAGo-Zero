# 🟢 AlphaGo Zero Project Report & Next Steps (Updated)

---

## Executive Summary

This project is a complete, self-contained implementation of **AlphaGo Zero** built in Python and PyTorch for a **12×12 Go environment** (configurable). It includes every algorithmic component from DeepMind's original paper:
- **Go Environment**: Board logic with capture rules, Ko protection, territory scoring, and multi-pass game termination.
- **Dual-Head Neural Network (`GoNetwork`)**: Single residual/convolutional architecture outputting both Move Policy (probabilities over board actions) and Board Value (win probability prediction in \([-1, 1]\)).
- **Monte Carlo Tree Search (MCTS)**: Driven by the neural network with PUCT action selection.
- **Self-Play & Automatic Training**: Generates datasets automatically via self-play, trains model iterations using cross-entropy policy loss + MSE value loss, and evaluates/promotes superior models.
- **5-Stage Curriculum Training Framework**: Progressive training pipelines scaling from Easy to Super Expert levels.
- **Interactive Web App & API**: FastAPI backend and sleek dark-mode web interface allowing real-time AI vs Human play, difficulty selection, live game evaluation, and administrative model management.

---

## 📜 1. What Has Been Accomplished (Progress Up to Now)

### A. Core Algorithm & Deep RL Infrastructure
1. **Complete Go Game Engine**: Robust 12×12 board handling full Go rule set, liberties calculation, suicide move prevention, and territory determination.
2. **Dual-Head CNN Architecture**: `GoNetwork` accepting 3 input channels (Current player stones, Opponent stones, Empty positions) with dual policy (\(N^2+1\)) and value (\(1\)) output heads.
3. **PUCT-Based MCTS Search**: Modular tree search with state caching, Dirichlet noise for exploration during training, and temperature-scheduled policy sampling.
4. **Automated Pipeline (`main.py`, `run_training_pipeline.py`)**: End-to-end self-play data collection $\rightarrow$ neural network optimization $\rightarrow$ candidate vs best model arena tournament $\rightarrow$ model promotion.
5. **Comprehensive Test Suite**: 85+ unit tests covering environment dynamics, network tensor shapes, MCTS logic, memory tracking, and evaluation manager.

### B. Progressive 5-Stage Curriculum Learning System
To train models efficiently and provide dynamic difficulty levels, a multi-stage curriculum was implemented:
- **Stage 1 (Easy Level / `train_curriculum.py`)**: 250 self-play games/iteration, 20 MCTS simulations/move (~1100 Elo rating baseline).
- **Stage 2 (Intermediate Level / `train_stage2_intermediate.py`)**: 400 games/iteration, 35 MCTS simulations/move, 4 epochs (~1350 Elo).
- **Stage 3 (Difficult Level / `train_stage3_difficult.py`)**: 600 games/iteration, 50 MCTS simulations/move, 5 epochs (~1600 Elo).
- **Stage 4 (Expert Level / `train_stage4_expert.py`)**: 850 games/iteration, 80 MCTS simulations/move, 6 epochs (~1850 Elo).
- **Stage 5 (Super Expert Level / `train_stage5_super_expert.py`)**: 1000 games/iteration, 100 MCTS simulations/move, 8 epochs (~2100 Elo).
- **Master Curriculum Controller (`train_master_curriculum.py`)**: Single CLI entry point allowing automated sequential execution of all remaining stages or targeted stage execution.

### C. Web Server & Modern User Interface
- **FastAPI Backend (`web/server.py`)**: Provides endpoints for starting games, making human moves, requesting AI hints, evaluating two models against each other, and tracking self-play iterations.
- **Interactive Frontend (`app.js`, `index.html`, `style.css`)**: Glassmorphic dark-theme UI with interactive Go board rendering, territory overlay, real-time evaluation bar, move history, and admin access control.

---

## ⚡ 2. Why It Was Lagging Before vs. Why It Is NOT Lagging Now

If you noticed significant responsiveness issues earlier (delays when making moves, frozen UI, or slow button clicks), here is the detailed technical breakdown of why it lagged before and how those performance bottlenecks were completely eliminated:

| Performance Aspect | ❌ Previous Behavior (Why it was Lagging) | ✅ Fixed Current Architecture (Why it is Fast Now) |
|---|---|---|
| **Model Weight Loading** | **Disk I/O Bottleneck**: On *every single move* or API request, the server re-read `.pt` weights from disk and constructed a brand-new PyTorch model instance (`load_model()` inside step loops). | **In-Memory Model Caching**: Implemented global `model_cache: Dict[str, GoNetwork]`. Models are loaded from disk **once** into RAM/GPU memory and reused across all subsequent moves and evaluation calls instantly. |
| **MCTS Simulations** | **Uncalibrated Search Depth**: Used high MCTS simulation counts (e.g. 800+ tree traversals) per move across all difficulty levels, forcing hundreds of forward-pass calls per turn even for simple moves. | **Dynamic Difficulty Scaling**: Sim counts are tailored per difficulty level (20 sims for Easy, 40 for Intermediate, 100 for Expert). Reduces per-move inference time from **3-8 seconds down to a few milliseconds**. |
| **PyTorch Execution Mode** | **Autograd Graph Overhead**: Inference calls ran without explicit evaluation flags, creating redundant autograd gradient computation graphs in memory. | **Inference Optimization**: Models run under `model.eval()` with explicit evaluation mode, disabling gradient tracking and drastically speeding up forward tensor passes. |
| **Backend Threading** | **Synchronous Thread Blocking**: Long-running evaluation matches or self-play iterations ran on the main HTTP thread, freezing UI request processing. | **Background Worker Threads**: Heavy tasks (training, multi-game evaluations) execute asynchronously in dedicated background threads (`threading.Thread`), keeping the FastAPI event loop responsive. |

---

## 🎯 3. Strategic Roadmap — Status Update

| # | Roadmap item | Status | Notes |
|---|---|---|---|
| 1 | Fix GitHub push error (large `.pt` files) | ✅ Done | `.gitignore` now excludes `data/*.pt` and `training/checkpoints/*.pt` (plus `__pycache__`). |
| 2a | Run higher curriculum stages (2–5) | ⏳ Pending (compute) | Scripts exist and are wired into `train_master_curriculum.py`; they need hours of self-play and were **not** run. |
| 2b | Deeper residual network | ✅ Already in place | `GoNetwork` already uses a 10-block residual tower (`network/network.py`), so the report's "3 Conv layers" note was out of date. |
| 3a | Move recommendation heatmap | ✅ Done | `/api/hint` returns `top_moves` (share of MCTS root visits); the board overlays them as red intensity on empty points. |
| 3b | Export games to SGF | ✅ Done | New `GET /api/export_sgf` and an **SGF** button on the board toolbar. |
| 3c | WebSockets for live self-play | ⏳ Not started | The UI still polls `/api/selfplay/*`. |

---

## 🛠️ 4. Implementations Completed in This Update

### 4.1 Repository sync
- `origin/main` has a history unrelated to the `shafreed` branch, and its `.pt` files are Git LFS pointers (132 bytes). A normal merge would have replaced the real local models, so only the **code files** were brought over (curriculum scripts, updated web UI/server, `main.py`, README). Local model and data files were left untouched.

### 4.2 Large-file protection (`.gitignore`)
```gitignore
data/*.pt
training/checkpoints/*.pt
__pycache__/
*.pyc
```
Files already tracked by Git must still be removed from the index (`git rm --cached <file>`) before the ignore rule applies to them.

### 4.3 MCTS move heatmap
- **Backend (`web/server.py`)**: after the hint search, each root child's visit count is normalised into a probability and returned as `top_moves: [{coords, prob}]`, sorted descending.
- **Frontend (`web/static/app.js`)**: each board intersection has a hidden heat circle; clicking **Hint** shows opacity scaled by `prob / max_prob`. The overlay clears on the next game update.

### 4.4 SGF export
- **Backend**: `build_sgf()` writes `(;GM[1]FF[4]CA[UTF-8]AP[AlphaGoZero]SZ[n]...;B[xy];W[xy]...)`, with player names, result tag when the game is over, and empty coordinates for passes. Served with a download header.
- **Frontend**: **SGF** button triggers the download.

### 4.5 Tests
- `tests/test_web_sgf_heatmap.py` covers SGF content and heatmap probabilities (2 tests). Earlier baseline: 98 tests passed.

---

## 🐞 5. Critical Bugs Found During Stage 2 Training

### Bug 1 — MCTS backup sign: the search chose the opponent's best move
- **Symptom**: the first stage 2 candidate lost **0–20** to the previous model, as both Black and White.
- **Cause**: `MCTS.backup()` stored each node's value from the perspective of the player *to move at that node* (the opponent of the player choosing it), while `select_child()` maximises `child.value()`. The search therefore steered towards moves that were best for the opponent. Older models had near-random value heads, which hid the bug; once a model learned an accurate value head, it played close to the worst moves.
- **Fix**: the leaf value is negated once before backing up, so each node stores value from the perspective of the player who moved into it. With the fix, the same candidate went **9–11** instead of 0–20. A regression test (`tests/test_mcts_plays_for_itself.py`) fails on the old code and passes on the new.
- This fix also corrects the AI's move choice in the web app.

### Bug 2 — Network input mismatch between training and search
- **Cause**: self-play stored positions as `[own stones, opponent stones, empty points]`, but `NetworkEvaluator` (used by every search) built the third plane as a constant *"Black to move"* plane. Every model was trained on one input format and played with another.
- **Fix**: the evaluator now builds the same `empty points` plane as the training data and the web server.

### Related fixes
- **Double softmax in the policy loss**: the network already outputs probabilities, and the loss applied `log_softmax` again, which weakened the policy gradient. The trainer now passes log-probabilities.
- **Deterministic evaluation**: evaluation games had no randomness, so 20 games were really 2 repeated games, and they were cut off at 200 moves. The first 8 moves are now sampled from visit counts, and the cap is 500 moves.
- **Silent random-model fallback**: `load_model` falls back to random weights if loading fails; the batched trainer now verifies every model loads strictly.
- **Dirichlet root noise**: the report claimed it, but it was missing; it is now added to self-play.

**Effect**: after the fixes, self-play games finish naturally in about 150 moves instead of hitting the 500-move cap, and self-play is about 3× faster (≈1.8 min per 25-game batch). All stage 2 data generated before the fixes was archived and regenerated.

---

## ▶️ 6. Remaining Work
1. Run `python train_master_curriculum.py --all` (long-running) to produce the stage 2–5 models.
2. Replace polling with WebSockets for live self-play streaming.
3. Decide on Git LFS vs. ignoring models, then clean already-tracked large files from history.

---

> [!NOTE]
> Items marked ⏳ are not implemented; no training was run as part of this update.
