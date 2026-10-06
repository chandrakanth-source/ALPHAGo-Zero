# AlphaGo Zero – From Scratch in Python

A clean, fully self-contained implementation of the **AlphaGo Zero** algorithm
applied to a **12×12 Go** board (configurable), written in pure Python /
PyTorch.  The project covers every component from the original paper – from
the Go environment and neural network to MCTS, self-play, training, evaluation,
and model promotion – all wired together into one automatic loop.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Environment** | 12×12 Go with captures, ko, two-pass termination, territory scoring |
| **Neural network** | Dual-head CNN (policy + value) — `GoNetwork` |
| **MCTS** | PUCT-based Monte Carlo Tree Search driven by the neural network |
| **Self-play** | Temperature-scheduled move selection (τ=1 for exploration, τ→0 after threshold) |
| **Training** | SGD with momentum, L2 regularisation, combined policy + value loss |
| **Checkpointing** | Epoch-level checkpoints with automatic resume (`CheckpointManager`) |
| **Evaluation** | Alternating-colour match between candidate and current best model |
| **Promotion** | Candidate promoted when win rate ≥ threshold (default 55 %) |
| **Full pipeline** | Single `python main.py` or `python iteration/run_iteration.py` |
| **Demo** | `python demo.py` — watch two AIs play a full game |
| **Tests** | 85 + pytest tests covering every component |

---

## 📁 Project Structure

```
alphago-zero/
│
├── main.py                       ← Full training pipeline (original)
├── demo.py                       ← Final demonstration: AI vs AI game
│
├── config/
│   └── training_config.py        ← Hyperparameter constants
│
├── environment/
│   └── go_game.py                ← Go game rules & state
│
├── mcts/
│   ├── mcts.py                   ← PUCT MCTS implementation
│   ├── node.py                   ← MCTSNode
│   ├── puct.py                   ← PUCT score formula
│   └── network_Evaluator.py      ← NetworkEvaluator (state → policy, value)
│
├── network/
│   ├── network.py                ← GoNetwork (dual-head CNN)
│   └── model_loader.py           ← Safe checkpoint / state-dict loading
│
├── self_play/
│   ├── self_play.py              ← SelfPlay with temperature scheduling
│   ├── trained_self_play.py      ← CLI demo for trained-model self-play
│   └── generate_data.py          ← Quick data-generation helper
│
├── training/
│   ├── trainer.py                ← Trainer with checkpoint integration
│   ├── train_iteration.py        ← Single-iteration train function
│   ├── train.py                  ← Standalone training script
│   ├── dataset.py                ← SelfPlayDataset (PyTorch Dataset)
│   ├── checkpoint_manager.py     ← CheckpointManager (save/load/resume)
│   └── checkpoints/              ← Auto-created checkpoint directory
│
├── evaluation/
│   ├── arena.py                  ← Generic Arena for two-model matches
│   ├── model_match.py            ← ModelPlayer + play_game
│   ├── model_evaluator.py        ← ModelEvaluator utilities
│   ├── promotion.py              ← PromotionManager
│   └── model_promotion.py        ← ModelPromotion (win-rate decision)
│
├── iteration/
│   └── run_iteration.py          ← Day 26: complete automatic pipeline
│
├── iterations/
│   └── iteration.py              ← AlphaGoZeroIteration (one outer loop)
│
├── models/                       ← Saved model weights
│   ├── model_iteration_N.pt
│   ├── checkpoint.pt
│   └── latest_model.pt
│
├── data/                         ← Self-play datasets
│   └── self_play_iteration_N.pt
│
└── tests/                        ← pytest test suite (85+ tests)
```

---

## 🏗️ Architecture

```
             ┌───────────────┐
             │   Go Board    │
             │  (12×12 Go)   │
             └───────┬───────┘
                     │  state
                     ▼
             ┌───────────────┐
             │ Neural Network│
             │  GoNetwork    │
             └───────┬───────┘
                     │
              Policy + Value
                     │
                     ▼
             ┌───────────────┐
             │     MCTS      │
             │ PUCT formula  │
             └───────┬───────┘
                     │ best move
                     ▼
                 Self-Play
                     │ (state, policy, value) triples
                     ▼
             Training Dataset
                     │
                     ▼
                 Training
               (SGD + L2 reg)
                     │
                     ▼
               Candidate Model
                     │
                     ▼
                Evaluation
           (Candidate vs Current)
                     │
                     ▼
              Promote / Reject
                     │
                     └──────────► Repeat (next iteration)
```

### Neural Network (`GoNetwork`)

```
Input: (B, 3, N, N)  – [current-player stones | opponent stones | empty]
    ↓
Conv(3→64, 3×3) → ReLU
Conv(64→64, 3×3) → ReLU
Conv(64→64, 3×3) → ReLU
    ↓
┌──────────────────────┬────────────────────────┐
│  Policy head         │  Value head            │
│  Conv(64→2, 1×1)     │  Conv(64→1, 1×1)       │
│  Flatten             │  Flatten → Linear(64)  │
│  Linear(N²+1)        │  Linear(1) → Tanh      │
│  Softmax             │                        │
└──────────────────────┴────────────────────────┘

Output:
  policy: (B, N²+1)   – probability over all moves + PASS
  value:  (B, 1)       – position evaluation in [-1, 1]
```

For the default 12×12 board: input = **(B, 3, 12, 12)**, policy = **(B, 145)**, value = **(B, 1)**.

### MCTS (PUCT)

Each move:
1. **Select** – traverse tree by highest PUCT score:
   ```
   PUCT(s,a) = Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))
   ```
2. **Expand** – call neural network to get `(policy, value)`, create child nodes
3. **Backup** – propagate value back, flipping sign at each player boundary

### Temperature Scheduling

| Move number | Temperature | Behaviour |
|---|---|---|
| < `temp_threshold` (default 30) | τ = 1.0 | Stochastic – explore diverse moves |
| ≥ `temp_threshold` | τ = 0 | Greedy – pick highest-visit move |

### Training Loss (AlphaGo Zero objective)

```
L = cross_entropy(π_mcts, π_network) + MSE(z, v_network)
```

- `π_mcts` = MCTS visit-count policy
- `z` = game outcome from self-play (+1 / -1 / 0)
- `v_network` = network value prediction

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch numpy pytest
```

### 2. Run the full training pipeline

```bash
# Default: 5 iterations, 12×12 board
python main.py

# Or use the Day 26 clean pipeline:
python iteration/run_iteration.py
```

### 3. Resume interrupted training

Training automatically resumes from the last checkpoint:

```bash
python main.py          # detects training/checkpoints/checkpoint_latest.pt
python iteration/run_iteration.py   # same
```

Force a fresh start:

```bash
python main.py --fresh
python iteration/run_iteration.py --fresh
```

### 4. Watch a demo game

```bash
# Two AIs play against each other using the latest trained model
python demo.py

# Small board for a quick demo
python demo.py --board-size 5 --sims 20

# Show only the final board
python demo.py --quiet
```

### 5. Command-line options (main.py / run_iteration.py)

```
--iterations INT          Number of training iterations (default: 5)
--board-size INT          Board side length (default: 12)
--games INT               Self-play games per iteration (default: 5)
--simulations INT         MCTS simulations per move (default: 10)
--epochs INT              Training epochs per iteration (default: 5)
--batch-size INT          Mini-batch size (default: 32)
--eval-games INT          Evaluation games (default: 20)
--promotion-threshold F   Win rate to promote (default: 0.55)
--lr FLOAT                Learning rate (default: 0.001)
--temperature FLOAT       Move selection temperature (default: 1.0)
--temp-threshold INT      Moves before temperature → 0 (default: 30)
--fresh                   Ignore checkpoints, restart from scratch
--start-iteration INT     Override starting iteration number
```

---

## ⏸️ Checkpoint & Resume (Day 25)

The `CheckpointManager` saves the complete training state after every epoch:

```
training/checkpoints/
├── checkpoint_iter0001_epoch0000.pt
├── checkpoint_iter0001_epoch0001.pt
├── checkpoint_iter0001_epoch0002.pt
├── checkpoint_latest.pt      ← always the most recent
└── checkpoint_meta.json      ← rolling history list
```

Each checkpoint contains:
- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Current epoch number
- Current iteration number
- Loss metrics

Automatic resume:

```python
from training.trainer import Trainer
trainer = Trainer(model)
iteration, epoch = trainer.resume_from_latest()
# → prints: "Resumed from checkpoint: iteration=2, epoch=3"
```

---

## 🔄 Training Pipeline (Day 26)

The complete automatic loop (`iteration/run_iteration.py`):

```
Iteration 1
  ↓
  [1] Self-play (current best model generates games)
  ↓
  [2] Training  (new candidate model trained on those games)
  ↓
  [3] Evaluation (candidate vs current best, alternating colours)
  ↓
  [4] Promotion  (promote if win rate ≥ threshold, else reject)
  ↓
Iteration 2 (using the new best model)
  ↓
  ...
```

Model files:

```
models/
├── model_iteration_0.pt   ← bootstrap (random)
├── model_iteration_1.pt   ← trained candidate after iteration 1
├── model_iteration_2.pt   ← trained candidate after iteration 2
├── checkpoint.pt          ← copy of most recently promoted model
└── latest_model.pt        ← always the current best model
```

---

## 🧪 Testing (Day 27)

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_mcts_benchmark.py -v     # MCTS benchmarks
python -m pytest tests/test_self_play_validation.py -v  # self-play correctness
python -m pytest tests/test_network_shapes.py -v     # network I/O shapes
python -m pytest tests/test_checkpoints.py -v        # checkpoint round-trips
python -m pytest tests/test_pipeline_integration.py -v  # end-to-end loop
```

### Test coverage

| Area | Tests |
|---|---|
| Go game rules (captures, ko, scoring) | `tests_go_game.py` |
| MCTS node, PUCT formula | `tests_mcts_node.py`, `test_puct.py` |
| MCTS search correctness & benchmarks | `tests_mcts_network.py`, `test_mcts_benchmark.py` |
| Network I/O shapes (B, 3, 12, 12) → (B, 145), (B, 1) | `test_network_shapes.py`, `tests_network.py` |
| NetworkEvaluator | `test_network_evaluator.py` |
| Self-play policy/value correctness | `tests_self_play.py`, `test_self_play_validation.py` |
| Temperature scheduling | `test_temperature_selection.py` |
| Training loss, weight updates | `test_training.py` |
| Checkpoint save / load / resume | `test_checkpoints.py`, `test_model_checkpoint.py` |
| Model loader | `test_model_loader.py` |
| Model evaluation & promotion | `test_model_evaluator.py`, `test_model_promotion.py` |
| Evaluation report | `test_evaluation_report.py` |
| Full self-play → train → evaluate loop | `test_pipeline_integration.py` |
| Iteration runner | `test_iteration.py`, `test_train_iteration.py` |
| MCTS with trained model | `test_trained_mcts.py` |

---

## ⚙️ Configuration

Edit `config/training_config.py`:

```python
BOARD_SIZE          = 12    # Go board side length
SELF_PLAY_GAMES     = 5     # Games per self-play phase
MCTS_SIMULATIONS    = 10    # MCTS rollouts per move
TRAINING_EPOCHS     = 5     # Epochs per training phase
BATCH_SIZE          = 32    # Mini-batch size
EVALUATION_GAMES    = 20    # Games for model evaluation
PROMOTION_THRESHOLD = 0.55  # Win rate to promote candidate
```

---

## 📊 Results

After each completed iteration you will see output like:

```
======================================================================
                           ITERATION  1
======================================================================

  Current best model: models/latest_model.pt

  ----------------------------------------------------------------------
    SELF-PLAY  |  Iteration 1  |  5 games
  ----------------------------------------------------------------------
    Game   1/5 |   87 examples |   8.3s | Total: 87
    Game   2/5 |   64 examples |   6.1s | Total: 151
    ...

  ----------------------------------------------------------------------
    TRAINING  |  Iteration 1  |  5 epochs
  ----------------------------------------------------------------------
    Epoch   1/5 | Loss: 5.0231 | Policy: 4.9764 | Value: 0.0467 | 0.4s
    ...

  ----------------------------------------------------------------------
    EVALUATION  |  10 games
  ----------------------------------------------------------------------
    Game   1/10 | Candidate=Black | Candidate wins
    ...
    Candidate win rate: 60.00%

  ----------------------------------------------------------------------
    PROMOTION
  ----------------------------------------------------------------------
    ✓ Candidate PROMOTED  (win rate 60.00% ≥ 55.00%)
```

---

## 🌐 Web Application & Deployment

The web interface is composed of:
1. **Frontend**: Hosted on Vercel static CDN (`index.html`, `app.js`, `style.css`).
2. **Backend**: FastAPI REST engine hosted on Render (`web/server.py`).

### ⚡ Render Cold Start & Health Keep-Alive Service

Free Render containers sleep after 15 minutes of inactivity, causing visitors landing on the page to wait 30–60 seconds for a cold start.

To keep the engine instantly responsive:
1. **Health Check Endpoints**:
   - `GET /health` or `GET /api/ping` returns JSON status `{ "status": "ok", "engine_ready": true, "uptime_seconds": 120 }`.
2. **Free Ping Service Setup (UptimeRobot or cron-job.org)**:
   - Create a free HTTP monitor pointing to `https://alphago-zero.onrender.com/health`.
   - Set the check interval to **every 10 minutes**.
   - This prevents Render from putting the container to sleep, ensuring instantaneous load times for all visitors.

### 🔐 Security & Admin Mode Passcode

Heavy server jobs (Self-play data generation, PyTorch iteration training, and evaluation benchmarks) write to disk and consume CPU/RAM resources. To prevent public server overload or unauthorized triggers:
- Admin password can be set via `ADMIN_PASSWORD` env var on Render (defaults to `alphago2026`).
- Visitors can view live self-play logs, leaderboard Elo ratings, and play against AI without any password.
- Clicking "Start self-play", "Train iteration", or "Run evaluation" requires entering the Admin Passcode via the topbar **Admin Mode** button or popup.

---

## 📖 References

- [Mastering the Game of Go without Human Knowledge — Silver et al., 2017](https://www.nature.com/articles/nature24270)
- [A Simple Alpha(Go) Zero Tutorial — David Foster](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)
