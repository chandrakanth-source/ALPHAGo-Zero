# AlphaGo Zero — From Scratch in Python

A self-contained implementation of the AlphaGo Zero training loop for a configurable Go board. The project combines Go rules, a dual-head PyTorch policy/value network, PUCT Monte Carlo Tree Search, self-play, training, evaluation, and model promotion.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

For the complete iterative pipeline, run:

```bash
python iteration/run_iteration.py
```

## What is included

- Go environment with captures, ko, two-pass termination, and scoring
- Neural policy and value heads implemented in PyTorch
- PUCT MCTS with self-play data generation
- Training checkpoints, automated evaluation, and promotion thresholds
- A web interface backed by FastAPI

## Testing

```bash
python -m pytest tests/ -v
```

## Configuration

Training defaults such as board size, self-play games, MCTS simulations, epochs, and promotion threshold are defined in `config/training_config.py`.

## Team Members

| Name | Roll number |
|---|---|
| Shaik Shafreed | 2024BCS0113 |
| Harshithan Lakavath | 2024BCS0321 |
| Shaik Humair | 2024BCS0105 |
| Chandrakanth Reddy | 2024BCS0125 |
| Arya Vardhan | 2024BCS0129 |
