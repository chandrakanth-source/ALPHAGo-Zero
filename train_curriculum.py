"""
AlphaGo Zero - Easy Level Stage Training (250 Games @ 20 Sims/Move)
===================================================================

This script trains Stage 1 (Easy Level):
  - 2 Iterations: model_iteration_1.pt & model_iteration_2.pt
  - 20 MCTS Simulations / move
  - 250 Self-play games per iteration (~25,000 board positions per iteration)
  - 4 Training Epochs per iteration

Usage:
------
Run from terminal:
    python train_curriculum.py
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.training_config import (
    BOARD_SIZE, BATCH_SIZE, EVALUATION_GAMES, PROMOTION_THRESHOLD
)
from main import run_pipeline

EASY_STAGE_CONFIG = {
    "stage": 1,
    "name": "Easy Level",
    "iterations": 2,          # 2 Iterations (model_iteration_1.pt & model_iteration_2.pt)
    "simulations": 20,         # 20 MCTS simulations per move
    "self_play_games": 250,    # 250 self-play games per iteration
    "epochs": 4,               # 4 training epochs
    "description": "High-variety 250-game self-play dataset with 20 MCTS sims lookahead"
}

def run_easy_level_training(board_size: int = BOARD_SIZE):
    cfg = EASY_STAGE_CONFIG
    print("=" * 70)
    print(f" ALPHAGO ZERO — STAGE 1: {cfg['name'].upper()}")
    print("=" * 70)
    print(f" Board size:        {board_size}x{board_size}")
    print(f" Iterations:        {cfg['iterations']} (model_iteration_1.pt & model_iteration_2.pt)")
    print(f" Self-play games:   {cfg['self_play_games']} games / iteration")
    print(f" MCTS simulations:  {cfg['simulations']} sims / move")
    print(f" Training epochs:   {cfg['epochs']} epochs / iteration")
    print(f" Total positions:   ~{cfg['self_play_games'] * 100 * cfg['iterations']:,} training samples")
    print("=" * 70 + "\n")

    start_time = time.time()

    for iter_num in range(1, cfg["iterations"] + 1):
        t0 = time.time()
        print(f"\n" + "#" * 70)
        print(f" STARTING EASY LEVEL — ITERATION {iter_num}/{cfg['iterations']}")
        print(f" 1. Generating {cfg['self_play_games']} self-play games with {cfg['simulations']} MCTS sims...")
        print(f" 2. Training PyTorch ResNet for {cfg['epochs']} epochs...")
        print(f" 3. Evaluating candidate vs current best ({EVALUATION_GAMES} games)...")
        print("#" * 70 + "\n")

        run_pipeline(
            num_iterations=1,
            board_size=board_size,
            self_play_games=cfg["self_play_games"],
            simulations=cfg["simulations"],
            epochs=cfg["epochs"],
            batch_size=BATCH_SIZE,
            eval_games=EVALUATION_GAMES,
            promotion_threshold=PROMOTION_THRESHOLD,
            force_fresh=False
        )

        elapsed_iter = time.time() - t0
        print(f"\n>>> Iteration {iter_num} completed in {elapsed_iter / 60:.1f} minutes!")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(" STAGE 1 (EASY LEVEL) TRAINING COMPLETE!")
    print(f" Total runtime: {total_elapsed / 60:.1f} minutes")
    print(f" Checkpoints saved in models/:")
    print(f"   • models/model_iteration_1.pt")
    print(f"   • models/model_iteration_2.pt")
    print(f"   • models/latest_model.pt")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_easy_level_training()