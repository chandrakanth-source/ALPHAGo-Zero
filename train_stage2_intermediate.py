"""
AlphaGo Zero - Stage 2: Intermediate Level Training (Iterations 3 & 4)
======================================================================
Configuration:
  - Iterations: 2 (model_iteration_3.pt & model_iteration_4.pt)
  - MCTS Simulations: 35 sims/move
  - Self-play games: 400 games / iteration
  - Epochs: 4 training epochs / iteration
  - Target Elo: ~1350 Elo

Usage:
------
Run from terminal:
    python train_stage2_intermediate.py
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

INTERMEDIATE_STAGE_CONFIG = {
    "stage": 2,
    "name": "Intermediate Level",
    "start_iteration": 3,
    "iterations": 2,           # Iterations 3 & 4
    "simulations": 35,          # 35 MCTS simulations per move
    "self_play_games": 400,     # 400 games per iteration
    "epochs": 4,                # 4 epochs per iteration
    "description": "Tactical 400-game dataset with 35 MCTS sims lookahead"
}

def run_intermediate_level_training(board_size: int = BOARD_SIZE):
    cfg = INTERMEDIATE_STAGE_CONFIG
    print("=" * 70)
    print(f" ALPHAGO ZERO — STAGE 2: {cfg['name'].upper()}")
    print("=" * 70)
    print(f" Board size:        {board_size}x{board_size}")
    print(f" Target Checkpoints:{cfg['iterations']} iterations (model_iteration_3.pt & model_iteration_4.pt)")
    print(f" Self-play games:   {cfg['self_play_games']} games / iteration")
    print(f" MCTS simulations:  {cfg['simulations']} sims / move")
    print(f" Training epochs:   {cfg['epochs']} epochs / iteration")
    print(f" Total positions:   ~{cfg['self_play_games'] * 120 * cfg['iterations']:,} training samples")
    print("=" * 70 + "\n")

    start_time = time.time()

    for i in range(1, cfg["iterations"] + 1):
        t0 = time.time()
        print(f"\n" + "#" * 70)
        print(f" STARTING STAGE 2 ({cfg['name'].upper()}) — STEP {i}/{cfg['iterations']}")
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
            force_fresh=False,
            start_iteration=cfg["start_iteration"] + i - 1
        )

        elapsed_iter = time.time() - t0
        print(f"\n>>> Stage 2 Step {i} completed in {elapsed_iter / 60:.1f} minutes!")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(" STAGE 2 (INTERMEDIATE LEVEL) TRAINING COMPLETE!")
    print(f" Total runtime: {total_elapsed / 60:.1f} minutes")
    print(f" Checkpoints saved in models/:")
    print(f"   • models/model_iteration_3.pt")
    print(f"   • models/model_iteration_4.pt")
    print(f"   • models/latest_model.pt")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_intermediate_level_training()
