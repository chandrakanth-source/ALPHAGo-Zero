"""
AlphaGo Zero - Stage 3: Difficult Level Training (Iterations 5 & 6)
===================================================================
Configuration:
  - Iterations: 2 (model_iteration_5.pt & model_iteration_6.pt)
  - MCTS Simulations: 50 sims/move
  - Self-play games: 600 games / iteration
  - Epochs: 5 training epochs / iteration
  - Target Elo: ~1600 Elo

Usage:
------
Run from terminal:
    python train_stage3_difficult.py
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

DIFFICULT_STAGE_CONFIG = {
    "stage": 3,
    "name": "Difficult Level",
    "iterations": 2,           # Iterations 5 & 6
    "simulations": 50,          # 50 MCTS simulations per move
    "self_play_games": 600,     # 600 games per iteration
    "epochs": 5,                # 5 epochs per iteration
    "description": "Deep tactical 600-game dataset with 50 MCTS sims lookahead"
}

def run_difficult_level_training(board_size: int = BOARD_SIZE):
    cfg = DIFFICULT_STAGE_CONFIG
    print("=" * 70)
    print(f" ALPHAGO ZERO — STAGE 3: {cfg['name'].upper()}")
    print("=" * 70)
    print(f" Board size:        {board_size}x{board_size}")
    print(f" Target Checkpoints:{cfg['iterations']} iterations (model_iteration_5.pt & model_iteration_6.pt)")
    print(f" Self-play games:   {cfg['self_play_games']} games / iteration")
    print(f" MCTS simulations:  {cfg['simulations']} sims / move")
    print(f" Training epochs:   {cfg['epochs']} epochs / iteration")
    print(f" Total positions:   ~{cfg['self_play_games'] * 130 * cfg['iterations']:,} training samples")
    print("=" * 70 + "\n")

    start_time = time.time()

    for i in range(1, cfg["iterations"] + 1):
        t0 = time.time()
        print(f"\n" + "#" * 70)
        print(f" STARTING STAGE 3 ({cfg['name'].upper()}) — STEP {i}/{cfg['iterations']}")
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
        print(f"\n>>> Stage 3 Step {i} completed in {elapsed_iter / 60:.1f} minutes!")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(" STAGE 3 (DIFFICULT LEVEL) TRAINING COMPLETE!")
    print(f" Total runtime: {total_elapsed / 60:.1f} minutes")
    print(f" Checkpoints saved in models/:")
    print(f"   • models/model_iteration_5.pt")
    print(f"   • models/model_iteration_6.pt")
    print(f"   • models/latest_model.pt")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_difficult_level_training()
