"""
AlphaGo Zero – Full Iterative Training Pipeline  (Task 26)
===========================================================

This is the single entry-point for end-to-end training.

Loop (repeated for *num_iterations* rounds):
  1. Self-play    – the current best model plays games against itself.
  2. Training     – a new candidate model is trained on the collected data.
  3. Evaluation   – the candidate plays against the current best.
  4. Promotion    – if the candidate wins ≥ *promotion_threshold* of games
                    it becomes the new best model; otherwise it is rejected.

Usage
-----
Run directly::

    python main.py

Or import and call :func:`run_pipeline` with custom arguments.

Resume behaviour
----------------
If checkpoints exist in ``training/checkpoints/``, training automatically
resumes from the last saved checkpoint.  Pass ``--fresh`` on the command
line (or set ``force_fresh=True``) to ignore existing checkpoints and
restart from scratch.
"""

import argparse
import os
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader

from config.training_config import (
    BATCH_SIZE,
    BOARD_SIZE,
    EVALUATION_GAMES,
    MCTS_SIMULATIONS,
    PROMOTION_THRESHOLD,
    SELF_PLAY_GAMES,
    TRAINING_EPOCHS,
)
from environment.go_game import GoGame
from evaluation.model_match import ModelPlayer, play_game
from evaluation.promotion import PromotionManager
from mcts.network_Evaluator import NetworkEvaluator
from network.model_loader import load_model
from network.network import GoNetwork
from self_play.self_play import SelfPlay
from training.checkpoint_manager import CheckpointManager
from training.dataset import SelfPlayDataset
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

MODELS_DIR = "models"
DATA_DIR = "data"
CHECKPOINT_DIR = "training/checkpoints"
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, "latest_model.pt")
LATEST_MODEL = LATEST_MODEL_PATH


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _banner(text, width=70):
    print()
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def _section(text, width=70):
    print()
    print("-" * width)
    print(f"  {text}")
    print("-" * width)


def ensure_dirs():
    """Create output directories if they do not exist."""
    for d in (MODELS_DIR, DATA_DIR, CHECKPOINT_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 – Self-play
# ---------------------------------------------------------------------------


def run_self_play(
    model,
    iteration,
    num_games=SELF_PLAY_GAMES,
    simulations=MCTS_SIMULATIONS,
    board_size=BOARD_SIZE,
    temperature=1.0,
    temp_threshold=30,
):
    """
    Generate self-play training data using *model*.

    Args:
        model:          GoNetwork to use as the policy/value evaluator.
        iteration:      Current training iteration number (used for filename).
        num_games:      Number of games to play.
        simulations:    MCTS simulations per move.
        board_size:     Size of the Go board.
        temperature:    Exploration temperature for move selection.
        temp_threshold: After this many moves per game, switch to greedy.

    Returns:
        str: Path of the saved dataset file.
    """
    _section(f"SELF-PLAY  |  Iteration {iteration}  |  {num_games} games")

    evaluator = NetworkEvaluator(model)

    self_play = SelfPlay(
        board_size=board_size,
        simulations=simulations,
        evaluator=evaluator,
        temperature=temperature,
        temp_threshold=temp_threshold,
    )

    examples = []

    t_start = time.time()

    for game_idx in range(num_games):
        t0 = time.time()
        game_data = self_play.generate_game()
        elapsed = time.time() - t0

        examples.extend(game_data)

        print(
            f"  Game {game_idx + 1:>3}/{num_games} | "
            f"{len(game_data):>4} examples | "
            f"{elapsed:5.1f}s | "
            f"Total examples: {len(examples)}"
        )

    total_elapsed = time.time() - t_start

    print()
    print(
        f"  Self-play finished in {total_elapsed:.1f}s  "
        f"({len(examples)} examples total)"
    )

    # Save dataset.
    data_path = os.path.join(
        DATA_DIR, f"self_play_iteration_{iteration}.pt"
    )
    torch.save(examples, data_path)

    print(f"  Dataset saved -> {data_path}")

    return data_path


# ---------------------------------------------------------------------------
# Step 2 – Training
# ---------------------------------------------------------------------------


def run_training(
    data_path,
    iteration,
    previous_model_path=None,
    epochs=TRAINING_EPOCHS,
    batch_size=BATCH_SIZE,
    board_size=BOARD_SIZE,
    learning_rate=0.001,
    resume=True,
):
    """
    Train a new candidate model on self-play data.

    Args:
        data_path:            Path to the ``.pt`` dataset file.
        iteration:            Current training iteration number.
        previous_model_path:  If supplied, initialise weights from this
                              model (warm-start) instead of random init.
        epochs:               Number of training epochs.
        batch_size:           Batch size for the DataLoader.
        board_size:           Board size (must match the dataset).
        learning_rate:        Initial learning rate.
        resume:               If *True* and a checkpoint exists for this
                              iteration, resume from it.

    Returns:
        str: Path of the saved candidate model file.
    """
    _section(f"TRAINING  |  Iteration {iteration}  |  {epochs} epochs")

    # ---- Dataset ----
    examples = torch.load(data_path, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(examples)} training examples")

    dataset = SelfPlayDataset(examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---- Model ----
    if previous_model_path and os.path.exists(previous_model_path):
        model = load_model(previous_model_path, board_size=board_size)
        print(f"  Warm-started from: {previous_model_path}")
    else:
        model = GoNetwork(board_size=board_size)
        print("  Initialised fresh model")

    trainer = Trainer(
        model,
        learning_rate=learning_rate,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # ---- Resume from checkpoint ----
    start_epoch = 0
    if resume:
        ckpt_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)
        if ckpt_manager.checkpoint_exists():
            try:
                payload = ckpt_manager.load_latest(
                    model=model,
                    optimizer=trainer.optimizer,
                    device=str(trainer.device),
                )
                if payload and payload.get("iteration") == iteration:
                    start_epoch = int(payload.get("epoch", 0)) + 1
                    print(
                        f"  Resumed from checkpoint at epoch {start_epoch}"
                    )
            except RuntimeError:
                print("  WARNING: Checkpoint board-size mismatch – starting from epoch 0")


    # ---- Training loop ----
    best_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        metrics = trainer.train_epoch(dataloader)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch + 1:>3}/{epochs} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Policy: {metrics['policy_loss']:.4f} | "
            f"Value: {metrics['value_loss']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save epoch-level checkpoint.
        trainer.save_checkpoint(
            path=os.path.join(
                CHECKPOINT_DIR,
                f"checkpoint_iter{iteration:04d}_epoch{epoch:04d}.pt",
            ),
            epoch=epoch,
            iteration=iteration,
            metrics=metrics,
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]

    # ---- Save candidate model ----
    candidate_path = os.path.join(
        MODELS_DIR, f"model_iteration_{iteration}.pt"
    )
    torch.save(model.state_dict(), candidate_path)
    print(f"  Candidate model saved -> {candidate_path}")

    return candidate_path


# ---------------------------------------------------------------------------
# Step 3 – Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    current_model_path,
    candidate_model_path,
    num_games=EVALUATION_GAMES,
    simulations=MCTS_SIMULATIONS,
    board_size=BOARD_SIZE,
):
    """
    Pit the candidate model against the current best.

    Games are played with alternating colours to remove first-move bias.

    Args:
        current_model_path:   Path to the current best model.
        candidate_model_path: Path to the newly trained model.
        num_games:            Total number of evaluation games.
        simulations:          MCTS simulations per move.
        board_size:           Board size.

    Returns:
        dict: ``{"wins": …, "losses": …, "draws": …, "win_rate": …}``
    """
    _section(f"EVALUATION  |  {num_games} games")

    current_model = load_model(current_model_path, board_size=board_size)
    candidate_model = load_model(candidate_model_path, board_size=board_size)

    current_player = ModelPlayer(
        model=current_model,
        board_size=board_size,
        simulations=simulations,
    )
    candidate_player = ModelPlayer(
        model=candidate_model,
        board_size=board_size,
        simulations=simulations,
    )

    candidate_wins = 0
    current_wins = 0
    draws = 0

    for game_idx in range(1, num_games + 1):
        game = GoGame(board_size=board_size)

        # Alternate colours.
        if game_idx % 2 == 1:
            black_player = candidate_player
            white_player = current_player
            candidate_color = "Black"
        else:
            black_player = current_player
            white_player = candidate_player
            candidate_color = "White"

        winner = play_game(
            model_black=black_player,
            model_white=white_player,
            game=game,
        )

        if winner == 0:
            draws += 1
            outcome = "Draw"
        elif (winner == 1 and candidate_color == "Black") or (
            winner == -1 and candidate_color == "White"
        ):
            candidate_wins += 1
            outcome = "Candidate wins"
        else:
            current_wins += 1
            outcome = "Current wins"

        print(
            f"  Game {game_idx:>3}/{num_games} | "
            f"Candidate={candidate_color:<5} | "
            f"{outcome}"
        )

    win_rate = candidate_wins / num_games if num_games > 0 else 0.0

    print()
    print(f"  Candidate wins : {candidate_wins}")
    print(f"  Current wins   : {current_wins}")
    print(f"  Draws          : {draws}")
    print(f"  Candidate win rate: {win_rate:.2%}")

    return {
        "wins": candidate_wins,
        "losses": current_wins,
        "draws": draws,
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# Step 4 – Promotion
# ---------------------------------------------------------------------------


def run_promotion(
    candidate_path,
    eval_result,
    threshold=PROMOTION_THRESHOLD,
):
    """
    Promote the candidate if its win rate exceeds *threshold*.

    Args:
        candidate_path: Path to the candidate model.
        eval_result:    Dict returned by :func:`run_evaluation`.
        threshold:      Win-rate threshold for promotion.

    Returns:
        bool: ``True`` if the candidate was promoted.
    """
    _section("PROMOTION")

    win_rate = eval_result["win_rate"]
    promoted = win_rate >= threshold

    promoter = PromotionManager(models_dir=MODELS_DIR)

    if promoted:
        promoter.promote(candidate_path)
        print(
            f"  ✓  Candidate PROMOTED  (win rate {win_rate:.2%} ≥ {threshold:.2%})"
        )
    else:
        promoter.reject(candidate_path)
        print(
            f"  ✗  Candidate REJECTED  (win rate {win_rate:.2%} < {threshold:.2%})"
        )

    return promoted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    num_iterations=5,
    board_size=BOARD_SIZE,
    self_play_games=SELF_PLAY_GAMES,
    simulations=MCTS_SIMULATIONS,
    epochs=TRAINING_EPOCHS,
    batch_size=BATCH_SIZE,
    eval_games=EVALUATION_GAMES,
    promotion_threshold=PROMOTION_THRESHOLD,
    learning_rate=0.001,
    temperature=1.0,
    temp_threshold=30,
    force_fresh=False,
    start_iteration=None,
):
    """
    Run the full AlphaGo Zero training pipeline.

    Args:
        num_iterations:       Number of self-play → train → evaluate loops.
        board_size:           Go board side length.
        self_play_games:      Games per self-play phase.
        simulations:          MCTS simulations per move.
        epochs:               Training epochs per iteration.
        batch_size:           Mini-batch size.
        eval_games:           Evaluation games per promotion check.
        promotion_threshold:  Win-rate required to promote a candidate.
        learning_rate:        Initial SGD learning rate.
        temperature:          Self-play move-selection temperature.
        temp_threshold:       Move count after which temperature → 0.
        force_fresh:          If *True*, ignore existing checkpoints.
        start_iteration:      Override the starting iteration number.
    """
    ensure_dirs()

    _banner("ALPHAGO ZERO – FULL TRAINING PIPELINE")

    # ---- Print configuration ----
    print()
    print("Configuration")
    print(f"  Board size         : {board_size}")
    print(f"  Iterations         : {num_iterations}")
    print(f"  Self-play games    : {self_play_games}")
    print(f"  MCTS simulations   : {simulations}")
    print(f"  Training epochs    : {epochs}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Evaluation games   : {eval_games}")
    print(f"  Promotion threshold: {promotion_threshold:.0%}")
    print(f"  Learning rate      : {learning_rate}")
    print(f"  Temperature        : {temperature}")
    print(f"  Temp threshold     : {temp_threshold} moves")

    # ---- Determine starting iteration ----
    ckpt_manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)

    if start_iteration is not None:
        current_iteration = start_iteration
    elif force_fresh:
        current_iteration = 1
    else:
        current_iteration = ckpt_manager.latest_iteration() + 1

    print(f"\n  Starting at iteration: {current_iteration}")

    # ---- Ensure a bootstrap model exists ----
    if not os.path.exists(LATEST_MODEL_PATH):
        print()
        print("  No latest_model.pt found — bootstrapping random model …")
        bootstrap = GoNetwork(board_size=board_size)
        torch.save(bootstrap.state_dict(), LATEST_MODEL_PATH)
        # Also save as iteration 0.
        iter0_path = os.path.join(MODELS_DIR, "model_iteration_0.pt")
        torch.save(bootstrap.state_dict(), iter0_path)
        print(f"  Bootstrap model saved -> {LATEST_MODEL_PATH}")

    # ---- Main loop ----
    for iteration in range(current_iteration, current_iteration + num_iterations):
        _banner(f"ITERATION  {iteration}", width=70)

        t_iter_start = time.time()

        # 1. Load current best model.
        current_best_path = LATEST_MODEL_PATH
        current_best = load_model(current_best_path, board_size=board_size)
        print(f"\n  Current best model: {current_best_path}")

        # 2. Self-play.
        data_path = run_self_play(
            model=current_best,
            iteration=iteration,
            num_games=self_play_games,
            simulations=simulations,
            board_size=board_size,
            temperature=temperature,
            temp_threshold=temp_threshold,
        )

        # 3. Training.
        candidate_path = run_training(
            data_path=data_path,
            iteration=iteration,
            previous_model_path=current_best_path,
            epochs=epochs,
            batch_size=batch_size,
            board_size=board_size,
            learning_rate=learning_rate,
            resume=(not force_fresh),
        )

        # 4. Evaluation.
        eval_result = run_evaluation(
            current_model_path=current_best_path,
            candidate_model_path=candidate_path,
            num_games=eval_games,
            simulations=simulations,
            board_size=board_size,
        )

        # 5. Promotion.
        promoted = run_promotion(
            candidate_path=candidate_path,
            eval_result=eval_result,
            threshold=promotion_threshold,
        )

        iter_elapsed = time.time() - t_iter_start

        _banner(
            f"ITERATION {iteration} COMPLETE  "
            f"({'PROMOTED' if promoted else 'REJECTED'})  "
            f"{iter_elapsed:.0f}s"
        )

    _banner("TRAINING PIPELINE COMPLETE")
    print()
    print(f"  Final model: {LATEST_MODEL_PATH}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="AlphaGo Zero – full iterative training pipeline"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of self-play → train → evaluate iterations",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=BOARD_SIZE,
        help="Go board side length",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=SELF_PLAY_GAMES,
        help="Self-play games per iteration",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=MCTS_SIMULATIONS,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_EPOCHS,
        help="Training epochs per iteration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=EVALUATION_GAMES,
        help="Evaluation games per iteration",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=PROMOTION_THRESHOLD,
        help="Win-rate threshold to promote a candidate (0.0 – 1.0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Move-selection temperature",
    )
    parser.add_argument(
        "--temp-threshold",
        type=int,
        default=30,
        help="Move count after which temperature switches to 0 (greedy)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints and restart from scratch",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="Override the starting iteration number",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        num_iterations=args.iterations,
        board_size=args.board_size,
        self_play_games=args.games,
        simulations=args.simulations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_games=args.eval_games,
        promotion_threshold=args.promotion_threshold,
        learning_rate=args.lr,
        temperature=args.temperature,
        temp_threshold=args.temp_threshold,
        force_fresh=args.fresh,
        start_iteration=args.start_iteration,
    )
