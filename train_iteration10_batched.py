"""
Resumable Online-Batched Training for Iteration 10.
===================================================
Features:
- Starts from models/latest_model.pt (Iteration 9 baseline).
- Plays games in batches (default 25 games/batch).
- Online Model Update: After each batch, trains on recent replay data
  and saves an atomic checkpoint to training/checkpoints/iter10_online_checkpoint.pt.
- Fully Resumable: If interrupted or laptop closes, re-running the script
  automatically loads the latest checkpoint and skips already completed batches.
- Final Polish & Arena Evaluation: Once all batches finish (1000 games),
  performs final training epochs, saves models/model_iteration_10.pt,
  evaluates vs latest_model.pt, and promotes if win rate >= 55%.
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import main as pipeline
from config.training_config import BATCH_SIZE, BOARD_SIZE, PROMOTION_THRESHOLD
from evaluation.parallel_eval import evaluate_parallel
from evaluation.promotion import PromotionManager
from network.model_loader import load_model
from network.network import GoNetwork
from self_play.parallel import default_workers, play_games_parallel
from training.dataset import SelfPlayDataset
from training.trainer import Trainer

STATE_FILE = os.path.join(PROJECT_ROOT, "training", "iter10_batched_state.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "training", "checkpoints", "iter10_online_checkpoint.pt")
BATCH_DIR = os.path.join(PROJECT_ROOT, "data", "batches")
ITERATION = 10
EVAL_GAMES = 40


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def verify_model(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    net = GoNetwork(board_size=BOARD_SIZE)
    net.load_state_dict(sd, strict=True)
    return net


def load_training_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"Warning: could not read {STATE_FILE}: {e}")
    return {
        "iteration": ITERATION,
        "completed_batches": [],
        "total_games_played": 0,
        "status": "in_progress",
    }


def save_training_state(state):
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_FILE)


def main():
    parser = argparse.ArgumentParser(description="Resumable batched training for Iteration 10")
    parser.add_argument("--games", type=int, default=1000, help="Total self-play games (default: 1000)")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move (default: 100)")
    parser.add_argument("--batch-size", type=int, default=25, help="Games per batch (default: 25)")
    parser.add_argument("--replay-window", type=int, default=4, help="Batches in sliding replay window for online training")
    parser.add_argument("--workers", type=int, default=default_workers(), help="Parallel worker processes")
    parser.add_argument("--epochs-per-batch", type=int, default=1, help="Training epochs after each batch (default: 1)")
    parser.add_argument("--final-epochs", type=int, default=8, help="Final epochs on consolidated dataset (default: 8)")
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run (4 games, 5 sims)")
    args = parser.parse_args()

    if args.smoke:
        args.games = 4
        args.sims = 5
        args.batch_size = 2
        args.final_epochs = 1
        args.epochs_per_batch = 1

    os.makedirs(os.path.join(PROJECT_ROOT, "training", "checkpoints"), exist_ok=True)
    os.makedirs(BATCH_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

    state = load_training_state()
    total_batches = -(-args.games // args.batch_size)

    log("=" * 70)
    log(f" ALPHAGO ZERO — ITERATION {ITERATION} BATCHED TRAINING")
    log("=" * 70)
    log(f" Board size:          {BOARD_SIZE}x{BOARD_SIZE}")
    log(f" Total games:         {args.games} in {total_batches} batches of {args.batch_size}")
    log(f" MCTS simulations:    {args.sims} sims / move")
    log(f" Workers:             {args.workers} CPU processes")
    log(f" Checkpoint path:     {CHECKPOINT_PATH}")
    log(f" State file:          {STATE_FILE}")
    log("=" * 70)

    # 1. Determine starting model
    baseline_path = pipeline.LATEST_MODEL_PATH
    if not os.path.exists(baseline_path):
        fallback_path = os.path.join(PROJECT_ROOT, "models", "model_iteration_9.pt")
        if os.path.exists(fallback_path):
            shutil.copy2(fallback_path, baseline_path)
            log(f"Copied {fallback_path} -> {baseline_path}")
        else:
            raise FileNotFoundError(f"Neither {baseline_path} nor {fallback_path} found!")

    verify_model(baseline_path)

    # If an online checkpoint exists from an interrupted run, resume from it
    if os.path.exists(CHECKPOINT_PATH):
        current_model_path = CHECKPOINT_PATH
        log(f"Resuming working model from online checkpoint: {CHECKPOINT_PATH}")
    else:
        current_model_path = baseline_path
        log(f"Starting working model from baseline: {baseline_path}")

    model = load_model(current_model_path, board_size=BOARD_SIZE)
    trainer = Trainer(model, learning_rate=0.001, checkpoint_dir=os.path.join(PROJECT_ROOT, "training", "checkpoints"))

    # 2. Batched Self-Play Loop
    completed_batches = set(state.get("completed_batches", []))

    for b in range(total_batches):
        batch_filename = f"iter{ITERATION}_batch_{b:03d}.pt"
        batch_path = os.path.join(BATCH_DIR, batch_filename)
        n_games = min(args.batch_size, args.games - b * args.batch_size)

        if b in completed_batches and os.path.exists(batch_path):
            log(f"Batch {b + 1}/{total_batches} ({n_games} games) already completed — skipping.")
            continue

        log("-" * 70)
        log(f"Starting Batch {b + 1}/{total_batches} ({n_games} games, {args.sims} sims/move)...")
        t0 = time.time()

        # Generate self-play games using the current model weights
        results = play_games_parallel(
            model=model,
            num_games=n_games,
            simulations=args.sims,
            board_size=BOARD_SIZE,
            workers=args.workers,
        )

        batch_examples = [ex for game in results for ex in game]
        elapsed = time.time() - t0

        # Save batch atomically
        tmp_batch = batch_path + ".tmp"
        torch.save(batch_examples, tmp_batch)
        os.replace(tmp_batch, batch_path)

        completed_batches.add(b)
        state["completed_batches"] = sorted(list(completed_batches))
        state["total_games_played"] = len(completed_batches) * args.batch_size

        eta_hours = (elapsed * (total_batches - (b + 1))) / 3600.0
        log(f"Batch {b + 1}/{total_batches} completed in {elapsed / 60:.1f} min "
            f"({len(batch_examples)} positions). ETA remaining: ~{eta_hours:.1f} h")

        # 3. Online model update across recent replay window
        replay_start = max(0, b - args.replay_window + 1)
        recent_examples = []
        for rb in range(replay_start, b + 1):
            r_path = os.path.join(BATCH_DIR, f"iter{ITERATION}_batch_{rb:03d}.pt")
            if os.path.exists(r_path):
                recent_examples.extend(torch.load(r_path, weights_only=False))

        if recent_examples:
            log(f"Online training update on {len(recent_examples)} recent positions ({args.epochs_per_batch} epoch(s))...")
            dataset = SelfPlayDataset(recent_examples)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for ep in range(args.epochs_per_batch):
                metrics = trainer.train_epoch(dataloader)
                log(f"  Online Epoch {ep + 1}/{args.epochs_per_batch} | Loss: {metrics['loss']:.4f} "
                    f"(Policy: {metrics['policy_loss']:.4f}, Value: {metrics['value_loss']:.4f})")

            # Save model checkpoint atomically
            tmp_ckpt = CHECKPOINT_PATH + ".tmp"
            torch.save(model.state_dict(), tmp_ckpt)
            os.replace(tmp_ckpt, CHECKPOINT_PATH)
            state["latest_checkpoint"] = CHECKPOINT_PATH
            state["latest_loss"] = metrics["loss"]

        save_training_state(state)
        log(f">> Checkpoint saved! If you close your laptop, training resumes from Batch {b + 2} without data loss.")

    log("=" * 70)
    log("ALL SELF-PLAY BATCHES COMPLETED!")
    log("=" * 70)

    # 4. Consolidate Full Dataset for Final Training
    final_data_path = os.path.join(PROJECT_ROOT, "data", f"self_play_iteration_{ITERATION}.pt")
    if os.path.exists(final_data_path) and os.path.getsize(final_data_path) > 1000000:
        log(f"Consolidated dataset already exists: {final_data_path}")
    else:
        all_examples = []
        for b in range(total_batches):
            b_path = os.path.join(BATCH_DIR, f"iter{ITERATION}_batch_{b:03d}.pt")
            if os.path.exists(b_path):
                all_examples.extend(torch.load(b_path, weights_only=False))
        torch.save(all_examples, final_data_path)
        log(f"Merged {len(all_examples)} positions into {final_data_path}")

    # 5. Final Training Epochs
    log(f"Running final consolidation training ({args.final_epochs} epochs)...")
    candidate_path = pipeline.run_training(
        data_path=final_data_path,
        iteration=ITERATION,
        previous_model_path=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else baseline_path,
        epochs=args.final_epochs,
        batch_size=BATCH_SIZE,
        board_size=BOARD_SIZE,
        resume=True,
    )
    verify_model(candidate_path)

    # 6. Evaluation Match (Candidate vs Current Best Baseline)
    eval_sims = args.sims
    eval_games = 4 if args.smoke else EVAL_GAMES
    log("=" * 70)
    log(f" EVALUATING CANDIDATE (Iteration {ITERATION}) vs BASELINE ({os.path.basename(baseline_path)})")
    log(f" Games: {eval_games} | Simulations: {eval_sims}")
    log("=" * 70)

    result = evaluate_parallel(
        cur_path=baseline_path,
        cand_path=candidate_path,
        num_games=eval_games,
        simulations=eval_sims,
        board_size=BOARD_SIZE,
        workers=args.workers,
    )

    # Record history
    hist_path = os.path.join(PROJECT_ROOT, "data", "evaluation_history.json")
    hist = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
        except Exception:
            hist = []

    hist.append({
        "timestamp": int(time.time()),
        "model_a": os.path.basename(candidate_path),
        "model_b": os.path.basename(baseline_path),
        "sims_a": eval_sims,
        "sims_b": eval_sims,
        "num_games": eval_games,
        "model_a_wins": result["wins"],
        "model_b_wins": result["losses"],
        "draws": result["draws"],
        "win_rate_a": round(result["win_rate"] * 100, 1),
        "win_rate_b": round(result["losses"] / eval_games * 100, 1) if eval_games else 0.0,
    })
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    # 7. Promotion
    promoted = result["win_rate"] >= PROMOTION_THRESHOLD
    promoter = PromotionManager(models_dir=os.path.join(PROJECT_ROOT, "models"))
    if promoted:
        promoter.promote(candidate_path)
        log(f"SUCCESS: Candidate PROMOTED! (Win rate: {result['win_rate']:.1%} >= {PROMOTION_THRESHOLD:.0%})")
        log(f"Updated {pipeline.LATEST_MODEL_PATH} with upgraded Iteration {ITERATION} model.")
    else:
        promoter.reject(candidate_path)
        log(f"Candidate REJECTED (Win rate: {result['win_rate']:.1%} < {PROMOTION_THRESHOLD:.0%}).")

    state["status"] = "completed"
    state["promoted"] = bool(promoted)
    state["eval_result"] = result
    save_training_state(state)
    log("ITERATION 10 TRAINING RUN COMPLETE!")


if __name__ == "__main__":
    main()
