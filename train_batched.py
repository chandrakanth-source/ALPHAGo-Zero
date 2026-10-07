"""
Batched, resumable curriculum training (stages 2-5).

Self-play runs in parallel worker processes and is saved every --batch-size
games (default 25), so a crash or reboot loses at most one batch.  Re-running
the same command resumes where it stopped.

Usage:
    python train_batched.py --stage 2            # real run
    python train_batched.py --stage 2 --smoke    # tiny isolated test run
    python train_batched.py --all                # stages 2..5 in order
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import main as pipeline
from config.training_config import BATCH_SIZE, BOARD_SIZE, PROMOTION_THRESHOLD
from evaluation.parallel_eval import evaluate_parallel
from network.model_loader import load_model
from self_play.parallel import default_workers, play_games_parallel

# stage -> iterations (fixed numbers), games/iter, sims/move, epochs
STAGES = {
    2: dict(iterations=[3, 4], games=400, sims=35, epochs=4),
    3: dict(iterations=[5, 6], games=600, sims=50, epochs=5),
    4: dict(iterations=[7, 8], games=850, sims=80, epochs=6),
    5: dict(iterations=[9, 10], games=1000, sims=100, epochs=8),
}
EVAL_GAMES = 40


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def load_state(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"done_iterations": {}}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def selfplay_batches(model, iteration, games, sims, batch_size, workers, batch_dir):
    os.makedirs(batch_dir, exist_ok=True)
    n_batches = -(-games // batch_size)
    for b in range(n_batches):
        path = os.path.join(batch_dir, f"iter{iteration}_batch{b:03d}.pt")
        if os.path.exists(path):
            log(f"batch {b + 1}/{n_batches} already done - skipping")
            continue
        n = min(batch_size, games - b * batch_size)
        t0 = time.time()
        results = play_games_parallel(model, n, sims, board_size=BOARD_SIZE, workers=workers)
        examples = [ex for g in results for ex in g]
        torch.save(examples, path + ".tmp")
        os.replace(path + ".tmp", path)
        el = time.time() - t0
        log(f"batch {b + 1}/{n_batches}: {n} games, {len(examples)} examples, {el / 60:.1f} min "
            f"(ETA this iteration ~{el * (n_batches - b - 1) / 3600:.1f} h)")


def verify_model(path):
    """load_model silently falls back to random weights; refuse that here."""
    from network.network import GoNetwork

    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    GoNetwork(board_size=BOARD_SIZE).load_state_dict(sd, strict=True)


def run_iteration(iteration, cfg, args, state, state_path, batch_dir):
    key = str(iteration)
    if key in state["done_iterations"]:
        log(f"iteration {iteration} already complete - skipping")
        return
    log(f"=== ITERATION {iteration}: {cfg['games']} games, {cfg['sims']} sims, {cfg['epochs']} epochs ===")
    best_path = pipeline.LATEST_MODEL_PATH
    verify_model(best_path)
    model = load_model(best_path, board_size=BOARD_SIZE)

    selfplay_batches(model, iteration, cfg["games"], cfg["sims"], args.batch_size, args.workers, batch_dir)

    # Replay window: train on this iteration's games plus the previous
    # (replay_window - 1) iterations' games, as AlphaGo Zero does.
    window = range(iteration - args.replay_window + 1, iteration + 1)
    examples = []
    for it in window:
        n_before = len(examples)
        for f in sorted(os.listdir(batch_dir)):
            if f.startswith(f"iter{it}_batch") and f.endswith(".pt"):
                examples.extend(torch.load(os.path.join(batch_dir, f), weights_only=False))
        if len(examples) > n_before:
            log(f"replay window: iteration {it} -> {len(examples) - n_before} examples")
    data_path = os.path.join(pipeline.DATA_DIR, f"self_play_iteration_{iteration}.pt")
    torch.save(examples, data_path)
    log(f"merged {len(examples)} examples -> {data_path}")

    cand_path = pipeline.run_training(
        data_path=data_path, iteration=iteration, previous_model_path=best_path,
        epochs=cfg["epochs"], batch_size=BATCH_SIZE, board_size=BOARD_SIZE, resume=True,
    )

    verify_model(cand_path)
    result = evaluate_parallel(best_path, cand_path, cfg["eval_games"], cfg["sims"],
                               board_size=BOARD_SIZE, workers=args.workers)
    hist_path = os.path.join(pipeline.DATA_DIR, "evaluation_history.json")
    hist = []
    if os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as f:
            hist = json.load(f)
    n = cfg["eval_games"]
    hist.append({
        "timestamp": int(time.time()), "model_a": os.path.basename(cand_path),
        "model_b": os.path.basename(best_path), "sims_a": cfg["sims"], "sims_b": cfg["sims"],
        "num_games": n, "model_a_wins": result["wins"], "model_b_wins": result["losses"],
        "draws": result["draws"], "win_rate_a": round(result["win_rate"] * 100, 1),
        "win_rate_b": round(result["losses"] / n * 100, 1),
    })
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    promoted = pipeline.run_promotion(cand_path, result, PROMOTION_THRESHOLD)
    state["done_iterations"][key] = {"promoted": bool(promoted), "win_rate": result["win_rate"]}
    save_state(state_path, state)
    log(f"iteration {iteration} done - {'PROMOTED' if promoted else 'REJECTED'} ({result['win_rate']:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=sorted(STAGES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--replay-window", type=int, default=2,
                    help="train on the last N iterations' self-play games")
    ap.add_argument("--smoke", action="store_true", help="tiny run in an isolated temp dir")
    args = ap.parse_args()
    stages = [2, 3, 4, 5] if args.all else ([args.stage] if args.stage else [])
    if not stages:
        ap.error("give --stage N or --all")

    if args.smoke:
        root = os.path.join(PROJECT_ROOT, "training", "smoke_run")
        shutil.rmtree(root, ignore_errors=True)
        for d in ("models", "data", "checkpoints"):
            os.makedirs(os.path.join(root, d))
        shutil.copy2(pipeline.LATEST_MODEL_PATH, os.path.join(root, "models", "latest_model.pt"))
        pipeline.MODELS_DIR = os.path.join(root, "models")
        pipeline.DATA_DIR = os.path.join(root, "data")
        pipeline.CHECKPOINT_DIR = os.path.join(root, "checkpoints")
        pipeline.LATEST_MODEL_PATH = os.path.join(root, "models", "latest_model.pt")
        state_path = os.path.join(root, "state.json")
        batch_dir = os.path.join(root, "batches")
        args.batch_size = 2
    else:
        state_path = os.path.join(PROJECT_ROOT, "training", "batched_state.json")
        batch_dir = os.path.join(PROJECT_ROOT, "data", "batches")
        for d in (pipeline.MODELS_DIR, pipeline.DATA_DIR, pipeline.CHECKPOINT_DIR):
            os.makedirs(d, exist_ok=True)

    state = load_state(state_path)
    log(f"workers={args.workers} batch_size={args.batch_size} cuda={torch.cuda.is_available()} smoke={args.smoke}")
    for s in stages:
        cfg = dict(STAGES[s])
        cfg["eval_games"] = EVAL_GAMES
        if args.smoke:
            cfg.update(games=4, sims=5, epochs=1, eval_games=4, iterations=cfg["iterations"][:1])
        else:
            backup = os.path.join(pipeline.MODELS_DIR, f"latest_model.before_stage{s}.pt")
            if not os.path.exists(backup):
                shutil.copy2(pipeline.LATEST_MODEL_PATH, backup)
        log(f"##### STAGE {s} #####")
        for it in cfg["iterations"]:
            run_iteration(it, cfg, args, state, state_path, batch_dir)
    log("ALL REQUESTED STAGES COMPLETE")


if __name__ == "__main__":
    main()
