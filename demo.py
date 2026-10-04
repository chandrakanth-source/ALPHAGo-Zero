"""
AlphaGo Zero – Final Demonstration  (Day 28)
=============================================

Demonstrates the complete AI in action:

    Start program
      ↓
    Load model  (or bootstrap a random one)
      ↓
    MCTS searches moves
      ↓
    AI chooses move
      ↓
    Two AIs play against each other  (or AI vs random)
      ↓
    Game continues until terminal
      ↓
    Print winner + board state

Usage
-----
AI vs AI (both use the same latest model)::

    python demo.py

AI vs AI with a specific model::

    python demo.py --model models/model_iteration_3.pt

Show only the final board::

    python demo.py --quiet

Use fewer MCTS simulations for speed::

    python demo.py --sims 10

Play on a smaller board for faster demo::

    python demo.py --board-size 5 --sims 20
"""

import argparse
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment.go_game import GoGame
from evaluation.model_match import ModelPlayer, play_game
from mcts.mcts import MCTS
from mcts.network_Evaluator import NetworkEvaluator
from network.model_loader import load_model
from network.network import GoNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = {0: ".", 1: "B", -1: "W"}
PLAYER_NAME = {1: "Black (B)", -1: "White (W)"}

MODELS_DIR = "models"
LATEST_MODEL = os.path.join(MODELS_DIR, "latest_model.pt")


def banner(text, width=70):
    print()
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def print_board(game, move_history=None, last_move=None):
    """Pretty-print the Go board with coordinates."""
    size = game.board_size
    col_labels = "  " + " ".join(
        chr(ord("A") + c) for c in range(size)
    )
    print(col_labels)
    for r in range(size):
        row_str = f"{r + 1:>2} "
        for c in range(size):
            stone = game.board[r, c]
            ch = SYMBOLS[int(stone)]
            if last_move is not None and last_move == (r, c):
                # Highlight last move with brackets
                ch = f"[{ch}]"
                row_str += ch
            else:
                row_str += f" {ch} "
        print(row_str + f" {r + 1}")
    print(col_labels)


def load_or_bootstrap(model_path, board_size):
    """Load a model from *model_path*, or create a random one if not found.

    If the checkpoint was saved for a different board size, a fresh random
    model is used instead (rather than crashing).
    """
    if model_path and os.path.exists(model_path):
        try:
            print(f"  Loading model: {model_path}")
            return load_model(model_path, board_size=board_size)
        except RuntimeError as exc:
            print(
                f"  WARNING: Could not load '{model_path}' for board_size={board_size} "
                f"({exc.__class__.__name__}: {exc!s:.120}).  "
                f"Using a fresh random model instead."
            )

    print(f"  Bootstrapping a random model for board_size={board_size}")
    model = GoNetwork(board_size=board_size)
    # Save only if we are using the default path (don't overwrite an existing
    # model that might be for a different board size).
    if model_path == LATEST_MODEL:
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(model.state_dict(), LATEST_MODEL)
        print(f"  Bootstrap model saved -> {LATEST_MODEL}")
    return model


# ---------------------------------------------------------------------------
# Demo: AI vs AI game with move-by-move display
# ---------------------------------------------------------------------------

def demo_game(
    model_black,
    model_white,
    board_size=12,
    simulations=50,
    max_moves=200,
    quiet=False,
    delay=0.0,
):
    """
    Play one complete AI vs AI game, printing board after each move.

    Args:
        model_black:  GoNetwork playing as Black (player +1).
        model_white:  GoNetwork playing as White (player -1).
        board_size:   Board side length.
        simulations:  MCTS simulations per move.
        max_moves:    Move cap.
        quiet:        If True, only print the final board + result.
        delay:        Seconds to pause between moves (for animated viewing).

    Returns:
        dict: game result with keys ``winner``, ``black_score``, ``white_score``,
              ``total_moves``.
    """
    game = GoGame(board_size=board_size)

    players = {
        1:  ModelPlayer(model=model_black, board_size=board_size, simulations=simulations),
        -1: ModelPlayer(model=model_white, board_size=board_size, simulations=simulations),
    }

    move_number = 0
    last_move_pos = None

    if not quiet:
        banner(f"ALPHAGO ZERO DEMO  —  {board_size}×{board_size} Board")
        print(f"  Black: MCTS ({simulations} sims)  |  White: MCTS ({simulations} sims)")
        print(f"  Max moves: {max_moves}")
        print()
        print_board(game)

    while not game.is_terminal() and move_number < max_moves:
        current = game.current_player
        player = players[current]

        t0 = time.time()
        move = player.select_move(game)
        elapsed = time.time() - t0

        # Decode move for display
        if move == game.get_pass_action():
            move_desc = "PASS"
            last_move_pos = None
        else:
            row, col = divmod(move, board_size)
            col_label = chr(ord("A") + col)
            move_desc = f"{col_label}{row + 1}"
            last_move_pos = (row, col)

        # Apply move
        if not game.is_legal(move):
            game.pass_move()
            move_desc = "PASS (forced)"
        else:
            game.play(move)

        move_number += 1

        if not quiet:
            print(
                f"\n  Move {move_number:>3}: "
                f"{PLAYER_NAME[current]:>14}  →  {move_desc:<10}  "
                f"({elapsed:.2f}s)"
            )
            print_board(game, last_move=last_move_pos)

            if delay > 0:
                time.sleep(delay)

    # ---- Result ----
    result = game.get_result()
    winner = result["winner"]
    black_score = result["black_score"]
    white_score = result["white_score"]

    banner("GAME OVER")
    print()
    print_board(game, last_move=last_move_pos)
    print()
    print(f"  Total moves  : {move_number}")
    print(f"  Black score  : {black_score}")
    print(f"  White score  : {white_score}")
    print()
    if winner == 1:
        print("  *** BLACK WINS! ***")
    elif winner == -1:
        print("  *** WHITE WINS! ***")
    else:
        print("  === DRAW ===")
    print()

    return {
        "winner": winner,
        "black_score": black_score,
        "white_score": white_score,
        "total_moves": move_number,
    }


# ---------------------------------------------------------------------------
# Checkpoint verification helper
# ---------------------------------------------------------------------------

def print_model_info(model, label="Model"):
    """Print model parameter count and a sample inference."""
    n = sum(p.numel() for p in model.parameters())
    print(f"  {label}: {n:,} parameters")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AlphaGo Zero – final demonstration (AI vs AI)"
    )
    p.add_argument("--model", default=LATEST_MODEL,
                   help=f"Path to model weights (default: {LATEST_MODEL})")
    p.add_argument("--model-black", default=None,
                   help="Path to Black's model (overrides --model)")
    p.add_argument("--model-white", default=None,
                   help="Path to White's model (overrides --model)")
    p.add_argument("--board-size", type=int, default=12,
                   help="Board side length (default: 12)")
    p.add_argument("--sims", type=int, default=50,
                   help="MCTS simulations per move (default: 50)")
    p.add_argument("--max-moves", type=int, default=200,
                   help="Move cap per game (default: 200)")
    p.add_argument("--quiet", action="store_true",
                   help="Only show the final board and result")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to pause between moves (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()

    banner("ALPHAGO ZERO – PROJECT DEMO")

    print()
    print("  Loading models …")

    black_path = args.model_black or args.model
    white_path = args.model_white or args.model

    model_black = load_or_bootstrap(black_path, args.board_size)
    model_white = load_or_bootstrap(white_path, args.board_size)

    model_black.eval()
    model_white.eval()

    print_model_info(model_black, "Black's model")
    print_model_info(model_white, "White's model")

    demo_game(
        model_black=model_black,
        model_white=model_white,
        board_size=args.board_size,
        simulations=args.sims,
        max_moves=args.max_moves,
        quiet=args.quiet,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
