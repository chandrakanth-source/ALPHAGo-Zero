"""Parallel candidate-vs-current evaluation (one game per worker task)."""

import contextlib
import io
import multiprocessing as mp

import torch

_W = {}


def _init(cur_path, cand_path, board_size, simulations):
    torch.set_num_threads(1)
    from evaluation.model_match import ModelPlayer
    from network.model_loader import load_model

    _W["board_size"] = board_size
    _W["cur"] = ModelPlayer(load_model(cur_path, board_size=board_size), board_size, simulations)
    _W["cand"] = ModelPlayer(load_model(cand_path, board_size=board_size), board_size, simulations)


def _play(game_idx):
    from environment.go_game import GoGame
    from evaluation.model_match import play_game

    candidate_black = game_idx % 2 == 0
    black, white = (_W["cand"], _W["cur"]) if candidate_black else (_W["cur"], _W["cand"])
    with contextlib.redirect_stdout(io.StringIO()):
        winner = play_game(black, white, GoGame(board_size=_W["board_size"]))
    if winner == 0:
        return "draw", candidate_black
    cand_won = (winner == 1) == candidate_black
    return ("cand" if cand_won else "cur"), candidate_black


def evaluate_parallel(cur_path, cand_path, num_games, simulations, board_size=12, workers=None):
    """Same result dict as main.run_evaluation."""
    from self_play.parallel import default_workers

    workers = min(workers or default_workers(), num_games)
    wins = losses = draws = 0
    with mp.get_context("spawn").Pool(
        workers, initializer=_init, initargs=(cur_path, cand_path, board_size, simulations)
    ) as pool:
        for i, (res, cand_black) in enumerate(pool.imap_unordered(_play, range(num_games)), 1):
            wins += res == "cand"
            losses += res == "cur"
            draws += res == "draw"
            print(f"  Eval game {i}/{num_games} | candidate={'Black' if cand_black else 'White'} | {res}", flush=True)
    rate = wins / num_games if num_games else 0.0
    print(f"  Candidate wins {wins} | current wins {losses} | draws {draws} | win rate {rate:.2%}", flush=True)
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": rate}
