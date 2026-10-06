"""Parallel candidate-vs-current evaluation (one game per worker task)."""

import contextlib
import io
import multiprocessing as mp

import numpy as np
import torch

_W = {}


def _init(cur_path, cand_path, board_size, simulations):
    torch.set_num_threads(1)
    np.random.seed(int.from_bytes(__import__("os").urandom(4), "little"))
    from evaluation.model_match import ModelPlayer
    from network.model_loader import load_model

    _W["board_size"] = board_size
    _W["cur"] = ModelPlayer(load_model(cur_path, board_size=board_size), board_size, simulations)
    _W["cand"] = ModelPlayer(load_model(cand_path, board_size=board_size), board_size, simulations)


OPENING_SAMPLED_MOVES = 8  # sample these from visit counts so games differ
MAX_MOVES = 500            # same cap as self-play


def _select(player, game, sample):
    from mcts.mcts import MCTS

    searcher = MCTS(model=player.model, game=game, board_size=player.board_size,
                    simulations=player.simulations)
    best = searcher.search(game)
    if not sample or searcher.root is None or not searcher.root.children:
        return best
    moves = list(searcher.root.children.keys())
    visits = np.array([c.visit_count for c in searcher.root.children.values()], dtype=np.float64)
    if visits.sum() <= 0:
        return best
    return int(moves[np.random.choice(len(moves), p=visits / visits.sum())])


def _play(game_idx):
    from environment.go_game import GoGame

    candidate_black = game_idx % 2 == 0
    black, white = (_W["cand"], _W["cur"]) if candidate_black else (_W["cur"], _W["cand"])
    game = GoGame(board_size=_W["board_size"])
    with contextlib.redirect_stdout(io.StringIO()):
        for move_no in range(MAX_MOVES):
            if game.is_terminal():
                break
            player = black if game.current_player == 1 else white
            move = _select(player, game, move_no < OPENING_SAMPLED_MOVES)
            if not game.is_legal(move) or not game.play(move):
                game.play(game.get_pass_action())
        winner = game.get_winner()
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
