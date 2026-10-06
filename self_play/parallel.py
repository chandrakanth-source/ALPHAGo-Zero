"""
Parallel self-play: plays games in separate worker processes.

Each worker holds its own copy of the model and runs single-threaded, so N
workers use N CPU cores instead of one process fighting over PyTorch's
intra-op thread pool.  Games are independent, so speed-up is close to linear.
"""

import multiprocessing as mp
import os

import numpy as np
import torch

_WORKER = {}


def _init_worker(state_dict, board_size, simulations, temperature, temp_threshold):
    torch.set_num_threads(1)
    # Distinct RNG stream per worker (spawned processes would otherwise be
    # seeded independently anyway, but make it explicit).
    np.random.seed((os.getpid() * 2654435761) % (2**32))

    from network.network import GoNetwork
    from mcts.network_Evaluator import NetworkEvaluator
    from self_play.self_play import SelfPlay

    model = GoNetwork(board_size=board_size)
    model.load_state_dict(state_dict)
    model.eval()
    _WORKER["sp"] = SelfPlay(
        board_size=board_size,
        simulations=simulations,
        evaluator=NetworkEvaluator(model),
        temperature=temperature,
        temp_threshold=temp_threshold,
        dirichlet_alpha=10.0 / (board_size * board_size),
    )


def _play_one(_):
    return _WORKER["sp"].generate_game()


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)


def play_games_parallel(
    model,
    num_games,
    simulations,
    board_size=12,
    temperature=1.0,
    temp_threshold=30,
    workers=None,
    on_game_done=None,
):
    """
    Play *num_games* self-play games in parallel and return a list with one
    list of (state, policy, value) examples per game.

    on_game_done(game_index, examples) is called in the parent as each game
    finishes (completion order, not submission order).
    """
    workers = min(workers or default_workers(), num_games)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=(state_dict, board_size, simulations, temperature, temp_threshold),
    ) as pool:
        for i, game_data in enumerate(pool.imap_unordered(_play_one, range(num_games))):
            results.append(game_data)
            if on_game_done:
                on_game_done(i, game_data)
    return results
