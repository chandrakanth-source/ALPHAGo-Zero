"""
Day 27 - MCTS simulation benchmarks and correctness checks.

Tests:
  1. MCTS with different simulation counts (10, 20, 50, 100)
  2. Search time scales with simulation count
  3. Policy distribution sums to 1
  4. Selected move is always legal
  5. Root visit counts equal simulations
"""

import time

import numpy as np
import pytest

from environment.go_game import GoGame
from mcts.mcts import MCTS
from network.network import GoNetwork


BOARD_SIZE = 5  # small board for fast tests


@pytest.fixture(scope="module")
def model():
    return GoNetwork(board_size=BOARD_SIZE)


@pytest.fixture
def fresh_game():
    return GoGame(board_size=BOARD_SIZE)


@pytest.mark.parametrize("simulations", [10, 20, 50, 100])
def test_mcts_runs_for_simulation_counts(simulations, model, fresh_game):
    """MCTS must complete and return a legal move for each sim count."""
    mcts = MCTS(
        model=model,
        game=fresh_game,
        board_size=BOARD_SIZE,
        simulations=simulations,
    )
    move = mcts.search(fresh_game)
    assert fresh_game.is_legal(move), (
        f"Illegal move {move} returned with {simulations} simulations"
    )


def test_mcts_search_time_scales_with_simulations(model):
    """100-sim search must take longer than 10-sim search."""
    times = {}
    for sims in [10, 100]:
        game = GoGame(board_size=BOARD_SIZE)
        mcts = MCTS(model=model, game=game, board_size=BOARD_SIZE, simulations=sims)
        t0 = time.time()
        mcts.search(game)
        times[sims] = time.time() - t0

    assert times[100] > times[10], (
        "100-sim search was not slower than 10-sim search"
    )


@pytest.mark.parametrize("simulations", [10, 50, 100])
def test_mcts_policy_sums_to_one(simulations, model, fresh_game):
    """Visit-count policy from get_mcts_policy must sum to 1."""
    from self_play.self_play import SelfPlay

    sp = SelfPlay(
        board_size=BOARD_SIZE,
        simulations=simulations,
        model=model,
    )
    mcts = MCTS(
        model=model,
        game=fresh_game,
        board_size=BOARD_SIZE,
        simulations=simulations,
    )
    mcts.evaluator = sp.evaluator
    mcts.search(fresh_game)

    policy = sp.get_mcts_policy(mcts)
    assert abs(policy.sum() - 1.0) < 1e-5, (
        f"Policy sum {policy.sum()} != 1.0"
    )


@pytest.mark.parametrize("simulations", [10, 50, 100])
def test_root_visit_count_equals_simulations(simulations, model, fresh_game):
    """Total child visits at root should equal simulations."""
    mcts = MCTS(
        model=model,
        game=fresh_game,
        board_size=BOARD_SIZE,
        simulations=simulations,
    )
    mcts.search(fresh_game)

    total_child_visits = sum(
        c.visit_count for c in mcts.root.children.values()
    )
    assert total_child_visits == simulations, (
        f"Expected {simulations} total child visits, got {total_child_visits}"
    )


def test_mcts_returns_legal_move_after_several_moves(model):
    """MCTS should keep returning legal moves as the game progresses."""
    game = GoGame(board_size=BOARD_SIZE)
    for _ in range(5):
        if game.is_terminal():
            break
        mcts = MCTS(model=model, game=game, board_size=BOARD_SIZE, simulations=20)
        move = mcts.search(game)
        assert game.is_legal(move), f"Illegal move {move} returned"
        game.play(move)


def test_mcts_on_near_terminal_game(model):
    """MCTS should return a legal move when game has one pass already."""
    game = GoGame(board_size=BOARD_SIZE)
    game.pass_move()  # one pass; game not over yet
    mcts = MCTS(model=model, game=game, board_size=BOARD_SIZE, simulations=10)
    move = mcts.search(game)
    assert game.is_legal(move), f"Illegal move {move} on near-terminal board"
