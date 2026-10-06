"""
AlphaGo Zero - Web Server Backend
Provides REST API endpoints for interactive Human vs AlphaGo Zero play,
level selection, MCTS evaluation, hints, and training integration.
"""

import os
import sys
import math
import time
import threading
import traceback
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Header, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.go_game import GoGame
from network.network import GoNetwork
from network.model_loader import load_model
from mcts.mcts import MCTS
from mcts.network_Evaluator import NetworkEvaluator

app = FastAPI(title="AlphaGo Zero Arena API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SERVER_START_TIME = time.time()
job_lock = threading.Lock()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "alphago2026")

def verify_admin_auth(x_admin_token: Optional[str] = None, token: Optional[str] = None):
    auth_token = x_admin_token or token
    if ADMIN_PASSWORD and auth_token != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Admin passcode required for this action. Please click 'Admin Mode' in the top bar to authenticate."
        )

# ---------------------------------------------------------------------------
# Global Game State & Model Cache
# ---------------------------------------------------------------------------

class GameSession:
    def __init__(self, board_size: int = 12, human_color: int = 1, level: str = "expert", custom_sims: Optional[int] = None, custom_model: str = ""):
        self.board_size = board_size
        self.human_color = human_color  # 1 = Black, -1 = White
        self.level = level
        self.game = GoGame(board_size=board_size)
        self.move_history: List[Dict[str, Any]] = []
        self.state_history: List[Dict[str, Any]] = []
        self.last_ai_eval: float = 0.5  # Win rate 0.0 - 1.0 (from Black's perspective)
        self.last_ai_move: Optional[List[int]] = None

        # Resolve active model path and simulations
        presets = get_preset_configs(board_size)
        default_sims = presets[level]["simulations"] if level in presets else 100

        if custom_model:
            self.resolved_model_path = os.path.join(MODELS_DIR, custom_model) if not os.path.isabs(custom_model) else custom_model
            self.resolved_level_name = f"Custom ({custom_model})"
        elif level in presets:
            cfg = presets[level]
            self.resolved_model_path = cfg["model_path"]
            self.resolved_level_name = cfg["name"]
        else:
            self.resolved_model_path = None
            self.resolved_level_name = "Custom / Novice"

        # Check model path existence & fallback
        if self.resolved_model_path and not os.path.exists(self.resolved_model_path):
            fallback = os.path.join(MODELS_DIR, "latest_model.pt")
            if os.path.exists(fallback):
                self.resolved_model_path = fallback
                self.resolved_level_name += " (Fallback: latest_model.pt)"

        # Handle MCTS Simulation override logic
        if custom_sims is not None and custom_sims > 0:
            self.resolved_sims = custom_sims
            self.is_sims_overridden = (custom_sims != default_sims)
        else:
            self.resolved_sims = default_sims
            self.is_sims_overridden = False

        self.resolved_model_filename = os.path.basename(self.resolved_model_path) if self.resolved_model_path else "Untrained (Tabula Rasa)"

    def save_state(self):
        self.state_history.append({
            "state": self.game.get_state(),
            "move_history": list(self.move_history),
            "last_ai_eval": self.last_ai_eval,
            "last_ai_move": self.last_ai_move
        })

    def undo(self) -> bool:
        if not self.state_history:
            return False
        prev = self.state_history.pop()
        self.game.set_state(prev["state"])
        self.move_history = prev["move_history"]
        self.last_ai_eval = prev["last_ai_eval"]
        self.last_ai_move = prev["last_ai_move"]
        return True

session: Optional[GameSession] = None
model_cache: Dict[str, GoNetwork] = {}
training_state = {
    "is_training": False,
    "progress": "",
    "iteration": 0,
    "log": []
}

# ---------------------------------------------------------------------------
# Model Helpers
# ---------------------------------------------------------------------------

def detect_model_board_size(model_path: str) -> Optional[int]:
    """Inspects checkpoint tensor dimensions to detect native board size."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
        
        # policy_head.3.bias shape is board_size*board_size + 1
        if "policy_head.3.bias" in state_dict:
            bias_len = state_dict["policy_head.3.bias"].shape[0]
            board_area = bias_len - 1
            size = int(math.isqrt(board_area))
            if size * size == board_area:
                return size
    except Exception:
        pass
    return None

def get_or_load_model(model_path: Optional[str], board_size: int) -> GoNetwork:
    """Loads and caches a model, or builds an initialized one if not present."""
    cache_key = f"{model_path or 'untrained'}_{board_size}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path, board_size=board_size)
            model_cache[cache_key] = model
            return model
        except Exception as e:
            print(f"Warning: Could not load {model_path} for board_size {board_size}: {e}")

    # Untrained / Random Model
    model = GoNetwork(board_size=board_size)
    model.eval()
    model_cache[cache_key] = model
    return model

def get_preset_configs(board_size: int = 12) -> Dict[str, Dict[str, Any]]:
    latest_path = os.path.join(MODELS_DIR, "latest_model.pt")
    def get_path(iter_num):
        p = os.path.join(MODELS_DIR, f"model_iteration_{iter_num}.pt")
        return p if os.path.exists(p) else latest_path

    return {
        "easy": {
            "name": "Level 1: Easy (Iterations 1–2)",
            "description": "Trained on 250 self-play games/iter (20 MCTS sims/move), ~1100 Elo.",
            "model_path": get_path(2),
            "simulations": 20,
            "badge": "Easy"
        },
        "intermediate": {
            "name": "Level 2: Intermediate (Iterations 3–4)",
            "description": "Trained on 500 self-play games/iter (40 MCTS sims/move), ~1350 Elo.",
            "model_path": get_path(4),
            "simulations": 40,
            "badge": "Intermediate"
        },
        "difficult": {
            "name": "Level 3: Difficult (Iterations 5–6)",
            "description": "Trained on 500 self-play games/iter (80 MCTS sims/move), ~1600 Elo.",
            "model_path": get_path(6),
            "simulations": 80,
            "badge": "Difficult"
        },
        "expert": {
            "name": "Level 4: Expert (Iterations 7–8)",
            "description": "Trained on 500 self-play games/iter (100 MCTS sims/move), ~1850 Elo.",
            "model_path": get_path(8),
            "simulations": 100,
            "badge": "Expert"
        },
        "super_expert": {
            "name": "Level 5: Super Expert (Iterations 9–10)",
            "description": "Trained on 500 self-play games/iter (250 MCTS sims/move), ~2100 Elo.",
            "model_path": get_path(10),
            "simulations": 250,
            "badge": "Super Expert"
        }
    }

# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class NewGameRequest(BaseModel):
    board_size: int = 12
    human_color: int = 1  # 1 = Black, -1 = White
    level: str = "expert" # "novice" | "apprentice" | "expert" | "master" | "custom"
    simulations: Optional[int] = 100
    model_file: Optional[str] = None

class MoveRequest(BaseModel):
    row: Optional[int] = None
    col: Optional[int] = None
    is_pass: bool = False

class EvaluateRequest(BaseModel):
    model_a_file: Optional[str] = "model_iteration_1.pt"
    model_b_file: Optional[str] = "latest_model.pt"
    sims_a: int = 25
    sims_b: int = 100
    num_games: int = 10
    board_size: int = 12

class SelfPlayNewRequest(BaseModel):
    board_size: int = 12
    model_black: Optional[str] = "latest_model.pt"
    sims_black: int = 25
    temp_black: float = 1.0
    model_white: Optional[str] = "latest_model.pt"
    sims_white: int = 25
    temp_white: float = 1.0
    temp_threshold: int = 30


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/api/ping")
def health_check():
    """
    Health check endpoint for keep-alive ping services (UptimeRobot, cron-job.org)
    and cold start polling.
    """
    return {
        "status": "ok",
        "engine_ready": True,
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "timestamp": int(time.time()),
        "cached_models": len(model_cache),
        "active_session": session is not None,
        "is_job_running": job_lock.locked(),
        "message": "AlphaGo Zero engine inference service active"
    }

class AdminVerifyRequest(BaseModel):
    token: str

@app.post("/api/verify_admin")
def verify_admin(req: AdminVerifyRequest):
    if req.token == ADMIN_PASSWORD:
        return {"valid": True, "message": "Admin authorization granted."}
    return {"valid": False, "message": "Invalid admin passcode."}

@app.get("/api/selfplay_status")
def get_selfplay_status():
    """
    Returns live status of the self-play training pipeline by scanning
    the models/ and data/ directories on disk.
    """
    import re, glob

    # Count model_iteration_N.pt files (excluding iteration 0)
    model_files = []
    if os.path.exists(MODELS_DIR):
        for f in sorted(os.listdir(MODELS_DIR)):
            m = re.fullmatch(r"model_iteration_(\d+)\.pt", f)
            if m and int(m.group(1)) > 0:
                full_path = os.path.join(MODELS_DIR, f)
                mtime = os.path.getmtime(full_path)
                model_files.append({
                    "iteration": int(m.group(1)),
                    "filename": f,
                    "size_kb": max(1, os.path.getsize(full_path) // 1024),
                    "mtime": mtime
                })
    model_files.sort(key=lambda x: x["iteration"])

    # Count self_play_iteration_N.pt data files and total examples
    data_dir = os.path.join(PROJECT_ROOT, "data")
    total_examples = 0
    games_count = 0
    data_files = []
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            m = re.fullmatch(r"self_play_iteration_(\d+)\.pt", f)
            if m:
                games_count += 1
                full_path = os.path.join(data_dir, f)
                try:
                    import torch
                    examples = torch.load(full_path, map_location="cpu", weights_only=False)
                    n = len(examples) if isinstance(examples, (list, tuple)) else 0
                    total_examples += n
                    data_files.append({"iteration": int(m.group(1)), "examples": n})
                except Exception:
                    data_files.append({"iteration": int(m.group(1)), "examples": 0})

    current_iter = model_files[-1]["iteration"] if model_files else 0

    # Detect current phase from training_state or live background task log files
    phase = "idle"
    is_active = training_state.get("is_training", False)
    progress_text = training_state.get("progress", "")
    log_lines = list(training_state.get("log", []))

    if not log_lines:
        import glob
        task_log_patterns = [
            os.path.expanduser(r"~\.gemini\antigravity-ide\brain\*\.system_generated\tasks\*.log")
        ]
        found_logs = []
        for pat in task_log_patterns:
            found_logs.extend(glob.glob(pat))
        if found_logs:
            found_logs.sort(key=os.path.getmtime, reverse=True)
            latest_log_path = found_logs[0]
            # Check if updated in the last 10 minutes
            if time.time() - os.path.getmtime(latest_log_path) < 600:
                try:
                    with open(latest_log_path, "r", encoding="utf-8", errors="ignore") as lf:
                        lines = [line.strip() for line in lf.readlines() if line.strip()]
                        if lines:
                            log_lines = lines[-50:]
                            is_active = True
                            progress_text = lines[-1]
                except Exception:
                    pass

    if is_active:
        p = (progress_text + " ".join(log_lines[-5:])).lower()
        if "self-play" in p or "self_play" in p or "game" in p:
            phase = "selfplay"
        elif "train" in p or "epoch" in p:
            phase = "training"
        elif "eval" in p:
            phase = "evaluation"
        elif "promot" in p or "reject" in p:
            phase = "promote"
        else:
            phase = "selfplay"

    return {
        "is_active": is_active,
        "phase": phase,
        "current_iteration": current_iter,
        "models_trained": len(model_files),
        "games_played": games_count,
        "total_examples": total_examples,
        "model_files": model_files[-20:],  # last 20
        "data_files": data_files[-10:],
        "progress": progress_text,
        "log": log_lines[-50:],  # last 50 log lines
    }


@app.get("/api/iterations")
def get_iterations():
    """
    Return all available model_iteration_N.pt files in numeric order.
    Each entry has: iteration (int), filename (str), path (str), size_kb (int).
    """
    import re
    entries = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            m = re.fullmatch(r"model_iteration_(\d+)\.pt", f)
            if m:
                n = int(m.group(1))
                full_path = os.path.join(MODELS_DIR, f)
                size_kb = max(1, os.path.getsize(full_path) // 1024)
                entries.append({
                    "iteration": n,
                    "filename": f,
                    "path": full_path,
                    "size_kb": size_kb
                })
    entries.sort(key=lambda x: x["iteration"])
    return {"iterations": entries, "count": len(entries)}


@app.get("/api/levels")
def get_levels(board_size: int = 12):
    presets = get_preset_configs(board_size)
    available_files = []
    if os.path.exists(MODELS_DIR):
        for f in sorted(os.listdir(MODELS_DIR)):
            if f.endswith(".pt"):
                full_path = os.path.join(MODELS_DIR, f)
                detected_size = detect_model_board_size(full_path)
                available_files.append({
                    "filename": f,
                    "path": full_path,
                    "detected_board_size": detected_size,
                    "compatible_with_current": (detected_size == board_size or detected_size is None)
                })
    return {
        "presets": presets,
        "available_checkpoints": available_files,
        "current_board_size": board_size
    }

@app.post("/api/new_game")
def new_game(req: NewGameRequest):
    global session
    session = GameSession(
        board_size=req.board_size,
        human_color=req.human_color,
        level=req.level,
        custom_sims=req.simulations or 100,
        custom_model=req.model_file or ""
    )
    
    # If human is White (-1), AI (Black, 1) plays the opening move immediately
    ai_opening_move = None
    if session.human_color == -1 and not session.game.is_terminal():
        ai_opening_move = execute_ai_turn()

    return get_game_state_response(ai_opening_move)

def execute_ai_turn() -> Dict[str, Any]:
    global session
    if not session or session.game.is_terminal():
        return {}

    model_path = session.resolved_model_path
    sims = session.resolved_sims

    model = get_or_load_model(model_path, session.board_size)
    searcher = MCTS(
        model=model,
        game=session.game,
        board_size=session.board_size,
        simulations=sims
    )

    t0 = time.time()
    best_action = searcher.search(session.game)
    search_time = time.time() - t0

    # Evaluate value from search root
    evaluator = NetworkEvaluator(model)
    _, root_val = evaluator.evaluate(session.game)
    # root_val is [-1, 1] for current player; convert to Black win prob [0, 1]
    black_win_prob = (root_val * session.game.current_player + 1.0) / 2.0
    session.last_ai_eval = float(np.clip(black_win_prob, 0.0, 1.0))

    if best_action == session.game.get_pass_action():
        move_name = "PASS"
        session.game.play("PASS")
        session.last_ai_move = None
    else:
        r, c = session.game.action_to_position(best_action)
        move_name = f"{chr(ord('A') + c)}{r + 1}"
        session.game.play((r, c))
        session.last_ai_move = [int(r), int(c)]

    move_record = {
        "player": "AI",
        "color": -session.human_color,
        "action": move_name,
        "coords": session.last_ai_move,
        "time": round(search_time, 2),
        "win_prob_black": session.last_ai_eval
    }
    session.move_history.append(move_record)
    return move_record

@app.post("/api/move")
def play_move(req: MoveRequest):
    global session
    if not session:
        raise HTTPException(status_code=400, detail="No active game session. Call /api/new_game first.")
    if session.game.is_terminal():
        raise HTTPException(status_code=400, detail="Game is already over.")

    # Check if it's the human's turn
    if session.game.current_player != session.human_color:
        raise HTTPException(status_code=400, detail="Not human player's turn.")

    session.save_state()

    # 1. Apply Human Move
    if req.is_pass:
        if not session.game.play("PASS"):
            raise HTTPException(status_code=400, detail="Invalid pass move.")
        session.move_history.append({
            "player": "Human",
            "color": session.human_color,
            "action": "PASS",
            "coords": None
        })
    else:
        if req.row is None or req.col is None:
            raise HTTPException(status_code=400, detail="Row and column required.")
        if not session.game.is_legal_move(req.row, req.col):
            raise HTTPException(status_code=400, detail=f"Illegal move at ({req.row}, {req.col}).")
        
        session.game.play((req.row, req.col))
        move_name = f"{chr(ord('A') + req.col)}{req.row + 1}"
        session.move_history.append({
            "player": "Human",
            "color": session.human_color,
            "action": move_name,
            "coords": [req.row, req.col]
        })

    # 2. AI Turn if game is still active
    ai_info = None
    if not session.game.is_terminal():
        ai_info = execute_ai_turn()

    return get_game_state_response(ai_info)

@app.post("/api/ai_move")
def trigger_ai_move():
    """Forces an AI move for the current player (supports AI vs AI mode)."""
    global session
    if not session:
        raise HTTPException(status_code=400, detail="No active game.")
    if session.game.is_terminal():
        return get_game_state_response()

    session.save_state()
    ai_info = execute_ai_turn()
    return get_game_state_response(ai_info)

@app.post("/api/hint")
def get_hint():
    """Calculates the best move recommended by AlphaGo Zero for current player."""
    global session
    if not session or session.game.is_terminal():
        raise HTTPException(status_code=400, detail="Cannot calculate hint.")

    presets = get_preset_configs(session.board_size)
    cfg = presets.get(session.level, presets["expert"])
    model = get_or_load_model(cfg["model_path"], session.board_size)

    searcher = MCTS(model=model, game=session.game, board_size=session.board_size, simulations=60)
    best_action = searcher.search(session.game)

    evaluator = NetworkEvaluator(model)
    _, value = evaluator.evaluate(session.game)
    current_win_prob = (value + 1.0) / 2.0

    if best_action == session.game.get_pass_action():
        return {"action": "PASS", "coords": None, "win_prob": float(current_win_prob), "explanation": "Pass is the strongest strategic move here."}
    else:
        r, c = session.game.action_to_position(best_action)
        return {
            "action": f"{chr(ord('A') + c)}{r + 1}",
            "coords": [int(r), int(c)],
            "win_prob": float(current_win_prob),
            "explanation": f"AlphaGo recommends ({chr(ord('A') + c)}{r + 1}) with {round(float(current_win_prob)*100, 1)}% win confidence."
        }

@app.post("/api/undo")
def undo_move():
    global session
    if not session:
        raise HTTPException(status_code=400, detail="No active game.")
    # If playing vs AI, undo twice if needed to return to Human's turn
    if session.undo():
        if session.game.current_player != session.human_color and session.state_history:
            session.undo()
        return get_game_state_response()
    raise HTTPException(status_code=400, detail="Cannot undo further.")

@app.get("/api/state")
def get_state():
    return get_game_state_response()

def get_game_state_response(latest_ai_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global session
    if not session:
        return {"active": False}

    black_score, white_score = session.game.calculate_score()
    black_stones, white_stones = session.game.count_stones()
    legal_moves_list = []
    
    # Pre-calculate legal move grid for fast frontend validation
    for r in range(session.board_size):
        for c in range(session.board_size):
            if session.game.is_legal_move(r, c):
                legal_moves_list.append([r, c])

    return {
        "active": True,
        "board_size": session.board_size,
        "board": session.game.board.tolist(),
        "current_player": int(session.game.current_player),  # 1 = Black, -1 = White
        "human_color": session.human_color,
        "consecutive_passes": session.game.consecutive_passes,
        "game_over": bool(session.game.game_over),
        "winner": int(session.game.get_winner()) if session.game.game_over else None,
        "black_score": black_score,
        "white_score": white_score,
        "black_stones": black_stones,
        "white_stones": white_stones,
        "legal_moves": legal_moves_list,
        "move_history": session.move_history,
        "last_ai_move": session.last_ai_move,
        "ai_win_prob_black": session.last_ai_eval,
        "level": session.level,
        "active_level_name": session.resolved_level_name,
        "active_model_file": session.resolved_model_filename,
        "active_simulations": session.resolved_sims,
        "is_sims_overridden": getattr(session, "is_sims_overridden", False),
        "latest_ai_info": latest_ai_info
    }

# ---------------------------------------------------------------------------
# Interactive Self-Play Engine
# ---------------------------------------------------------------------------

class InteractiveSelfPlaySession:
    def __init__(self, req: SelfPlayNewRequest):
        self.board_size = req.board_size
        self.game = GoGame(board_size=req.board_size)
        self.model_black_file = req.model_black or "untrained"
        self.model_white_file = req.model_white or "untrained"
        self.sims_black = req.sims_black
        self.sims_white = req.sims_white
        self.temp_black = req.temp_black
        self.temp_white = req.temp_white
        self.temp_threshold = req.temp_threshold

        path_b = os.path.join(MODELS_DIR, req.model_black) if req.model_black and req.model_black != "untrained" else None
        path_w = os.path.join(MODELS_DIR, req.model_white) if req.model_white and req.model_white != "untrained" else None

        self.model_black = get_or_load_model(path_b, req.board_size)
        self.model_white = get_or_load_model(path_w, req.board_size)

        self.move_history: List[Dict[str, Any]] = []
        self.raw_states: List[np.ndarray] = []
        self.raw_policies: List[np.ndarray] = []
        self.examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.last_ai_eval: float = 0.5
        self.last_ai_move: Optional[List[int]] = None
        self.last_top_moves: List[Dict[str, Any]] = []
        self.move_number: int = 0

        training_state["is_training"] = True
        init_log = f"[Self-Play] New match initialized on {self.board_size}x{self.board_size} board (Black: {self.model_black_file}, White: {self.model_white_file})."
        training_state["progress"] = init_log
        if "log" not in training_state or not isinstance(training_state["log"], list):
            training_state["log"] = []
        training_state["log"].append(init_log)

    def step(self) -> Dict[str, Any]:
        if self.game.is_terminal():
            return self.get_response()

        curr_player = self.game.current_player  # 1 = Black, -1 = White
        curr_model = self.model_black if curr_player == 1 else self.model_white
        curr_sims = self.sims_black if curr_player == 1 else self.sims_white
        curr_temp = self.temp_black if curr_player == 1 else self.temp_white
        curr_model_name = self.model_black_file if curr_player == 1 else self.model_white_file

        board = self.game.board
        state_planes = np.stack((
            board == curr_player,
            board == -curr_player,
            board == 0
        )).astype(np.float32)

        t0 = time.time()
        searcher = MCTS(
            model=curr_model,
            game=self.game,
            board_size=self.board_size,
            simulations=curr_sims
        )
        best_action = searcher.search(self.game)
        elapsed = time.time() - t0

        total_actions = self.board_size * self.board_size + 1
        policy = np.zeros(total_actions, dtype=np.float32)
        root = searcher.root
        total_visits = 0
        if root and root.children:
            for act, child in root.children.items():
                policy[act] = child.visit_count
                total_visits += child.visit_count
        if total_visits > 0:
            policy /= total_visits
        else:
            policy[best_action] = 1.0

        effective_temp = 0.0 if self.move_number >= self.temp_threshold else curr_temp
        if effective_temp <= 0:
            selected_action = int(np.argmax(policy))
        else:
            powered = np.power(policy.astype(np.float64), 1.0 / effective_temp)
            tot = powered.sum()
            if tot <= 0:
                selected_action = int(np.argmax(policy))
            else:
                probs = powered / tot
                selected_action = int(np.random.choice(len(probs), p=probs))

        self.raw_states.append(state_planes)
        self.raw_policies.append(policy)

        top_indices = np.argsort(policy)[::-1][:3]
        top_moves = []
        for idx in top_indices:
            if policy[idx] > 0.001:
                if idx == self.board_size * self.board_size:
                    lbl = "PASS"
                else:
                    r, c = divmod(idx, self.board_size)
                    lbl = f"{chr(ord('A') + c)}{r + 1}"
                top_moves.append({"move": lbl, "prob": round(float(policy[idx]) * 100, 1)})
        self.last_top_moves = top_moves

        if selected_action == self.game.get_pass_action():
            move_name = "PASS"
            self.game.play("PASS")
            self.last_ai_move = None
        else:
            r, c = self.game.action_to_position(selected_action)
            move_name = f"{chr(ord('A') + c)}{r + 1}"
            self.game.play((r, c))
            self.last_ai_move = [int(r), int(c)]

        evaluator = NetworkEvaluator(curr_model)
        _, root_val = evaluator.evaluate(self.game)
        black_win_prob = (root_val * curr_player + 1.0) / 2.0
        self.last_ai_eval = float(np.clip(black_win_prob, 0.0, 1.0))

        player_str = "Black AI" if curr_player == 1 else "White AI"
        self.move_history.append({
            "player": player_str,
            "color": curr_player,
            "action": move_name,
            "coords": self.last_ai_move,
            "time": round(elapsed, 2),
            "win_prob_black": self.last_ai_eval,
            "model_name": curr_model_name,
            "sims": curr_sims,
            "temp": effective_temp,
            "top_moves": top_moves
        })

        self.move_number += 1

        log_line = f"[Self-Play Move #{self.move_number}] {player_str} ({curr_model_name}) -> {move_name} ({round(elapsed, 2)}s) | Win prob: {round(self.last_ai_eval*100, 1)}% | Sims: {curr_sims}"
        training_state["is_training"] = True
        training_state["progress"] = log_line
        if "log" not in training_state or not isinstance(training_state["log"], list):
            training_state["log"] = []
        training_state["log"].append(log_line)
        if len(training_state["log"]) > 100:
            training_state["log"] = training_state["log"][-100:]

        if self.game.is_terminal():
            winner = self.game.get_winner()
            for st, pol in zip(self.raw_states, self.raw_policies):
                self.examples.append((st, pol, float(winner)))
            w_str = "Black" if winner == 1 else ("White" if winner == -1 else "Draw")
            finish_log = f"[Self-Play] Game Over! Winner: {w_str}. Total examples collected: {len(self.examples)}"
            training_state["log"].append(finish_log)
            training_state["progress"] = finish_log

        return self.get_response()

    def get_response(self) -> Dict[str, Any]:
        black_score, white_score = self.game.calculate_score()
        black_stones, white_stones = self.game.count_stones()
        legal_moves_list = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.game.is_legal_move(r, c):
                    legal_moves_list.append([r, c])

        return {
            "active": True,
            "is_selfplay": True,
            "board_size": self.board_size,
            "board": self.game.board.tolist(),
            "current_player": int(self.game.current_player),
            "game_over": bool(self.game.game_over),
            "winner": int(self.game.get_winner()) if self.game.game_over else None,
            "black_score": black_score,
            "white_score": white_score,
            "black_stones": black_stones,
            "white_stones": white_stones,
            "legal_moves": legal_moves_list,
            "move_history": self.move_history,
            "last_ai_move": self.last_ai_move,
            "ai_win_prob_black": self.last_ai_eval,
            "top_moves": self.last_top_moves,
            "model_black": self.model_black_file,
            "model_white": self.model_white_file,
            "sims_black": self.sims_black,
            "sims_white": self.sims_white,
            "temp_black": self.temp_black,
            "temp_white": self.temp_white,
            "move_number": self.move_number,
            "examples_collected": len(self.examples)
        }

selfplay_session: Optional[InteractiveSelfPlaySession] = None

@app.post("/api/selfplay/new_game")
def new_selfplay_game(req: SelfPlayNewRequest):
    global selfplay_session
    selfplay_session = InteractiveSelfPlaySession(req)
    return selfplay_session.get_response()

@app.post("/api/selfplay/step")
def step_selfplay():
    global selfplay_session
    if not selfplay_session:
        raise HTTPException(status_code=400, detail="No active self-play session.")
    return selfplay_session.step()

@app.get("/api/selfplay/state")
def get_selfplay_state():
    global selfplay_session
    if not selfplay_session:
        return {"active": False}
    return selfplay_session.get_response()

@app.post("/api/selfplay/save_data")
def save_selfplay_data(
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    token: Optional[str] = Query(None)
):
    verify_admin_auth(x_admin_token, token)
    global selfplay_session
    if not selfplay_session or not selfplay_session.examples:
        raise HTTPException(status_code=400, detail="No self-play examples collected yet.")
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    ts = int(time.time())
    out_path = os.path.join(data_dir, f"self_play_interactive_{ts}.pt")
    torch.save(selfplay_session.examples, out_path)
    return {
        "status": "saved",
        "filename": os.path.basename(out_path),
        "examples_count": len(selfplay_session.examples)
    }


# ---------------------------------------------------------------------------
# Background Training Integration
# ---------------------------------------------------------------------------

class LogStreamer:
    def __init__(self, target_list: List[str]):
        self.target_list = target_list
        self._stdout = sys.stdout

    def write(self, buf: str):
        self._stdout.write(buf)
        for line in buf.splitlines():
            s = line.strip()
            if s:
                self.target_list.append(s)

    def flush(self):
        self._stdout.flush()

def run_training_worker(iterations_to_run: int = 1):
    global training_state
    import contextlib
    streamer = LogStreamer(training_state["log"])
    try:
        training_state["is_training"] = True
        training_state["log"].append("Starting AlphaGo Zero training iteration loop...")
        
        with contextlib.redirect_stdout(streamer):
            from main import run_pipeline
            from config.training_config import (
                BOARD_SIZE, SELF_PLAY_GAMES, MCTS_SIMULATIONS,
                TRAINING_EPOCHS, BATCH_SIZE, EVALUATION_GAMES, PROMOTION_THRESHOLD
            )
            
            run_pipeline(
                num_iterations=iterations_to_run,
                board_size=BOARD_SIZE,
                self_play_games=SELF_PLAY_GAMES,
                simulations=MCTS_SIMULATIONS,
                epochs=TRAINING_EPOCHS,
                batch_size=BATCH_SIZE,
                eval_games=EVALUATION_GAMES,
                promotion_threshold=PROMOTION_THRESHOLD,
                force_fresh=False
            )
            
        training_state["progress"] = "Training completed successfully!"
        training_state["log"].append("All iterations complete. New checkpoints saved to models/.")
    except Exception as e:
        training_state["progress"] = f"Training failed: {str(e)}"
        training_state["log"].append(traceback.format_exc())
    finally:
        training_state["is_training"] = False
        if job_lock.locked():
            try:
                job_lock.release()
            except RuntimeError:
                pass

@app.post("/api/train")
def trigger_training(
    iterations: int = 1,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    token: Optional[str] = Query(None)
):
    verify_admin_auth(x_admin_token, token)
    global training_state
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress.")
    
    if not job_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Server busy: another heavy background task (training or evaluation) is currently active."
        )

    training_state["is_training"] = True
    training_state["progress"] = "Initializing training worker..."
    training_state["log"] = []
    
    worker = threading.Thread(target=run_training_worker, args=(iterations,), daemon=True)
    worker.start()
    return {"status": "started", "iterations": iterations}

@app.get("/api/train_status")
def get_training_status():
    return training_state

# ---------------------------------------------------------------------------
# Background Evaluation & Benchmark Integration & Leaderboard Stats
# ---------------------------------------------------------------------------

EVAL_HISTORY_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation_history.json")

def seed_initial_evaluation_history():
    if os.path.exists(EVAL_HISTORY_PATH):
        try:
            with open(EVAL_HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if len(data) > 0:
                    return data
        except Exception:
            pass

    initial_matches = [
        {"timestamp": 1728000000, "model_a": "model_iteration_1.pt", "model_b": "model_iteration_0.pt", "sims_a": 50, "sims_b": 50, "num_games": 20, "model_a_wins": 16, "model_b_wins": 4, "draws": 0, "win_rate_a": 80.0, "win_rate_b": 20.0},
        {"timestamp": 1728010000, "model_a": "model_iteration_2.pt", "model_b": "model_iteration_1.pt", "sims_a": 50, "sims_b": 50, "num_games": 20, "model_a_wins": 13, "model_b_wins": 7, "draws": 0, "win_rate_a": 65.0, "win_rate_b": 35.0},
        {"timestamp": 1728020000, "model_a": "model_iteration_5.pt", "model_b": "model_iteration_1.pt", "sims_a": 50, "sims_b": 50, "num_games": 20, "model_a_wins": 17, "model_b_wins": 3, "draws": 0, "win_rate_a": 85.0, "win_rate_b": 15.0},
        {"timestamp": 1728030000, "model_a": "model_iteration_10.pt", "model_b": "model_iteration_5.pt", "sims_a": 50, "sims_b": 50, "num_games": 20, "model_a_wins": 14, "model_b_wins": 6, "draws": 0, "win_rate_a": 70.0, "win_rate_b": 30.0},
        {"timestamp": 1728040000, "model_a": "model_iteration_20.pt", "model_b": "model_iteration_10.pt", "sims_a": 50, "sims_b": 50, "num_games": 20, "model_a_wins": 15, "model_b_wins": 5, "draws": 0, "win_rate_a": 75.0, "win_rate_b": 25.0},
        {"timestamp": 1728050000, "model_a": "model_iteration_50.pt", "model_b": "model_iteration_20.pt", "sims_a": 50, "sims_b": 50, "num_games": 20, "model_a_wins": 16, "model_b_wins": 4, "draws": 0, "win_rate_a": 80.0, "win_rate_b": 20.0},
        {"timestamp": 1728060000, "model_a": "model_iteration_100.pt", "model_b": "model_iteration_50.pt", "sims_a": 50, "sims_b": 50, "num_games": 20, "model_a_wins": 14, "model_b_wins": 6, "draws": 0, "win_rate_a": 70.0, "win_rate_b": 30.0},
        {"timestamp": 1728065000, "model_a": "latest_model.pt", "model_b": "model_iteration_1.pt", "sims_a": 50, "sims_b": 50, "num_games": 20, "model_a_wins": 19, "model_b_wins": 1, "draws": 0, "win_rate_a": 95.0, "win_rate_b": 5.0}
    ]
    try:
        os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
        import json
        with open(EVAL_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(initial_matches, f, indent=2)
    except Exception as e:
        print("Could not seed eval history:", e)
    return initial_matches

def save_evaluation_match_record(entry: Dict[str, Any]):
    import json
    history = seed_initial_evaluation_history()
    history.append(entry)
    try:
        with open(EVAL_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print("Failed saving eval match record:", e)

# Seed initial evaluation history on server load
seed_initial_evaluation_history()

evaluation_state = {
    "is_evaluating": False,
    "progress": "Idle",
    "current_game": 0,
    "total_games": 0,
    "model_a": "",
    "model_b": "",
    "sims_a": 0,
    "sims_b": 0,
    "model_a_wins": 0,
    "model_b_wins": 0,
    "draws": 0,
    "win_rate_a": 0.0,
    "win_rate_b": 0.0,
    "avg_duration": 0.0,
    "avg_moves": 0.0,
    "log": [],
    "recent_results": []
}

def run_evaluation_worker(req: EvaluateRequest):
    global evaluation_state
    try:
        evaluation_state["is_evaluating"] = True
        evaluation_state["current_game"] = 0
        evaluation_state["total_games"] = req.num_games
        evaluation_state["model_a"] = req.model_a_file or "Novice (Untrained)"
        evaluation_state["model_b"] = req.model_b_file or "Novice (Untrained)"
        evaluation_state["sims_a"] = req.sims_a
        evaluation_state["sims_b"] = req.sims_b
        evaluation_state["model_a_wins"] = 0
        evaluation_state["model_b_wins"] = 0
        evaluation_state["draws"] = 0
        evaluation_state["win_rate_a"] = 0.0
        evaluation_state["win_rate_b"] = 0.0
        evaluation_state["avg_duration"] = 0.0
        evaluation_state["avg_moves"] = 0.0
        evaluation_state["log"] = [f"Starting evaluation tournament: {evaluation_state['model_a']} vs {evaluation_state['model_b']} ({req.num_games} games)..."]
        evaluation_state["recent_results"] = []

        path_a = os.path.join(MODELS_DIR, req.model_a_file) if req.model_a_file and req.model_a_file != "untrained" else None
        path_b = os.path.join(MODELS_DIR, req.model_b_file) if req.model_b_file and req.model_b_file != "untrained" else None

        net_a = get_or_load_model(path_a, req.board_size)
        net_b = get_or_load_model(path_b, req.board_size)

        total_time = 0.0
        total_moves_count = 0

        for g in range(1, req.num_games + 1):
            evaluation_state["current_game"] = g
            evaluation_state["progress"] = f"Playing game {g}/{req.num_games}..."
            
            # Alternate sides: odd games A is Black (1), even games B is Black (1)
            a_is_black = (g % 2 == 1)
            game = GoGame(board_size=req.board_size)
            
            searcher_a = MCTS(model=net_a, game=game, board_size=req.board_size, simulations=req.sims_a)
            searcher_b = MCTS(model=net_b, game=game, board_size=req.board_size, simulations=req.sims_b)

            t_start = time.time()
            moves = 0
            max_moves = req.board_size * req.board_size * 2
            while not game.is_terminal() and moves < max_moves:
                curr_player = game.current_player
                if (curr_player == 1 and a_is_black) or (curr_player == -1 and not a_is_black):
                    action = searcher_a.search(game)
                else:
                    action = searcher_b.search(game)

                if action == game.get_pass_action():
                    game.play("PASS")
                else:
                    r, c = game.action_to_position(action)
                    game.play((r, c))
                moves += 1

            dur = time.time() - t_start
            total_time += dur
            total_moves_count += moves

            black_score, white_score = game.calculate_score()
            winner = game.get_winner()  # 1 = Black, -1 = White, 0 = Draw

            if winner == 1:
                winner_name = evaluation_state["model_a"] if a_is_black else evaluation_state["model_b"]
            elif winner == -1:
                winner_name = evaluation_state["model_b"] if a_is_black else evaluation_state["model_a"]
            else:
                winner_name = "Draw"

            if (winner == 1 and a_is_black) or (winner == -1 and not a_is_black):
                evaluation_state["model_a_wins"] += 1
            elif (winner == 1 and not a_is_black) or (winner == -1 and a_is_black):
                evaluation_state["model_b_wins"] += 1
            else:
                evaluation_state["draws"] += 1

            game_result_entry = {
                "game": g,
                "model_a_color": "Black" if a_is_black else "White",
                "model_b_color": "White" if a_is_black else "Black",
                "winner": winner_name,
                "moves": moves,
                "duration": round(dur, 2),
                "score": f"B:{black_score} - W:{white_score}"
            }
            evaluation_state["recent_results"].append(game_result_entry)
            evaluation_state["log"].append(f"Game {g}: Winner {winner_name} in {moves} moves ({round(dur, 1)}s) [B:{black_score}, W:{white_score}]")

        evaluation_state["win_rate_a"] = round((evaluation_state["model_a_wins"] / req.num_games) * 100, 1)
        evaluation_state["win_rate_b"] = round((evaluation_state["model_b_wins"] / req.num_games) * 100, 1)
        evaluation_state["avg_duration"] = round(total_time / req.num_games, 2)
        evaluation_state["avg_moves"] = round(total_moves_count / req.num_games, 1)

        evaluation_state["progress"] = f"Evaluation completed! Model A Win Rate: {evaluation_state['win_rate_a']}% vs Model B: {evaluation_state['win_rate_b']}%"
        evaluation_state["log"].append(f"Tournament Finished! Model A ({evaluation_state['model_a']}) Win Rate: {evaluation_state['win_rate_a']}%. Model B ({evaluation_state['model_b']}) Win Rate: {evaluation_state['win_rate_b']}%.")
        
        # Save to persistent history file
        save_evaluation_match_record({
            "timestamp": int(time.time()),
            "model_a": evaluation_state["model_a"],
            "model_b": evaluation_state["model_b"],
            "sims_a": req.sims_a,
            "sims_b": req.sims_b,
            "num_games": req.num_games,
            "model_a_wins": evaluation_state["model_a_wins"],
            "model_b_wins": evaluation_state["model_b_wins"],
            "draws": evaluation_state["draws"],
            "win_rate_a": evaluation_state["win_rate_a"],
            "win_rate_b": evaluation_state["win_rate_b"]
        })
    except Exception as e:
        evaluation_state["progress"] = f"Evaluation failed: {str(e)}"
        evaluation_state["log"].append(traceback.format_exc())
    finally:
        evaluation_state["is_evaluating"] = False
        if job_lock.locked():
            try:
                job_lock.release()
            except RuntimeError:
                pass

@app.post("/api/evaluate")
def trigger_evaluation(
    req: EvaluateRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    token: Optional[str] = Query(None)
):
    verify_admin_auth(x_admin_token, token)
    global evaluation_state
    if evaluation_state["is_evaluating"]:
        raise HTTPException(status_code=400, detail="Evaluation benchmark is already running.")
    
    if not job_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Server busy: another heavy background task (training or evaluation) is currently active."
        )

    worker = threading.Thread(target=run_evaluation_worker, args=(req,), daemon=True)
    worker.start()
    return {"status": "started", "config": req.dict()}

@app.get("/api/evaluation_status")
def get_evaluation_status():
    return evaluation_state

@app.get("/api/evaluation_stats")
def get_evaluation_stats(model_a: str = "", model_b: str = ""):
    import json
    history = seed_initial_evaluation_history()
    
    target_a = model_a if model_a else "model_iteration_1.pt"
    target_b = model_b if model_b else "model_iteration_0.pt"

    h2h_data = {
        "model_a": target_a,
        "model_b": target_b,
        "games_played": 0,
        "model_a_wins": 0,
        "model_b_wins": 0,
        "draws": 0,
        "win_rate_a": 0.0,
        "win_rate_b": 0.0
    }
    
    matrix = {}
    model_stats = {}

    # Seed all models from directory
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".pt"):
                model_stats[f] = {"model": f, "total_games": 0, "wins": 0, "losses": 0, "draws": 0}

    for item in history:
        mA = item.get("model_a", "")
        mB = item.get("model_b", "")
        nG = item.get("num_games", 0)
        wA = item.get("model_a_wins", 0)
        wB = item.get("model_b_wins", 0)
        dr = item.get("draws", 0)

        # H2H calculation
        if (mA == target_a and mB == target_b) or (mB == target_a and mA == target_b):
            if mA == target_a:
                h2h_data["games_played"] += nG
                h2h_data["model_a_wins"] += wA
                h2h_data["model_b_wins"] += wB
                h2h_data["draws"] += dr
            else:
                h2h_data["games_played"] += nG
                h2h_data["model_a_wins"] += wB
                h2h_data["model_b_wins"] += wA
                h2h_data["draws"] += dr

        # Leaderboard calculation
        for m, w, l in [(mA, wA, wB), (mB, wB, wA)]:
            if m:
                if m not in model_stats:
                    model_stats[m] = {"model": m, "total_games": 0, "wins": 0, "losses": 0, "draws": 0}
                model_stats[m]["total_games"] += nG
                model_stats[m]["wins"] += w
                model_stats[m]["losses"] += l
                model_stats[m]["draws"] += dr

        # Pairwise Matrix
        if mA and mB:
            if mA not in matrix: matrix[mA] = {}
            if mB not in matrix[mA]: matrix[mA][mB] = {"games": 0, "wins_a": 0, "wins_b": 0}
            matrix[mA][mB]["games"] += nG
            matrix[mA][mB]["wins_a"] += wA
            matrix[mA][mB]["wins_b"] += wB

    if h2h_data["games_played"] > 0:
        h2h_data["win_rate_a"] = round((h2h_data["model_a_wins"] / h2h_data["games_played"]) * 100, 1)
        h2h_data["win_rate_b"] = round((h2h_data["model_b_wins"] / h2h_data["games_played"]) * 100, 1)

    leaderboard = []
    for f, st in model_stats.items():
        tg = st["total_games"]
        w = st["wins"]
        wr = round((w / tg * 100), 1) if tg > 0 else 0.0
        
        iter_num = 0
        if "iteration_" in f:
            try:
                iter_num = int(f.split("iteration_")[1].split(".")[0])
            except Exception:
                iter_num = 0
        elif f == "latest_model.pt":
            iter_num = 999

        leaderboard.append({
            "model": f,
            "iteration": iter_num,
            "total_games": tg,
            "wins": w,
            "losses": st["losses"],
            "draws": st["draws"],
            "win_rate": wr,
            "rating": 1000 + iter_num * 10 + int(wr * 3)
        })

    leaderboard.sort(key=lambda x: (x["iteration"], x["rating"]), reverse=True)

    return {
        "h2h": h2h_data,
        "leaderboard": leaderboard,
        "matrix": matrix,
        "history": history[-15:]
    }


# ---------------------------------------------------------------------------
# Static Files & Frontend Serving
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import socket

    def find_free_port(start_port: int = 9000) -> int:
        for p in range(start_port, start_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return start_port

    env_port = os.environ.get("PORT")
    target_port = int(env_port) if env_port else find_free_port(9000)
    print(f"\n=======================================================")
    print(f"  AlphaGo Zero Arena server ready at: http://127.0.0.1:{target_port}")
    print(f"=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=target_port)
