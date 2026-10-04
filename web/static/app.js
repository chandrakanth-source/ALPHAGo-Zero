const API_BASE = location.hostname.endsWith("vercel.app")
  ? "https://alphago-zero.onrender.com"
  : "";

const MOCK = new URLSearchParams(location.search).get("mock") === "1";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {
  boardSize: 12,
  humanColor: 1,
  simulations: 50,
  modelFile: "",
  iterations: [],
  board: [],
  legalMoves: [],
  currentPlayer: 1,
  lastAiMove: null,
  moveHistory: [],
  gameOver: false,
  thinking: false,
  valueHistory: [50],
  moveNumber: 0,
  mock: MOCK,
  mockSelfplay: null,
};
let boardCells = new Map();
let valueChart = null;
let thinkTimer = null;
let wakeAttempt = 0;
let selfplayPoll = null;
let evaluationPoll = null;

function mockIterations() {
  return [0, 1, 2, 3, 5, 10, 20].map((iteration) => ({
    iteration,
    filename: `model_iteration_${iteration}.pt`,
    size_kb: 3020,
  }));
}
function mockGame() {
  const board = Array.from({ length: state.boardSize }, () =>
    Array(state.boardSize).fill(0),
  );
  return {
    active: true,
    board_size: state.boardSize,
    board,
    current_player: 1,
    human_color: state.humanColor,
    consecutive_passes: 0,
    game_over: false,
    winner: null,
    black_score: 0,
    white_score: 0,
    black_stones: 0,
    white_stones: 0,
    legal_moves: [],
    move_history: [],
    last_ai_move: null,
    ai_win_prob_black: 0.5,
    active_level_name: state.modelFile || "Untrained",
    active_model_file: state.modelFile || "untrained",
    active_simulations: state.simulations,
    latest_ai_info: null,
  };
}
async function mockRequest(endpoint, method, data) {
  if (endpoint.startsWith("/api/iterations"))
    return { iterations: mockIterations(), count: 7 };
  if (endpoint.startsWith("/api/levels"))
    return {
      current_board_size: state.boardSize,
      presets: {},
      available_checkpoints: mockIterations().map((item) => ({
        filename: item.filename,
        detected_board_size: state.boardSize,
        compatible_with_current: true,
      })),
    };
  if (endpoint === "/api/new_game") {
    Object.assign(state, {
      boardSize: data.board_size,
      humanColor: data.human_color,
      simulations: data.simulations,
      modelFile: data.model_file || "untrained",
      moveHistory: [],
      moveNumber: 0,
      lastAiMove: null,
    });
    return mockGame();
  }
  if (endpoint === "/api/state") return mockGame();
  if (
    endpoint === "/api/move" ||
    endpoint === "/api/ai_move" ||
    endpoint === "/api/undo"
  )
    return mockGame();
  if (endpoint === "/api/hint")
    return {
      coords: [
        Math.floor(state.boardSize / 2),
        Math.floor(state.boardSize / 2),
      ],
      action: "F6",
      win_prob: 0.54,
      explanation: "A central move keeps influence balanced across the board.",
    };
  if (endpoint === "/api/selfplay_status")
    return {
      current_iteration: 20,
      games_played: 138,
      total_examples: 18420,
      models_trained: 20,
      is_active: false,
      progress: "Pipeline ready / idle",
      log: ["Mock telemetry enabled", "No backend writes performed."],
      model_files: mockIterations().map((item) => item),
    };
  if (endpoint === "/api/selfplay/new_game") {
    state.mockSelfplay = mockGame();
    state.mockSelfplay.is_selfplay = true;
    return state.mockSelfplay;
  }
  if (endpoint === "/api/selfplay/step")
    return state.mockSelfplay || mockGame();
  if (endpoint === "/api/selfplay/save_data")
    return { status: "saved", filename: "mock-preview.pt", examples_count: 64 };
  if (endpoint === "/api/evaluation_stats") return mockEvaluation();
  if (endpoint === "/api/evaluate") return { status: "started" };
  if (endpoint === "/api/evaluation_status")
    return {
      is_evaluating: false,
      model_a: "model_iteration_1.pt",
      model_b: "model_iteration_2.pt",
      sims_a: 25,
      sims_b: 50,
      win_rate_a: 46,
      win_rate_b: 54,
      draws: 0,
      avg_duration: 1.2,
      log: ["Mock evaluation complete"],
    };
  if (endpoint === "/api/train") return { status: "started" };
  if (endpoint === "/api/train_status")
    return { is_training: false, progress: "Mock preview", log: [] };
  return {};
}
function mockEvaluation() {
  const leaderboard = mockIterations().map((item, index) => ({
    model: item.filename,
    iteration: item.iteration,
    total_games: 20,
    wins: 8 + index,
    losses: 12 - index,
    draws: 0,
    win_rate: 40 + index * 4,
    rating: 980 + index * 24,
  }));
  return {
    h2h: {
      model_a: "model_iteration_1.pt",
      model_b: "model_iteration_2.pt",
      games_played: 20,
      model_a_wins: 9,
      model_b_wins: 11,
      draws: 0,
      win_rate_a: 45,
      win_rate_b: 55,
    },
    leaderboard,
    matrix: {},
    history: [],
  };
}
async function apiRequest(endpoint, method = "GET", data = null, signal) {
  if (MOCK) return mockRequest(endpoint, method, data);
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
    signal,
  };
  if (data) options.body = JSON.stringify(data);
  const response = await fetch(`${API_BASE}${endpoint}`, options);
  if (!response.ok) {
    const errorData = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    const error = new Error(String(errorData.detail || "API request failed"));
    error.isNoSession =
      error.message.includes("No active game session") ||
      error.message.includes("Call /api/new_game");
    throw error;
  }
  return response.json();
}
function toast(message, kind = "error") {
  const item = document.createElement("div");
  item.className = `toast ${kind === "ok" ? "ok" : ""}`;
  item.textContent = message;
  $("#toasts").append(item);
  setTimeout(() => item.remove(), 5000);
}
function setEngineStatus(label, online = true) {
  $("#engine-status label").textContent = label;
  $("#engine-status").classList.toggle("offline", !online);
}
function setWakeProgress(percent, message) {
  $("#wake-progress").style.width = `${percent}%`;
  $("#wake-message").textContent = message;
}
async function wakeEngine() {
  if (MOCK) {
    setWakeProgress(
      100,
      "Mock preview is local and never writes to the backend.",
    );
    await new Promise((resolve) => setTimeout(resolve, 450));
    await boot();
    return;
  }
  wakeAttempt += 1;
  setWakeProgress(
    Math.min(92, 18 + wakeAttempt * 13),
    wakeAttempt === 1
      ? "Connecting to the Render inference service."
      : `Render is waking up. Retrying in a moment (${wakeAttempt}).`,
  );
  setEngineStatus("Waking engine", false);
  try {
    await apiRequest("/api/iterations");
    setWakeProgress(100, "Engine online. Loading reported checkpoints.");
    await boot();
  } catch (error) {
    const delay = Math.min(15000, 900 * 2 ** Math.min(wakeAttempt - 1, 4));
    $("#wake-retry").textContent = `Retry in ${Math.ceil(delay / 1000)}s`;
    setTimeout(wakeEngine, delay);
  }
}
async function boot() {
  try {
    await Promise.all([loadIterations(), loadLevels(), loadEvaluation()]);
    await startGame();
    setEngineStatus("Engine online", true);
    $("#mode-note").textContent = MOCK
      ? "Mock preview"
      : "MCTS inference ready";
    $("#engine-overlay").classList.add("ready");
  } catch (error) {
    setEngineStatus("Engine error", false);
    toast(error.message);
    $("#wake-retry").textContent = "Retry now";
  }
}
async function loadIterations() {
  const data = await apiRequest("/api/iterations");
  state.iterations = data.iterations || [];
  const options = state.iterations
    .map(
      (item) =>
        `<option value="${item.filename}">Iteration ${item.iteration} / ${item.size_kb} KB</option>`,
    )
    .join("");
  [
    $("#model-select"),
    $("#sp-model-black"),
    $("#sp-model-white"),
    $("#eval-model-a"),
    $("#eval-model-b"),
  ].forEach((select) => {
    if (select)
      select.innerHTML =
        options || `<option value="">No checkpoints reported</option>`;
  });
  if (state.iterations.length) {
    state.modelFile = state.iterations[0].filename;
    $("#model-select").value = state.modelFile;
    $("#eval-model-a").value = state.iterations[0].filename;
    $("#eval-model-b").value = state.iterations.at(-1).filename;
  }
  renderCheckpoints();
}
async function loadLevels() {
  await apiRequest(`/api/levels?board_size=${state.boardSize}`);
}
async function loadEvaluation() {
  try {
    renderEvaluation(
      await apiRequest(
        `/api/evaluation_stats?model_a=${encodeURIComponent($("#eval-model-a")?.value || "")}&model_b=${encodeURIComponent($("#eval-model-b")?.value || "")}`,
      ),
    );
  } catch (error) {
    console.warn(error);
  }
}
function renderCheckpoints() {
  $("#checkpoint-grid").innerHTML = state.iterations.length
    ? state.iterations
        .map(
          (item) =>
            `<div class="checkpoint"><strong>ITER ${item.iteration}</strong><span>${item.filename} / ${item.size_kb} KB</span></div>`,
        )
        .join("")
    : '<p class="muted">No checkpoints reported by the backend.</p>';
}
async function startGame() {
  await requestGame({
    board_size: state.boardSize,
    human_color: state.humanColor,
    level: "custom",
    simulations: state.simulations,
    model_file: state.modelFile || null,
  });
}
async function requestGame(payload) {
  setThinking(true);
  try {
    const data = await apiRequest("/api/new_game", "POST", payload);
    updateGame(data);
  } catch (error) {
    toast(`Could not start game: ${error.message}`);
  } finally {
    setThinking(false);
  }
}
function updateGame(data) {
  if (!data || data.active === false) return;
  state.boardSize = data.board_size || state.boardSize;
  state.board = data.board || [];
  state.legalMoves = data.legal_moves || [];
  state.currentPlayer = data.current_player || 1;
  state.lastAiMove = data.last_ai_move;
  state.moveHistory = data.move_history || [];
  state.gameOver = Boolean(data.game_over);
  state.moveNumber = state.moveHistory.length;
  state.modelFile = data.active_model_file || state.modelFile;
  state.valueHistory.push(Number(data.ai_win_prob_black || 0.5) * 100);
  if (state.valueHistory.length > 24) state.valueHistory.shift();
  $("#board-size-label").textContent =
    `${state.boardSize} x ${state.boardSize}`;
  $("#game-clock").textContent =
    `GAME 001 / MOVE ${String(state.moveNumber).padStart(2, "0")}`;
  $("#active-model").textContent = state.modelFile || "Untrained";
  $("#active-config").textContent =
    `${data.active_simulations || state.simulations} simulations / move`;
  $("#telemetry-model").textContent = state.modelFile || "--";
  $("#telemetry-sims").textContent =
    `${data.active_simulations || state.simulations} / move`;
  $("#telemetry-time").textContent = data.latest_ai_info?.time
    ? `${data.latest_ai_info.time}s`
    : "--";
  $("#telemetry-passes").textContent = `${data.consecutive_passes || 0} / 2`;
  const blackProb = Number(data.ai_win_prob_black ?? 0.5) * 100;
  $("#value-fill").style.width = `${blackProb}%`;
  $("#value-readout").textContent = `${blackProb.toFixed(1)}%`;
  $("#black-prob").textContent = `${blackProb.toFixed(1)}%`;
  $("#white-prob").textContent = `${(100 - blackProb).toFixed(1)}%`;
  $("#black-score").textContent =
    `${data.black_score || 0} points / ${data.black_stones || 0} stones`;
  $("#white-score").textContent =
    `${data.white_score || 0} points / ${data.white_stones || 0} stones`;
  $("#black-name").textContent =
    state.humanColor === 1 ? "You" : "AlphaGo Zero";
  $("#white-name").textContent =
    state.humanColor === -1 ? "You" : "AlphaGo Zero";
  $("#turn-chip").innerHTML =
    `<i class="stone-dot ${state.currentPlayer === 1 ? "black" : "white"}"></i> ${state.currentPlayer === 1 ? "Black" : "White"} to move`;
  renderBoard();
  renderHistory();
  updateChart();
  if (state.gameOver) showGameOver(data);
}
function boardLetter(col) {
  const letters = "ABCDEFGHJKLMNOPQRST";
  return letters[col] || "?";
}
function starPoints(size) {
  return size === 12
    ? [
        [3, 3],
        [3, 8],
        [8, 3],
        [8, 8],
      ]
    : size === 5
      ? [[2, 2]]
      : [];
}
function buildBoard() {
  const svg = $("#goban");
  svg.innerHTML = "";
  boardCells = new Map();
  const size = state.boardSize;
  const margin = 55;
  const step = (720 - margin * 2) / (size - 1);
  const ns = "http://www.w3.org/2000/svg";
  const defs = document.createElementNS(ns, "defs");
  defs.innerHTML =
    '<radialGradient id="stone-black" cx="32%" cy="25%"><stop stop-color="#67717d"/><stop offset=".35" stop-color="#242c35"/><stop offset="1" stop-color="#07090c"/></radialGradient><radialGradient id="stone-white" cx="30%" cy="23%"><stop stop-color="#fff"/><stop offset=".55" stop-color="#e0ddd2"/><stop offset="1" stop-color="#a6a196"/></radialGradient><filter id="stone-shadow"><feDropShadow dx="2" dy="4" stdDeviation="4" flood-color="#2d1608" flood-opacity=".55"/></filter>';
  svg.append(defs);
  const grid = document.createElementNS(ns, "g");
  grid.setAttribute("stroke", "#4f2d16");
  grid.setAttribute("stroke-width", "1.5");
  grid.setAttribute("opacity", ".88");
  for (let index = 0; index < size; index += 1) {
    const position = margin + index * step;
    const horizontal = document.createElementNS(ns, "line");
    horizontal.setAttribute("x1", margin);
    horizontal.setAttribute("x2", 720 - margin);
    horizontal.setAttribute("y1", position);
    horizontal.setAttribute("y2", position);
    grid.append(horizontal);
    const vertical = document.createElementNS(ns, "line");
    vertical.setAttribute("x1", position);
    vertical.setAttribute("x2", position);
    vertical.setAttribute("y1", margin);
    vertical.setAttribute("y2", 720 - margin);
    grid.append(vertical);
  }
  svg.append(grid);
  const points = document.createElementNS(ns, "g");
  points.setAttribute("fill", "#3d210e");
  starPoints(size).forEach(([row, col]) => {
    const point = document.createElementNS(ns, "circle");
    point.setAttribute("cx", margin + col * step);
    point.setAttribute("cy", margin + row * step);
    point.setAttribute("r", size === 5 ? 5 : 4);
    points.append(point);
  });
  svg.append(points);
  const labels = document.createElementNS(ns, "g");
  labels.setAttribute("fill", "#5b351a");
  labels.setAttribute("font-size", "12");
  labels.setAttribute("font-family", "JetBrains Mono, monospace");
  for (let index = 0; index < size; index += 1) {
    const x = margin + index * step;
    const colLabel = boardLetter(index);
    [margin - 29, 720 - margin + 28].forEach((y) => {
      const text = document.createElementNS(ns, "text");
      text.setAttribute("x", x);
      text.setAttribute("y", y);
      text.setAttribute("text-anchor", "middle");
      text.textContent = colLabel;
      labels.append(text);
    });
    const y = margin + index * step;
    [margin - 20, 720 - margin + 20].forEach((xPos) => {
      const text = document.createElementNS(ns, "text");
      text.setAttribute("x", xPos);
      text.setAttribute("y", y + 4);
      text.setAttribute("text-anchor", "middle");
      text.textContent = String(index + 1);
      labels.append(text);
    });
  }
  svg.append(labels);
  for (let row = 0; row < size; row += 1)
    for (let col = 0; col < size; col += 1) {
      const group = document.createElementNS(ns, "g");
      group.dataset.row = row;
      group.dataset.col = col;
      group.setAttribute(
        "transform",
        `translate(${margin + col * step},${margin + row * step})`,
      );
      group.classList.add("intersection");
      group.addEventListener("click", () => playMove(row, col));
      const ghost = document.createElementNS(ns, "circle");
      ghost.setAttribute("r", step * 0.39);
      ghost.classList.add("ghost");
      group.append(ghost);
      const stone = document.createElementNS(ns, "circle");
      stone.setAttribute("r", step * 0.39);
      stone.setAttribute("filter", "url(#stone-shadow)");
      group.append(stone);
      const last = document.createElementNS(ns, "circle");
      last.setAttribute("r", step * 0.12);
      last.classList.add("last-marker");
      group.append(last);
      boardCells.set(`${row},${col}`, { group, stone, last });
      svg.append(group);
    }
}
function renderBoard() {
  if (!boardCells.size || boardCells.size !== state.boardSize ** 2)
    buildBoard();
  const legal = new Set(state.legalMoves.map(([row, col]) => `${row},${col}`));
  boardCells.forEach(({ group, stone, last }, key) => {
    const [row, col] = key.split(",").map(Number);
    const value = state.board[row]?.[col] || 0;
    group.classList.toggle("legal", legal.has(key));
    group.classList.toggle("occupied", Boolean(value));
    stone.setAttribute(
      "fill",
      value === 1 ? "url(#stone-black)" : "url(#stone-white)",
    );
    stone.style.display = value ? "block" : "none";
    last.style.display =
      state.lastAiMove?.[0] === row && state.lastAiMove?.[1] === col
        ? "block"
        : "none";
  });
}
function renderHistory() {
  const list = $("#move-list");
  $("#move-count").textContent = state.moveHistory.length;
  if (!state.moveHistory.length) {
    list.innerHTML = '<p class="muted">First move awaits the human player.</p>';
    return;
  }
  list.innerHTML = state.moveHistory
    .slice(-14)
    .map(
      (move, index) =>
        `<div class="move-row ${move.player === "AI" ? "ai" : ""}"><span class="num">${String(Math.max(1, state.moveHistory.length - 13 + index)).padStart(2, "0")}</span><span>[${move.color === 1 ? "B" : "W"}] ${move.action || "PASS"}</span><span class="meta">${move.player || ""} ${move.time ? `${move.time}s` : ""}</span></div>`,
    )
    .join("");
  list.scrollTop = list.scrollHeight;
}
function updateChart() {
  if (!window.Chart) return;
  const canvas = $("#value-chart");
  if (!valueChart)
    valueChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: state.valueHistory.map((_, index) => index + 1),
        datasets: [
          {
            data: state.valueHistory,
            borderColor: "#f5b84b",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35,
            fill: true,
            backgroundColor: "rgba(245,184,75,.08)",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: { display: false, min: 0, max: 100 },
        },
      },
    });
  else {
    valueChart.data.labels = state.valueHistory.map((_, index) => index + 1);
    valueChart.data.datasets[0].data = state.valueHistory;
    valueChart.update("none");
  }
}
function setThinking(active) {
  state.thinking = active;
  const overlay = $("#board-thinking");
  overlay.classList.toggle("hidden", !active);
  $$("[data-action]").forEach((button) => {
    button.disabled = active;
  });
  if (active) {
    const started = performance.now();
    clearInterval(thinkTimer);
    thinkTimer = setInterval(() => {
      $("#think-time").textContent =
        `${((performance.now() - started) / 1000).toFixed(1)}s`;
    }, 100);
  } else clearInterval(thinkTimer);
}
async function playMove(row, col) {
  if (
    state.thinking ||
    state.gameOver ||
    !state.legalMoves.some(
      ([legalRow, legalCol]) => legalRow === row && legalCol === col,
    )
  )
    return;
  await performAction(
    "/api/move",
    "POST",
    { row, col, is_pass: false },
    "Illegal move",
  );
}
async function performAction(endpoint, method, data, fallback) {
  setThinking(true);
  try {
    updateGame(await apiRequest(endpoint, method, data));
  } catch (error) {
    if (error.isNoSession) await startGame();
    else toast(`${fallback}: ${error.message}`);
  } finally {
    setThinking(false);
  }
}
async function passMove() {
  if (!state.thinking && !state.gameOver)
    await performAction("/api/move", "POST", { is_pass: true }, "Pass failed");
}
async function undoMove() {
  if (!state.thinking)
    await performAction("/api/undo", "POST", null, "Undo failed");
}
async function aiMove() {
  if (!state.thinking && !state.gameOver)
    await performAction("/api/ai_move", "POST", null, "AI move failed");
}
async function hint() {
  if (state.thinking || state.gameOver) return;
  try {
    const data = await apiRequest("/api/hint", "POST");
    state.lastAiMove = data.coords;
    $("#hint-copy").textContent =
      data.explanation || "The engine found a promising continuation.";
    renderBoard();
    toast(data.explanation || "Hint ready", "ok");
  } catch (error) {
    toast(`Hint unavailable: ${error.message}`);
  }
}
function showGameOver(data) {
  const dialog = $("#game-dialog");
  $("#dialog-title").textContent =
    data.winner === state.humanColor
      ? "A measured victory"
      : data.winner === 0
        ? "A balanced game"
        : "The policy takes it";
  $("#dialog-copy").textContent = data.resignation
    ? "The match ended by resignation."
    : "The final territory count is in.";
  $("#dialog-black").textContent = data.black_score ?? 0;
  $("#dialog-white").textContent = data.white_score ?? 0;
  if (!dialog.open) dialog.showModal();
}
function renderEvaluation(data) {
  if (!data) return;
  const h2h = data.h2h || {};
  $("#h2h-title").textContent = `${h2h.games_played || 0} games played`;
  $("#rate-a").textContent =
    h2h.win_rate_a == null ? "--" : `${h2h.win_rate_a}%`;
  $("#rate-b").textContent =
    h2h.win_rate_b == null ? "--" : `${h2h.win_rate_b}%`;
  $("#name-a").textContent = h2h.model_a || "Model A";
  $("#name-b").textContent = h2h.model_b || "Model B";
  const total = Math.max(1, (h2h.win_rate_a || 0) + (h2h.win_rate_b || 0));
  $("#match-a").style.width = `${((h2h.win_rate_a || 0) / total) * 100}%`;
  $("#match-b").style.width = `${((h2h.win_rate_b || 0) / total) * 100}%`;
  $("#draws").textContent = `${h2h.draws || 0} draws`;
  $("#promotion-badge").textContent =
    (h2h.win_rate_a || 0) >= 55
      ? "PROMOTED A"
      : (h2h.win_rate_b || 0) >= 55
        ? "PROMOTED B"
        : "Below threshold";
  const rows = data.leaderboard || [];
  $("#leaderboard").innerHTML = rows.length
    ? rows
        .map(
          (item, index) =>
            `<tr><td>${index + 1}</td><td><b>${item.model}</b></td><td>${item.wins}-${item.losses}-${item.draws}</td><td>${item.win_rate}%</td><td>${item.rating}</td></tr>`,
        )
        .join("")
    : '<tr><td colspan="5" class="muted">No evaluation results reported.</td></tr>';
}
async function loadSelfplayStatus() {
  try {
    const data = await apiRequest("/api/selfplay_status");
    $("#sp-iteration").textContent = data.current_iteration ?? "--";
    $("#sp-games").textContent = data.games_played ?? "--";
    $("#sp-examples").textContent =
      data.total_examples == null
        ? "--"
        : Number(data.total_examples).toLocaleString();
    $("#sp-models").textContent = data.models_trained ?? "--";
    $("#sp-status").textContent = data.is_active ? "Running" : "Idle";
    $("#sp-log").textContent =
      data.log?.join("\n") || data.progress || "No log lines reported.";
    if (data.model_files) {
      $("#checkpoint-grid").innerHTML = data.model_files
        .map(
          (item) =>
            `<div class="checkpoint"><strong>ITER ${item.iteration}</strong><span>${item.filename} / ${item.size_kb} KB</span></div>`,
        )
        .join("");
    }
  } catch (error) {
    toast(`Self-play telemetry unavailable: ${error.message}`);
  }
}
let autoSelfplayTimer = null;

async function startSelfplay() {
  const btn = $("#sp-start");
  if (autoSelfplayTimer) {
    clearInterval(autoSelfplayTimer);
    autoSelfplayTimer = null;
    if (btn) btn.textContent = "Start self-play";
    toast("Self-play paused.");
    return;
  }

  const payload = {
    board_size: 12,
    model_black: $("#sp-model-black")?.value || "untrained",
    sims_black: Number($("#sp-sims-black")?.value) || 25,
    temp_black: Number($("#sp-temp-black")?.value) || 1,
    model_white: $("#sp-model-white")?.value || "untrained",
    sims_white: Number($("#sp-sims-white")?.value) || 25,
    temp_white: Number($("#sp-temp-white")?.value || $("#sp-temp-black")?.value) || 1,
    temp_threshold: Number($("#sp-temp-threshold")?.value) || 30,
  };

  try {
    const initRes = await apiRequest("/api/selfplay/new_game", "POST", payload);
    updateGame(initRes);
    if (btn) btn.textContent = "Pause self-play";
    toast("Self-play started", "ok");

    autoSelfplayTimer = setInterval(async () => {
      if (state.gameOver) {
        clearInterval(autoSelfplayTimer);
        autoSelfplayTimer = null;
        if (btn) btn.textContent = "Start self-play";
        toast("Self-play match finished!", "ok");
        return;
      }
      await selfplayStep();
    }, 400);
  } catch (error) {
    if (btn) btn.textContent = "Start self-play";
    toast(`Self-play could not start: ${error.message}`);
  }
}

async function selfplayStep() {
  try {
    const res = await apiRequest("/api/selfplay/step", "POST");
    updateGame(res);
  } catch (error) {
    if (autoSelfplayTimer) {
      clearInterval(autoSelfplayTimer);
      autoSelfplayTimer = null;
      const btn = $("#sp-start");
      if (btn) btn.textContent = "Start self-play";
    }
    toast(`Self-play step failed: ${error.message}`);
  }
}

async function saveSelfplay() {
  try {
    const data = await apiRequest("/api/selfplay/save_data", "POST");
    toast(`Saved ${data.examples_count || 0} examples`, "ok");
  } catch (error) {
    toast(`No self-play data saved: ${error.message}`);
  }
}

async function startTraining() {
  const btn = $("#sp-train");
  try {
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Training in progress...";
    }
    await apiRequest("/api/train?iterations=1", "POST");
    toast("Training worker started", "ok");
    const poll = setInterval(async () => {
      const data = await apiRequest("/api/train_status");
      $("#sp-status").textContent = data.is_training ? "Training" : "Idle";
      $("#sp-log").textContent =
        data.log?.join("\n") || data.progress || "No training log reported.";
      if (!data.is_training) {
        clearInterval(poll);
        if (btn) {
          btn.disabled = false;
          btn.textContent = "Train one iteration";
        }
      }
    }, 2000);
  } catch (error) {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Train one iteration";
    }
    toast(`Training could not start: ${error.message}`);
  }
}
async function runEvaluation() {
  const payload = {
    model_a_file: $("#eval-model-a").value,
    model_b_file: $("#eval-model-b").value,
    sims_a: 25,
    sims_b: 100,
    num_games: Number($("#eval-games").value) || 10,
    board_size: Number($("#eval-board").value) || 12,
  };
  $("#eval-status").textContent = "Running";
  try {
    await apiRequest("/api/evaluate", "POST", payload);
    evaluationPoll = setInterval(async () => {
      const data = await apiRequest("/api/evaluation_status");
      if (!data.is_evaluating) {
        clearInterval(evaluationPoll);
        $("#eval-status").textContent = "Complete";
        renderEvaluation(
          await apiRequest(
            `/api/evaluation_stats?model_a=${encodeURIComponent(payload.model_a_file)}&model_b=${encodeURIComponent(payload.model_b_file)}`,
          ),
        );
      }
    }, 1500);
  } catch (error) {
    clearInterval(evaluationPoll);
    toast(`Evaluation could not start: ${error.message}`);
    $("#eval-status").textContent = "Error";
  }
}
function changeTab(tab) {
  $$(".tab").forEach((button) =>
    button.classList.toggle("active", button.dataset.tab === tab),
  );
  $$(".tab-panel").forEach((panel) =>
    panel.classList.toggle("active", panel.dataset.panel === tab),
  );
  if (tab === "selfplay") {
    loadSelfplayStatus();
    clearInterval(selfplayPoll);
    selfplayPoll = setInterval(loadSelfplayStatus, 5000);
  } else clearInterval(selfplayPoll);
  if (tab === "evaluation") loadEvaluation();
}
function bind() {
  $$(".tab").forEach((button) =>
    button.addEventListener("click", () => changeTab(button.dataset.tab)),
  );
  $$("[data-board]").forEach((button) =>
    button.addEventListener("click", () => {
      $$("[data-board]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.boardSize = Number(button.dataset.board);
      loadLevels();
    }),
  );
  $$("[data-color]").forEach((button) =>
    button.addEventListener("click", () => {
      $$("[data-color]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.humanColor = Number(button.dataset.color);
    }),
  );
  $("#model-select").addEventListener("change", (event) => {
    state.modelFile = event.target.value;
  });
  $("#sims-range").addEventListener("input", (event) => {
    state.simulations = Number(event.target.value);
    $("#sims-value").textContent = state.simulations;
  });
  $("#new-game").addEventListener("click", startGame);
  $("[data-action=pass]").addEventListener("click", passMove);
  $("[data-action=undo]").addEventListener("click", undoMove);
  $("[data-action=hint]").addEventListener("click", hint);
  $("[data-action=ai]").addEventListener("click", aiMove);
  $("#cancel-thinking").addEventListener("click", () =>
    toast(
      "The current request cannot be cancelled by the backend; waiting for its response.",
    ),
  );
  $("#wake-retry").addEventListener("click", () => {
    wakeAttempt = 0;
    wakeEngine();
  });
  $("#sp-start").addEventListener("click", startSelfplay);
  $("#sp-step").addEventListener("click", selfplayStep);
  $("#sp-save").addEventListener("click", saveSelfplay);
  $("#sp-train").addEventListener("click", startTraining);
  $("#eval-start").addEventListener("click", runEvaluation);
  $$("[data-close-dialog]").forEach((btn) =>
    btn.addEventListener("click", () => $("#game-dialog")?.close()),
  );
  $("#dialog-new-game").addEventListener("click", () => {
    $("#game-dialog").close();
    startGame();
  });
  window.addEventListener("keydown", (event) => {
    if (event.target.matches("input,select,textarea")) return;
    if (event.key.toLowerCase() === "p") passMove();
    if (event.key.toLowerCase() === "u") undoMove();
    if (event.key.toLowerCase() === "h") hint();
  });
}
bind();
wakeEngine();
