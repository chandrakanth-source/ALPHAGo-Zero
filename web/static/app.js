/**
 * AlphaGo Zero Arena - Frontend Controller
 * Manages Goban rendering, stone placement, AI level switching,
 * MCTS hint requests, audio synthesis, and training loops.
 */

// ---------------------------------------------------------------------------
// Audio Synthesizer (Realistic Go Stone "Clack")
// ---------------------------------------------------------------------------

class SoundEngine {
    constructor() {
        this.ctx = null;
    }

    init() {
        if (!this.ctx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioContext();
        }
    }

    playStoneClick() {
        try {
            this.init();
            if (!this.ctx) return;
            const now = this.ctx.currentTime;

            // Transient click (wood collision)
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            const filter = this.ctx.createBiquadFilter();

            osc.type = 'triangle';
            osc.frequency.setValueAtTime(320 + Math.random() * 80, now);
            osc.frequency.exponentialRampToValueAtTime(60, now + 0.08);

            filter.type = 'lowpass';
            filter.frequency.setValueAtTime(1400, now);

            gain.gain.setValueAtTime(0.45, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.09);

            osc.connect(filter);
            filter.connect(gain);
            gain.connect(this.ctx.destination);

            osc.start(now);
            osc.stop(now + 0.09);
        } catch (e) {
            // Audio policy might block until user gesture
        }
    }
}

const soundEngine = new SoundEngine();

// ---------------------------------------------------------------------------
// App State
// ---------------------------------------------------------------------------

const state = {
    boardSize: 12,
    humanColor: 1,         // 1 = Black, -1 = White
    selectedLevel: 'custom',
    simulations: 50,
    selectedModelFile: '',  // used by legacy paths
    iterationModelFile: 'model_iteration_1.pt', // active iteration .pt filename
    selectedIteration: 1,
    boardState: Array(12).fill(0).map(() => Array(12).fill(0)),
    legalMoves: [],
    lastAiMove: null,
    currentHint: null,
    isAiThinking: false,
    gameOver: false,
    levelsData: null,
    iterationsData: []      // [{iteration, filename, size_kb}, ...]
};

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------

const gobanEl = document.getElementById('goban-board');
const aiThinkingOverlay = document.getElementById('ai-thinking-overlay');
const serverStatusDot = document.getElementById('server-status-dot');
const serverStatusText = document.getElementById('server-status-text');
const currentTurnBadge = document.getElementById('current-turn-badge');

// Level & Settings (legacy hidden elements for compat)
const toggleCustomBtn = document.getElementById('toggle-custom-config');
const customConfigBody = document.getElementById('custom-config-body');
const selectModelFile = document.getElementById('select-model-file');
const inputSimulations = document.getElementById('input-simulations');
const simsValLabel = document.getElementById('sims-val');
const colorBtns = document.querySelectorAll('.color-btn');
const sizeBtns = document.querySelectorAll('.size-btn');
const btnNewGame = document.getElementById('btn-new-game');

// Iteration Picker
const iterSlider     = document.getElementById('iter-slider');
const iterSelect     = document.getElementById('iter-select');
const iterBadge      = document.getElementById('iter-badge');
const iterMaxLabel   = document.getElementById('iter-max-label');
const iterSimsInput  = document.getElementById('iter-sims');
const iterSimsVal    = document.getElementById('iter-sims-val');

// Live Eval & Scores
const evalBarFill = document.getElementById('eval-bar-fill');
const evalBlackPct = document.getElementById('eval-black-pct');
const evalWhitePct = document.getElementById('eval-white-pct');
const evalStatusNote = document.getElementById('eval-status-note');
const blackScoreVal = document.getElementById('black-score-val');
const whiteScoreVal = document.getElementById('white-score-val');
const blackStoneCount = document.getElementById('black-stone-count');
const whiteStoneCount = document.getElementById('white-stone-count');
const blackPlayerName = document.getElementById('black-player-name');
const whitePlayerName = document.getElementById('white-player-name');

// Toolbar Controls
const btnHint = document.getElementById('btn-hint');
const btnPass = document.getElementById('btn-pass');
const btnUndo = document.getElementById('btn-undo');
const btnAutoAi = document.getElementById('btn-auto-ai');
const btnResign = document.getElementById('btn-resign');
const hintBanner = document.getElementById('hint-banner');
const hintText = document.getElementById('hint-text');
const btnCloseHint = document.getElementById('btn-close-hint');

// History & Telemetry
const moveHistoryList = document.getElementById('move-history-list');
const moveCountBadge = document.getElementById('move-count-badge');
const telemetryLevel = document.getElementById('telemetry-level');
const telemetrySims = document.getElementById('telemetry-sims');
const telemetryTime = document.getElementById('telemetry-time');
const telemetryPasses = document.getElementById('telemetry-passes');

// Modals
const trainingModal = document.getElementById('training-modal');
const infoModal = document.getElementById('info-modal');
const gameoverModal = document.getElementById('gameover-modal');
const evalModal = document.getElementById('eval-modal');
const selfplayModal = document.getElementById('selfplay-modal');
const btnOpenTraining = document.getElementById('btn-open-training');
const btnCloseTraining = document.getElementById('btn-close-training');
const btnOpenInfo = document.getElementById('btn-open-info');
const btnCloseInfo = document.getElementById('btn-close-info');
const btnOpenEval = document.getElementById('btn-open-eval');
const btnCloseEval = document.getElementById('btn-close-eval');
const btnOpenSelfplay = document.getElementById('btn-open-selfplay');
const btnCloseSelfplay = document.getElementById('btn-close-selfplay');
const btnStartTraining = document.getElementById('btn-start-training');
const inputTrainIterations = document.getElementById('input-train-iterations');
const trainStatusBadge = document.getElementById('train-status-badge');
const trainProgressFill = document.getElementById('train-progress-fill');
const trainLogTerminal = document.getElementById('train-log-terminal');
const btnModalNewGame = document.getElementById('btn-modal-newgame');

// Evaluation Modal Controls
const evalSelectModelA = document.getElementById('eval-select-model-a');
const evalSelectModelB = document.getElementById('eval-select-model-b');
const evalSimsA = document.getElementById('eval-sims-a');
const evalSimsB = document.getElementById('eval-sims-b');
const evalNumGames = document.getElementById('eval-num-games');
const btnStartEval = document.getElementById('btn-start-eval');
const evalStatusBadge = document.getElementById('eval-status-badge');
const evalRateA = document.getElementById('eval-rate-a');
const evalRateB = document.getElementById('eval-rate-b');
const evalNameA = document.getElementById('eval-name-a');
const evalNameB = document.getElementById('eval-name-b');
const evalDrawsCount = document.getElementById('eval-draws-count');
const evalAvgDur = document.getElementById('eval-avg-dur');
const evalLogTerminal = document.getElementById('eval-log-terminal');

// Play vs Opponent Modal
const opponentModal = document.getElementById('opponent-modal');
const btnOpenOpponent = document.getElementById('btn-open-opponent');
const btnCloseOpponent = document.getElementById('btn-close-opponent');
const oppModalIterSlider = document.getElementById('opp-modal-iter-slider');
const oppModalIterSelect = document.getElementById('opp-modal-iter-select');
const oppModalIterBadge = document.getElementById('opp-modal-iter-badge');
const oppModalIterMax = document.getElementById('opp-modal-iter-max');
const oppModalSimsSlider = document.getElementById('opp-modal-sims-slider');
const oppModalSimsVal = document.getElementById('opp-modal-sims-val');
const oppModalColorBlack = document.getElementById('opp-modal-color-black');
const oppModalColorWhite = document.getElementById('opp-modal-color-white');
const oppModalSize12 = document.getElementById('opp-modal-size-12');
const oppModalSize5 = document.getElementById('opp-modal-size-5');
const btnOppModalStart = document.getElementById('btn-opp-modal-start');

// ---------------------------------------------------------------------------
// API Client
// ---------------------------------------------------------------------------

async function apiRequest(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: { 'Content-Type': 'application/json' }
        };
        if (data) options.body = JSON.stringify(data);
        const res = await fetch(endpoint, options);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            const detailStr = String(err.detail || 'API request failed');
            const error = new Error(detailStr);
            if (detailStr.includes('No active game session') || detailStr.includes('Call /api/new_game first')) {
                error.isNoSession = true;
            }
            throw error;
        }
        return await res.json();
    } catch (e) {
        console.error(`Error during API ${endpoint}:`, e);
        throw e;
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

async function init() {
    renderGoban();
    setupEventListeners();
    setupSelfPlayEventListeners();

    try {
        await fetchLevels();
    } catch (e) {
        console.warn("fetchLevels failed:", e);
    }

    try {
        await fetchIterations();
    } catch (e) {
        console.warn("fetchIterations failed:", e);
    }

    try {
        await startNewGame();
    } catch (e) {
        console.warn("startNewGame failed:", e);
    }
}

async function fetchLevels() {
    try {
        const data = await apiRequest(`/api/levels?board_size=${state.boardSize}`);
        state.levelsData = data;

        // Populate eval modal model selects
        if (evalSelectModelA && evalSelectModelB) {
            evalSelectModelA.innerHTML = '<option value="untrained">Novice (Untrained / Tabula Rasa)</option>';
            evalSelectModelB.innerHTML = '<option value="untrained">Novice (Untrained / Tabula Rasa)</option>';
        }
        if (data.available_checkpoints) {
            data.available_checkpoints.forEach(cp => {
                if (evalSelectModelA && evalSelectModelB) {
                    const compatStr = cp.detected_board_size ? ` (${cp.detected_board_size}x${cp.detected_board_size})` : '';
                    const optA = document.createElement('option');
                    optA.value = cp.filename;
                    optA.textContent = `${cp.filename}${compatStr}`;
                    evalSelectModelA.appendChild(optA);

                    const optB = document.createElement('option');
                    optB.value = cp.filename;
                    optB.textContent = `${cp.filename}${compatStr}`;
                    evalSelectModelB.appendChild(optB);
                }
            });
            if (evalSelectModelA && data.available_checkpoints.find(c => c.filename === 'model_iteration_1.pt')) {
                evalSelectModelA.value = 'model_iteration_1.pt';
            }
            if (evalSelectModelB && data.available_checkpoints.find(c => c.filename === 'latest_model.pt')) {
                evalSelectModelB.value = 'latest_model.pt';
            }
        }
        updateActiveLevelUI();
    } catch (e) {
        serverStatusDot.classList.remove('active');
        serverStatusText.textContent = 'Engine Offline';
    }
}

// ---------------------------------------------------------------------------
// Iteration Picker Logic
// ---------------------------------------------------------------------------

async function fetchIterations() {
    try {
        const data = await apiRequest('/api/iterations');
        state.iterationsData = data.iterations || [];

        populateSelfPlayModels();

        if (state.iterationsData.length === 0) return;

        // Configure slider range
        const minIter = state.iterationsData[0].iteration;
        const maxIter = state.iterationsData[state.iterationsData.length - 1].iteration;
        iterSlider.min = minIter;
        iterSlider.max = maxIter;
        iterSlider.value = minIter;
        if (iterMaxLabel) iterMaxLabel.textContent = maxIter;

        // Populate dropdown
        if (iterSelect) {
            iterSelect.innerHTML = '';
            state.iterationsData.forEach(it => {
                const opt = document.createElement('option');
                opt.value = it.filename;
                opt.textContent = `Iteration ${it.iteration}  (${it.size_kb} KB)`;
                opt.dataset.iteration = it.iteration;
                iterSelect.appendChild(opt);
            });
        }

        // Set default to iteration 1
        applyIteration(minIter);
    } catch (e) {
        console.warn('Could not load iterations:', e);
    }
}

function applyIteration(iterNum) {
    const entry = state.iterationsData.find(it => it.iteration === iterNum);
    if (!entry) return;

    state.selectedIteration = iterNum;
    state.iterationModelFile = entry.filename;
    state.selectedLevel = 'custom';

    // Sync slider
    if (iterSlider) iterSlider.value = iterNum;
    // Sync dropdown
    if (iterSelect) iterSelect.value = entry.filename;
    // Badge
    if (iterBadge) iterBadge.textContent = `Iteration ${iterNum}`;

    updateActiveLevelUI();
    syncOpponentModalUI();
}

// ---------------------------------------------------------------------------
// Game Loop & Goban Rendering
// ---------------------------------------------------------------------------

async function startNewGame() {
    hideModals();
    state.currentHint = null;
    hintBanner.classList.add('hidden');
    setAiThinking(true);

    try {
        const payload = {
            board_size: state.boardSize,
            human_color: state.humanColor,
            level: state.selectedLevel,
            simulations: state.simulations,
            model_file: state.iterationModelFile || state.selectedModelFile || null
        };

        const res = await apiRequest('/api/new_game', 'POST', payload);
        updateGameState(res);
        renderGoban();
    } catch (e) {
        alert(`Failed to start game: ${e.message}`);
    } finally {
        setAiThinking(false);
    }
}

function updateGameState(data) {
    if (!data.active) return;

    state.boardSize = data.board_size;
    state.boardState = data.board;
    state.legalMoves = data.legal_moves || [];
    state.lastAiMove = data.last_ai_move;
    state.gameOver = data.game_over;

    // Current turn indicator
    const isBlackTurn = data.current_player === 1;
    const isHumanTurn = data.current_player === state.humanColor;
    
    currentTurnBadge.innerHTML = `
        <span class="stone-preview ${isBlackTurn ? 'black' : 'white'}"></span>
        <span class="turn-text">${isBlackTurn ? "Black's Turn" : "White's Turn"} ${isHumanTurn ? '(You)' : '(AI)'}</span>
    `;

    // Names in player cards
    if (state.humanColor === 1) {
        blackPlayerName.textContent = 'Human (You)';
        whitePlayerName.textContent = `AlphaGo (${capitalize(state.selectedLevel)})`;
    } else {
        blackPlayerName.textContent = `AlphaGo (${capitalize(state.selectedLevel)})`;
        whitePlayerName.textContent = 'Human (You)';
    }

    // Live Scores & Stones
    blackScoreVal.textContent = `${data.black_score} pts`;
    whiteScoreVal.textContent = `${data.white_score} pts`;
    blackStoneCount.textContent = `Stones: ${data.black_stones}`;
    whiteStoneCount.textContent = `Stones: ${data.white_stones}`;

    // Win rate evaluation
    const blackWinPct = (data.ai_win_prob_black * 100).toFixed(1);
    const whiteWinPct = (100 - parseFloat(blackWinPct)).toFixed(1);
    evalBarFill.style.width = `${blackWinPct}%`;
    evalBlackPct.textContent = `${blackWinPct}%`;
    evalWhitePct.textContent = `${whiteWinPct}%`;

    if (blackWinPct > 65) {
        evalStatusNote.textContent = 'Black has strategic advantage';
    } else if (blackWinPct < 35) {
        evalStatusNote.textContent = 'White has strategic advantage';
    } else {
        evalStatusNote.textContent = 'Even tactical position';
    }

    // Telemetry
    const activeLevelText = data.active_level_name || `Level: ${capitalize(state.selectedLevel)}`;
    const activeModelFile = data.active_model_file ? ` [${data.active_model_file}]` : '';
    telemetryLevel.textContent = `${activeLevelText}${activeModelFile}`;
    telemetrySims.textContent = `${data.active_simulations || state.simulations} / move`;
    telemetryPasses.textContent = `${data.consecutive_passes} / 2`;
    if (data.latest_ai_info && data.latest_ai_info.time) {
        telemetryTime.textContent = `${data.latest_ai_info.time}s`;
    }

    updateActiveLevelUI(data);

    // Move History
    renderMoveHistory(data.move_history);

    // Game Over check
    if (data.game_over) {
        showGameOverModal(data);
    }

    renderGoban();
}

function renderGoban() {
    const size = state.boardSize;
    gobanEl.innerHTML = '';

    const width = gobanEl.clientWidth || 580;
    const padding = 36;
    const cellSize = (width - padding * 2) / (size - 1);

    // 1. Draw SVG Grid Lines
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('class', 'grid-svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${width}`);

    for (let i = 0; i < size; i++) {
        const pos = padding + i * cellSize;

        // Horizontal line
        const hLine = document.createElementNS(svgNS, 'line');
        hLine.setAttribute('x1', padding);
        hLine.setAttribute('y1', pos);
        hLine.setAttribute('x2', width - padding);
        hLine.setAttribute('y2', pos);
        hLine.setAttribute('stroke', '#42240c');
        hLine.setAttribute('stroke-width', '1.5');
        svg.appendChild(hLine);

        // Vertical line
        const vLine = document.createElementNS(svgNS, 'line');
        vLine.setAttribute('x1', pos);
        vLine.setAttribute('y1', padding);
        vLine.setAttribute('x2', pos);
        vLine.setAttribute('y2', width - padding);
        vLine.setAttribute('stroke', '#42240c');
        vLine.setAttribute('stroke-width', '1.5');
        svg.appendChild(vLine);
    }

    // 2. Draw Star Points (Hoshi)
    const starPoints = getStarPoints(size);
    starPoints.forEach(([r, c]) => {
        const cx = padding + c * cellSize;
        const cy = padding + r * cellSize;
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('cx', cx);
        circle.setAttribute('cy', cy);
        circle.setAttribute('r', '4');
        circle.setAttribute('fill', '#2a1403');
        svg.appendChild(circle);
    });

    gobanEl.appendChild(svg);

    // 3. Render Coordinate Headers (A-L, 1-12)
    for (let i = 0; i < size; i++) {
        const colLetter = String.fromCharCode(65 + i);
        const colX = padding + i * cellSize;

        // Top & Bottom Col labels
        const topLabel = document.createElement('span');
        topLabel.className = 'coord-label';
        topLabel.style.left = `${colX}px`;
        topLabel.style.top = '12px';
        topLabel.textContent = colLetter;
        gobanEl.appendChild(topLabel);

        const botLabel = document.createElement('span');
        botLabel.className = 'coord-label';
        botLabel.style.left = `${colX}px`;
        botLabel.style.bottom = '12px';
        botLabel.textContent = colLetter;
        gobanEl.appendChild(botLabel);

        // Row Numbers (1 at bottom or 1 at top)
        const rowNum = (i + 1).toString();
        const rowY = padding + i * cellSize;

        const leftLabel = document.createElement('span');
        leftLabel.className = 'coord-label';
        leftLabel.style.left = '14px';
        leftLabel.style.top = `${rowY - 6}px`;
        leftLabel.textContent = rowNum;
        gobanEl.appendChild(leftLabel);

        const rightLabel = document.createElement('span');
        rightLabel.className = 'coord-label';
        rightLabel.style.right = '14px';
        rightLabel.style.top = `${rowY - 6}px`;
        rightLabel.textContent = rowNum;
        gobanEl.appendChild(rightLabel);
    }

    // 4. Render Intersections & Stones
    const stoneRadius = cellSize * 0.44;

    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            const x = padding + c * cellSize;
            const y = padding + r * cellSize;
            const val = state.boardState[r] ? state.boardState[r][c] : 0;

            if (val === 0) {
                // Empty Intersection for Clicking
                const inter = document.createElement('div');
                inter.className = `intersection hover-${state.humanColor === 1 ? 'black' : 'white'}`;
                inter.style.left = `${x}px`;
                inter.style.top = `${y}px`;
                inter.style.width = `${cellSize}px`;
                inter.style.height = `${cellSize}px`;

                inter.addEventListener('click', () => handleHumanMove(r, c));
                gobanEl.appendChild(inter);
            } else {
                // Stone Placed
                const stone = document.createElement('div');
                stone.className = `stone ${val === 1 ? 'black' : 'white'}`;
                stone.style.left = `${x}px`;
                stone.style.top = `${y}px`;
                stone.style.width = `${stoneRadius * 2}px`;
                stone.style.height = `${stoneRadius * 2}px`;

                // Highlight last move
                if (state.lastAiMove && state.lastAiMove[0] === r && state.lastAiMove[1] === c) {
                    stone.classList.add('last-move');
                }

                gobanEl.appendChild(stone);
            }

            // Hint highlight
            if (state.currentHint && state.currentHint[0] === r && state.currentHint[1] === c) {
                const hintEl = document.createElement('div');
                hintEl.className = 'hint-glow';
                hintEl.style.left = `${x}px`;
                hintEl.style.top = `${y}px`;
                hintEl.style.width = `${stoneRadius * 2.2}px`;
                hintEl.style.height = `${stoneRadius * 2.2}px`;
                gobanEl.appendChild(hintEl);
            }
        }
    }
}

function getStarPoints(size) {
    if (size === 12) {
        return [[3, 3], [3, 8], [8, 3], [8, 8]];
    } else if (size === 5) {
        return [[2, 2]];
    }
    return [];
}

// ---------------------------------------------------------------------------
// Human & AI Interactions
// ---------------------------------------------------------------------------

async function handleMoveError(e, defaultMsg = 'Action failed') {
    if (e.isNoSession || (e.message && (e.message.includes('No active game session') || e.message.includes('Call /api/new_game')))) {
        console.warn('Backend session missing. Auto-reinitializing new game session...');
        await startNewGame();
    } else {
        alert(e.message || defaultMsg);
    }
}

async function handleHumanMove(row, col) {
    if (state.isAiThinking || state.gameOver) return;

    soundEngine.playStoneClick();
    setAiThinking(true);
    state.currentHint = null;
    hintBanner.classList.add('hidden');

    try {
        const res = await apiRequest('/api/move', 'POST', { row, col, is_pass: false });
        soundEngine.playStoneClick();
        updateGameState(res);
    } catch (e) {
        await handleMoveError(e, 'Illegal move');
    } finally {
        setAiThinking(false);
    }
}

async function handlePass() {
    if (state.isAiThinking || state.gameOver) return;

    if (!confirm('Are you sure you want to PASS your turn?')) return;

    setAiThinking(true);
    try {
        const res = await apiRequest('/api/move', 'POST', { is_pass: true });
        updateGameState(res);
    } catch (e) {
        await handleMoveError(e, 'Pass failed');
    } finally {
        setAiThinking(false);
    }
}

async function handleUndo() {
    if (state.isAiThinking) return;

    try {
        const res = await apiRequest('/api/undo', 'POST');
        updateGameState(res);
    } catch (e) {
        await handleMoveError(e, 'Cannot undo move.');
    }
}

async function handleAutoAi() {
    if (state.isAiThinking || state.gameOver) return;

    setAiThinking(true);
    try {
        const res = await apiRequest('/api/ai_move', 'POST');
        soundEngine.playStoneClick();
        updateGameState(res);
    } catch (e) {
        await handleMoveError(e, 'AI move failed');
    } finally {
        setAiThinking(false);
    }
}

async function handleHint() {
    if (state.isAiThinking || state.gameOver) return;

    hintBanner.classList.remove('hidden');
    hintText.textContent = 'AlphaGo MCTS search in progress...';

    try {
        const hint = await apiRequest('/api/hint', 'POST');
        if (hint.coords) {
            state.currentHint = hint.coords;
            hintText.textContent = `${hint.explanation}`;
            renderGoban();
        } else {
            hintText.textContent = `${hint.explanation}`;
        }
    } catch (e) {
        if (e.isNoSession) {
            await startNewGame();
        } else {
            hintText.textContent = 'Could not calculate hint.';
        }
    }
}

function handleResign() {
    if (state.gameOver) return;
    if (confirm('Are you sure you want to resign the game?')) {
        showGameOverModal({
            winner: -state.humanColor,
            black_score: state.humanColor === 1 ? 0 : 99,
            white_score: state.humanColor === -1 ? 0 : 99,
            resignation: true
        });
    }
}

// ---------------------------------------------------------------------------
// Move History Rendering
// ---------------------------------------------------------------------------

function renderMoveHistory(history) {
    if (!history || history.length === 0) {
        moveHistoryList.innerHTML = '<div class="empty-history">Game started. Waiting for first move...</div>';
        moveCountBadge.textContent = '0 Moves';
        return;
    }

    moveCountBadge.textContent = `${history.length} Moves`;
    moveHistoryList.innerHTML = '';

    history.forEach((m, idx) => {
        const row = document.createElement('div');
        row.className = `move-row ${m.player === 'AI' ? 'ai-move' : ''}`;

        const isBlack = m.color === 1;
        const colorLabel = isBlack ? 'B' : 'W';
        const metaStr = m.time ? `${m.time}s` : '';

        row.innerHTML = `
            <span class="move-num">#${idx + 1}</span>
            <span class="move-action">[${colorLabel}] ${m.action}</span>
            <span class="move-meta">${m.player} ${metaStr}</span>
        `;
        moveHistoryList.appendChild(row);
    });

    moveHistoryList.scrollTop = moveHistoryList.scrollHeight;
}

// ---------------------------------------------------------------------------
// Modals & Training UI
// ---------------------------------------------------------------------------

function showGameOverModal(data) {
    state.gameOver = true;
    const humanWon = data.winner === state.humanColor;
    const titleEl = document.getElementById('gameover-title');
    const subtitleEl = document.getElementById('gameover-subtitle');
    const iconEl = document.getElementById('gameover-icon');

    if (data.resignation) {
        titleEl.textContent = humanWon ? 'Victory!' : 'Resigned';
        subtitleEl.textContent = humanWon ? 'Opponent resigned.' : 'You resigned the game.';
        iconEl.textContent = humanWon ? '🏆' : '🏳️';
    } else if (data.winner === 0) {
        titleEl.textContent = 'Draw Game (Jigo)';
        subtitleEl.textContent = 'Equal territory scores.';
        iconEl.textContent = '🤝';
    } else {
        titleEl.textContent = humanWon ? 'Victory!' : 'AlphaGo Wins!';
        subtitleEl.textContent = `Winner: ${data.winner === 1 ? 'Black' : 'White'}`;
        iconEl.textContent = humanWon ? '🏆' : '🤖';
    }

    document.getElementById('final-black-score').textContent = data.black_score;
    document.getElementById('final-white-score').textContent = data.white_score;

    gameoverModal.classList.remove('hidden');
}

function hideModals() {
    trainingModal.classList.add('hidden');
    infoModal.classList.add('hidden');
    gameoverModal.classList.add('hidden');
    if (evalModal) evalModal.classList.add('hidden');
    if (opponentModal) opponentModal.classList.add('hidden');
    if (selfplayModal) {
        selfplayModal.classList.add('hidden');
        stopSelfplayStatusPolling();
    }
}

function syncOpponentModalUI() {
    if (!opponentModal) return;

    if (state.iterationsData.length > 0) {
        const minIter = state.iterationsData[0].iteration;
        const maxIter = state.iterationsData[state.iterationsData.length - 1].iteration;
        if (oppModalIterSlider) {
            oppModalIterSlider.min = minIter;
            oppModalIterSlider.max = maxIter;
            oppModalIterSlider.value = state.selectedIteration || minIter;
        }
        if (oppModalIterMax) oppModalIterMax.textContent = `Iter ${maxIter}`;

        if (oppModalIterSelect) {
            oppModalIterSelect.innerHTML = '';
            state.iterationsData.forEach(it => {
                const opt = document.createElement('option');
                opt.value = it.filename;
                opt.textContent = `Iteration ${it.iteration} (${it.size_kb} KB)`;
                if (it.iteration === state.selectedIteration) opt.selected = true;
                oppModalIterSelect.appendChild(opt);
            });
        }
    }

    if (oppModalIterBadge) oppModalIterBadge.textContent = `Iteration ${state.selectedIteration || 1}`;
    if (oppModalSimsSlider) oppModalSimsSlider.value = state.simulations || 50;
    if (oppModalSimsVal) oppModalSimsVal.textContent = `${state.simulations || 50} sims / move`;

    if (oppModalColorBlack && oppModalColorWhite) {
        oppModalColorBlack.classList.toggle('active', state.humanColor === 1);
        oppModalColorWhite.classList.toggle('active', state.humanColor === -1);
    }

    if (oppModalSize12 && oppModalSize5) {
        oppModalSize12.classList.toggle('active', state.boardSize === 12);
        oppModalSize5.classList.toggle('active', state.boardSize === 5);
    }
}

let trainingPollInterval = null;
let evalPollInterval = null;

async function startTraining() {
    const iters = parseInt(inputTrainIterations.value) || 1;
    btnStartTraining.disabled = true;
    trainStatusBadge.textContent = 'Running...';
    trainStatusBadge.className = 'badge badge-gold';

    try {
        await apiRequest(`/api/train?iterations=${iters}`, 'POST');
        pollTrainingStatus();
    } catch (e) {
        alert(`Training launch failed: ${e.message}`);
        btnStartTraining.disabled = false;
    }
}

function pollTrainingStatus() {
    if (trainingPollInterval) clearInterval(trainingPollInterval);

    trainingPollInterval = setInterval(async () => {
        try {
            const status = await apiRequest('/api/train_status');
            trainProgressFill.style.width = status.is_training ? '65%' : '100%';
            trainLogTerminal.textContent = status.log.join('\n') || status.progress;
            trainLogTerminal.scrollTop = trainLogTerminal.scrollHeight;

            if (!status.is_training) {
                clearInterval(trainingPollInterval);
                btnStartTraining.disabled = false;
                trainStatusBadge.textContent = 'Finished';
                trainStatusBadge.className = 'badge badge-green';
                fetchLevels(); // Refresh new models
            }
        } catch (e) {
            clearInterval(trainingPollInterval);
            btnStartTraining.disabled = false;
        }
    }, 2000);
}

async function startEvaluation() {
    const payload = {
        model_a_file: evalSelectModelA.value,
        model_b_file: evalSelectModelB.value,
        sims_a: parseInt(evalSimsA.value) || 25,
        sims_b: parseInt(evalSimsB.value) || 100,
        num_games: parseInt(evalNumGames.value) || 20,
        board_size: state.boardSize
    };

    btnStartEval.disabled = true;
    evalStatusBadge.textContent = 'Running...';
    evalStatusBadge.className = 'badge badge-gold';

    try {
        await apiRequest('/api/evaluate', 'POST', payload);
        pollEvaluationStatus();
    } catch (e) {
        alert(`Evaluation launch failed: ${e.message}`);
        btnStartEval.disabled = false;
    }
}

function pollEvaluationStatus() {
    if (evalPollInterval) clearInterval(evalPollInterval);

    evalPollInterval = setInterval(async () => {
        try {
            const status = await apiRequest('/api/evaluation_status');
            evalNameA.textContent = `${status.model_a || 'Model A'} (${status.sims_a} Sims)`;
            evalNameB.textContent = `${status.model_b || 'Model B'} (${status.sims_b} Sims)`;
            evalRateA.textContent = `${status.win_rate_a}%`;
            evalRateB.textContent = `${status.win_rate_b}%`;
            evalDrawsCount.textContent = status.draws;
            evalAvgDur.textContent = `${status.avg_duration}s`;

            evalLogTerminal.textContent = status.log.join('\n') || status.progress;
            evalLogTerminal.scrollTop = evalLogTerminal.scrollHeight;

            if (!status.is_evaluating) {
                clearInterval(evalPollInterval);
                btnStartEval.disabled = false;
                evalStatusBadge.textContent = 'Completed';
                evalStatusBadge.className = 'badge badge-green';
                fetchEvaluationStats();
            }
        } catch (e) {
            clearInterval(evalPollInterval);
            btnStartEval.disabled = false;
        }
    }, 1500);
}

async function fetchEvaluationStats() {
    try {
        const modelA = evalSelectModelA ? evalSelectModelA.value : '';
        const modelB = evalSelectModelB ? evalSelectModelB.value : '';
        const data = await apiRequest(`/api/evaluation_stats?model_a=${encodeURIComponent(modelA)}&model_b=${encodeURIComponent(modelB)}`);

        // Update H2H Pairwise Card
        const h2h = data.h2h;
        if (h2h) {
            const h2hBadge = document.getElementById('h2h-match-count-badge');
            const h2hModelA = document.getElementById('h2h-model-a-lbl');
            const h2hModelB = document.getElementById('h2h-model-b-lbl');
            const h2hScoreSummary = document.getElementById('h2h-score-summary');
            const h2hBarA = document.getElementById('h2h-bar-a');
            const h2hBarB = document.getElementById('h2h-bar-b');
            const h2hWinsA = document.getElementById('h2h-wins-a');
            const h2hWinsB = document.getElementById('h2h-wins-b');
            const h2hDraws = document.getElementById('h2h-draws-label');

            if (h2hBadge) h2hBadge.textContent = `${h2h.games_played} Games Played`;
            if (h2hModelA) h2hModelA.textContent = h2h.model_a;
            if (h2hModelB) h2hModelB.textContent = h2h.model_b;
            if (h2hScoreSummary) h2hScoreSummary.textContent = `${h2h.model_a_wins} Wins - ${h2h.model_b_wins} Wins`;
            if (h2hWinsA) h2hWinsA.textContent = `${h2h.model_a_wins} Wins (${h2h.win_rate_a}%)`;
            if (h2hWinsB) h2hWinsB.textContent = `${h2h.model_b_wins} Wins (${h2h.win_rate_b}%)`;
            if (h2hDraws) h2hDraws.textContent = `${h2h.draws} Draws`;

            if (h2hBarA && h2hBarB) {
                const total = (h2h.model_a_wins + h2h.model_b_wins) || 1;
                const pctA = Math.round((h2h.model_a_wins / total) * 100);
                const pctB = 100 - pctA;
                h2hBarA.style.width = `${pctA}%`;
                h2hBarB.style.width = `${pctB}%`;
            }
        }

        // Update Leaderboard Table
        const leaderboardBody = document.getElementById('leaderboard-table-body');
        if (leaderboardBody && data.leaderboard) {
            leaderboardBody.innerHTML = '';
            data.leaderboard.forEach((item, index) => {
                const tr = document.createElement('tr');
                tr.style.cssText = 'border-bottom: 1px solid var(--border-color); font-size: 0.85rem;';
                
                const rankColor = index === 0 ? '#ffd700' : (index === 1 ? '#c0c0c0' : (index === 2 ? '#cd7f32' : 'var(--text-muted)'));
                const isBest = item.model === 'latest_model.pt';
                const badgeTag = isBest ? '<span class="badge badge-green" style="font-size:0.7rem;margin-left:6px;">Current Best</span>' : '';

                tr.innerHTML = `
                    <td style="padding: 10px 14px; font-weight: 700; color: ${rankColor};">#${index + 1}</td>
                    <td style="padding: 10px 14px; font-weight: 600;"><code>${item.model}</code> ${badgeTag}</td>
                    <td style="padding: 10px 14px;"><span class="badge" style="background:rgba(56,139,253,0.15);color:#58a6ff;">Iter ${item.iteration}</span></td>
                    <td style="padding: 10px 14px; text-align: center; font-weight: 700; color:#58a6ff;">${item.total_games}</td>
                    <td style="padding: 10px 14px; text-align: center;">${item.wins} W - ${item.losses} L - ${item.draws} D</td>
                    <td style="padding: 10px 14px; text-align: center; font-weight: 700; color: ${item.win_rate >= 50 ? '#39d353' : '#f85149'};">${item.win_rate}%</td>
                    <td style="padding: 10px 14px; text-align: center; font-weight: 600; color: #a371f7;">${item.rating}</td>
                    <td style="padding: 10px 14px; text-align: center;">
                        <button class="btn btn-sm btn-secondary btn-set-model-a" data-model="${item.model}" style="padding: 2px 8px; font-size: 0.75rem; margin-right: 4px;">Set Model A</button>
                        <button class="btn btn-sm btn-secondary btn-set-model-b" data-model="${item.model}" style="padding: 2px 8px; font-size: 0.75rem;">Set Model B</button>
                    </td>
                `;
                leaderboardBody.appendChild(tr);
            });

            leaderboardBody.querySelectorAll('.btn-set-model-a').forEach(btn => {
                btn.addEventListener('click', () => {
                    if (evalSelectModelA) evalSelectModelA.value = btn.dataset.model;
                    fetchEvaluationStats();
                });
            });
            leaderboardBody.querySelectorAll('.btn-set-model-b').forEach(btn => {
                btn.addEventListener('click', () => {
                    if (evalSelectModelB) evalSelectModelB.value = btn.dataset.model;
                    fetchEvaluationStats();
                });
            });
        }
    } catch (e) {
        console.warn("Failed fetching evaluation stats:", e);
    }
}


function updateActiveLevelUI(data = null) {
    const titleEl = document.getElementById('active-level-title');
    const subEl = document.getElementById('active-level-sub');

    if (!titleEl || !subEl) return;

    let levelTitle, checkpoint, sims;

    if (data && data.active_model_file) {
        // Prefer live server response
        checkpoint = data.active_model_file;
        sims = data.active_simulations;
        const iterMatch = checkpoint.match(/model_iteration_(\d+)\.pt/);
        levelTitle = iterMatch ? `Iteration ${iterMatch[1]}` : (data.active_level_name || checkpoint);
    } else {
        checkpoint = state.iterationModelFile || state.selectedModelFile || 'Untrained';
        sims = state.simulations;
        const iterMatch = checkpoint.match(/model_iteration_(\d+)\.pt/);
        levelTitle = iterMatch ? `Iteration ${iterMatch[1]}` : checkpoint;
    }

    titleEl.textContent = levelTitle;
    subEl.innerHTML = `Checkpoint: <code>${checkpoint}</code> (${sims} MCTS Sims)`;
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

function setupEventListeners() {
    // ------------------------------------------------------------------
    // Iteration Picker — slider
    // ------------------------------------------------------------------
    if (iterSlider) {
        let sliderDebounce = null;
        iterSlider.addEventListener('input', () => {
            const n = parseInt(iterSlider.value);
            if (iterBadge) iterBadge.textContent = `Iteration ${n}`;
            // Sync dropdown immediately for visual feedback
            const entry = state.iterationsData.find(it => it.iteration === n);
            if (entry && iterSelect) iterSelect.value = entry.filename;
            // Debounce game start
            clearTimeout(sliderDebounce);
            sliderDebounce = setTimeout(async () => {
                applyIteration(n);
                await startNewGame();
            }, 400);
        });
    }

    // Iteration Picker — dropdown
    if (iterSelect) {
        iterSelect.addEventListener('change', async () => {
            const filename = iterSelect.value;
            const entry = state.iterationsData.find(it => it.filename === filename);
            if (entry) {
                applyIteration(entry.iteration);
                await startNewGame();
            }
        });
    }

    // Iteration Picker — MCTS simulations slider
    if (iterSimsInput) {
        iterSimsInput.addEventListener('input', () => {
            state.simulations = parseInt(iterSimsInput.value);
            if (iterSimsVal) iterSimsVal.textContent = state.simulations;
            updateActiveLevelUI();
        });
        iterSimsInput.addEventListener('change', async () => {
            state.simulations = parseInt(iterSimsInput.value);
            if (iterSimsVal) iterSimsVal.textContent = state.simulations;
            updateActiveLevelUI();
            await startNewGame();
        });
    }

    // Color Toggle
    colorBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            colorBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.humanColor = parseInt(btn.dataset.color);
        });
    });

    // Board Size Toggle
    sizeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            sizeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.boardSize = parseInt(btn.dataset.size);
            fetchLevels();
        });
    });

    // Start New Game
    if (btnNewGame) btnNewGame.addEventListener('click', startNewGame);
    if (btnModalNewGame) btnModalNewGame.addEventListener('click', startNewGame);

    // Toolbar buttons
    if (btnPass) btnPass.addEventListener('click', handlePass);
    if (btnUndo) btnUndo.addEventListener('click', handleUndo);
    if (btnAutoAi) btnAutoAi.addEventListener('click', handleAutoAi);
    if (btnHint) btnHint.addEventListener('click', handleHint);
    if (btnResign) btnResign.addEventListener('click', handleResign);
    if (btnCloseHint && hintBanner) btnCloseHint.addEventListener('click', () => hintBanner.classList.add('hidden'));

    // Modals
    if (btnOpenTraining && trainingModal) btnOpenTraining.addEventListener('click', () => trainingModal.classList.remove('hidden'));
    if (btnCloseTraining && trainingModal) btnCloseTraining.addEventListener('click', () => trainingModal.classList.add('hidden'));
    if (btnOpenInfo && infoModal) btnOpenInfo.addEventListener('click', () => infoModal.classList.remove('hidden'));
    if (btnCloseInfo && infoModal) btnCloseInfo.addEventListener('click', () => infoModal.classList.add('hidden'));
    if (btnStartTraining) btnStartTraining.addEventListener('click', startTraining);

    if (btnOpenEval && evalModal) {
        btnOpenEval.addEventListener('click', () => {
            evalModal.classList.remove('hidden');
            fetchEvaluationStats();
        });
    }
    if (evalSelectModelA) evalSelectModelA.addEventListener('change', fetchEvaluationStats);
    if (evalSelectModelB) evalSelectModelB.addEventListener('change', fetchEvaluationStats);

    if (btnCloseEval && evalModal) {
        btnCloseEval.addEventListener('click', () => evalModal.classList.add('hidden'));
    }
    if (btnStartEval) {
        btnStartEval.addEventListener('click', startEvaluation);
    }

    if (btnOpenSelfplay && selfplayModal) {
        btnOpenSelfplay.addEventListener('click', () => {
            selfplayModal.classList.remove('hidden');
            startSelfplayStatusPolling();
        });
    }
    if (btnCloseSelfplay && selfplayModal) {
        btnCloseSelfplay.addEventListener('click', () => {
            selfplayModal.classList.add('hidden');
            stopSelfplayStatusPolling();
        });
    }

    // Play vs Opponent Modal Listeners
    if (btnOpenOpponent && opponentModal) {
        btnOpenOpponent.addEventListener('click', () => {
            syncOpponentModalUI();
            opponentModal.classList.remove('hidden');
        });
    }
    if (btnCloseOpponent && opponentModal) {
        btnCloseOpponent.addEventListener('click', () => opponentModal.classList.add('hidden'));
    }
    if (oppModalIterSlider) {
        oppModalIterSlider.addEventListener('input', () => {
            const n = parseInt(oppModalIterSlider.value);
            applyIteration(n);
        });
    }
    if (oppModalIterSelect) {
        oppModalIterSelect.addEventListener('change', () => {
            const filename = oppModalIterSelect.value;
            const entry = state.iterationsData.find(it => it.filename === filename);
            if (entry) applyIteration(entry.iteration);
        });
    }
    if (oppModalSimsSlider) {
        oppModalSimsSlider.addEventListener('input', () => {
            state.simulations = parseInt(oppModalSimsSlider.value);
            if (iterSimsInput) iterSimsInput.value = state.simulations;
            if (iterSimsVal) iterSimsVal.textContent = state.simulations;
            if (oppModalSimsVal) oppModalSimsVal.textContent = `${state.simulations} sims / move`;
            updateActiveLevelUI();
        });
    }
    if (oppModalColorBlack && oppModalColorWhite) {
        oppModalColorBlack.addEventListener('click', () => {
            state.humanColor = 1;
            colorBtns.forEach(b => b.classList.toggle('active', b.dataset.color == '1'));
            syncOpponentModalUI();
        });
        oppModalColorWhite.addEventListener('click', () => {
            state.humanColor = -1;
            colorBtns.forEach(b => b.classList.toggle('active', b.dataset.color == '-1'));
            syncOpponentModalUI();
        });
    }
    if (oppModalSize12 && oppModalSize5) {
        oppModalSize12.addEventListener('click', () => {
            state.boardSize = 12;
            sizeBtns.forEach(b => b.classList.toggle('active', b.dataset.size == '12'));
            syncOpponentModalUI();
            fetchLevels();
        });
        oppModalSize5.addEventListener('click', () => {
            state.boardSize = 5;
            sizeBtns.forEach(b => b.classList.toggle('active', b.dataset.size == '5'));
            syncOpponentModalUI();
            fetchLevels();
        });
    }
    if (btnOppModalStart) {
        btnOppModalStart.addEventListener('click', async () => {
            opponentModal.classList.add('hidden');
            await startNewGame();
        });
    }

    // Backdrop click (click outside modal box) to close
    document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
        backdrop.addEventListener('click', (e) => {
            if (e.target === backdrop) {
                hideModals();
            }
        });
    });

    // Close buttons (.btn-modal-close)
    document.querySelectorAll('.btn-modal-close').forEach(btn => {
        btn.addEventListener('click', hideModals);
    });

    // Escape to close modals
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') hideModals();
    });
}

let spStatusPollInterval = null;

function startSelfplayStatusPolling() {
    fetchSelfplayStatus();
    clearInterval(spStatusPollInterval);
    spStatusPollInterval = setInterval(fetchSelfplayStatus, 2000);
}

function stopSelfplayStatusPolling() {
    clearInterval(spStatusPollInterval);
}

async function fetchSelfplayStatus() {
    try {
        const data = await apiRequest('/api/selfplay_status');
        const statIter = document.getElementById('sp-stat-iteration');
        const statGames = document.getElementById('sp-stat-games');
        const statExamples = document.getElementById('sp-stat-examples');
        const statModels = document.getElementById('sp-stat-models');

        if (statIter) statIter.textContent = data.current_iteration || (state.iterationsData ? state.iterationsData.length : 0);
        if (statGames) statGames.textContent = data.games_played || 0;
        if (statExamples) statExamples.textContent = (data.total_examples || 0).toLocaleString();
        if (statModels) statModels.textContent = data.models_trained || 0;

        const progressLbl = document.getElementById('sp-progress-label');
        const progressFill = document.getElementById('sp-progress-fill');
        if (progressLbl) progressLbl.textContent = data.progress || (data.is_active ? 'Training in progress...' : 'Pipeline Ready / Idle');
        if (progressFill) progressFill.style.width = data.is_active ? '65%' : '100%';

        const logTerm = document.getElementById('sp-log-terminal');
        if (logTerm && data.log && data.log.length > 0) {
            logTerm.innerHTML = data.log.join('\n');
            logTerm.scrollTop = logTerm.scrollHeight;
        }

        const modelsGrid = document.getElementById('sp-models-grid');
        if (modelsGrid && data.model_files) {
            modelsGrid.innerHTML = '';
            data.model_files.forEach(m => {
                const item = document.createElement('div');
                item.className = 'sp-model-item';
                item.style.cssText = 'background:rgba(255,255,255,0.04);padding:8px 12px;border-radius:8px;border:1px solid var(--border-color);font-size:0.8rem;';
                item.innerHTML = `<strong>${m.filename}</strong><div style="font-size:0.75rem;color:var(--text-muted);">Iter ${m.iteration} • ${m.size_kb} KB</div>`;
                modelsGrid.appendChild(item);
            });
        }
    } catch (e) {
        console.warn('Could not fetch selfplay status:', e);
    }
}

// ---------------------------------------------------------------------------
// Interactive Self-Play Controller
// ---------------------------------------------------------------------------

let selfplayState = {
    isRunning: false,
    isPaused: false,
    timer: null,
    speedMs: 800
};

function populateSelfPlayModels() {
    const spModelBlack = document.getElementById('sp-model-black');
    const spModelWhite = document.getElementById('sp-model-white');

    if (!spModelBlack || !spModelWhite) return;

    let optionsHtml = '<option value="latest_model.pt">latest_model.pt (Current Best)</option>';
    optionsHtml += '<option value="untrained">Untrained / Random Network</option>';

    if (state.iterationsData && state.iterationsData.length > 0) {
        state.iterationsData.forEach(item => {
            optionsHtml += `<option value="${item.filename}">${item.filename} (Iter ${item.iteration})</option>`;
        });
    }

    spModelBlack.innerHTML = optionsHtml;
    spModelWhite.innerHTML = optionsHtml;

    if (state.iterationsData.length > 0) {
        spModelBlack.value = state.iterationsData[0].filename;
        spModelWhite.value = state.iterationsData[state.iterationsData.length - 1].filename;
    }
}

async function startSelfPlayMatch() {
    const spModelBlack = document.getElementById('sp-model-black').value;
    const spModelWhite = document.getElementById('sp-model-white').value;
    const spSimsBlack = parseInt(document.getElementById('sp-sims-black').value) || 25;
    const spSimsWhite = parseInt(document.getElementById('sp-sims-white').value) || 25;
    const spTempBlack = parseFloat(document.getElementById('sp-temp-black').value) || 1.0;
    const spTempWhite = parseFloat(document.getElementById('sp-temp-white').value) || 1.0;
    const spBoardSize = parseInt(document.getElementById('sp-board-size').value) || 12;
    const spTempThreshold = parseInt(document.getElementById('sp-temp-threshold').value) || 30;
    const spPlaySpeed = parseInt(document.getElementById('sp-play-speed').value) || 800;

    selfplayState.speedMs = spPlaySpeed;
    selfplayState.isRunning = true;
    selfplayState.isPaused = false;

    const btnStart = document.getElementById('btn-sp-start');
    const btnPause = document.getElementById('btn-sp-pause');
    const btnStep = document.getElementById('btn-sp-step');
    const btnStop = document.getElementById('btn-sp-stop');

    if (btnStart) btnStart.classList.add('hidden');
    if (btnPause) {
        btnPause.classList.remove('hidden');
        btnPause.innerHTML = '<span class="icon">⏸</span> Pause';
    }
    if (btnStep) btnStep.classList.remove('hidden');
    if (btnStop) btnStop.classList.remove('hidden');

    try {
        const payload = {
            board_size: spBoardSize,
            model_black: spModelBlack,
            sims_black: spSimsBlack,
            temp_black: spTempBlack,
            model_white: spModelWhite,
            sims_white: spSimsWhite,
            temp_white: spTempWhite,
            temp_threshold: spTempThreshold
        };

        const res = await apiRequest('/api/selfplay/new_game', 'POST', payload);
        updateGameState(res);

        if (spPlaySpeed > 0) {
            runSelfPlayLoop();
        }
    } catch (e) {
        alert(`Failed to start self-play: ${e.message}`);
        stopSelfPlayMatch();
    }
}

function runSelfPlayLoop() {
    clearTimeout(selfplayState.timer);
    if (!selfplayState.isRunning || selfplayState.isPaused) return;

    selfplayState.timer = setTimeout(async () => {
        if (!selfplayState.isRunning || selfplayState.isPaused) return;
        const gameOver = await stepSelfPlayMatch();
        if (!gameOver && selfplayState.speedMs > 0) {
            runSelfPlayLoop();
        }
    }, selfplayState.speedMs);
}

async function stepSelfPlayMatch() {
    try {
        setAiThinking(true);
        const res = await apiRequest('/api/selfplay/step', 'POST');
        soundEngine.playStoneClick();
        updateGameState(res);

        if (res.game_over) {
            stopSelfPlayMatch();
            return true;
        }
        return false;
    } catch (e) {
        console.error("Self-play step failed:", e);
        stopSelfPlayMatch();
        return true;
    } finally {
        setAiThinking(false);
    }
}

function togglePauseSelfPlay() {
    const btnPause = document.getElementById('btn-sp-pause');
    selfplayState.isPaused = !selfplayState.isPaused;

    if (selfplayState.isPaused) {
        clearTimeout(selfplayState.timer);
        if (btnPause) btnPause.innerHTML = '<span class="icon">▶</span> Resume';
    } else {
        if (btnPause) btnPause.innerHTML = '<span class="icon">⏸</span> Pause';
        runSelfPlayLoop();
    }
}

function stopSelfPlayMatch() {
    selfplayState.isRunning = false;
    selfplayState.isPaused = false;
    clearTimeout(selfplayState.timer);

    const btnStart = document.getElementById('btn-sp-start');
    const btnPause = document.getElementById('btn-sp-pause');
    const btnStep = document.getElementById('btn-sp-step');
    const btnStop = document.getElementById('btn-sp-stop');

    if (btnStart) btnStart.classList.remove('hidden');
    if (btnPause) btnPause.classList.add('hidden');
    if (btnStep) btnStep.classList.add('hidden');
    if (btnStop) btnStop.classList.add('hidden');
}

async function saveSelfPlayDataset() {
    try {
        const res = await apiRequest('/api/selfplay/save_data', 'POST');
        alert(`Dataset saved: ${res.filename} (${res.examples_count} training triples)`);
    } catch (e) {
        alert(`Save failed: ${e.message}`);
    }
}

function applySelfPlayPreset(presetType) {
    const spModelBlack = document.getElementById('sp-model-black');
    const spModelWhite = document.getElementById('sp-model-white');
    const spSimsBlack = document.getElementById('sp-sims-black');
    const spSimsWhite = document.getElementById('sp-sims-white');
    const spTempBlack = document.getElementById('sp-temp-black');
    const spTempWhite = document.getElementById('sp-temp-white');
    const spBoardSize = document.getElementById('sp-board-size');

    if (presetType === 'duel') {
        if (state.iterationsData.length > 0) {
            spModelBlack.value = state.iterationsData[0].filename;
            spModelWhite.value = state.iterationsData[state.iterationsData.length - 1].filename;
        }
        spSimsBlack.value = 25;
        spSimsWhite.value = 100;
        spTempBlack.value = 0.5;
        spTempWhite.value = 0.5;
        spBoardSize.value = "12";
    } else if (presetType === 'self') {
        spModelBlack.value = "latest_model.pt";
        spModelWhite.value = "latest_model.pt";
        spSimsBlack.value = 25;
        spSimsWhite.value = 25;
        spTempBlack.value = 1.0;
        spTempWhite.value = 1.0;
        spBoardSize.value = "12";
    } else if (presetType === 'fast') {
        spModelBlack.value = "latest_model.pt";
        spModelWhite.value = "latest_model.pt";
        spSimsBlack.value = 10;
        spSimsWhite.value = 10;
        spTempBlack.value = 1.0;
        spTempWhite.value = 1.0;
        spBoardSize.value = "5";
    }

    document.getElementById('sp-sims-black-val').textContent = spSimsBlack.value;
    document.getElementById('sp-sims-white-val').textContent = spSimsWhite.value;
    document.getElementById('sp-temp-black-val').textContent = spTempBlack.value;
    document.getElementById('sp-temp-white-val').textContent = spTempWhite.value;
}

function setupSelfPlayEventListeners() {
    const btnStart = document.getElementById('btn-sp-start');
    const btnPause = document.getElementById('btn-sp-pause');
    const btnStep = document.getElementById('btn-sp-step');
    const btnStop = document.getElementById('btn-sp-stop');
    const btnSaveData = document.getElementById('btn-sp-save-data');

    const btnPresetDuel = document.getElementById('btn-sp-preset-duel');
    const btnPresetSelf = document.getElementById('btn-sp-preset-self');
    const btnPresetFast = document.getElementById('btn-sp-preset-fast');

    const spSimsBlack = document.getElementById('sp-sims-black');
    const spSimsWhite = document.getElementById('sp-sims-white');
    const spTempBlack = document.getElementById('sp-temp-black');
    const spTempWhite = document.getElementById('sp-temp-white');

    if (btnStart) btnStart.addEventListener('click', startSelfPlayMatch);
    if (btnPause) btnPause.addEventListener('click', togglePauseSelfPlay);
    if (btnStep) btnStep.addEventListener('click', stepSelfPlayMatch);
    if (btnStop) btnStop.addEventListener('click', stopSelfPlayMatch);
    if (btnSaveData) btnSaveData.addEventListener('click', saveSelfPlayDataset);

    if (btnPresetDuel) btnPresetDuel.addEventListener('click', () => applySelfPlayPreset('duel'));
    if (btnPresetSelf) btnPresetSelf.addEventListener('click', () => applySelfPlayPreset('self'));
    if (btnPresetFast) btnPresetFast.addEventListener('click', () => applySelfPlayPreset('fast'));

    if (spSimsBlack) spSimsBlack.addEventListener('input', () => {
        document.getElementById('sp-sims-black-val').textContent = spSimsBlack.value;
    });
    if (spSimsWhite) spSimsWhite.addEventListener('input', () => {
        document.getElementById('sp-sims-white-val').textContent = spSimsWhite.value;
    });
    if (spTempBlack) spTempBlack.addEventListener('input', () => {
        document.getElementById('sp-temp-black-val').textContent = spTempBlack.value;
    });
    if (spTempWhite) spTempWhite.addEventListener('input', () => {
        document.getElementById('sp-temp-white-val').textContent = spTempWhite.value;
    });
}

function setAiThinking(thinking) {
    state.isAiThinking = thinking;
    if (thinking) {
        aiThinkingOverlay.classList.remove('hidden');
    } else {
        aiThinkingOverlay.classList.add('hidden');
    }
}

function capitalize(s) {
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
}

// Launch
window.addEventListener('DOMContentLoaded', init);

