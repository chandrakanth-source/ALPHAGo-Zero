"""Plain-language explanations of each slide, plus likely questions and answers.

EXPLANATIONS maps slide number -> dict(simple=[paragraphs], remember=[facts], qa=[(question, answer)]).
"""

EXPLANATIONS = {
    1: dict(
        simple=["The paper we reviewed is DeepMind's 2017 Nature paper on AlphaGo Zero. We did not just read it: "
                "we rebuilt the whole system ourselves and trained it on a smaller 12×12 board on one laptop."],
        remember=["Paper: Silver et al., “Mastering the game of Go without human knowledge,” Nature 550, 2017."],
        qa=[("Why a 12×12 board instead of 19×19?",
             "Compute. The paper used thousands of times more computing power. A 12×12 board keeps every game "
             "short enough to train on a laptop while keeping all the rules and the full algorithm.")],
    ),
    2: dict(
        simple=["The talk goes from the problem, to the idea, to the six algorithms, to the results, and ends with "
                "lessons learned and the conclusion."],
        remember=["Six algorithms: environment, network, MCTS, exploration, training, evaluation."],
        qa=[],
    ),
    3: dict(
        simple=["Go has so many possible positions that no computer can look through them all. Earlier programs, "
                "including the first AlphaGo, needed millions of human expert positions to learn which moves are "
                "good. AlphaGo Zero learns with no human games at all, only the rules."],
        remember=["≈2×10^170 legal positions on 19×19 (more than atoms in the observable universe, ≈10^80).",
                  "145 actions on our board: 144 points + pass.",
                  "AlphaGo (2016) used 30 million human positions; AlphaGo Zero uses none."],
        qa=[("Why can't we just search the whole game tree like in tic-tac-toe?",
             "The tree is astronomically large: about 145 choices per move and games of 150–225 moves. Even "
             "looking 5 moves ahead is about 145^5 ≈ 64 billion positions. We must search selectively, which is "
             "what MCTS does."),
            ("What does “without human knowledge” mean exactly?",
             "No human games and no hand-made features or strategies. The program is given only the rules of Go "
             "(legal moves, captures, scoring) and learns everything else from self-play.")],
    ),
    4: dict(
        simple=["Think of a student who plays against themselves, studies their own games, and then takes a test "
                "against their previous self. If they win clearly, the improved version becomes the new “best”. "
                "Repeating this makes the player stronger each round, with no teacher."],
        remember=["Loop: self-play → train → evaluate (40 games) → promote if ≥ 55%."],
        qa=[("Why does training on its own games make it better instead of just repeating its mistakes?",
             "Because the training target is the result of tree search, not the network's raw guess. The search "
             "looks ahead and finds better moves than the network alone, so the network learns from a stronger "
             "version of itself every time."),
            ("Why the 55% threshold and not 50%?",
             "To avoid promoting a model that only looks better by luck. A candidate must be clearly stronger. "
             "The paper uses the same 55% rule.")],
    ),
    5: dict(
        simple=["We programmed the Go rules ourselves. The agent sees the board as three 12×12 grids of 0s and 1s: "
                "where its own stones are, where the opponent's are, and which points are empty. It always sees the "
                "board from the side of the player whose turn it is, so one network can play both colours."],
        remember=["Positional superko: no move may repeat any earlier board position.",
                  "Area scoring, no komi; games capped at 500 moves.",
                  "Reward only at the end: +1 win, −1 loss, 0 draw."],
        qa=[("What is ko, and why superko?",
             "Ko is a situation where two players could capture back and forth forever. Simple ko only forbids "
             "immediately repeating the previous position; positional superko forbids repeating any earlier "
             "position, which also prevents longer cycles. It is simpler to implement correctly."),
            ("Why is the reward only given at the end? Isn't that hard to learn from?",
             "Yes, that is the credit-assignment problem. AlphaGo Zero handles it with the value head, which "
             "learns to predict the final result from any position, so search gets a useful signal at every move."),
            ("What is komi, and why didn't you use it?",
             "Komi is a points bonus for White to compensate for Black moving first. We kept scoring simple and "
             "alternated colours in evaluation, so neither model benefits from the first-move advantage overall.")],
    ),
    6: dict(
        simple=["The network is the program's intuition. Given a board, it instantly answers two questions: “Which "
                "moves look good?” (the policy, a probability for each of the 145 moves) and “Who is winning?” (the "
                "value, a number from −1 to +1). Residual blocks are layers with shortcuts that let information and "
                "gradients pass through easily, so a deep network still trains well."],
        remember=["Input 3×12×12 → 3×3 conv, 64 filters → 10 residual blocks → policy head (145) + value head (tanh).",
                  "793,144 parameters, about 3.2 MB per model.",
                  "Paper: 20 or 40 blocks with 256 filters, 17 input planes."],
        qa=[("Why one network with two heads instead of two networks?",
             "Both tasks need the same understanding of the board. Sharing the body makes training more efficient "
             "and each task helps the other; the paper reports that the combined network played better."),
            ("What does batch normalisation do?",
             "It rescales each layer's outputs to a stable range during training, which makes deep networks train "
             "faster and more reliably."),
            ("Why tanh in the value head?",
             "tanh squashes any number into the range −1 to +1, which matches the game result: −1 loss, +1 win.")],
    ),
    7: dict(
        simple=["Before each move the program “thinks ahead” by building a small tree of possible continuations. "
                "It does not explore every move equally: it spends most of its time on moves that the network likes "
                "or that have worked well so far, but still tries less-visited moves sometimes. After many "
                "simulations, the move that was explored the most is the one it trusts."],
        remember=["PUCT: Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a)), c_puct = 1.5.",
                  "Steps: select → expand and evaluate (one network call, no rollouts) → back up (flip sign each level).",
                  "35 → 100 simulations per move across stages (paper: 1,600)."],
        qa=[("What is the difference between Q and U in the formula?",
             "Q is exploitation: the average result of this move in earlier simulations. U is exploration: it is "
             "high for moves the network rates highly but that have few visits, and it shrinks as a move gets "
             "visited more."),
            ("Why choose the move by visit count rather than by Q value?",
             "Visit counts are more robust. A move visited many times has a reliable estimate; a move with a high "
             "Q but only one or two visits may just be lucky."),
            ("Why flip the sign during backup?",
             "Players alternate, so a position that is good for one player is bad for the other. Each node must "
             "store the value from the view of the player who made the move into it. Getting this wrong was our "
             "most serious bug (slide 17)."),
            ("How is this different from classical MCTS?",
             "Classical MCTS evaluates a leaf by playing random moves to the end of the game (a rollout). AlphaGo "
             "Zero replaces rollouts with the value head, which is far more accurate and faster.")],
    ),
    8: dict(
        simple=["If the program always plays its favourite move, every self-play game is the same and it never "
                "discovers anything new. So we add controlled randomness: a little random noise on the move "
                "preferences at the start of each search, and random sampling of moves early in each game."],
        remember=["Dirichlet noise at the root: P′ = 0.75·P + 0.25·noise, α = 10/144 ≈ 0.069 (paper: 0.03 on 19×19).",
                  "Temperature: sample by visit counts for the first 30 moves (τ = 1), then play the best move (τ → 0)."],
        qa=[("Why is α different from the paper?",
             "The paper's rule of thumb scales α inversely with the number of legal moves: about 10 divided by the "
             "number of moves. For 19×19 that gives 0.03; for our 144 points it gives about 0.069."),
            ("Is the noise used during evaluation and real games?",
             "No. Noise is only for self-play, to generate diverse training data. Evaluation instead samples the "
             "first 8 moves so games differ, then plays its best moves.")],
    ),
    9: dict(
        simple=["Self-play is the data factory. Eighteen CPU processes play games at the same time. Every position "
                "from every game is saved with two labels: what the search thought was best there, and who "
                "eventually won. Games are saved every 25 so a crash never loses more than one batch."],
        remember=["18 workers, batches of 25 games; example = (board planes, π, z).",
                  "Replay window: current + previous iteration's games.",
                  "Stage 5: 1,000 games ≈ 222,000 positions per iteration; iteration 10 trained on 446,449."],
        qa=[("Why keep the previous iteration's games as well?",
             "More data and more variety, which reduces overfitting to the latest games. The paper keeps the last "
             "500,000 games; we keep the last two iterations because of memory and time limits."),
            ("Why run self-play on the CPU when you have a GPU?",
             "MCTS evaluates one position at a time, which is a poor fit for a GPU. Running many independent games "
             "on many CPU cores gave better throughput. Batching leaf evaluations for the GPU is listed as future "
             "work.")],
    ),
    10: dict(
        simple=["Training teaches the network to imitate the search and to predict the winner. If the search, after "
                "thinking, preferred a move the network did not rate highly, the network is corrected toward the "
                "search's choice. If the network predicted a win but the game was lost, the value is corrected too."],
        remember=["Loss = cross-entropy(π, p) + (z − v)² + c‖θ‖², c = 10⁻⁴ (as weight decay).",
                  "SGD, momentum 0.9, learning rate 0.001, batch 32, 4–8 epochs; warm start from the best model."],
        qa=[("Why cross-entropy for the policy and squared error for the value?",
             "The policy is a probability distribution, and cross-entropy is the standard way to compare two "
             "distributions. The value is a single number, so squared error is natural."),
            ("What is the L2 term for?",
             "It penalises large weights, which reduces overfitting. We apply it through the optimiser's weight "
             "decay setting."),
            ("Why start each candidate from the best model instead of from scratch?",
             "It keeps what has already been learned and makes each iteration much cheaper; the new data only "
             "has to improve an already good network.")],
    ),
    11: dict(
        simple=["Training can make a network worse as well as better, so every candidate must prove itself in a "
                "match against the current champion before it is used to generate new games."],
        remember=["40 games from iteration 8 (20 before); colours alternate; first 8 moves sampled; 500-move cap.",
                  "Promote only if win rate ≥ 55%.",
                  "Standard error with 40 games ≈ 8 percentage points."],
        qa=[("Where does “about 8 percentage points” come from?",
             "For an even match the standard error of a win rate over n games is √(0.5 × 0.5 / n). With n = 40 "
             "that is √(0.00625) ≈ 0.079, so about 8 points."),
            ("Why sample the first 8 moves in evaluation?",
             "Without randomness both models would play the same game every time. We found that our early "
             "20-game evaluations were really only 2 different games repeated."),
            ("AlphaZero dropped this evaluation step. Why keep it?",
             "Gating protects against a bad update ruining the self-play data. It is the AlphaGo Zero method we were "
             "reviewing; comparing it with AlphaZero-style continuous updates is in our future work.")],
    ),
    12: dict(
        simple=["We started with cheap settings and made the task harder as the network improved, like moving a "
                "student from easy to hard exercises. Deeper search only helps once the network is good enough to "
                "guide it well."],
        remember=["Stage 2: 400 games, 35 sims, 4 epochs (iterations 3–4).",
                  "Stage 3: 600, 50, 5 (5–6). Stage 4: 850, 80, 6 (7–8). Stage 5: 1,000, 100, 8 (9–10)."],
        qa=[("Is the curriculum part of the original paper?",
             "No. The paper trains with fixed settings on huge compute. The curriculum is our adaptation to make "
             "good use of a single laptop.")],
    ),
    13: dict(
        simple=["Three of the eight candidates were clearly better and were promoted: iterations 4, 7 and 9. "
                "Iteration 9 is the final model. The rejected ones mostly scored about 50%, which means they were "
                "about as strong as the champion, not weaker."],
        remember=["Promoted: iter 4 (70%), iter 7 (55%), iter 9 (72.5%, 29–11 vs iter 7).",
                  "Rejected: iter 3 (30%), 5 (45%), 6 (50%), 8 (52.5%), 10 (50%).",
                  "Iteration 9's result is ≈2.8 standard errors above 50%."],
        qa=[("How do you know the model actually got better and it is not just luck?",
             "Iteration 9 won 29 of 40 games. If the two models were equal, a result that far from 50% would "
             "happen by chance well under 1% of the time (about 2.8 standard errors)."),
            ("Iteration 7 was promoted with exactly 55% on only 20 games. Is that reliable?",
             "Not very, and we say so. With 20 games the standard error is about 11 points, which is why we "
             "raised evaluations to 40 games from iteration 8. Iteration 9 then beat iteration 7 decisively, which "
             "confirms the overall progress."),
            ("Did you compare against a human or another program?",
             "No; all results are relative to our own earlier models. Testing against an external program such as "
             "GNU Go is our first item of future work.")],
    ),
    14: dict(
        simple=["The loss measures how far the network's predictions are from the targets. It dropped sharply in "
                "the first stages and then levelled off, which matches the results: big early gains, then slower "
                "progress."],
        remember=["Total loss 4.83 (iter 3) → 3.27 (iter 9); policy loss 4.78 → 3.24; value loss ≈ 0.03.",
                  "Uniform guess over 145 moves: ln 145 ≈ 4.98."],
        qa=[("Why doesn't the policy loss go close to zero?",
             "Because the targets are themselves spread out. In the first 30 moves moves are sampled, and a short "
             "search gives several moves similar visit counts. Even a perfect network cannot predict a random "
             "choice exactly."),
            ("Why is the value loss so small compared with the policy loss?",
             "It is a single number per position, and many positions late in a game have an obvious winner. The "
             "policy is a distribution over 145 moves, so its loss is naturally larger.")],
    ),
    15: dict(
        simple=["Self-play is by far the most expensive part. Later iterations play more games with deeper search, "
                "so each iteration took longer: from half an hour to almost seven hours."],
        remember=["Iterations 3–10: 5,700 games, ≈1.23 million positions, ≈24.5 hours of self-play.",
                  "Games grew from ≈150 moves (stage 2) to ≈225 moves later."],
        qa=[("Why did games get longer?",
             "Stronger models play closer, more contested games and stop passing early; deeper search also delays "
             "passing.")],
    ),
    16: dict(
        simple=["We ran the same code on the same laptop under Windows and then Linux, and Linux was clearly "
                "faster. We also measured that the GPU was idle during evaluation, because tree search runs on the "
                "CPU one position at a time."],
        remember=["Self-play batch: 10.0 min (Windows) vs 7.1 min (Linux), ≈30% faster.",
                  "Training epoch: ≈136 s vs ≈78 s.",
                  "GPU utilisation during evaluation: 0%."],
        qa=[("Why is Linux faster?",
             "We did not profile it, so we cannot say for certain. Python starts worker processes more slowly on "
             "Windows, and our pipeline starts many processes, so that is a likely factor. We state it as a likely "
             "cause, not a proven one.")],
    ),
    17: dict(
        simple=["Our first version had bugs that did not crash anything but quietly ruined learning. The worst one "
                "made the search pick the opponent's best move. It stayed hidden until the value network became "
                "accurate, and then a strong candidate suddenly lost every game."],
        remember=["Backup sign bug: 0–20 → 9–11 after the fix.",
                  "Input mismatch: trained on an empty-points plane, played with a black-to-move plane.",
                  "Also: double softmax, identical evaluation games, silent random-weight fallback, missing noise.",
                  "All data was regenerated after the fixes; a regression test guards the MCTS fix."],
        qa=[("Why was the sign bug invisible at first?",
             "With an untrained value head, values are almost random, so choosing the “opponent's best” move is no "
             "worse than random. Once the value head became accurate, the bug made the search actively choose bad "
             "moves."),
            ("What was the double softmax problem?",
             "The network already outputs probabilities, and the loss applied softmax again, which flattens the "
             "distribution and weakens the gradient. We now pass log-probabilities to the loss, which makes it "
             "exact.")],
    ),
    18: dict(
        simple=["Anyone can reproduce our work from the GitHub repository. The trained models are stored with Git "
                "LFS because model files are large, and training can be stopped and resumed safely."],
        remember=["github.com/chandrakanth-source/ALPHAGo-Zero, branch shafreed.",
                  "Run: git lfs pull · pip install -r requirements.txt · python train_batched.py --stage N · pytest.",
                  "99 unit tests pass."],
        qa=[("What is Git LFS?",
             "Large File Storage: Git keeps a small pointer in the repository and stores the large file on a "
             "separate server. It keeps the repository small and avoids GitHub's 100 MB file limit."),
            ("Would rerunning give exactly the same results?",
             "No. Self-play is random, so a rerun plays different games; win rates can differ by several points, "
             "but the overall trend should be the same.")],
    ),
    19: dict(
        simple=["These are the papers the project builds on: the AlphaGo Zero paper itself, its predecessor and "
                "successor, and the papers behind residual networks and Monte Carlo tree search."],
        remember=["Main paper [1]: Silver et al., Nature 550, 354–359, 2017."],
        qa=[("What is the difference between AlphaGo, AlphaGo Zero and AlphaZero?",
             "AlphaGo (2016) learned from human games and used two networks plus rollouts. AlphaGo Zero (2017) "
             "used no human data, one network and no rollouts. AlphaZero (2018) applied the same method to chess "
             "and shogi as well, and replaced the evaluation gate with continuous updates.")],
    ),
    20: dict(
        simple=["We built the complete AlphaGo Zero method and showed that it learns from nothing but self-play, "
                "even at small scale. The main limits are compute and how precisely small improvements can be "
                "measured."],
        remember=["Best model: iteration 9, 72.5% vs iteration 7.",
                  "Future work: longer evaluations or Elo, external baseline, batched GPU search, larger networks "
                  "and boards, live self-play in the web app."],
        qa=[("If you had ten times more compute, what would you change first?",
             "More evaluation games, to measure progress reliably; then more simulations per move and more games "
             "per iteration, which give better training targets."),
            ("How strong is your final model?",
             "We can only say it is clearly stronger than our earlier models. An absolute rating would need games "
             "against an external program or human players, which we list as future work.")],
    ),
}

GLOSSARY = [
    ("Policy (p)", "The network's probability for each possible move: which moves look promising."),
    ("Value (v)", "The network's prediction of the final result from the current position, between −1 and +1."),
    ("MCTS", "Monte Carlo tree search: builds a search tree selectively, focusing on promising moves."),
    ("PUCT", "The rule MCTS uses to pick which move to explore, balancing the prior P, the value Q and visit counts."),
    ("Simulation", "One pass of MCTS: select down the tree, evaluate a leaf with the network, back up the value."),
    ("π (pi)", "The improved policy produced by MCTS: visit counts at the root, normalised."),
    ("z", "The final result of a self-play game from a given player's point of view (+1, −1 or 0)."),
    ("Residual block", "Two convolution layers plus a shortcut that adds the input back to the output."),
    ("Dirichlet noise", "Random noise added to the root move priors during self-play to force exploration."),
    ("Temperature (τ)", "Controls randomness when choosing moves: τ = 1 samples by visit counts, τ → 0 picks the best."),
    ("Epoch", "One full pass over the training data."),
    ("Replay window", "The set of recent games the network is trained on."),
    ("Promotion / gating", "Replacing the best model with a candidate only if it wins at least 55% of the evaluation."),
    ("Superko", "Rule forbidding any move that recreates an earlier board position."),
    ("Area scoring", "Score = own stones on the board + empty points surrounded only by own stones."),
    ("Standard error", "How much a measured win rate is expected to vary by chance; ≈8 points for 40 games."),
]
