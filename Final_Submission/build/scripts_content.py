"""Team members and the spoken script for each slide of AlphaGoZero_Presentation.pptx.

SCRIPTS maps slide number -> (presenter key, minutes, [paragraphs]).
Speaking pace assumed: about 130 words per minute.
"""

MEMBERS = {
    "shafreed": ("Shaik Shafreed", "2024BCS0113"),
    "chandrakanth": ("Chandrakanth", "2024BCS0125"),
    "arya": ("Arya Vardhan", "2024BCS0129"),
    "humair": ("Shaik Humair Fatah", "2024BCS0105"),
    "harshithan": ("Lakavath Harshithan", "2024BCS0321"),
}

# Order in which members are listed on the cover and title slide
MEMBER_ORDER = ["shafreed", "chandrakanth", "arya", "humair", "harshithan"]

SLIDE_TITLES = {
    1: "Title", 2: "Agenda", 3: "Why Go is hard for AI", 4: "The core idea: a self-improving loop",
    5: "The environment: 12×12 Go", 6: "Algorithm 1: dual-head residual network",
    7: "Algorithm 2: Monte Carlo Tree Search with PUCT", 8: "Algorithm 3: exploration controls",
    9: "Algorithm 4: self-play data generation", 10: "Algorithm 5: training the network",
    11: "Algorithm 6: evaluation and promotion", 12: "Curriculum: four stages of growing difficulty",
    13: "Results: three candidates promoted", 14: "Training loss fell across the curriculum",
    15: "Self-play volume per iteration", 16: "Same code, different speed on Windows and Linux",
    17: "Bugs we found and fixed", 18: "Reproducibility", 19: "References", 20: "Conclusion and future work",
}

SCRIPTS = {
    # ---------------------------------------------------------------- Arya Vardhan
    1: ("arya", 0.5, [
        "Good morning everyone. We are presenting our course project for CSE 311, Artificial Intelligence: a "
        "review and implementation of AlphaGo Zero, from the paper “Mastering the game of Go without human "
        "knowledge” by Silver and colleagues, published in Nature in 2017.",
        "I am Arya Vardhan, and with me are Shaik Shafreed, Chandrakanth, Shaik Humair Fatah and Lakavath "
        "Harshithan.",
    ]),
    2: ("arya", 0.5, [
        "Here is our plan. We start with the problem and why AlphaGo Zero was a breakthrough, then the "
        "self-play loop at the heart of the method.",
        "Next come the six algorithms we implemented: the Go environment, the neural network, Monte Carlo tree "
        "search, the exploration controls, training and evaluation. Then our curriculum and results over ten "
        "iterations, some engineering lessons, and finally our references and conclusions.",
    ]),
    3: ("arya", 0.75, [
        "Why is Go hard for AI? On a full 19 by 19 board there are about 2 times 10 to the power 170 legal "
        "positions, far more than any computer could ever search. Even on our smaller 12 by 12 board, every "
        "move has 145 options: 144 points plus pass.",
        "AlphaGo, in 2016, handled this partly by learning from 30 million positions of human expert games. "
        "AlphaGo Zero's key idea is to use zero human games. It starts from random play and learns only from "
        "games it plays against itself.",
    ]),
    4: ("arya", 0.75, [
        "This slide shows the core idea: a loop that improves itself. First, self-play: the current best "
        "network, together with tree search, plays games against itself. Second, training: the network learns "
        "to predict what the search chose and who won. Third, evaluation: the new candidate plays 40 games "
        "against the current best. Fourth, promotion: if it wins at least 55 percent, it becomes the new best, "
        "and the loop starts again.",
        "A better network makes the search stronger, and a stronger search produces better training data. "
        "Shafreed will now explain how each part works.",
    ]),
    # ---------------------------------------------------------------- Shaik Shafreed
    5: ("shafreed", 1.0, [
        "Thank you, Arya. Everything starts with the game environment, which we wrote from scratch. The board "
        "is a 12 by 12 grid. Stones are captured when their group loses all its liberties. Suicide moves are "
        "illegal unless they capture. For ko we use positional superko: a move may never recreate an earlier "
        "board position, which prevents endless repetition. Passing is legal, consecutive passes end the game, "
        "and the winner is decided by area scoring, without komi. Games are capped at 500 moves.",
        "On the right is how the agent sees the board: three 12 by 12 planes, its own stones, the opponent's "
        "stones and the empty points, always from the point of view of the player to move. There are 145 "
        "possible actions, and the only reward is the final result: plus one for a win, minus one for a loss.",
    ]),
    6: ("shafreed", 1.25, [
        "Algorithm one is the neural network, which we call GoNetwork. The three input planes go through a "
        "3 by 3 convolution with 64 filters, then a tower of 10 residual blocks. Each block has two "
        "convolutions with batch normalisation and a skip connection, which lets a deep network train stably. "
        "That is the ResNet idea from He and colleagues.",
        "The tower then splits into two heads. The policy head gives a probability for each of the 145 moves: "
        "which moves look promising. The value head ends in tanh and gives one number between minus one and "
        "plus one: who is likely to win from here. AlphaGo used two separate networks for this; AlphaGo Zero "
        "uses one. Ours has about 793 thousand parameters, roughly 3.2 megabytes per saved model.",
    ]),
    7: ("shafreed", 1.5, [
        "Algorithm two is Monte Carlo tree search, which turns the network into a strong player. For every "
        "move we run many simulations.",
        "In each simulation we first select: starting at the root, we follow the move with the highest PUCT "
        "score, shown at the top. Q is how good the move has been so far, which is exploitation. The second "
        "term is exploration: it is large when the network's prior P is high but the move has few visits N. "
        "c-puct, set to 1.5, balances the two.",
        "When we reach a leaf, we expand and evaluate it with a single network call. There are no random "
        "rollouts, unlike classical MCTS. Then we back up the value along the path, flipping its sign at every "
        "level because the players alternate. After all simulations we choose the move from the visit counts. "
        "The number of simulations grows from 35 to 100 per move across our curriculum.",
    ]),
    8: ("shafreed", 1.0, [
        "Algorithm three is about exploration. If self-play always picks the same moves, the games repeat and "
        "the network stops learning. We use three controls from the paper.",
        "First, Dirichlet noise: at the root, 25 percent of the prior comes from random noise. Its parameter "
        "alpha is 10 divided by the number of moves, about 0.069 for our board; on a 19 by 19 board the same "
        "rule gives the paper's 0.03. Second, temperature: for the first 30 moves of each game we sample moves "
        "in proportion to their visit counts, and after that we play the most-visited move. Third, the "
        "exploration constant c-puct of 1.5 inside the search.",
        "Now Chandrakanth will explain how we generate the data and train the network.",
    ]),
    # ---------------------------------------------------------------- Chandrakanth
    9: ("chandrakanth", 1.0, [
        "Thanks, Shafreed. Algorithm four is self-play data generation. We run 18 worker processes in parallel "
        "on the CPU, each with its own random seed. Games are played in batches of 25 and saved after every "
        "batch, so if a run stops, and ours did stop a few times, it resumes from the last batch instead of "
        "starting over.",
        "Every position in every game becomes one training example: the board planes, the search's visit "
        "distribution pi, and the final result z. We also use a replay window: each iteration trains on its "
        "own games plus the previous iteration's games. At stage 5 that is 1,000 games and about 222 thousand "
        "new positions per iteration, and iteration 10 trained on almost 450 thousand positions.",
    ]),
    10: ("chandrakanth", 1.0, [
        "Algorithm five is training. The loss at the top has three parts. The first is the cross-entropy "
        "between the search policy pi and the network's policy p, so the network learns to predict what the "
        "search would choose. The second is the squared error between the game result z and the predicted "
        "value v. The third is L2 regularisation, which prevents overfitting.",
        "We train with stochastic gradient descent with momentum 0.9, a learning rate of 0.001, weight decay "
        "of 10 to the minus 4 and mini-batches of 32, on the laptop's RTX 4050 GPU. Each candidate starts from "
        "the current best model's weights and trains for 4 to 8 epochs depending on the stage, with a "
        "checkpoint after every epoch.",
    ]),
    11: ("chandrakanth", 1.0, [
        "Algorithm six decides whether a new network is really better. The candidate plays an arena match "
        "against the current best. Colours alternate, the first 8 moves are sampled so that the games differ, "
        "and games are capped at 500 moves. From iteration 8 onward we play 40 games, and the candidate must "
        "win at least 55 percent to be promoted.",
        "One important point: with 40 games, a measured win rate has a standard error of about 8 percentage "
        "points. So a 50 percent result does not mean the candidate is worse, only that it was not clearly "
        "better.",
    ]),
    12: ("chandrakanth", 0.75, [
        "To make good use of a single laptop, we trained through a curriculum of four stages. Stage 2 plays "
        "400 games per iteration with 35 simulations per move; stage 5 plays 1,000 games with 100 simulations "
        "and 8 epochs. Each stage trains two candidates, so iterations 3 to 10 in total.",
        "Early stages are cheap and fast. Later stages search deeper, which produces stronger training targets "
        "once the network is good enough to use them.",
    ]),
    13: ("chandrakanth", 1.25, [
        "Here are our results. Each bar is one candidate's win rate against the current best model. Three "
        "candidates were promoted: iteration 4 with 70 percent, iteration 7 with 55 percent, and iteration 9 "
        "with 72.5 percent, winning 29 games to 11 against iteration 7.",
        "Iteration 9 is our final best model, and that result is statistically clear: about 2.8 standard "
        "errors above an even match. The other five candidates were rejected, but most of them scored between "
        "45 and 52.5 percent, which is essentially a tie with the model they played, not a regression. This "
        "shows that the self-play loop really does produce stronger networks without any human data.",
    ]),
    14: ("chandrakanth", 1.0, [
        "This chart shows the training loss at the end of each iteration. It fell from 4.83 at iteration 3 to "
        "3.27 at iteration 9. Almost all of it is policy loss. For comparison, guessing uniformly over 145 "
        "moves would score about 4.98, so the network has learned clear move preferences. The value loss "
        "settled near 0.03.",
        "The curve flattens after iteration 6, which matches the near-50-percent evaluation results: the easy "
        "gains have been made, and further progress needs more games or deeper search. Now Humair will talk "
        "about the cost of running all this.",
    ]),
    # ---------------------------------------------------------------- Shaik Humair Fatah
    15: ("humair", 0.75, [
        "Thank you, Chandrakanth. This table shows how much self-play each iteration needed. The number of "
        "games grew from 400 to 1,000 per iteration, and so did the time: from about half an hour at "
        "iteration 3 to more than six and a half hours at iteration 9.",
        "In total, iterations 3 to 10 played 5,700 games and produced about 1.23 million training positions, "
        "which took roughly 24 and a half hours of self-play.",
    ]),
    16: ("humair", 1.0, [
        "We noticed something interesting when we moved from Windows to Linux on the same laptop. With exactly "
        "the same code and the same 18 workers, a batch of 25 self-play games took 10 minutes on Windows but "
        "only about 7 minutes on Linux, around 30 percent faster. Training differed even more: about 136 "
        "seconds per epoch on Windows against 78 seconds on Linux.",
        "We did not profile the exact cause, but slower process creation on Windows is a likely factor. We "
        "also monitored the GPU during evaluation and found it at 0 percent, because the search evaluates one "
        "position at a time on the CPU. Shafreed will now explain the bugs we found along the way.",
    ]),
    # ---------------------------------------------------------------- Shaik Shafreed
    17: ("shafreed", 1.25, [
        "Implementing the paper ourselves exposed several bugs that never crashed the program but silently "
        "stopped it from learning. The most serious was the sign in the MCTS backup. Nodes stored the value "
        "from the wrong player's point of view, so the search actually picked the opponent's best move. We "
        "only noticed when a well-trained candidate lost 0 to 20. After the fix, the same model scored 9 to 11.",
        "Second, the network was trained with an empty-points input plane but played with a different plane. "
        "Third, there were smaller issues: a double softmax in the loss, evaluation games that were all "
        "identical, a silent fallback to random weights, and missing Dirichlet noise. We regenerated all the "
        "data after these fixes, and a regression test now protects the MCTS fix.",
        "Harshithan will take us through reproducibility and our conclusions.",
    ]),
    # ---------------------------------------------------------------- Lakavath Harshithan
    18: ("harshithan", 0.75, [
        "Thank you, Shafreed. All our code is on GitHub, in the repository chandrakanth-source slash "
        "ALPHAGo-Zero, on the shafreed branch. The trained models are stored with Git LFS, and the full "
        "results are in Final_Training_Report.md.",
        "To run it, you pull the models, install the requirements and run train_batched.py with a stage "
        "number. Because self-play is saved in batches, a stopped run can be restarted with the same command. "
        "The project also has a test suite of 99 passing unit tests.",
    ]),
    19: ("harshithan", 0.5, [
        "These are the main references we used. The core paper is Silver and colleagues, 2017, in Nature. We "
        "also studied the original AlphaGo paper from 2016 and AlphaZero from 2018, the ResNet paper by He and "
        "colleagues, and the two foundational Monte Carlo tree search papers, by Kocsis and Szepesvári and by "
        "Coulom.",
    ]),
    20: ("harshithan", 0.75, [
        "To conclude: we implemented the complete AlphaGo Zero loop, with Go rules, a dual-head residual "
        "network, PUCT tree search, self-play, training and arena evaluation, and trained it from scratch on a "
        "single laptop. Over ten iterations three models were promoted, and iteration 9 is our best, winning "
        "72.5 percent against its predecessor.",
        "In the future we would like to use longer evaluations or an Elo ladder, compare against an external "
        "program, use larger networks and boards, and show live self-play in our web app. Thank you for "
        "listening. We are happy to take your questions.",
    ]),
}
