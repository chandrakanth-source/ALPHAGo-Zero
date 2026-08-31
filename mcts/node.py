class MCTSNode:

    def __init__(self, state=None, parent=None, move=None, action=None, prior=0.0):

        self.state = state
        self.parent = parent
        self.move = move if move is not None else action
        self.action = self.move

        self.children = {}

        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):

        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def add_child(self, move, state, prior=0.0):
        child = MCTSNode(
            state=state,
            parent=self,
            move=move,
            prior=prior
        )

        self.children[move] = child

        return child
    def is_expanded(self):
        return len(self.children) > 0