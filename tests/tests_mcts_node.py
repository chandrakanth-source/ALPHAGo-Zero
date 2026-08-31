from mcts.node import MCTSNode


def test_node_creation():

    state = "initial"

    node = MCTSNode(state)

    assert node.state == "initial"
    assert node.parent is None
    assert node.move is None
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert len(node.children) == 0
def test_add_child():

    root = MCTSNode("root")

    child = root.add_child(
        move=57,
        state="after_move"
    )

    assert child.parent == root
    assert child.move == 57
    assert child.state == "after_move"

    assert 57 in root.children
    assert root.children[57] == child
def test_value():

    node = MCTSNode("state")

    assert node.value() == 0.0

    node.visit_count = 4
    node.value_sum = 2.0

    assert node.value() == 0.5