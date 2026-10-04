import math


def puct_score(
    parent,
    child,
    c_puct=1.5
):

    exploitation = child.value()

    exploration = (
        c_puct
        * child.prior
        * math.sqrt(
            parent.visit_count
        )
        / (
            1
            + child.visit_count
        )
    )

    return (
        exploitation
        + exploration
    )


def select_child(
    node,
    c_puct=1.5
):

    best_move = None
    best_child = None
    best_score = float("-inf")

    for move, child in node.children.items():

        score = puct_score(
            node,
            child,
            c_puct
        )

        if score > best_score:

            best_score = score

            best_move = move

            best_child = child

    return best_move, best_child