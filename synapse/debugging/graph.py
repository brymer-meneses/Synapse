from synapse import Tensor

def show_parents(tensor: Tensor, all_nodes = []) -> None:
    """Shows the nodes of a parent graph"""

    assert isinstance(tensor, Tensor), ValueError(f"Expected \
        Type Tensor got, {type(tensor)}")

    for node in tensor._parent_nodes:
        all_nodes.append(node)

    if tensor._parent_nodes == []:
        for member in reversed(all_nodes):
            print(member)
    else:
        show_parents(node.tensor)
