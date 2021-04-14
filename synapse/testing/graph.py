from synapse import Tensor

def showParents(tensor: Tensor, all_nodes = []) -> None:
    for node in tensor._parent_nodes:
        all_nodes.append(node)

    if tensor._parent_nodes == []:
        for member in reversed(all_nodes):
            print(member)
    else:
        showParents(node.tensor)
