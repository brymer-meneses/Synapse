from synapse import Tensor

def showParents(tensor: Tensor, allNodes = []) -> None:
    for node in tensor.parentNodes:
        allNodes.append(node)

    if tensor.parentNodes == []:
        for member in reversed(allNodes):
            print(member)
    else:
        showParents(node.tensor)
