

def ShowGraph(tensor, allNodes = []):
    for node in tensor.parentNodes:
        allNodes.append(node)

    if tensor.parentNodes == []:
        for member in reversed(allNodes):
            print(member)
    else:
        nodePrint(node.tensor)
    return
