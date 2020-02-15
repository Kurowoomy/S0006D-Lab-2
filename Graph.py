

class SparseGraph:
    def __init__(self, nextNodeIndex):
        self.nextNodeIndex = 0
        self.graph = Graph()

    def load(self, fileName):
        file = open(fileName, "r+")

        row = file.readline()
        x = 0
        y = 0
        while row != "":
            for symbol in row:
                # create node
                if symbol != "\n":
                    # add node to nodes
                    self.graph.nodes.append([x, y])
                    if symbol == "X":
                        self.graph.nonWalkables.append([x, y])
                    elif symbol == "S":
                        self.graph.startNode.append(x)
                        self.graph.startNode.append(y)
                    elif symbol == "G":
                        self.graph.goalNode.append(x)
                        self.graph.goalNode.append(y)

                    x += 1
                    self.nextNodeIndex += 1
            x = 0
            y += 1
            row = file.readline()

        file.close()

        return self.graph


class Graph:


    def __init__(self):
        self.nodes = []
        self.nonWalkables = []
        self.startNode = []
        self.goalNode = []

    def neighbours(self, node):
        directions = [[-1, -1], [0, -1], [1, -1],
                      [-1, 0], [1, 0],
                      [-1, 1], [0, 1], [1, 1]]
        corners = [[-1, -1], [1, -1],
                   [-1, 1], [1, 1]]
        result = []
        for pos in directions:
            neighbour = [node[0] + pos[0], node[1] + pos[1]]
            if neighbour in self.nodes and neighbour not in self.nonWalkables:
                if pos in corners and not self.cornerIsReachable(neighbour, node):
                    pass
                else:
                    result.append(neighbour)  # add only walkable nodes to neighbours

        return result

    def cornerIsReachable(self, corner, node):
        if corner[0] is node[0] - 1 and corner[1] is node[1] - 1:  # upper left
            if [node[0], node[1] - 1] in self.nonWalkables or [node[0] - 1, node[1]] in self.nonWalkables:
                return False
        elif corner[0] is node[0] + 1 and corner[1] is node[1] - 1:  # upper right
            if [node[0], node[1] - 1] in self.nonWalkables or [node[0] + 1, node[1]] in self.nonWalkables:
                return False
        elif corner[0] is node[0] - 1 and corner[1] is node[1] + 1:  # lower left
            if [node[0], node[1] + 1] in self.nonWalkables or [node[0] - 1, node[1]] in self.nonWalkables:
                return False
        elif corner[0] is node[0] + 1 and corner[1] is node[1] + 1:  # lower right
            if [node[0], node[1] + 1] in self.nonWalkables or [node[0] + 1, node[1]] in self.nonWalkables:
                return False
        return True
