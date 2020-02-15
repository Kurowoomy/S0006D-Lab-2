import timeit

def measureAlgorithmAndMap():
    mySetup = '''
    import Algorithms
    
    
    class SparseGraph:
        def __init__(self, nextNodeIndex):
            self.nextNodeIndex = 0
    
        def load(self, fileName):
            graph = Graph()
            file = open(fileName, "r+")
    
            row = file.readline()
            x = 0
            y = 0
            while row != "":
                for symbol in row:
                    # create node
                    if symbol != "\\n":
                        # add node to nodes
                        graph.nodes.append([x, y])
                        if symbol == "X":
                            graph.nonWalkables.append([x, y])
                        elif symbol == "S":
                            graph.startNode.append(x)
                            graph.startNode.append(y)
                        elif symbol == "G":
                            graph.goalNode.append(x)
                            graph.goalNode.append(y)
    
                        x += 1
                        self.nextNodeIndex += 1
                x = 0
                y += 1
                row = file.readline()
    
            file.close()
    
            return graph
    
    
    class Graph:
    nodes = []
    nonWalkables = []
    startNode = []
    goalNode = []

    def __init__(self):
        pass

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
    
    
    mapName = "Map1.txt"  # edit string manually here to change map
    graph = SparseGraph.load(SparseGraph(0), mapName)
    '''
    myCode = '''
    path = Algorithms.AStar(graph, graph.startNode, graph.goalNode)  # change algorithm manually here
    route = Algorithms.getPath(graph.startNode, graph.goalNode, path)
    '''
    num = 10000
    print(timeit.timeit(stmt=myCode, setup=mySetup, number=num)/num)
