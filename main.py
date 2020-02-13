import pygame
import Algorithms
import TimeMeasure


class Graphics:
    squareSize = 15

    def __init__(self):
        pass

    def drawSquare(self, x, y, symbol, screen):
        if symbol == "X":
            pygame.draw.rect(screen, (0, 0, 0), [x, y, self.squareSize, self.squareSize])
        elif symbol == "0":
            pygame.draw.rect(screen, (255, 255, 255), [x, y, self.squareSize, self.squareSize])
        elif symbol == "S":
            pygame.draw.rect(screen, (0, 255, 0), [x, y, self.squareSize, self.squareSize])
        elif symbol == "G":
            pygame.draw.rect(screen, (255, 0, 0), [x, y, self.squareSize, self.squareSize])


class Parser:
    def __init__(self):
        self.squareDistance = Graphics.squareSize + 1
        self.nodePos = {}

    def parse(self, fileName, screen):
        file = open(fileName, "r+")

        row = file.readline()
        x = 0
        y = 0
        posX = 0
        posY = 0
        while row != "":
            for symbol in row:
                Graphics.drawSquare(Graphics(), x, y, symbol, screen)

                if symbol != "\n":  # to access the middle of a square via the position of a node
                    self.nodePos[(posX, posY)] = [x + Graphics.squareSize/2, y + Graphics.squareSize/2]
                    posX += 1

                x += self.squareDistance
            x = 0
            posX = 0
            y += self.squareDistance
            posY += 1
            row = file.readline()

        file.close()

    def drawPath(self, path, goal, screen):
        start_pos = self.nodePos[tuple(goal)]
        for node in path:
            pygame.draw.line(screen, (0, 0, 255), start_pos, self.nodePos[tuple(node)])
            start_pos = self.nodePos[tuple(node)]

    def drawVisited(self, path, screen):
        for node in path:
            pos = self.nodePos[tuple(node)]
            intPos = (int(pos[0]), int(pos[1]))
            pygame.draw.circle(screen, (255, 0, 255), intPos, 2)


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
                if symbol != "\n":
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


# ---------------main start------------------------
pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Path Finder")

# measure search time:
# TimeMeasure.measureAlgorithmAndMap()

# map parsing
parser = Parser()
mapName = "Map1.txt"  # edit string manually here to change map
graph = SparseGraph.load(SparseGraph(0), mapName)

# algorithm
path = Algorithms.AStar(graph, graph.startNode, graph.goalNode)  # change algorithm manually here
route = Algorithms.getPath(graph.startNode, graph.goalNode, path)

# -------------game loop start---------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # drawing-------------------------
    screen.fill((255, 255, 255))
    parser.parse(mapName, screen)

    # draw path
    parser.drawVisited(path, screen)
    parser.drawPath(route, graph.goalNode, screen)

    pygame.display.update()
    # --------------------------------
