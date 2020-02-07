import pygame


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
        else:
            pass

class Parser:
    def __init__(self):
        self.squareDistance = Graphics.squareSize + 1

    def parse(self, fileName, screen):
        file = open(fileName, "r+")

        row = file.readline()
        x = 0
        y = 0
        while row != "":
            for symbol in row:
                Graphics.drawSquare(Graphics(), x, y, symbol, screen)
                x += self.squareDistance
            x = 0
            y += self.squareDistance
            row = file.readline()

        file.close()

class GraphNode:
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

class GraphEdge:
    def __init__(self, iFrom, iTo, cost):
        self.iFrom = iFrom
        self.iTo = iTo
        self.cost = cost

class SparseGraph:
    nodes = []
    edges = []

    def __init__(self, nextNodeIndex):
        self.nextNodeIndex = 0
        self.squareDistance = Graphics.squareSize + 1

    def load(self, fileName):
        file = open(fileName, "r+")

        row = file.readline()
        x = self.squareDistance/2
        y = self.squareDistance/2
        while row != "":
            for symbol in row:
                # create node
                if symbol != "\n":
                    node = GraphNode(x, y, self.nextNodeIndex)

                    # add node to nodes
                    self.nodes.append(node)
                    self.edges.append([])

                    x += self.squareDistance
                    self.nextNodeIndex += 1
            x = self.squareDistance/2
            y += self.squareDistance
            row = file.readline()

        file.close()
        file = open(fileName, "r+")

        row = file.readline()


# ---------------main start------------------------
pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Path Finder")

# map parser
parser = Parser()
mapName = "Map1.txt"  # edit this manually here to change map
parseGraph = SparseGraph(0)
parseGraph.load(mapName)

# -------------game loop start---------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # drawing-------------------------
    screen.fill((255, 255, 255))
    parser.parse(mapName, screen)

    pygame.display.update()
    # --------------------------------
