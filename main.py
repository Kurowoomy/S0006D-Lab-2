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


# ---------------main start------------------------
pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Path Finder")

# map parser
parser = Parser()
mapName = "Map1.txt"  # edit this manually here to change map

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
