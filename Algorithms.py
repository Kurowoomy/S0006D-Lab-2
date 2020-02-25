import collections
import heapq


# import torch
# import torchvision
# from torchvision import transforms, datasets
# import torch.nn
# import torch.nn.functional
# from torch.utils.data import Dataset
# import torch.optim
# import Graph
# import random
# import time
# uncomment all of above to train the neural network


def breadthFirstSearch(graph, start, goal):
    queue = collections.deque()
    queue.append(start)
    path = {tuple(start): None}

    while len(queue) != 0:
        currentNode = queue.popleft()  # treat queue as a stack by using popleft

        if currentNode == goal:
            break

        for neighbour in graph.neighbours(currentNode):
            if tuple(neighbour) not in path:
                queue.append(neighbour)
                path[tuple(neighbour)] = currentNode

    return path


def depthFirstSearch(graph, start, goal):
    queue = collections.deque()
    queue.append(start)
    path = {tuple(start): None}

    while len(queue) != 0:
        currentNode = queue.pop()

        if currentNode == goal:
            break

        for neighbour in graph.neighbours(currentNode):
            if tuple(neighbour) not in path:
                queue.append(neighbour)
                path[tuple(neighbour)] = currentNode

    return path


def AStar(graph, start, goal):
    priorityQ = []
    heapq.heappush(priorityQ, (0, tuple(start)))
    path = {tuple(start): None}
    costSoFar = {tuple(start): 0}

    while len(priorityQ) != 0:
        currentNode = heapq.heappop(priorityQ)[1]

        if currentNode == tuple(goal):
            break

        for neighbour in graph.neighbours(currentNode):
            newCost = costSoFar[currentNode] + heuristic(neighbour, currentNode)
            if (tuple(neighbour) not in costSoFar) or (newCost < costSoFar[tuple(neighbour)]):
                costSoFar[tuple(neighbour)] = newCost
                priority = newCost + heuristic(goal, neighbour)
                path[tuple(neighbour)] = currentNode
                heapq.heappush(priorityQ, (priority, tuple(neighbour)))

    return path, costSoFar[tuple(goal)]


def AStar2(graph, start, goal, distance):
    priorityQ = []
    heapq.heappush(priorityQ, (0, tuple(start)))
    path = {tuple(start): None}
    costSoFar = {tuple(start): 0}

    while len(priorityQ) != 0:
        currentNode = heapq.heappop(priorityQ)[1]

        if currentNode == tuple(goal):
            break

        for neighbour in graph.neighbours(currentNode):
            newCost = costSoFar[currentNode] + heuristic(neighbour, currentNode)
            # eventually skip a few nodes by also checking if the cost has gone past the distance
            if (tuple(neighbour) not in costSoFar) or (distance < costSoFar[tuple(neighbour)]) \
                    or (newCost < costSoFar[tuple(neighbour)]):
                costSoFar[tuple(neighbour)] = newCost
                priority = newCost + heuristic(goal, neighbour)
                path[tuple(neighbour)] = currentNode
                heapq.heappush(priorityQ, (priority, tuple(neighbour)))

    return path


# goes from goal node and follows the path from there to return the final route
def getPath(start, goal, path):
    if len(path) <= 0:
        return []

    node = tuple(goal)
    route = [node]
    while node != tuple(start):
        node = tuple(path[node])
        route.append(node)

    return route


# calculates heuristic cost between two nodes, including diagonal cost
def heuristic(goal, next):
    remaining = abs(abs(goal[0] - next[0]) - abs(goal[1] - next[1]))
    if abs(goal[0] - next[0]) < abs(goal[1] - next[1]):
        return 14 * abs(goal[0] - next[0]) + remaining * 10
    else:
        return 14 * abs(goal[1] - next[1]) + remaining * 10

# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(256, 64)
#         self.fc2 = torch.nn.Linear(64, 64)
#         self.fc3 = torch.nn.Linear(64, 64)
#         self.fc4 = torch.nn.Linear(64, 500)
#
#     def forward(self, x):
#         x = torch.nn.functional.relu(self.fc1(x))
#         x = torch.nn.functional.relu(self.fc2(x))
#         x = torch.nn.functional.relu(self.fc3(x))
#         x = self.fc4(x)
#         return torch.nn.functional.log_softmax(x, dim=1)
#
#
# class CustomDataset(Dataset):
#     def __init__(self, numberofDataPoints):
#         # need a Graph() for my AStar
#         # otherwise use mapt for X
#         self.graph = None
#         self.loadMap("Map2.txt")
#         X, y = self.RMList(numberofDataPoints)
#         self.X = X
#         self.y = y
#
#     def loadMap(self, mapName):
#         # create the graph to operate on inside RandAStar
#         self.graph = Graph.SparseGraph.load(Graph.SparseGraph(0), mapName)
#
#     def RandAStar(self):
#         s_ok = False
#         g_ok = False
#         while not s_ok:
#             x1 = int((random.random()) * 16)
#             y1 = int((random.random()) * 16)
#             if [x1, y1] not in self.graph.nonWalkables:
#                 s_ok = True
#                 self.graph.startNode[0] = x1
#                 self.graph.startNode[1] = y1
#
#         while not g_ok:
#             x2 = int((random.random()) * 16)
#             y2 = int((random.random()) * 16)
#             if [x2, y2] not in self.graph.nonWalkables:
#                 g_ok = True
#                 self.graph.goalNode[0] = x2
#                 self.graph.goalNode[1] = y2
#
#         path, distance = AStar(self.graph, self.graph.startNode, self.graph.goalNode)
#         # vet fortfarande inte om detta är rätt men jag prövar med costSoFar[goal] som distance
#
#         mapt = []
#         for i in range(16):
#             mapt = mapt + [[0]*16]
#
#         mapt[x1][y1] = 2
#         mapt[x2][y2] = 2
#
#         # X är kartans alla noder, y är en int(kortaste A*-sträckan man vill få ut)
#         return mapt, distance
#
#     def RMList(self, count):
#         x = []
#         y = []
#         for i in range(count):
#             a, b = self.RandAStar()
#             x = x + [a]
#             y = y + [b]
#         return x, y
#
#     def __getitem__(self, index):
#         return torch.FloatTensor(self.X[index]), self.y[index]
#
#     def __len__(self):
#         return len(self.y)
#
#
# def neuralNetwork():
#     device = torch.device("cuda:0")
#     start = time.clock()
#
#     test = CustomDataset(200)
#     testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)
#     net = Net().to(device)
#     # net = Net()
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#
#     EPOCHS = 10
#     for datasets in range(5):
#         print("Epoch #", datasets)
#         train = CustomDataset(200)
#         trainSet = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
#         for epoch in range(EPOCHS):
#             for data in trainSet:
#                 X, y = data
#                 X, y = X.to(device), y.to(device)
#                 net.zero_grad()
#                 output = net(X.view(-1, 256))
#                 loss = torch.nn.functional.nll_loss(output, y)
#                 loss.backward()
#                 optimizer.step()
#             print(loss)
#
#     correct = 0
#     total = 0
#     net.eval()
#     with torch.no_grad():
#         for data in testSet:
#             X, y = data
#             X, y = X.to(device), y.to(device)
#             output = net(X.view(-1, 256))
#             for idx, i in enumerate(output):
#                 print(torch.argmax(i), y[idx])
#                 if torch.argmax(i) == y[idx]:
#                     correct += 1
#                 total += 1
#
#     print("Accuracy: ", round(correct / total, 3)*100, "%")
#     print(round(time.clock() - start, 3))
