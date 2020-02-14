import collections
import heapq
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset


def breadthFirstSearch(graph, start, goal):
    queue = collections.deque()
    queue.append(start)
    path = {tuple(start): None}

    while len(queue) != 0:
        currentNode = queue.popleft()

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

    return path


def getPath(start, goal, path):
    if len(path) <= 0:
        return []

    node = tuple(goal)
    route = [node]
    while node != tuple(start):
        node = tuple(path[node])
        route.append(node)

    return route


def heuristic(goal, next):
    remaining = abs(abs(goal[0] - next[0]) - abs(goal[1] - next[1]))
    if abs(goal[0] - next[0]) < abs(goal[1] - next[1]):
        return 14 * abs(goal[0] - next[0]) + remaining * 10
    else:
        return 14 * abs(goal[1] - next[1]) + remaining * 10


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class CustomDataset(Dataset):
    def __init__(self):
        pass

    def AStarPath(self):
        pass  # X är kartans alla noder, y är en int(kortaste A*-sträckan man vill få ut)
        # return map #X, alla [][]#,  distance #y#


def neuralNetwork():
    train = datasets.MNIST("", download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

    trainSet = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
    testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    EPOCHS = 3

    for epoch in range(EPOCHS):
        for data in trainSet:
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(loss)

    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for data in testSet:
            X, y = data
            output = net(X.view(-1, 28*28))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))
