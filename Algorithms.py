import collections
import heapq


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
