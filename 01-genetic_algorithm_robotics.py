import random
import math
import time
import matplotlib.pyplot as plt

N = 10
START, GOAL = 0, N - 1

OBSTACLES = {(2,2), (2,3), (2,4), (3,4), (4,4),
             (5,6), (6,6), (7,6), (7,7)}

RMIN = 1
POP_SIZE = 100
GENERATIONS = 150
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.3
RESTART_LIMIT = 10

def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def obstacle_distance(p):
    if p in OBSTACLES:
        return 0
    return min(distance(p, o) for o in OBSTACLES)

def fitness(path):
    length = sum(distance(a, b) for a, b in zip(path, path[1:]))
    safety_penalty = sum(
        max(0, RMIN - obstacle_distance(p)) * 1000
        for p in path
    )
    return length + safety_penalty

def create_path():
    path = [(0, 0)]
    x, y = 0, 0

    while (x, y) != (N - 1, N - 1):
        choices = []

        if x < N - 1:
            choices.append((x + 1, y))
        if y < N - 1:
            choices.append((x, y + 1))
        if x > 0:
            choices.append((x - 1, y))
        if y > 0:
            choices.append((x, y - 1))

        choices = [p for p in choices
                   if p not in OBSTACLES]

        if not choices:
            return create_path()

        p = random.choice(choices)
        path.append(p)
        x, y = p

    return path

def crossover(a, b):
    if random.random() > CROSSOVER_RATE:
        return a[:]

    n = min(len(a), len(b))
    point = random.randint(1, n - 2)

    child = a[:point] + b[point:]

    if child[-1] != GOAL_POINT:
        child.append(GOAL_POINT)

    return child

def mutate(path):
    path = path[:]

    for i in range(1, len(path) - 1):
        if random.random() < MUTATION_RATE:
            x, y = path[i]

            neighbours = [
                (x+1,y), (x-1,y),
                (x,y+1), (x,y-1)
            ]

            neighbours = [
                p for p in neighbours
                if 0 <= p[0] < N
                and 0 <= p[1] < N
                and p not in OBSTACLES
            ]

            if neighbours:
                path[i] = random.choice(neighbours)

    return path

GOAL_POINT = (N - 1, N - 1)

def run_ga():
    start_time = time.time()

    population = [create_path() for _ in range(POP_SIZE)]
    best = min(population, key=fitness)
    history = []
    stagnation = 0

    for generation in range(GENERATIONS):

        population.sort(key=fitness)

        if fitness(population[0]) < fitness(best):
            best = population[0][:]
            stagnation = 0
        else:
            stagnation += 1

        history.append(fitness(best))

        if stagnation >= RESTART_LIMIT:
            population = [create_path() for _ in range(POP_SIZE)]
            stagnation = 0
            continue

        parents = population[:20]
        new_population = parents[:]

        while len(new_population) < POP_SIZE:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    elapsed = time.time() - start_time

    return best, elapsed, history

best, elapsed, history = run_ga()

L = fitness(best)
P = POP_SIZE
G = GENERATIONS
C = int(P * CROSSOVER_RATE)
M = int(P * MUTATION_RATE)
T = elapsed

Q = (0.6 * (1 / L) + 0.1 * P +
     0.1 * G + 0.1 * C + 0.1 * M) / T

print("Best Path:", best)
print("Path Length:", L)
print("Population:", P)
print("Generations:", G)
print("Crossover Chromosomes:", C)
print("Mutation Chromosomes:", M)
print("Execution Time:", round(T, 4), "seconds")
print("Quality Function Q:", round(Q, 4))

x, y = zip(*best)

plt.figure(figsize=(7, 7))

for ox, oy in OBSTACLES:
    plt.scatter(ox, oy, c="black", s=250, marker="s")

plt.plot(x, y, "b-o")
plt.scatter(*best[0], c="green", s=150, label="Start")
plt.scatter(*best[-1], c="red", s=150, label="Goal")

plt.grid()
plt.legend()
plt.title("GA-Based Mobile Robot Path Planning")
plt.show()

plt.plot(history)
plt.xlabel("Generation")
plt.ylabel("Path Cost")
plt.title("GA Convergence")
plt.grid()
plt.show()
