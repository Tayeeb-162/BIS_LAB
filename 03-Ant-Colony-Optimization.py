import itertools
import numpy as np
import matplotlib.pyplot as plt


# 1. Define cities using (x, y) coordinates
CITIES = np.array([
    [0, 0],
    [2, 6],
    [5, 3],
    [6, 8],
    [8, 2],
    [1, 4],
    [7, 6],
    [3, 1]
], dtype=float)

CITY_NAMES = [f"City {i}" for i in range(len(CITIES))]


def calculate_distance_matrix(cities):
    difference = cities[:, None, :] - cities[None, :, :]
    distances = np.sqrt(np.sum(difference ** 2, axis=2))
    np.fill_diagonal(distances, 0.0)
    return distances


class AntColonySystem:
    def __init__(
        self,
        distances,
        n_ants=None,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q0=0.9,
        phi=0.1,
        iterations=300,
        seed=42,
        symmetric=False
    ):
        self.distances = np.asarray(distances, dtype=float)
        self.n = len(self.distances)

        if self.n < 2 or self.distances.shape != (self.n, self.n):
            raise ValueError("Distance matrix must be square with at least 2 cities.")
        if not np.all(np.isfinite(self.distances)):
            raise ValueError("Distances must be finite.")
        if np.any(self.distances < 0) or np.any(np.diag(self.distances) != 0):
            raise ValueError("Distances must be nonnegative with a zero diagonal.")
        if np.any(self.distances[~np.eye(self.n, dtype=bool)] <= 0):
            raise ValueError("Distances between different cities must be positive.")
        if symmetric and not np.allclose(self.distances, self.distances.T):
            raise ValueError("Symmetric TSP requires symmetric distances.")
        if not (0 < rho <= 1 and 0 < phi <= 1):
            raise ValueError("rho and phi must be in (0, 1].")
        if not (0 <= q0 <= 1) or alpha < 0 or beta < 0 or iterations < 1:
            raise ValueError("Invalid algorithm parameters.")

        self.n_ants = n_ants if n_ants is not None else self.n
        if self.n_ants < 1:
            raise ValueError("At least one ant is required.")

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.phi = phi
        self.iterations = iterations
        self.symmetric = symmetric
        self.rng = np.random.default_rng(seed)

        self.heuristic = np.zeros_like(self.distances)
        mask = self.distances > 0
        self.heuristic[mask] = 1.0 / self.distances[mask]

        nearest_tour, nearest_length = self.nearest_neighbor_tour()
        self.tau0 = 1.0 / (self.n * nearest_length)

        self.pheromone = np.full(
            (self.n, self.n), self.tau0, dtype=float
        )
        np.fill_diagonal(self.pheromone, 0.0)

        self.best_tour = nearest_tour
        self.best_length = nearest_length
        self.history = []

    def tour_length(self, tour):
        return sum(
            self.distances[tour[i], tour[(i + 1) % self.n]]
            for i in range(self.n)
        )

    def nearest_neighbor_tour(self):
        best_tour = None
        best_length = float("inf")

        for start in range(self.n):
            tour = [start]
            unvisited = set(range(self.n))
            unvisited.remove(start)

            while unvisited:
                current = tour[-1]
                next_city = min(
                    unvisited,
                    key=lambda city: self.distances[current, city]
                )
                tour.append(next_city)
                unvisited.remove(next_city)

            length = self.tour_length(tour)

            if length < best_length:
                best_tour = tour
                best_length = length

        return best_tour, best_length

    def select_next_city(self, current, unvisited):
        candidates = np.array(sorted(unvisited), dtype=int)

        weights = (
            self.pheromone[current, candidates] ** self.alpha
            * self.heuristic[current, candidates] ** self.beta
        )

        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            probabilities = np.full(len(candidates), 1.0 / len(candidates))
        else:
            probabilities = weights / weights.sum()

        if self.rng.random() <= self.q0:
            return int(candidates[np.argmax(weights)])

        return int(self.rng.choice(candidates, p=probabilities))

    def local_update(self, i, j):
        self.pheromone[i, j] = (
            (1 - self.phi) * self.pheromone[i, j]
            + self.phi * self.tau0
        )

        if self.symmetric:
            self.pheromone[j, i] = self.pheromone[i, j]

    def construct_tour(self):
        current = int(self.rng.integers(self.n))
        tour = [current]
        unvisited = set(range(self.n))
        unvisited.remove(current)

        while unvisited:
            next_city = self.select_next_city(current, unvisited)
            self.local_update(current, next_city)
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        self.local_update(tour[-1], tour[0])
        return tour

    def global_update(self):
        reinforcement = 1.0 / self.best_length

        for i in range(self.n):
            a = self.best_tour[i]
            b = self.best_tour[(i + 1) % self.n]

            self.pheromone[a, b] = (
                (1 - self.rho) * self.pheromone[a, b]
                + self.rho * reinforcement
            )

            if self.symmetric:
                self.pheromone[b, a] = self.pheromone[a, b]

        np.fill_diagonal(self.pheromone, 0.0)

    def solve(self, verbose=True):
        for iteration in range(1, self.iterations + 1):
            iteration_best_tour = None
            iteration_best_length = float("inf")

            for _ in range(self.n_ants):
                tour = self.construct_tour()
                length = self.tour_length(tour)

                if length < iteration_best_length:
                    iteration_best_tour = tour.copy()
                    iteration_best_length = length

            if iteration_best_length < self.best_length:
                self.best_tour = iteration_best_tour.copy()
                self.best_length = iteration_best_length

            self.global_update()
            self.history.append(self.best_length)

            if verbose and (
                iteration == 1
                or iteration % 25 == 0
                or iteration == self.iterations
            ):
                print(
                    f"Iteration {iteration:4d} | "
                    f"Best distance: {self.best_length:.4f}"
                )

        return self.best_tour, self.best_length


def exact_tsp(distances, max_cities=10):
    n = len(distances)

    if n > max_cities:
        raise ValueError(f"Exact solver is limited to {max_cities} cities.")

    best_tour = None
    best_length = float("inf")

    for permutation in itertools.permutations(range(1, n)):
        tour = (0,) + permutation
        length = sum(
            distances[tour[i], tour[(i + 1) % n]]
            for i in range(n)
        )

        if length < best_length:
            best_length = length
            best_tour = list(tour)

    return best_tour, best_length


def plot_results(cities, best_tour, history):
    closed_tour = best_tour + [best_tour[0]]
    route = cities[closed_tour]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        route[:, 0], route[:, 1],
        "-o", color="royalblue", linewidth=1.8
    )

    for i, (x, y) in enumerate(cities):
        axes[0].annotate(
            CITY_NAMES[i], (x, y),
            xytext=(5, 5), textcoords="offset points"
        )

    axes[0].set_title("Best TSP Tour Found by ACS")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history, color="darkorange")
    axes[1].set_title("Convergence of Best Tour Length")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best Distance")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    distances = calculate_distance_matrix(CITIES)

    acs = AntColonySystem(
        distances=distances,
        n_ants=len(CITIES),
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q0=0.9,
        phi=0.1,
        iterations=300,
        seed=42,
        symmetric=True
    )

    best_tour, best_length = acs.solve()

    print("\n========== ACS RESULTS ==========")
    print(
        "Best route: "
        + " -> ".join(CITY_NAMES[i] for i in best_tour)
        + f" -> {CITY_NAMES[best_tour[0]]}"
    )
    print(f"Best tour length: {best_length:.4f}")

    _, baseline_length = acs.nearest_neighbor_tour()
    print(f"Nearest-neighbor tour length: {baseline_length:.4f}")

    exact_tour, exact_length = exact_tsp(distances)
    print(
        "Exact optimal route: "
        + " -> ".join(CITY_NAMES[i] for i in exact_tour)
        + f" -> {CITY_NAMES[exact_tour[0]]}"
    )
    print(f"Exact optimal length: {exact_length:.4f}")

    gap = 100 * (best_length - exact_length) / exact_length
    print(f"ACS optimality gap: {gap:.2f}%")

    plot_results(CITIES, best_tour, acs.history)