
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# 1. PROBLEM DEFINITION: SMART ENERGY MANAGEMENT
# ============================================================

np.random.seed(42)

HOURS = 24
NUM_NESTS = 30
MAX_ITERATIONS = 300
PA = 0.25
ALPHA = 1.0

# Objective weights (must sum to 1)
W_PEAK = 0.35
W_COST = 0.45
W_COMFORT = 0.20

assert abs(W_PEAK + W_COST + W_COMFORT - 1.0) < 1e-9

# ============================================================
# 2. TIME-OF-USE ELECTRICITY PRICES
# Illustrative currency units per kWh
# ============================================================

price = np.full(HOURS, 6.0)

# Off-peak hours
price[0:6] = 3.0
price[22:24] = 3.0

# Shoulder hours
price[6:17] = 6.0

# Peak-price hours
price[17:22] = 10.0

# ============================================================
# 3. APPLIANCE INFORMATION
# Power: kW
# Duration: hours
# earliest: earliest permitted start hour
# latest: latest permitted start hour
# preferred: preferred start hour
# ============================================================

appliances = [
    {
        "name": "Washing machine",
        "power": 0.8,
        "duration": 2,
        "earliest": 6,
        "latest": 18,
        "preferred": 9
    },
    {
        "name": "Dishwasher",
        "power": 1.2,
        "duration": 2,
        "earliest": 8,
        "latest": 21,
        "preferred": 20
    },
    {
        "name": "Water heater",
        "power": 2.0,
        "duration": 2,
        "earliest": 5,
        "latest": 20,
        "preferred": 7
    },
    {
        "name": "Electric vehicle",
        "power": 3.5,
        "duration": 4,
        "earliest": 0,
        "latest": 18,
        "preferred": 18
    },
    {
        "name": "Air conditioner",
        "power": 1.5,
        "duration": 5,
        "earliest": 12,
        "latest": 18,
        "preferred": 15
    }
]

N = len(appliances)

# Permitted start times for each appliance
ALLOWED = [
    np.arange(
        a["earliest"],
        min(a["latest"], HOURS - a["duration"]) + 1
    )
    for a in appliances
]

# ============================================================
# 4. NORMALIZATION CONSTANTS
# ============================================================

TOTAL_ENERGY = sum(
    a["power"] * a["duration"]
    for a in appliances
)

MAX_PEAK_REFERENCE = sum(
    a["power"] for a in appliances
)

MAX_COST_REFERENCE = TOTAL_ENERGY * np.max(price)

# ============================================================
# 5. SOLAR PV GENERATION PROFILE
# Set pv to zeros to disable solar generation.
# Values represent available solar power in kW.
# ============================================================

pv = np.maximum(
    0.0,
    3.0 * np.sin(
        np.pi * (np.arange(HOURS) - 6) / 12
    )
)

# ============================================================
# 6. REPAIR SOLUTION
# Enforce valid appliance operating windows.
# ============================================================

def repair_solution(solution):
    repaired = []

    for i, appliance in enumerate(appliances):
        choices = ALLOWED[i]
        proposed = int(round(solution[i]))

        closest_index = np.argmin(
            np.abs(choices - proposed)
        )

        repaired.append(int(choices[closest_index]))

    return np.array(repaired, dtype=int)


# ============================================================
# 7. CALCULATE HOURLY APPLIANCE LOAD
# ============================================================

def calculate_load(solution):
    solution = repair_solution(solution)
    load = np.zeros(HOURS)

    for i, appliance in enumerate(appliances):
        start = solution[i]
        end = start + appliance["duration"]

        load[start:end] += appliance["power"]

    return load


# ============================================================
# 8. FITNESS FUNCTION
# Objectives:
# f1 = peak demand minimization
# f2 = electricity cost minimization
# f3 = consumer inconvenience minimization
# ============================================================

def evaluate(solution):
    solution = repair_solution(solution)

    appliance_load = calculate_load(solution)

    # Grid demand after solar generation
    grid_load = np.maximum(appliance_load - pv, 0.0)

    # Objective 1: minimize peak grid demand
    peak = float(np.max(grid_load))
    f_peak = peak / MAX_PEAK_REFERENCE

    # Objective 2: minimize electricity cost
    cost = float(np.sum(grid_load * price))
    f_cost = cost / MAX_COST_REFERENCE

    # Objective 3: minimize deviation from preferred times
    deviations = []
    max_deviations = []

    for i, appliance in enumerate(appliances):
        preferred = appliance["preferred"]

        deviations.append(
            abs(solution[i] - preferred)
        )

        max_deviations.append(
            max(
                abs(ALLOWED[i][0] - preferred),
                abs(ALLOWED[i][-1] - preferred)
            )
        )

    denominator = max(sum(max_deviations), 1)
    inconvenience = sum(deviations) / denominator

    # Combined weighted objective
    fitness = (
        W_PEAK * f_peak
        + W_COST * f_cost
        + W_COMFORT * inconvenience
    )

    return (
        float(fitness),
        peak,
        cost,
        float(inconvenience),
        appliance_load,
        grid_load
    )


# ============================================================
# 9. LEVY FLIGHT
# FIXED: use Python math module instead of np.math
# ============================================================

def levy_flight(size, beta=1.5):

    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)

    step = u / (
        np.abs(v) ** (1 / beta) + 1e-12
    )

    return step


# ============================================================
# 10. INITIALIZE CUCKOO NESTS
# ============================================================

nests = np.array([
    [
        np.random.choice(ALLOWED[i])
        for i in range(N)
    ]
    for _ in range(NUM_NESTS)
])

fitness = np.array([
    evaluate(nest)[0]
    for nest in nests
])

best_index = np.argmin(fitness)
best_solution = nests[best_index].copy()
best_fitness = fitness[best_index]

convergence = []

# ============================================================
# 11. CUCKOO SEARCH ALGORITHM
# ============================================================

for iteration in range(MAX_ITERATIONS):

    # Generate new solutions using Levy flights
    for i in range(NUM_NESTS):

        step = levy_flight(N)

        candidate = nests[i] + ALPHA * step * (
            nests[i] - best_solution
        )

        # Occasional random exploration
        if np.random.rand() < 0.10:
            j = np.random.randint(N)
            candidate[j] = np.random.choice(ALLOWED[j])

        candidate = repair_solution(candidate)
        candidate_fitness = evaluate(candidate)[0]

        # Compare with a randomly selected nest
        j = np.random.randint(NUM_NESTS)

        if candidate_fitness < fitness[j]:
            nests[j] = candidate.copy()
            fitness[j] = candidate_fitness

    # Abandon and replace a fraction of worst nests
    abandon_count = max(
        1,
        int(PA * NUM_NESTS)
    )

    worst_indices = np.argsort(fitness)[-abandon_count:]

    for idx in worst_indices:
        nests[idx] = np.array([
            np.random.choice(ALLOWED[i])
            for i in range(N)
        ])

        fitness[idx] = evaluate(nests[idx])[0]

    # Update global best solution
    current_best_index = np.argmin(fitness)

    if fitness[current_best_index] < best_fitness:
        best_fitness = fitness[current_best_index]
        best_solution = nests[current_best_index].copy()

    convergence.append(best_fitness)

    if (iteration + 1) % 50 == 0:
        print(
            f"Iteration {iteration + 1:3d} | "
            f"Best fitness = {best_fitness:.6f}"
        )


# ============================================================
# 12. BASELINE: PREFERRED APPLIANCE SCHEDULE
# ============================================================

baseline = np.array([
    appliance["preferred"]
    for appliance in appliances
])

base_result = evaluate(baseline)
best_result = evaluate(best_solution)


# ============================================================
# 13. DISPLAY OPTIMIZED SCHEDULE
# ============================================================

print("\n" + "=" * 55)
print("CUCKOO SEARCH: OPTIMIZED ENERGY SCHEDULE")
print("=" * 55)

for i, appliance in enumerate(appliances):

    start = best_solution[i]
    end = start + appliance["duration"]

    print(
        f"{appliance['name']:20s} | "
        f"Start: {start:02d}:00 | "
        f"End: {end:02d}:00 | "
        f"Power: {appliance['power']:.1f} kW"
    )

print("\n" + "=" * 55)
print("PERFORMANCE COMPARISON")
print("=" * 55)

print(
    f"Baseline peak demand : {base_result[1]:.2f} kW"
)

print(
    f"Optimized peak demand: {best_result[1]:.2f} kW"
)

print(
    f"\nBaseline electricity cost : {base_result[2]:.2f}"
)

print(
    f"Optimized electricity cost: {best_result[2]:.2f}"
)

print(
    f"\nBaseline inconvenience : {base_result[3]:.4f}"
)

print(
    f"Optimized inconvenience: {best_result[3]:.4f}"
)


# ============================================================
# 14. CALCULATE PERCENTAGE REDUCTION
# ============================================================

def percentage_reduction(before, after):

    if before <= 1e-12:
        return 0.0

    return 100.0 * (before - after) / before


peak_reduction = percentage_reduction(
    base_result[1],
    best_result[1]
)

cost_reduction = percentage_reduction(
    base_result[2],
    best_result[2]
)

print(
    f"\nPeak demand reduction: {peak_reduction:.2f}%"
)

print(
    f"Electricity cost reduction: {cost_reduction:.2f}%"
)


# ============================================================
# 15. CONVERGENCE GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, MAX_ITERATIONS + 1),
    convergence,
    color="blue",
    linewidth=2
)

plt.xlabel("Iteration")
plt.ylabel("Best fitness value")
plt.title("Cuckoo Search Convergence Curve")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 16. HOURLY GRID DEMAND COMPARISON
# ============================================================

hours = np.arange(HOURS)

plt.figure(figsize=(12, 5))

plt.plot(
    hours,
    base_result[5],
    marker="o",
    label="Baseline grid demand"
)

plt.plot(
    hours,
    best_result[5],
    marker="s",
    label="Optimized grid demand"
)

plt.plot(
    hours,
    pv,
    linestyle="--",
    label="Solar PV generation"
)

plt.xlabel("Hour of day")
plt.ylabel("Power (kW)")
plt.title("Hourly Grid Demand Before and After Optimization")
plt.xticks(np.arange(0, 24, 2))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 17. ELECTRICITY COST COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["Baseline", "Optimized"],
    [base_result[2], best_result[2]],
    color=["gray", "green"]
)

plt.ylabel("Electricity cost (illustrative units)")
plt.title("Electricity Cost Before and After Optimization")
plt.tight_layout()
plt.show()


# ============================================================
# 18. OBJECTIVE VALUE COMPARISON
# ============================================================

labels = [
    "Peak demand",
    "Electricity cost",
    "Consumer inconvenience"
]

baseline_values = [
    base_result[1] / MAX_PEAK_REFERENCE,
    base_result[2] / MAX_COST_REFERENCE,
    base_result[3]
]

optimized_values = [
    best_result[1] / MAX_PEAK_REFERENCE,
    best_result[2] / MAX_COST_REFERENCE,
    best_result[3]
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    optimized_values,
    width,
    label="Optimized"
)

plt.xticks(x, labels)
plt.ylabel("Normalized objective value")
plt.title("Comparison of Optimization Objectives")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nOptimization completed successfully.")
