
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# =====================================================
# 1. INDUSTRIAL PROCESS MODEL
# =====================================================

# G(s) = 1 / (s^2 + 3s + 2)

T_END = 15
N_POINTS = 1000

t = np.linspace(0, T_END, N_POINTS)


# =====================================================
# 2. PID CONTROLLER SIMULATION
# =====================================================

def simulate_pid(Kp, Ki, Kd):

    # Closed-loop transfer function:
    # Gcl(s) = (Kd*s^2 + Kp*s + Ki) /
    #          (s^3 + (3+Kd)*s^2 + (2+Kp)*s + Ki)

    numerator = [Kd, Kp, Ki]
    denominator = [1, 3 + Kd, 2 + Kp, Ki]

    try:
        poles = np.roots(denominator)

        # Reject unstable systems
        if np.any(np.real(poles) >= 0):
            return None

        system = signal.TransferFunction(
            numerator, denominator
        )

        time, output = signal.step(system, T=t)

        if not np.all(np.isfinite(output)):
            return None

        return time, output

    except (ValueError, FloatingPointError):
        return None


# =====================================================
# 3. PERFORMANCE PARAMETERS
# =====================================================

def calculate_performance(Kp, Ki, Kd):

    result = simulate_pid(Kp, Ki, Kd)

    if result is None:
        return None

    time, output = result

    reference = 1.0
    error = reference - output

    # Integral Time Absolute Error
    ITAE = np.trapezoid(time * np.abs(error), time)

    # Maximum overshoot in percentage
    overshoot = max(
        0,
        (np.max(output) - reference) * 100
    )

    # Settling time using 2% criterion
    outside_band = np.where(
        np.abs(error) > 0.02 * reference
    )[0]

    if len(outside_band) == 0:
        settling_time = 0
    elif outside_band[-1] == len(time) - 1:
        settling_time = T_END
    else:
        settling_time = time[outside_band[-1] + 1]

    # Steady-state error
    steady_state_error = abs(reference - output[-1])

    # Weighted fitness function
    fitness = (
        ITAE
        + 0.05 * overshoot
        + 2.0 * settling_time
        + 100.0 * steady_state_error
    )

    return {
        "ITAE": ITAE,
        "Overshoot": overshoot,
        "Settling Time": settling_time,
        "Steady State Error": steady_state_error,
        "Fitness": fitness
    }


# =====================================================
# 4. GREY WOLF OPTIMIZATION (GWO)
# =====================================================

def GWO(objective, search_agents=30, max_iter=100):

    dimensions = 3

    # PID parameter boundaries
    lower_bound = np.array([0.01, 0.01, 0.01])
    upper_bound = np.array([20.0, 20.0, 10.0])

    # Initialize wolves
    wolves = np.random.uniform(
        lower_bound,
        upper_bound,
        (search_agents, dimensions)
    )

    alpha_pos = np.zeros(dimensions)
    beta_pos = np.zeros(dimensions)
    delta_pos = np.zeros(dimensions)

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    convergence_curve = []

    # Main optimization loop
    for iteration in range(max_iter):

        for i in range(search_agents):

            fitness = objective(*wolves[i])

            if not np.isfinite(fitness):
                fitness = float("inf")

            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = alpha_score
                beta_pos = alpha_pos.copy()

                alpha_score = fitness
                alpha_pos = wolves[i].copy()

            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = fitness
                beta_pos = wolves[i].copy()

            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = wolves[i].copy()

        # Linearly decreasing parameter
        a = 2 - 2 * iteration / max_iter

        # Update wolf positions
        for i in range(search_agents):

            new_positions = []

            for leader in [
                alpha_pos,
                beta_pos,
                delta_pos
            ]:

                r1 = np.random.random(dimensions)
                r2 = np.random.random(dimensions)

                A = 2 * a * r1 - a
                C = 2 * r2

                D = np.abs(C * leader - wolves[i])

                X = leader - A * D

                new_positions.append(X)

            wolves[i] = np.mean(
                new_positions, axis=0
            )

            # Maintain search boundaries
            wolves[i] = np.clip(
                wolves[i],
                lower_bound,
                upper_bound
            )

        convergence_curve.append(alpha_score)

        print(
            f"Iteration {iteration + 1}/{max_iter} "
            f"| Best Fitness: {alpha_score:.6f}"
        )

    return alpha_pos, alpha_score, convergence_curve


# =====================================================
# 5. FITNESS FUNCTION FOR GWO
# =====================================================

def objective(Kp, Ki, Kd):

    result = calculate_performance(Kp, Ki, Kd)

    if result is None:
        return 1e10

    return result["Fitness"]


# =====================================================
# 6. CONVENTIONAL PID PARAMETERS
# =====================================================

# Baseline PID settings for comparison.
# These are illustrative fixed gains, not a
# claimed Ziegler-Nichols tuning result.

Kp_conventional = 5.0
Ki_conventional = 2.0
Kd_conventional = 1.0

conventional_metrics = calculate_performance(
    Kp_conventional,
    Ki_conventional,
    Kd_conventional
)


# =====================================================
# 7. RUN GREY WOLF OPTIMIZATION
# =====================================================

print("\nStarting Grey Wolf Optimization...\n")

best_pid, best_fitness, convergence = GWO(
    objective,
    search_agents=30,
    max_iter=100
)

Kp_opt, Ki_opt, Kd_opt = best_pid

print("\nOPTIMAL PID PARAMETERS")
print("----------------------")
print("Kp =", Kp_opt)
print("Ki =", Ki_opt)
print("Kd =", Kd_opt)


# =====================================================
# 8. EVALUATE OPTIMIZED PID
# =====================================================

optimized_metrics = calculate_performance(
    Kp_opt,
    Ki_opt,
    Kd_opt
)

print("\nOPTIMIZED PID PERFORMANCE")
print("-------------------------")

for key, value in optimized_metrics.items():
    print(f"{key}: {value:.6f}")


# =====================================================
# 9. PERFORMANCE COMPARISON
# =====================================================

comparison = pd.DataFrame(
    [conventional_metrics, optimized_metrics],
    index=["Conventional PID", "GWO-PID"]
)

print("\nPERFORMANCE COMPARISON")
print(comparison.to_string())


# =====================================================
# 10. RESPONSE CURVE COMPARISON
# =====================================================

time1, output1 = simulate_pid(
    Kp_conventional,
    Ki_conventional,
    Kd_conventional
)

time2, output2 = simulate_pid(
    Kp_opt,
    Ki_opt,
    Kd_opt
)

plt.figure(figsize=(10, 6))

plt.plot(
    time1,
    output1,
    label="Conventional PID",
    linewidth=2
)

plt.plot(
    time2,
    output2,
    label="GWO Optimized PID",
    linewidth=2
)

plt.axhline(
    y=1.0,
    color="black",
    linestyle="--",
    label="Reference Input"
)

plt.xlabel("Time (seconds)")
plt.ylabel("System Output")
plt.title("Conventional PID vs GWO-PID")

plt.legend()
plt.grid(True)
plt.show()


# =====================================================
# 11. GWO CONVERGENCE CURVE
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(convergence) + 1),
    convergence,
    color="green",
    linewidth=2
)

plt.xlabel("Iteration")
plt.ylabel("Best Fitness Value")
plt.title("GWO Convergence Curve")

plt.grid(True)
plt.show()


# =====================================================
# 12. PERFORMANCE METRICS GRAPH
# =====================================================

metrics = [
    "ITAE",
    "Overshoot",
    "Settling Time",
    "Steady State Error"
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    [conventional_metrics[m] for m in metrics],
    width,
    label="Conventional PID"
)

plt.bar(
    x + width / 2,
    [optimized_metrics[m] for m in metrics],
    width,
    label="GWO-PID"
)

plt.xticks(x, metrics)
plt.ylabel("Performance Value")
plt.title("PID Controller Performance Comparison")

plt.legend()
plt.grid(axis="y")
plt.show()
