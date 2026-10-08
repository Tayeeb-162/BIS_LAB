import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# ANTENNA ARRAY DESIGN USING CAPSO
# Based on:
# "Antenna array design by a contraction adaptive
# particle swarm optimization algorithm"
# ============================================================


# ------------------------------------------------------------
# 1. PARAMETERS
# ------------------------------------------------------------

N = 20                  # Number of antenna elements
wavelength = 1.0        # Normalized wavelength

d_min = 0.25            # Minimum element spacing (lambda)
d_max = 0.50            # Maximum element spacing (lambda)

num_particles = 40      # Number of particles
max_iterations = 200    # Maximum PSO iterations

theta = np.linspace(-90, 90, 1801)
theta_rad = np.deg2rad(theta)

theta_0 = 0
theta_0_rad = np.deg2rad(theta_0)


# ------------------------------------------------------------
# 2. INITIAL UNIFORM ARRAY
# ------------------------------------------------------------

uniform_positions = np.arange(N) * 0.5


# ------------------------------------------------------------
# 3. ARRAY FACTOR
# ------------------------------------------------------------

def array_factor(positions, theta_rad):

    AF = np.zeros(len(theta_rad), dtype=complex)

    for x in positions:

        AF += np.exp(
            1j * 2 * np.pi * x *
            (np.sin(theta_rad) -
             np.sin(theta_0_rad))
        )

    AF = np.abs(AF)

    AF = AF / np.max(AF)

    return AF


# ------------------------------------------------------------
# 4. RADIATION PATTERN
# ------------------------------------------------------------

def radiation_pattern(positions):

    AF = array_factor(
        positions,
        theta_rad
    )

    AF_dB = 20 * np.log10(
        AF + 1e-12
    )

    AF_dB = np.maximum(
        AF_dB,
        -60
    )

    return AF_dB


# ------------------------------------------------------------
# 5. SIDE LOBE LEVEL
# ------------------------------------------------------------

def calculate_sll(positions):

    AF = array_factor(
        positions,
        theta_rad
    )

    AF_dB = 20 * np.log10(
        AF + 1e-12
    )

    # Main beam region
    main_lobe_region = (
        np.abs(theta) <= 5
    )

    # Remove main lobe
    sidelobes = AF_dB[
        ~main_lobe_region
    ]

    maximum_sidelobe = np.max(
        sidelobes
    )

    return maximum_sidelobe


# ------------------------------------------------------------
# 6. FITNESS FUNCTION
# ------------------------------------------------------------

def fitness(positions):

    spacing = np.diff(positions)

    # Spacing constraint
    if (
        np.any(spacing < d_min)
        or
        np.any(spacing > d_max)
    ):
        return 1000

    sll = calculate_sll(
        positions
    )

    # Lower SLL is better
    return sll


# ------------------------------------------------------------
# 7. GENERATE VALID PARTICLE
# ------------------------------------------------------------

def generate_particle():

    spacing = np.random.uniform(
        d_min,
        d_max,
        N - 1
    )

    positions = np.concatenate(
        ([0],
         np.cumsum(spacing))
    )

    return positions


# ============================================================
# 8. STANDARD PSO
# ============================================================

def standard_pso():

    particles = np.array([
        generate_particle()
        for _ in range(num_particles)
    ])

    velocities = np.zeros_like(
        particles
    )

    personal_best = particles.copy()

    personal_best_scores = np.array([
        fitness(p)
        for p in personal_best
    ])

    best_index = np.argmin(
        personal_best_scores
    )

    global_best = (
        personal_best[
            best_index
        ].copy()
    )

    global_best_score = (
        personal_best_scores[
            best_index
        ]
    )

    history = []

    # PSO coefficients
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    for iteration in range(
        max_iterations
    ):

        for i in range(
            num_particles
        ):

            r1 = np.random.rand(N)
            r2 = np.random.rand(N)

            velocities[i] = (
                w * velocities[i]
                +
                c1 * r1 *
                (
                    personal_best[i]
                    -
                    particles[i]
                )
                +
                c2 * r2 *
                (
                    global_best
                    -
                    particles[i]
                )
            )

            particles[i] += (
                velocities[i]
            )

            # Fix first element
            particles[i] -= (
                particles[i][0]
            )

            spacing = np.diff(
                particles[i]
            )

            # Repair invalid particle
            if (
                np.any(
                    spacing < d_min
                )
                or
                np.any(
                    spacing > d_max
                )
            ):

                particles[i] = (
                    generate_particle()
                )

            current_score = fitness(
                particles[i]
            )

            # Update personal best
            if (
                current_score
                <
                personal_best_scores[i]
            ):

                personal_best[i] = (
                    particles[i].copy()
                )

                personal_best_scores[i] = (
                    current_score
                )

            # Update global best
            if (
                current_score
                <
                global_best_score
            ):

                global_best = (
                    particles[i].copy()
                )

                global_best_score = (
                    current_score
                )

        history.append(
            global_best_score
        )

        print(
            f"PSO Iteration "
            f"{iteration + 1:3d} | "
            f"SLL = "
            f"{global_best_score:.2f} dB"
        )

    return (
        global_best,
        global_best_score,
        history
    )


# ============================================================
# 9. CONTRACTION ADAPTIVE PSO
# ============================================================

def capso():

    particles = np.array([
        generate_particle()
        for _ in range(num_particles)
    ])

    velocities = np.zeros_like(
        particles
    )

    personal_best = particles.copy()

    personal_best_scores = np.array([
        fitness(p)
        for p in personal_best
    ])

    best_index = np.argmin(
        personal_best_scores
    )

    global_best = (
        personal_best[
            best_index
        ].copy()
    )

    global_best_score = (
        personal_best_scores[
            best_index
        ]
    )

    history = []

    c1 = 2.0
    c2 = 2.0

    for iteration in range(
        max_iterations
    ):

        # Adaptive contraction factor
        progress = (
            iteration /
            max_iterations
        )

        contraction = (
            0.9 -
            0.7 * progress
        )

        for i in range(
            num_particles
        ):

            r1 = np.random.rand(N)
            r2 = np.random.rand(N)

            velocities[i] = (
                contraction *
                (
                    velocities[i]
                    +
                    c1 * r1 *
                    (
                        personal_best[i]
                        -
                        particles[i]
                    )
                    +
                    c2 * r2 *
                    (
                        global_best
                        -
                        particles[i]
                    )
                )
            )

            particles[i] += (
                velocities[i]
            )

            # Fix first element
            particles[i] -= (
                particles[i][0]
            )

            spacing = np.diff(
                particles[i]
            )

            # Repair invalid particle
            if (
                np.any(
                    spacing < d_min
                )
                or
                np.any(
                    spacing > d_max
                )
            ):

                particles[i] = (
                    generate_particle()
                )

            current_score = fitness(
                particles[i]
            )

            # Update personal best
            if (
                current_score
                <
                personal_best_scores[i]
            ):

                personal_best[i] = (
                    particles[i].copy()
                )

                personal_best_scores[i] = (
                    current_score
                )

            # Update global best
            if (
                current_score
                <
                global_best_score
            ):

                global_best = (
                    particles[i].copy()
                )

                global_best_score = (
                    current_score
                )

        history.append(
            global_best_score
        )

        print(
            f"CAPSO Iteration "
            f"{iteration + 1:3d} | "
            f"SLL = "
            f"{global_best_score:.2f} dB"
        )

    return (
        global_best,
        global_best_score,
        history
    )


# ============================================================
# 10. RUN STANDARD PSO
# ============================================================

print("\n==============================")
print("Running Standard PSO")
print("==============================")

(
    pso_positions,
    pso_sll,
    pso_history
) = standard_pso()


# ============================================================
# 11. RUN CAPSO
# ============================================================

print("\n==============================")
print("Running CAPSO")
print("==============================")

(
    capso_positions,
    capso_sll,
    capso_history
) = capso()


# ============================================================
# 12. RADIATION PATTERNS
# ============================================================

uniform_pattern = radiation_pattern(
    uniform_positions
)

pso_pattern = radiation_pattern(
    pso_positions
)

capso_pattern = radiation_pattern(
    capso_positions
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(
    f"Uniform Array SLL : "
    f"{calculate_sll(uniform_positions):.2f} dB"
)

print(
    f"PSO Optimized SLL : "
    f"{pso_sll:.2f} dB"
)

print(
    f"CAPSO Optimized SLL : "
    f"{capso_sll:.2f} dB"
)

print("\nCAPSO Element Positions:")

print(
    np.round(
        capso_positions,
        4
    )
)

print("\nCAPSO Element Spacing:")

print(
    np.round(
        np.diff(capso_positions),
        4
    )
)


# ============================================================
# 14. RADIATION PATTERN COMPARISON
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    theta,
    uniform_pattern,
    label="Uniform Array",
    linewidth=2
)

plt.plot(
    theta,
    pso_pattern,
    label="PSO Optimized",
    linewidth=2
)

plt.plot(
    theta,
    capso_pattern,
    label="CAPSO Optimized",
    linewidth=2
)

plt.xlabel(
    "Angle (degrees)"
)

plt.ylabel(
    "Normalized Radiation Pattern (dB)"
)

plt.title(
    "Antenna Array Radiation Pattern Optimization"
)

plt.grid(True)

plt.legend()

plt.ylim(
    -50,
    2
)

plt.show()


# ============================================================
# 15. CONVERGENCE COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    pso_history,
    label="Standard PSO",
    linewidth=2
)

plt.plot(
    capso_history,
    label="CAPSO",
    linewidth=2
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Best SLL (dB)"
)

plt.title(
    "PSO vs CAPSO Convergence"
)

plt.grid(True)

plt.legend()

plt.show()


# ============================================================
# 16. ELEMENT POSITION COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 4)
)

plt.scatter(
    uniform_positions,
    np.ones(N),
    s=80,
    label="Uniform Array"
)

plt.scatter(
    capso_positions,
    np.ones(N) * 0.8,
    s=80,
    label="CAPSO Array"
)

plt.xlabel(
    "Element Position (lambda)"
)

plt.yticks([])

plt.title(
    "Antenna Element Position Optimization"
)

plt.grid(True)

plt.legend()

plt.show()