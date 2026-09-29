from functions1_6 import *

import numpy as np
import scipy
import matplotlib.pyplot as plt

fig_size = (4, 4)
fsz = 10  # font size
plt.rcParams.update({"font.size": fsz})


# Problem 1.6
points = np.linspace(-1, 1, 1000)

original_points = (5 / 2) * (points + 1)


N_list = [4, 5, 10, 50]
approx_values_per_N = [problem_1_6_solve_for_N(N=n, points=points) for n in N_list]

# actual function
actual_values = actual_function(original_points)

fig, axs = plt.subplots(2, 2)
for i, ax in enumerate(axs.flat):
    ax.plot(original_points, approx_values_per_N[i], label=f"Approximation")
    ax.plot(original_points, actual_values, label="Actual")

    max_error = np.max(np.abs(approx_values_per_N[i] - actual_values))

    ax.set_title(f"N={N_list[i]},max_error={max_error:.2e}")


plt.legend()
plt.tight_layout()
plt.savefig("./figures/problem1_6_1.pdf")

# Finding N where roundoff error starts to worsen results

N_list = range(10, 100)
max_error_list = []
for n in N_list:
    approx_values = problem_1_6_solve_for_N(N=n, points=points)
    max_error = np.max(np.abs(approx_values - actual_values))
    max_error_list.append(max_error)

plt.figure()
plt.plot(N_list, max_error_list)
plt.xlabel("N")
plt.ylabel("Max error per N")
plt.yscale("log")
plt.savefig("./figures/problem1_6_2.pdf")
