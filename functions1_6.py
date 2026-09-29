import numpy as np
import scipy


# Chebyshev polynomials
def Tsch(n, x):
    return np.cos(n * np.arccos(x))


def cheb(N) -> tuple[np.ndarray, np.ndarray]:
    if N == 0:
        D = 0
        x = 1
        return np.array(D), np.array(x)
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones((N + 1,))
    c[0] = 2
    c[N] = 2
    c[1::2] = -c[1::2]
    # create an (N+1)-by-(N+1) matrix whose columns are the vectors of Chebychev nodes
    X = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        X[:, j] = x
    dX = X - np.transpose(X)
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))  # off-diagonal entries
    D = D - np.diag(np.sum(D, axis=1))  # D[i,i] = -\sum_{j\neq i} D[i,j]
    return D, x


def get_coeffs(x, f_x, N):

    coeffs = []
    for k in range(0, N + 1):
        T_vec = Tsch(k, x)
        T_vec[0] /= 2
        T_vec[N] /= 2
        c_k = 2 / N * (T_vec.T @ f_x)
        coeffs.append(c_k)

    return np.array(coeffs)


def evaluate_point_clenshaw(coeffs, N, x):
    A = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(N + 1):

            if i == j:
                A[i, j] = 1
            elif i == j + 1:
                A[i, j] = -2 * x
            elif i == j + 2:
                A[i, j] = 1

    b = np.linalg.solve(A.T, coeffs)

    return 1 / 2 * (b[0] - b[2] - b[N] * np.cos(N * np.arccos(x)))


def problem_1_6_solve_for_N(N, points):
    # First, get points and differentiation matrix
    D, x = cheb(N)

    # G points
    G = get_G(x)

    diff_eq_mat = 16 / 625 * D @ D @ D @ D + 2 / 5 * D + np.eye(N + 1)

    # boundary condition equations

    # zero order
    b_cond_1 = np.zeros(N + 1)
    b_cond_1[0] = 1
    diff_eq_mat[0, :] = b_cond_1
    G[0] = 0

    b_cond_2 = np.zeros(N + 1)
    b_cond_2[N] = 1
    diff_eq_mat[N, :] = b_cond_2
    G[N] = 0

    # first order
    diff_eq_mat[1, :] = D[0, :]
    G[1] = 0

    diff_eq_mat[N - 1, :] = D[N, :]
    G[N - 1] = 0

    # solve
    U = np.linalg.solve(diff_eq_mat, G)

    # now get interpolating polynomial
    coeffs = get_coeffs(x, U, N)

    approx_values = []

    for x in points:
        y = evaluate_point_clenshaw(coeffs, N, x)
        approx_values.append(y)

    approx_values = np.array(approx_values)

    # now, we have V, but we have to add B back

    def b(x):
        return (
            (x - 1) ** 2 * (x + 2) / 4
            + (x + 1) ** 2 * (2 - x) / 104
            - 25 / 676 / 4 * (x + 1) ** 2 * (x - 1)
        )

    return approx_values + b(points)


def get_G(x):

    def g(x):

        return (
            (24 - 240 * x**2 + 120 * x**4) / (1 + x**2) ** 5
            + 1 / (1 + x**2)
            - (2 * x) / (1 + x**2) ** 2
        )

    return (
        g(5 / 2 * (x + 1))
        - (x - 1) ** 2 * (x + 2) / 4
        - (x + 1) ** 2 * (2 - x) / 104
        + 25 / 2704 * (x + 1) ** 2 * (x - 1)
        - 2 / 5 * (1875 * x**2 - 50 * x - 1925) / 2704
    )


def actual_function(x):
    return 1 / (1 + x**2)
