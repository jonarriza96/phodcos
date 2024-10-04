import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from papor.utils.quaternion import *


def Bezier(t, p, d=9):
    bz = p[:, 0]
    if d == 1:
        bz = (1 - t) * p[0, :] + t * p[1, :]
    elif d == 2:
        bz = p[0, :] * (1 - t) ** 2 + 2 * p[1, :] * (1 - t) * t + p[2, :] * t**2
    elif d == 4:
        bz = (
            (1 - t) ** 4 * p[0, :]
            + 4 * (1 - t) ** 3 * t * p[1, :]
            + 6 * (1 - t) ** 2 * t**2 * p[2, :]
            + 4 * (1 - t) * t**3 * p[3, :]
            + t**4 * p[4, :]
        )
    elif d == 5:
        bz = (
            (1 - t) ** 5 * p[0, :]
            + 5 * t * (1 - t) ** 4 * p[1, :]
            + 10 * t**2 * (1 - t) ** 3 * p[2, :]
            + 10 * t**3 * (1 - t) ** 2 * p[3, :]
            + 5 * t**4 * (1 - t) * p[4, :]
            + t**5 * p[5, :]
        )
    elif d == 8:
        bz = (
            p[0, :] * (1 - t) ** 8
            + 8 * p[1, :] * (1 - t) ** 7 * t
            + 28 * p[2, :] * (1 - t) ** 6 * t**2
            + 56 * p[3, :] * (1 - t) ** 5 * t**3
            + 70 * p[4, :] * (1 - t) ** 4 * t**4
            + 56 * p[5, :] * (1 - t) ** 3 * t**5
            + 28 * p[6, :] * (1 - t) ** 2 * t**6
            + 8 * p[7, :] * (1 - t) * t**7
            + p[8, :] * t**8
        )
    elif d == 9:
        bz = (
            (1 - t) ** 9 * p[0, :]
            + 9 * (1 - t) ** 8 * t * p[1, :]
            + 36 * (1 - t) ** 7 * t**2 * p[2, :]
            + 84 * (1 - t) ** 6 * t**3 * p[3, :]
            + 126 * (1 - t) ** 5 * t**4 * p[4, :]
            + 126 * (1 - t) ** 4 * t**5 * p[5, :]
            + 84 * (1 - t) ** 3 * t**6 * p[6, :]
            + 36 * (1 - t) ** 2 * t**7 * p[7, :]
            + 9 * (1 - t) * t**8 * p[8, :]
            + t**9 * p[9, :]
        )
    elif d == 17:
        bz = (
            p[0, :] * (1 - t) ** 17
            + 17 * p[1, :] * (1 - t) ** 16 * t
            + 136 * p[2, :] * (1 - t) ** 15 * t**2
            + 680 * p[3, :] * (1 - t) ** 14 * t**3
            + 2380 * p[4, :] * (1 - t) ** 13 * t**4
            + 6188 * p[5, :] * (1 - t) ** 12 * t**5
            + 12376 * p[6, :] * (1 - t) ** 11 * t**6
            + 19448 * p[7, :] * (1 - t) ** 10 * t**7
            + 24310 * p[8, :] * (1 - t) ** 9 * t**8
            + 24310 * p[9, :] * (1 - t) ** 8 * t**9
            + 19448 * p[10, :] * (1 - t) ** 7 * t**10
            + 12376 * p[11, :] * (1 - t) ** 6 * t**11
            + 6188 * p[12, :] * (1 - t) ** 5 * t**12
            + 2380 * p[13, :] * (1 - t) ** 4 * t**13
            + 680 * p[14, :] * (1 - t) ** 3 * t**14
            + 136 * p[15, :] * (1 - t) ** 2 * t**15
            + 17 * p[16, :] * (1 - t) * t**16
            + p[17, :] * t**17
        )
    return bz


def Z_to_erf(t, Z):
    # ERF
    sigma = Z[0] ** 2 + Z[1] ** 2 + Z[2] ** 2 + Z[3] ** 2
    e1 = (quat_AiconjA(Z, "i") / sigma)[1:]
    e2 = (quat_AiconjA(Z, "j") / sigma)[1:]
    e3 = (quat_AiconjA(Z, "k") / sigma)[1:]
    erf = cs.horzcat(e1, e2, e3)

    # Darboux vector
    # e1_d = cs.jacobian(e1, t)  # .T
    # e2_d = cs.jacobian(e2, t)  # .T
    # e3_d = cs.jacobian(e3, t)  # .T

    # X1 = cs.dot(e2_d, e3)
    # X2 = cs.dot(e3_d, e1)
    # X3 = cs.dot(e1_d, e2)
    # # omega =  X1*e1+X2*e2+X3*e3
    # X = cs.horzcat(X1, X2, X3)

    return erf  # , X  # ,omega


def normalize_halfspace(a, b):
    """
    Normalizes a halfspace A*x - b < 0 for ONE sample. The pseudo-code is:

            if b != 0:                                   # CASE 1
                C = sign(b)*(A/b)/norm(A/b)
                d = sign(b)*1/norm(A/b)
            else:
                if norm(A) > 0:                         # CASE 2
                    C = A/norm(A); D = 0
                else:                                   # CASE 3
                    C = 0; D = 0

    Notice that the norms are applied over the x.y,z columns

    Args:
        a: [n_sides_total x 3] or a list [n_poly][n_sides_per_poly x 3]
        b: [n_sides_total] or a list [n_poly][n_sides_per_poly]

    Returns:
        c: [n_sides_total x 3] or a list [n_poly][n_sides_per_poly x 3]
        d: [n_sides_total] or a list [n_poly][n_sides_per_poly]

    """

    # concatenate into a matrix if input is a list
    if isinstance(a, list):
        C = []
        D = []
        for aa, bb in zip(a, b):
            # declare matrixes and identify case indexes
            n_sides = aa.shape[0]
            c = np.zeros((n_sides, 3))
            d = np.zeros(n_sides)

            ind_case1 = np.squeeze(np.argwhere((bb > 1e-6) | (bb < -1e-6)))
            ind_case2 = np.squeeze(
                np.stack([np.linalg.norm(aa, axis=1) > 0, bb <= 0]).all(axis=0)
            )

            # case 1
            c[ind_case1, :] = aa[ind_case1, :] / bb[ind_case1][:, np.newaxis]
            c_norm = np.linalg.norm(c[ind_case1, :], axis=1)
            c[ind_case1, :] = (
                np.sign(bb[ind_case1])[:, np.newaxis]
                * c[ind_case1, :]
                / c_norm[:, np.newaxis]
            )
            d[ind_case1] = np.sign(bb[ind_case1]) * 1 / c_norm

            # case 2
            c[ind_case2, :] = (
                aa[ind_case2, :]
                / np.linalg.norm(aa[ind_case2, :], axis=1)[:, np.newaxis]
            )

            C += [c]
            D += [d]

    else:
        # declare matrixes and identify case indexes
        n_sides = a.shape[0]
        c = np.zeros((n_sides, 3))
        d = np.zeros(n_sides)

        ind_case1 = np.squeeze(np.argwhere((b > 1e-6) | (b < -1e-6)))
        ind_case2 = np.squeeze(
            np.stack([np.linalg.norm(a, axis=1) > 0, b <= 0]).all(axis=0)
        )

        # case 1
        c[ind_case1, :] = a[ind_case1, :] / b[ind_case1][:, np.newaxis]
        c_norm = np.linalg.norm(c[ind_case1, :], axis=1)
        c[ind_case1, :] = (
            np.sign(b[ind_case1])[:, np.newaxis]
            * c[ind_case1, :]
            / c_norm[:, np.newaxis]
        )
        d[ind_case1] = np.sign(b[ind_case1]) * 1 / c_norm

        # case 2
        c[ind_case2, :] = (
            a[ind_case2, :] / np.linalg.norm(a[ind_case2, :], axis=1)[:, np.newaxis]
        )

        C = c
        D = d

    return C, D


def transform_to_initial_pose(A, b, wp, q):
    R_i = quaternion_to_rotation(q[0])
    wp_i = np.vstack([np.zeros(3), np.dot(R_i.T, wp[1] - wp[0])])
    q_i = np.vstack(
        [
            rotation_to_quaternion(np.eye(3)),
            rotation_to_quaternion(np.dot(R_i.T, quaternion_to_rotation(q[1]))),
        ]
    )

    A_i = []
    b_i = []
    for A_w, b_w in zip(A, b):
        A_i += [np.dot(A_w, R_i)]
        b_i += [-np.dot(A_w, wp[0]) + b_w]

    return A_i, b_i, wp_i, q_i
