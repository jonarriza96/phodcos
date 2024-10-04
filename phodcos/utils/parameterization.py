import numpy as np
import matplotlib.pyplot as plt
import casadi as cs

from phodcos.utils.ph_curve import *
from phodcos.utils.quaternion import *
from phodcos.utils.visualize import *


def preimage_continuity(
    A0_rot,
    A1_rot,
    A2_rot,
    A3_rot,
    A4_rot,
    A5_rot,
    A6_rot,
    A7_rot,
    A8_rot,
    interp_order,
    m,
    parametric_funcs,
):
    """Negative sign to preimage (rotation by 2pi) coefficients for continutiy"""

    if m > 0:
        if interp_order == 2:
            A = parametric_funcs["Z"](
                0,
                A0_rot[m],
                A1_rot[m],
                A2_rot[m],
                A3_rot[m],
                A4_rot[m],
            )
            A_old = parametric_funcs["Z"](
                1,
                A0_rot[m - 1],
                A1_rot[m - 1],
                A2_rot[m - 1],
                A3_rot[m - 1],
                A4_rot[m - 1],
            )
        elif interp_order == 4:
            A = parametric_funcs["Z"](
                0,
                A0_rot[m],
                A1_rot[m],
                A2_rot[m],
                A3_rot[m],
                A4_rot[m],
                A5_rot[m],
                A6_rot[m],
                A7_rot[m],
                A8_rot[m],
            )
            A_old = parametric_funcs["Z"](
                1,
                A0_rot[m - 1],
                A1_rot[m - 1],
                A2_rot[m - 1],
                A3_rot[m - 1],
                A4_rot[m - 1],
                A5_rot[m - 1],
                A6_rot[m - 1],
                A7_rot[m - 1],
                A8_rot[m - 1],
            )
        Ak = np.squeeze([A]) / np.linalg.norm(np.squeeze([A]))
        Ak_old = np.squeeze([A_old]) / np.linalg.norm(np.squeeze([A_old]))

        # rotate if flipped
        # diff_q = quat_mult(Ak, quat_conj(Ak_old))
        # diff_q /= np.linalg.norm(diff_q)
        # print(np.linalg.norm(diff_q - np.array([-1, 0, 0, 0])))
        # if (
        #     np.linalg.norm(diff_q - np.array([-1, 0, 0, 0])) < 1
        # ):  # TODO --> flip threshold

        flipped = True
        for j in range(4):
            if Ak_old[j] != 0 and Ak[j] != 0:
                if np.sign(Ak_old[j]) == np.sign(Ak[j]):
                    flipped = False
                    break
        if flipped:
            A0_rot[m, :] *= -1
            A1_rot[m, :] *= -1
            A2_rot[m, :] *= -1
            A3_rot[m, :] *= -1
            A4_rot[m, :] *= -1
            if interp_order == 4:
                A5_rot[m, :] *= -1
                A6_rot[m, :] *= -1
                A7_rot[m, :] *= -1
                A8_rot[m, :] *= -1

    return A0_rot, A1_rot, A2_rot, A3_rot, A4_rot, A5_rot, A6_rot, A7_rot, A8_rot


def erf_continuity(
    A0_rot,
    A1_rot,
    A2_rot,
    A3_rot,
    A4_rot,
    A5_rot,
    A6_rot,
    A7_rot,
    A8_rot,
    interp_order,
    m,
    parametric_funcs,
):
    """Rotate adapted frame around e1 (roll) for continuity in e2 and e3"""
    n_segments = A0_rot.shape[0]
    Rot = np.zeros((n_segments, 3, 3))
    Rot[0, :, :] = np.eye(3)
    if m > 0:
        if interp_order == 2:
            erf = parametric_funcs["erf"](
                1e-6,
                A0_rot[m],
                A1_rot[m],
                A2_rot[m],
                A3_rot[m],
                A4_rot[m],
            )
            erf_old = parametric_funcs["erf"](
                1,
                A0_rot[m - 1],
                A1_rot[m - 1],
                A2_rot[m - 1],
                A3_rot[m - 1],
                A4_rot[m - 1],
            )
        elif interp_order == 4:
            erf = parametric_funcs["erf"](
                1e-6,
                A0_rot[m],
                A1_rot[m],
                A2_rot[m],
                A3_rot[m],
                A4_rot[m],
                A5_rot[m],
                A6_rot[m],
                A7_rot[m],
                A8_rot[m],
            )
            erf_old = parametric_funcs["erf"](
                1,
                A0_rot[m - 1],
                A1_rot[m - 1],
                A2_rot[m - 1],
                A3_rot[m - 1],
                A4_rot[m - 1],
                A5_rot[m - 1],
                A6_rot[m - 1],
                A7_rot[m - 1],
                A8_rot[m - 1],
            )
        Rk = np.squeeze([erf])
        Rk_old = np.squeeze([erf_old])

        # roll --> rotation around e1
        theta = angle_between(Rk[:, 1], Rk_old[:, 1])
        Rot[m, :, :] = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

        # roll clockwise or anti clockwise
        if np.linalg.norm(np.matmul(Rk, Rot[m]) - Rk_old) < np.linalg.norm(
            np.matmul(Rk, Rot[m].T) - Rk_old
        ):
            qrot = rotation_to_quaternion(Rot[m])
        else:
            qrot = rotation_to_quaternion(Rot[m].T)
        # Rq = ((Quaternion(matrix=Rk)*qrot)).rotation_matrix
        # print(Rq - Rk_old)

        # rotate ctrlpts of quaternion polynomial
        A0_rot[m, :] = quat_mult(A0_rot[m, :], qrot)
        A1_rot[m, :] = quat_mult(A1_rot[m, :], qrot)
        A2_rot[m, :] = quat_mult(A2_rot[m, :], qrot)
        A3_rot[m, :] = quat_mult(A3_rot[m, :], qrot)
        A4_rot[m, :] = quat_mult(A4_rot[m, :], qrot)
        if interp_order == 4:
            A5_rot[m, :] = quat_mult(A5_rot[m, :], qrot)
            A6_rot[m, :] = quat_mult(A6_rot[m, :], qrot)
            A7_rot[m, :] = quat_mult(A7_rot[m, :], qrot)
            A8_rot[m, :] = quat_mult(A8_rot[m, :], qrot)

    return A0_rot, A1_rot, A2_rot, A3_rot, A4_rot, A5_rot, A6_rot, A7_rot, A8_rot


def parametric_segment_path_functions(interp_order):
    """Generates symbolic functions for C2 or C4 hermite interpolation of PH curves
        interp_order: Order of hermite interpolation (2 or 4)

        NOTE: These symbolic functions are PER SEGMENT.
    Returns:
        dict: Dictionary with hodograph, curve and PH construction functions
    """
    if interp_order == 2:
        # symbolic variables and constants
        pb = cs.MX.sym("pb", 3)
        pe = cs.MX.sym("pe", 3)
        vb = cs.MX.sym("vb", 3)
        ve = cs.MX.sym("ve", 3)
        ab = cs.MX.sym("ab", 3)
        ae = cs.MX.sym("ae", 3)

        o0 = cs.MX.sym("o0")
        o4 = cs.MX.sym("o4")
        o2 = cs.MX.sym("o2")
        t1 = cs.MX.sym("t1")
        t3 = cs.MX.sym("t3")

        i = cs.DM([0, 1, 0, 0])

        # ------------------- function for C2 hermite interpolation ------------------ #
        # Step 1
        h0 = cs.vertcat(0, vb)
        h8 = cs.vertcat(0, ve)
        h1 = cs.vertcat(0, ab / 8) + h0
        h7 = -(cs.vertcat(0, ae / 8) - h8)

        # Step 2
        h0_norm = cs.norm_2(h0)
        numden = (h0 / h0_norm) + i
        Qo0 = cs.vertcat(cs.cos(o0), cs.sin(o0), 0, 0)
        A0 = quat_mult(cs.sqrt(h0_norm) * (numden / cs.norm_2(numden)), Qo0)

        h8_norm = cs.norm_2(h8)
        numden = (h8 / h8_norm) + i
        Qo4 = cs.vertcat(cs.cos(o4), cs.sin(o4), 0, 0)
        A4 = quat_mult(cs.sqrt(h8_norm) * (numden / cs.norm_2(numden)), Qo4)

        # Step 3
        A1 = quat_mult(quat_mult(-(cs.vertcat(t1, 0, 0, 0) + h1), A0), i) / (
            cs.norm_2(A0) ** 2
        )
        A3 = quat_mult(quat_mult(-(cs.vertcat(t3, 0, 0, 0) + h7), A4), i) / (
            cs.norm_2(A4) ** 2
        )

        # Step 4
        delta_p = cs.vertcat(0, pe - pb)
        delta_v = cs.vertcat(0, ve + vb)
        delta_a = cs.vertcat(0, ae - ab)

        def AstB(A, B):
            return 1 / 2 * (quat_AiB(A, quat_conj(B)) + quat_AiB(B, quat_conj(A)))

        # alpha = 2520 * delta_p - 435 * delta_v + 45 / 2 * delta_a - (
        #     60 * AstB(A1, A1) - 60 * AstB(A0, A3) - 60 * AstB(A1, A4) + 60 *
        #     AstB(A3, A3) - 42 * AstB(A0, A4) - 72 * AstB(A1, A3))

        # alpha_norm = cs.norm_2(alpha)
        # numden = (alpha/alpha_norm) + i
        # Qo2 = cs.vertcat(cs.cos(o2), cs.sin(o2), 0, 0)
        # A2 = 1/12*(quat_mult(cs.sqrt(alpha_norm)*(numden/cs.norm_2(numden)), Qo2) -
        #            (10*A1 + 5*A0 + 5*A4 + 10*A3))

        def ZiZ(A, B):
            return quat_AiB(A, quat_conj(B)).T

        alpha2 = 27 * (
            85 * ZiZ(A0, A0)
            + 30 * ZiZ(A0, A1)
            - 10 * ZiZ(A0, A3)
            - 7 * ZiZ(A0, A4)
            + 30 * ZiZ(A1, A0)
            + 20 * ZiZ(A1, A1)
            - 12 * ZiZ(A1, A3)
            - 10 * ZiZ(A1, A4)
            - 10 * ZiZ(A3, A0)
            - 12 * ZiZ(A3, A1)
            + 20 * ZiZ(A3, A3)
            + 30 * ZiZ(A3, A4)
            - 7 * ZiZ(A4, A0)
            - 10 * ZiZ(A4, A1)
            + 30 * ZiZ(A4, A3)
            + 85 * ZiZ(A4, A4)
        )
        alpha = 22680 * delta_p - alpha2.T
        alpha_norm = cs.norm_2(alpha)
        numden = (alpha / alpha_norm) + i
        Qo2 = cs.vertcat(cs.cos(o2), cs.sin(o2), 0, 0)
        A2 = (
            1
            / 36
            * (
                quat_mult(cs.sqrt(alpha_norm) * (numden / cs.norm_2(numden)), Qo2)
                - (15 * A0 + 30 * A1 + 30 * A3 + 15 * A4)
            )
        )

        f_A = cs.Function(
            "f_A",
            [pb, pe, vb, ve, ab, ae, o0, o2, o4, t1, t3],
            [A0, A1, A2, A3, A4],
            ["pb", "pe", "vb", "ve", "ab", "ae", "o0", "o2", "o3", "t1", "t3"],
            ["A0", "A1", "A2", "A3", "A4"],
        )

        # ------------- quaternion coefficients to quaternion polynomial ------------- #

        xi = cs.MX.sym("xi")  # path parameter
        A0 = cs.MX.sym("A0", 4)  # quat. coeff 0
        A1 = cs.MX.sym("A1", 4)  # quat. coeff 1
        A2 = cs.MX.sym("A2", 4)  # quat. coeff 2
        A3 = cs.MX.sym("A3", 4)  # quat. coeff 3
        A4 = cs.MX.sym("A4", 4)  # quat. coeff 4
        Zcoeff = cs.horzcat(A0, A1, A2, A3, A4).T

        U = Bezier(xi, Zcoeff[:, 0], 4)
        V = Bezier(xi, Zcoeff[:, 1], 4)
        G = Bezier(xi, Zcoeff[:, 2], 4)
        H = Bezier(xi, Zcoeff[:, 3], 4)
        Z = cs.vertcat(U, V, G, H)
        Z_d = cs.jacobian(Z, xi)
        Z_dd = cs.jacobian(Z_d, xi)
        Z_ddd = cs.jacobian(Z_dd, xi)
        f_Z = cs.Function(
            "f_Z",
            [xi, A0, A1, A2, A3, A4],
            [Z],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["Z"],
        )

        # ----------------------------- parametric speed ----------------------------- #
        sigma = Z[0] ** 2 + Z[1] ** 2 + Z[2] ** 2 + Z[3] ** 2
        f_sigma = cs.Function(
            "f_sigma",
            [xi, A0, A1, A2, A3, A4],
            [sigma],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["sigma"],
        )

        # ------------------- quaternion control points to position ------------------ #

        # hodograph controlpoints
        h0 = AstB(A0, A0)
        h1 = AstB(A0, A1)
        h2 = 1 / 7 * (4 * AstB(A1, A1) + 3 * AstB(A0, A2))
        h3 = 1 / 7 * (AstB(A0, A3) + 6 * AstB(A1, A2))
        h4 = 1 / 35 * (18 * AstB(A2, A2) + AstB(A0, A4) + 16 * AstB(A1, A3))
        h5 = 1 / 7 * (AstB(A1, A4) + 6 * AstB(A2, A3))
        h6 = 1 / 7 * (4 * AstB(A3, A3) + 3 * AstB(A2, A4))
        h7 = AstB(A3, A4)
        h8 = AstB(A4, A4)
        hodograph = [h0, h1, h2, h3, h4, h5, h6, h7, h8]
        f_h = cs.Function(
            "f_h",
            [A0, A1, A2, A3, A4],
            hodograph,
            ["A0", "A1", "A2", "A3", "A4"],
            ["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"],
        )

        # curve controlpoints
        p0 = cs.MX.sym("p0", 3)
        p = cs.MX.zeros(10, 3)
        p[0, :] = p0
        for j in range(1, 10):
            # sum_hodograph = cs.MX.zeros(1,3)
            # for i in range(j):
            #    sum_hodograph += hodograph[i][1:,0].T
            p[j, :] = p[j - 1, :] + 1 / 9 * hodograph[j - 1][1:, 0].T

        # f_h = cs.Function('f_h', [p0, A0, A1, A2, A3, A4], hodograph)
        # f_p = cs.Function('f_p', [p0, A0, A1, A2, A3, A4], [p], [
        #    'p0', 'A0', 'A1', 'A2', 'A3', 'A4'], ['p'])

        # curve
        gamma = Bezier(xi, p).T
        f_gamma = cs.Function(
            "f_gamma",
            [xi, p0, A0, A1, A2, A3, A4],
            [gamma],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4"],
            ["gamma"],
        )

        # ------------------------------- adapted frame ------------------------------ #
        sigma = Z[0] ** 2 + Z[1] ** 2 + Z[2] ** 2 + Z[3] ** 2
        e1 = (quat_AiconjA(Z, "i") / sigma)[1:]
        e2 = (quat_AiconjA(Z, "j") / sigma)[1:]
        e3 = (quat_AiconjA(Z, "k") / sigma)[1:]
        erf = cs.horzcat(e1, e2, e3)

        f_erf = cs.Function(
            "f_erf",
            [xi, A0, A1, A2, A3, A4],
            [erf],
            ["xi", "A0", "A1", "A2", "A3", "A4"],
            ["erf"],
        )

        # f_X = cs.Function(
        #     "f_X",
        #     [xi, A0, A1, A2, A3, A4],
        #     [X],
        #     ["xi", "A0", "A1", "A2", "A3", "A4"],
        #     ["X"],
        # )

        # -------------------------- dictionary of functions ------------------------- #
        func_dict = {
            "Z": f_Z,
            "A": f_A,
            "gamma": f_gamma,
            "hodograph": f_h,
            "sigma": f_sigma,
            "erf": f_erf,
            # "X": f_X,
        }

    elif interp_order == 4:

        def AstB(A, B):
            return 1 / 2 * (quat_AiB(A, quat_conj(B)) + quat_AiB(B, quat_conj(A)))

        def ZiZ(A, B):
            return quat_AiB(A, quat_conj(B)).T

        # symbolic variables and constants
        pb = cs.MX.sym("pb", 3)
        pe = cs.MX.sym("pe", 3)
        vb = cs.MX.sym("vb", 3)
        ve = cs.MX.sym("ve", 3)
        ab = cs.MX.sym("ab", 3)
        ae = cs.MX.sym("ae", 3)
        jb = cs.MX.sym("jb", 3)
        je = cs.MX.sym("je", 3)
        sb = cs.MX.sym("sb", 3)
        se = cs.MX.sym("se", 3)

        o0 = cs.MX.sym("o0")
        t1 = cs.MX.sym("t1")
        t2 = cs.MX.sym("t2")
        t3 = cs.MX.sym("t3")
        o4 = cs.MX.sym("o4")
        t5 = cs.MX.sym("t5")
        t6 = cs.MX.sym("t6")
        t7 = cs.MX.sym("t7")
        o8 = cs.MX.sym("o8")

        i = cs.DM([0, 1, 0, 0])

        # ------------------- function for C2 hermite interpolation ------------------ #
        # Step 1
        h0 = cs.vertcat(0, vb)
        h16 = cs.vertcat(0, ve)
        h1 = 1 / 16 * (cs.vertcat(0, ab) - (-16 * h0))
        h15 = 1 / -16 * (cs.vertcat(0, ae) - (16 * h16))
        h2 = 1 / 240 * (cs.vertcat(0, jb) - (240 * h0 - 480 * h1))
        h14 = 1 / 240 * (cs.vertcat(0, je) - (-480 * h15 + 240 * h16))
        h3 = 1 / 3360 * (cs.vertcat(0, sb) - (-3360 * h0 + 10080 * h1 - 10080 * h2))
        h13 = 1 / -3360 * (cs.vertcat(0, se) - (10080 * h14 - 10080 * h15 + 3360 * h16))

        # Step 2
        h0_norm = cs.norm_2(h0)
        numden = (h0 / h0_norm) + i
        Qo0 = cs.vertcat(cs.cos(o0), cs.sin(o0), 0, 0)
        A0 = quat_mult(cs.sqrt(h0_norm) * (numden / cs.norm_2(numden)), Qo0)

        h16_norm = cs.norm_2(h16)
        numden = (h16 / h16_norm) + i
        Qo8 = cs.vertcat(cs.cos(o8), cs.sin(o8), 0, 0)
        A8 = quat_mult(cs.sqrt(h16_norm) * (numden / cs.norm_2(numden)), Qo8)

        # Step 3
        A1 = quat_mult(quat_mult(-(cs.vertcat(t1, 0, 0, 0) + h1), A0), i) / (
            cs.norm_2(A0) ** 2
        )
        A7 = quat_mult(quat_mult(-(cs.vertcat(t7, 0, 0, 0) + h15), A8), i) / (
            cs.norm_2(A8) ** 2
        )
        # Step 4
        h2_2 = 1 / 7 * (15 * h2 - 8 * AstB(A1, A1))
        h14_2 = 1 / 7 * (15 * h14 - 8 * AstB(A7, A7))
        A2 = quat_mult(quat_mult(-(cs.vertcat(t2, 0, 0, 0) + h2_2), A0), i) / (
            cs.norm_2(A0) ** 2
        )
        A6 = quat_mult(quat_mult(-(cs.vertcat(t6, 0, 0, 0) + h14_2), A8), i) / (
            cs.norm_2(A8) ** 2
        )

        # Step 5
        h3_2 = 1 / 2 * (10 * h3 - 8 * AstB(A1, A2))
        h13_2 = 1 / 2 * (10 * h13 - 8 * AstB(A6, A7))
        A3 = quat_mult(quat_mult(-(cs.vertcat(t3, 0, 0, 0) + h3_2), A0), i) / (
            cs.norm_2(A0) ** 2
        )
        A5 = quat_mult(quat_mult(-(cs.vertcat(t5, 0, 0, 0) + h13_2), A8), i) / (
            cs.norm_2(A8) ** 2
        )

        # Step 6
        delta_p = cs.vertcat(0, pe - pb)

        alpha2 = (1 / 112633092) * (
            147807 * ZiZ(A0, A0)
            + 72270 * ZiZ(A0, A1)
            + 30954 * ZiZ(A0, A2)
            + 9702 * ZiZ(A0, A3)
            - 3234 * ZiZ(A0, A5)
            - 3150 * ZiZ(A0, A6)
            - 1818 * ZiZ(A0, A7)
            - 565 * ZiZ(A0, A8)
            + 72270 * ZiZ(A1, A0)
            + 72732 * ZiZ(A1, A1)
            + 47124 * ZiZ(A1, A2)
            + 19404 * ZiZ(A1, A3)
            - 8820 * ZiZ(A1, A5)
            - 9324 * ZiZ(A1, A6)
            - 5668 * ZiZ(A1, A7)
            - 1818 * ZiZ(A1, A8)
            + 30954 * ZiZ(A2, A0)
            + 47124 * ZiZ(A2, A1)
            + 40572 * ZiZ(A2, A2)
            + 20580 * ZiZ(A2, A3)
            - 12348 * ZiZ(A2, A5)
            - 14308 * ZiZ(A2, A6)
            - 9324 * ZiZ(A2, A7)
            - 3150 * ZiZ(A2, A8)
            + 9702 * ZiZ(A3, A0)
            + 19404 * ZiZ(A3, A1)
            + 20580 * ZiZ(A3, A2)
            + 12348 * ZiZ(A3, A3)
            - 9604 * ZiZ(A3, A5)
            - 12348 * ZiZ(A3, A6)
            - 8820 * ZiZ(A3, A7)
            - 3234 * ZiZ(A3, A8)
            - 3234 * ZiZ(A5, A0)
            - 8820 * ZiZ(A5, A1)
            - 12348 * ZiZ(A5, A2)
            - 9604 * ZiZ(A5, A3)
            + 12348 * ZiZ(A5, A5)
            + 20580 * ZiZ(A5, A6)
            + 19404 * ZiZ(A5, A7)
            + 9702 * ZiZ(A5, A8)
            - 3150 * ZiZ(A6, A0)
            - 9324 * ZiZ(A6, A1)
            - 14308 * ZiZ(A6, A2)
            - 12348 * ZiZ(A6, A3)
            + 20580 * ZiZ(A6, A5)
            + 40572 * ZiZ(A6, A6)
            + 47124 * ZiZ(A6, A7)
            + 30954 * ZiZ(A6, A8)
            - 1818 * ZiZ(A7, A0)
            - 5668 * ZiZ(A7, A1)
            - 9324 * ZiZ(A7, A2)
            - 8820 * ZiZ(A7, A3)
            + 19404 * ZiZ(A7, A5)
            + 47124 * ZiZ(A7, A6)
            + 72732 * ZiZ(A7, A7)
            + 72270 * ZiZ(A7, A8)
            - 565 * ZiZ(A8, A0)
            - 1818 * ZiZ(A8, A1)
            - 3150 * ZiZ(A8, A2)
            - 3234 * ZiZ(A8, A3)
            + 9702 * ZiZ(A8, A5)
            + 30954 * ZiZ(A8, A6)
            + 72270 * ZiZ(A8, A7)
            + 147807 * ZiZ(A8, A8)
        )

        alpha = 490 / 21879 * delta_p - alpha2.T
        alpha_norm = cs.norm_2(alpha)
        numden = (alpha / alpha_norm) + i
        Qo4 = cs.vertcat(cs.cos(o4), cs.sin(o4), 0, 0)
        A4 = (
            21879
            / 490
            * (
                quat_mult(cs.sqrt(alpha_norm) * (numden / cs.norm_2(numden)), Qo4)
                - (
                    (1 / 442) * A0
                    + (5 / 663) * A1
                    + (35 / 2431) * A2
                    + (49 / 2431) * A3
                    + (49 / 2431) * A5
                    + (35 / 2431) * A6
                    + (5 / 663) * A7
                    + (1 / 442) * A8
                )
            )
        )

        f_A = cs.Function(
            "f_A",
            [
                pb,
                pe,
                vb,
                ve,
                ab,
                ae,
                jb,
                je,
                sb,
                se,
                o0,
                t1,
                t2,
                t3,
                o4,
                t5,
                t6,
                t7,
                o8,
            ],
            [A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [
                "pb",
                "pe",
                "vb",
                "ve",
                "ab",
                "ae",
                "jb",
                "je",
                "sb",
                "se",
                "o0",
                "t1",
                "t2",
                "t3",
                "o4",
                "t5",
                "t6",
                "t7",
                "o8",
            ],
            ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
        )

        # ------------- quaternion coefficients to quaternion polynomial ------------- #

        xi = cs.MX.sym("xi")  # path parameter
        A0 = cs.MX.sym("A0", 4)  # quat. coeff 0
        A1 = cs.MX.sym("A1", 4)  # quat. coeff 1
        A2 = cs.MX.sym("A2", 4)  # quat. coeff 2
        A3 = cs.MX.sym("A3", 4)  # quat. coeff 3
        A4 = cs.MX.sym("A4", 4)  # quat. coeff 4
        A5 = cs.MX.sym("A5", 4)  # quat. coeff 5
        A6 = cs.MX.sym("A6", 4)  # quat. coeff 6
        A7 = cs.MX.sym("A7", 4)  # quat. coeff 7
        A8 = cs.MX.sym("A8", 4)  # quat. coeff 8
        Zcoeff = cs.horzcat(A0, A1, A2, A3, A4, A5, A6, A7, A8).T

        U = Bezier(xi, Zcoeff[:, 0], 8)
        V = Bezier(xi, Zcoeff[:, 1], 8)
        G = Bezier(xi, Zcoeff[:, 2], 8)
        H = Bezier(xi, Zcoeff[:, 3], 8)

        Z = cs.vertcat(U, V, G, H)
        Z_d = cs.jacobian(Z, xi)
        Z_dd = cs.jacobian(Z_d, xi)
        Z_ddd = cs.jacobian(Z_dd, xi)

        f_Z = cs.Function(
            "f_Z",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [Z],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["Z"],
        )

        # ----------------------------- parametric speed ----------------------------- #
        sigma = Z[0] ** 2 + Z[1] ** 2 + Z[2] ** 2 + Z[3] ** 2

        f_sigma = cs.Function(
            "f_sigma",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [sigma],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["sigma"],
        )

        # ------------------- quaternion control points to position ------------------ #

        # hodograph controlpoints
        h0 = AstB(A0, A0)
        h1 = AstB(A0, A1)
        h2 = (1 / 15) * (7 * AstB(A0, A2) + 8 * AstB(A1, A1))
        h3 = (1 / 10) * (2 * AstB(A0, A3) + 8 * AstB(A1, A2))
        h4 = (1 / 65) * (5 * AstB(A0, A4) + 32 * AstB(A1, A3) + 28 * AstB(A2, A2))
        h5 = (1 / 39) * (AstB(A0, A5) + 10 * AstB(A1, A4) + 28 * AstB(A2, A3))
        h6 = (1 / 143) * (
            AstB(A0, A6) + 16 * AstB(A1, A5) + 70 * AstB(A2, A4) + 56 * AstB(A3, A3)
        )
        h7 = (1 / 715) * (
            AstB(A0, A7) + 28 * AstB(A1, A6) + 196 * AstB(A2, A5) + 490 * AstB(A3, A4)
        )
        h8 = (1 / 6435) * (
            AstB(A0, A8)
            + 64 * AstB(A1, A7)
            + 784 * AstB(A2, A6)
            + 3136 * AstB(A3, A5)
            + 2450 * AstB(A4, A4)
        )
        h9 = (1 / 715) * (
            AstB(A1, A8) + 28 * AstB(A2, A7) + 196 * AstB(A3, A6) + 490 * AstB(A4, A5)
        )
        h10 = (1 / 143) * (
            AstB(A2, A8) + 16 * AstB(A3, A7) + 70 * AstB(A4, A6) + 56 * AstB(A5, A5)
        )
        h11 = (1 / 39) * (AstB(A3, A8) + 10 * AstB(A4, A7) + 28 * AstB(A5, A6))
        h12 = (1 / 65) * (5 * AstB(A4, A8) + 32 * AstB(A5, A7) + 28 * AstB(A6, A6))
        h13 = (1 / 10) * (2 * AstB(A5, A8) + 8 * AstB(A6, A7))
        h14 = (1 / 15) * (7 * AstB(A6, A8) + 8 * AstB(A7, A7))
        h15 = AstB(A7, A8)
        h16 = AstB(A8, A8)
        hodograph = [
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            h6,
            h7,
            h8,
            h9,
            h10,
            h11,
            h12,
            h13,
            h14,
            h15,
            h16,
        ]

        # curve controlpoints
        p0 = cs.MX.sym("p0", 3)
        p = cs.MX.zeros(18, 3)
        p[0, :] = p0
        for j in range(1, 18):
            # sum_hodograph = cs.MX.zeros(1,3)
            # for i in range(j):
            #    sum_hodograph += hodograph[i][1:,0].T
            p[j, :] = p[j - 1, :] + 1 / 17 * hodograph[j - 1][1:, 0].T

        f_h = cs.Function("f_h", [p0, A0, A1, A2, A3, A4, A5, A6, A7, A8], hodograph)
        f_p = cs.Function(
            "f_p",
            [p0, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [p],
            ["p0", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["p"],
        )

        # curve
        gamma = Bezier(xi, p, 17).T
        f_gamma = cs.Function(
            "f_gamma",
            [xi, p0, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [gamma],
            ["xi", "p0", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["gamma"],
        )
        # ------------------------------- adapted frame ------------------------------ #
        sigma = Z[0] ** 2 + Z[1] ** 2 + Z[2] ** 2 + Z[3] ** 2
        e1 = (quat_AiconjA(Z, "i") / sigma)[1:]
        e2 = (quat_AiconjA(Z, "j") / sigma)[1:]
        e3 = (quat_AiconjA(Z, "k") / sigma)[1:]
        erf = cs.horzcat(e1, e2, e3)

        f_erf = cs.Function(
            "f_erf",
            [xi, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [erf],
            ["xi", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            ["erf"],
        )

        # -------------------------- dictionary of functions ------------------------- #
        func_dict = {
            "Z": f_Z,
            "A": f_A,
            "gamma": f_gamma,
            "hodograph": f_h,
            "sigma": f_sigma,
            "erf": f_erf,
        }
    return func_dict


def parametric_path_functions(n, parametric_segment_funcs, interp_order):
    """Generates symbolic functions for C2 or C4 hermite interpolation of PH curves
    interp_order: Order of hermite interpolation (2 or 4)

    NOTE: These symbolic functions are FOR THE ENTIRE CURVE.
    """
    n_segments = 2**n

    XI = cs.MX.sym("xi")
    XI_grid = cs.MX.sym("xi_grid", n_segments + 1)
    TR = cs.MX.sym("tr", n_segments, 3)
    A0 = cs.MX.sym("A0", n_segments, 4)
    A1 = cs.MX.sym("A1", n_segments, 4)
    A2 = cs.MX.sym("A2", n_segments, 4)
    A3 = cs.MX.sym("A3", n_segments, 4)
    A4 = cs.MX.sym("A4", n_segments, 4)
    if interp_order == 4:
        A5 = cs.MX.sym("A5", n_segments, 4)
        A6 = cs.MX.sym("A6", n_segments, 4)
        A7 = cs.MX.sym("A7", n_segments, 4)
        A8 = cs.MX.sym("A8", n_segments, 4)

    ind = cs.find(cs.sign(XI_grid - XI) + 1) - 1  # TODO: Not robust if XI=0

    xi_i = (XI - XI_grid[ind]) / (XI_grid[ind + 1] - XI_grid[ind])
    A0_i = A0[ind, :]
    A1_i = A1[ind, :]
    A2_i = A2[ind, :]
    A3_i = A3[ind, :]
    A4_i = A4[ind, :]
    if interp_order == 4:
        A5_i = A5[ind, :]
        A6_i = A6[ind, :]
        A7_i = A7[ind, :]
        A8_i = A8[ind, :]

    if interp_order == 2:
        gamma = (
            parametric_segment_funcs["gamma"](
                xi_i,
                cs.DM.zeros(3),
                A0_i,
                A1_i,
                A2_i,
                A3_i,
                A4_i,
            )
            + TR[ind, :].T
        )
        gamma_d = cs.jacobian(gamma, XI)
        gamma_dd = cs.jacobian(gamma_d, XI)
        gamma_ddd = cs.jacobian(gamma_dd, XI)
        gamma_dddd = cs.jacobian(gamma_ddd, XI)

        Z = parametric_segment_funcs["Z"](xi_i, A0_i, A1_i, A2_i, A3_i, A4_i)
        Z_d = cs.jacobian(Z, XI)
        Z_dd = cs.jacobian(Z_d, XI)
        Z_ddd = cs.jacobian(Z_dd, XI)

        R_erf = parametric_segment_funcs["erf"](xi_i, A0_i, A1_i, A2_i, A3_i, A4_i)
        R_erf_d = cs.jacobian(R_erf, XI).reshape((3, 3))
        R_erf_dd = cs.jacobian(R_erf_d, XI).reshape((3, 3))
        R_erf_ddd = cs.jacobian(R_erf_dd, XI).reshape((3, 3))

        e1 = R_erf[:, 0]
        e2 = R_erf[:, 1]
        e3 = R_erf[:, 2]
        e1_d = R_erf_d[:, 0]
        e2_d = R_erf_d[:, 1]
        e3_d = R_erf_d[:, 2]
        X1 = cs.dot(e2_d, e3)
        X2 = cs.dot(e3_d, e1)
        X3 = cs.dot(e1_d, e2)
        # # omega =  X1*e1+X2*e2+X3*e3
        X = cs.horzcat(X1, X2, X3)
        X_d = cs.jacobian(X, XI)
        X_dd = cs.jacobian(X_d, XI)
        ######
        skew_symm_X = R_erf.T @ R_erf_d
        X_v2 = cs.vertcat(skew_symm_X[2, 1], skew_symm_X[0, 2], skew_symm_X[1, 0])
        ######

        sigma = parametric_segment_funcs["sigma"](xi_i, A0_i, A1_i, A2_i, A3_i, A4_i)
        sigma_d = cs.jacobian(sigma, XI)
        sigma_dd = cs.jacobian(sigma_d, XI)
        sigma_ddd = cs.jacobian(sigma_dd, XI)

        f_Gamma = cs.Function(
            "f_Gamma",
            [XI, XI_grid, TR, A0, A1, A2, A3, A4],
            [gamma, gamma_d, gamma_dd, gamma_ddd, gamma_dddd],
        )
        f_Preimage = cs.Function(
            "f_Z", [XI, XI_grid, A0, A1, A2, A3, A4], [Z, Z_d, Z_dd, Z_ddd]
        )
        f_Rerf = cs.Function(
            "f_Rerf",
            [XI, XI_grid, A0, A1, A2, A3, A4],
            [R_erf, R_erf_d, R_erf_dd, R_erf_ddd],
        )
        f_X = cs.Function(
            "f_X",
            [XI, XI_grid, A0, A1, A2, A3, A4],
            [X, X_d, X_dd, X_v2],
        )
        f_Sigma = cs.Function(
            "f_Sigma",
            [XI, XI_grid, A0, A1, A2, A3, A4],
            [sigma, sigma_d, sigma_dd, sigma_ddd],
        )

    elif interp_order == 4:
        gamma = (
            parametric_segment_funcs["gamma"](
                xi_i,
                cs.DM.zeros(3),
                A0_i,
                A1_i,
                A2_i,
                A3_i,
                A4_i,
                A5_i,
                A6_i,
                A7_i,
                A8_i,
            )
            + TR[ind, :].T
        )
        gamma_d = cs.jacobian(gamma, XI)
        gamma_dd = cs.jacobian(gamma_d, XI)
        gamma_ddd = cs.jacobian(gamma_dd, XI)
        gamma_dddd = cs.jacobian(gamma_ddd, XI)

        Z = parametric_segment_funcs["Z"](
            xi_i, A0_i, A1_i, A2_i, A3_i, A4_i, A5_i, A6_i, A7_i, A8_i
        )
        Z_d = cs.jacobian(Z, XI)
        Z_dd = cs.jacobian(Z_d, XI)
        Z_ddd = cs.jacobian(Z_dd, XI)

        R_erf = parametric_segment_funcs["erf"](
            xi_i, A0_i, A1_i, A2_i, A3_i, A4_i, A5_i, A6_i, A7_i, A8_i
        )
        R_erf_d = cs.jacobian(R_erf, XI).reshape((3, 3))
        R_erf_dd = cs.jacobian(R_erf_d, XI).reshape((3, 3))
        R_erf_ddd = cs.jacobian(R_erf_dd, XI).reshape((3, 3))

        e1 = R_erf[:, 0]
        e2 = R_erf[:, 1]
        e3 = R_erf[:, 2]
        e1_d = R_erf_d[:, 0]
        e2_d = R_erf_d[:, 1]
        e3_d = R_erf_d[:, 2]
        X1 = cs.dot(e2_d, e3)
        X2 = cs.dot(e3_d, e1)
        X3 = cs.dot(e1_d, e2)
        # # omega =  X1*e1+X2*e2+X3*e3
        X = cs.horzcat(X1, X2, X3)
        X_d = cs.jacobian(X, XI)
        X_dd = cs.jacobian(X_d, XI)
        ######
        skew_symm_X = R_erf.T @ R_erf_d
        X_v2 = cs.vertcat(skew_symm_X[2, 1], skew_symm_X[0, 2], skew_symm_X[1, 0])
        ######

        sigma = parametric_segment_funcs["sigma"](
            xi_i, A0_i, A1_i, A2_i, A3_i, A4_i, A5_i, A6_i, A7_i, A8_i
        )
        sigma_d = cs.jacobian(sigma, XI)
        sigma_dd = cs.jacobian(sigma_d, XI)
        sigma_ddd = cs.jacobian(sigma_dd, XI)

        f_Gamma = cs.Function(
            "f_Gamma",
            [XI, XI_grid, TR, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [gamma, gamma_d, gamma_dd, gamma_ddd, gamma_dddd],
        )
        f_Preimage = cs.Function(
            "f_Z",
            [XI, XI_grid, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [Z, Z_d, Z_dd, Z_ddd],
        )
        f_Rerf = cs.Function(
            "f_Rerf",
            [XI, XI_grid, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [R_erf, R_erf_d, R_erf_dd, R_erf_ddd],
        )
        f_X = cs.Function(
            "f_X",
            [XI, XI_grid, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [X, X_d, X_dd, X_v2],
        )
        f_Sigma = cs.Function(
            "f_Sigma",
            [XI, XI_grid, A0, A1, A2, A3, A4, A5, A6, A7, A8],
            [sigma, sigma_d, sigma_dd, sigma_ddd],
        )

    parametric_funcs = {
        "gamma": f_Gamma,
        "A": f_Preimage,
        "sigma": f_Sigma,
        "erf": f_Rerf,
        "X": f_X,
    }
    return parametric_funcs


def visualize_parameterization(nominal_path, parametric_path):
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["r", "g", "b"])

    # ------------------------------- 3D, gamma, A ------------------------------- #
    fig = plt.figure()
    # 3d visualization
    ax = fig.add_subplot(131, projection="3d")
    ax.plot(
        parametric_path["p"][:, 0],
        parametric_path["p"][:, 1],
        parametric_path["p"][:, 2],
        "b-",
    )
    ax.plot(
        nominal_path["p"][:, 0], nominal_path["p"][:, 1], nominal_path["p"][:, 2], "k--"
    )
    ax = plot_frames(
        parametric_path["p"],
        parametric_path["erf"][:, :, 0],
        parametric_path["erf"][:, :, 1],
        parametric_path["erf"][:, :, 2],
        interval=0.99,
        scale=0.25,
        ax=ax,
    )
    ax = axis_equal(
        parametric_path["p"][:, 0],
        parametric_path["p"][:, 1],
        parametric_path["p"][:, 2],
        ax=ax,
    )

    # position
    ax = fig.add_subplot(532)
    ax.plot(parametric_path["xi"], parametric_path["p"])
    ax.plot(nominal_path["xi"], nominal_path["p"], "k--")
    ax.set_ylabel("p")

    ax = fig.add_subplot(535)
    ax.plot(parametric_path["xi"], parametric_path["v"])
    ax.plot(nominal_path["xi"], nominal_path["v"], "k--")
    ax.set_ylabel("v")

    ax = fig.add_subplot(538)
    ax.plot(parametric_path["xi"], parametric_path["a"])
    ax.plot(nominal_path["xi"], nominal_path["a"], "k--")
    ax.set_ylabel("a")

    ax = fig.add_subplot(5, 3, 11)
    ax.plot(parametric_path["xi"], parametric_path["j"])
    ax.plot(nominal_path["xi"], nominal_path["j"], "k--")
    ax.set_ylabel("j")

    ax = fig.add_subplot(5, 3, 14)
    ax.plot(parametric_path["xi"], parametric_path["s"])
    ax.plot(nominal_path["xi"], nominal_path["s"], "k--")
    ax.set_ylabel("s")

    # preimage
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["k", "r", "g", "b"])

    ax = fig.add_subplot(433)
    ax.plot(parametric_path["xi"], parametric_path["A"])
    ax.set_ylabel(r"$A$")

    ax = fig.add_subplot(436)
    ax.plot(parametric_path["xi"], parametric_path["Ad"])
    ax.set_ylabel(r"$A^'$")

    ax = fig.add_subplot(439)
    ax.plot(parametric_path["xi"], parametric_path["Add"])
    ax.set_ylabel(r"$A^{''}$")

    ax = fig.add_subplot(4, 3, 12)
    ax.plot(parametric_path["xi"], parametric_path["Addd"])
    ax.set_ylabel(r"$A^{'''}$")

    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["r", "g", "b"])
    # ---------------------------------- ERF, X ---------------------------------- #
    fig = plt.figure()

    ax = fig.add_subplot(451)
    ax.plot(parametric_path["xi"], parametric_path["erf"][:, :, 0], ".")
    ax.set_ylabel(r"$e_1$")
    ax = fig.add_subplot(452)
    ax.plot(parametric_path["xi"], parametric_path["erf"][:, :, 1], ".")
    ax.set_ylabel(r"$e_2$")
    ax = fig.add_subplot(453)
    ax.plot(parametric_path["xi"], parametric_path["erf"][:, :, 2], ".")
    ax.set_ylabel(r"$e_3$")
    ax = fig.add_subplot(454)
    ax.plot(parametric_path["xi"], parametric_path["sigma"])
    ax.set_ylabel(r"$\sigma$")
    ax = fig.add_subplot(455)
    ax.plot(parametric_path["xi"], parametric_path["X"])
    ax.plot(parametric_path["xi"], parametric_path["Xv2"], "--")
    ax.set_ylabel(r"$\chi$")

    ax = fig.add_subplot(456)
    ax.plot(parametric_path["xi"], parametric_path["erfd"][:, :, 0])
    ax.set_ylabel(r"$e_1^'$")
    ax = fig.add_subplot(457)
    ax.plot(parametric_path["xi"], parametric_path["erfd"][:, :, 1])
    ax.set_ylabel(r"$e_2^'$")
    ax = fig.add_subplot(458)
    ax.plot(parametric_path["xi"], parametric_path["erfd"][:, :, 2])
    ax.set_ylabel(r"$e_3^'$")
    ax = fig.add_subplot(459)
    ax.plot(parametric_path["xi"], parametric_path["sigmad"])
    ax.set_ylabel(r"$\sigma^'$")
    ax = fig.add_subplot(4, 5, 10)
    ax.plot(parametric_path["xi"], parametric_path["Xd"])
    ax.set_ylabel(r"$\chi^'$")

    ax = fig.add_subplot(4, 5, 11)
    ax.plot(parametric_path["xi"], parametric_path["erfdd"][:, :, 0])
    ax.set_ylabel(r"$e_1^{''}$")
    ax = fig.add_subplot(4, 5, 12)
    ax.plot(parametric_path["xi"], parametric_path["erfdd"][:, :, 1])
    ax.set_ylabel(r"$e_2^{''}$")
    ax = fig.add_subplot(4, 5, 13)
    ax.plot(parametric_path["xi"], parametric_path["erfdd"][:, :, 2])
    ax.set_ylabel(r"$e_3^{''}$")
    ax = fig.add_subplot(4, 5, 14)
    ax.plot(parametric_path["xi"], parametric_path["sigmadd"])
    ax.set_ylabel(r"$\sigma^{''}$")
    ax = fig.add_subplot(4, 5, 15)
    ax.plot(parametric_path["xi"], parametric_path["Xdd"])
    ax.set_ylabel(r"$\chi^{''}$")

    ax = fig.add_subplot(4, 5, 16)
    ax.plot(parametric_path["xi"], parametric_path["erfddd"][:, :, 0])
    ax.set_ylabel(r"$e_1^{'''}$")
    ax = fig.add_subplot(4, 5, 17)
    ax.plot(parametric_path["xi"], parametric_path["erfddd"][:, :, 1])
    ax.set_ylabel(r"$e_2^{'''}$")
    ax = fig.add_subplot(4, 5, 18)
    ax.plot(parametric_path["xi"], parametric_path["erfddd"][:, :, 2])
    ax.set_ylabel(r"$e_3^{'''}$")
    ax = fig.add_subplot(4, 5, 19)
    ax.plot(parametric_path["xi"], parametric_path["sigmaddd"])
    ax.set_ylabel(r"$\sigma^{'''}$")

    plt.show()


def nominal_path(equation=None, path_parameter=None):
    if equation is None:
        xi = cs.MX.sym("xi")
        r = cs.vertcat(1.5 * cs.sin(7.2 * xi), cs.cos(9 * xi), cs.exp(cs.cos(1.8 * xi)))
    else:
        xi = path_parameter
        r = equation
    dr = cs.jacobian(r, xi)  # first derivative of analytical curve
    ddr = cs.jacobian(dr, xi)  # second derivative of analytical curve
    dddr = cs.jacobian(ddr, xi)  # third derivative of analytical curve
    ddddr = cs.jacobian(dddr, xi)  # fourth derivative of analytical curve

    f_p = cs.Function(
        "f_p", [xi], [r, dr, ddr, dddr, ddddr], ["xi"], ["p", "v", "a", "j", "s"]
    )

    no_path = {"f_p": f_p}  # , "f_v": v, "f_a": a}

    return no_path
    # return 1e16 * p, 1e16 * v, 1e16 * a


def parameterize_path(nominal_path, parametric_funcs, n, interp_order, rotate_erf=True):
    n_segments = 2**n
    xi_grid = np.zeros(n_segments + 1)
    for i in range(n_segments):
        xi_grid[i + 1] = (i + 1) / (n_segments)
    # delta_xi = xi_grid[1]

    A0 = np.zeros((n_segments, 4))
    A1 = np.zeros((n_segments, 4))
    A2 = np.zeros((n_segments, 4))
    A3 = np.zeros((n_segments, 4))
    A4 = np.zeros((n_segments, 4))
    if interp_order == 4:
        A5 = np.zeros((n_segments, 4))
        A6 = np.zeros((n_segments, 4))
        A7 = np.zeros((n_segments, 4))
        A8 = np.zeros((n_segments, 4))

    translation = np.zeros((n_segments, 3))
    rotation = np.zeros((n_segments, 3, 3))

    h = []
    for m in range(n_segments):
        # get hermite data
        pb = np.squeeze(nominal_path["f_p"](xi=xi_grid[m])["p"])
        pe = np.squeeze(nominal_path["f_p"](xi=xi_grid[m + 1])["p"])
        vb = np.squeeze(nominal_path["f_p"](xi=xi_grid[m])["v"])
        ve = np.squeeze(nominal_path["f_p"](xi=xi_grid[m + 1])["v"])
        ab = np.squeeze(nominal_path["f_p"](xi=xi_grid[m])["a"])
        ae = np.squeeze(nominal_path["f_p"](xi=xi_grid[m + 1])["a"])
        jb = np.squeeze(nominal_path["f_p"](xi=xi_grid[m])["j"])
        je = np.squeeze(nominal_path["f_p"](xi=xi_grid[m + 1])["j"])
        sb = np.squeeze(nominal_path["f_p"](xi=xi_grid[m])["s"])
        se = np.squeeze(nominal_path["f_p"](xi=xi_grid[m + 1])["s"])

        delta_xi = xi_grid[m + 1] - xi_grid[m]
        vb = vb * delta_xi
        ve = ve * delta_xi
        ab = ab * (delta_xi**2)
        ae = ae * (delta_xi**2)
        jb = jb * (delta_xi**3)
        je = je * (delta_xi**3)
        sb = sb * (delta_xi**4)
        se = se * (delta_xi**4)

        # choice of interpolants
        if interp_order == 2:
            o0 = 0.0
            o2 = 0.0
            o4 = 0.0
            t1 = 0.0
            t3 = 0.0
        elif interp_order == 4:
            o0 = 0.0
            t1 = 0.0
            t2 = 0.0
            t3 = 0.0
            o4 = 0.0
            t5 = 0.0
            t6 = 0.0
            t7 = 0.0
            o8 = 0.0

        # translate hermite data into standard position
        translation[m, :] = np.squeeze(pb)
        rotation[m, :, :] = tangent_to_rotation(np.squeeze(vb + ve))
        pb = pb - translation[m, :]
        pe = pe - translation[m, :]
        pb = np.zeros(3)  # np.matmul(np.squeeze(pb), rotation[m, :, :])
        pe = np.matmul(np.squeeze(pe), rotation[m, :, :])
        vb = np.matmul(np.squeeze(vb), rotation[m, :, :])
        ve = np.matmul(np.squeeze(ve), rotation[m, :, :])
        ab = np.matmul(np.squeeze(ab), rotation[m, :, :])
        ae = np.matmul(np.squeeze(ae), rotation[m, :, :])
        jb = np.matmul(np.squeeze(jb), rotation[m, :, :])
        je = np.matmul(np.squeeze(je), rotation[m, :, :])
        sb = np.matmul(np.squeeze(sb), rotation[m, :, :])
        se = np.matmul(np.squeeze(se), rotation[m, :, :])

        # compute quaternion polynomial coefficients
        if interp_order == 2:
            a0, a1, a2, a3, a4 = parametric_funcs["A"](
                pb, pe, vb, ve, ab, ae, o0, o2, o4, t1, t3
            )
            A0[m, :] = np.squeeze(a0)
            A1[m, :] = np.squeeze(a1)
            A2[m, :] = np.squeeze(a2)
            A3[m, :] = np.squeeze(a3)
            A4[m, :] = np.squeeze(a4)
        elif interp_order == 4:
            a0, a1, a2, a3, a4, a5, a6, a7, a8 = parametric_funcs["A"](
                pb,
                pe,
                vb,
                ve,
                ab,
                ae,
                jb,
                je,
                sb,
                se,
                o0,
                t1,
                t2,
                t3,
                o4,
                t5,
                t6,
                t7,
                o8,
            )
            A0[m, :] = np.squeeze(a0)
            A1[m, :] = np.squeeze(a1)
            A2[m, :] = np.squeeze(a2)
            A3[m, :] = np.squeeze(a3)
            A4[m, :] = np.squeeze(a4)
            A5[m, :] = np.squeeze(a5)
            A6[m, :] = np.squeeze(a6)
            A7[m, :] = np.squeeze(a7)
            A8[m, :] = np.squeeze(a8)

    # rotate back the quaternion polynomial from the standard from to original
    q_rot = []
    A0_rot = np.zeros((n_segments, 4))
    A1_rot = np.zeros((n_segments, 4))
    A2_rot = np.zeros((n_segments, 4))
    A3_rot = np.zeros((n_segments, 4))
    A4_rot = np.zeros((n_segments, 4))
    A5_rot = np.zeros((n_segments, 4))
    A6_rot = np.zeros((n_segments, 4))
    A7_rot = np.zeros((n_segments, 4))
    A8_rot = np.zeros((n_segments, 4))
    for m in range(n_segments):
        # rotate back to original pose
        q_rot += [quat_conj(rotation_to_quaternion(rotation[m, :, :].T))]
        A0_rot[m, :] = quat_mult(q_rot[m], A0[m])
        A1_rot[m, :] = quat_mult(q_rot[m], A1[m])
        A2_rot[m, :] = quat_mult(q_rot[m], A2[m])
        A3_rot[m, :] = quat_mult(q_rot[m], A3[m])
        A4_rot[m, :] = quat_mult(q_rot[m], A4[m])
        if interp_order == 4:
            A5_rot[m, :] = quat_mult(q_rot[m], A5[m])
            A6_rot[m, :] = quat_mult(q_rot[m], A6[m])
            A7_rot[m, :] = quat_mult(q_rot[m], A7[m])
            A8_rot[m, :] = quat_mult(q_rot[m], A8[m])

        # rotate adapted frame around e1 (roll) for continuity in e2 and e3
        if rotate_erf:
            (
                A0_rot,
                A1_rot,
                A2_rot,
                A3_rot,
                A4_rot,
                A5_rot,
                A6_rot,
                A7_rot,
                A8_rot,
            ) = erf_continuity(
                A0_rot=A0_rot.copy(),
                A1_rot=A1_rot.copy(),
                A2_rot=A2_rot.copy(),
                A3_rot=A3_rot.copy(),
                A4_rot=A4_rot.copy(),
                A5_rot=A5_rot.copy(),
                A6_rot=A6_rot.copy(),
                A7_rot=A7_rot.copy(),
                A8_rot=A8_rot.copy(),
                interp_order=interp_order,
                m=m,
                parametric_funcs=parametric_funcs,
            )

        # flip coefficients for preimage continuity
        (
            A0_rot,
            A1_rot,
            A2_rot,
            A3_rot,
            A4_rot,
            A5_rot,
            A6_rot,
            A7_rot,
            A8_rot,
        ) = preimage_continuity(
            A0_rot=A0_rot.copy(),
            A1_rot=A1_rot.copy(),
            A2_rot=A2_rot.copy(),
            A3_rot=A3_rot.copy(),
            A4_rot=A4_rot.copy(),
            A5_rot=A5_rot.copy(),
            A6_rot=A6_rot.copy(),
            A7_rot=A7_rot.copy(),
            A8_rot=A8_rot.copy(),
            interp_order=interp_order,
            m=m,
            parametric_funcs=parametric_funcs,
        )

    pa_path = {
        "A0": A0_rot,
        "A1": A1_rot,
        "A2": A2_rot,
        "A3": A3_rot,
        "A4": A4_rot,
        "A5": A5_rot,
        "A6": A6_rot,
        "A7": A7_rot,
        "A8": A8_rot,
        "xi_grid": xi_grid,
        "rotation": rotation,
        "translation": translation,
        "q_rot": q_rot,
    }

    return pa_path


def evaluate_parameterization(
    nominal_path, parametric_path, parametric_funcs, n_eval, interp_order
):
    no_p = np.zeros((n_eval, 3))
    no_v = np.zeros((n_eval, 3))
    no_a = np.zeros((n_eval, 3))
    no_j = np.zeros((n_eval, 3))
    no_s = np.zeros((n_eval, 3))

    pa_p = np.zeros((n_eval, 3))
    pa_v = np.zeros((n_eval, 3))
    pa_a = np.zeros((n_eval, 3))
    pa_j = np.zeros((n_eval, 3))
    pa_s = np.zeros((n_eval, 3))
    pa_A = np.zeros((n_eval, 4))
    pa_Ad = np.zeros((n_eval, 4))
    pa_Add = np.zeros((n_eval, 4))
    pa_Addd = np.zeros((n_eval, 4))
    pa_erf = np.zeros((n_eval, 3, 3))
    pa_erfd = np.zeros((n_eval, 3, 3))
    pa_erfdd = np.zeros((n_eval, 3, 3))
    pa_erfddd = np.zeros((n_eval, 3, 3))
    pa_X = np.zeros((n_eval, 3))
    pa_Xd = np.zeros((n_eval, 3))
    pa_Xdd = np.zeros((n_eval, 3))
    pa_Xv2 = np.zeros((n_eval, 3))
    pa_sigma = np.zeros(n_eval)
    pa_sigmad = np.zeros(n_eval)
    pa_sigmadd = np.zeros(n_eval)
    pa_sigmaddd = np.zeros(n_eval)

    dists = np.zeros((n_eval))

    # loop over evaluation parametric path
    xi_eval = np.linspace(1e-6, 1, n_eval)  # path parameter values to evaluate on
    for k, xik in enumerate(xi_eval):
        # parametric path
        if interp_order == 2:
            p, v, a, j, s = parametric_funcs["gamma"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["translation"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
            )

            A, A_d, A_dd, A_ddd = parametric_funcs["A"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
            )

            erf, erf_d, erf_dd, erf_ddd = parametric_funcs["erf"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
            )

            X, X_d, X_dd, X_v2 = parametric_funcs["X"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
            )

            sigma, sigma_d, sigma_dd, sigma_ddd = parametric_funcs["sigma"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
            )
        elif interp_order == 4:
            p, v, a, j, s = parametric_funcs["gamma"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["translation"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
                parametric_path["A5"],
                parametric_path["A6"],
                parametric_path["A7"],
                parametric_path["A8"],
            )

            A, A_d, A_dd, A_ddd = parametric_funcs["A"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
                parametric_path["A5"],
                parametric_path["A6"],
                parametric_path["A7"],
                parametric_path["A8"],
            )

            erf, erf_d, erf_dd, erf_ddd = parametric_funcs["erf"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
                parametric_path["A5"],
                parametric_path["A6"],
                parametric_path["A7"],
                parametric_path["A8"],
            )

            X, X_d, X_dd, X_v2 = parametric_funcs["X"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
                parametric_path["A5"],
                parametric_path["A6"],
                parametric_path["A7"],
                parametric_path["A8"],
            )

            sigma, sigma_d, sigma_dd, sigma_ddd = parametric_funcs["sigma"](
                xik,
                parametric_path["xi_grid"],
                parametric_path["A0"],
                parametric_path["A1"],
                parametric_path["A2"],
                parametric_path["A3"],
                parametric_path["A4"],
                parametric_path["A5"],
                parametric_path["A6"],
                parametric_path["A7"],
                parametric_path["A8"],
            )
        pa_p[k, :] = np.squeeze([p])
        pa_v[k, :] = np.squeeze([v])
        pa_a[k, :] = np.squeeze([a])
        pa_j[k, :] = np.squeeze([j])
        pa_s[k, :] = np.squeeze([s])

        pa_A[k, :] = np.squeeze([A])
        pa_Ad[k, :] = np.squeeze([A_d])
        pa_Add[k, :] = np.squeeze([A_dd])
        pa_Addd[k, :] = np.squeeze([A_ddd])

        pa_erf[k, :, :] = np.squeeze([erf])
        pa_erfd[k, :, :] = np.squeeze([erf_d])
        pa_erfdd[k, :, :] = np.squeeze([erf_dd])
        pa_erfddd[k, :, :] = np.squeeze([erf_ddd])

        pa_X[k, :] = np.squeeze([X])
        pa_Xd[k, :] = np.squeeze([X_d])
        pa_Xdd[k, :] = np.squeeze([X_dd])
        pa_Xv2[k, :] = np.squeeze([X_v2])

        pa_sigma[k] = np.squeeze([sigma])
        pa_sigmad[k] = np.squeeze([sigma_d])
        pa_sigmadd[k] = np.squeeze([sigma_dd])
        pa_sigmaddd[k] = np.squeeze([sigma_ddd])

        # nominal path
        no_p[k, :] = np.squeeze(nominal_path["f_p"](xi=xik)["p"])
        no_v[k, :] = np.squeeze(nominal_path["f_p"](xi=xik)["v"])
        no_a[k, :] = np.squeeze(nominal_path["f_p"](xi=xik)["a"])
        no_j[k, :] = np.squeeze(nominal_path["f_p"](xi=xik)["j"])
        no_s[k, :] = np.squeeze(nominal_path["f_p"](xi=xik)["s"])

        # difference between original and ph curves
        dists[k] = np.linalg.norm(no_p[k, :] - pa_p[k, :])

    # calculate the error: maximum distance between analytical curve and PH spline
    error = max(dists)

    # store results
    nominal_path["xi"] = xi_eval
    nominal_path["p"] = no_p
    nominal_path["v"] = no_v
    nominal_path["a"] = no_a
    nominal_path["j"] = no_j
    nominal_path["s"] = no_s

    parametric_path["xi"] = xi_eval
    parametric_path["p"] = pa_p
    parametric_path["v"] = pa_v
    parametric_path["a"] = pa_a
    parametric_path["j"] = pa_j
    parametric_path["s"] = pa_s
    parametric_path["erf"] = pa_erf
    parametric_path["erfd"] = pa_erfd
    parametric_path["erfdd"] = pa_erfdd
    parametric_path["erfddd"] = pa_erfddd
    parametric_path["X"] = pa_X
    parametric_path["Xd"] = pa_Xd
    parametric_path["Xdd"] = pa_Xdd
    parametric_path["Xv2"] = pa_Xv2
    parametric_path["A"] = pa_A
    parametric_path["Ad"] = pa_Ad
    parametric_path["Add"] = pa_Add
    parametric_path["Addd"] = pa_Addd
    parametric_path["sigma"] = pa_sigma
    parametric_path["sigmad"] = pa_sigmad
    parametric_path["sigmadd"] = pa_sigmadd
    parametric_path["sigmaddd"] = pa_sigmaddd

    return nominal_path, parametric_path, error


def phodcos(n, interp_order, N, no_path=None, pa_segment_funcs=None, rotate_erf=True):
    # nominal path
    if no_path is None:
        no_path = nominal_path()

    # segment functions
    if pa_segment_funcs is None:
        pa_segment_funcs = parametric_segment_path_functions(interp_order=interp_order)

    # parameterize path
    pa_path = parameterize_path(
        nominal_path=no_path,
        parametric_funcs=pa_segment_funcs,
        n=n,
        interp_order=interp_order,
        rotate_erf=rotate_erf,
    )

    # obtain parametric path functions
    pa_funcs = parametric_path_functions(
        n=n, parametric_segment_funcs=pa_segment_funcs, interp_order=interp_order
    )

    # evaluate parameterization
    no_path, pa_path, error = evaluate_parameterization(
        nominal_path=no_path,
        parametric_path=pa_path,
        parametric_funcs=pa_funcs,
        n_eval=(2**n) * N,
        interp_order=interp_order,
    )

    return no_path, pa_path, error
