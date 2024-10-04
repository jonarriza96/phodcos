# %%
import argparse
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs

import pickle
import pyquaternion

from scipy.spatial.distance import cdist
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


from phodcos.utils.visualize import plot_frames, axis_equal
from phodcos.utils.utils import get_phodcos_path


x = cs.MX.sym("x", 9)
u = cs.MX.sym("u", 3)

dist_scaling = 389703
t_scaling = 382981


def spherical_to_cartesian(center, ang1, ang2, r):
    n = np.array(
        [np.sin(ang2) * np.cos(ang1), np.sin(ang2) * np.sin(ang1), np.cos(ang2)]
    )
    x = center[0] + r * np.sin(ang2) * np.cos(ang1)
    y = center[1] + r * np.sin(ang2) * np.sin(ang1)
    z = center[2] + r * np.cos(ang2)
    return np.array([x, y, z]), n


def plot_sphere(center, radius, color, ax=None, planar=False):
    def get_moon_surface(center, radius):

        # Create a meshgrid for the sphere
        u = np.linspace(0, 2 * np.pi, 200)
        v = np.linspace(0, np.pi, 200)

        # Parametric equations for a sphere
        xs = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        ys = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        zs = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        return xs, ys, zs

    # get moon
    xs, ys, zs = get_moon_surface(center, radius)

    # Plotting the sphere
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    if not planar:
        ax.plot_surface(xs, ys, zs, color=color)
    else:
        circle = plt.Circle((center[0], center[1]), radius, color=color, fill=True)
        ax.add_patch(circle)

    # Set plot labels
    ax.set_xlabel("X ")
    ax.set_ylabel("Y")
    if not planar:
        ax.set_zlabel("Z ")

    return ax


def generate_interpolators(ppr):
    """
    Ideally we would use, but their evaluation is slow. Therefore, we create
    interpolators to speed things up.

    Ideally, this should be replaced by:
    f_p, f_R, f_sigma, f_X = simplify_parametric_functions(ppr)
    """

    xi = ppr.parametric_path["xi"]

    p = ppr.parametric_path["p"]
    f_px = cs.interpolant("f_px", "bspline", [xi], p[:, 0])
    f_py = cs.interpolant("f_py", "bspline", [xi], p[:, 1])
    f_pz = cs.interpolant("f_pz", "bspline", [xi], p[:, 2])

    sigma = ppr.parametric_path["sigma"]
    f_sigma = cs.interpolant("f_sigma", "bspline", [xi], sigma)

    erf = ppr.parametric_path["erf"]
    f_e1y = cs.interpolant("fi_e1y", "bspline", [xi], erf[:, 1, 0])
    f_e1z = cs.interpolant("fi_e1y", "bspline", [xi], erf[:, 2, 0])
    f_e2x = cs.interpolant("fi_e2x", "bspline", [xi], erf[:, 0, 1])
    f_e2y = cs.interpolant("fi_e2y", "bspline", [xi], erf[:, 1, 1])
    f_e1x = cs.interpolant("fi_e1x", "bspline", [xi], erf[:, 0, 0])
    f_e2z = cs.interpolant("fi_e2z", "bspline", [xi], erf[:, 2, 1])
    f_e3x = cs.interpolant("fi_e2x", "bspline", [xi], erf[:, 0, 2])
    f_e3y = cs.interpolant("fi_e2y", "bspline", [xi], erf[:, 1, 2])
    f_e3z = cs.interpolant("fi_e2z", "bspline", [xi], erf[:, 2, 2])

    X = ppr.parametric_path["X"]
    f_X1 = cs.interpolant("f_X1", "bspline", [xi], X[:, 0])
    f_X2 = cs.interpolant("f_X2", "bspline", [xi], X[:, 1])
    f_X3 = cs.interpolant("f_X3", "bspline", [xi], X[:, 2])

    xi_cs = cs.MX.sym("xi")
    p = cs.vertcat(f_px(xi_cs), f_py(xi_cs), f_pz(xi_cs))
    v = cs.jacobian(p, xi_cs)
    a = cs.jacobian(v, xi_cs)
    e1 = cs.vertcat(f_e1x(xi_cs), f_e1y(xi_cs), f_e1z(xi_cs))
    e2 = cs.vertcat(f_e2x(xi_cs), f_e2y(xi_cs), f_e2z(xi_cs))
    e3 = cs.vertcat(f_e3x(xi_cs), f_e3y(xi_cs), f_e3z(xi_cs))
    R = cs.horzcat(e1, e2, e3)
    X = cs.vertcat(f_X1(xi_cs), f_X2(xi_cs), f_X3(xi_cs))

    f_p = cs.Function("f_p", [xi_cs], [p])
    f_v = cs.Function("f_v", [xi_cs], [v])
    f_a = cs.Function("f_a", [xi_cs], [a])
    f_sigma = cs.Function("f_sigma", [xi_cs], [f_sigma(xi_cs)])
    f_R = cs.Function("f_eq", [xi_cs], [R])
    f_X = cs.Function("f_X", [xi_cs], [X])

    return f_p, f_v, f_a, f_R, f_sigma, f_X


def rocket_gimbaled(x, u):

    # states and inputs
    p = x[:3]
    theta = x[3]
    phi = x[4]
    v = x[5]

    a = u[0]
    theta_rate = u[1]
    phi_rate = u[2]

    # equations of motion
    p_dot = v * cs.vertcat(
        cs.cos(phi) * cs.cos(theta), cs.sin(phi), cs.cos(phi) * cs.sin(theta)
    )
    theta_dot = theta_rate
    phi_dot = phi_rate
    v_dot = a

    x_dot = cs.vertcat(p_dot, theta_dot, phi_dot, v_dot)

    return x_dot


def forward_integrate(f, x, u, dt, RK4=False):

    if RK4:
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)

        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    else:
        x_next = x + dt * f(x, u)

    return x_next


def vec_to_gimbal(v):
    V = np.linalg.norm(v)
    n = v / V
    # phi = np.arcsin(n[1])
    # phi = np.arctan2(n[1], n[0])
    # if n[0] == 0:
    #     theta = np.arccos(n[0] / np.cos(phi))
    # else:
    theta = np.arctan2(n[2], n[0])
    if np.cos(theta) != 0:
        phi = np.arctan2(n[1], n[0] / np.cos(theta))
    elif np.sin(theta) != 0:
        phi = np.arctan2(n[1], n[2] / np.sin(theta))
    return V, theta, phi


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case_study",
        type=int,
        default=4,
        help="Case study (0, 1, 2, 3, 4)",
    )
    args = parser.parse_args()
    case_study = args.case_study

    # import parameterization
    path = get_phodcos_path() + "/examples/data/"
    with open(path + "nrho.pkl", "rb") as f:
        data = pickle.load(f)
    nrho = data["ppr_nrho"]
    moon_center = data["m2"]  # * dist_scaling
    moon_r = 1740 / dist_scaling

    ###### TODO: replace this by the commented line below
    f_p, f_v, f_a, f_R, f_sigma, f_X = generate_interpolators(nrho)
    # f_p, f_R, f_sigma, f_X = simplify_parametric_functions(nrho)
    ######

    # intial state
    angs = np.array(
        [
            [0, 0],
            [0, np.pi],
            [0, np.pi / 2],
            [0, 3 * np.pi / 2],
            [-np.pi / 2, np.pi / 2],
        ]
    )
    p0, n0 = spherical_to_cartesian(
        center=moon_center,
        r=moon_r * 1.05,
        ang1=angs[case_study, 0],
        ang2=angs[case_study, 1],
    )
    v0, theta0, phi0 = vec_to_gimbal(0.01 * n0)
    x0 = np.squeeze(np.hstack([p0, theta0, phi0, v0]))

    # settings
    XI = 0.02
    N = 100
    xi0 = 0.125

    # other variables
    nx = 6
    nu = 3
    d_xi = XI / N

    # set reference
    xi_ref = np.zeros((N, 1))
    x_ref = np.zeros((N, nx))
    u_ref = np.zeros((N - 1, nu))
    e1_ref = np.zeros((N, 3))
    for k in range(N):
        xi_k = (xi0 + k * d_xi) % 1.0
        p_ref = f_p(xi_k)
        v_ref, theta_ref, phi_ref = vec_to_gimbal(f_v(xi_k))

        xi_ref[k] = xi_k
        x_ref[k, :] = np.squeeze(np.vstack([p_ref, theta_ref, phi_ref, v_ref]))
        e1_ref[k, :] = np.squeeze(f_R(xi_ref[k])[:, 0])
        if k < N - 1:
            u_ref[k] = np.array([0, 0, 0])

    # ------------------------------------ NLP ----------------------------------- #
    # formulate NLP
    x = cs.MX.sym("x", N, nx)
    u = cs.MX.sym("u", N - 1, nu)
    x_nlp = cs.vertcat(
        cs.reshape(x, x.size1() * x.size2(), 1), cs.reshape(u, u.size1() * u.size2(), 1)
    )  # [px_0,...,px_N,py_0,...py_N,...]

    f_nlp = 0
    g_nlp = []
    lbg = []
    ubg = []

    for k in range(N - 1):

        # cost function
        e_p = x_ref[k, :3] - x[k, :3].T
        f_nlp += e_p.T @ cs.DM.eye(3) @ e_p
        f_nlp += 1e-10 * (u[k, :] @ cs.DM.eye(nu) @ u[k, :].T)

        # initial state
        if k == 0:
            g_nlp = cs.vertcat(g_nlp, x[0, :].T - x0)
            lbg = cs.vertcat(lbg, [0] * nx)  # , [0]*nu)
            ubg = cs.vertcat(ubg, [0] * nx)  # , [0]*nu)

        # dynamics
        x_next = forward_integrate(
            f=rocket_gimbaled, x=x[k, :].T, u=u[k, :].T, dt=d_xi, RK4=True
        ).T
        g_nlp = cs.vertcat(g_nlp, (x[k + 1, :] - x_next).T)
        lbg = cs.vertcat(lbg, [0] * nx)
        ubg = cs.vertcat(ubg, [0] * nx)

        # input constraints
        gimbal_max_rate = 500
        g_nlp = cs.vertcat(g_nlp, x[k, 5], u[k, 1], u[k, 2])
        lbg = cs.vertcat(lbg, [0.001], -gimbal_max_rate, -gimbal_max_rate)
        ubg = cs.vertcat(ubg, [50], gimbal_max_rate, gimbal_max_rate)

    # generate solver
    nlp_dict = {"x": x_nlp, "f": f_nlp, "g": g_nlp, "p": cs.vertcat()}
    nlp_opts = {
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
        "ipopt.print_level": 5,  # if nlp_params["verbose"] else 0,
        "print_time": False,
    }
    nlp_solver = cs.nlpsol("rocket_nlp", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": lbg, "ubg": ubg}

    # solve
    x_warm = np.hstack([x_ref.T.flatten(), u_ref.T.flatten()])
    sol = nlp_solver(x0=x_warm, lbg=lbg, ubg=ubg)
    status = nlp_solver.stats()["success"]
    if not status:
        print("NLP solver failed")

    # restructure output
    x = np.squeeze(sol["x"])[: nx * N].reshape(nx, N).T
    u = np.squeeze(sol["x"])[nx * N :].reshape(nu, N - 1).T

    # %% --------------------------------- Visualize -------------------------------- #
    ax = plt.figure().add_subplot(111, projection="3d")
    ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], "r--")
    ax.plot(x[:, 0], x[:, 1], x[:, 2], "b-")
    ax = plot_sphere(center=moon_center, radius=moon_r, color="gray", ax=ax)
    ax.plot(
        nrho.parametric_path["p"][:, 0],
        nrho.parametric_path["p"][:, 1],
        nrho.parametric_path["p"][:, 2],
        "-",
        color="gray",
        alpha=0.5,
    )
    axis_equal(
        nrho.parametric_path["p"][:, 0],
        nrho.parametric_path["p"][:, 1],
        nrho.parametric_path["p"][:, 2],
        ax=ax,
    )

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(xi_ref[:-1], u[:, 0])
    ax.set_ylabel(r"$a$")
    ax = fig.add_subplot(312)
    ax.plot(xi_ref[:-1], u[:, 1])
    ax.set_ylabel(r"$\dot{\theta}$")
    ax = fig.add_subplot(313)
    ax.plot(xi_ref[:-1], u[:, 2])
    ax.set_ylabel(r"$\dot{\phi}$")

    e1 = np.zeros((N, 3))
    for k in range(N):
        theta = x[k, 3]
        phi = x[k, 4]
        e1[k, :] = np.array(
            [np.cos(phi) * np.cos(theta), np.sin(phi), np.cos(phi) * np.sin(theta)]
        )
        e_e1 = 1 - np.dot(e1, e1_ref[k])

    fig, axs = plt.subplots(5, 1, sharex=True)
    axs[0].plot(xi_ref, np.linalg.norm(x[:, :3] - x_ref[:, :3], axis=1))
    axs[0].set_ylabel(r"$||e_p||$")
    axs[1].plot(xi_ref, e_e1)
    axs[1].set_ylabel(r"$||e_{e1}||$")
    axs[2].plot(xi_ref, x[:, 5])
    axs[2].set_ylabel("v")
    axs[3].plot(xi_ref, x[:, 3])
    axs[3].set_ylabel(r"$\theta$")
    axs[4].plot(xi_ref, x[:, 4])
    axs[4].set_ylabel(r"$\phi$")
    axs[4].set_xlabel(r"$\xi$")

    plt.show()

    # ----------------------------------- Save ----------------------------------- #
    with open(
        get_phodcos_path() + "/examples/data/" + str(case_study) + ".pkl",
        "wb",
    ) as f:
        pickle.dump(
            {
                "nrho": nrho,
                "moon_center": moon_center,
                "moon_r": moon_r,
                "x": x,
                "u": u,
                "x_ref": x_ref,
                "xi_ref": xi_ref,
                "e1_ref": e1_ref,
            },
            f,
        )
