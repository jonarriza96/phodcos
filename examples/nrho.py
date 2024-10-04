# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import (
    LSQUnivariateSpline,
    UnivariateSpline,
    InterpolatedUnivariateSpline,
)

import pickle

import casadi as cs

from phodcos.phodcos import PHODCOS
from phodcos.utils.visualize import axis_equal
from phodcos.utils.utils import get_phodcos_path


def EMBC_to_ECI(r_c3bp, t, T):

    theta = 2 * np.pi / T * t
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    return R @ (r_c3bp + np.array([mu, 0, 0]))


def convert_orbit_to_spline(t, p):
    order = 4

    # get scipy splines (for knots and coefficients)
    spl_x = InterpolatedUnivariateSpline(x=t / t[-1], y=p[0], k=order)
    spl_y = InterpolatedUnivariateSpline(x=t / t[-1], y=p[1], k=order)
    spl_z = InterpolatedUnivariateSpline(x=t / t[-1], y=p[2], k=order)
    knots_x = np.hstack(
        [
            np.tile(spl_x.get_knots()[0], order),
            spl_x.get_knots(),
            np.tile(spl_x.get_knots()[-1], order),
        ]
    )
    knots_y = np.hstack(
        [
            np.tile(spl_y.get_knots()[0], order),
            spl_y.get_knots(),
            np.tile(spl_y.get_knots()[-1], order),
        ]
    )
    knots_z = np.hstack(
        [
            np.tile(spl_z.get_knots()[0], order),
            spl_z.get_knots(),
            np.tile(spl_z.get_knots()[-1], order),
        ]
    )

    # convert to casadi
    xi = cs.MX.sym("xi")
    cspl_x = cs.bspline(xi, cs.DM(spl_x.get_coeffs()), [knots_x], [order], 1)
    cspl_y = cs.bspline(xi, cs.DM(spl_y.get_coeffs()), [knots_y], [order], 1)
    cspl_z = cs.bspline(xi, cs.DM(spl_z.get_coeffs()), [knots_z], [order], 1)
    r = cs.vertcat(cspl_x, cspl_y, cspl_z)

    dr = cs.jacobian(r, xi)  # first derivative of analytical curve
    ddr = cs.jacobian(dr, xi)  # second derivative of analytical curve
    dddr = cs.jacobian(ddr, xi)  # third derivative of analytical curve
    ddddr = cs.jacobian(dddr, xi)  # fourth derivative of analytical curve
    f_p = cs.Function(
        "f_p", [xi], [r, dr, ddr, dddr, ddddr], ["xi"], ["p", "v", "a", "j", "s"]
    )

    return {"f_p": f_p}


n_tile = 4  # [4 for a single moon orbit, 9 for 2 moon orbits]

n_segments = 8
n_eval = 10

dist_scalings = [1, 389703]
t_scalings = [1, 382981]
file_names = ["nrho", "nrho_SI"]

for k, (dist_scaling, t_scaling, file_name) in enumerate(
    zip(dist_scalings, t_scalings, file_names)
):

    if k == 0:
        print("\nNRHO in normalized units:")
    else:
        print("\nNRHO in SI units:")

    # -------------------------------- Import data ------------------------------- #
    # Load the CSV file
    data = np.genfromtxt(
        get_phodcos_path() + "/examples/data/nrho.csv", delimiter=",", skip_header=1
    )
    data[:, 0] *= t_scaling
    data[:, 1:4] *= dist_scaling

    # Extract columns as separate NumPy arrays
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    # lagrange points

    # periods
    T_nrho = t[-1]
    ratio = 27.3 / 7.007  # 9/2
    T_moon = ratio * T_nrho

    # --------------------- Expand orbit to full moon period --------------------- #
    # main bodies
    mu = 1.125e-2
    m1 = np.array([-mu, 0, 0])
    m2 = np.array([1 - mu, 0, 0])

    n_nrho = data.shape[0]
    X = np.tile(data, (n_tile, 1))
    for k in range(9):
        X[k * n_nrho : (k + 1) * n_nrho, 0] += T_nrho * k
    t_sim = X[:, 0]
    X = X[:, 1:]
    p_nrho = X[:, :3]

    # rotate to ECI
    p_eci = np.zeros((len(t_sim), 3))
    p_m2 = np.zeros((len(t_sim), 3))
    for k in range(len(t_sim)):
        p_eci[k, :] = EMBC_to_ECI(r_c3bp=X[k, :3], t=t_sim[k], T=T_moon)
        p_m2[k, :] = EMBC_to_ECI(r_c3bp=np.array([1 - mu, 0, 0]), t=t_sim[k], T=T_moon)

    # ---------------------------------- PHODCOS --------------------------------- #

    # convert orbits to parametric equations
    unique_t_sim, indices = np.unique(t_sim, return_index=True)
    p_eci_cs = convert_orbit_to_spline(t=t_sim[indices], p=p_eci[indices].T)
    p_nrho_cs = convert_orbit_to_spline(t=t_sim[indices], p=p_nrho[indices].T)

    print("\n\tParameterizing NRHO in ECI frame ...")
    ppr_eci = PHODCOS(interp_order=4, n_segments=2**n_segments, n_eval=n_eval)
    ppr_eci.parameterize_path(nominal_path=p_eci_cs)
    ppr_eci.evaluate_parameterization()
    print("\tDone.")

    print("\n\tParameterizing NRHO in EMBR frame ...")
    ppr_nrho = PHODCOS(interp_order=4, n_segments=2**n_segments, n_eval=n_eval)
    ppr_nrho.parameterize_path(nominal_path=p_nrho_cs)
    ppr_nrho.evaluate_parameterization()
    print("\tDone.")

    ppr_eci.visualize_parameterization()
    ppr_nrho.visualize_parameterization()

    # ----------------------------------- Save ----------------------------------- #
    with open(get_phodcos_path() + "/examples/data/" + file_name + ".pkl", "wb") as f:
        pickle.dump(
            {
                "ppr_eci": ppr_eci,
                "ppr_nrho": ppr_nrho,
                "p_eci": p_eci,
                "p_nrho": p_nrho,
                "m1": m1,
                "m2": m2,
                "p_m2": p_m2,
                "t_nrho_period": t,
                "t_nrho": t_sim,
            },
            f,
        )
