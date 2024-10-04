import argparse
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from phodcos.utils.parameterization import nominal_path
from phodcos.phodcos import PHODCOS

if __name__ == "__main__":
    # -------------------------------- User inputs ------------------------------- #
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interp_order",
        type=int,
        default=4,
        help="Hermite data interpolation order (2 or 4)",
    )
    parser.add_argument(
        "--n_start",
        type=int,
        default=1,
        help="Minimum number of segments when benchmarking",
    )
    parser.add_argument(
        "--n_end",
        type=int,
        default=2,
        help="Maximum number of segments when benchmarking",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=100,
        help="Multiplicity of discretization (per segment) when evaluating",
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    # Assign args to variables
    interp_order = args.interp_order
    n_eval = args.n_eval
    visualize = args.visualize
    n_sweep = [args.n_start, args.n_end]

    # --------------------- Initialize phodcos and nominal path ------------------ #
    phd = PHODCOS(interp_order=interp_order, n_segments=2 ** n_sweep[0], n_eval=n_eval)
    no_path = nominal_path()
    # ------------------------------ Benchmark loop ------------------------------ #

    # loop over different number of segments
    error_old = 0
    for n in range(n_sweep[0], n_sweep[1]):
        # set number of segments
        phd.n_segments = 2**n

        # parameterize and evaluate path
        phd.parameterize_path(nominal_path=no_path)
        phd.evaluate_parameterization()

        # print results
        error = phd.error
        print("n,segments,error,ratio:", n, 2**n, error, error_old / error)
        error_old = error

        # visualize
        if visualize:
            phd.visualize_parameterization()
