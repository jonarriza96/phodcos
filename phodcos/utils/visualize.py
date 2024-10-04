import numpy as np
import casadi as cs


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from phodcos.utils.quaternion import quaternion_to_rotation


def axis_equal(X, Y, Z, ax=None):
    """
    Sets axis bounds to "equal" according to the limits of X,Y,Z.
    If axes are not given, it generates and labels a 3D figure.

    Args:
        X: Vector of points in coord. x
        Y: Vector of points in coord. y
        Z: Vector of points in coord. z
        ax: Axes to be modified

    Returns:
        ax: Axes with "equal" aspect


    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - 1.2 * max_range, mid_x + 1.2 * max_range)
    ax.set_ylim(mid_y - 1.2 * max_range, mid_y + 1.2 * max_range)
    ax.set_zlim(mid_z - 1.2 * max_range, mid_z + 1.2 * max_range)

    return ax


def plot_frames(
    r, e1, e2, e3, interval=0.9, scale=1.0, ax=None, ax_equal=True, planar=False
):
    """
    Plots the moving frame [e1,e2,e3] of the curve r. The amount of frames to
    be plotted can be controlled with "interval".

    Args:
        r: Vector of 3d points (x,y,z) of curve
        e1: Vector of first component of frame
        e2: Vector of second component of frame
        e3: Vector of third component of frame
        interval: Percentage of frames to be plotted, i.e, 1 plots a frame in
                  every point of r, while 0 does not plot any.
        scale: Float to size components of frame
        ax: Axis where plot will be modified

    Returns:
        ax: Modified plot
    """
    # scale = 0.1
    nn = r.shape[0]
    tend = r + e1 * scale
    nend = r + e2 * scale
    bend = r + e3 * scale
    # interval = 1
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")

    if planar:
        if interval == 1:
            rng = range(nn)
        else:
            rng = range(0, nn, int(nn * (1 - interval)))
        for i in rng:  # if nn >1 else 1):
            if planar == 1:
                ax.plot([r[i, 0], tend[i, 0]], [r[i, 1], tend[i, 1]], "r")
                ax.plot([r[i, 0], nend[i, 0]], [r[i, 1], nend[i, 1]], "g")
            elif planar == 2:
                ax.plot(
                    [r[i, 0], tend[i, 0]], [r[i, 2], tend[i, 2]], "r"
                )  # , linewidth=2)
                ax.plot(
                    [r[i, 0], bend[i, 0]], [r[i, 2], bend[i, 2]], "g"
                )  # , linewidth=2)
            elif planar == 3:
                ax.plot(
                    [r[i, 1], tend[i, 1]], [r[i, 2], tend[i, 2]], "r"
                )  # , linewidth=2)
                ax.plot(
                    [r[i, 1], bend[i, 1]], [r[i, 2], bend[i, 2]], "b"
                )  # , linewidth=2)
        ax.set_aspect("equal")

    else:
        if ax_equal:
            ax = axis_equal(r[:, 0], r[:, 1], r[:, 2], ax=ax)
        if interval == 1:
            rng = range(nn)
        else:
            rng = range(0, nn, int(nn * (1 - interval)) if nn > 1 else 1)

        for i in rng:
            ax.plot(
                [r[i, 0], tend[i, 0]], [r[i, 1], tend[i, 1]], [r[i, 2], tend[i, 2]], "r"
            )
            ax.plot(
                [r[i, 0], nend[i, 0]], [r[i, 1], nend[i, 1]], [r[i, 2], nend[i, 2]], "g"
            )
            ax.plot(
                [r[i, 0], bend[i, 0]], [r[i, 1], bend[i, 1]], [r[i, 2], bend[i, 2]], "b"
            )

    return ax
