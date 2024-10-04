# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

from phodcos.utils.utils import get_phodcos_path
from phodcos.utils.visualize import plot_frames, axis_equal


def plot_sphere(x, y, z, radius, color, ax=None, planar=False):

    # Create a meshgrid for the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Parametric equations for a sphere
    xs = radius * np.outer(np.cos(u), np.sin(v)) + x
    ys = radius * np.outer(np.sin(u), np.sin(v)) + y
    zs = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z

    # Plotting the sphere
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    if not planar:
        ax.plot_surface(xs, ys, zs, color=color)
    else:
        # ax.plot_surface(xs, ys, color=color)
        circle = plt.Circle((x, y), radius, color=color, fill=True)
        ax.add_patch(circle)

    # Set plot labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if not planar:
        ax.set_zlabel("Z")

    return ax


path = get_phodcos_path() + "/examples/data/"
with open(path + "nrho.pkl", "rb") as f:
    data = pickle.load(f)

ppr_eci = data["ppr_eci"]
ppr_nrho = data["ppr_nrho"]
p_eci = data["p_eci"]
p_nrho = data["p_nrho"]
m1 = data["m1"]
m2 = data["m2"]
p_m2 = data["p_m2"]


scaling = 389703
moon_r = 1740 / scaling
earth_r = 6378 / scaling

ind = 75

# ------------------------------------ ECI ----------------------------------- #
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    ppr_eci.parametric_path["p"][:-ind, 0],
    ppr_eci.parametric_path["p"][:-ind, 1],
    ppr_eci.parametric_path["p"][:-ind, 2],
    "b-",
)
ax.plot(
    ppr_eci.nominal_path["p"][:-ind, 0],
    ppr_eci.nominal_path["p"][:-ind, 1],
    ppr_eci.nominal_path["p"][:-ind, 2],
    "k--",
)
ax = plot_frames(
    ppr_eci.parametric_path["p"][:-ind],
    ppr_eci.parametric_path["erf"][:-ind, :, 0],
    ppr_eci.parametric_path["erf"][:-ind, :, 1],
    ppr_eci.parametric_path["erf"][:-ind, :, 2],
    interval=0.99,  # 0.99,
    scale=0.075,  # 0.25,
    ax=ax,
)
ax = axis_equal(
    ppr_eci.parametric_path["p"][:-ind, 0],
    ppr_eci.parametric_path["p"][:-ind, 1],
    ppr_eci.parametric_path["p"][:-ind, 2],
    ax=ax,
)

ax.plot(p_m2[:-10, 0], p_m2[:-10, 1], p_m2[:-10, 2], "--", color="gray")
ax = plot_sphere(0, 0, 0, earth_r, color="green", ax=ax)
ax.set_axis_off()
fig.set_size_inches(35, 10)
ax.view_init(azim=-58, elev=15)
ax.dist = 5.5
# fig.savefig(path + "figures/eci.pdf", dpi=1800)
plt.show()

# %%
# ---------------------------------------------------------------------------- #
#                                     NRHO                                     #
# ---------------------------------------------------------------------------- #

# ------------------------------ Isometric view ------------------------------ #
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    ppr_nrho.parametric_path["p"][:-ind, 0],
    ppr_nrho.parametric_path["p"][:-ind, 1],
    ppr_nrho.parametric_path["p"][:-ind, 2],
    "b-",
)
ax.plot(
    ppr_nrho.nominal_path["p"][:-ind, 0],
    ppr_nrho.nominal_path["p"][:-ind, 1],
    ppr_nrho.nominal_path["p"][:-ind, 2],
    "k--",
)
ax = plot_frames(
    ppr_nrho.parametric_path["p"][:-ind],
    ppr_nrho.parametric_path["erf"][:-ind, :, 0],
    ppr_nrho.parametric_path["erf"][:-ind, :, 1],
    ppr_nrho.parametric_path["erf"][:-ind, :, 2],
    interval=0.99,  # 0.99,
    scale=0.005,  # 0.25,
    ax=ax,
)
ax = axis_equal(
    ppr_nrho.parametric_path["p"][:-ind, 0],
    ppr_nrho.parametric_path["p"][:-ind, 1],
    ppr_nrho.parametric_path["p"][:-ind, 2],
    ax=ax,
)


ax.set_axis_off()
fig.set_size_inches(35, 10)
ax.view_init(azim=-62, elev=7)
ax.dist = 5.5
ax = plot_sphere(m2[0], m2[1], m2[2], moon_r, color="gray", ax=ax)
# fig.savefig(path + "figures/nrho_iso.pdf", dpi=1800)
plt.show()

# --------------------------------- Side view -------------------------------- #
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(
    # ppr_nrho.parametric_path["p"][:-ind, 0],
    ppr_nrho.parametric_path["p"][:-ind, 1],
    ppr_nrho.parametric_path["p"][:-ind, 2],
    "b-",
)
ax.plot(
    # ppr_nrho.nominal_path["p"][:-ind, 0],
    ppr_nrho.nominal_path["p"][:-ind, 1],
    ppr_nrho.nominal_path["p"][:-ind, 2],
    "k--",
)
ax = plot_frames(
    ppr_nrho.parametric_path["p"][:-ind],
    ppr_nrho.parametric_path["erf"][:-ind, :, 0],
    ppr_nrho.parametric_path["erf"][:-ind, :, 1],
    ppr_nrho.parametric_path["erf"][:-ind, :, 2],
    interval=0.99,  # 0.99,
    scale=0.005,  # 0.25,
    ax=ax,
    planar=3,
    ax_equal=False,
)

ax = plot_sphere(x=m2[1], y=m2[2], z=0, radius=moon_r, color="gray", ax=ax, planar=True)
ax.set_aspect("equal")
ax.set_axis_off()
# fig.savefig(path + "figures/nrho_front.pdf", dpi=1800)

# --------------------------------- Top view --------------------------------- #
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(
    ppr_nrho.parametric_path["p"][:-ind, 0],
    # ppr_nrho.parametric_path["p"][:-ind, 1],
    ppr_nrho.parametric_path["p"][:-ind, 2],
    "b-",
)
ax.plot(
    ppr_nrho.nominal_path["p"][:-ind, 0],
    # ppr_nrho.nominal_path["p"][:-ind, 1],
    ppr_nrho.nominal_path["p"][:-ind, 2],
    "k--",
)
ax = plot_frames(
    ppr_nrho.parametric_path["p"][:-ind],
    ppr_nrho.parametric_path["erf"][:-ind, :, 0],
    ppr_nrho.parametric_path["erf"][:-ind, :, 1],
    ppr_nrho.parametric_path["erf"][:-ind, :, 2],
    interval=0.99,  # 0.99,
    scale=0.005,  # 0.25,
    ax=ax,
    planar=2,
    ax_equal=False,
)

ax = plot_sphere(x=m2[0], y=m2[2], z=0, radius=moon_r, color="gray", ax=ax, planar=True)
ax.set_aspect("equal")
ax.set_axis_off()
# fig.savefig(path + "figures/nrho_side.pdf", dpi=1800)


# ---------------------------------------------------------------------------- #
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(p_nrho[:, 0], p_nrho[:, 1], p_nrho[:, 2])
# ax.plot([0], [0], [0], marker="o", markersize=3, color="k")
# ax = plot_sphere(m1[0], m1[1], m1[2], earth_r, color="green", ax=ax)
# ax = plot_sphere(m2[0], m2[1], m2[2], moon_r, color="gray", ax=ax)
# axis_equal(p_nrho[:, 0], p_nrho[:, 1], p_nrho[:, 2], ax=ax)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(p_eci[:, 0], p_eci[:, 1], p_eci[:, 2], color="blue")
# ax.plot(p_m2[:, 0], p_m2[:, 1], p_m2[:, 2], color="gray")
# ax = plot_sphere(0, 0, 0, earth_r, color="green", ax=ax)
# axis_equal(p_eci[:, 0], p_eci[:, 1], p_eci[:, 2], ax=ax)


# plt.show()

# %% ----------------------------- Parametric speed ----------------------------- #
fig = plt.figure()  # figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(ppr_nrho.parametric_path["xi"], ppr_nrho.parametric_path["sigma"])
ax.set_ylabel(r"$\sigma$", fontsize=14)
ax.set_xlabel(r"$\xi$", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)

# %% ----------------------------- Angular velocity ----------------------------- #
fig = plt.figure()  # figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(ppr_nrho.parametric_path["xi"], ppr_nrho.parametric_path["X"][:, 0], color="r")
ax.plot(ppr_nrho.parametric_path["xi"], ppr_nrho.parametric_path["X"][:, 1], color="g")
ax.plot(ppr_nrho.parametric_path["xi"], ppr_nrho.parametric_path["X"][:, 2], color="b")
ax.set_ylabel(r"$\omega$", fontsize=14)
ax.set_xlabel(r"$\xi$", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)
