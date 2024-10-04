# %%
import numpy as np
import matplotlib.pyplot as plt

import pickle

from phodcos.utils.utils import get_phodcos_path
from phodcos.utils.visualize import plot_frames, axis_equal


def plot_sphere(center, radius, color, ax=None, planar=False, alpha=1):
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
        ax.plot_surface(xs, ys, zs, color=color, alpha=alpha)
    else:
        circle = plt.Circle(
            (center[0], center[1]), radius, color=color, fill=True, alpha=alpha
        )
        ax.add_patch(circle)

    # Set plot labels
    ax.set_xlabel("X ")
    ax.set_ylabel("Y")
    if not planar:
        ax.set_zlabel("Z ")

    return ax


# import
case_study = 4
x = {"0": None, "1": None, "2": None, "3": None, "4": None}
u = {"0": None, "1": None, "2": None, "3": None, "4": None}
keys = ["0", "1", "2", "3", "4"]
for key in keys:
    path = get_phodcos_path() + "/examples/data/"
    with open(path + key + ".pkl", "rb") as f:
        data = pickle.load(f)
    nrho = data["nrho"]
    moon_center = data["moon_center"]
    moon_r = data["moon_r"]
    x_ref = data["x_ref"]
    xi_ref = data["xi_ref"]
    e1_ref = data["e1_ref"]
    N = x_ref.shape[0]

    x[key] = data["x"]
    u[key] = data["u"]

# %% Navigation - Isometric view

ax = plt.figure().add_subplot(111, projection="3d")
ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], "k--", linewidth=2)
ax = plot_sphere(center=moon_center, radius=moon_r, color="gray", ax=ax, alpha=0.5)
ax.plot(
    nrho.parametric_path["p"][:, 0],
    nrho.parametric_path["p"][:, 1],
    nrho.parametric_path["p"][:, 2],
    "-",
    color="gray",
    alpha=0.5,
)
for key in keys:
    ax.plot(x[key][:, 0], x[key][:, 1], x[key][:, 2], "-")
axis_equal(
    nrho.parametric_path["p"][:, 0],
    nrho.parametric_path["p"][:, 1],
    nrho.parametric_path["p"][:, 2],
    ax=ax,
)

ax.view_init(azim=-62, elev=7)
# ax.view_init(azim=-80, elev=5) #zoomed in
# ax.dist = 2.5
ax.set_axis_off()
# fig.savefig("figure.pdf", dpi=1800)

# %% Navigation - Side view
ax = plt.figure().add_subplot(111)
ax.plot(x_ref[:, 0], x_ref[:, 2], "k--", linewidth=2)
ax = plot_sphere(
    center=moon_center, radius=moon_r, color="gray", ax=ax, alpha=0.5, planar=True
)
ax.plot(
    nrho.parametric_path["p"][:, 0],
    nrho.parametric_path["p"][:, 2],
    "-",
    color="gray",
    alpha=0.5,
)
for key in keys:
    ax.plot(x[key][:, 0], x[key][:, 2], "-")

ax.set_aspect("equal")
ax.set_axis_off()
# fig.savefig("figure.pdf", dpi=1800)

# %% Navigation - Top view
ax = plt.figure().add_subplot(111)
ax.plot(x_ref[:, 0], x_ref[:, 1], "k--", linewidth=2)
ax = plot_sphere(
    center=moon_center, radius=moon_r, color="gray", ax=ax, alpha=0.5, planar=True
)
ax.plot(
    nrho.parametric_path["p"][:, 0],
    nrho.parametric_path["p"][:, 1],
    "-",
    color="gray",
    alpha=0.5,
)
for key in keys:
    ax.plot(x[key][:, 0], x[key][:, 1], "-")

ax.set_aspect("equal")
ax.set_axis_off()
# fig.savefig("figure.pdf", dpi=1800)

# %% Navigation - Front view
ax = plt.figure().add_subplot(111)
ax.plot(x_ref[:, 1], x_ref[:, 2], "k--", linewidth=2)
ax = plot_sphere(
    center=np.array([moon_center[1], moon_center[2], 0]),
    radius=moon_r,
    color="gray",
    ax=ax,
    alpha=0.5,
    planar=True,
)
ax.plot(
    nrho.parametric_path["p"][:, 1],
    nrho.parametric_path["p"][:, 2],
    "-",
    color="gray",
    alpha=0.5,
)
for key in keys:
    ax.plot(x[key][:, 1], x[key][:, 2], "-")

ax.set_aspect("equal")
ax.set_axis_off()
# fig.savefig("figure.pdf", dpi=1800)


# %%


fig, axs = plt.subplots(5, 1, sharex=True, figsize=(9, 4))
for key in keys:
    xx = x[key]

    e1 = np.zeros((N, 3))
    for k in range(N):
        theta = xx[k, 3]
        phi = xx[k, 4]
        e1[k, :] = np.array(
            [np.cos(phi) * np.cos(theta), np.sin(phi), np.cos(phi) * np.sin(theta)]
        )
        e_e1 = 1 - np.dot(e1, e1_ref[k])

    axs[0].plot(xi_ref * 4, np.linalg.norm(xx[:, :3] - x_ref[:, :3], axis=1))
    axs[0].set_ylabel(r"$||e_p||$")
    axs[1].plot(xi_ref * 4, e_e1)
    axs[1].set_ylabel(r"$||e_R||$")
    axs[2].plot(xi_ref * 4, xx[:, 5])
    axs[2].set_ylabel("v")
    axs[3].plot(xi_ref * 4, xx[:, 3])
    axs[3].set_ylabel(r"$\theta$")
    axs[4].plot(xi_ref * 4, xx[:, 4])
    axs[4].set_ylabel(r"$\phi$")
    axs[4].set_xlabel(r"$\xi$")
