import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


def visualize_solution(mxyzBeam: np.ndarray,
                     vx: np.ndarray,
                     vy: np.ndarray,
                     vZslice: np.ndarray,
                     core: tuple[float, float, float],
                     xlims: tuple[float, float],
                     ylims: tuple[float, float],
                     secpoints: tuple[float, float]):
    """
    Visualize solution of the Beam Propagation Method.
    :param mxyzBeam: 3D distribution of the complex beam amplitude (3D complex numpy array)
    :param vx: x-coordinate (1D float numpy array) [m]
    :param vy: y-coordinate (1D float numpy array) [m]
    :param vZslice: array of z-slices (1D float numpy array) [m]
    :param core: tuple of fiber's core parameters (x0 - center point [um], y0 - center point [um], diemeter [um])
    :param xlims: tuple of x-limits to display plots ([um], [um])
    :param ylims: tuple of y-limits to display plots ([um], [um])
    :param secpoints: tuple of points where YZ and XZ-section is taken ([m], [m])
    :return:
    """

    # Set colormap for plots
    colormap = 'jet'

    # Parameters for the plots
    x0, y0 = core[0], core[1]  # Center of the fiber core [um], [um]
    dcr_um = core[2]  # Convert core diameter to microns [um]
    xlim1, xlim2 = xlims[0], xlims[1] # x-limits to display plots [um], [um]
    ylim1, ylim2 = ylims[0], ylims[1] # x-limits to display plots [um], [um]
    zlim1, zlim2 = vZslice[0] * 1e3, vZslice[-1] * 1e3 # z-limits to display plots [mm], [mm]
    xpoint, ypoint = secpoints[0], secpoints[1]  # points where YZ and XZ-section is taken ([m], [m])

    xticks = [xlim1, xlim1 / 2, 0, xlim2 / 2, xlim2]
    yticks = [ylim1, ylim1 / 2, 0, ylim2 / 2, ylim2]
    zticks = list(range(int(zlim1), int(zlim2) + 1))

    xticklabels = [str(num) for num in xticks]
    yticklabels = [str(num) for num in yticks]
    zticklabels = [str(num) for num in zticks]

    xind = np.argmin(np.abs(vx - xpoint))  # Find the index for xpoint
    yind = np.argmin(np.abs(vy - ypoint)) # Find the index for ypoint

    # Mesh for pcolormesh
    mx, my = np.meshgrid(vx, vy)
    mzx, mxz = np.meshgrid(vZslice, vx)
    mzy, myz = np.meshgrid(vZslice, vy)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

    # Data for pcolormesh
    intcoeff = 1e-4  # intensity coefficient [cm ** 2]

    xyBeam = mxyzBeam[:, :, -1]
    int_xyBeam = abs(xyBeam) ** 2 * intcoeff

    xzBeam = mxyzBeam[yind, :, :]
    int_xzBeam = abs(xzBeam) ** 2 * intcoeff

    yzBeam = mxyzBeam[:, xind, :]
    int_yzBeam = abs(yzBeam) ** 2 * intcoeff

    # --- 2D (X,Y) BEAM PROFILE ---
    c1 = axes[0].pcolormesh(mx * 1e6, my * 1e6, int_xyBeam, shading='auto', cmap=colormap)
    axes[0].set_aspect(1)
    axes[0].set_title(r'2D (XY) Output Intensity [W/$cm^{2}$]', fontsize=9)
    axes[0].set_xlabel('x (um)', fontsize=9)
    axes[0].set_ylabel('y (um)', fontsize=9)
    axes[0].set_xlim(xlim1, xlim2)
    axes[0].set_ylim(ylim1, ylim2)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels, fontsize=9)
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticklabels, fontsize=9)
    fig.colorbar(c1, orientation='vertical')
    # Draw core boundaries
    axes[0].add_patch(patches.Circle((x0, y0), dcr_um / 2, edgecolor='w', linestyle='-', fill=False))
    axes[0].add_patch(patches.Circle((x0, y0), dcr_um / 2, edgecolor='k', linestyle='--', fill=False))

    # --- 2D (X,Z) BEAM PROFILE ---
    c2 = axes[1].pcolormesh(mxz * 1e6, mzx * 1e3, int_xzBeam, shading='auto', cmap=colormap)
    axes[1].set_title(r'2D (XZ) Intensity [W/$cm^{2}$]', fontsize=9)
    axes[1].set_xlabel('x (um)', fontsize=9)
    axes[1].set_ylabel('z (mm)', fontsize=9)
    axes[1].set_xlim(xlim1, xlim2)
    axes[1].set_ylim(zlim1, zlim2)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticklabels, fontsize=9)
    axes[1].set_yticks(zticks)
    axes[1].set_yticklabels(zticklabels, fontsize=9)
    fig.colorbar(c2, orientation='vertical')
    # Draw core boundaries
    axes[1].axvline(-dcr_um / 2, color='w', linestyle='-')
    axes[1].axvline(-dcr_um / 2, color='k', linestyle='--')
    axes[1].axvline(dcr_um / 2, color='w', linestyle='-')
    axes[1].axvline(dcr_um / 2, color='k', linestyle='--')

    # --- 2D (Y,Z) BEAM PROFILE ---
    c3 = axes[2].pcolormesh(myz * 1e6, mzy * 1e3, int_yzBeam, shading='auto', cmap=colormap)
    axes[2].set_title(r'2D (YZ) Intensity [W/$cm^{2}$]', fontsize=9)
    axes[2].set_xlabel('y (um)', fontsize=9)
    axes[2].set_ylabel('z (mm)', fontsize=9)
    axes[2].set_xlim(xlim1, xlim2)
    axes[2].set_ylim(zlim1, zlim2)
    axes[2].set_xticks(yticks)
    axes[2].set_xticklabels(yticklabels, fontsize=9)
    axes[2].set_yticks(zticks)
    axes[2].set_yticklabels(zticklabels, fontsize=9)
    fig.colorbar(c3, orientation='vertical')
    # Draw core boundaries
    axes[2].axvline(-dcr_um / 2, color='w', linestyle='-')
    axes[2].axvline(-dcr_um / 2, color='k', linestyle='--')
    axes[2].axvline(dcr_um / 2, color='w', linestyle='-')
    axes[2].axvline(dcr_um / 2, color='k', linestyle='--')

    plt.show()


def animate_solution(mxyzBeam: np.ndarray,
                     vx: np.ndarray,
                     vy: np.ndarray,
                     vZslice: np.ndarray,
                     core: tuple[float, float, float],
                     xlims: tuple[float, float],
                     ylims: tuple[float, float],
                     secpoints: tuple[float, float]):
    """
    Animate solution of the Beam Propagation Method.
    :param mxyzBeam: 3D distribution of the complex beam amplitude (3D complex numpy array)
    :param vx: x-coordinate (1D float numpy array) [m]
    :param vy: y-coordinate (1D float numpy array) [m]
    :param vZslice: array of z-slices (1D float numpy array) [m]
    :param core: tuple of fiber's core parameters (x0 - center point [um], y0 - center point [um], diemeter [um])
    :param xlims: tuple of x-limits to display plots ([um], [um])
    :param ylims: tuple of y-limits to display plots ([um], [um])
    :param secpoints: tuple of points where YZ and XZ-section is taken ([m], [m])
    """

    # Set colormap for plots
    colormap = 'jet'

    # Parameters for the plots
    x0, y0 = core[0], core[1]  # Center of the fiber core [um], [um]
    dcr_um = core[2]  # Convert core diameter to microns [um]
    xlim1, xlim2 = xlims[0], xlims[1] # x-limits to display plots [um], [um]
    ylim1, ylim2 = ylims[0], ylims[1] # x-limits to display plots [um], [um]
    zlim1, zlim2 = vZslice[0] * 1e3, vZslice[-1] * 1e3 # z-limits to display plots [mm], [mm]
    xpoint, ypoint = secpoints[0], secpoints[1]  # points where YZ and XZ-section is taken ([m], [m])

    xticks = [xlim1, xlim1 / 2, 0, xlim2 / 2, xlim2]
    yticks = [ylim1, ylim1 / 2, 0, ylim2 / 2, ylim2]
    zticks = list(range(int(zlim1), int(zlim2) + 1))

    xticklabels = [str(num) for num in xticks]
    yticklabels = [str(num) for num in yticks]
    zticklabels = [str(num) for num in zticks]

    xind = np.argmin(np.abs(vx - xpoint))  # Find the index for xpoint
    yind = np.argmin(np.abs(vy - ypoint)) # Find the index for ypoint

    # Mesh for pcolormesh
    mx, my = np.meshgrid(vx, vy)
    mzx, mxz = np.meshgrid(vZslice, vx)
    mzy, myz = np.meshgrid(vZslice, vy)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)

    # Initial data for pcolormesh
    int_xyBeam_init = np.zeros((len(vy), len(vx)))
    int_xzBeam_init = np.zeros((len(vx), len(vZslice)))
    int_yzBeam_init = np.zeros((len(vy), len(vZslice)))

    # --- 2D (X,Y) BEAM PROFILE ---
    c1 = axes[0].pcolormesh(mx * 1e6, my * 1e6, int_xyBeam_init, shading='auto', cmap=colormap)
    axes[0].set_aspect(1)
    axes[0].set_xlabel('x (um)', fontsize=9)
    axes[0].set_ylabel('y (um)', fontsize=9)
    axes[0].set_xlim(xlim1, xlim2)
    axes[0].set_ylim(ylim1, ylim2)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels, fontsize=9)
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticklabels, fontsize=9)
    # Draw core boundaries
    axes[0].add_patch(patches.Circle((x0, y0), dcr_um / 2, edgecolor='w', linestyle='-', fill=False))
    axes[0].add_patch(patches.Circle((x0, y0), dcr_um / 2, edgecolor='k', linestyle='--', fill=False))

    # --- 2D (X,Z) BEAM PROFILE ---
    c2 = axes[1].pcolormesh(mxz * 1e6, mzx * 1e3, int_xzBeam_init, shading='auto', cmap=colormap)
    axes[1].set_xlabel('x (um)', fontsize=9)
    axes[1].set_ylabel('z (mm)', fontsize=9)
    axes[1].set_xlim(xlim1, xlim2)
    axes[1].set_ylim(zlim1, zlim2)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticklabels, fontsize=9)
    axes[1].set_yticks(zticks)
    axes[1].set_yticklabels(zticklabels, fontsize=9)
    # Draw core boundaries
    axes[1].axvline(-dcr_um / 2, color='w', linestyle='-')
    axes[1].axvline(-dcr_um / 2, color='k', linestyle='--')
    axes[1].axvline(dcr_um / 2, color='w', linestyle='-')
    axes[1].axvline(dcr_um / 2, color='k', linestyle='--')

    # --- 2D (Y,Z) BEAM PROFILE ---
    c3 = axes[2].pcolormesh(myz * 1e6, mzy * 1e3, int_yzBeam_init, shading='auto', cmap=colormap)
    axes[2].set_xlabel('y (um)', fontsize=9)
    axes[2].set_ylabel('z (mm)', fontsize=9)
    axes[2].set_xlim(xlim1, xlim2)
    axes[2].set_ylim(zlim1, zlim2)
    axes[2].set_xticks(yticks)
    axes[2].set_xticklabels(yticklabels, fontsize=9)
    axes[2].set_yticks(zticks)
    axes[2].set_yticklabels(zticklabels, fontsize=9)
    # Draw core boundaries
    axes[2].axvline(-dcr_um / 2, color='w', linestyle='-')
    axes[2].axvline(-dcr_um / 2, color='k', linestyle='--')
    axes[2].axvline(dcr_um / 2, color='w', linestyle='-')
    axes[2].axvline(dcr_um / 2, color='k', linestyle='--')

    # Arrays for XZ-section and YZ-section
    xzBeam = np.zeros((len(vx), len(vZslice)), dtype=complex)
    yzBeam = np.zeros((len(vy), len(vZslice)), dtype=complex)


    def update(iz: int):
        """
        Update data for the plots
        :param iz: index of the z-sample
        :return: List of QuadMesh objects representing the updated intensity plots for XY, XZ, and YZ planes.
        """

        intcoeff = 1e-4  # intensity coefficient [cm ** 2]

        # Update XY-section
        xyBeam = mxyzBeam[:, :, iz]
        int_xyBeam = abs(xyBeam) ** 2 * intcoeff
        c1.set_array(int_xyBeam.ravel())
        c1.set_clim(int_xyBeam.min(), int_xyBeam.max())
        axes[0].set_title(f'2D (XY) Intensity, z = {vZslice[iz] * 1e3:.2f} mm', fontsize=9)

        # Update XZ-section
        xzBeam[:, iz] = xyBeam[yind, :]
        int_xzBeam = abs(xzBeam) ** 2 * intcoeff
        c2.set_array(int_xzBeam.ravel())
        c2.set_clim(int_xzBeam.min(), int_xzBeam.max())
        axes[1].set_title(f'2D (XZ) Intensity, z = {vZslice[iz] * 1e3:.2f} mm', fontsize=9)

        # Update YZ-section
        yzBeam[:, iz] = xyBeam[:, xind]
        int_yzBeam = abs(yzBeam) ** 2 * intcoeff
        c3.set_array(int_yzBeam.ravel())
        c3.set_clim(int_yzBeam.min(), int_yzBeam.max())
        axes[2].set_title(f'2D (YZ) Intensity, z = {vZslice[iz] * 1e3:.2f} mm', fontsize=9)

        return [c1, c2, c3]


    # run animation
    ani = FuncAnimation(fig, update, frames=len(vZslice), interval=50, blit=False, repeat=False)
    plt.show()

