import numpy as np
import cupy as cp
import time
import visualization

#GPU-accelerated 3D Beam Propagation Method (FFT BPM)

def bpm_loop_gpu(Beam, vH, mxyn, ncl, k0, zd, mABC, vz, vZslice):

    """
    GPU-accelerated Beam Propagation Method (BPM).
    :param Beam: 2D Input beam profile
    :param vH: Free-space transfer function for a half-step
    :param mxyn: 2D Refractive index profile
    :param ncl: cladding index
    :param k0: vacuum wavevector [1/m]
    :param zd: step size over z [m]
    :param mABC: 2D Distribution of ABC
    :param vz: z space vector [m]
    :param vZslice: Z slice space vector [m]
    :return: mxyzBeam: 3D distribution of the complex beam amplitude (Solution)
    """

    # Transfer data to GPU
    Beam_gpu = cp.asarray(Beam)
    vH_gpu = cp.asarray(vH)
    mxyn_gpu = cp.asarray(mxyn)
    mABC_gpu = cp.asarray(mABC)

    # Initialize GPU memory for storing beam propagation results
    mxyzBeam_gpu = cp.zeros((Beam.shape[0], Beam.shape[1], len(vZslice)), dtype=cp.complex64)
    mxyzBeam_gpu[:, :, 0] = Beam_gpu

    nZslice = 1
    for iz in range(1, len(vz)):
        # FIRST HALF-STEP FREE-SPACE PROPAGATION (DIFFRACTION)
        FBeam = cp.fft.fft2(Beam_gpu)
        FBeam *= cp.fft.fftshift(vH_gpu)
        Beam_gpu = cp.fft.ifft2(FBeam)

        # STEP PROPAGATION THROUGH INHOMOGENEOUS MEDIUM
        # constant 2D refractive index profile is used (mxyn) for the fiber simulation
        Beam_gpu *= cp.exp(-1j * k0 * zd * (mxyn_gpu - ncl))

        # SECOND HALF-STEP FREE-SPACE PROPAGATION (DIFFRACTION)
        FBeam = cp.fft.fft2(Beam_gpu)
        FBeam *= cp.fft.fftshift(vH_gpu)
        Beam_gpu = cp.fft.ifft2(FBeam)

        # ABSORBING BOUNDARY CONDITIONS
        Beam_gpu *= mABC_gpu

        # COLLECT SOLUTION
        # Store the beam at specific z-slices for later visualization
        if vz[iz] >= vZslice[nZslice]:
            mxyzBeam_gpu[:, :, nZslice] = Beam_gpu
            nZslice += 1

    # Transfer final result back to CPU
    return cp.asnumpy(mxyzBeam_gpu)



#################################################################
#INITIALIZE GRID, BEAM, REFRACTIVE INDEX PROFILE, ABC
#################################################################

# TRANSVERSE GRID X
x_window = 80e-6  # window size over x [m]
x_points = 256  # number of points
xd = x_window / x_points  # step size over x [m]
vx = ((np.arange(1, x_points + 1) - (x_points + 1) / 2) * xd)  # x space vector [m]
vKx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(x_points, xd))  # spatial frequency [1/m]

# TRANSVERSE GRID Y
y_window = 80e-6  # window size over y [m]
y_points = 256  # number of points
yd = y_window / y_points  # step size over y [m]
vy = ((np.arange(1, y_points + 1) - (y_points + 1) / 2) * yd)  # y space vector [m]
vKy = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(y_points, yd))  # spatial frequency [1/m]

mx, my = np.meshgrid(vx, vy)
mKx, mKy = np.meshgrid(vKx, vKy)

# LONGITUDINAL GRID Z
z_window = 4e-3  # window size over z [m]
z_points = 2001  # number of points
zd = z_window / z_points  # step size over z [m]
vz = np.linspace(0, z_window, z_points)  # z space vector [m]

# Samples (Slices) vector to collect solution for visualization
z_slice_point = 251
vZslice = np.linspace(0, z_window, z_slice_point)  # Z slice space vector [m]

# INITIAL BEAM
waist = 3e-6  # Beam waist [m]
P0 = 0.1e-3  # Power of Input beam [W]
A0 = np.sqrt(2 * P0 / (np.pi * waist ** 2))  # amplitude of Input beam
intcoeff = 1e-4 # intensity coefficient [cm ** 2]

x_shift = 2e-6  # shift of the input beam over x [m]
y_shift = 0  # shift of the input beam over y [m]
Beam = A0 * np.exp(-(((mx - x_shift) / waist) ** 2 + ((my - y_shift) / waist) ** 2))  # Gaussian beam amplitude

# REFRACTIVE INDEX PROFILE
ncr = 1.503  # core index
ncl = 1.500  # cladding index
dcr = 10e-6  # waveguide width [m]
L0 = 1e-6  # vacuum wavelength [m]
k0 = (2 * np.pi) / L0  # vacuum wavevector [1/m]
B = k0 * ncl  # reference wavevector [1/m]

# Initial 2D XY Refractive index profile (Super-Gaussian)
mxyn = ncl + (ncr - ncl) * np.exp(-np.log(2) * ((2 * mx / dcr) ** 2 + (2 * my / dcr) ** 2) ** 40)

# Initialize 3D Beam profile (solution)
mxyzBeam = np.zeros((x_points, y_points, z_slice_point), dtype=complex)
mxyzBeam[:, :, 0] = Beam

# ABSORBING BOUNDARY CONDITIONS (ABC)
wABC = 2 * np.max(vx) * (1 - 0.15)  # 15 percent cutoff
wMin = 0.985
mABC = wMin + (1 - wMin) * np.exp(-np.log(2) * ((2 * mx / wABC) ** 2 + (2 * my / wABC) ** 2) ** 7)

# Free-space transfer function for a half-step
vH = np.exp(1j / (4 * B) * (mKx ** 2 + mKy ** 2) * zd)


if __name__ == '__main__':

    #################################################################
    # ITERATIVE BPM LOOP
    #################################################################

    start_time = time.perf_counter()

    #run BPM simulations
    mxyzBeam = bpm_loop_gpu(Beam, vH, mxyn, ncl, k0, zd, mABC, vz, vZslice)

    end_time = time.perf_counter()
    print(f"elapsed time: {end_time - start_time}, s")

    #################################################################
    # VISUALIZATION OF SOLUTION
    #################################################################

    #static plot
    visualization.visualize_solution(
        mxyzBeam,
        vx,
        vy,
        vZslice,
        [0, 0, dcr * 1e6],
        [-16, 16],
        [-16, 16],
        [0, 0]
    )

    #animated plot
    # visualization.animate_solution(
    #     mxyzBeam,
    #     vx,
    #     vy,
    #     vZslice,
    #     [0, 0, dcr * 1e6],
    #     [-16, 16],
    #     [-16, 16],
    #     [0, 0]
    # )

