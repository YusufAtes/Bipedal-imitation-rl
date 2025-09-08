
import numpy as np
import os
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle
def create_noisy_plane(gamma,omega,count, row_size = 32, col_size = 1024,simulation_res = 0.05):    #5 cm resolutions defined for the simulation
    # Create the plane
    """
    Create a heightfield plane with noise.
    Gamma is the ground resolution, omega is the noise level.
    The plane is created with a cubic spline interpolation.
    The plane is created with a resolution of 5 cm (in simulation).
    The plane is created with a size of 32 rows and 1024 columns.
    """
    full_plane = np.zeros(col_size)
    end_point = col_size * simulation_res 
    mid_point = end_point / 2
    safety_margin = 0.25 # 25 cm safety margin
    plane_coarse = np.arange(mid_point , end_point, gamma)
    plane_fine = np.arange(mid_point  , end_point, simulation_res)
    plane = np.zeros(len(plane_coarse))
    prev_height = 0.0

    for i in range(len(plane_coarse)-1):
        #truncated normal noise
        noise = np.random.normal()  # Adjust the standard
        noise = np.clip(noise, -omega, omega)  # Clip to a range
        height = prev_height + noise
        plane[i+1] = height
        prev_height = height
    cs = CubicSpline(plane_coarse, plane, bc_type='natural')
    full_plane[-len(plane_fine):] = cs(plane_fine)

    plane_x_axis = np.arange(0, 25.6, simulation_res)
    
    # # Generate dictionary mapping x-axis to height values
    # plane_dict = dict(zip(plane_x_axis, full_plane[-int(len(full_plane)/2):]))
    # # Save dictionary as .npy and .png (using same base name)
    # np.save(f"noise_planes/plane_{gamma}_{count}_dict.npy", plane_dict)

    # create a plot of the plane
    plt.figure(figsize=(10, 5))
    plt.ylim(-0.1, 0.1)
    plt.xlim(0, 10)
    plt.plot(plane_x_axis, full_plane[-int(len(full_plane)/2):])
    plt.title(f"Heightfield Plane for Resolution {gamma}")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.grid()
    plt.savefig(f"noise_planes/plane_{gamma}_{count}.png")
    plt.close()
    full_plane_data = np.repeat(full_plane, row_size)  # Repeat the plane data for each row
    np.save(f"noise_planes/plane_{gamma}_{count}.npy", full_plane_data)

    return full_plane_data

if __name__ == "__main__":
    # Example usage
    gammas = [0.25,0.5, 1.0, 1.5, 2.0]  # resolution in meters
    omega = 0.01
    for gamma in gammas:
        for scenario in range(4):
            heightfield_data = create_noisy_plane(gamma, omega,scenario)
            print("Heightfield data created with shape:", heightfield_data.shape)