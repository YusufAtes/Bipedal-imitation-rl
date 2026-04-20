
import numpy as np
import os
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle

def create_noisy_plane(omega,count, row_size = 32, col_size = 1024,simulation_res = 0.05):    #5 cm resolutions defined for the simulation
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
    safety_margin = 0.25 # 25 cm safety margin
    mid_point = end_point / 2 + safety_margin
    org_noise = np.random.normal(0, omega, size=1000)

    for gamma in [0.25,0.5, 1.0, 1.5, 2.0]:  # resolution in meters
        plane_coarse = np.arange(mid_point , end_point, gamma)
        plane_fine = np.arange(mid_point  , end_point, simulation_res)
        plane = np.zeros(len(plane_coarse))
        prev_height = 0.0

        for i in range(len(plane_coarse)-1):
            #truncated normal noise
            noise = org_noise[i]
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

def create_step_plane(count, row_size=32, col_size=1024, simulation_res=0.05):
    """
    Create a stair-like heightfield plane.
    Similar to create_noisy_plane, this function generates and saves one plane
    per gamma in [0.25, 0.5, 1.0, 2.0].
    After the safety margin, each step is randomly flat, +1 cm, or -1 cm
    relative to the previous step. Each gamma gets its own random stair profile.
    """
    end_point = col_size * simulation_res
    safety_margin = 0.25  # 25 cm safety margin
    for gamma in [0.25, 0.5, 1.0, 2.0]:
        full_plane = np.zeros(col_size)

        # Coarse x positions define where the step value can change.
        plane_coarse = np.arange(safety_margin, end_point, gamma)
        if len(plane_coarse) == 0 or plane_coarse[-1] < end_point:
            plane_coarse = np.append(plane_coarse, end_point)

        # Height value for each coarse segment.
        step_heights = np.zeros(len(plane_coarse))
        for i in range(1, len(plane_coarse)):
            delta_h = np.random.choice([-0.01, 0.0, 0.01])  # -1cm, flat, +1cm
            step_heights[i] = step_heights[i - 1] + delta_h

        # Fill fine-resolution plane with piecewise-constant stair values.
        for i in range(len(plane_coarse) - 1):
            start_x = plane_coarse[i]
            end_x = plane_coarse[i + 1]
            start_idx = int(np.floor(start_x / simulation_res))
            end_idx = int(np.floor(end_x / simulation_res))
            full_plane[start_idx:end_idx] = step_heights[i]

        # Ensure tail uses the last available step height.
        if len(plane_coarse) > 1:
            last_idx = int(np.floor(plane_coarse[-2] / simulation_res))
            full_plane[last_idx:] = step_heights[-2]

        plane_x_axis = np.arange(0, 25.6, simulation_res)

        plt.figure(figsize=(10, 5))
        plt.ylim(-0.2, 0.2)
        plt.xlim(0, 10)
        plt.step(plane_x_axis, full_plane[-int(len(full_plane) / 2):], where="post")
        plt.title(f"Stair Plane for Resolution {gamma}")
        plt.xlabel("Distance (m)")
        plt.ylabel("Height (m)")
        plt.grid()
        plt.savefig(f"noise_planes/plane_step_{gamma}_{count}.png")
        plt.close()

        full_plane_data = np.repeat(full_plane, row_size)
        np.save(f"noise_planes/plane_step_{gamma}_{count}.npy", full_plane_data)

    return full_plane_data

if __name__ == "__main__":
    # Example usage
    omega = 0.01

    for scenario in range(1):
        # heightfield_data = create_noisy_plane(omega,scenario)
        heightfield_data = create_step_plane(scenario)
        print("Heightfield data created with shape:", heightfield_data.shape)