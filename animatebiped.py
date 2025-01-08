
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate_biped(angles,filename):
    """
    Animate a 2D bipedal robot given an n*4 array of angles.
    
    Parameters:
    -----------
    angles : np.ndarray
        A numpy array of shape (n,4), where n is the number of time steps.
        Each row: [right_shank_angle, right_thigh_angle, left_shank_angle, left_thigh_angle]
    """
    num_frames = angles.shape[0]
    seg_length = 1.0

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True)

    # Lines
    torso_line, = ax.plot([], [], 'k-', lw=2)
    right_thigh_line, = ax.plot([], [], 'b-', lw=2)
    right_shank_line, = ax.plot([], [], 'r-', lw=2)
    left_thigh_line, = ax.plot([], [], 'g-', lw=2)
    left_shank_line, = ax.plot([], [], 'm-', lw=2)

    # Scatter points for joints
    hip_marker = ax.scatter([], [], c='black', s=50)
    right_knee_marker = ax.scatter([], [], c='blue', s=50)
    right_foot_marker = ax.scatter([], [], c='red', s=50)
    left_knee_marker = ax.scatter([], [], c='green', s=50)
    left_foot_marker = ax.scatter([], [], c='magenta', s=50)

    # Initialization function for FuncAnimation
    def init():
        torso_line.set_data([], [])
        right_thigh_line.set_data([], [])
        right_shank_line.set_data([], [])
        left_thigh_line.set_data([], [])
        left_shank_line.set_data([], [])

        # Set empty offsets as a (0,2) shaped array
        hip_marker.set_offsets(np.empty((0, 2)))
        right_knee_marker.set_offsets(np.empty((0, 2)))
        right_foot_marker.set_offsets(np.empty((0, 2)))
        left_knee_marker.set_offsets(np.empty((0, 2)))
        left_foot_marker.set_offsets(np.empty((0, 2)))

        return (torso_line, right_thigh_line, right_shank_line, 
                left_thigh_line, left_shank_line, hip_marker, 
                right_knee_marker, right_foot_marker, 
                left_knee_marker, left_foot_marker)

    def update(frame):
        ax.set_title(f"Time Step: {frame + 1}")
        r_thigh_angle, r_shank_angle, l_thigh_angle, l_shank_angle = angles[frame]

        # Hip at origin
        hip_pos = np.array([0.0, 0.0])
        # Simple torso line upwards
        torso_top = np.array([0.0, 1.0])
        torso_line.set_data([torso_top[0], hip_pos[0]], [torso_top[1], hip_pos[1]])

        # Right leg
        right_knee = hip_pos + seg_length * np.array([np.sin(r_thigh_angle), -np.cos(r_thigh_angle)])
        right_foot = right_knee + seg_length * np.array([np.sin(r_thigh_angle + r_shank_angle), -np.cos(r_thigh_angle + r_shank_angle)])
        right_thigh_line.set_data([hip_pos[0], right_knee[0]], [hip_pos[1], right_knee[1]])
        right_shank_line.set_data([right_knee[0], right_foot[0]], [right_knee[1], right_foot[1]])

        # Left leg
        left_knee = hip_pos + seg_length * np.array([np.sin(l_thigh_angle), -np.cos(l_thigh_angle)])
        left_foot = left_knee + seg_length * np.array([np.sin(l_thigh_angle + l_shank_angle), -np.cos(l_thigh_angle + l_shank_angle)])
        left_thigh_line.set_data([hip_pos[0], left_knee[0]], [hip_pos[1], left_knee[1]])
        left_shank_line.set_data([left_knee[0], left_foot[0]], [left_knee[1], left_foot[1]])

        # Update markers with a 2D array shape
        hip_marker.set_offsets([hip_pos])
        right_knee_marker.set_offsets([right_knee])
        right_foot_marker.set_offsets([right_foot])
        left_knee_marker.set_offsets([left_knee])
        left_foot_marker.set_offsets([left_foot])

        return (torso_line, right_thigh_line, right_shank_line, 
                left_thigh_line, left_shank_line, hip_marker, 
                right_knee_marker, right_foot_marker, 
                left_knee_marker, left_foot_marker)

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=1, blit=False)
    ani.save(filename, fps=100)