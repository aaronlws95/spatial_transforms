from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import animation
from pathlib import Path

import src.spatial_transform as st
import src.utils as utils

# Setup
start_pt = st.Vector3D(0.5, 0.5, 0.5)
start_euler = st.EulerAngle(0, 0, 0).to_rad()
start_quat = start_euler.to_quat()

end_euler = st.EulerAngle(180, 45, -60).to_rad()
end_quat = end_euler.to_quat()

# Slerp
slerp = utils.quat_slerp(start_quat.numpy(), end_quat.numpy(), 0, 1, 0.05)

# Calculate points and directions for plotting
pos_pts = [start_pt]
x_dir_pts = [st.Vector3D(1, 0, 0)]
y_dir_pts = [st.Vector3D(0, 1, 0)]
z_dir_pts = [st.Vector3D(0, 0, 1)]
euler_angles = [start_euler.to_deg()]

new_quat = st.Quaternion()
new_quat.from_numpy(slerp[-1])
new_y_dir = y_dir_pts[-1].rotate_by_quat(new_quat)

for quat in slerp:

    new_quat = st.Quaternion()
    new_quat.from_numpy(quat)
    new_pt = pos_pts[0].rotate_by_quat(new_quat)
    new_x_dir = x_dir_pts[0].rotate_by_quat(new_quat)
    new_y_dir = y_dir_pts[0].rotate_by_quat(new_quat)
    new_z_dir = z_dir_pts[0].rotate_by_quat(new_quat)
    euler_angles.append(new_quat.to_euler().to_deg())

    pos_pts.append(new_pt)
    x_dir_pts.append(new_x_dir)
    y_dir_pts.append(new_y_dir)
    z_dir_pts.append(new_z_dir)

pos_pts = np.asarray(pos_pts)
x_dir_pts = np.asarray(x_dir_pts)
y_dir_pts = np.asarray(y_dir_pts)
z_dir_pts = np.asarray(z_dir_pts)

# Plot setup
fig = plt.figure()
ax = Axes3D(fig)
txt = fig.suptitle('')

# Plots
num = 0
line_length = 1

def init_plt():
    # Clear previous plots
    plt.cla()

    # Axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set limit
    lim = 1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # Origin axis
    ax.quiver(0, 0, 0, 1, 0, 0, color='black', alpha=0.5, length=1, normalize=True)
    ax.quiver(0, 0, 0, 0, 1, 0, color='black', alpha=0.5, length=1, normalize=True)
    ax.quiver(0, 0, 0, 0, 0, 1, color='black', alpha=0.5, length=1, normalize=True)

# Animation function
def update_points(num):
    txt.set_text('x={:0.3f}, y={:0.3f}, z={:0.3f}, roll={:0.3f}, pitch={:0.3f}, yaw={:0.3f}'.format(
        pos_pts[num].x, pos_pts[num].y, pos_pts[num].z,
        euler_angles[num].roll, euler_angles[num].pitch, euler_angles[num].yaw))

    init_plt()

    ax.plot([pos_pts[num].x], [pos_pts[num].y], [pos_pts[num].z], 'ro', markersize=1)
    ax.quiver([pos_pts[num].x], [pos_pts[num].y], [pos_pts[num].z],
              [x_dir_pts[num].x], [x_dir_pts[num].y], [x_dir_pts[num].z],
              color='r', length=line_length, normalize=True)

    ax.quiver([pos_pts[num].x], [pos_pts[num].y], [pos_pts[num].z],
              [y_dir_pts[num].x], [y_dir_pts[num].y], [y_dir_pts[num].z],
              color='g', length=line_length, normalize=True)

    ax.quiver([pos_pts[num].x], [pos_pts[num].y], [pos_pts[num].z],
              [z_dir_pts[num].x], [z_dir_pts[num].y], [z_dir_pts[num].z],
              color='b', length=line_length, normalize=True)
    return txt
anim = animation.FuncAnimation(fig, update_points, frames=len(slerp))

# Save as gif
anim.save(str(Path('media')/'vis.gif'), dpi=80, writer='pillow')

plt.show()