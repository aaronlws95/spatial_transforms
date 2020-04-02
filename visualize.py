from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import animation

import src.spatial_transform as st
import src.utils as utils

fig = plt.figure()
ax = Axes3D(fig)
txt = fig.suptitle('')

a1 = st.EulerAngle(30, 45, -60).to_rad()
q1 = a1.to_quat()
a2 = st.EulerAngle(30, 0, 0).to_rad()
q2 = a2.to_quat()

start_pt = st.Quaternion(0, 4, 0, 0)
end_pt = st.Quaternion(0, 0, 0, 4)

start_pt = q1
end_pt = q2

slerp = utils.quat_slerp(start_pt.numpy(), end_pt.numpy(), 0, 1, 0.1)

x_history = []
y_history = []
z_history = []

last_pt_x = start_pt.x
last_pt_y = start_pt.y
last_pt_z = start_pt.z
for pt in slerp:
    x_history.append([last_pt_x, pt[1]])
    y_history.append([last_pt_y, pt[2]])
    z_history.append([last_pt_z, pt[3]])

    last_pt_x = pt[1]
    last_pt_y = pt[2]
    last_pt_z = pt[3]

x_history = np.stack(x_history)
y_history = np.stack(y_history)
z_history = np.stack(z_history)

def update_points(num):
    if num == 0:
        plt.cla()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    txt.set_text('t={:d}, x={:f}, y={:f}, z={:f}'.format(num, slerp[num, 1], slerp[num, 2], slerp[num, 3]))
    ax.plot(x_history[num], y_history[num], z_history[num], 'ro-', markersize=5)
    return txt

anim = animation.FuncAnimation(fig, update_points, frames=len(slerp))
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('line.gif', dpi=80, writer=writer)
plt.show()