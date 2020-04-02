import numpy as np

import src.spatial_transform as st

class TestRotMat:
    def test_to_euler(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        r1 = a1.to_rotmat()
        ans = r1.to_euler()
        assert a1 == ans

    def test_to_quat(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        r1 = a1.to_rotmat()
        q1 = r1.to_quat().round(3)
        ans = st.Quaternion(0.723,  0.392, 0.201, -0.532)
        assert q1 == ans

    def test_to_axis_angle(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        r1 = a1.to_rotmat()
        aa1 = r1.to_axis_angle().round(7)
        ans = st.AxisAngle(0.5675524,  0.2904527, -0.7704035, 1.5244035)
        assert aa1 == ans