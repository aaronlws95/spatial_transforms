import numpy as np

import src.spatial_transform as st

class TestEulerAngle:
    def test_to_rad(self):
        deg = st.EulerAngle(30, 45, -60)
        rad = deg.to_rad()
        ans = st.EulerAngle(0.523599, 0.785398, -1.0472)
        assert rad == ans

    def test_to_deg(self):
        rad = st.EulerAngle(0.523599, 0.785398, -1.0472)
        deg = rad.to_deg()
        ans = st.EulerAngle(30, 45, -60)
        assert deg == ans

    def test_to_quat(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        q1 = a1.to_quat().round(3)
        ans = st.Quaternion(0.723,  0.392, 0.201, -0.532)
        assert q1 == ans

    def test_to_rotmat(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        r1 = a1.to_rotmat()
        ans = st.RotMat()
        ans.set_rotmat(0.3535534,  0.9267767, -0.1268265, -0.6123725,  0.1268265, -0.7803301, -0.7071068,  0.3535534,  0.6123725)
        assert r1 == ans

    def test_to_axis_angle(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        aa1 = a1.to_axis_angle()
        ans = st.AxisAngle(0.5675524,  0.2904527, -0.7704035, 1.5244035)
        assert aa1 == ans