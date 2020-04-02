import numpy as np

import src.spatial_transform as st

class TestQuaternion:
    def test_to_euler(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        q1 = a1.to_quat()
        ans = q1.to_euler()
        assert a1 == ans

    def test_to_rotmat(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        q1 = a1.to_quat()
        r1 = q1.to_rotmat()
        ans = st.RotMat()
        ans.set_rotmat(0.3535534,  0.9267767, -0.1268265, -0.6123725,  0.1268265, -0.7803301, -0.7071068,  0.3535534,  0.6123725)
        assert r1 == ans

    def test_to_axis_angle(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        q1 = a1.to_quat()
        aa1 = q1.to_axis_angle()
        ans = st.AxisAngle(0.5675524,  0.2904527, -0.7704035, 1.5244035)
        assert aa1 == ans

    def test_mult(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        q1 = a1.to_quat()
        a2 = st.EulerAngle(30, 0, 0).to_rad()
        q2 = a2.to_quat()
        q3 = q1.quat_mult(q2)
        a3 = q3.to_euler()
        ans = a1 + a2
        assert a3 == ans

    def test_inverse_mult(self):
        a1 = st.EulerAngle(30, 45, -60).to_rad()
        q1 = a1.to_quat()
        a2 = st.EulerAngle(30, 0, 0).to_rad()
        q2 = a2.to_quat()
        q3 = q1.quat_mult(q2)
        q4 = q3.quat_mult(q2.inverse())
        a3 = q4.to_euler()
        assert a3 == a1