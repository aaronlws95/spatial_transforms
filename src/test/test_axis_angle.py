import numpy as np

import src.spatial_transform as st

class TestAxisAngle:
    def test_to_rotmat(self):
        aa1 = st.AxisAngle(0.5675524,  0.2904527, -0.7704035, 1.5244035)
        r1 = aa1.to_rotmat()
        ans = st.RotMat()
        ans.set_rotmat(0.3535534,  0.9267767, -0.1268265, -0.6123725,  0.1268265, -0.7803301, -0.7071068,  0.3535534,  0.6123725)
        assert r1 == ans

    def test_to_quat(self):
        aa1 = st.AxisAngle(0.5675524,  0.2904527, -0.7704035, 1.5244035)
        q1 = aa1.to_quat().round(3)
        ans = st.Quaternion(0.723,  0.392, 0.201, -0.532)
        assert q1 == ans

    def test_to_euler(self):
        aa1 = st.AxisAngle(0.5675524,  0.2904527, -0.7704035, 1.5244035)
        a1 = aa1.to_euler()
        ans = st.EulerAngle(0.5235988, 0.7853981, -1.0471976)
        assert a1 == ans