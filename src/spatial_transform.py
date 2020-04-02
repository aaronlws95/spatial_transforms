import numpy as np

# Tait-Bryan ZYX (intrinsic)
class EulerAngle:
    def __init__(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __str__(self):
        return '({}, {}, {})'.format(self.roll, self.pitch, self.yaw)

    def __repr__(self):
        return '({}, {}, {})'.format(self.roll, self.pitch, self.yaw)

    def __eq__(self, other):
        if isinstance(other, EulerAngle):
            if np.isclose(self.roll, other.roll) and np.isclose(self.pitch, other.pitch) and np.isclose(self.yaw, other.yaw):
                return True
            else:
                return False

    def __add__(self, val):
        if isinstance(val, EulerAngle):
            return EulerAngle(self.roll + val.roll,
                              self.pitch + val.pitch,
                              self.yaw + val.yaw)
        elif isinstance(val, (int, float)):
            return EulerAngle(self.roll + val,
                              self.pitch + val,
                              self.yaw + val)

    def __sub__(self, val):
        if isinstance(val, EulerAngle):
            return EulerAngle(self.roll - val.roll,
                              self.pitch - val.pitch,
                              self.yaw - val.yaw)
        elif isinstance(val, (int, float)):
            return EulerAngle(self.roll - val,
                              self.pitch - val,
                              self.yaw - val)

    def to_rad(self):
        roll = self.roll * np.pi / 180
        pitch = self.pitch * np.pi / 180
        yaw = self.yaw * np.pi / 180

        return EulerAngle(roll, pitch, yaw)

    def to_deg(self):
        roll = self.roll * 180 / np.pi
        pitch = self.pitch * 180 / np.pi
        yaw = self.yaw * 180 / np.pi

        return EulerAngle(roll, pitch, yaw)

    def to_rotmat(self):
        rot_x = [[1, 0, 0],
                [0, np.cos(self.roll), -np.sin(self.roll)],
                [0, np.sin(self.roll), np.cos(self.roll)]]
        rot_x = np.asarray(rot_x)

        rot_y = [[np.cos(self.pitch), 0, np.sin(self.pitch)],
                [0, 1, 0],
                [-np.sin(self.pitch), 0, np.cos(self.pitch)]]
        rot_y = np.asarray(rot_y)

        rot_z = [[np.cos(self.yaw), -np.sin(self.yaw), 0],
                [np.sin(self.yaw), np.cos(self.yaw), 0],
                [0, 0, 1]]
        rot_z = np.asarray(rot_z)

        rotmat = rot_z @ rot_y @ rot_x
        return RotMat(rotmat)

    def to_axis_angle(self):
        x = np.sin(self.roll / 2) * np.cos(self.pitch / 2) * np.cos(self.yaw / 2) \
            - np.cos(self.roll /2) * np.sin(self.pitch /2 ) * np.sin(self.yaw / 2)

        y = np.cos(self.roll / 2) * np.sin(self.pitch / 2) * np.cos(self.yaw / 2) \
            + np.sin(self.roll /2) * np.cos(self.pitch /2 ) * np.sin(self.yaw / 2)

        z = np.cos(self.roll / 2) * np.cos(self.pitch / 2) * np.sin(self.yaw / 2) \
            - np.sin(self.roll / 2) * np.sin(self.pitch / 2) * np.cos(self.yaw / 2)

        angle = 2 * np.arccos(np.cos(self.roll / 2) * np.cos(self.pitch / 2) * np.cos(self.yaw / 2) \
            + np.sin(self.roll / 2) * np.sin(self.pitch / 2) * np.sin(self.yaw / 2))

        return AxisAngle(x, y, z, angle).normalize()

    def to_quat(self):
        w = np.cos(self.roll / 2) * np.cos(self.pitch / 2) * np.cos(self.yaw / 2) \
            + np.sin(self.roll / 2) * np.sin(self.pitch / 2) * np.sin(self.yaw / 2)

        x = np.sin(self.roll / 2) * np.cos(self.pitch / 2) * np.cos(self.yaw / 2) \
            - np.cos(self.roll /2) * np.sin(self.pitch /2 ) * np.sin(self.yaw / 2)

        y = np.cos(self.roll / 2) * np.sin(self.pitch / 2) * np.cos(self.yaw / 2) \
            + np.sin(self.roll /2) * np.cos(self.pitch /2 ) * np.sin(self.yaw / 2)

        z = np.cos(self.roll / 2) * np.cos(self.pitch / 2) * np.sin(self.yaw / 2) \
            - np.sin(self.roll / 2) * np.sin(self.pitch / 2) * np.cos(self.yaw / 2)

        return Quaternion(w, x, y, z)

    def round(self, n):
        return EulerAngle(np.round(self.roll, n), np.round(self.pitch, n), np.round(self.yaw, n))

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, val):
        if isinstance(val, Quaternion):
            return self.quat_add(val)
        elif isinstance(val, (int, float)):
            return self.scalar_add(val)

    def __sub__(self, val):
        if isinstance(val, Quaternion):
            return self.quat_sub(val)
        elif isinstance(val, (int, float)):
            return self.scalar_sub(val)

    def __mul__(self, val):
        if isinstance(val, Quaternion):
            return self.quat_mult(val)
        elif isinstance(val, (int, float)):
            return self.scalar_mult(val)

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            if np.isclose(self.w, other.w) and np.isclose(self.x, other.x) and np.isclose(self.y, other.y) and np.isclose(self.z, other.z):
                return True
            else:
                return False

    def __radd__(self, val):
        if isinstance(val, (int, float)):
            return self.scalar_add(val)

    def __rsub__(self, val):
        if isinstance(val, (int, float)):
            return self.scalar_rsub(val)

    def __rmul__(self, val):
        if isinstance(val, (int, float)):
            return self.scalar_mult(val)

    def __floordiv__(self, val):
        if isinstance(val, (int, float)):
            return self.scalar_floordiv(val)

    def __truediv__(self, val):
        if isinstance(val, (int, float)):
            return self.scalar_truediv(val)

    def __str__(self):
        return '[{}, {}, {}, {}]'.format(self.w, self.x, self.y, self.z)

    def __repr__(self):
        return '[{}, {}, {}, {}]'.format(self.w, self.x, self.y, self.z)

    def scalar_floordiv(self, val):
        return Quaternion(self.w // val, self.x // val, self.y // val, self.z // val)

    def scalar_truediv(self, val):
        return Quaternion(self.w / val, self.x / val, self.y / val, self.z / val)

    def quat_add(self, quat):
        return Quaternion(self.w + quat.w, self.x + quat.x, self.y + quat.y, self.z + quat.z)

    def quat_sub(self, quat):
        return Quaternion(self.w - quat.w, self.x - quat.x, self.y - quat.y, self.z - quat.z)

    def scalar_rsub(self, val):
        return Quaternion(val - w, val - x, val - y, val - z)

    def scalar_sub(self, val):
        return Quaternion(self.w - val, self.x - val, self.y - val, self.z - val)

    def scalar_add(self, val):
        return Quaternion(self.w + val, self.x + val, self.y + val, self.z + val)

    def scalar_mult(self, val):
        return Quaternion(self.w * val, self.x * val, self.y * val, self.z * val)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.y)

    def norm(self):
        return np.sqrt(1 / (self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2))

    def inverse(self):
        conj = self.conjugate()
        scale = 1 / (self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        return scale * conj

    def dot(self, quat):
        return quat.w * self.w + quat.x * self.x + quat.y * self.y + quat.z * self.z

    def numpy(self):
        return np.asarray([self.w, self.x, self.y, self.z])

    def quat_mult(self, quat):
        w = quat.w * self.w - quat.x * self.x - quat.y * self.y - quat.z * self.z
        x = quat.w * self.x + quat.x * self.w + quat.z * self.y - quat.y * self.z
        y = quat.y * self.w - quat.z * self.x + quat.w * self.y + quat.x * self.z
        z = quat.z * self.w + quat.y * self.x - quat.x * self.y + quat.w * self.z
        return Quaternion(w, x, y, z)

    def to_rotmat(self):
        r00 = 1 - 2 * self.y**2 - 2 * self.z**2
        r01 = 2 * self.x * self.y - 2 * self.z * self.w
        r02 = 2 * self.x * self.z + 2 * self.y * self.w

        r10 = 2 * self.x * self.y + 2 * self.z * self.w
        r11 = 1 - 2 * self.x**2 - 2 * self.z**2
        r12 = 2 * self.y * self.z - 2 * self.x * self.w

        r20 = 2 * self.x * self.z - 2 * self.y * self.w
        r21 = 2 * self.y * self.z + 2 * self.x * self.w
        r22 = 1 - 2 * self.x**2 - 2 * self.y**2

        rotmat = [[r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22]]
        rotmat = np.asarray(rotmat)

        return RotMat(rotmat)

    def to_euler(self):
        t0 = 2.0 * (self.w * self.x + self.y * self.z)
        t1 = 1.0 - 2.0 * (self.x**2 + self.y**2)
        roll = np.arctan2(t0, t1)

        t2 = 2.0 * (self.w * self.y - self.z * self.x)
        t2 = 1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)

        t3 = 2.0 * (self.x * self.y + self.w * self.z)
        t4 = 1.0 - 2.0 * (self.y**2 + self.z**2)
        yaw = np.arctan2(t3, t4)

        return EulerAngle(roll, pitch, yaw)

    def round(self, n):
        return Quaternion(np.round(self.w, n), np.round(self.x, n), np.round(self.y, n), np.round(self.z, n))

    def normalize(self):
        mag = self.norm()
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def to_axis_angle(self):
        mag = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        angle = 2 * np.arccos(self.w)
        return AxisAngle(self.x / mag, self.y / mag, self.z / mag, angle)

class RotMat:
    def __init__(self, rotmat=np.identity(3)):
        self.matrix = rotmat

    def set_rotmat(self, r00, r01, r02, r10, r11, r12, r20, r21, r22):
        rotmat = [[r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22]]
        self.matrix = np.asarray(rotmat)

    def __str__(self):
        return str(self.matrix)

    def __repr__(self):
        return str(self.matrix)

    def __eq__(self, other):
        if isinstance(other, RotMat):
            if np.isclose(self.matrix, other.matrix).all():
                return True
            else:
                return False

    def inv(self):
        return np.linalg.inv(self.matrix)

    def __matmul__(self, val):
        if isinstance(val, RotMat):
            return self.matrix @ RotMat.matrix

    def to_euler(self):
        sy = np.sqrt(self.matrix[0, 0] * self.matrix[0, 0] +  self.matrix[1, 0] * self.matrix[1, 0])
        singular = sy < 1e-6

        if not singular :
            roll = np.arctan2(self.matrix[2, 1] , self.matrix[2, 2])
            pitch = np.arctan2(-self.matrix[2, 0], sy)
            yaw = np.arctan2(self.matrix[1, 0], self.matrix[0, 0])
        else :
            roll = np.arctan2(-self.matrix[1, 2], self.matrix[1, 1])
            pitch = np.arctan2(-self.matrix[2, 0], sy)
            yaw = 0

        return EulerAngle(roll, pitch, yaw)

    def to_quat(self):
        tr = self.matrix[0, 0] + self.matrix[1, 1] + self.matrix[2, 2]

        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (self.matrix[2, 1] - self.matrix[1, 2]) / s
            qy = (self.matrix[0, 2] - self.matrix[2, 0]) / s
            qz = (self.matrix[1, 0] - self.matrix[0, 1]) / s
        elif self.matrix[0, 0] > self.matrix[1, 1] and self.matrix[0, 0] > self.matrix[2, 2]:
            s = np.sqrt(1.0 + self.matrix[0, 0] - self.matrix[1, 1] - self.matrix[2, 2]) * 2
            qw = (self.matrix[2, 1] - self.matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (self.matrix[0, 1] + self.matrix[1, 0]) / s
            qz = (self.matrix[0, 2] + self.matrix[2, 0]) / s
        elif self.matrix[1, 1] > self.matrix[2, 2]:
            s = np.sqrt(1.0 + self.matrix[1, 1] - self.matrix[0, 0] - self.matrix[2, 2]) * 2
            qw = (self.matrix[0, 2] - self.matrix[2, 0]) / s
            qx = (self.matrix[0, 1] + self.matrix[1, 0]) / s
            qy =  0.25 * s
            qz = (self.matrix[1, 2] + self.matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + self.matrix[2, 2] - self.matrix[0, 0] - self.matrix[1, 1]) * 2
            qw = (self.matrix[1, 0] - self.matrix[0, 1]) / s
            qx = (self.matrix[0, 2] + self.matrix[2, 0]) / s
            qy = (self.matrix[1, 2] + self.matrix[2, 1]) / s
            qz =  0.25 * s

        return Quaternion(qw, qx, qy, qz)

    def to_axis_angle(self):
        angle = np.arccos((self.matrix[0, 0] + self.matrix[1, 1] + self.matrix[2, 2] - 1) / 2)
        sn = 2 * np.sin(angle)
        x = ((self.matrix[2, 1] - self.matrix[1, 2]) / sn)
        y = ((self.matrix[0, 2] - self.matrix[2, 0]) / sn)
        z = ((self.matrix[1, 0] - self.matrix[0, 1]) / sn)
        return AxisAngle(x, y, z, angle)

class AxisAngle:
    def __init__(self, x, y, z, angle):
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle

    def __str__(self):
        return 'x: {}, y: {}, z: {}, angle: {}'.format(self.x, self.y, self.z, self.angle)

    def __repr__(self):
        return 'x: {}, y: {}, z: {}, angle: {}'.format(self.x, self.y, self.z, self.angle)

    def to_rad():
        self.angle = self.angle * np.pi / 180

    def to_deg():
        self.angle = self.angle * 180 / np.pi

    def __eq__(self, other):
        if isinstance(other, AxisAngle):
            if np.isclose(self.x, other.x) and np.isclose(self.y, other.y) and np.isclose(self.z, other.z) and np.isclose(self.angle, other.angle):
                return True
            else:
                return False

    def normalize(self):
        mag = np.sqrt(self.x**2 + self.y**2 + self.z**2) + 1e-10
        return AxisAngle(self.x / mag, self.y / mag, self.z / mag, self.angle)

    def to_rotmat(self):
        cs = np.cos(self.angle)
        sn = np.sin(self.angle)
        r00 = (1 - cs) * self.x * self.x + cs
        r01 = (1 - cs) * self.x * self.y - self.z * sn
        r02 = (1 - cs) * self.x * self.z + self.y * sn
        r10 = (1 - cs) * self.y * self.x + self.z * sn
        r11 = (1 - cs) * self.y * self.y + cs
        r12 = (1 - cs) * self.y * self.z - self.x * sn
        r20 = (1 - cs) * self.z * self.x - self.y * sn
        r21 = (1 - cs) * self.z *self.y + self.x * sn
        r22 = (1 -cs) * self.z * self.z + cs

        rotmat = [[r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22]]
        rotmat = np.asarray(rotmat)

        return RotMat(rotmat)

    def to_quat(self):
        norm = self.normalize()
        w = np.cos(norm.angle / 2)
        sn = np.sin(norm.angle / 2)

        x = norm.x * sn
        y = norm.y * sn
        z = norm.z * sn

        return Quaternion(w, x, y, z)

    def to_euler(self):
        norm = self.normalize()
        w = np.cos(norm.angle / 2)
        sn = np.sin(norm.angle / 2)
        x = norm.x * sn
        y = norm.y * sn
        z = norm.z * sn

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x**2 + y**2)
        roll = np.arctan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)

        t3 = 2.0 * (x * y + w * z)
        t4 = 1.0 - 2.0 * (y**2 + z**2)
        yaw = np.arctan2(t3, t4)

        return EulerAngle(roll, pitch, yaw)

    def round(self, n):
        return AxisAngle(np.round(self.x, n), np.round(self.y, n), np.round(self.z, n), np.round(self.angle, n))