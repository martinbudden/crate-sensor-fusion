use num_traits::{ConstOne, ConstZero};
use vqm::{MathMethods, Matrix3x3f32, RollPitchYawf32, UnitQuaternionf32, Vector3f32};

/// Converts acceleration from sensor frame to NED frame, and integrates it
/// to give `delta_pos` and `delta_vel` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuIntegrator {
    dt_sum: f32,
    count: u32,
    denominator: u32,
    acc_bias_ned_mps2: Vector3f32,
    integral: ImuIntegral,
}

impl Default for ImuIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ImuIntegrator {
    const DEFAULT_DENOMINATOR: u32 = 16;

    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dt_sum: 0.0,
            count: 0,
            denominator: Self::DEFAULT_DENOMINATOR,
            acc_bias_ned_mps2: Vector3f32::ZERO,
            integral: ImuIntegral::new(),
        }
    }

    /// Set the `denominator` of a newly constructed `ImuIntegrator`.
    #[must_use]
    pub fn with_denominator(mut self, denominator: u32) -> Self {
        self.denominator = denominator;
        self
    }

    pub fn set_denominator(&mut self, denominator: u32) {
        self.denominator = denominator;
    }

    pub fn set_acc_bias(&mut self, acc_bias_ned_mps2: Vector3f32) {
        self.acc_bias_ned_mps2 = acc_bias_ned_mps2;
    }

    #[inline]
    pub fn reset(&mut self) {
        self.count = 0;
        self.dt_sum = 0.0;
        self.integral.reset();
    }

    #[inline]
    #[must_use]
    pub fn integration_complete(&self) -> bool {
        self.count >= self.denominator
    }

    /// Integrates acceleration over one IMU sample interval.
    ///
    /// Uses simple Euler integration, assuming acceleration is constant over `dt`.
    /// Generally called inside the main IMU execution loop (eg `gyro_pid` loop).
    ///
    /// # Assumptions
    /// - `acc_body_mps2`, `orientation`, and `dt` have already been validated by the caller.
    /// - `acc_body_mps2` is in the body frame.
    /// - The caller does not integrate beyond interval completion.
    #[inline]
    pub fn integrate(&mut self, acc_body_mps2: Vector3f32, orientation: UnitQuaternionf32, dt: f32) {
        debug_assert!(acc_body_mps2.is_finite());
        debug_assert!(dt > 0.0);
        debug_assert!(dt.is_finite());

        self.count += 1;
        self.integral.integrate(acc_body_mps2, self.acc_bias_ned_mps2, orientation, dt);
    }

    #[must_use]
    pub fn imu_integral(&self) -> ImuIntegral {
        self.integral
    }
}

/// Integral values for IMU integrator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuIntegral {
    pub dt_sum: f32,
    pub delta_pos_ned_m: Vector3f32,
    pub delta_vel_ned_mps: Vector3f32,
    pub orientation: UnitQuaternionf32,
    // Jacobians (ie matrices of partial derivatives).
    /// Sensitivity of integrated position to a common NED-frame small-angle orientation error, units m/radian.
    orientation_pos_jacobian: Matrix3x3f32,
    /// Sensitivity of integrated velocity to a common NED-frame small-angle orientation error, units m/s/radian.
    orientation_vel_jacobian: Matrix3x3f32,
}

//#[cfg(not(any(feature = "servos", feature = "eight_motors")))]
const _: () = assert!(size_of::<ImuIntegral>() == 128);

impl Default for ImuIntegral {
    fn default() -> Self {
        Self::new()
    }
}

impl ImuIntegral {
    const G_MPS2: f32 = 9.80665;
    // NED: positive Z points down.
    const GRAVITY_NED_MPS2: Vector3f32 = Vector3f32 { x: 0.0, y: 0.0, z: Self::G_MPS2 };

    #[must_use]
    pub const fn new() -> Self {
        Self {
            dt_sum: 0.0,
            delta_pos_ned_m: Vector3f32::ZERO,
            delta_vel_ned_mps: Vector3f32::ZERO,
            orientation: UnitQuaternionf32::ONE,
            orientation_pos_jacobian: Matrix3x3f32::ZERO,
            orientation_vel_jacobian: Matrix3x3f32::ZERO,
        }
    }
    #[inline]
    pub fn reset(&mut self) {
        self.dt_sum = 0.0;
        self.delta_vel_ned_mps = Vector3f32::ZERO;
        self.delta_pos_ned_m = Vector3f32::ZERO;
        self.orientation_pos_jacobian = Matrix3x3f32::ZERO;
        self.orientation_vel_jacobian = Matrix3x3f32::ZERO;
    }

    #[inline]
    pub fn integrate(
        &mut self,
        acc_body_mps2: Vector3f32,
        acc_bias_ned_mps2: Vector3f32,
        orientation: UnitQuaternionf32,
        dt: f32,
    ) {
        self.dt_sum += dt;
        self.orientation = orientation;

        // Rotate acceleration into NED.
        let acc_rotated_ned = orientation.rotate(acc_body_mps2);

        // Apply gravity compensation and accelerometer bias correction.
        let acc_ned_mps2 = acc_rotated_ned + Self::GRAVITY_NED_MPS2 - acc_bias_ned_mps2;

        // Calculate sensitivity of NED acceleration to a small
        // orientation error expressed in the NED frame and multiply by `dt`.
        let j_acc_dt = -Matrix3x3f32::from_skew_symmetric(acc_rotated_ned * dt);

        // Integrate orientation sensitivities using the same kinematic equations as position and velocity.
        self.orientation_pos_jacobian += self.orientation_vel_jacobian * dt + j_acc_dt * (0.5 * dt);
        self.orientation_vel_jacobian += j_acc_dt;

        // Integrate acceleration.
        let dv_ned_mps = acc_ned_mps2 * dt;
        self.delta_pos_ned_m += (self.delta_vel_ned_mps + 0.5 * dv_ned_mps) * dt;
        self.delta_vel_ned_mps += dv_ned_mps;
    }

    /// Called within Kalman filter `predict` function.
    #[must_use]
    pub fn calculate_orientation_process_noise(
        &self,
        orientation: UnitQuaternionf32,
        euler_variance: RollPitchYawf32,
    ) -> OrientationNoise {
        let orientation_covariance = Self::orientation_variance_to_ned(orientation, euler_variance);
        let jp = self.orientation_pos_jacobian;
        let jv = self.orientation_vel_jacobian;
        let jp_oc = jp * orientation_covariance;

        OrientationNoise {
            position: jp_oc * jp.transpose(),
            position_velocity: jp_oc * jv.transpose(),
            velocity: jv * orientation_covariance * jv.transpose(),
        }
    }

    #[must_use]
    fn orientation_variance_to_ned(
        orientation: UnitQuaternionf32,
        euler_variance: RollPitchYawf32,
    ) -> Matrix3x3f32 {
        let q = orientation.q();

        let a = q.w * q.z + q.x * q.y;
        let b = 0.5 - q.y * q.y - q.z * q.z;
        let sq = a * a + b * b;

        let (sin_yaw, cos_yaw) = if sq < 1.0e-6 {
            // Near Euler singularity.
            // Fall back values of sin_yaw and cos_yaw
            (0.0, 1.0)
        } else {
            let norm_reciprocal = sq.sqrt_reciprocal();
            (a * norm_reciprocal, b * norm_reciprocal)
        };

        let cos_yaw_cos_pitch = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let sin_yaw_cos_pitch = 2.0 * (q.x * q.y + q.w * q.z);
        let sin_pitch = 2.0 * (q.w * q.y - q.x * q.z);
        #[rustfmt::skip]
        let j = Matrix3x3f32::new([
            cos_yaw_cos_pitch, -sin_yaw, 0.0,
            sin_yaw_cos_pitch,  cos_yaw, 0.0,
           -sin_pitch,          0.0,     1.0,
        ]);
        j.mul_diagonal_array([euler_variance.roll, euler_variance.pitch, euler_variance.yaw]) * j.transpose()
    }
}

// Orientation noise:
// matrices that represent the orientation-induced covariance contributions to integrated position and velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientationNoise {
    pub position: Matrix3x3f32,
    pub position_velocity: Matrix3x3f32,
    pub velocity: Matrix3x3f32,
}

impl Default for OrientationNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl OrientationNoise {
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self { position: Matrix3x3f32::ZERO, position_velocity: Matrix3x3f32::ZERO, velocity: Matrix3x3f32::ZERO }
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<ImuIntegrator>();
        is_full::<ImuIntegral>();
    }
}
