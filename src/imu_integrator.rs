use vqm::{Quaternionf32, Vector3f32};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuIntegrator {
    pub count: u32,
    pub denominator: u32,
    pub dt_sum: f32,
    pub delta_vel_ned_mps: Vector3f32,
    pub delta_pos_ned_m: Vector3f32,
    pub acc_bias_ned_mps2: Vector3f32,
}

impl Default for ImuIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ImuIntegrator {
    const DEFAULT_DENOMINATOR: u32 = 16;
    const G_MPS2: f32 = 9.80665;
    // NED: positive Z points down.
    const GRAVITY_NED_MPS2: Vector3f32 = Vector3f32 { x: 0.0, y: 0.0, z: Self::G_MPS2 };

    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            denominator: Self::DEFAULT_DENOMINATOR,
            dt_sum: 0.0,
            delta_vel_ned_mps: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            delta_pos_ned_m: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            acc_bias_ned_mps2: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
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
        self.delta_vel_ned_mps = Vector3f32 { x: 0.0, y: 0.0, z: 0.0 };
        self.delta_pos_ned_m = Vector3f32 { x: 0.0, y: 0.0, z: 0.0 };
    }

    #[inline]
    #[must_use]
    pub fn integration_complete(&self) -> bool {
        self.count >= self.denominator
    }

    #[inline]
    fn integrate_acc(&mut self, acc_ned_mps2: Vector3f32, dt: f32) {
        self.count += 1;
        self.dt_sum += dt;

        let dv_ned_mps = acc_ned_mps2 * dt;
        self.delta_pos_ned_m += (self.delta_vel_ned_mps + 0.5 * dv_ned_mps) * dt;
        self.delta_vel_ned_mps += dv_ned_mps;
    }

    /// Integrates acceleration over one IMU sample interval.
    ///
    /// Uses simple Euler integration, assuming acceleration is constant over `dt`.
    ///
    /// # Assumptions
    /// - `acc_body_mps2` and `dt` have already been validated by the caller.
    /// - `acc_body_mps2` is in the body frame.
    /// - The caller does not integrate beyond interval completion.
    #[inline]
    pub fn integrate(&mut self, acc_body_mps2: Vector3f32, orientation: Quaternionf32, dt: f32) {
        // Rotate acc_body_mps2 into NED and compensate for gravity.
        // Rotate takes ~45 arithmetic operations.
        let mut acc_ned_mps2 = orientation.rotate(acc_body_mps2) + Self::GRAVITY_NED_MPS2;

        // Correct bias in NED.
        acc_ned_mps2 -= self.acc_bias_ned_mps2;

        // Integrate the unfiltered acc in the NED frame to provide delta_pos and delta_vel for the Kalman filter.
        self.integrate_acc(acc_ned_mps2, dt);
    }

    #[must_use]
    pub fn imu_integral(&self) -> ImuIntegral {
        ImuIntegral {
            delta_pos_ned_m: self.delta_pos_ned_m,
            delta_vel_ned_mps: self.delta_vel_ned_mps,
            delta_t: self.dt_sum,
            count: self.count,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ImuIntegral {
    pub delta_pos_ned_m: Vector3f32,
    pub delta_vel_ned_mps: Vector3f32,
    pub delta_t: f32,
    pub count: u32,
}

//#[cfg(not(any(feature = "servos", feature = "eight_motors")))]
const _: () = assert!(size_of::<ImuIntegral>() == 32);

impl Default for ImuIntegral {
    fn default() -> Self {
        Self::new()
    }
}

impl ImuIntegral {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            delta_pos_ned_m: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            delta_vel_ned_mps: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            delta_t: 0.0,
            count: 0,
        }
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<ImuIntegrator>();
        is_full::<ImuIntegral>();
    }
}
