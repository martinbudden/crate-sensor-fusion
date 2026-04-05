use crate::{SensorFusion, SensorFusionMath};
use core::ops::{Div, Neg, Sub};
use num_traits::{One, Zero};
use vector_quaternion_matrix::{Quaternion, QuaternionMath, SqrtMethods, Vector3d, Vector3dMath};

pub type MadgwickFilterf32 = MadgwickFilter<f32>;
pub type MadgwickFilterf64 = MadgwickFilter<f64>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MadgwickFilter<T> {
    // orientation quaternion
    q: Quaternion<T>,
    acc_magnitude_squared_max: T,
    beta: T,
}

impl<T> Default for MadgwickFilter<T>
where
    T: Copy + Zero + One + Default,
{
    fn default() -> Self {
        MadgwickFilter {
            q: Quaternion::default(),
            acc_magnitude_squared_max: T::one() + T::one() + T::one() + T::one(),
            beta: T::one(),
        }
    }
}

impl<T> MadgwickFilter<T>
where
    T: Copy
        + One
        + Zero
        + Neg<Output = T>
        + PartialOrd
        + Sub<Output = T>
        + Div<Output = T>
        + Vector3dMath
        + QuaternionMath
        + SqrtMethods
        + SensorFusionMath,
{
    pub fn set_beta(&mut self, beta: T) {
        self.set_free_parameters(beta, T::zero());
    }
}

/// Madgwick AHRS algorithm, calculates orientation by fusing output from gyroscope and accelerometer.
/// (No magnetometer is used in this implementation.)
///
/// The orientation is calculated as the integration of the gyroscope measurements summed with the measurement from the accelerometer multiplied by a gain.
/// A low gain gives more weight to the gyroscope more and so is more susceptible to drift.
/// A high gain gives more weight to the accelerometer and so is more susceptible to accelerometer noise, lag, and other accelerometer errors.
/// A gain of zero means that orientation is determined by solely by the gyroscope.
///
/// See [Sebastian Madgwick's Phd thesis](https://x-io.co.uk/downloads/madgwick-phd-thesis.pdf)
/// and also x-io Technologies [sensor fusion library](https://github.com/xioTechnologies/Fusion)
///
/// For computation efficiency this code refactors the code used in many implementations (Arduino, Adafruit, M5Stack, Reefwing-AHRS),
/// [see MadgwickRefactoring](../../../documents/MadgwickRefactoring.md)
///
impl<T> SensorFusion<T> for MadgwickFilter<T>
where
    T: Copy + Zero + PartialOrd + QuaternionMath + SqrtMethods + SensorFusionMath,
{
    fn set_free_parameters(&mut self, parameter0: T, _parameter1: T) {
        self.beta = parameter0;
    }

    fn requires_initialization() -> bool {
        true
    }

    /// Fuses accelerometer and gyroscope readings to give the orientation quaternion.
    fn fuse_acc_gyro(&mut self, acc: Vector3d<T>, gyro_rps: Vector3d<T>, delta_t: T) -> Quaternion<T> {
        let step = SensorFusionMath::madgwick_step_acc(self.q, acc);
        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate and subtract the corrective step.
        let q_dot = SensorFusionMath::derivative(self.q, gyro_rps) - step * self.beta;

        // Update the orientation quaternion using simple Euler integration
        self.q += q_dot * delta_t;

        // Return the normalized orientation quaternion
        self.q.normalized()
    }

    /// Fuses accelerometer, gyroscope, and magnetometer readings to give the orientation quaternion.
    fn fuse_acc_gyro_mag(
        &mut self,
        acc: Vector3d<T>,
        gyro_rps: Vector3d<T>,
        mag: Vector3d<T>,
        delta_t: T,
    ) -> Quaternion<T> {
        let step = SensorFusionMath::madgwick_step_acc_mag(self.q, acc, mag, self.acc_magnitude_squared_max);
        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate and subtract the corrective step.
        let q_dot = SensorFusionMath::derivative(self.q, gyro_rps) - step * self.beta;

        // Update the orientation quaternion using simple Euler integration
        self.q += q_dot * delta_t;

        // Return the normalized orientation quaternion
        self.q.normalized()
    }
}

#[cfg(any(debug_assertions, test))]
mod tests {
    #![allow(unused)]
    use crate::FuseAccGyro;

    use super::*;
    use imu_sensors::ImuReadingf32;
    use vector_quaternion_matrix::{Quaternionf32, Vector3df32};

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<MadgwickFilter<f32>>();
    }
    #[test]
    fn update_orientation() {
        let mut madgwick_filter = MadgwickFilterf32::default();
        let requires_initialization = MadgwickFilterf32::requires_initialization();
        assert_eq!(requires_initialization, true);

        madgwick_filter.set_beta(1.0);

        let delta_t: f32 = 0.0;
        let acc = Vector3df32::default();
        let gyro_rps = Vector3df32::default();

        let orientation = madgwick_filter.fuse_acc_gyro(acc, gyro_rps, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    }
    #[test]
    fn fuse_acc_gyro_using() {
        let mut madgwick_filter = MadgwickFilterf32::default();

        let delta_t: f32 = 0.0;
        let acc = Vector3df32::default();
        let gyro_rps = Vector3df32::default();

        let orientation = (acc, gyro_rps).fuse_acc_gyro_using(&mut madgwick_filter, delta_t);
        assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
    }
}
