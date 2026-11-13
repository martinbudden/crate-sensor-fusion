use core::{f32, f64, ops::Sub};

use super::SensorFusion;
use num_traits::{ConstOne, ConstZero, float::FloatCore};
use vqm::{MathMethods, Quaternion, QuaternionMath, UnitQuaternion, Vector3};

/// Madgwick filter for `f32`<br>
pub type MadgwickFilterf32 = MadgwickFilter<f32>;
/// Madgwick filter for `f64`<br><br>
pub type MadgwickFilterf64 = MadgwickFilter<f64>;

pub trait MadgwickConstants {
    const G_MPS2: Self;
    const ACC_DIRECTIONAL_WEIGHT_DENOMINATOR_DEFAULT: Self;
    const COS_5_DEGREES: Self;
}

impl MadgwickConstants for f32 {
    const G_MPS2: f32 = 9.80665;
    const ACC_DIRECTIONAL_WEIGHT_DENOMINATOR_DEFAULT: Self = 0.0565;
    #[allow(clippy::excessive_precision)]
    const COS_5_DEGREES: Self = 0.996_194_698;
}

impl MadgwickConstants for f64 {
    const G_MPS2: f64 = 9.80665;
    const ACC_DIRECTIONAL_WEIGHT_DENOMINATOR_DEFAULT: Self = 0.0565;
    const COS_5_DEGREES: Self = 0.996_194_698;
}

/// [Madgwick filter](https://ahrs.readthedocs.io/en/latest/filters/madgwick.html).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MadgwickFilter<T> {
    /// orientation quaternion.
    q: UnitQuaternion<T>,
    /// Lower confidence angle for accelerometer, below this angle accelerometer is fully trusted.
    cos_theta1: T,
    /// Upper confidence angle for accelerometer, above this angle accelerometer is completely rejected.
    acc_directional_weight_denominator: T,
    /*/// Typically e0 = 0.05 (5% deviation).
    e0: T,
    /// Typically e1 = 0.35 (30% deviation).
    e1: T,
    /// Beta gain tuned for fast tracking of pitch and roll (typically 0.01 to 0.1).*/
    beta: T,
    /// Beta yaw gain, should be significantly smaller than beta (e.g., 0.001 to 0.005).
    /// GPS latency can cause the filter to over correct and oscillate if this value is too aggressive.
    beta_yaw: T,
}

impl<T> Default for MadgwickFilter<T>
where
    T: Copy + ConstZero + ConstOne + MadgwickConstants,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MadgwickFilter<T> {
    /// Constructor.
    #[must_use]
    pub fn new() -> Self
    where
        T: Copy + ConstZero + ConstOne + MadgwickConstants,
    {
        MadgwickFilter {
            q: UnitQuaternion::default(),
            cos_theta1: MadgwickConstants::COS_5_DEGREES,
            acc_directional_weight_denominator: (MadgwickConstants::ACC_DIRECTIONAL_WEIGHT_DENOMINATOR_DEFAULT),
            beta: T::ONE,
            beta_yaw: T::ZERO,
        }
    }

    /// Set the beta gain of a newly constructed filter.
    #[must_use]
    pub const fn with_beta(mut self, beta: T) -> Self
    where
        T: Copy,
    {
        self.beta = beta;
        self
    }

    /// Set the orientation of a newly constructed filter.
    #[must_use]
    pub const fn with_orientation(mut self, orientation: UnitQuaternion<T>) -> Self
    where
        T: Copy,
    {
        self.q = orientation;
        self
    }

    /*/// Set accelerometer directional magnitude weights of a newly constructed filter.
    #[must_use]
    pub fn with_accelerometer_magnitude_confidence_weights(mut self, e0: T, e1: T) -> Self
    where
        T: Copy + Sub<Output = T> + MathMethods,
    {
        self.e0 = e0;
        self.e1 = e1;
        self
    }*/

    /// Set accelerometer directional confidence weights of a newly constructed filter.
    #[must_use]
    pub fn with_accelerometer_directional_confidence_weights(mut self, theta0: T, theta1: T) -> Self
    where
        T: Copy + Sub<Output = T> + MathMethods,
    {
        self.cos_theta1 = theta1.cos();
        self.acc_directional_weight_denominator = theta0.cos() - self.cos_theta1;
        self
    }
}

impl<T> MadgwickFilter<T> {
    pub fn set_orientation(&mut self, orientation: UnitQuaternion<T>)
    where
        T: Copy,
    {
        self.q = orientation;
    }

    pub fn set_beta(&mut self, beta: T)
    where
        T: Copy,
    {
        self.beta = beta;
    }

    pub fn set_beta_yaw(&mut self, beta_yaw: T)
    where
        T: Copy,
    {
        self.beta_yaw = beta_yaw;
    }

    pub fn correct_yaw(&mut self, yaw_radians: T, dt: T) -> UnitQuaternion<T>
    where
        T: Copy + FloatCore + QuaternionMath + MathMethods + MadgwickConstants,
    {
        self.correct_yaw_with_gain(yaw_radians, self.beta_yaw, dt)
    }

    // Calculate the acc directional confidence weight, `w_dir`.
    // In high g maneuvers the accelerometer is a less reliable direction indicator.
    // In these maneuvers the acc reading deviates from the estimated gravity direction.
    pub fn acc_directional_confidence_weight(&self, acc: Vector3<T>) -> T
    where
        T: FloatCore,
    {
        let gravity = self.q.gravity(); // 13 arithmetic operations
        let acc_dot_gravity = acc.dot(gravity).clamp(-T::one(), T::one());
        ((acc_dot_gravity - self.cos_theta1) / self.acc_directional_weight_denominator).clamp(-T::one(), T::one())
    }

    pub fn step_acc(q: UnitQuaternion<T>, a: Vector3<T>) -> Quaternion<T>
    where
        T: FloatCore,
    {
        let two = T::one() + T::one();
        let q = q.q();
        let wz_common = two * (q.x * q.x + q.y * q.y);
        let xy_common = two * (q.w * q.w + q.z * q.z - T::one() + two * wz_common + a.z);
        Quaternion {
            w: q.w * wz_common + q.y * a.x - q.x * a.y,
            z: q.z * wz_common - q.x * a.x - q.y * a.y,
            x: q.x * xy_common - q.z * a.x - q.w * a.y,
            y: q.y * xy_common + q.w * a.x - q.z * a.y,
        }
    }

    #[allow(clippy::many_single_char_names)]
    #[inline]
    pub fn step_acc_mag(q: Quaternion<T>, a: Vector3<T>, m: Vector3<T>) -> Quaternion<T>
    where
        T: FloatCore + MathMethods,
    {
        // make copies of the components of q to simplify the algebraic expressions
        let q0 = q.w;
        let q1 = q.x;
        let q2 = q.y;
        let q3 = q.z;
        // Auxiliary variables to avoid repeated arithmetic
        let q0q0 = q0 * q0;
        let q0q1 = q0 * q1;
        let q0q2 = q0 * q2;
        let q0q3 = q0 * q3;
        let q1q1 = q1 * q1;
        let q1q2 = q1 * q2;
        let q1q3 = q1 * q3;
        let q2q2 = q2 * q2;
        let q2q3 = q2 * q3;
        let q3q3 = q3 * q3;

        let q1q1_plus_q2q2 = q1q1 + q2q2;
        let q2q2_plus_q3q3 = q2q2 + q3q3;

        let two = T::one() + T::one();
        let four = two + two;
        let half = T::one() / two;
        // Reference direction of Earth's magnetic field
        let h = Vector3 {
            x: m.x * (q0q0 + q1q1 - q2q2_plus_q3q3) + two * (m.y * (q1q2 - q0q3) + m.z * (q0q2 + q1q3)),
            y: (m.x * (q0q3 + q1q2) + m.y * (q0q0 - q1q1 + q2q2 - q3q3) + m.z * (q2q3 - q0q1)) * two,
            z: T::zero(),
        };

        let bx_bx = h.x * h.x + h.y * h.y;
        let b = Vector3 {
            x: bx_bx.sqrt(),
            y: T::zero(),
            z: two * (m.x * (q1q3 - q0q2) + m.y * (q0q1 + q2q3)) + m.z * (q0q0 - q1q1_plus_q2q2 + q3q3),
        };

        let a_dash = Vector3 { x: a.x + m.x * b.z, y: a.y + m.y * b.z, z: T::zero() };
        let bz_bz = b.z * b.z;
        let _4bx_bz = four * b.x * b.z;

        let m_bx = m * b.x;
        let mz_bz = m.z * b.z;

        let sum_squares_minus_one = q0q0 + q1q1_plus_q2q2 + q3q3 - T::one();
        let xy_common = sum_squares_minus_one + q1q1_plus_q2q2;
        let yz_common = sum_squares_minus_one + q2q2_plus_q3q3;
        let wz_common = q1q1_plus_q2q2 * (T::one() + bz_bz) + bx_bx;

        // Gradient decent algorithm corrective step
        #[allow(clippy::used_underscore_binding)]
        Quaternion {
            w: q0 * two * (wz_common * q2q2_plus_q3q3) - q1 * a_dash.y
                + q2 * (a_dash.x - m_bx.z)
                + q3 * (m_bx.y - _4bx_bz * q0q1),

            x: -q0 * a_dash.y
                + q1 * two * (xy_common * (T::one() + two * bz_bz) + mz_bz + bx_bx * q2q2_plus_q3q3 + a.z)
                - q2 * m_bx.y
                - q3 * (a_dash.x + m_bx.z + _4bx_bz * (half * sum_squares_minus_one + q1q1)),

            y: q0 * (a_dash.x - m_bx.z) - q1 * m_bx.y
                + q2 * two * (xy_common * (T::one() + bz_bz) + mz_bz + m_bx.x + bx_bx * yz_common + a.z)
                - q3 * (a_dash.y + _4bx_bz * q1q2),

            z: q0 * m_bx.y - q1 * (a_dash.x + m_bx.z + _4bx_bz * (half * sum_squares_minus_one + q3q3)) - q2 * a_dash.y
                + q3 * two * (wz_common + m_bx.x * yz_common),
        }
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
/// and also x-io Technologies [sensor fusion library](https://github.com/xioTechnologies/Fusion).
///
/// For computation efficiency this code refactors the code used in many implementations (Arduino, Adafruit, M5 Stack, Reefwing-AHRS),
/// [see Madgwick refactoring](../../../documents/MadgwickRefactoring.md).
///
impl<T> SensorFusion<T> for MadgwickFilter<T>
where
    T: Copy + FloatCore + QuaternionMath + MathMethods,
{
    fn set_gains(&mut self, gain0: T, gain1: T) {
        self.beta = gain0;
        self.beta_yaw = gain1;
    }

    fn gains(&self) -> (T, T) {
        (self.beta, self.beta_yaw)
    }

    fn requires_initialization() -> bool {
        true
    }

    /// Fuses accelerometer and gyroscope readings to give the orientation quaternion.
    fn fuse_acc_gyro(&mut self, acc_mps2: Vector3<T>, gyro_rps: Vector3<T>, dt: T) -> UnitQuaternion<T> {
        // Calculate the corrective step.
        let acc_norm = acc_mps2.norm();
        let mut acc = acc_mps2;
        if acc_norm == T::zero() {
            acc = Vector3 { x: T::zero(), y: T::zero(), z: T::zero() };
        } else {
            acc /= acc_norm;
        }

        // Calculate the acc magnitude confidence weight, `w_mag`.
        //let e = (acc_norm - MadgwickConstants::G_MPS2).abs();
        //let w_mag = if e > self.e0 { (self.e1 - e) / (self.e1 - self.e0) } else { T::one() };

        let w_dir = self.acc_directional_confidence_weight(acc);

        // Calculate the adaptive step.
        let step = Self::step_acc(self.q, acc);
        let mut q = self.q.q();
        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate
        // and subtract the corrective step attenuated by the acc directional confidence.
        let q_dot = q.derivative(gyro_rps) - step * (self.beta * w_dir);

        // Update the orientation quaternion using simple Euler integration
        q += q_dot * dt;

        // normalize the orientation quaternion and return it
        self.q = UnitQuaternion::from_quaternion(q);
        self.q
    }

    /// Fuses accelerometer, gyroscope, and magnetometer readings to give the orientation quaternion.
    fn fuse_acc_gyro_mag(
        &mut self,
        acc_mps2: Vector3<T>,
        gyro_rps: Vector3<T>,
        mag: Vector3<T>,
        dt: T,
    ) -> UnitQuaternion<T> {
        // Calculate the corrective step.
        let acc_magnitude_squared = acc_mps2.norm_squared();
        let mut acc = acc_mps2;
        if acc_magnitude_squared == T::zero() {
            acc = Vector3 { x: T::zero(), y: T::zero(), z: T::zero() };
        } else {
            acc *= acc_magnitude_squared.sqrt_reciprocal();
        }

        let w_dir = self.acc_directional_confidence_weight(acc);

        let mut q = self.q.q();
        let step = Self::step_acc_mag(q, acc, mag.normalize());

        // Calculate quaternion derivative (q_dot, aka dq/dt) from the angular rate and subtract the corrective step.
        let q_dot = q.derivative(gyro_rps) - step * (self.beta * w_dir);

        // Update the orientation quaternion using simple Euler integration
        q += q_dot * dt;

        // normalize the orientation quaternion and return it

        self.q = UnitQuaternion::from_quaternion(q);
        self.q
    }

    /// Note `dt` is the time since `correct_yaw` was last called, distinct from the `dt` in `fuse_acc_gyro`.
    /// This function called at a frequency of ~10Hz if called from a GPS update, or ~200Hz for a magnetometer update.
    /// It is normally called withing the Gyro/PID loop and so should be as fast as possible to minimize jitter.
    fn correct_yaw_with_gain(&mut self, yaw_radians: T, gain: T, dt: T) -> UnitQuaternion<T> {
        // Create a target yaw quaternion from the yaw heading.
        // This represents a flat orientation pointing in the direction of travel.
        let mut target = UnitQuaternion::<T>::from_yaw_radians(yaw_radians);

        // The dot product between two quaternions tells you how well aligned they are.
        // If the dot product is positive, the two quaternions are on the same hemisphere, and moving directly toward the target is the shortest path.
        // If the dot product is negative, the quaternions are pointing the same way geometrically,
        // but traveling along the raw path would take the "long way around" the 4D sphere
        // (due to the double-cover property of quaternions where `q` and `-q` represent the same 3D rotation).
        // If the dot product is negative, we simply negate the target quaternion.
        // This flips it to the closer hemisphere, ensuring the gradient descent step takes the shortest possible path.
        if self.q.dot(target) < T::zero() {
            target = -target;
        }

        // Calculate the error quaternion between current orientation and the target (measured) yaw heading
        let error = self.q * target.conjugate();

        // Construct a pure rotational velocity quaternion around the Z-axis
        let omega = Quaternion::<T>::new(T::zero(), T::zero(), T::zero(), -error.q().z);

        // Calculate kinematic derivative: q_dot = 0.5 * q * omega
        // The 0.5 factor is absorbed into the `beta_yaw` value, so does not appear here.
        let mut q = self.q.q();
        let q_dot = q * omega;

        //let error_z = -self.q.w * target.z - self.q.x * target.y + self.q.y * target.x + self.q.z * target.w;
        //let q_dot = Quaternion { w: self.q.z, x: -self.q.y, y: self.q.x, z: -self.q.w } * (error_z * self.beta_yaw);

        // Euler integration & renormalization
        q += q_dot * (dt * gain);
        self.q = UnitQuaternion::from_quaternion(q);

        self.q
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<MadgwickFilter<f32>>();
    }
}
#[cfg(any(debug_assertions, test))]
mod tests {
    #[allow(unused)]
    use {
        super::*,
        crate::FuseAccGyro,
        vqm::{Quaternionf32, UnitQuaternionf32, Vector3f32},
    };

    #[test]
    #[allow(clippy::float_cmp)]
    fn q_dot() {
        let q = Quaternionf32::new(2.0, 3.0, 5.0, 7.0);
        let mut target = Quaternionf32::new(11.0, 13.0, 17.0, 19.0);
        if q.dot(target) < 0.0 {
            target = -target;
        }

        let error = q * target.conjugate();
        assert_eq!(53.0, error.z);
        let omega = Quaternionf32::new(0.0, 0.0, 0.0, -error.z);
        let q_dot = q * omega;
        assert_eq!(Quaternionf32 { w: 371.0, x: -265.0, y: 159.0, z: -106.0 }, q_dot);

        let error_z = -q.w * target.z - q.x * target.y + q.y * target.x + q.z * target.w;
        assert_eq!(53.0, error_z);
        assert_eq!(error.z, error_z);
        let q_dot2 = Quaternion { w: q.z * error_z, x: -q.y * error_z, y: q.x * error_z, z: -q.w * error_z };
        assert_eq!(q_dot, q_dot2);
    }

    #[test]
    fn readme() {
        let mut madgwick_filter = MadgwickFilterf32::default();

        let dt: f32 = 0.001;
        let acc_mps2 = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = madgwick_filter.fuse_acc_gyro(acc_mps2, gyro_rps, dt);
        //assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
        assert!(orientation.is_near_one(1.2e-4));
    }

    #[test]
    fn update_orientation() {
        let mut madgwick_filter = MadgwickFilterf32::default();
        let requires_initialization = MadgwickFilterf32::requires_initialization();
        assert!(requires_initialization);

        madgwick_filter.set_beta(1.0);

        let dt: f32 = 0.0;
        let acc_mps2 = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = madgwick_filter.fuse_acc_gyro(acc_mps2, gyro_rps, dt);
        //assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
        assert!(orientation.is_near_one(1.2e-4));
    }

    #[test]
    fn fuse_acc_gyro_using() {
        let mut madgwick_filter = MadgwickFilterf32::default();

        let dt: f32 = 0.0;
        let acc_mps2 = Vector3f32::default();
        let gyro_rps = Vector3f32::default();

        let orientation = (acc_mps2, gyro_rps).fuse_acc_gyro_using(&mut madgwick_filter, dt);
        //assert_eq!(orientation, Quaternion { w: 1.0, x: 0.0, y: 0.0, z: 0.0 });
        assert!(orientation.is_near_one(1.2e-4));
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn yaw_correction() {
        let initial_yaw = 179.0f32;
        let initial_q = UnitQuaternionf32::from_yaw_degrees(initial_yaw);
        //println!("Initial q: {initial_q}");

        let mut filter = MadgwickFilterf32::new().with_orientation(initial_q);
        let gain = 5.0;
        let dt = 0.1;
        filter.set_beta_yaw(gain);

        let target_yaw = 178.0_f32;
        let filtered_q = filter.correct_yaw_with_gain(target_yaw.to_radians(), gain, dt);
        let final_yaw = filtered_q.calculate_yaw_degrees();
        println!("Yaw, initial: {initial_yaw:.4}° | target: {target_yaw:.4}° | filtered: {final_yaw:.4}°");
        assert!((178.5 - final_yaw).abs() < 5e-4);

        filter.set_orientation(initial_q);
        let target_yaw = 179.5_f32;
        let filtered_q = filter.correct_yaw_with_gain(target_yaw.to_radians(), gain, dt);
        let final_yaw = filtered_q.calculate_yaw_degrees();
        println!("Yaw, initial: {initial_yaw:.4}° | target: {target_yaw:.4}° | filtered: {final_yaw:.4}°");
        assert!((179.25 - final_yaw).abs() < 2e-4);

        let initial_yaw = 179.75f32;
        let initial_q = UnitQuaternionf32::from_yaw_degrees(initial_yaw);
        filter.set_orientation(initial_q);
        let target_yaw = 181.0_f32;
        let filtered_q = filter.correct_yaw_with_gain(target_yaw.to_radians(), gain, dt);
        let final_yaw = filtered_q.calculate_yaw_degrees();
        println!("Yaw, initial: {initial_yaw:.4}° | target: {target_yaw:.4}° | filtered: {final_yaw:.4}°");
        assert!((-179.625 - final_yaw).abs() < 4e-4);
        // A correct step moves across the 180° threshold. Because of how atan2 wraps,
        // moving from 179.75° towards 181.0° means it wraps cleanly into the negative space (e.g. -179.625°).
        // If it took the wrong long path, the angle would drop below 179.75° (e.g. 179.25°).
        assert!(
            !(0.0..=179.0).contains(&final_yaw),
            "\n**** Filter took the long way around! Yaw dropped down to {final_yaw:.2}°"
        );
    }
}
