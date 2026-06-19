#![allow(unused)]
use num_traits::{ConstZero, Float, One};
use vqm::{Matrix3x3f32, Vector3df32};
pub type AltitudeKalmanFilterf32 = AltitudeKalmanFilter;

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AltitudeKalmanFilter {
    predicted: Vector3df32,
    estimated: Vector3df32,
    beta: f32,
    /// Predicted System Uncertainty Covariance Matrix (P).
    P: Matrix3x3f32,
    /// Estimated Post-Correction Error Covariance Matrix (E).
    E: Matrix3x3f32,

    // --- Hyperparameters & Tuning Constants ---
    q_velocity: f32,
    q_bias: f32,
    /// Barometer measurement variance.
    r_barometer: f32,
    /// Rangefinder measurement variance.
    r_rangefinder: f32,
    /// GPS measurement variance.
    r_gps: f32,
}

impl Default for AltitudeKalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl AltitudeKalmanFilter {
    /// Q, process noise covariance matrix.
    const Q1: f32 = 0.01;
    const Q3: f32 = 1.0;

    /// R, measurement covariance matrix.
    const R: f32 = 0.004 * 0.004;
    /// indices to access matrix rows.
    const VELOCITY_ROW: usize = 0;
    const ALTITUDE_ROW: usize = 1;
    const BIAS_ROW: usize = 2;

    /// Constructor.
    pub const fn new() -> Self {
        Self {
            predicted: Vector3df32::ZERO,
            estimated: Vector3df32::ZERO,
            beta: 0.0,
            q_velocity: Self::Q1,
            q_bias: Self::Q3,
            r_barometer: 0.0,
            r_rangefinder: 0.0,
            r_gps: 0.0,
            E: Matrix3x3f32::ZERO,
            P: Matrix3x3f32::ZERO,
        }
    }
}

impl AltitudeKalmanFilter {
    /// Initializer targeting steady-state baseline parameters.
    pub fn new_steady_state(
        initial_altitude: f32,
        q_velocity: f32,
        q_bias: f32,
        r_barometer: f32,
        r_rangefinder: f32,
        r_gps: f32,
    ) -> Self {
        // 1. Calculate analytical steady-state variance bounds.
        // Higher sensor noise (R) increases state uncertainty boundaries.
        // Higher process noise (Q) indicates dynamic, fast-changing states.
        let steady_state_alt_variance = (q_velocity * r_barometer).sqrt();
        let steady_state_vel_variance = q_velocity;
        let steady_state_bias_variance = q_bias;

        // 2. Map variances to the diagonal elements of the Covariance Matrices
        #[rustfmt::skip]
        let initial_covariance = Matrix3x3f32::new([
            steady_state_vel_variance, 0.0,                       0.0,
            0.0,                       steady_state_alt_variance, 0.0,
            0.0,                       0.0,                       steady_state_bias_variance,
        ]);

        Self {
            estimated: Vector3df32 { x: 0.0, y: initial_altitude, z: 0.0 },
            predicted: Vector3df32 { x: 0.0, y: initial_altitude, z: 0.0 },
            P: initial_covariance,
            E: initial_covariance,
            beta: 0.1, // Damping factor configuration baseline
            q_velocity,
            q_bias,
            r_barometer,
            r_rangefinder,
            r_gps,
        }
    }

    pub fn set_velocity(&mut self, velocity: f32) {
        self.estimated.x = velocity;
    }

    pub fn reset(&mut self) {
        self.E = Matrix3x3f32::one() * 100.0;
    }

    /// Returns doublet `(estimated velocity, estimated altitude)`.
    pub fn state(&self) -> (f32, f32) {
        (self.estimated.x, self.estimated.y)
    }
}

// **** Predict ****

impl AltitudeKalmanFilter {
    /// Phase 1: Predict state forward using IMU/Physics
    /// Call this at your IMU frequency or fixed control loop rate.
    #[allow(non_snake_case)]
    #[rustfmt::skip]
    pub fn predict(&mut self, acceleration_measurement: f32, delta_t: f32) -> Vector3df32 {
        // States are a 3d vector with components: velocity, altitude, and bias.
        // Destructure the state vectors as references with meaningful names, for code legibility (Zero cost abstraction).
        let Vector3df32 { x: estimated_velocity, y: estimated_altitude, z: estimated_bias } = self.estimated;
        let Vector3df32 { x: ref mut predicted_velocity, y: ref mut predicted_altitude, z: ref mut predicted_bias } =
            self.predicted;

        // Kinematic Euler integration for velocity and altitude.
        *predicted_velocity = estimated_velocity + (acceleration_measurement - estimated_bias) * delta_t;
        *predicted_altitude = estimated_altitude + estimated_velocity * delta_t;
        *predicted_bias = estimated_bias * (1.0 + self.beta * delta_t);

        // State Transition Matrix (A)
        let A = Matrix3x3f32::new([
            1.0,     0.0, -delta_t,
            delta_t, 1.0, 0.0,
            0.0,     0.0, 1.0 + self.beta * delta_t,
        ]);

        // Process Noise Matrix (Q)
        let dt2 = delta_t * delta_t;
        let Q = Matrix3x3f32::new([
            dt2 * self.q_velocity, 0.0, 0.0, // Fixed negative sign from original code if standard variance
            0.0,                   0.0, 0.0,
            0.0,                   0.0, dt2 * self.q_bias,
        ]);

        // Project error covariance: P = A * E * A^T + Q
        self.P = (A * self.E * A.transpose()) + Q;

        // Safety: If no measurement arrives, the estimate tracks tje prediction
        self.estimated = self.predicted;
        self.E = self.P;

        self.predicted
    }
}

// **** Correct ***

impl AltitudeKalmanFilter {
    /// Phase 2 Altitude Correction using new measurement.
    #[allow(non_snake_case)]
    pub fn correct_altitude(&mut self, altitude: f32, R: f32) {
        // H vector for altitude: [0, 1, 0]
        let H_transpose = Vector3df32 { x: 0.0, y: 1.0, z: 0.0 };

        // Innovation covariance: S = H * P * H^T + R
        let S = self.P[Matrix3x3f32::M22] + R;

        // Kalman Gain: K = P * H^T / S
        let K = (self.P * H_transpose) * (1.0 / S);

        // Update state estimate
        let predicted_altitude = self.predicted.y;

        let error = altitude - predicted_altitude;
        self.estimated = self.predicted + K * error;

        // Update error covariance: E = (I - KH)P
        self.E = self.P - K.outer_product(self.P.row(Self::ALTITUDE_ROW));

        // Prepare for next cycle if multiple corrections happen sequentially
        self.predicted = self.estimated;
        self.P = self.E;
    }

    /// Phase 2: Correct altitude using the barometer measurement.
    #[inline]
    pub fn correct_altitude_using_barometer(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_barometer);
    }

    /// Phase 2: Correct altitude using the rangefinder measurement.
    #[inline]
    pub fn correct_altitude_using_rangefinder(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_barometer);
    }
    /// Phase 2: Correct altitude using GPS vertical measurement.
    #[inline]
    pub fn correct_altitude_using_gps(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_gps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<AltitudeKalmanFilter>();
    }
    #[test]
    fn test_new() {
        let _kalman_filter = AltitudeKalmanFilter::new();
    }

    #[test]
    fn kalman_covariance_update() {
        // Initialize the Kalman Gain vector (K)
        let k = Vector3df32 { x: 3.0, y: 7.0, z: 13.0 };

        // 2. Initialize a starting Covariance Matrix (P)
        // We set the 2nd row to [2.0, 5.0, 11.0] to match our proven outer product values
        let p = Matrix3x3f32::new([
            10.0, 20.0, 30.0, // Row 1
            2.0, 5.0, 11.0, // Row 2 (altitude row)
            50.0, 60.0, 70.0, // Row 3
        ]);

        // Extract altitude row from the P matrix
        let altitude_row = p.row(AltitudeKalmanFilter::ALTITUDE_ROW);
        assert_eq!(Vector3df32 { x: 2.0, y: 5.0, z: 11.0 }, altitude_row);

        // Compute the updated Covariance Matrix (E).
        let kh_p = k.outer_product(altitude_row);

        let e = p - kh_p;

        // 5. Calculate the mathematically expected output data layout:
        // Row 1: [10, 20, 30] - [6,  15, 33]  = [4,   5,  -3]
        // Row 2: [2,  5,  11] - [14, 35, 77]  = [-12, -30, -66]
        // Row 3: [50, 60, 70] - [26, 65, 143] = [24,  -5,  -73]
        assert_eq!(e, Matrix3x3f32::from([4.0, 5.0, -3.0, -12.0, -30.0, -66.0, 24.0, -5.0, -73.0]));
    }
}
