#![allow(unused)]
use num_traits::{ConstZero, Float, One};
use vqm::{Matrix3x3f32, Vector3df32};
pub type AltitudeKalmanFilterf32 = AltitudeKalmanFilter;

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AltitudeKalmanFilter {
    /// States are a 3d vector with components: velocity, altitude, and bias.
    /// Time-propagated prediction state vector.
    predicted: Vector3df32,
    /// Current best estimation state vector.
    estimated: Vector3df32,
    /// Bias.
    beta: f32,
    //. R, measurement covariance matrix.
    R: f32,
    q_velocity: f32,
    // note q_altitude is assumed zero and so omitted.
    q_bias: f32,

    /// Matrices have 3 rows: velocity, altitude, and bias.
    /// estimated.
    E: Matrix3x3f32,
    /// predicted.
    P: Matrix3x3f32,
}

impl Default for AltitudeKalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl AltitudeKalmanFilter {
    // Q, process noise covariance matrix
    const Q1: f32 = 0.01;
    const Q3: f32 = 1.0;

    // R, measurement covariance matrix
    const R: f32 = 0.004 * 0.004;
    // indices to access matrix rows.
    const VELOCITY_ROW: usize = 0;
    const ALTITUDE_ROW: usize = 1;
    const BIAS_ROW: usize = 2;

    pub const fn new() -> Self {
        Self {
            predicted: Vector3df32::ZERO,
            estimated: Vector3df32::ZERO,
            beta: 0.0,
            R: Self::R,
            q_velocity: Self::Q1,
            q_bias: Self::Q3,
            E: Matrix3x3f32::ZERO,
            P: Matrix3x3f32::ZERO,
        }
    }
}

impl AltitudeKalmanFilter {
    /// Initializer targeting steady-state baseline parameters.
    pub fn new_steady_state(initial_altitude: f32, r_sensor_noise: f32, q_velocity: f32, q_bias: f32) -> Self {
        // 1. Calculate analytical steady-state variance bounds.
        // Higher sensor noise (R) increases state uncertainty boundaries.
        // Higher process noise (Q) indicates dynamic, fast-changing states.
        let steady_state_alt_variance = (q_velocity * r_sensor_noise).sqrt();
        let steady_state_vel_variance = q_velocity;
        let steady_state_bias_variance = q_bias;

        // 2. Map variances to the diagonal elements of the Covariance Matrices
        let initial_covariance = Matrix3x3f32::new([
            steady_state_vel_variance,
            0.0,
            0.0,
            0.0,
            steady_state_alt_variance,
            0.0,
            0.0,
            0.0,
            steady_state_bias_variance,
        ]);

        Self {
            estimated: Vector3df32 { x: 0.0, y: initial_altitude, z: 0.0 },
            predicted: Vector3df32 { x: 0.0, y: initial_altitude, z: 0.0 },
            P: initial_covariance,
            E: initial_covariance,
            R: r_sensor_noise,
            q_velocity,
            q_bias,
            beta: 0.1, // Damping factor configuration baseline
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

    pub fn update(&mut self, altitude_measurement: f32, acceleration_measurement: f32, delta_t: f32) -> Vector3df32 {
        // States are a 3d vector with components: velocity, altitude, and bias.
        // Destructure the state vectors as references with meaningful names, for code legibility.
        // This is a zero-cost abstraction.
        let Vector3df32 { x: estimated_velocity, y: estimated_altitude, z: estimated_bias } = self.estimated;
        let Vector3df32 { x: ref mut predicted_velocity, y: ref mut predicted_altitude, z: ref mut predicted_bias } =
            self.predicted;

        // Calculate the predicted state using the meaningful names, rather than use the vectors directly.
        // This is simple Euler integration for velocity and altitude.
        *predicted_velocity = estimated_velocity + (acceleration_measurement - estimated_bias) * delta_t;
        *predicted_altitude = estimated_altitude + estimated_velocity * delta_t;
        *predicted_bias = estimated_bias * (1.0 + self.beta * delta_t);

        // updated predicted P
        self.P = Self::predict_covariance(self.E, delta_t, self.beta, self.q_velocity, self.q_bias);

        // update the Kalman gain, k
        // h_transposed selects the second column of P during multiplication
        let h_transposed = Vector3df32 { x: 0.0, y: 1.0, z: 0.0 };
        // s is the scalar P22 + r
        let s = self.P[Matrix3x3f32::M22] + self.R;
        // K = (P * H^T) / S
        let k = (self.P * h_transposed) * (1.0 / s);

        // update estimate
        let error = altitude_measurement - *predicted_altitude;
        self.estimated = self.predicted + k * error;

        // Extract the altitude row of the P matrix as a 3d vector.
        let altitude_row = self.P.row(Self::ALTITUDE_ROW);

        // update estimated P using outer product of k and the altitude row.
        // outer_product(k, altitude_row) creates a 3x3 matrix, since both k and altitude_row are 3d vectors.
        self.E = self.P - Matrix3x3f32::outer_product(k, altitude_row);

        self.estimated
    }

    #[allow(non_snake_case)]
    #[rustfmt::skip]
    pub fn predict_covariance(E: Matrix3x3f32, dt: f32, beta: f32, q_velocity: f32, q_bias: f32) -> Matrix3x3f32 {
        // Define the State Transition Matrix (A) based on system physics
        let A = Matrix3x3f32::new([
            1.0, 0.0, -dt,
            dt,  1.0, 0.0,
            0.0, 0.0, 1.0 + beta * dt,
        ]);

        // Define the Process Noise Matrix (Q)
        let dt2 = dt * dt;
        let Q = Matrix3x3f32::new([
            -dt2 * q_velocity, 0.0, 0.0,
            0.0,               0.0, 0.0,
            0.0,               0.0, dt2 * q_bias,
        ]);

    // 3. Textbook Kalman prediction: P = A * E * A^T + Q
    (A * E * A.transpose()) + Q
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
            2.0, 5.0, 11.0, // Row 2 (Matched to our row vector)
            50.0, 60.0, 70.0, // Row 3
        ]);

        // Extract altitude row from the flat P matrix
        let altitude_row = p.row(AltitudeKalmanFilter::ALTITUDE_ROW);
        assert_eq!(Vector3df32 { x: 2.0, y: 5.0, z: 11.0 }, altitude_row);

        // Compute the updated Covariance Matrix (E).
        let kh_p = Matrix3x3f32::outer_product(k, altitude_row);

        let e = p - kh_p;

        // 5. Calculate the mathematically expected output data layout:
        // Row 1: [10, 20, 30] - [6,  15, 33]  = [4,   5,  -3]
        // Row 2: [2,  5,  11] - [14, 35, 77]  = [-12, -30, -66]
        // Row 3: [50, 60, 70] - [26, 65, 143] = [24,  -5,  -73]
        assert_eq!(e, Matrix3x3f32::from([4.0, 5.0, -3.0, -12.0, -30.0, -66.0, 24.0, -5.0, -73.0]));
    }
}
