#![allow(unused)]
use num_traits::One;
use vqm::{Matrix3x3f32, Matrix9x9f32, Vector3df32};

/*
The system is traditionally split into two cleanly decoupled steps to avoid managing a massive 15-state matrix:

  ┌───────────┐
  │ IMU Gyro  ├──► [ 1. ATTITUDE FILTER ] ──► Attitude (Quaternion / Rotation Matrix)
  └───────────┘                │
                               ▼
  ┌───────────┐         Transforms Body
  │ IMU Accel ├──► [ 2. Acceleration ] ──► [ 3. POSITION KALMAN FILTER ]
  └───────────┘         to Earth Frame                    ▲
                                                          │
                    GPS & Barometer Measurements ─────────┘
*/
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionKalmanFilter {
    // 3D Kinematic State Vectors
    pub pos: Vector3df32,      // Position (x, y, z)
    pub vel: Vector3df32,      // Velocity (x, y, z)
    pub acc_bias: Vector3df32, // Accelerometer Bias (x, y, z)

    // Covariance blocks can be kept as a flat array, or partitioned
    // For simplicity, we use a 9x9 flat layout.
    pub P: Matrix9x9f32,
    pub E: Matrix9x9f32,

    // Hyperparameters
    pub q_velocity: f32,
    pub q_bias: f32,
    pub r_gps_horiz: f32,
    pub r_gps_vert: f32,
    pub r_baro: f32,
}

impl Default for PositionKalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionKalmanFilter {
    // 1-indexed constants to map sensor states for students
    pub const Z_POS_INDEX: usize = 3; // Row 3 / Col 3 corresponds to Z position (Altitude)
    pub const Z_POS_ROW: usize = 3; // H vector selects the 3rd row of P
    pub fn new() -> Self {
        Self {
            pos: Vector3df32::default(),
            vel: Vector3df32::default(),
            acc_bias: Vector3df32::default(),
            q_velocity: 0.0,
            q_bias: 0.0,
            r_gps_horiz: 0.0,
            r_gps_vert: 0.0,
            r_baro: 0.0,
            E: Matrix9x9f32::default(),
            P: Matrix9x9f32::default(),
        }
    }
}

impl PositionKalmanFilter {
    // Propagates the state vector forward using IMU acceleration inputs.
    /// Integrates raw IMU accelerometer data to predict new position and velocity vectors.
    ///
    /// This forms the continuous dead reckoning pipeline driving vehicle kinetics forward.
    ///
    /// ### Physical Mechanics
    /// ```math
    /// pos_k = pos_k₋₁ + vel_k₋₁ * dT + 0.5 * acc_true * dT²
    /// vel_k = vel_k₋₁ + acc_true * dT
    /// ```
    pub fn predict_states(&mut self, acc_measurement: Vector3df32, dt: f32) {
        let gravity = Vector3df32 { x: 0.0, y: 0.0, z: 9.80665 };

        // 1. Calculate true physical acceleration by removing bias and adding gravity
        let acc_true = acc_measurement - self.acc_bias - gravity;

        // 2. High-level vector physics integration
        self.pos = self.pos + self.vel * dt + acc_true * (0.5 * dt * dt);
        self.vel += acc_true * dt;
        // Bias remains constant during prediction, modeled as a random walk in covariance
    }

    /// Executes an asynchronous measurement update when a new Barometer reading arrives.
    /// Updates only the vertical Z axis components across all tracking states.
    #[allow(non_snake_case)]
    pub fn update_barometer(&mut self, baro_altitude: f32) {
        // Calculate the scalar innovation covariance: S = P_zz + R
        // Matrix9x9f32::index(3, 3) gives the flat array position for the Z variance (index 20)
        let flat_z_idx = Matrix9x9f32::index(Self::Z_POS_INDEX, Self::Z_POS_INDEX);
        let S = self.P[flat_z_idx] + self.r_baro;

        // 2. Compute the 9-element Kalman Gain vector: K = (P * H^T) / S
        // Multiplying P by H^T is mathematically identical to extracting the 3rd column of P
        let K = self.P.column_vector(Self::Z_POS_INDEX) * (1.0 / S);

        // 3. Compute the scalar innovation error
        let error = baro_altitude - self.pos.z;

        // 4. Update the state vectors across all physical domains simultaneously
        self.pos += K.pos * error;
        self.vel += K.vel * error;
        self.acc_bias += K.bias * error;

        // 5. Extract the altitude row of P to compute the error covariance: E = P - K * H * P
        let altitude_row = self.P.row_vector(Self::Z_POS_ROW);

        // Matrix9x9f32::outer_product(K, altitude_row) generates an 9x9 correction matrix
        self.E = self.P - Matrix9x9f32::outer_product(K, altitude_row);
    }
}

impl PositionKalmanFilter {
    /// When a GPS packet arrives (typically at a slower 1Hz to 10Hz rate), it updates x, y, z all at once.
    /// The error becomes a 3D vector, and the update updates the whole kinematic tree.
    /// Executes an asynchronous measurement update when a new 3D GPS reading arrives.
    /// Corrects 3D Position, Velocity, and Accelerometer Bias states simultaneously.
    /// Processes slow-rate, multi-dimensional coordinate packets arriving from a GPS module.
    ///
    /// Processes slow-rate, multi-dimensional coordinate packets arriving from a GPS module.
    ///
    /// Evaluates concurrent corrections across the whole position, velocity, and bias system tree.
    ///
    /// ### Core Operations
    /// *  Extracts top-left 3x3 position sub-block from P.
    /// *  `S = H * P * Hᵀ + R_gps` (Yields a 3x3 Innovation Matrix)
    /// *  `S_inv = try_inverse(S)`
    /// *  `K = (P * Hᵀ) * S_inv` (Yields a 9x3 block matrix representation)
    #[allow(non_snake_case)]
    pub fn update_gps(&mut self, gps_position: Vector3df32) {
        // 1. Extract the 3x3 Position sub-block from the top-left of the 9x9 P matrix.
        // This corresponds to rows 1-3, columns 1-3.
        let mut P_pos = Matrix3x3f32::new(
            [
                self.P[0], self.P[1], self.P[2],
                self.P[9], self.P[10], self.P[11],
                self.P[18], self.P[19], self.P[20],
            ]
        );

        // 2. Compute the 3x3 Innovation Covariance matrix: S = H * P * H^T + R
        // In our model, R is a diagonal matrix containing horizontal and vertical GPS noise.
        P_pos[0] += self.r_gps_horiz; // S_xx
        P_pos[4] += self.r_gps_horiz; // S_yy
        P_pos[8] += self.r_gps_vert;  // S_zz

        // 3. Compute the analytic 3x3 matrix inverse of S.
        // If S is singular (e.g. sensor fault), we safely abort to prevent system crash.
        let Some(S_inv) = P_pos.try_invert() else { return; };

        // 4. Extract the first 3 columns of the 9x9 P matrix (a 9x3 block of 27 elements).
        // This is mathematically equivalent to computing P * H^T.
        let mut P_HT = [0.0; 27];
        for r in 0..9 {
            let offset9 = r * 9;
            let offset3 = r * 3;
            P_HT[offset3]     = self.P[offset9];     // Column 1
            P_HT[offset3 + 1] = self.P[offset9 + 1]; // Column 2
            P_HT[offset3 + 2] = self.P[offset9 + 2]; // Column 3
        }

        // 5. Compute the 9x3 Kalman Gain block: K = (P * H^T) * S_inv
        let K_block = Matrix9x9f32::multiply_9x3_by_3x3(P_HT, S_inv);

        // 6. Compute the 3D Innovation Error vector
        let error = gps_position - self.pos;

        // 7. Update the state vectors across all three physical domains.
        // Each domain extracts its corresponding rows from the 9x3 K_block.
        self.pos.x += K_block[0] * error.x + K_block[1] * error.y + K_block[2] * error.z;
        self.pos.y += K_block[3] * error.x + K_block[4] * error.y + K_block[5] * error.z;
        self.pos.z += K_block[6] * error.x + K_block[7] * error.y + K_block[8] * error.z;

        self.vel.x += K_block[9] * error.x + K_block[10] * error.y + K_block[11] * error.z;
        self.vel.y += K_block[12] * error.x + K_block[13] * error.y + K_block[14] * error.z;
        self.vel.z += K_block[15] * error.x + K_block[16] * error.y + K_block[17] * error.z;

        self.acc_bias.x += K_block[18] * error.x + K_block[19] * error.y + K_block[20] * error.z;
        self.acc_bias.y += K_block[21] * error.x + K_block[22] * error.y + K_block[23] * error.z;
        self.acc_bias.z += K_block[24] * error.x + K_block[25] * error.y + K_block[26] * error.z;

        // 8. Update Covariance Matrix: E = P - K * (H * P)
        // Since (H * P) is simply the top 3 rows of the 9x9 P matrix, we perform a 
        // direct 9x3 by 3x9 block multiplication to generate the 9x9 correction matrix.
        let mut KH_P = Matrix9x9f32::default();
        for r in 0..9 {
            let k_offset = r * 3;
            let out_offset = r * 9;
            
            let k1 = K_block[k_offset];
            let k2 = K_block[k_offset + 1];
            let k3 = K_block[k_offset + 2];

            // Multiply across the 9 columns of the top 3 rows of P
            for c in 0..9 {
                KH_P[out_offset + c] = 
                    k1 * self.P[c] +       // Row 1 of P
                    k2 * self.P[c + 9] +   // Row 2 of P
                    k3 * self.P[c + 18];   // Row 3 of P
            }
        }

        // Apply final covariance matrix subtraction
        self.E = self.P - KH_P;
    }
}

#[cfg(test)]
mod position_filter_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /*
    The Core Magic of Kalman Filtering:

    Note that we passed only an altitude reading (110.0) into the function.
    Yet, because the matrix infrastructure maintains cross-covariance terms,
    the math automatically calculated how that altitude discrepancy should
    adjust the estimated vertical velocity and sensor bias registers simultaneously
    without writing explicit scalar formulas.

    Orthogonality Isolation: 
    This test proves that a pure vertical sensor step doesn't bleed data into
    or corrupt the horizontal dimensions (pos.x, vel.y), verifying that the multi-sensor isolation layers are mathematically sound
    */
    #[test]
    fn test_barometer_measurement_update() {
        // 1. Initialize a baseline filter instance with realistic toy values
        let mut filter = PositionKalmanFilter {
            pos: Vector3df32 { x: 10.0, y: 20.0, z: 100.0 }, // Initial Altitude is 100m
            vel: Vector3df32 { x: 1.0, y: 2.0, z: 5.0 },     // Initial Vertical Velocity is 5m/s
            acc_bias: Vector3df32 { x: 0.01, y: 0.02, z: 0.1 }, // Initial Vertical Bias is 0.1m/s²

            // Setting up a predictable diagonal P matrix
            P: Matrix9x9f32::new({
                let mut d = [0.0; 81];
                // Fill indices corresponding to x, y, z variances
                d[Matrix9x9f32::index(1, 1)] = 2.0; // X position variance
                d[Matrix9x9f32::index(2, 2)] = 2.0; // Y position variance
                d[Matrix9x9f32::index(3, 3)] = 4.0; // Z position variance (P_zz)

                d[Matrix9x9f32::index(4, 4)] = 1.0; // X velocity variance
                d[Matrix9x9f32::index(5, 5)] = 1.0; // Y velocity variance
                d[Matrix9x9f32::index(6, 6)] = 2.0; // Z velocity variance

                d[Matrix9x9f32::index(7, 7)] = 0.1; // X bias variance
                d[Matrix9x9f32::index(8, 8)] = 0.1; // Y bias variance
                d[Matrix9x9f32::index(9, 9)] = 0.5; // Z bias variance

                // Inject a cross-covariance between Z-position (3) and Z-velocity (6)
                // and Z-position (3) and Z-bias (9) to prove the matrix update paths works!
                d[Matrix9x9f32::index(6, 3)] = 1.0; // Column 3, Row 6 (cross term)
                d[Matrix9x9f32::index(9, 3)] = -0.5; // Column 3, Row 9 (cross term)
                d
            }),
            E: Matrix9x9f32::default(),
            q_velocity: 0.1,
            q_bias: 0.01,
            r_gps_horiz: 1.0,
            r_gps_vert: 2.0,
            r_baro: 4.0, // Measurement noise R = 4.0
        };

        // 2. Introduce an altitude measurement with a 10-meter error step
        // Expected: Innovation Error = 110.0 - 100.0 = 10.0
        let barometer_reading = 110.0;

        // 3. Run the update loop
        // S = P_zz + R = 4.0 + 4.0 = 8.0
        // K_pos_z = P_zz / S = 4.0 / 8.0 = 0.5
        // K_vel_z = P_vz / S = 1.0 / 8.0 = 0.125
        // K_bias_z = P_bz / S = -0.5 / 8.0 = -0.0625
        filter.update_barometer(barometer_reading);

        // 4. Verify State Corrections: New Value = Old Value + (K * Error)
        // Expected Z position = 100.0 + (0.5 * 10.0) = 105.0
        assert_abs_diff_eq!(filter.pos.z, 105.0, epsilon = 1e-5);

        // Expected Z velocity = 5.0 + (0.125 * 10.0) = 6.25
        assert_abs_diff_eq!(filter.vel.z, 6.25, epsilon = 1e-5);

        // Expected Z bias = 0.1 + (-0.0625 * 10.0) = -0.525
        assert_abs_diff_eq!(filter.acc_bias.z, -0.525, epsilon = 1e-5);

        // 5. Verify that Horizontal States remain completely untouched by the vertical sensor
        assert_abs_diff_eq!(filter.pos.x, 10.0, epsilon = 1e-5);
        assert_abs_diff_eq!(filter.vel.y, 2.0, epsilon = 1e-5);

        // 6. Verify Covariance Matrix Extraction Reduction (E = P - K * H * P)
        // New P_zz (index 3,3) = 4.0 - (0.5 * 4.0) = 2.0
        let flat_z_idx = Matrix9x9f32::index(PositionKalmanFilter::Z_POS_INDEX, PositionKalmanFilter::Z_POS_INDEX);
        assert_abs_diff_eq!(filter.E[flat_z_idx], 2.0, epsilon = 1e-5);
    }
}

#[cfg(test)]
mod gps_filter_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gps_3d_measurement_update() {
        // 1. Initialize a baseline filter instance with clean, predictable values
        let mut filter = PositionKalmanFilter {
            pos: Vector3df32 { x: 10.0, y: 20.0, z: 30.0 },       // Initial Position (m)
            vel: Vector3df32 { x: 1.0,  y: 2.0,  z: 3.0 },         // Initial Velocity (m/s)
            acc_bias: Vector3df32 { x: 0.0,  y: 0.0,  z: 0.0 },         // Initial Accelerometer Bias

            // Setup a predictable P matrix layout
            // For educational clarity, we start with a clean diagonal covariance
            P: Matrix9x9f32::new({
                    let mut d = [0.0; 81];
                    // Fill the diagonal variances for Position (Rows 1-3)
                    d[Matrix9x9f32::index(1, 1)] = 2.0;  // X variance
                    d[Matrix9x9f32::index(2, 2)] = 2.0;  // Y variance
                    d[Matrix9x9f32::index(3, 3)] = 3.0;  // Z variance

                    // Fill diagonal variances for Velocity (Rows 4-6)
                    d[Matrix9x9f32::index(4, 4)] = 1.0;
                    d[Matrix9x9f32::index(5, 5)] = 1.0;
                    d[Matrix9x9f32::index(6, 6)] = 1.0;

                    // Fill diagonal variances for Bias (Rows 7-9)
                    d[Matrix9x9f32::index(7, 7)] = 0.1;
                    d[Matrix9x9f32::index(8, 8)] = 0.1;
                    d[Matrix9x9f32::index(9, 9)] = 0.1;

                    // Inject cross-covariances to test the multidimensional gain path:
                    // Here, we simulate that an error in X-position correlates with X-velocity,
                    // and an error in Z-position correlates with Z-bias.
                    d[Matrix9x9f32::index(4, 1)] = 0.5;  // Row 4, Col 1 (X-vel vs X-pos)
                    d[Matrix9x9f32::index(9, 3)] = -0.2; // Row 9, Col 3 (Z-bias vs Z-pos)
                    d
                }
            ),
            E: Matrix9x9f32::default(),
            q_velocity: 0.1,
            q_bias: 0.01,
            r_gps_horiz: 2.0, // Horizontal GPS noise covariance
            r_gps_vert:  3.0, // Vertical GPS noise covariance
            r_baro: 1.0,
        };

        // 2. Introduce a 3D GPS packet with a specific translation step error
        // Innovation Error = GPS - Predicted = [14.0 - 10.0, 20.0 - 20.0, 36.0 - 30.0]
        //                  = [4.0, 0.0, 6.0]
        let gps_reading = Vector3df32 { x: 14.0, y: 20.0, z: 36.0 };

        // 3. Run the multidimensional update logic
        // Hand-calculating the top-left block behavior for verification:
        // S = P_pos + R_gps
        // S_xx = 2.0 (P_xx) + 2.0 (R_horiz) = 4.0  => S_inv_xx = 1.0 / 4.0 = 0.25
        // S_zz = 3.0 (P_zz) + 3.0 (R_vert)  = 6.0  => S_inv_zz = 1.0 / 6.0 = 0.166666
        // K_pos_xx = P_xx * S_inv_xx = 2.0 * 0.25 = 0.5
        // K_pos_zz = P_zz * S_inv_zz = 3.0 * 0.166666 = 0.5
        filter.update_gps(gps_reading);

        // 4. Verify Position Updates: New = Old + (K * Error)
        // New X pos = 10.0 + (0.5 * 4.0) + (0.0 * 0.0) + (0.0 * 6.0) = 12.0
        assert_abs_diff_eq!(filter.pos.x, 12.0, epsilon = 1e-5);
        
        // New Y pos = 20.0 + (0.5 * 0.0) = 20.0 (Error was 0, should stay locked)
        assert_abs_diff_eq!(filter.pos.y, 20.0, epsilon = 1e-5);
        
        // New Z pos = 30.0 + (0.5 * 6.0) = 33.0
        assert_abs_diff_eq!(filter.pos.z, 33.0, epsilon = 1e-5);

        // 5. Verify Cross-State Corrections (Velocity and Bias channels)
        // K_vel_x = P_vx_px * S_inv_xx = 0.5 * 0.25 = 0.125
        // New X velocity = 1.0 + (0.125 * 4.0) = 1.5
        assert_abs_diff_eq!(filter.vel.x, 1.5, epsilon = 1e-5);

        // K_bias_z = P_bz_pz * S_inv_zz = -0.2 * 0.166666 = -0.033333
        // New Z bias = 0.0 + (-0.033333 * 6.0) = -0.2
        assert_abs_diff_eq!(filter.acc_bias.z, -0.2, epsilon = 1e-5);

        // 6. Verify Post-Correction Covariance Matrix Changes (E = P - K * H * P)
        // New P_xx = 2.0 - (K_pos_xx * P_xx) = 2.0 - (0.5 * 2.0) = 1.0
        let flat_x_idx = Matrix9x9f32::index(1, 1);
        assert_abs_diff_eq!(filter.E[flat_x_idx], 1.0, epsilon = 1e-5);
    }
}
