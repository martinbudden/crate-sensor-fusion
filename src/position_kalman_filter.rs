#![allow(unused)]
use num_traits::One;
use vqm::{Matrix3x3f32, Matrix9x9f32, Vector3df32};

use crate::KalmanStateVector9f32;

pub type PositionKalmanFilterf32 = PositionKalmanFilter;

/// The system is split into two cleanly decoupled steps. This:
/// 1. avoids managing a massive 15x15 state matrix.
/// 2. linearizes the attitude so a Kalman Filter (rather than an Extended Kalman Filter) can be used.
/// ```text
///   ┌──────────────┐
///   │ IMU Acc/Gyro ├──► [ 1. ATTITUDE (MADGWICK) FILTER ] ──► Attitude Quaternion
///   └──────────────┘                │
///                                   ▼
///   ┌───────────┐       [    Transform Body ]
///   │ IMU Acc   ├─────► [ 2. Acceleration   ] ──► [ 3. POSITION KALMAN FILTER ]
///   └───────────┘       [    to Earth Frame ]                  ▲
///                                                              │
///                        GPS & Barometer Measurements ─────────┘
/// ```
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionKalmanFilter {
    // 3D Kinematic State Vectors
    //// Position (x, y, z).
    pub pos: Vector3df32,
    /// Velocity (x, y, z).
    pub vel: Vector3df32,
    /// Accelerometer Bias (x, y, z).
    pub acc_bias: Vector3df32,

    /// Predicted System Uncertainty Covariance Matrix (P).
    pub P: Matrix9x9f32,
    /// Estimated Post-Correction Error Covariance Matrix (E).
    pub E: Matrix9x9f32,

    // --- Hyperparameters & Tuning Constants ---
    /// Process Noise spectral density mapping to Velocity variance.
    pub q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    pub q_bias: f32,
    /// Absolute Measurement Noise variance for horizontal GPS channels.
    pub r_gps_horizontal: f32,
    /// Absolute Measurement Noise variance for vertical GPS channels.
    pub r_gps_vertical: f32,
    /// Absolute Measurement Noise variance for barometric pressure altimeter.
    pub r_barometer: f32,
    /// Absolute Measurement Noise variance for rangefinder.
    pub r_rangefinder: f32,
    /// Absolute Measurement Noise variance for optical flow.
    pub r_optical_flow: Vector3df32,
}

impl Default for PositionKalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionKalmanFilter {
    pub const Z_POS_ROW: usize = 2; // H vector selects the 3rd row of P
    pub const Z_POS_COL: usize = 2; // 3rd column corresponds to Z position (Altitude)
    pub const S_XX: usize = Matrix3x3f32::M11;
    pub const S_YY: usize = Matrix3x3f32::M22;
    pub const S_ZZ: usize = Matrix3x3f32::M33;

    /// Constructor.
    pub const fn new() -> Self {
        Self {
            pos: Vector3df32 { x: 0.0, y: 0.0, z: 0.0 },
            vel: Vector3df32 { x: 0.0, y: 0.0, z: 0.0 },
            acc_bias: Vector3df32 { x: 0.0, y: 0.0, z: 0.0 },
            q_velocity: 0.0,
            q_bias: 0.0,
            r_gps_horizontal: 0.0,
            r_gps_vertical: 0.0,
            r_barometer: 0.0,
            r_rangefinder: 0.0,
            r_optical_flow: Vector3df32 { x: 0.0, y: 0.0, z: 0.0 },
            E: Matrix9x9f32::new([0.0; 81]),
            P: Matrix9x9f32::new([0.0; 81]),
        }
    }
}

// **** Predict ****

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
        self.pos += (self.vel + 0.5 * acc_true * dt) * dt;
        self.vel += acc_true * dt;
        // Bias remains constant during prediction, it is modeled as a random walk in covariance.
    }
    /*
    Our state vector is organized as [{p}, {v}, {b}]^T.
    The kinematic transition equations using simple Euler integration are:
    {p}_k = {p}_{k-1} + {v}_{k-1}Delta T
    {v}_k = {v}_{k-1} - vec{b}_{k-1}Delta T (assuming acceleration is updated via the control loop)
    {b}_k = {b}_{k-1}
    This means our 9x9 matrix **A** is incredibly sparse, containing only a few *dt* terms on the off-diagonals.
     If we write out the math for \(AEA^{T}\) manually using 3x3 blocks, the matrix operations simplify into a clean sequence of 3x3 array updates.
    */
    /// Propagates the 9x9 covariance matrix forward in time.
    ///
    /// A full 9x9 matrix multiplication involves 729 individual matrix multiplications.
    /// Instead the matrix is divided into 9 separate 3x3 sub-matrices (blocks),
    /// and each block is processed separately.
    /// ```text
    /// ┌                 9x9                  ┐
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │ Position ││ Position ││ Position │ │
    /// │ │ Position ││ Velocity ││ Bias     │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │ Velocity ││ Velocity ││ Velocity │ │
    /// │ │ Position ││ Velocity ││ Bias     │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// │ │ Bias     ││ Bias     ││ Bias     │ │
    /// │ │ Position ││ Velocity ││ Bias     │ │
    /// │ └          ┘└          ┘└          ┘ │
    /// └                                      ┘
    /// ```
    /// ## Formula
    /// *  `P_k = A * E_k₋₁ * Aᵀ + Q`
    #[allow(non_snake_case)]
    #[rustfmt::skip]
    pub fn predict_covariance(&mut self, dt: f32) {
        let mut P = Matrix9x9f32::default();

        // Calculate
        //    P = A * E * A^T + Q
        // avoiding expensive 9x9 matrix multiplication by splitting the 9x9 matrices into sub-matrices (blocks).

        // =====================================================================
        // BLOCK 1: POSITION STATES (Rows r = 0..=2)
        // =====================================================================

        for (r, P_row) in P.chunks_exact_mut(9).enumerate().take(3) {
            let r_plus_3 = r + 3;
            let E_row_r = &self.E[r * 9..(r + 1) * 9];
            let E_row_r3 = &self.E[r_plus_3 * 9..(r_plus_3 + 1) * 9];

            // PositionPosition Block
            for c in 0..3 {
                let c_plus_3 = c + 3;
                P_row[c] = E_row_r[c] + dt * ((E_row_r3[c] + E_row_r[c_plus_3]) + dt * E_row_r3[c_plus_3]);
            }
            // PositionVelocity Block
            for c in 3..6 {
                let c_plus_3 = c + 3;
                P_row[c] = E_row_r[c] + dt * ((E_row_r3[c] - E_row_r[c_plus_3]) - dt * E_row_r3[c_plus_3]);
            }
            // PositionBias Block
            for c in 6..9 {
                P_row[c] = E_row_r[c] + dt * E_row_r3[c];
            }
        }

        // =====================================================================
        // BLOCK 2: VELOCITY STATES (Rows r = 3..=5)
        // =====================================================================

        for (r_offset, P_row) in P.chunks_exact_mut(9).enumerate().skip(3).take(3) {
            let r = r_offset;
            let r_plus_3 = r + 3;
            let E_row_r = &self.E[r * 9..(r + 1) * 9];
            let E_row_r3 = &self.E[r_plus_3 * 9..(r_plus_3 + 1) * 9];

            // VelocityPosition Block
            for c in 0..3 {
                let c_plus_3 = c + 3;
                P_row[c] = E_row_r[c] + dt * ((E_row_r[c_plus_3] - E_row_r3[c]) - dt * E_row_r3[c_plus_3]);
            }
            // VelocityVelocity Block
            for c in 3..6 {
                let c_plus_3 = c + 3;
                P_row[c] = E_row_r[c] - dt * ((E_row_r3[c] + E_row_r[c_plus_3]) + dt * E_row_r3[c_plus_3]);
            }
            // VelocityBias Block (Static transition)
            P_row[6..9].copy_from_slice(&E_row_r[6..9]);
        }

        // =====================================================================
        // BLOCK 3: HARDWARE ACCELEROMETER BIAS STATES (Rows r = 6..=8)
        // =====================================================================

        // Bias rows remain unchanged by the kinematic state transition matrix A
        // Since this is just a slice copy, we can bypass the loop and do it in one line.
        P[Matrix9x9f32::M71..=Matrix9x9f32::M99].copy_from_slice(&self.E[Matrix9x9f32::M71..=Matrix9x9f32::M99]);

        // =====================================================================
        // PROCESS NOISE INJECTION (Additive Q terms on the active diagonals)
        // =====================================================================

        // Add Q.

        // Velocity random walk noise maps to diagonal states 4, 5, and 6
        let q_velocity_dt2 = self.q_velocity * dt * dt;
        P[Matrix9x9f32::M44] += q_velocity_dt2;
        P[Matrix9x9f32::M55] += q_velocity_dt2;
        P[Matrix9x9f32::M66] += q_velocity_dt2;

        // Accelerometer bias random walk noise maps to diagonal states 7, 8, and 9
        let q_bias_dt2 = self.q_bias * dt * dt;
        P[Matrix9x9f32::M77] += q_bias_dt2;
        P[Matrix9x9f32::M88] += q_bias_dt2;
        P[Matrix9x9f32::M99] += q_bias_dt2;

        self.P = P;
    }
}

// **** Correct ***

impl PositionKalmanFilter {
    /// Phase 2 Altitude Correction using new measurement.
    /// Updates only the vertical Z axis components across all tracking states.
    ///
    /// ### Core Operations
    /// *  `S = P₂₂ + R` (Innovation Variance calculation)
    /// *  `K = P_column_2 * (1.0 / S)` (Kalman Gain column selection extraction)
    /// *  `E = P - K * H * P` (Covariance correction step)
    #[allow(non_snake_case)]
    pub fn correct_altitude(&mut self, altitude: f32, R: f32) {
        // Calculate the scalar innovation covariance: S = P_zz + R
        let S = self.P[Matrix9x9f32::M33] + R;

        // Calculate the 9-element Kalman Gain vector: K = (P * H^T) / S
        // Multiplying P by H^T is mathematically identical to extracting the 3rd column of P
        let K = KalmanStateVector9f32::from(self.P.column_tuple3d(Self::Z_POS_COL)) * (1.0 / S);

        // Calculate the scalar innovation error
        let error = altitude - self.pos.z;

        // Update the state vectors
        self.pos += K.pos * error;
        self.vel += K.vel * error;
        self.acc_bias += K.bias * error;

        // Extract the altitude row of P to compute the error covariance: E = P - K * H * P
        let altitude_row = KalmanStateVector9f32::from(self.P.row_tuple3d(Self::Z_POS_ROW));

        // K.outer_product(altitude_row) generates the 9x9 correction matrix
        self.E = self.P - K.outer_product(altitude_row);
    }

    /// Phase 2: Correct altitude using the barometer measurement.
    #[inline]
    pub fn correct_altitude_using_barometer(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_barometer);
    }

    /// Phase 2: Correct altitude using the rangefinder measurement.
    #[inline]
    pub fn correct_altitude_using_rangefinder(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_rangefinder);
    }

    /// Phase 2: Correct altitude using GPS vertical measurement.
    #[inline]
    pub fn correct_altitude_using_gps(&mut self, altitude: f32) {
        self.correct_altitude(altitude, self.r_gps_vertical);
    }

    /// Executes an asynchronous measurement update when a new 3D GPS reading arrives
    /// (typically at a slower 1Hz to 10Hz rate).
    /// The error becomes a 3D vector, and the 3D Position, Velocity, and Accelerometer Bias states.
    ///
    /// ### Core Operations
    /// *  Extracts `PositionPosition` sub-block from P (top left 3x3 matrix).
    /// *  `S = H * P * Hᵀ + R_gps` (yields a 3x3 innovation matrix)
    /// *  `S_inv = try_inverse(S)`
    /// *  `K = (P * Hᵀ) * S_inv` (yields a 9x3 block matrix representation)
    #[allow(non_snake_case)]
    pub fn correct_position(&mut self, position: Vector3df32, R: Vector3df32) {
        // 1. Extract the PositionPosition 3x3 sub-matrix from the top-left of the 9x9 P matrix.
        let mut P_pos = Matrix3x3f32::from(self.P);

        // Calculate the 3x3 Innovation Covariance matrix: S = H * P * H^T + R
        // In our model, R is a diagonal matrix containing horizontal and vertical GPS noise.
        P_pos[Self::S_XX] += R.x;
        P_pos[Self::S_YY] += R.y;
        P_pos[Self::S_ZZ] += R.z;

        // Calculate inverse of S.
        // If S is singular (eg sensor fault), we safely return to prevent system crash.
        let Some(S_inv) = P_pos.try_inverse() else {
            return;
        };

        // Calculate the Kalman Gain: K = (P * H^T) * S_inv, and split it into 3 separate 3x3 matrices.
        // We do this by extracting the first 3 columns of P, which is mathematically equivalent to calculating P * H^T
        // and then multiplying by S_inv.
        let (K_pos, K_vel, K_acc_bias) = Matrix9x9f32::multiply_9x3_by_3x3(&self.P, S_inv);

        // Calculate the error vector.
        let error = position - self.pos;

        // Update the state vectors across all three physical domains.
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_acc_bias * error;

        // Calculate K * (H * P) by re-assembling the 3x3 K_matrices into the 9x9 KH_P matrix.
        let KH_P = self.reassemble_k_matrices(K_pos, K_vel, K_acc_bias);

        // Update Covariance Matrix: E = P - K * (H * P)
        self.E = self.P - KH_P;
    }

    pub fn correct_position_using_gps(&mut self, position: Vector3df32) {
        let r_gps = Vector3df32 { x: self.r_gps_horizontal, y: self.r_gps_horizontal, z: self.r_gps_vertical };
        self.correct_position(position, r_gps);
    }

    pub fn correct_position_using_optical_flow(&mut self, position: Vector3df32) {
        self.correct_position(position, self.r_optical_flow);
    }

    #[allow(non_snake_case)]
    #[inline]
    fn reassemble_k_matrices(
        &mut self,
        K_pos: Matrix3x3f32,
        K_vel: Matrix3x3f32,
        K_acc_bias: Matrix3x3f32,
    ) -> Matrix9x9f32 {
        let mut KH_P = [0.0_f32; 81];

        // Cache the first 3 rows of self.P once to eliminate 24 redundant row-lookup steps
        let P_row0 = &self.P[0..9];
        let P_row1 = &self.P[9..18];
        let P_row2 = &self.P[18..27];

        // Break the destination array into 9 row slices natively
        let mut rows = KH_P.chunks_exact_mut(9);

        // --- Loop 1: Position States (Rows 0 to 2) ---
        // Chunk K_pos into rows of 3 elements using its Deref slice behavior
        for (out_row, K_row) in rows.by_ref().take(3).zip(K_pos.chunks_exact(3)) {
            let [k1, k2, k3] = [K_row[0], K_row[1], K_row[2]];

            // Loop unrolls perfectly; layout enables direct SIMD fused multiply-add (FMA)
            for c in 0..9 {
                out_row[c] = k1 * P_row0[c] + k2 * P_row1[c] + k3 * P_row2[c];
            }
        }

        // --- Loop 2: Velocity States (Rows 3 to 5) ---
        for (out_row, K_row) in rows.by_ref().take(3).zip(K_vel.chunks_exact(3)) {
            let [k1, k2, k3] = [K_row[0], K_row[1], K_row[2]];

            for c in 0..9 {
                out_row[c] = k1 * P_row0[c] + k2 * P_row1[c] + k3 * P_row2[c];
            }
        }

        // --- Loop 3: Accelerometer Bias States (Rows 6 to 8) ---
        for (out_row, K_row) in rows.take(3).zip(K_acc_bias.chunks_exact(3)) {
            let [k1, k2, k3] = [K_row[0], K_row[1], K_row[2]];

            for c in 0..9 {
                out_row[c] = k1 * P_row0[c] + k2 * P_row1[c] + k3 * P_row2[c];
            }
        }
        Matrix9x9f32::from(KH_P)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<PositionKalmanFilter>();
    }
}
