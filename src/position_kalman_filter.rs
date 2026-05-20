#![allow(unused)]
use num_traits::One;
use vqm::{Matrix3x3f32, Matrix9x9f32, StateVector9, Vector3df32};

pub type StateVector9f32 = StateVector9<f32>;
pub type StateVector9f64 = StateVector9<f64>;
pub type PositionKalmanFilterf32 = PositionKalmanFilter;
/*
The system is traditionally split into two cleanly decoupled steps to avoid managing a massive 15-state matrix:

  ┌──────────────┐
  │ IMU Acc/Gyro ├──► [ 1. ATTITUDE (MADGWICK) FILTER ] ──► Attitude Quaternion
  └──────────────┘                │
                                  ▼
  ┌───────────┐       [    Transform Body ]
  │ IMU Acc   ├─────► [ 2. Acceleration   ] ──► [ 3. POSITION KALMAN FILTER ]
  └───────────┘       [    to Earth Frame ]                  ▲
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
    /// 9x9 Predicted System Uncertainty Covariance Matrix (P).
    pub P: Matrix9x9f32,
    /// 9x9 Estimated Post-Correction Error Covariance Matrix (E).
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
    pub const S_XX: usize = Matrix3x3f32::M11;
    pub const S_YY: usize = Matrix3x3f32::M22;
    pub const S_ZZ: usize = Matrix3x3f32::M33;
    pub fn new() -> Self {
        Self {
            pos: Vector3df32::default(),
            vel: Vector3df32::default(),
            acc_bias: Vector3df32::default(),
            q_velocity: 0.0,
            q_bias: 0.0,
            r_gps_horizontal: 0.0,
            r_gps_vertical: 0.0,
            r_barometer: 0.0,
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
        self.pos += self.vel * dt + acc_true * (0.5 * dt * dt);
        self.vel += acc_true * dt;
        // Bias remains constant during prediction, modeled as a random walk in covariance
    }

    /// Propagates the 9x9 covariance matrix forward in time.
    /// 
    /// A full 9x9 matrix multiplication involves 729 individual matrix multiplications,
    /// The matrix is sparsely populated and so is divided 9 separate 3x3 sub-matrices (blocks),
    /// allowing us to skip the zero-multiplications altogether.
    /// an analytically expanded kinematic sparse row-block sequence.
    ///
    /// ### Formula
    /// *  `P_k = A * E_k₋₁ * Aᵀ + Q`
    #[allow(non_snake_case)]
    pub fn predict_covariance(&mut self, dt: f32) {
        let dt2 = dt * dt;
        let mut P_new = Matrix9x9f32::default();

        // =====================================================================
        // BLOCK 1: POSITION STATES (Rows r = 1..=3)
        // =====================================================================
        for r in 1..=3 {
            let r_plus_3 = r + 3;

            // Columns 1..=3: Position / Position Block
            for c in 1..=3 {
                let idx = Matrix9x9f32::index(r, c);
                let c_plus_3 = c + 3;
                P_new[idx] = self.E[idx]
                    + dt * (self.E[Matrix9x9f32::index(r_plus_3, c)] + self.E[Matrix9x9f32::index(r, c_plus_3)])
                    + dt2 * self.E[Matrix9x9f32::index(r_plus_3, c_plus_3)];
            }

            // Columns 4..=6: Position / Velocity Block
            for c in 4..=6 {
                let idx = Matrix9x9f32::index(r, c);
                let c_plus_3 = c + 3;
                P_new[idx] = self.E[idx] + dt * self.E[Matrix9x9f32::index(r_plus_3, c)]
                    - dt * self.E[Matrix9x9f32::index(r, c_plus_3)]
                    - dt2 * self.E[Matrix9x9f32::index(r_plus_3, c_plus_3)];
            }

            // Columns 7..=9: Position / Bias Block
            for c in 7..=9 {
                let idx = Matrix9x9f32::index(r, c);
                P_new[idx] = self.E[idx] + dt * self.E[Matrix9x9f32::index(r_plus_3, c)];
            }
        }

        // =====================================================================
        // BLOCK 2: VELOCITY STATES (Rows r = 4..=6)
        // =====================================================================
        for r in 4..=6 {
            let r_plus_3 = r + 3;

            // Columns 1..=3: Velocity / Position Block
            for c in 1..=3 {
                let idx = Matrix9x9f32::index(r, c);
                let c_plus_3 = c + 3;
                P_new[idx] = self.E[idx] + dt * self.E[Matrix9x9f32::index(r, c_plus_3)]
                    - dt * self.E[Matrix9x9f32::index(r_plus_3, c)]
                    - dt2 * self.E[Matrix9x9f32::index(r_plus_3, c_plus_3)];
            }

            // Columns 4..=6: Velocity / Velocity Block
            for c in 4..=6 {
                let idx = Matrix9x9f32::index(r, c);
                let c_plus_3 = c + 3;
                P_new[idx] = self.E[idx]
                    - dt * (self.E[Matrix9x9f32::index(r_plus_3, c)] + self.E[Matrix9x9f32::index(r, c_plus_3)])
                    - dt2 * self.E[Matrix9x9f32::index(r_plus_3, c_plus_3)];
            }

            // Columns 7..=9: Velocity / Bias Block (Static transition)
            for c in 7..=9 {
                let idx = Matrix9x9f32::index(r, c);
                P_new[idx] = self.E[idx];
            }
        }

        // =====================================================================
        // BLOCK 3: HARDWARE ACCELEROMETER BIAS STATES (Rows r = 7..=9)
        // =====================================================================
        // Bias rows remain unchanged by the kinematic state transition matrix A
        for r in 7..=9 {
            for c in 1..=9 {
                let idx = Matrix9x9f32::index(r, c);
                P_new[idx] = self.E[idx];
            }
        }

        // =====================================================================
        // PROCESS NOISE INJECTION (Additive Q terms on the active diagonals)
        // =====================================================================
        // Velocity random walk noise maps to diagonal states 4, 5, and 6
        P_new[Matrix9x9f32::M44] += dt2 * self.q_velocity;
        P_new[Matrix9x9f32::M55] += dt2 * self.q_velocity;
        P_new[Matrix9x9f32::M66] += dt2 * self.q_velocity;

        // Accelerometer bias random walk noise maps to diagonal states 7, 8, and 9
        P_new[Matrix9x9f32::M77] += dt2 * self.q_bias;
        P_new[Matrix9x9f32::M88] += dt2 * self.q_bias;
        P_new[Matrix9x9f32::M99] += dt2 * self.q_bias;

        // Save directly back to the active tracking state in-place
        self.P = P_new;
    }
}

impl PositionKalmanFilter {
    /// Executes an asynchronous measurement update when a new Barometer reading arrives.
    /// Updates only the vertical Z axis components across all tracking states.
    /// Updates vertical axis tracking components `(z, v_z, b_z)` when a Barometer altitude drop occurs.
    ///
    /// Demonstrates cross-covariance correction propagation via a single measurement scalar.
    ///
    /// ### Core Operations
    /// *  `S = P₂₂ + R_barometer` (Innovation Variance calculation)
    /// *  `K = P_column_2 * (1.0 / S)` (Kalman Gain column selection extraction)
    /// *  `E = P - K * H * P` (Covariance correction step)
    #[allow(non_snake_case)]
    pub fn update_barometer(&mut self, barometer_altitude: f32) {
        // Calculate the scalar innovation covariance: S = P_zz + R
        // Matrix9x9f32::index(3, 3) gives the flat array position for the Z variance (index 20)
        let z_idx = Matrix9x9f32::index(Self::Z_POS_INDEX, Self::Z_POS_INDEX);
        let S = self.P[z_idx] + self.r_barometer;

        // 2. Compute the 9-element Kalman Gain vector: K = (P * H^T) / S
        // Multiplying P by H^T is mathematically identical to extracting the 3rd column of P
        let K = StateVector9f32::from(self.P.column_tuple3d(Self::Z_POS_INDEX)) * (1.0 / S);

        // 3. Compute the scalar innovation error
        let error = barometer_altitude - self.pos.z;

        // 4. Update the state vectors across all physical domains simultaneously
        self.pos += K.pos * error;
        self.vel += K.vel * error;
        self.acc_bias += K.bias * error;

        // 5. Extract the altitude row of P to compute the error covariance: E = P - K * H * P
        let altitude_row = StateVector9f32::from(self.P.row_tuple3d(Self::Z_POS_ROW));

        // Matrix9x9f32::outer_product(K, altitude_row) generates an 9x9 correction matrix
        self.E = self.P - Matrix9x9f32::outer_product(K, altitude_row);
    }

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
        // 1. Extract the 3x3 Position sub-matrix from the top-left of the 9x9 P matrix.
        // This corresponds to rows 1-3, columns 1-3.
        let mut P_pos = Matrix3x3f32::from(self.P);

        // 2. Compute the 3x3 Innovation Covariance matrix: S = H * P * H^T + R
        // In our model, R is a diagonal matrix containing horizontal and vertical GPS noise.
        P_pos[Self::S_XX] += self.r_gps_horizontal; // S_xx
        P_pos[Self::S_YY] += self.r_gps_horizontal; // S_yy
        P_pos[Self::S_ZZ] += self.r_gps_vertical; // S_zz

        // 3. Compute the analytic 3x3 matrix inverse of S.
        // If S is singular (e.g. sensor fault), we safely abort to prevent system crash.
        let Some(S_inv) = P_pos.try_inverse() else {
            return;
        };

        // 4. Extract the first 3 columns of the 9x9 P matrix as a 9x3 block of 27 elements.
        // This is mathematically equivalent to computing P * H^T.
        let P_HT = self.P.extract_9x3_array();

        // 5. Compute the Kalman Gain: K = (P * H^T) * S_inv, and split it into 3 separate 3x3 matrices.
        let (K_pos, K_vel, K_acc_bias) = Matrix9x9f32::multiply_9x3_by_3x3(P_HT, S_inv);

        // 6. Compute the 3D Innovation Error vector
        let error = gps_position - self.pos;

        // 7. Update the state vectors across all three physical domains.
        self.pos += K_pos * error;
        self.vel += K_vel * error;
        self.acc_bias += K_acc_bias * error;

        // 8. Calculate K * (H * P) by re-assembling the 3x3 K_matrices into the 9x9 KH_P matrix.
        let KH_P = self.re_assemble_k_matrices(K_pos, K_vel, K_acc_bias);

        // 9. Update Covariance Matrix: E = P - K * (H * P)
        self.E = self.P - KH_P;
    }

    #[allow(non_snake_case)]
    #[inline]
    fn re_assemble_k_matrices(
        &mut self,
        K_pos: Matrix3x3f32,
        K_vel: Matrix3x3f32,
        K_acc_bias: Matrix3x3f32,
    ) -> Matrix9x9f32 {
        let mut kh_p = [0.0_f32; 81];

        // --- Loop 1: Position States (Rows 0 to 2) ---
        for r in 0..3 {
            let out_offset = r * 9;
            let k_row_offset = r * 3;
            let k1 = K_pos[k_row_offset];
            let k2 = K_pos[k_row_offset + 1];
            let k3 = K_pos[k_row_offset + 2];

            for c in 0..9 {
                kh_p[out_offset + c] = k1 * self.P[c] +       // Row 1 of P
                    k2 * self.P[c + 9] +   // Row 2 of P
                    k3 * self.P[c + 18]; // Row 3 of P
            }
        }

        // --- Loop 2: Velocity States (Rows 3 to 5) ---
        for r in 3..6 {
            let out_offset = r * 9;
            let k_row_offset = (r - 3) * 3; // Shift index back down branchlessly
            let k1 = K_vel[k_row_offset];
            let k2 = K_vel[k_row_offset + 1];
            let k3 = K_vel[k_row_offset + 2];

            for c in 0..9 {
                kh_p[out_offset + c] = k1 * self.P[c] +       // Row 1 of P
                    k2 * self.P[c + 9] +   // Row 2 of P
                    k3 * self.P[c + 18]; // Row 3 of P
            }
        }

        // --- Loop 3: Accelerometer Bias States (Rows 6 to 8) ---
        for r in 6..9 {
            let out_offset = r * 9;
            let k_row_offset = (r - 6) * 3; // Shift index back down branchlessly
            let k1 = K_acc_bias[k_row_offset];
            let k2 = K_acc_bias[k_row_offset + 1];
            let k3 = K_acc_bias[k_row_offset + 2];

            for c in 0..9 {
                kh_p[out_offset + c] = k1 * self.P[c] +       // Row 1 of P
                    k2 * self.P[c + 9] +   // Row 2 of P
                    k3 * self.P[c + 18]; // Row 3 of P
            }
        }
        Matrix9x9f32::from(kh_p)
    }
}

