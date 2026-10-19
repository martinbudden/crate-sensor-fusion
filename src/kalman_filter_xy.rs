use vqm::{Matrix2x2f32, Matrix3x3xM2x2, Matrix3x3xM2x2f32, Vector2f32};

use super::KalmanFilterDiagnostics;

/// `f32` variant of `KalmanFilterXy`.
pub type KalmanFilterXyf32 = KalmanFilterXy;

/// `f32` variant of `KalmanStateXy`.
pub type KalmanStateXyf32 = KalmanStateXy;

/// Linear Kalman Filter in 2 dimensions.
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXy {
    /// 2D Kinematic State Vectors.
    pub state: KalmanStateXy,

    // Predicted System Uncertainty Covariance Matrix (P).
    /// `P`: Prediction error covariance (aka state covariance matrix, the system's internal uncertainty).
    pub P: Matrix3x3xM2x2f32,

    // state transition noise covariance Matrix `Q`
    /// Process Noise spectral density mapping to Velocity variance.
    pub Q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    pub Q_bias: f32,
}

impl Default for KalmanFilterXy {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl KalmanFilterXy {
    const G_MPS2: f32 = 9.80665;

    const M11: usize = Matrix2x2f32::M11;
    const M22: usize = Matrix2x2f32::M22;

    const PP: usize = 0;
    const VP: usize = 1;
    const BP: usize = 2;

    const PV: usize = 3;
    const VV: usize = 4;
    const BV: usize = 5;

    const PB: usize = 6;
    const VB: usize = 7;
    const BB: usize = 8;
}

impl KalmanFilterXy {
    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        #[allow(non_snake_case)]
        let mut P = Matrix3x3xM2x2f32::default();

        // Seed initial Position uncertainty (ie, confident within 1 meter)
        P[Self::PP][Self::M11] = 1.0;
        P[Self::PP][Self::M22] = 1.0;

        // Seed initial Velocity uncertainty (ie, confident within 0.5 m/s)
        P[Self::VV][Self::M11] = 0.25;
        P[Self::VV][Self::M22] = 0.25;

        // Seed initial Bias uncertainty (ie, accelerometer bias)
        P[Self::BB][Self::M11] = 0.01;
        P[Self::BB][Self::M22] = 0.01;

        Self {
            state: KalmanStateXy::new(),
            // A value of 0.05 implies that every second, we expect aerodynamic buffeting, vibration, or wind to naturally perturb the velocity
            // by roughly 0.22 m/s ie sqrt(0.05).
            Q_velocity: 0.05,
            // Sensor bias shifts very slowly due to thermal changes as the silicon heats up.
            // So this value should be tiny so the filter treats bias as a near-constant,
            // shifting it incrementally over minutes rather than fluctuating on every single vibration loop.
            Q_bias: 1e-4,
            P,
        }
    }

    /// Set the state of a newly constructed `KalmanFilterXy`.
    #[must_use]
    pub fn with_state(mut self, state: KalmanStateXy) -> Self {
        self.state = state;
        self
    }

    /// Set `Q_velocity`, the spectral density of the acceleration noise.
    /// Input is accelerometer noise density in g per root-Hertz.
    /// Datasheets typically specify accelerometer noise density in micro-g per root-Hertz  or mg per root-Hertz.
    pub fn set_q_velocity(&mut self, acc_noise_density_g_per_root_hz: f32, denominator: u32) {
        let acc_noise_density_mps2_per_root_hz = acc_noise_density_g_per_root_hz * Self::G_MPS2;
        #[allow(clippy::cast_precision_loss)]
        {
            self.Q_velocity =
                acc_noise_density_mps2_per_root_hz * acc_noise_density_mps2_per_root_hz / (denominator as f32);
        }
    }

    /// Set `Q_bias`, the spectral density of the acceleration bias random walk.
    /// Input is accelerometer bias instability in g per time period tau.
    /// Datasheets typically specify bias instability in micro-g over a time constant tau, typically 100 seconds.
    pub fn set_q_bias(&mut self, bias_instability_g: f32, tau: f32) {
        let bias_instability_mps2 = bias_instability_g * Self::G_MPS2;
        self.Q_velocity = bias_instability_mps2 * bias_instability_mps2 / tau;
    }

    pub fn set_position(&mut self, pos_ned_m: Vector2f32) {
        self.state.pos_m = pos_ned_m;
    }

    pub fn set_initial_position(&mut self, initial_position_m: Vector2f32) {
        self.state.pos_m = initial_position_m;

        // --- Define Initial Uncertainties (Variances) ---
        let p_pos_init = 25.0; // ±5 meters initial position uncertainty, typical of GPS
        let p_vel_init = 0.25; // ±0.5 m/s initial velocity uncertainty
        let p_bias_init = 0.04; // ±0.2 m/s² initial accelerometer bias uncertainty

        // --- Populate the Diagonal Core Blocks ---
        // Setting these ensures the X, Y, and Z axes are uniformly weighted at startup.
        self.P[Self::PP].add_diagonal_scalar_in_place(p_pos_init);
        self.P[Self::VV].add_diagonal_scalar_in_place(p_vel_init);
        self.P[Self::BB].add_diagonal_scalar_in_place(p_bias_init);

        // Off-diagonal blocks (PV, PB, VP, VB, BP, BV) remain pure zero matrices, as no cross-correlations exist yet.
    }
}

impl KalmanFilterXy {
    #[inline]
    #[must_use]
    pub fn state(&self) -> KalmanStateXy {
        self.state
    }

    #[inline]
    #[must_use]
    pub fn pos_m(&self) -> Vector2f32 {
        self.state.pos_m
    }

    #[inline]
    #[must_use]
    pub fn vel_mps(&self) -> Vector2f32 {
        self.state.vel_mps
    }

    #[inline]
    #[must_use]
    pub fn acc_bias_mps2(&self) -> Vector2f32 {
        self.state.acc_bias_mps2
    }
}

// **** Predict ****

impl KalmanFilterXy {
    /// Propagates the state vector forward using `delta_vel` and `delta_pos` from the IMU integrator.
    /// ```text
    ///     ┌          ┐
    ///     │  ½·dt·I₂ │
    /// B = │    I₂    │
    ///     │    0₂    │
    ///     └          ┘.
    /// ```
    pub fn predict_state(&mut self, delta_vel_mps: Vector2f32, delta_pos_m: Vector2f32, dt: f32) {
        // Gravity and acc bias were not included in the IMU integration, so we need to correct for them here.

        // Correct the integrated position for acc bias.
        self.state.pos_m += delta_pos_m + dt * (self.state.vel_mps - 0.5 * self.state.acc_bias_mps2 * dt);

        // Correct the integrated velocity for acc bias.
        self.state.vel_mps += delta_vel_mps - self.state.acc_bias_mps2 * dt;

        // Bias remains constant during prediction: its uncertainty is modelled as a random walk in the covariance.
        // So we don't update `self.state.acc_bias` here.
    }

    /// Predict covariance:
    /// ```text
    /// P = F·P·Fᵀ + Q
    ///
    /// P: State covariance matrix
    /// F: State transition matrix
    /// Q: Process noise covariance
    ///
    ///     ┌                            ┐
    ///     │  I₂     dt·I₂   -½·dt²·I₂  │
    /// F = │  0₂       I₂      -dt·I₂   │
    ///     │  0₂       0₂         I₂    │
    ///     └                            ┘
    ///     ┌                                ┐
    ///     │  ⅓·Q_v·dt³  ½·Q_v·dt²       0  │
    /// Q = │  ½·Q_v·dt²    Q_v·dt        0  │
    ///     │         0          0   Q_b·dt  │
    ///     └                                ┘
    /// ```
    #[allow(non_snake_case)]
    pub fn predict_covariance(&mut self, dt: f32) {
        // =====================================================================
        // PROPAGATE THE COVARIANCE P_new = F·P_old·Fᵀ
        // =====================================================================

        let dt2 = dt * dt;

        // alias self.P for readability.
        let P = &mut self.P;

        // We can update P in place by ensuring RHS of all P assignments refers to P_old.
        //
        // We use temporary indices and shadow them out of scope when they have been used.
        // This ensures we cannot use an updated value in a subsequent update, ie P_new can not be used on RHS.
        let (PP, VP, BP, PV, VV, BV, PB, VB, BB) =
            (Self::PP, Self::VP, Self::BP, Self::PV, Self::VV, Self::BV, Self::PB, Self::VB, Self::BB);

        // --- POSITION COLUMNS (PP, VP, BP) ---
        P[PP] = P[PP] + (P[VP] + P[PV]) * dt + P[VV] * dt2;
        // Prevent P_new[PP] being accidentally used in subsequent calculations by retiring PP.
        retire!(PP);

        P[VP] = P[VP] + (P[VV] - P[BP]) * dt - P[BV] * dt2;
        retire!(VP);

        P[BP] = P[BP] + P[BV] * dt;
        retire!(BP);

        // --- VELOCITY COLUMNS (PV, VV, BV) ---
        P[PV] = P[PV] + (P[VV] - P[PB]) * dt - P[VB] * dt2;
        retire!(PV);

        P[VV] = P[VV] - (P[BV] + P[VB]) * dt + P[BB] * dt2;
        retire!(VV);

        P[BV] = P[BV] - P[BB] * dt;
        retire!(BV);

        // --- BIAS COLUMNS (PB, VB, BB) ---
        P[PB] = P[PB] + P[VB] * dt;
        retire!(PB);

        P[VB] = P[VB] - P[BB] * dt;
        retire!(VB);
        // P[BB] = P[BB]; so don't need to update BB.

        // =====================================================================
        // APPLY PROCESS NOISE (Q)
        // =====================================================================

        // Continuous process noise integrated over dt maps to the diagonal variance slots of Velocity and Bias.
        // P += Q
        P[Self::VV].add_diagonal_scalar_in_place(self.Q_velocity * dt);
        P[Self::BB].add_diagonal_scalar_in_place(self.Q_bias * dt);

        // Time propagation is highly sensitive to asymmetric shearing, so enforce symmetry on the covariance matrix.
        P.enforce_symmetry();
    }
}

// **** Correct ***

#[allow(non_snake_case)]
impl KalmanFilterXy {
    /// Covariance correction step.
    ///
    /// `S = H·P·Hᵀ + R`
    /// `K = P·Hᵀ·S⁻¹`
    /// `P = P - K·H·P`.
    ///
    /// R is measurement noise covariance.
    pub fn correct_position_short_form(
        &mut self,
        measurement_m: Vector2f32,
        predicted_measurement_m: Vector2f32,
        R: Vector2f32,
    ) -> KalmanFilterDiagnostics {
        // Extract the submatrices representing H * P.
        let P_pos = self.P[Self::PP];
        let P_vel = self.P[Self::PV];
        let P_acc_bias = self.P[Self::PB];

        // Calculate S, the Residual Covariance matrix:
        // S = H·P·Hᵀ + R
        let S = P_pos.add_diagonal_vector(R);

        // The the Residual Covariance matrix may be non-invertible.
        // This happens very rarely and is due to rounding errors when the process noise covariance Q is small.
        let Some(S_inv) = S.try_inverse() else {
            return KalmanFilterDiagnostics::default();
        };

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measurement_m - predicted_measurement_m;

        // Calculate K, the Kalman gain.
        // Calculate the 3x3 segmented Kalman Gain pieces
        //
        // K = P·Hᵀ·S⁻¹
        //
        // H selects the position column block stack (Column 0: PP, VP, BP)
        // H selects the position states, so P * Hᵀ is simply the first three columns of P,
        // represented by the first three Matrix3x3 blocks.
        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // Update the physical state vectors
        // x += K·y
        self.state.pos_m += K_pos * innovation_m;
        self.state.vel_mps += K_vel * innovation_m;
        self.state.acc_bias_mps2 += K_acc_bias * innovation_m;

        // P -= K·(H·P)
        // Calculate by subtracting the K·(H·P) submatrices one by one.

        // Column 0: position column submatrices
        self.P[Self::PP] -= K_pos * P_pos;
        self.P[Self::VP] -= K_vel * P_pos;
        self.P[Self::BP] -= K_acc_bias * P_pos;

        // Column 1: velocity column submatrices
        self.P[Self::PV] -= K_pos * P_vel;
        self.P[Self::VV] -= K_vel * P_vel;
        self.P[Self::BV] -= K_acc_bias * P_vel;

        // Column 2: bias column submatrices
        self.P[Self::PB] -= K_pos * P_acc_bias;
        self.P[Self::VB] -= K_vel * P_acc_bias;
        self.P[Self::BB] -= K_acc_bias * P_acc_bias;

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();

        KalmanFilterDiagnostics::default()
    }

    /// Joseph's Stabilized Form for the covariance update step:
    ///
    /// `P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ`.
    ///
    /// While computationally more expensive, it guarantees the result remains positive-definite.
    /// That is it ensures the covariance matrix has positive eigenvalues and remains valid and invertible for future updates.
    pub fn correct_position_joseph_stabilized_form(
        &mut self,
        measurement_m: Vector2f32,
        predicted_measurement_m: Vector2f32,
        R: Vector2f32,
    ) -> KalmanFilterDiagnostics {
        // Calculate S, the Residual Covariance matrix:
        // S = H·P·Hᵀ + R
        let S = self.P[Self::PP].add_diagonal_vector(R);

        // The the Residual Covariance matrix may be non-invertible.
        // This happens very rarely and is due to rounding errors when the process noise covariance Q is small.
        let Some(S_inv) = S.try_inverse() else {
            return KalmanFilterDiagnostics::default();
        };

        // Calculate K, the Kalman gain.
        // `K = P·Hᵀ· S⁻¹`
        let K_pos = self.P[Self::PP] * S_inv;
        let K_vel = self.P[Self::VP] * S_inv;
        let K_acc_bias = self.P[Self::BP] * S_inv;

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measurement_m - predicted_measurement_m;

        // Update the physical state vectors
        // x += K·y
        self.state.pos_m += K_pos * innovation_m;
        self.state.vel_mps += K_vel * innovation_m;
        self.state.acc_bias_mps2 += K_acc_bias * innovation_m;

        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ

        // Calculate the columns of intermediate matrix A = (I - KH) * P
        let A = Matrix3x3xM2x2::from_column_array([
            // Column 0
            self.P[Self::PP] - K_pos * self.P[Self::PP],
            self.P[Self::VP] - K_vel * self.P[Self::PP],
            self.P[Self::BP] - K_acc_bias * self.P[Self::PP],
            // Column 1
            self.P[Self::PV] - K_pos * self.P[Self::PV],
            self.P[Self::VV] - K_vel * self.P[Self::PV],
            self.P[Self::BV] - K_acc_bias * self.P[Self::PV],
            // Column 2
            self.P[Self::PB] - K_pos * self.P[Self::PB],
            self.P[Self::VB] - K_vel * self.P[Self::PB],
            self.P[Self::BB] - K_acc_bias * self.P[Self::PB],
        ]);

        // Precompute transposes and helper terms
        let K_pos_t = K_pos.transpose();
        let K_vel_t = K_vel.transpose();
        let K_acc_bias_t = K_acc_bias.transpose();
        let I_minus_K_pos_t = Matrix2x2f32::identity() - K_pos_t; // Simplified distribution

        // Calculate J = A·(I - KH)ᵀ.
        // Only calculate the values that are different from A.
        let J = [
            A[Self::PP] * I_minus_K_pos_t - A[Self::PV] * K_vel_t - A[Self::PB] * K_acc_bias_t,
            A[Self::VP] * I_minus_K_pos_t - A[Self::VV] * K_vel_t - A[Self::VB] * K_acc_bias_t,
            A[Self::BP] * I_minus_K_pos_t - A[Self::BV] * K_vel_t - A[Self::BB] * K_acc_bias_t,
        ];

        // Calculate K·R·Kᵀ blocks
        let K_pos_R = K_pos.mul_diagonal_vector(R);
        let K_vel_R = K_vel.mul_diagonal_vector(R);
        let K_acc_bias_R = K_acc_bias.mul_diagonal_vector(R);

        // Reassemble J and A into P.
        self.P[Self::PP] = J[Self::PP] + K_pos_R * K_pos_t;
        self.P[Self::VP] = J[Self::VP] + K_vel_R * K_pos_t;
        self.P[Self::BP] = J[Self::BP] + K_acc_bias_R * K_pos_t;

        self.P[Self::PV] = A[Self::PV] + K_pos_R * K_vel_t;
        self.P[Self::VV] = A[Self::VV] + K_vel_R * K_vel_t;
        self.P[Self::BV] = A[Self::BV] + K_acc_bias_R * K_vel_t;

        self.P[Self::PB] = A[Self::PB] + K_pos_R * K_acc_bias_t;
        self.P[Self::VB] = A[Self::VB] + K_vel_R * K_acc_bias_t;
        self.P[Self::BB] = A[Self::BB] + K_acc_bias_R * K_acc_bias_t;

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();

        KalmanFilterDiagnostics::default()
    }

    pub fn correct_position(&mut self, position_m: Vector2f32, R: Vector2f32) -> KalmanFilterDiagnostics {
        self.correct_position_short_form(position_m, self.state.pos_m, R)
    }

    pub fn correct_position_joseph(&mut self, position_m: Vector2f32, R: Vector2f32) -> KalmanFilterDiagnostics {
        self.correct_position_joseph_stabilized_form(position_m, self.state.pos_m, R)
    }
}

#[allow(unused)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStateXy {
    // 3D Kinematic State Vectors
    /// Position {x, y}, meters.
    pub pos_m: Vector2f32,
    /// Velocity {x, y}, meters per second.
    pub vel_mps: Vector2f32,
    /// Accelerometer Bias {x, y}, meters per second squared.
    pub acc_bias_mps2: Vector2f32,
}

impl Default for KalmanStateXy {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanStateXy {
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pos_m: Vector2f32 { x: 0.0, y: 0.0 },
            vel_mps: Vector2f32 { x: 0.0, y: 0.0 },
            acc_bias_mps2: Vector2f32 { x: 0.0, y: 0.0 },
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
        is_full::<KalmanFilterXy>();
        is_full::<KalmanStateXy>();
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_position_convergence() {
        let mut filter = KalmanFilterXy::new();

        // Capture initial uncertainty variance values from the main diagonal blocks
        let init_p_pos_x = filter.P[KalmanFilterXy::PP][KalmanFilterXy::M11];
        //let init_p_vel_x = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M11];

        // Setup simulation pacing parameters
        let dt = 0.01; // 100 Hz tracking thread loop

        // Define realistic GPS/UWB measurement noise variances (R)
        // Tracks ±1.5m horizontal variance
        let r_gps = Vector2f32 { x: 2.25, y: 2.25 };

        // Define the ground-truth physical state parameters
        let mut true_pos = Vector2f32 { x: 10.0, y: -5.0 }; // Start at an offset
        let mut true_vel = Vector2f32::default();
        let true_bias = Vector2f32 { x: -0.03, y: 0.04 }; // System accelerometer bias

        // Prime the filter's initial position guess to match our starting point
        filter.state.pos_m = true_pos;

        println!("\n--- PHASE 1: STATIONARY GPS LOCK (1 SECOND) ---");
        #[allow(clippy::cast_possible_truncation)]
        let loops_stationary = (1.0 / dt) as i32;
        for _ in 0..loops_stationary {
            // Raw IMU reading = Kinematic Acc (0) + Bias
            #[allow(unused)]
            let acc_measurement = Vector2f32::default() + true_bias;

            // TODO: filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            //filter.correct_position(true_pos, r_gps);
            _ = filter.correct_position_joseph(true_pos, r_gps);
        }

        // --- PHASE 1 AUDIT LOGS ---
        println!("Stationary -> True Pos X: {:.4}, Estimated Pos X: {:.4}", true_pos.x, filter.pos_m().x);
        println!("Stationary -> True Vel X: {:.4}, Estimated Vel X: {:.4}", true_vel.x, filter.vel_mps().x);
        println!("Stationary -> True Bias X: {:.4}, Estimated Bias X: {:.4}", true_bias.x, filter.acc_bias_mps2().x);

        // Verification assertions for Phase 1
        assert!((filter.pos_m().x - true_pos.x).abs() < 0.05, "Position drifted while stationary!");
        assert!((filter.vel_mps().x - true_vel.x).abs() < 0.05, "Velocity accumulated noise while stationary!");

        // Ensure covariance has shrunk significantly below the starting bounds
        let post_stat_p_pos_x = filter.P[KalmanFilterXy::PP][KalmanFilterXy::M11];
        assert!(post_stat_p_pos_x < init_p_pos_x, "Position covariance failed to contract under GPS track!");

        println!("\n--- PHASE 2: 2D DYNAMIC SLIDE MANEUVER (2 SECONDS) ---");
        // Simulate a diagonal lateral acceleration profile (+X, +Y)
        #[allow(clippy::cast_possible_truncation)]
        let loops_maneuver = (2.0 / dt) as i32;
        let true_acc_kinematic = Vector2f32 { x: 1.5, y: 2.0 };

        for _ in 0..loops_maneuver {
            // Update truth kinematics using the exact model equations (Trapezoidal Rule)
            true_pos += (true_vel + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel += true_acc_kinematic * dt;

            // Raw IMU specific force generation
            #[allow(unused)]
            let acc_measurement = true_acc_kinematic + true_bias;

            // TODO: filter.predict_state(acc_measurement, dt);
            filter.predict_covariance(dt);

            _ = filter.correct_position(true_pos, r_gps);
        }

        // --- PHASE 2 AUDIT LOGS ---
        println!("Maneuver -> True Final Pos X: {:.4}, Estimated Final Pos X: {:.4}", true_pos.x, filter.pos_m().x);
        println!("Maneuver -> True Final Pos Y: {:.4}, Estimated Final Pos Y: {:.4}", true_pos.y, filter.pos_m().y);
        println!("Maneuver -> True Final Vel X: {:.4}, Estimated Final Vel X: {:.4}", true_vel.x, filter.vel_mps().x);
        println!("Maneuver -> True Final Vel Y: {:.4}, Estimated Final Vel Y: {:.4}", true_vel.y, filter.vel_mps().y);

        // Tracking precision assertions
        assert!((filter.pos_m().x - true_pos.x).abs() < 0.1, "Filter missed true physical X position track!");
        assert!((filter.pos_m().y - true_pos.y).abs() < 0.1, "Filter missed true physical Y position track!");
        assert!((filter.vel_mps().x - true_vel.x).abs() < 0.1, "Filter velocity tracking tracking failed on X axis!");
        assert!((filter.vel_mps().y - true_vel.y).abs() < 0.1, "Filter velocity tracking tracking failed on Y axis!");

        println!("\n✅ 2D position state propagation and measurement updates verified successfully!");
    }
}
*/
