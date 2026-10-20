use vqm::{Matrix2x2f32, Matrix3x3xM2x2f32, Vector2f32};

use super::KalmanFilterDiagnostics;

/// `f32` variant of `KalmanFilterXy`.
pub type KalmanFilterXyf32 = KalmanFilterXy;

/// `f32` variant of `KalmanStateXy`.
pub type KalmanStateXyf32 = KalmanStateXy;

/// Kalman Filter in 2 dimensions.
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXy {
    /// 2D Kinematic State Vectors, in the NED frame.
    state: KalmanStateXy,

    // Predicted System Uncertainty Covariance Matrix (P).
    /// `P`: Prediction error covariance (aka state covariance matrix, the system's internal uncertainty).
    P: Matrix3x3xM2x2f32,

    // state transition noise covariance Matrix `Q`
    /// Process Noise spectral density mapping to Velocity variance.
    Q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    /// Units: (m/s³)² / Hz.
    Q_bias: f32,
    /// Count to avoid filter latching.
    altitude_consecutive_rejection_count: u32,
    position_consecutive_rejection_count: u32,
}

impl Default for KalmanFilterXy {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl KalmanFilterXy {
    const G_MPS2: f32 = 9.80665;

    pub const M11: usize = Matrix2x2f32::M11;
    pub const M22: usize = Matrix2x2f32::M22;

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
            altitude_consecutive_rejection_count: 0,
            position_consecutive_rejection_count: 0,
        }
    }

    /// Set the state of a newly constructed `KalmanFilterXy`.
    #[must_use]
    pub fn with_state(mut self, state: KalmanStateXy) -> Self {
        self.state = state;
        self
    }

    /// Set the state covariance matrix of a newly constructed `KalmanFilterXyz`.
    #[must_use]
    pub fn with_p(mut self, p: Matrix3x3xM2x2f32) -> Self {
        self.P = p;
        self
    }

    /// Set `Q_velocity`, the continuous-time acceleration-noise Power Spectral Density (PSD -units of m²/s³)
    /// Input is accelerometer noise density in  g/√Hz (g per root-Hertz).
    /// Datasheets typically specify accelerometer noise density in micro-g per root-Hertz or mg per root-Hertz.
    pub fn set_q_velocity(&mut self, acc_noise_density_g_per_root_hz: f32) {
        let acc_noise_density_mps2_per_root_hz = acc_noise_density_g_per_root_hz * Self::G_MPS2;
        self.Q_velocity = acc_noise_density_mps2_per_root_hz * acc_noise_density_mps2_per_root_hz;
    }

    /// Set the continuous-time bias random-walk Power Spectral Density (PSD - units of m²/s⁵).
    ///
    /// Datasheets typically specify bias instability in micro-g over a time constant tau, typically 100 seconds.
    pub fn set_q_bias(&mut self, q_bias: f32) {
        self.Q_bias = q_bias;
    }

    pub fn set_position(&mut self, pos_ned_m: Vector2f32) {
        self.state.pos_ned_m = pos_ned_m;
    }

    pub fn set_initial_position(&mut self, initial_position_m: Vector2f32) {
        self.state.pos_ned_m = initial_position_m;

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
        self.state.pos_ned_m
    }

    #[inline]
    #[must_use]
    pub fn vel_mps(&self) -> Vector2f32 {
        self.state.vel_ned_mps
    }

    #[inline]
    #[must_use]
    pub fn acc_bias_mps2(&self) -> Vector2f32 {
        self.state.acc_bias_ned_mps2
    }

    #[inline]
    #[must_use]
    pub fn p(&self) -> &Matrix3x3xM2x2f32 {
        &self.P
    }

    #[inline]
    #[must_use]
    pub fn pp(&self) -> Matrix2x2f32 {
        self.P[Self::PP]
    }

    #[inline]
    #[must_use]
    pub fn pv(&self) -> Matrix2x2f32 {
        self.P[Self::PV]
    }

    #[inline]
    #[must_use]
    pub fn bb(&self) -> Matrix2x2f32 {
        self.P[Self::BB]
    }
}

// **** Predict ****

impl KalmanFilterXy {
    /// Predicts the state vectors using `delta_vel_mps` and `delta_pos_m` from the IMU integrator.
    /// ```text
    ///     ┌          ┐
    ///     │  ½·dt·I₂ │
    /// B = │    I₂    │
    ///     │    0₂    │
    ///     └          ┘.
    /// ```
    /// Assumptions:
    /// - `delta_pos_ned_m` is the integrated displacement contribution from acceleration in NED,
    ///   including gravity compensation and correction using the estimated accelerometer bias.
    /// - `delta_vel_ned_mps` is the corresponding integrated velocity increment, with the same corrections.
    /// - Both increments are expressed in NED.
    /// - `dt` equals the total integration duration represented by these increments.
    pub fn predict_state(&mut self, delta_vel_ned_mps: Vector2f32, delta_pos_ned_m: Vector2f32, dt: f32) {
        // Predict the position using standard kinematics, note we use the value of `state.vel_ned_mps` before it is updated.
        self.state.pos_ned_m += delta_pos_ned_m + dt * self.state.vel_ned_mps;

        // Predict the velocity, using standard kinematics.
        self.state.vel_ned_mps += delta_vel_ned_mps;

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
    /// The process noise matrix Q
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

        // alias self.P for readability.
        let P = &mut self.P;

        // We can update P in place by ensuring RHS of all P assignments refers to P_old.
        //
        // We use temporary indices and shadow them out of scope when they have been used.
        // This ensures we cannot use an updated value in a subsequent update, ie P_new cannot be used on RHS.
        let (PP, VP, BP, PV, VV, BV, PB, VB, BB) =
            (Self::PP, Self::VP, Self::BP, Self::PV, Self::VV, Self::BV, Self::PB, Self::VB, Self::BB);

        // --- POSITION COLUMNS (PP, VP, BP) ---
        P[PP] = P[PP]
            + dt * (P[PV] + P[VP] + dt * (P[VV] - 0.5 * (P[PB] + P[BP] + dt * (P[VB] + P[BV] - dt * 0.5 * P[BB]))));
        // Prevent P_new[PP] being accidentally used in subsequent calculations by retiring PP.
        retire!(PP);

        P[VP] = P[VP] + dt * (P[VV] - P[BP] - dt * (P[BV] + 0.5 * (P[VB] - dt * P[BB])));
        retire!(VP);

        P[BP] = P[BP] + dt * (P[BV] - dt * 0.5 * P[BB]);
        retire!(BP);

        // --- VELOCITY COLUMNS (PV, VV, BV) ---
        P[PV] = P[PV] + dt * (P[VV] - P[PB] - dt * (P[VB] + 0.5 * (P[BV] - dt * P[BB])));
        retire!(PV);

        P[VV] = P[VV] - dt * (P[BV] + P[VB] + dt * P[BB]);
        retire!(VV);

        P[BV] = P[BV] - dt * P[BB];
        retire!(BV);

        // --- BIAS COLUMNS (PB, VB, BB) ---
        P[PB] = P[PB] + dt * (P[VB] - dt * 0.5 * P[BB]);
        retire!(PB);

        P[VB] = P[VB] - dt * P[BB];
        retire!(VB);
        // P[BB] = P[BB]; so don't need to update BB.

        // =====================================================================
        // APPLY PROCESS NOISE (Q)
        // =====================================================================

        // Continuous process noise integrated over dt maps to the diagonal variance slots of Velocity and Bias.
        // P += Q
        let q_vel = self.Q_velocity * dt;
        let q_cross = 0.5 * q_vel * dt;
        let q_pos = (2.0 / 3.0) * q_cross * dt;
        let q_bias = self.Q_bias * dt;

        P[Self::PP].add_diagonal_scalar_in_place(q_pos);
        P[Self::PV].add_diagonal_scalar_in_place(q_cross);
        P[Self::VP].add_diagonal_scalar_in_place(q_cross);
        P[Self::VV].add_diagonal_scalar_in_place(q_vel);

        // Random walk bias drift integration
        // Bias random-walk noise is added only to the bias covariance (BB).
        // Its indirect contribution to position, velocity, and their cross-covariances during this interval is neglected as an approximation,
        // since these are second order in dt.
        // This is reasonable when dt is small and the bias random-walk noise is sufficiently low.
        P[Self::BB].add_diagonal_scalar_in_place(q_bias);

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
        // H = [ I₂   0₂   0₂ ]
        // S = H·P·Hᵀ + R
        // y = z - H·x
        // K = P·Hᵀ·S⁻¹
        // x = x + K·y
        // P = P - K·H·P

        // Extract the submatrices representing H·P.
        let P_pos = self.P[Self::PP];
        let P_vel = self.P[Self::PV];
        let P_acc_bias = self.P[Self::PB];

        // Calculate S, the Innovation Covariance matrix:
        // S = H·P·Hᵀ + R
        let S = P_pos.add_diagonal_vector(R);
        let Some(S_inv) = S.try_inverse() else {
            // The the Innovation Covariance matrix may be non-invertible.
            // This happens very rarely and is due to rounding errors when the process noise covariance Q is small.
            return KalmanFilterDiagnostics::default();
        };

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measurement_m - predicted_measurement_m;

        // Normalized Innovation Squared (NIS) = yᵀ·S⁻¹·y  (aka Mahalanobis Distance)
        // Dot product results in a unitless scalar
        let nis = innovation_m.dot(S_inv * innovation_m);
        let diagnostics = KalmanFilterDiagnostics { position_trace: self.P[Self::PP].trace(), nis };

        if diagnostics.nis > KalmanFilterDiagnostics::NIS_THRESHOLD_3D
            && self.position_consecutive_rejection_count < KalmanFilterDiagnostics::MAX_CONSECUTIVE_REJECTION_COUNT
        {
            // If nis > NIS_THRESHOLD_3D, there is less than a 5% chance this is normal sensor noise,
            // so reject this measurement as an outlier.
            self.position_consecutive_rejection_count += 1;
            return diagnostics;
        }
        self.position_consecutive_rejection_count = 0;

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

        // Update the estimated state vector
        // x += K·y
        self.state.pos_ned_m += K_pos * innovation_m;
        self.state.vel_ned_mps += K_vel * innovation_m;
        self.state.acc_bias_ned_mps2 += K_acc_bias * innovation_m;

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

        diagnostics
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
        // H = [ I₂   0₂   0₂ ]
        // S = H·P·Hᵀ + R
        // y = z - H·x
        // K = P·Hᵀ·S⁻¹
        // x = x + K·y
        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ); equivalent to P - K·H·P

        // Calculate S, the Innovation Covariance matrix:
        // S = H·P·Hᵀ + R
        let S = self.P[Self::PP].add_diagonal_vector(R);

        // S is guaranteed to be positive-definite, since we are using Joseph's Stabilized Form, so we can use a Cholesky solver.
        let Some(S_cholesky) = S.cholesky() else {
            return KalmanFilterDiagnostics::default();
        };

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measurement_m - predicted_measurement_m;

        // Normalized Innovation Squared (NIS) = yᵀ·S⁻¹·y  (aka Mahalanobis Distance)
        let s_inv_times_innovation = S_cholesky.solve_vector(innovation_m);
        // Dot product results in a unitless scalar
        let nis = innovation_m.dot(s_inv_times_innovation);
        let diagnostics = KalmanFilterDiagnostics { position_trace: self.P[Self::PP].trace(), nis };

        if diagnostics.nis > KalmanFilterDiagnostics::NIS_THRESHOLD_3D
            && self.position_consecutive_rejection_count < KalmanFilterDiagnostics::MAX_CONSECUTIVE_REJECTION_COUNT
        {
            // If nis > NIS_THRESHOLD_3D, there is less than a 5% chance this is normal sensor noise,
            // so reject this measurement as an outlier.
            self.position_consecutive_rejection_count += 1;
            return diagnostics;
        }
        self.position_consecutive_rejection_count = 0;

        // Calculate K, the Kalman gain.
        // K = P·Hᵀ·S⁻¹
        let K_pos = S_cholesky.solve_matrix(self.P[Self::PP]); // 60 arithmetic operations, matrix multiply is 45
        let K_vel = S_cholesky.solve_matrix(self.P[Self::VP]);
        let K_acc_bias = S_cholesky.solve_matrix(self.P[Self::BP]);

        // Update the estimated state vector
        // x += K·y
        self.state.pos_ned_m += K_pos * innovation_m;
        self.state.vel_ned_mps += K_vel * innovation_m;
        self.state.acc_bias_ned_mps2 += K_acc_bias * innovation_m;

        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ

        // Calculate the columns of intermediate matrix A = (I - KH)·P
        let A = Matrix3x3xM2x2f32::from_column_array([
            // Column 0, position column submatrices.
            self.P[Self::PP] - K_pos * self.P[Self::PP],
            self.P[Self::VP] - K_vel * self.P[Self::PP],
            self.P[Self::BP] - K_acc_bias * self.P[Self::PP],
            // Column 1, velocity column submatrices.
            self.P[Self::PV] - K_pos * self.P[Self::PV],
            self.P[Self::VV] - K_vel * self.P[Self::PV],
            self.P[Self::BV] - K_acc_bias * self.P[Self::PV],
            // Column 2, bias column submatrices.
            self.P[Self::PB] - K_pos * self.P[Self::PB],
            self.P[Self::VB] - K_vel * self.P[Self::PB],
            self.P[Self::BB] - K_acc_bias * self.P[Self::PB],
        ]);

        // Precompute transposes and helper terms.
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

        diagnostics
    }

    pub fn correct_position(&mut self, position_m: Vector2f32, R: Vector2f32) -> KalmanFilterDiagnostics {
        self.correct_position_joseph_stabilized_form(position_m, self.state.pos_ned_m, R)
    }
}

#[allow(unused)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStateXy {
    // 3D Kinematic State Vectors
    /// Position {x, y}, meters, in the NED frame.
    pub pos_ned_m: Vector2f32,
    /// Velocity {x, y}, meters per second, in the NED frame.
    pub vel_ned_mps: Vector2f32,
    /// Accelerometer Bias {x, y}, meters per second squared, in the NED frame.
    pub acc_bias_ned_mps2: Vector2f32,
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
            pos_ned_m: Vector2f32 { x: 0.0, y: 0.0 },
            vel_ned_mps: Vector2f32 { x: 0.0, y: 0.0 },
            acc_bias_ned_mps2: Vector2f32 { x: 0.0, y: 0.0 },
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
