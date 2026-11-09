use vqm::{Matrix3x3, Matrix3x3f32, Matrix3x3xM3x3f32, Vector2f32, Vector3f32};

use super::{FusionStatus, KalmanFilterDiagnostics};

/// `f32` variant of `KalmanFilterXyz`.
pub type KalmanFilterXyzf32 = KalmanFilterXyz;

/// `f32` variant of `KalmanStateXyz`.
pub type KalmanStateXyzf32 = KalmanStateXyz;

/// Kalman Filter in 3 dimensions.
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXyz {
    /// 3D Kinematic State Vectors, in the NED frame.
    state: KalmanStateXyz,

    // Predicted System Uncertainty Covariance Matrix (P).
    /// `P`: Prediction error covariance (aka state covariance matrix, the system's internal uncertainty).
    P: Matrix3x3xM3x3f32,

    // state transition noise covariance Matrix `Q`
    /// Process Noise spectral density mapping to Velocity variance.
    Q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    /// Units: (m/s³)² / Hz.
    Q_bias: f32,
}

impl Default for KalmanFilterXyz {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl KalmanFilterXyz {
    const G_MPS2: f32 = 9.80665;
    /// Minimum reciprocal condition allowed for inverting innovation covariance matrix S.
    /// This represents a fairly ill-conditioned matrix, consider making it larger.
    const MIN_RECIPROCAL_CONDITION: f32 = 0.0001;

    pub const PP: usize = 0;
    pub const VP: usize = 1;
    pub const BP: usize = 2;

    pub const PV: usize = 3;
    pub const VV: usize = 4;
    pub const BV: usize = 5;

    pub const PB: usize = 6;
    pub const VB: usize = 7;
    pub const BB: usize = 8;
}

impl KalmanFilterXyz {
    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        #[allow(non_snake_case)]
        let mut P = Matrix3x3xM3x3f32::default();

        // Seed initial Position uncertainty (ie, confident within 1 meter)
        P[Self::PP].set_diagonal_from_vector(Vector3f32 { x: 1.0, y: 1.0, z: 1.0 });

        // Seed initial Velocity uncertainty (ie, confident within 0.5 m/s)
        P[Self::VV].set_diagonal_from_vector(Vector3f32 { x: 0.25, y: 0.25, z: 0.25 } * KalmanStateXyz::SCALE_VELOCITY);

        // Seed initial Bias uncertainty (ie, accelerometer bias)
        P[Self::BB].set_diagonal_from_vector(Vector3f32 { x: 0.01, y: 0.01, z: 0.01 } * KalmanStateXyz::SCALE_ACC_BIAS);

        Self {
            state: KalmanStateXyz::new(),
            P,
            // A value of 0.05 implies that every second, we expect aerodynamic buffeting, vibration, or wind to naturally perturb the velocity
            // by roughly 0.22 m/s ie sqrt(0.05).
            Q_velocity: 0.05,
            // Sensor bias shifts very slowly due to thermal changes as the silicon heats up.
            // So this value should be tiny so the filter treats bias as a near-constant,
            // shifting it incrementally over minutes rather than fluctuating on every single vibration loop.
            Q_bias: 1e-4,
        }
    }

    /// Set the state of a newly constructed `KalmanFilterXyz`.
    #[must_use]
    pub fn with_state(mut self, state: KalmanStateXyz) -> Self {
        self.set_state(&state);
        self
    }

    /// Set the state covariance matrix of a newly constructed `KalmanFilterXyz`.
    #[must_use]
    pub fn with_p(mut self, p: Matrix3x3xM3x3f32) -> Self {
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

    fn set_state(&mut self, state: &KalmanStateXyz) {
        self.state.set(state);
    }

    pub fn set_position(&mut self, pos_ned_m: Vector3f32) {
        self.state.pos_ned_m = pos_ned_m;
    }

    pub fn set_initial_position(&mut self, initial_position_m: Vector3f32) {
        self.state.pos_ned_m = initial_position_m;

        // --- Define Initial Uncertainties (Variances) ---
        let p_pos_init = 25.0; // ±5 meters initial position uncertainty, typical of GPS
        let p_vel_init = 0.25; // ±0.5 m/s initial velocity uncertainty
        let p_bias_init = 0.04; // ±0.2 m/s² initial accelerometer bias uncertainty

        // --- Populate the Diagonal Core Blocks ---
        // Setting these ensures the X, Y, and Z axes are uniformly weighted at startup.
        self.P[Self::PP].add_diagonal_scalar_in_place(p_pos_init);
        self.P[Self::VV].add_diagonal_scalar_in_place(p_vel_init * KalmanStateXyz::SCALE_VELOCITY);
        self.P[Self::BB].add_diagonal_scalar_in_place(p_bias_init * KalmanStateXyz::SCALE_ACC_BIAS);

        // Off-diagonal blocks (PV, PB, VP, VB, BP, BV) remain pure zero matrices, as no cross-correlations exist yet.
    }
}

impl KalmanFilterXyz {
    #[inline]
    #[must_use]
    pub fn state(&self) -> KalmanStateXyz {
        self.state
    }

    #[inline]
    #[must_use]
    pub fn pos_m(&self) -> Vector3f32 {
        self.state.pos_ned_m
    }

    #[inline]
    #[must_use]
    pub fn vel_mps(&self) -> Vector3f32 {
        self.state.vel_ned_mps()
    }

    #[inline]
    #[must_use]
    pub fn acc_bias_mps2(&self) -> Vector3f32 {
        self.state.acc_bias_ned_mps2()
    }

    #[inline]
    #[must_use]
    pub fn p(&self) -> &Matrix3x3xM3x3f32 {
        &self.P
    }

    #[inline]
    #[must_use]
    pub fn pp(&self) -> Matrix3x3f32 {
        self.P[Self::PP]
    }

    #[inline]
    #[must_use]
    pub fn pv(&self) -> Matrix3x3f32 {
        self.P[Self::PV]
    }

    #[inline]
    #[must_use]
    pub fn bb(&self) -> Matrix3x3f32 {
        self.P[Self::BB]
    }
}

// **** Predict ****

impl KalmanFilterXyz {
    /// Predicts the state vectors (`x`) using the control input vectors (`u`) `delta_vel_mps` and `delta_pos_m` from the IMU integrator.
    /// ```text
    /// x = F·x + B·u
    ///
    /// where
    ///     ┌          ┐
    ///     │  ½·dt·I₃ │
    /// B = │    I₃    │
    ///     │    0₃    │
    ///     └          ┘.
    /// ```
    /// Assumptions:
    /// - `delta_pos_ned_m` is the integrated displacement contribution from acceleration in NED,
    ///   including gravity compensation and correction using the estimated accelerometer bias.
    /// - `delta_vel_ned_mps` is the corresponding integrated velocity increment, with the same corrections.
    /// - Both increments are expressed in NED.
    /// - `dt` equals the total integration duration represented by these increments.
    ///
    /// In scaled form:
    /// ```text
    /// D is the scaling matrix:
    ///     ┌           ┐
    ///     │  1  0  0  │
    /// D = │  0  vₛ 0  │
    ///     │  0  0  bₛ │
    ///     └           ┘
    ///
    /// Fₛ = D·F·D⁻¹
    /// Bₛ = D·B
    /// xₛ = Fₛ·xₛ + Bₛ·u
    /// ```
    pub fn predict_state(&mut self, delta_vel_ned_mps: Vector3f32, delta_pos_ned_m: Vector3f32, dt: f32) {
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
    ///     │  I₃     dt·I₃   -½·dt²·I₃  │
    /// F = │  0₃       I₃      -dt·I₃   │
    ///     │  0₃       0₃         I₃    │
    ///     └                            ┘
    ///
    /// Explanation
    ///
    /// A naïve dense 9×9 matrix multiplication requires 729 scalar multiplications and 648 additions.
    /// Calculating F·P·Fᵀ is even more expensive still.
    /// Because F is sparse and block structured, the covariance can instead be propagated using explicit 3×3 block equations.
    ///
    /// Instead we can divide F into 9 separate 3x3 sub-matrices (blocks) and process each block using 3x3 matrix operations.
    ///
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
    ///
    /// In a Kalman filter tracking Position (P), Velocity (V), and an Accelerometer Bias (B)
    /// over a small time increment (dt), the physical laws of motion give:
    ///     P_new = P + V * dt
    ///     V_new = V - B * dt  (Bias is subtracted from acceleration)
    ///     B_new = B (Bias is modeled as a constant or random walk)
    ///
    /// Expressed as a 3x3 block state transition matrix, that gives:
    ///
    ///     ┌                 9x9                  ┐
    ///     │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    ///     │ │    I     ││   dt·I   ││ -½·dt²·I │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// F = │ │    0     ││    I     ││  -dt·I   │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    ///     │ │    0     ││    0     ││    I     │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     └                                      ┘
    ///
    /// Since this is a sparse matrix, we can calculate F·P·Fᵀ without doing full 9x9 matrix multiplications.
    /// Instead we perform the multiplication block by block.
    ///
    /// Expanding F·P·Fᵀ algebraically gives the following block updates:
    ///
    /// For the position columns (PP, VP, BP)
    ///     PP_new = PP + dt*(PV + VP) + dt2*(VV - 0.5*(PB + BP)) - 0.5*dt3*(VB + BV) + 0.25*dt4*BB
    ///     VP_new = VP + dt*(VV - BP) - dt2*(BV + 0.5*VB) +        0.5*dt3*BB
    ///     BP_new = BP + dt*BV  -       dt2*0.5*BB
    ///
    /// For the velocity columns (PV, VV, BV)
    ///     PV_new = PV + dt*(VV - PB) - dt2*(VB + 0.5*BV) + dt3*0.5*BB
    ///     VV_new = VV - dt*(BV + VB) + dt2*BB
    ///     BV_new = BV - dt*BB
    ///
    /// For the bias columns (PB, VB, BB)
    ///     PB_new = PB + dt*VB - dt2*0.5*BB
    ///     VB_new = VB - dt*BB
    ///     BB_new = BB
    ///
    /// The process noise matrix Q
    ///     ┌                                ┐
    ///     │  ⅓·Q_v·dt³  ½·Q_v·dt²       0  │
    /// Q = │  ½·Q_v·dt²    Q_v·dt        0  │
    ///     │         0          0   Q_b·dt  │
    ///     └                                ┘
    ///
    /// In scaled form:
    ///
    /// Fₛ = D·F·D⁻¹
    /// Qₛ = D·B·Dᵀ
    /// Pₛ = Fₛ·Pₛ·Fₛᵀ + Qₛ
    /// ```
    #[allow(non_snake_case)]
    pub fn predict_covariance(&mut self, dt: f32) {
        debug_assert!(dt != 0.0 && dt.is_finite());
        // =====================================================================
        // PROPAGATE THE COVARIANCE P_new = F·P_old·Fᵀ
        // =====================================================================

        // alias self.P for readability.
        let P = &mut self.P;

        // We can update P in place by ensuring RHS of all P assignments refers to P_old.
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

        P[VV] = P[VV] - dt * (P[BV] + P[VB] - dt * P[BB]);
        retire!(VV);

        P[BV] = P[BV] - dt * P[BB];
        retire!(BV);

        // --- BIAS COLUMNS (PB, VB, BB) ---
        P[PB] = P[PB] + dt * (P[VB] - dt * 0.5 * P[BB]);
        retire!(PB);

        P[VB] = P[VB] - dt * P[BB];
        retire!(VB);
        // P[BB] = P[BB]; so we don't need to update BB.

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
        /*
        Q_PP += (1/20) * Q_bias * dt⁵
        Q_PV += (1/8)  * Q_bias * dt⁴
        Q_VP += (1/8)  * Q_bias * dt⁴
        Q_PB += -(1/6) * Q_bias * dt³
        Q_BP += -(1/6) * Q_bias * dt³
        Q_VV += (1/3)  * Q_bias * dt³
        Q_VB += -(1/2) * Q_bias * dt²
        Q_BV += -(1/2) * Q_bias * dt²
        */
        P[Self::BB].add_diagonal_scalar_in_place(q_bias);

        // Time propagation is highly sensitive to asymmetric shearing, so enforce symmetry on the covariance matrix.
        P.enforce_symmetry();
    }
}

// **** Correct ***

#[allow(non_snake_case)]
impl KalmanFilterXyz {
    /// Phase 2 Altitude Correction using new measurement.
    /// Updates only the vertical Z axis components across all tracking states.
    ///
    /// ### Core Operations
    /// *  `S = P₂₂ + R` (Innovation Variance calculation)
    /// *  `K = P_column_2 * (1.0 / S)` (Kalman Gain column selection extraction)
    /// *  `P = P - K * (H * P)` (Covariance correction step)
    /*        let K_pos = self.P[Self::PP].column(2) * S_inv;
           let K_vel = self.P[Self::VP].column(2) * S_inv;
           let K_acc_bias = self.P[Self::BP].column(2) * S_inv;
    */
    pub fn correct_altitude(&mut self, altitude_m: f32, R: f32, nis_threshold: f32) -> KalmanFilterDiagnostics {
        debug_assert!(altitude_m.is_finite());
        // The altitude measurement directly observes position.z.
        //
        // H = [0 0 1   0 0 0   0 0 0]

        // S = H * P * Hᵀ + R
        //
        // Since H selects position.z:
        // S = P_PP[z,z] + R

        // Innovation Covariance (S) is a scalar
        // Pull the Z-axis variance element directly out of the PP 3x3 block matrix
        let S = self.P[Self::PP][Matrix3x3f32::M33] + R;

        if S.abs() <= f32::EPSILON {
            return KalmanFilterDiagnostics {
                status: FusionStatus::IllConditionedInnovationCovarianceMatrix,
                covariance_trace: self.P[Self::PP].trace(),
                ..Default::default()
            };
        }

        let S_inv = 1.0 / S;

        // P * Hᵀ
        //
        // H selects the Z position column (ie column 2), so these are:
        //
        // P_PP[:,z]
        // P_VP[:,z]
        // P_BP[:,z]

        // The innovation(y) is a scalar.
        let innovation_m = altitude_m - self.state.pos_ned_m.z;

        // Normalized Innovation Squared (NIS) = yᵀ·S⁻¹·y  (aka Mahalanobis Distance)
        let nis = (innovation_m * innovation_m) * S_inv;
        let mut diagnostics = KalmanFilterDiagnostics {
            status: FusionStatus::Accepted,
            covariance_trace: self.P[Self::PP].trace(),
            covariance_reciprocal_condition: None,
            nis,
            innovation_m: Vector3f32 { x: 0.0, y: 0.0, z: innovation_m },
        };
        if nis > nis_threshold {
            // nis exceeds nis_threshold, so reject this measurement as an outlier.
            diagnostics.status = FusionStatus::RejectedOutlier;
            return diagnostics;
        }

        // Calculate the Kalman Gain (K) as 3D vectors
        // H isolates row index 2, so P * Hᵀ extracts column 2 of the sub-blocks.
        let P_pos = self.P[Self::PP].column(2);
        let P_vel = self.P[Self::VP].column(2);
        let P_acc_bias = self.P[Self::BP].column(2);

        // Calculate K, the Kalman gain.
        //
        // K = P·Hᵀ·S⁻¹
        //
        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // Update the estimated state vector
        // x += K·y
        self.state.pos_ned_m += K_pos * innovation_m;
        self.state.vel_ned_mps += K_vel * innovation_m;
        self.state.acc_bias_ned_mps2 += K_acc_bias * innovation_m;

        // Covariance update (rank-1 outer product)
        //
        // P -= (P·Hᵀ)(H·P) / S
        //
        self.P[Self::PP] -= Matrix3x3f32::outer_product(K_pos, P_pos);
        self.P[Self::PV] -= Matrix3x3f32::outer_product(K_pos, P_vel);
        self.P[Self::PB] -= Matrix3x3f32::outer_product(K_pos, P_acc_bias);

        self.P[Self::VP] -= Matrix3x3f32::outer_product(K_vel, P_pos);
        self.P[Self::VV] -= Matrix3x3f32::outer_product(K_vel, P_vel);
        self.P[Self::VB] -= Matrix3x3f32::outer_product(K_vel, P_acc_bias);

        self.P[Self::BP] -= Matrix3x3f32::outer_product(K_acc_bias, P_pos);
        self.P[Self::BV] -= Matrix3x3f32::outer_product(K_acc_bias, P_vel);
        self.P[Self::BB] -= Matrix3x3f32::outer_product(K_acc_bias, P_acc_bias);

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();

        diagnostics
    }

    /// Executes an asynchronous measurement update when a new 3D GPS reading arrives
    /// (typically at a slower 1Hz to 10Hz rate).
    /// The error becomes a 3D vector, and the 3D Position, Velocity, and Accelerometer Bias states.
    ///
    /// ### Core Operations
    /// *  Extracts `PositionPosition` sub-block from P (top left 3x3 matrix).
    /// *  `S = H·P·Hᵀ + R` (yields a 3x3 innovation matrix)
    /// *  `S⁻¹ = try_inverse(S)`
    /// *  `K = (P·Hᵀ)·S⁻¹` (yields a 9x3 block matrix representation)
    ///
    /// Performs a full 3D Position Correction (e.g., GPS or Optical Flow position packet).
    /// Observations sample the full x, y, and z position channels simultaneously.
    ///
    /// Note: we assume the measurement errors are not cross-correlated (that is the x, y, and z sensor noises are independent),
    /// This means the 3x3 noise covariance matrix `R` is diagonal and so can be represented by a 3d vector.
    ///
    /// Layouts:
    /// * self.P: 9x9 Column-Major Covariance Matrix
    /// * position: Vector3f32 observation `[x, y, z]`
    /// * R: Vector3f32 diagonal measurement noise variance [R.x, R.y, R.z]
    pub fn correct_position_short_form(
        &mut self,
        measurement_m: Vector3f32,
        predicted_measurement_m: Vector3f32,
        R: Vector3f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        debug_assert!(measurement_m.is_finite());
        debug_assert!(predicted_measurement_m.is_finite());
        // H = [ I₃   0₃   0₃ ]
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
        // S is symmetric, so we can just use its upper triangle.
        let (S_adj, determinant, reciprocal_condition) = S.upper_triangle().adjugate();
        if reciprocal_condition < Self::MIN_RECIPROCAL_CONDITION {
            let diagnostics = KalmanFilterDiagnostics {
                status: FusionStatus::IllConditionedInnovationCovarianceMatrix,
                covariance_trace: self.P[Self::PP].trace(),
                covariance_reciprocal_condition: Some(reciprocal_condition),
                ..Default::default()
            };

            self.P.enforce_components_diagonal_lower_bound(1e-6);
            return diagnostics;
        }
        let S_inv = S_adj * (1.0 / determinant);

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measurement_m - predicted_measurement_m;

        // Normalized Innovation Squared (NIS) = yᵀ·S⁻¹·y  (aka Mahalanobis Distance)
        // Dot product results in a unitless scalar
        let nis = innovation_m.dot(S_inv * innovation_m);
        let mut diagnostics = KalmanFilterDiagnostics {
            status: FusionStatus::Accepted,
            covariance_trace: self.P[Self::PP].trace(),
            covariance_reciprocal_condition: Some(reciprocal_condition),
            nis,
            innovation_m,
        };

        if nis > nis_threshold {
            // nis exceeds nis_threshold, so reject this measurement as an outlier.
            diagnostics.status = FusionStatus::RejectedOutlier;
            return diagnostics;
        }

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
        measurement_m: Vector3f32,
        predicted_measurement_m: Vector3f32,
        R: Vector3f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        debug_assert!(measurement_m.is_finite());
        debug_assert!(predicted_measurement_m.is_finite());
        // H = [ I₃   0₃   0₃ ]
        // S = H·P·Hᵀ + R
        // y = z - H·x
        // K = P·Hᵀ·S⁻¹
        // x = x + K·y
        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ); equivalent to P - K·H·P

        // Calculate S, the Innovation Covariance matrix:
        // S = H·P·Hᵀ + R
        let S = self.P[Self::PP].add_diagonal_vector(R);

        // S is guaranteed to be positive-definite, since we are using Joseph's Stabilized Form, so we can use
        // * matrix inverse
        // * a Cholesky solver.
        // * inverse using upper triangle
        // Cholesky solver takes 14 arithmetic operations.
        // matrix inverse takes 39 multiplications, 19 additions/subtractions and 1 division (1 div ~= 4 mul).
        // upper triangle matrix inverse takes 21 multiplications, 8 additions/subtractions and 1 division (1 div ~= 4 mul).
        // Upper triangle matrix is overall computationally more efficient, and almost as numerically stable as Cholesky solver.
        let (S_adj, determinant, reciprocal_condition) = S.upper_triangle().adjugate();
        if reciprocal_condition < Self::MIN_RECIPROCAL_CONDITION {
            let diagnostics = KalmanFilterDiagnostics {
                status: FusionStatus::IllConditionedInnovationCovarianceMatrix,
                covariance_trace: self.P[Self::PP].trace(),
                covariance_reciprocal_condition: Some(reciprocal_condition),
                ..Default::default()
            };

            self.P.enforce_components_diagonal_lower_bound(1e-6);
            return diagnostics;
        }
        let S_inv = S_adj * (1.0 / determinant);

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measurement_m - predicted_measurement_m;

        // Normalized Innovation Squared (NIS) = yᵀ·S⁻¹·y  (aka Mahalanobis Distance)
        let s_inv_times_innovation = S_inv * innovation_m;
        // Dot product results in a unitless scalar
        let nis = innovation_m.dot(s_inv_times_innovation);
        let mut diagnostics = KalmanFilterDiagnostics {
            status: FusionStatus::Accepted,
            covariance_trace: self.P[Self::PP].trace(),
            covariance_reciprocal_condition: Some(reciprocal_condition),
            nis,
            innovation_m,
        };

        if nis > nis_threshold {
            // nis exceeds nis_threshold, so reject this measurement as an outlier.
            diagnostics.status = FusionStatus::RejectedOutlier;
            return diagnostics;
        }

        // Calculate K, the Kalman gain.
        // K = P·Hᵀ·S⁻¹
        // Cholesky solver is 60 arithmetic operations, matrix multiply is 45, symmetric matrix multiply is 27
        let K_pos = self.P[Self::PP] * S_inv;
        let K_vel = self.P[Self::VP] * S_inv;
        let K_acc_bias = self.P[Self::BP] * S_inv;

        // Update the estimated state vector
        // x += K·y
        self.state.pos_ned_m += K_pos * innovation_m;
        self.state.vel_ned_mps += K_vel * innovation_m;
        self.state.acc_bias_ned_mps2 += K_acc_bias * innovation_m;

        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ

        // Calculate the columns of intermediate matrix A = (I - KH)·P
        let A = Matrix3x3xM3x3f32::from_column_array([
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

        let I_minus_K_pos_t = Matrix3x3f32::identity() - K_pos_t;

        // Calculate J = A·(I - K·H)ᵀ.
        //
        // For a position measurement:
        //
        // I - K·H =
        //
        // [ I - K_pos  0   0 ]
        // [ - K_vel    I   0 ]
        // [ -K_bias    0   I ]
        //
        // Therefore:
        //
        // J = A·(I-K·H)ᵀ
        //
        // The resulting blocks are:
        //
        // Column 0:
        // J_PP = A_PP·(I-K_posᵀ)
        // J_VP = A_VP·(I-K_posᵀ)
        // J_BP = A_BP·(I-K_posᵀ)
        //
        // Column 1:
        // J_PV = A_PV - A_PP·K_velᵀ
        // J_VV = A_VV - A_VP·K_velᵀ
        // J_BV = A_BV - A_BP·K_velᵀ
        //
        // Column 2:
        // J_PB = A_PB - A_PP·K_biasᵀ
        // J_VB = A_VB - A_VP·K_biasᵀ
        // J_BB = A_BB - A_BP·K_biasᵀ

        let J = [
            // Column 0: position.
            A[Self::PP] * I_minus_K_pos_t,
            A[Self::VP] * I_minus_K_pos_t,
            A[Self::BP] * I_minus_K_pos_t,
            // Column 1: velocity.
            A[Self::PV] - A[Self::PP] * K_vel_t,
            A[Self::VV] - A[Self::VP] * K_vel_t,
            A[Self::BV] - A[Self::BP] * K_vel_t,
            // Column 2: accelerometer bias.
            A[Self::PB] - A[Self::PP] * K_acc_bias_t,
            A[Self::VB] - A[Self::VP] * K_acc_bias_t,
            A[Self::BB] - A[Self::BP] * K_acc_bias_t,
        ];

        // Calculate K·R·Kᵀ blocks.
        let K_pos_R = K_pos.mul_diagonal_vector(R);
        let K_vel_R = K_vel.mul_diagonal_vector(R);
        let K_acc_bias_R = K_acc_bias.mul_diagonal_vector(R);

        // Reassemble J and K·R·Kᵀ into P.

        self.P[Self::PP] = J[0] + K_pos_R * K_pos_t;
        self.P[Self::VP] = J[1] + K_vel_R * K_pos_t;
        self.P[Self::BP] = J[2] + K_acc_bias_R * K_pos_t;

        self.P[Self::PV] = J[3] + K_pos_R * K_vel_t;
        self.P[Self::VV] = J[4] + K_vel_R * K_vel_t;
        self.P[Self::BV] = J[5] + K_acc_bias_R * K_vel_t;

        self.P[Self::PB] = J[6] + K_pos_R * K_acc_bias_t;
        self.P[Self::VB] = J[7] + K_vel_R * K_acc_bias_t;
        self.P[Self::BB] = J[8] + K_acc_bias_R * K_acc_bias_t;

        // Ensure numerical stability by enforcing symmetry.
        self.P.enforce_symmetry();
        diagnostics
    }

    pub fn correct_position(
        &mut self,
        position_m: Vector3f32,
        R: Vector3f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        self.correct_position_joseph_stabilized_form(position_m, self.state.pos_ned_m, R, nis_threshold)
    }

    pub fn correct_position_and_velocity(
        &mut self,
        position_m: Vector3f32,
        R_position: Vector3f32,
        velocity_mps: Vector3f32,
        R_velocity: Vector3f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        self.correct_position_and_velocity_joseph_stabilized_form(
            position_m,
            self.state.pos_ned_m,
            R_position,
            velocity_mps,
            self.state.vel_ned_mps,
            R_velocity,
            nis_threshold,
        )
    }

    /// Joseph's Stabilized Form for the velocity covariance update:
    ///
    /// `P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ`.
    ///
    /// `H = [ 0₃   I₃   0₃ ]`.
    ///
    /// This function corrects the navigation state using a 3D velocity measurement.
    pub fn correct_velocity_joseph_stabilized_form(
        &mut self,
        velocity_mps: Vector3f32,
        predicted_velocity_mps: Vector3f32,
        R: Vector3f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        debug_assert!(velocity_mps.is_finite());
        debug_assert!(predicted_velocity_mps.is_finite());

        // H = [ 0₃   I₃   0₃ ]
        // S = H·P·Hᵀ + R
        // y = z - H·x
        // K = P·Hᵀ·S⁻¹
        // x = x + K·y
        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ

        // Calculate S, the Innovation Covariance matrix:
        // S = P_VV + R
        let S = self.P[Self::VV].add_diagonal_vector(R);

        // Calculate the upper-triangle adjugate.
        let (S_adj, determinant, reciprocal_condition) = S.upper_triangle().adjugate();

        if reciprocal_condition < Self::MIN_RECIPROCAL_CONDITION {
            let diagnostics = KalmanFilterDiagnostics {
                status: FusionStatus::IllConditionedInnovationCovarianceMatrix,
                covariance_trace: self.P[Self::PP].trace(),
                covariance_reciprocal_condition: Some(reciprocal_condition),
                ..Default::default()
            };

            self.P.enforce_components_diagonal_lower_bound(1e-6);
            return diagnostics;
        }

        let S_inv = S_adj * (1.0 / determinant);

        // Calculate the innovation.
        // y = z - H·x
        let innovation_velocity = velocity_mps - predicted_velocity_mps;

        // Normalized Innovation Squared (NIS):
        // NIS = yᵀ·S⁻¹·y
        let s_inv_times_innovation = S_inv * innovation_velocity;

        let nis = innovation_velocity.dot(s_inv_times_innovation);

        let mut diagnostics = KalmanFilterDiagnostics {
            status: FusionStatus::Accepted,
            covariance_trace: self.P[Self::PP].trace(),
            covariance_reciprocal_condition: Some(reciprocal_condition),
            nis,
            innovation_m: innovation_velocity,
        };

        if nis > nis_threshold {
            diagnostics.status = FusionStatus::RejectedOutlier;
            return diagnostics;
        }

        // Calculate K, the Kalman gain.
        // K = P·Hᵀ·S⁻¹
        let K_pos = self.P[Self::PV] * S_inv;
        let K_vel = self.P[Self::VV] * S_inv;
        let K_acc_bias = self.P[Self::BV] * S_inv;

        // Update the estimated state vector.
        // x += K·y
        self.state.pos_ned_m += K_pos * innovation_velocity;
        self.state.vel_ned_mps += K_vel * innovation_velocity;
        self.state.acc_bias_ned_mps2 += K_acc_bias * innovation_velocity;

        // Joseph covariance update:
        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ

        // Calculate the columns of intermediate matrix:
        // A = (I - K·H)·P

        let A = Matrix3x3xM3x3f32::from_column_array([
            // Column 0: position column submatrices.
            self.P[Self::PP] - K_pos * self.P[Self::VP],
            self.P[Self::VP] - K_vel * self.P[Self::VP],
            self.P[Self::BP] - K_acc_bias * self.P[Self::VP],
            // Column 1: velocity column submatrices.
            self.P[Self::PV] - K_pos * self.P[Self::VV],
            self.P[Self::VV] - K_vel * self.P[Self::VV],
            self.P[Self::BV] - K_acc_bias * self.P[Self::VV],
            // Column 2: bias column submatrices.
            self.P[Self::PB] - K_pos * self.P[Self::VB],
            self.P[Self::VB] - K_vel * self.P[Self::VB],
            self.P[Self::BB] - K_acc_bias * self.P[Self::VB],
        ]);

        // Precompute transposes and helper terms.
        let K_pos_t = K_pos.transpose();
        let K_vel_t = K_vel.transpose();
        let K_acc_bias_t = K_acc_bias.transpose();

        let I_minus_K_vel_t = Matrix3x3f32::identity() - K_vel_t;

        // Calculate J = A·(I - K·H)ᵀ.
        // Only calculate the values that differ from A.

        let J = [
            // Column 0: position column submatrices.
            A[Self::PP] - A[Self::PV] * K_pos_t,
            A[Self::VP] - A[Self::VV] * K_pos_t,
            A[Self::BP] - A[Self::BV] * K_pos_t,
            // Column 1: velocity column submatrices.
            A[Self::PV] * I_minus_K_vel_t,
            A[Self::VV] * I_minus_K_vel_t,
            A[Self::BV] * I_minus_K_vel_t,
            // Column 2: bias column submatrices.
            A[Self::PB] - A[Self::PV] * K_acc_bias_t,
            A[Self::VB] - A[Self::VV] * K_acc_bias_t,
            A[Self::BB] - A[Self::BV] * K_acc_bias_t,
        ];

        // Calculate K·R·Kᵀ blocks.
        let K_pos_R = K_pos.mul_diagonal_vector(R);
        let K_vel_R = K_vel.mul_diagonal_vector(R);
        let K_acc_bias_R = K_acc_bias.mul_diagonal_vector(R);

        // Reassemble J into P, adding K·R·Kᵀ.

        // Column 0: position.
        self.P[Self::PP] = J[Self::PP] + K_pos_R * K_pos_t;
        self.P[Self::VP] = J[Self::VP] + K_vel_R * K_pos_t;
        self.P[Self::BP] = J[Self::BP] + K_acc_bias_R * K_pos_t;

        // Column 1: velocity.
        self.P[Self::PV] = J[Self::PV] + K_pos_R * K_vel_t;
        self.P[Self::VV] = J[Self::VV] + K_vel_R * K_vel_t;
        self.P[Self::BV] = J[Self::BV] + K_acc_bias_R * K_vel_t;

        // Column 2: accelerometer bias.
        self.P[Self::PB] = J[Self::PB] + K_pos_R * K_acc_bias_t;
        self.P[Self::VB] = J[Self::VB] + K_vel_R * K_acc_bias_t;
        self.P[Self::BB] = J[Self::BB] + K_acc_bias_R * K_acc_bias_t;
        // Ensure numerical stability by enforcing symmetry.
        self.P.enforce_symmetry();

        diagnostics
    }

    /// Joseph's Stabilized Form for combined GPS position and velocity correction.
    ///
    /// Uses sequential conditioning to perform a mathematically equivalent
    /// 6-dimensional measurement update using 3x3 matrix operations.
    ///
    /// Assumes independent GPS position and velocity measurement errors.
    ///
    /// The combined NIS is:
    /// NIS = `NIS_position` + `NIS_velocity_conditional`.
    #[allow(clippy::too_many_arguments)]
    pub fn correct_position_and_velocity_joseph_stabilized_form(
        &mut self,
        measurement_m: Vector3f32,
        predicted_measurement_m: Vector3f32,
        R_position: Vector3f32,
        velocity_mps: Vector3f32,
        predicted_velocity_mps: Vector3f32,
        R_velocity: Vector3f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        debug_assert!(measurement_m.is_finite());
        debug_assert!(predicted_measurement_m.is_finite());
        debug_assert!(velocity_mps.is_finite());
        debug_assert!(predicted_velocity_mps.is_finite());

        // for position H_p = [I 0 0]
        // for velocity H_p = [0 I 0]

        // Work on a temporary copy so that both measurements are accepted
        // or rejected together.
        let mut candidate = *self;

        // ------------------------------------------------------------
        // 1. Perform the position correction.
        // ------------------------------------------------------------

        // Use an infinite threshold here because we want to evaluate
        // the combined 6-dimensional NIS rather than position NIS alone.
        let mut diagnostics = candidate.correct_position_joseph_stabilized_form(
            measurement_m,
            predicted_measurement_m,
            R_position,
            f32::INFINITY,
        );

        // Abort if the position innovation covariance is ill-conditioned.
        if diagnostics.status != FusionStatus::Accepted {
            return diagnostics;
        }

        let nis_position = diagnostics.nis;

        // ------------------------------------------------------------
        // 2. Calculate the conditional velocity innovation.
        // ------------------------------------------------------------

        // The position correction changes the estimated velocity.
        // Account for this when calculating the conditional velocity innovation.
        let velocity_correction_from_position = candidate.state.vel_ned_mps - self.state.vel_ned_mps;

        let innovation_velocity = velocity_mps - (predicted_velocity_mps + velocity_correction_from_position);

        // ------------------------------------------------------------
        // 3. Calculate the conditional velocity innovation covariance.
        // ------------------------------------------------------------

        // S_velocity = P_VV (after position correction) + R_velocity
        let S_velocity = candidate.P[Self::VV].add_diagonal_vector(R_velocity);

        let (S_velocity_adj, determinant, reciprocal_condition) = S_velocity.upper_triangle().adjugate();

        if reciprocal_condition < Self::MIN_RECIPROCAL_CONDITION {
            diagnostics.status = FusionStatus::IllConditionedInnovationCovarianceMatrix;

            diagnostics.covariance_reciprocal_condition = Some(reciprocal_condition);

            return diagnostics;
        }

        let S_velocity_inv = S_velocity_adj * (1.0 / determinant);

        // ------------------------------------------------------------
        // 4. Calculate the conditional velocity NIS.
        // ------------------------------------------------------------

        let s_velocity_inv_times_innovation = S_velocity_inv * innovation_velocity;

        let nis_velocity = innovation_velocity.dot(s_velocity_inv_times_innovation);

        // Combined NIS for the 6-dimensional measurement.
        let nis = nis_position + nis_velocity;

        diagnostics.nis = nis;

        // The existing diagnostics structure has only one innovation vector.
        // Preserve its existing position innovation.
        if nis > nis_threshold {
            diagnostics.status = FusionStatus::RejectedOutlier;
            return diagnostics;
        }

        // ------------------------------------------------------------
        // 5. Calculate the velocity Kalman gain.
        // ------------------------------------------------------------

        // H_velocity = [ 0  I  0 ]
        //
        // K = P * H_velocity^T * S_velocity^-1

        let K_pos = candidate.P[Self::PV] * S_velocity_inv;
        let K_vel = candidate.P[Self::VV] * S_velocity_inv;
        let K_acc_bias = candidate.P[Self::BV] * S_velocity_inv;

        // ------------------------------------------------------------
        // 6. Update the estimated state vector.
        // ------------------------------------------------------------

        candidate.state.pos_ned_m += K_pos * innovation_velocity;
        candidate.state.vel_ned_mps += K_vel * innovation_velocity;
        candidate.state.acc_bias_ned_mps2 += K_acc_bias * innovation_velocity;

        // ------------------------------------------------------------
        // 7. Joseph covariance update.
        //
        // P = (I - K*H) * P * (I - K*H)^T + K*R*K^T
        //
        // Here H = [ 0  I  0 ].
        // ------------------------------------------------------------

        // Calculate the columns of intermediate matrix:
        // A = (I - K*H) * P

        let A = Matrix3x3xM3x3f32::from_column_array([
            // Column 0: position column submatrices.
            candidate.P[Self::PP] - K_pos * candidate.P[Self::VP],
            candidate.P[Self::VP] - K_vel * candidate.P[Self::VP],
            candidate.P[Self::BP] - K_acc_bias * candidate.P[Self::VP],
            // Column 1: velocity column submatrices.
            candidate.P[Self::PV] - K_pos * candidate.P[Self::VV],
            candidate.P[Self::VV] - K_vel * candidate.P[Self::VV],
            candidate.P[Self::BV] - K_acc_bias * candidate.P[Self::VV],
            // Column 2: bias column submatrices.
            candidate.P[Self::PB] - K_pos * candidate.P[Self::VB],
            candidate.P[Self::VB] - K_vel * candidate.P[Self::VB],
            candidate.P[Self::BB] - K_acc_bias * candidate.P[Self::VB],
        ]);

        // Precompute transposes and helper terms.
        let K_pos_t = K_pos.transpose();
        let K_vel_t = K_vel.transpose();
        let K_acc_bias_t = K_acc_bias.transpose();

        let I_minus_K_vel_t = Matrix3x3f32::identity() - K_vel_t;

        // Calculate J = A * (I - K*H)^T.
        //
        // Only calculate the values that differ from A.

        let J = [
            // Column 0: position column submatrices.
            A[Self::PP] - A[Self::PV] * K_pos_t,
            A[Self::VP] - A[Self::VV] * K_pos_t,
            A[Self::BP] - A[Self::BV] * K_pos_t,
            // Column 1: velocity column submatrices.
            A[Self::PV] * I_minus_K_vel_t,
            A[Self::VV] * I_minus_K_vel_t,
            A[Self::BV] * I_minus_K_vel_t,
            // Column 2: bias column submatrices.
            A[Self::PB] - A[Self::PV] * K_acc_bias_t,
            A[Self::VB] - A[Self::VV] * K_acc_bias_t,
            A[Self::BB] - A[Self::BV] * K_acc_bias_t,
        ];

        // ------------------------------------------------------------
        // 8. Calculate K * R_velocity * K^T blocks.
        // ------------------------------------------------------------

        let K_pos_R = K_pos.mul_diagonal_vector(R_velocity);
        let K_vel_R = K_vel.mul_diagonal_vector(R_velocity);
        let K_acc_bias_R = K_acc_bias.mul_diagonal_vector(R_velocity);

        // ------------------------------------------------------------
        // 9. Reassemble J and A into P.
        // ------------------------------------------------------------

        candidate.P[Self::PP] = J[Self::PP] + K_pos_R * K_pos_t;
        candidate.P[Self::VP] = J[Self::VP] + K_vel_R * K_pos_t;
        candidate.P[Self::BP] = J[Self::BP] + K_acc_bias_R * K_pos_t;

        candidate.P[Self::PV] = J[Self::PV] + K_pos_R * K_vel_t;
        candidate.P[Self::VV] = J[Self::VV] + K_vel_R * K_vel_t;
        candidate.P[Self::BV] = J[Self::BV] + K_acc_bias_R * K_vel_t;

        candidate.P[Self::PB] = J[Self::PB] + K_pos_R * K_acc_bias_t;
        candidate.P[Self::VB] = J[Self::VB] + K_vel_R * K_acc_bias_t;
        candidate.P[Self::BB] = J[Self::BB] + K_acc_bias_R * K_acc_bias_t;

        // ------------------------------------------------------------
        // 10. Ensure numerical stability by enforcing symmetry.
        // ------------------------------------------------------------

        candidate.P.enforce_symmetry();

        // ------------------------------------------------------------
        // 11. Commit the accepted correction.
        // ------------------------------------------------------------

        *self = candidate;

        diagnostics
    }

    // TODO: correct_position_xy_joseph_stabilized_form to support optical flow corrections
    pub fn correct_position_xy_joseph_stabilized_form(
        &mut self,
        position_m: Vector2f32,
        past_pos_m: Vector3f32,
        R: Vector2f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        debug_assert!(position_m.is_finite());
        debug_assert!(past_pos_m.is_finite());
        // TODO: implement correct_position_xy_delayed.
        _ = self;
        _ = position_m;
        _ = past_pos_m;
        _ = R;
        _ = nis_threshold;
        unimplemented!();
    }

    pub fn correct_position_xy(
        &mut self,
        position_m: Vector2f32,
        R: Vector2f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        self.correct_position_xy_joseph_stabilized_form(position_m, self.state.pos_ned_m, R, nis_threshold)
    }
}

#[allow(non_snake_case)]
impl KalmanFilterXyz {
    /// Non-linear correction using anchor distance.
    pub fn correct_position_using_anchor_distance(
        &mut self,
        anchor_position_m: Vector3f32,
        measured_distance_m: f32,
        R: f32,
        nis_threshold: f32,
    ) -> KalmanFilterDiagnostics {
        const MIN_DISTANCE_M: f32 = 1.0e-4;
        debug_assert!(anchor_position_m.is_finite());
        debug_assert!(measured_distance_m.is_finite());

        if measured_distance_m < 0.0 {
            return KalmanFilterDiagnostics { status: FusionStatus::InvalidMeasurement, ..Default::default() };
        }
        // Position relative to the anchor.
        let position_m = self.state.pos_ned_m - anchor_position_m;

        // Predicted range.
        let predicted_distance_m = position_m.norm();

        // The range Jacobian is undefined at zero distance.
        if predicted_distance_m <= MIN_DISTANCE_M {
            return KalmanFilterDiagnostics { status: FusionStatus::InvalidMeasurement, ..Default::default() };
        }

        // Jacobian with respect to position:
        // H = [ dx/r, dy/r, dz/r ]
        //
        // Vector3 is used as a row vector for:
        //     H·P
        //
        // and as a column vector for:
        //     P·Hᵀ
        let H = position_m / predicted_distance_m;

        // P ·Hᵀ for each state block.
        // u = P·Hᵀ
        //
        //     [ P_PP·Hᵀ ]
        // u = [ P_VP·Hᵀ ]
        //     [ P_BP·Hᵀ ]

        // P·Hᵀ
        let P_pos = self.P[Self::PP] * H;
        let P_vel = self.P[Self::VP] * H;
        let P_acc_bias = self.P[Self::BP] * H;

        // Innovation covariance:
        //
        // S = H·P·Hᵀ + R
        //   = H·P_PP·Hᵀ + R
        let S = H.dot(P_pos) + R;
        if S <= f32::EPSILON {
            return KalmanFilterDiagnostics {
                status: FusionStatus::IllConditionedInnovationCovarianceMatrix,
                covariance_trace: self.P[Self::PP].trace(),
                ..Default::default()
            };
        }
        let S_inv = 1.0 / S;

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measured_distance_m - predicted_distance_m;

        // Normalized innovation squared (Mahalanobis distance squared).
        let nis = innovation_m * innovation_m * S_inv;
        let mut diagnostics = KalmanFilterDiagnostics {
            status: FusionStatus::Accepted,
            covariance_trace: self.P[Self::PP].trace(),
            covariance_reciprocal_condition: None,
            nis,
            innovation_m: Vector3f32 { x: 0.0, y: 0.0, z: innovation_m },
        };

        if nis > nis_threshold {
            // Reject this measurement as an outlier.
            diagnostics.status = FusionStatus::RejectedOutlier;
            return diagnostics;
        }

        // Calculate K, the Kalman gain.
        // K = P·Hᵀ·S⁻¹
        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // Update the estimated state vector
        // x += K·y
        self.state.pos_ned_m += K_pos * innovation_m;
        self.state.vel_ned_mps += K_vel * innovation_m;
        self.state.acc_bias_ned_mps2 += K_acc_bias * innovation_m;

        // Covariance update:
        //     P_new = P - K·H·P
        // where:
        //     K = P·Hᵀ / S
        // Therefore, assuming P is symmetric:
        //     P_new = P - (P·Hᵀ)·(P·Hᵀ)ᵀ / S
        // This is a rank-1 update of the full 9x9 covariance.
        self.P[Self::PP] -= Matrix3x3::outer_product(K_pos, P_pos);
        self.P[Self::PV] -= Matrix3x3::outer_product(K_pos, P_vel);
        self.P[Self::PB] -= Matrix3x3::outer_product(K_pos, P_acc_bias);

        self.P[Self::VP] -= Matrix3x3::outer_product(K_vel, P_pos);
        self.P[Self::VV] -= Matrix3x3::outer_product(K_vel, P_vel);
        self.P[Self::VB] -= Matrix3x3::outer_product(K_vel, P_acc_bias);

        self.P[Self::BP] -= Matrix3x3::outer_product(K_acc_bias, P_pos);
        self.P[Self::BV] -= Matrix3x3::outer_product(K_acc_bias, P_vel);
        self.P[Self::BB] -= Matrix3x3::outer_product(K_acc_bias, P_acc_bias);

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();

        diagnostics
    }
}

// **** Validate ***

impl KalmanFilterXyz {
    /// Evaluates if an incoming innovation vector satisfies chi-squared gating thresholds.
    /// Formula: `d² = yᵀ·S⁻¹·y`.
    ///
    /// Layouts:
    /// * `self.P`: 9x9 Column-Major Covariance Matrix (used to extract the 3x3 S matrix block)
    /// * `y`: 3-element measurement innovation `[y_x, y_y, y_z]`
    /// * `R`: 3-element diagonal measurement noise variance array `[R_x, R_y, R_z]`
    /// * `gate_threshold`: Chi-squared limit (e.g., 7.815 for 3 DOF at 95% confidence)
    #[must_use]
    #[allow(non_snake_case)]
    pub fn validate_measurement(&self, y: Vector3f32, R: Vector3f32, nis_threshold: f32) -> bool {
        // Collect the columns into an array using standard iteration.
        /*let mut col_iter = self.P.iter_columns();
        let (Some(col0), Some(col1), Some(col2)) = (col_iter.next(), col_iter.next(), col_iter.next()) else {
            return false; // Structured pipeline fallback safety
        };*/
        let col0 = self.P.column(0);
        let col1 = self.P.column(1);
        let col2 = self.P.column(2);

        // H selects position states (rows 0, 1, 2) from columns 0, 1, 2 of matrix P.
        // We pack these into our a Matrix3x3f32.
        #[rustfmt::skip]
        let S = Matrix3x3f32::from_column_array([
            col0[0] + R.x, col0[1],       col0[2],
            col1[0],       col1[1] + R.y, col1[2],
            col2[0],       col2[1],       col2[2] + R.z,
        ]);

        let Some(S_inv) = S.try_inverse(1e-5) else {
            return false;
        };

        // Calculate the vector solution product: x_sol = S⁻¹·y
        let x_sol = S_inv * y;

        // Normalized Innovation Squared (NIS) = yᵀ·S⁻¹·y  (aka Mahalanobis Distance)
        let nis = y.dot(x_sol);

        // Returns true if the measurement innovation vector fits inside standard tolerances
        nis <= nis_threshold
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStateXyz {
    // 3D Kinematic State Vectors
    /// Position {x, y, z}, meters, in the NED frame.
    pub pos_ned_m: Vector3f32,
    /// Velocity {x, y, z}, meters per second, in the NED frame.
    pub vel_ned_mps: Vector3f32,
    /// Accelerometer Bias {x, y, z}, meters per second squared, in the NED frame.
    pub acc_bias_ned_mps2: Vector3f32,
}

impl Default for KalmanStateXyz {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanStateXyz {
    // const SCALE_POSITION: f32 = 1.0; // not used, since this is set to 1.0
    //const SCALE_VELOCITY: f32 = 256.0;
    //const SCALE_ACC_BIAS: f32 = 65,536; // or 131_072.0;
    const SCALE_VELOCITY: f32 = 1.0;
    const SCALE_ACC_BIAS: f32 = 1.0;

    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pos_ned_m: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            vel_ned_mps: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
            acc_bias_ned_mps2: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
        }
    }
}
impl KalmanStateXyz {
    #[inline]
    pub fn set(&mut self, state: &Self) {
        self.pos_ned_m = state.pos_ned_m;
        self.vel_ned_mps = state.vel_ned_mps * Self::SCALE_VELOCITY;
        self.acc_bias_ned_mps2 = state.acc_bias_ned_mps2 * Self::SCALE_ACC_BIAS;
    }

    #[inline]
    #[must_use]
    pub fn get(&self) -> Self {
        KalmanStateXyz {
            pos_ned_m: self.pos_ned_m,
            vel_ned_mps: self.vel_ned_mps / Self::SCALE_VELOCITY,
            acc_bias_ned_mps2: self.acc_bias_ned_mps2 / Self::SCALE_ACC_BIAS,
        }
    }
    #[inline]
    #[must_use]
    pub fn vel_ned_mps(&self) -> Vector3f32 {
        self.vel_ned_mps / Self::SCALE_VELOCITY
    }
    #[inline]
    #[must_use]
    pub fn acc_bias_ned_mps2(&self) -> Vector3f32 {
        self.acc_bias_ned_mps2 / Self::SCALE_ACC_BIAS
    }
}

//#[cfg(test)]
impl KalmanFilterXyz {
    /// Constructor for test code.
    #[allow(non_snake_case)]
    #[must_use]
    pub fn new_for_test(state: KalmanStateXyz, P: Matrix3x3xM3x3f32, Q_velocity: f32, Q_bias: f32) -> Self {
        Self { state, P, Q_velocity, Q_bias }
    }
    /// Return state for test code.
    #[must_use]
    pub fn state_for_test(&self) -> KalmanStateXyz {
        self.state
    }
    /// Return covariance for test code.
    #[must_use]
    pub fn covariance_for_test(&self) -> Matrix3x3xM3x3f32 {
        self.P
    }
    /// Return covariance blocks for test code.
    #[must_use]
    pub fn covariance_blocks_for_test(&self) -> [Matrix3x3<f32>; 9] {
        [
            self.P[Self::PP],
            self.P[Self::VP],
            self.P[Self::BP],
            self.P[Self::PV],
            self.P[Self::VV],
            self.P[Self::BV],
            self.P[Self::PB],
            self.P[Self::VB],
            self.P[Self::BB],
        ]
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterXyz>();
        is_full::<KalmanStateXyz>();
    }
}
