use vqm::{Matrix3x3, Matrix3x3f32, Matrix3x3xM3x3, Matrix3x3xM3x3f32, Vector2f32, Vector3f32};

use super::KalmanFilterDiagnostics;

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
    /// Count to avoid filter latching.
    altitude_consecutive_rejection_count: u32,
    position_consecutive_rejection_count: u32,
}

impl Default for KalmanFilterXyz {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl KalmanFilterXyz {
    const G_MPS2: f32 = 9.80665;

    pub const M11: usize = Matrix3x3xM3x3f32::M11;
    pub const M22: usize = Matrix3x3xM3x3f32::M22;
    pub const M33: usize = Matrix3x3xM3x3f32::M33;

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
        P[Self::PP][Self::M11] = 1.0;
        P[Self::PP][Self::M22] = 1.0;
        P[Self::PP][Self::M33] = 1.0;

        // Seed initial Velocity uncertainty (ie, confident within 0.5 m/s)
        P[Self::VV][Self::M11] = 0.25;
        P[Self::VV][Self::M22] = 0.25;
        P[Self::VV][Self::M33] = 0.25;

        // Seed initial Bias uncertainty (ie, accelerometer bias)
        P[Self::BB][Self::M11] = 0.01;
        P[Self::BB][Self::M22] = 0.01;
        P[Self::BB][Self::M33] = 0.01;

        Self {
            state: KalmanStateXyz::new(),
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

    /// Set the state of a newly constructed `KalmanFilterXyz`.
    #[must_use]
    pub fn with_state(mut self, state: KalmanStateXyz) -> Self {
        self.state = state;
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

    /*/// Set `Q_bias`, the continuous-time spectral density of the accelerometer-bias random walk.
    /// Input is accelerometer bias instability in g per time period tau.
    pub fn set_q_bias(&mut self, bias_instability_g: f32, tau: f32) {
        let bias_instability_mps2 = bias_instability_g * Self::G_MPS2;
        self.Q_bias = bias_instability_mps2 * bias_instability_mps2 / tau;
    }*/

    /// Set the continuous-time bias random-walk Power Spectral Density (PSD - units of m²/s⁵).
    ///
    /// Datasheets typically specify bias instability in micro-g over a time constant tau, typically 100 seconds.
    pub fn set_q_bias(&mut self, q_bias: f32) {
        self.Q_bias = q_bias;
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
        self.P[Self::VV].add_diagonal_scalar_in_place(p_vel_init);
        self.P[Self::BB].add_diagonal_scalar_in_place(p_bias_init);

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
        self.state.vel_ned_mps
    }

    #[inline]
    #[must_use]
    pub fn acc_bias_mps2(&self) -> Vector3f32 {
        self.state.acc_bias_ned_mps2
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
    /// Predicts the state vectors using `delta_vel_mps` and `delta_pos_m` from the IMU integrator.
    /// ```text
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
    /// Q: Process noise covariance.
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
    /// or (more compactly):
    ///
    ///     ┌                            ┐
    ///     │  I₃     dt·I₃   -½·dt²·I₃  │
    /// F = │  0₃       I₃      -dt·I₃   │
    ///     │  0₃       0₃         I₃    │
    ///     └                            ┘
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
    ///     ┌                               ┐
    ///     │ ⅓·Q_v·dt³  ½·Q_v·dt²       0  │
    /// Q = │ ½·Q_v·dt²    Q_v·dt        0  │
    ///     │         0          0  Q_b·dt  │
    ///     └                               ┘
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
    pub fn correct_altitude(&mut self, altitude_m: f32, R: f32) -> KalmanFilterDiagnostics {
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

        if S <= f32::EPSILON {
            return KalmanFilterDiagnostics::default();
        }

        let S_inv = 1.0 / S;

        // P * Hᵀ
        //
        // H selects the Z position column (ie column 2), so these are:
        //
        // P_PP[:,z]
        // P_VP[:,z]
        // P_BP[:,z]

        // Calculate the Kalman Gain (K) as 3D vectors
        // H isolates row index 2, so P * Hᵀ extracts column 2 of the sub-blocks.
        let P_pos = self.P[Self::PP].column(2);
        let P_vel = self.P[Self::VP].column(2);
        let P_acc_bias = self.P[Self::BP].column(2);

        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // The innovation(y) is a scalar.
        let innovation_m = altitude_m - self.state.pos_ned_m.z;

        // NIS = yᵀ * S⁻¹ * y.
        let nis = (innovation_m * innovation_m) * S_inv;
        let diagnostics = KalmanFilterDiagnostics { position_trace: self.P[Self::PP].trace(), nis };
        if diagnostics.nis > KalmanFilterDiagnostics::NIS_THRESHOLD_1D
            && self.altitude_consecutive_rejection_count < KalmanFilterDiagnostics::MAX_CONSECUTIVE_REJECTION_COUNT
        {
            // If nis > NIS_THRESHOLD_1D, there is less than a 5% chance this is normal sensor noise,
            // so reject this measurement as an outlier.
            self.altitude_consecutive_rejection_count += 1;
            return diagnostics;
        }
        self.altitude_consecutive_rejection_count = 0;

        // Update state using vector scaling
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
        measurement: Vector3f32,
        predicted_measurement: Vector3f32,
        R: Vector3f32,
    ) -> KalmanFilterDiagnostics {
        // H = [ I₃   0₃   0₃ ]
        // S = H·P·Hᵀ + R
        // K = P·Hᵀ·S⁻¹
        // y = z - H·x
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
        let innovation_m = measurement - predicted_measurement;

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

        // Update the physical state vectors
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
    /// `P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ)`.
    ///
    /// While computationally more expensive, it guarantees the result remains positive-definite.
    /// That is it ensures the covariance matrix has positive eigenvalues and remains valid and invertible for future updates.
    pub fn correct_position_joseph_stabilized_form(
        &mut self,
        measurement_m: Vector3f32,
        predicted_measurement_m: Vector3f32,
        R: Vector3f32,
    ) -> KalmanFilterDiagnostics {
        // H = [ I₃   0₃   0₃ ]
        // S = H·P·Hᵀ + R
        // K = P·Hᵀ·S⁻¹
        // y = z - H·x
        // x = x + K·y
        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ); equivalent to P - K·H·P

        // Calculate S, the Innovation Covariance matrix:
        // S = H·P·Hᵀ + R
        let S = self.P[Self::PP].add_diagonal_vector(R);

        // S is guaranteed to be positive-definite, since we are using Joseph's Stabilized Form, so we can use a Cholesky solver.
        // 14 arithmetic operations, matrix inverse takes 39 multiplications, 19 additions/subtractions and 1 division (1 div ~= 4 mul)
        let Some(S_cholesky) = S.cholesky() else {
            return KalmanFilterDiagnostics::default();
        };

        // Calculate K, the Kalman gain.
        // `K = P·Hᵀ·S⁻¹`
        let K_pos = S_cholesky.solve_matrix(self.P[Self::PP]); // 60 arithmetic operations, matrix multiply is 45
        let K_vel = S_cholesky.solve_matrix(self.P[Self::VP]);
        let K_acc_bias = S_cholesky.solve_matrix(self.P[Self::BP]);

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

        // Update the physical state vectors
        // x += K ·y
        self.state.pos_ned_m += K_pos * innovation_m;
        self.state.vel_ned_mps += K_vel * innovation_m;
        self.state.acc_bias_ned_mps2 += K_acc_bias * innovation_m;

        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ

        // Calculate the columns of intermediate matrix A = (I - KH)·P
        let A = Matrix3x3xM3x3::from_column_array([
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
        let I_minus_K_pos_t = Matrix3x3f32::identity() - K_pos_t; // Simplified distribution

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

    pub fn correct_position_xy_joseph_stabilized_form(
        &mut self,
        position_m: Vector2f32,
        past_pos_m: Vector3f32,
        R: Vector2f32,
    ) -> KalmanFilterDiagnostics {
        // TODO: implement correct_position_xy_delayed.
        _ = self;
        _ = position_m;
        _ = R;
        _ = past_pos_m;
        KalmanFilterDiagnostics::default()
    }

    /// Non-linear correction using anchor distance.
    pub fn correct_position_using_anchor_distance(
        &mut self,
        anchor_position_m: Vector3f32,
        measured_distance_m: f32,
        mahalanobis_squared_threshold: f32,
        R: f32,
    ) -> KalmanFilterDiagnostics {
        const MIN_DISTANCE_M: f32 = 1.0e-4;

        if measured_distance_m < 0.0 {
            return KalmanFilterDiagnostics::default();
        }
        // Position relative to the anchor.
        let position_m = self.state.pos_ned_m - anchor_position_m;

        // Predicted range.
        let predicted_distance_m = position_m.norm();

        // The range Jacobian is undefined at zero distance.
        if predicted_distance_m <= MIN_DISTANCE_M {
            return KalmanFilterDiagnostics::default();
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
            return KalmanFilterDiagnostics::default();
        }
        let S_inv = 1.0 / S;

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measured_distance_m - predicted_distance_m;

        // Normalized innovation squared (Mahalanobis distance squared).
        let mahalanobis_squared = innovation_m * innovation_m * S_inv;

        if mahalanobis_squared > mahalanobis_squared_threshold {
            // Reject this measurement as an outlier.
            return KalmanFilterDiagnostics::default();
        }

        // Calculate K, the Kalman gain.
        // K = P·Hᵀ·S⁻¹
        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // Update the physical state vectors
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

        KalmanFilterDiagnostics::default()
    }

    pub fn correct_position(&mut self, position_m: Vector3f32, R: Vector3f32) -> KalmanFilterDiagnostics {
        self.correct_position_joseph_stabilized_form(position_m, self.state.pos_ned_m, R)
    }

    pub fn correct_position_xy(&mut self, position_m: Vector2f32, R: Vector2f32) -> KalmanFilterDiagnostics {
        self.correct_position_xy_joseph_stabilized_form(position_m, self.state.pos_ned_m, R)
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
    pub fn validate_measurement(&self, y: Vector3f32, R: Vector3f32, gate_threshold: f32) -> bool {
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

        let Some(S_inv) = S.try_inverse() else {
            return false;
        };

        // Calculate the vector solution product: x_sol = S⁻¹·y
        let x_sol = S_inv * y;

        // Complete the final quadratic form: d² = y · x_sol
        let mahalanobis_distance_sq = y.dot(x_sol);

        // Returns true if the measurement innovation vector fits inside standard tolerances
        mahalanobis_distance_sq <= gate_threshold
    }
}

#[allow(unused)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStateXyz {
    // 3D Kinematic State Vectors
    /// Position {x, y, z}, meters, in the NED frame..
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

#[cfg(test)]
mod test_traits {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterXyz>();
        is_full::<KalmanStateXyz>();
    }
}
