use vqm::Matrix3x3f32;

use super::KalmanFilterDiagnostics;

/// `f32` variant of `KalmanFilterZ`.
pub type KalmanFilterZf32 = KalmanFilterZ;

/// `f32` variant of `KalmanStateXyz`.
pub type KalmanStateZf32 = KalmanStateZ;

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterZ {
    /// 3D Kinematic State Vectors.
    state: KalmanStateZ,

    // Predicted System Uncertainty Covariance Matrix (P).
    /// `P`: Prediction error covariance (aka state covariance matrix, the system's internal uncertainty).
    pub P: Matrix3x3f32,

    // state transition noise covariance Matrix `Q`
    /// Process Noise spectral density mapping to Velocity variance.
    pub Q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    /// Units: (m/s³)² / Hz.
    pub Q_bias: f32,
    /// Count to avoid filter latching.
    pub consecutive_rejection_count: u32,
}

impl Default for KalmanFilterZ {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(missing_docs)]
impl KalmanFilterZ {
    const G_MPS2: f32 = 9.80665;

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

impl KalmanFilterZ {
    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        #[allow(non_snake_case)]
        let mut P = Matrix3x3f32::default();

        // Seed initial Position uncertainty (ie, confident within 1 meter)
        P[Self::PP] = 1.0;

        // Seed initial Velocity uncertainty (ie, confident within 0.5 m/s)
        P[Self::VV] = 0.25;

        // Seed initial Bias uncertainty (ie, accelerometer bias)
        P[Self::BB] = 0.01;

        Self {
            state: KalmanStateZ::new(),
            // A value of 0.05 implies that every second, we expect aerodynamic buffeting, vibration, or wind to naturally perturb the velocity
            // by roughly 0.22 m/s ie sqrt(0.05).
            Q_velocity: 0.05,
            // Sensor bias shifts very slowly due to thermal changes as the silicon heats up.
            // So this value should be tiny so the filter treats bias as a near-constant,
            // shifting it incrementally over minutes rather than fluctuating on every single vibration loop.
            Q_bias: 1e-4,
            P,
            consecutive_rejection_count: 0,
        }
    }

    /// Set the state of a newly constructed `KalmanFilterZ`.
    #[must_use]
    pub fn with_state(mut self, state: KalmanStateZ) -> Self {
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

    pub fn set_position(&mut self, pos_ned_m: f32) {
        self.state.pos_m = pos_ned_m;
    }

    pub fn set_initial_position(&mut self, initial_position_m: f32) {
        self.state.pos_m = initial_position_m;

        // --- Define Initial Uncertainties (Variances) ---
        let p_pos_init = 25.0; // ±5 meters initial position uncertainty, typical of GPS
        let p_vel_init = 0.25; // ±0.5 m/s initial velocity uncertainty
        let p_bias_init = 0.04; // ±0.2 m/s² initial accelerometer bias uncertainty

        // --- Populate the Diagonal Core Blocks ---
        // Setting these ensures the X, Y, and Z axes are uniformly weighted at startup.
        self.P[Self::PP] += p_pos_init;
        self.P[Self::VV] += p_vel_init;
        self.P[Self::BB] += p_bias_init;

        // Off-diagonal blocks (PV, PB, VP, VB, BP, BV) remain pure zero matrices, as no cross-correlations exist yet.
    }
}

impl KalmanFilterZ {
    #[inline]
    #[must_use]
    pub fn state(&self) -> KalmanStateZ {
        self.state
    }

    #[inline]
    #[must_use]
    pub fn pos_m(&self) -> f32 {
        self.P[Self::PP]
    }

    #[inline]
    #[must_use]
    pub fn vel_mps(&self) -> f32 {
        self.P[Self::VV]
    }

    #[inline]
    #[must_use]
    pub fn acc_bias_mps2(&self) -> f32 {
        self.P[Self::BB]
    }
}

// **** Predict ****

impl KalmanFilterZ {
    /// Propagates the state vector forward using `delta_vel` and `delta_pos` from the IMU integrator.
    /// ```text
    ///     ┌          ┐
    ///     │  ½·dt·I  │
    /// B = │    I     │
    ///     │    0     │
    ///     └          ┘.
    /// ```
    pub fn predict_state(&mut self, delta_vel_mps: f32, delta_pos_m: f32, dt: f32) {
        // Gravity and acc bias were not included in the IMU integration, so we need to correct for them here.

        // Correct the integrated position for acc bias and add the gravity-induced displacement.
        self.state.pos_m +=
            delta_pos_m + dt * (self.state.vel_mps + 0.5 * (Self::G_MPS2 - self.state.acc_bias_mps2) * dt);

        // Correct the integrated velocity for acc bias and add gravity.
        self.state.vel_mps += delta_vel_mps + (Self::G_MPS2 - self.state.acc_bias_mps2) * dt;

        // Bias remains constant during prediction: its uncertainty is modelled as a random walk in the covariance.
        // So we don't update `self.state.acc_bias` here.
    }

    /// Predict covariance:
    /// ```text
    /// P = F·P·Fᵀ·Q
    ///
    /// P: State covariance matrix
    /// F: State transition matrix
    /// Q: Process noise covariance
    ///
    ///     ┌                         ┐
    ///     │  I     dt·I   -½·dt²·I  │
    /// F = │  0       I      -dt·I   │
    ///     │  0       0         I    │
    ///     └                         ┘
    /// ```
    #[allow(non_snake_case)]
    pub fn predict_covariance(&mut self, dt: f32) {
        // =====================================================================
        // PROPAGATE THE COVARIANCE P_new = (F·P_old·Fᵀ)
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
        P[PP] = (P[VP] + P[PV]) * dt + P[VV] * dt2;
        retire!(PP);

        P[VP] = P[VP] + (P[VV] - P[BP]) * dt - P[BV] * dt2;
        retire!(VP);

        P[BP] += P[BV] * dt;
        retire!(BP);

        // --- VELOCITY COLUMNS (PV, VV, BV) ---
        P[PV] = P[PV] + (P[VV] - P[PB]) * dt - P[VB] * dt2;
        retire!(PV);

        P[VV] = P[VV] - (P[BV] + P[VB]) * dt + P[BB] * dt2;
        retire!(VV);

        P[BV] -= P[BB] * dt;
        retire!(BV);

        // --- BIAS COLUMNS (PB, VB, BB) ---
        P[PB] += P[VB] * dt;
        retire!(PB);

        P[VB] -= P[BB] * dt;
        retire!(BP);
        // P[Self::BB] = P[Self::BB]; don't need to update BB.

        // =====================================================================
        // APPLY PROCESS NOISE (Q)
        // =====================================================================

        // Continuous process noise integrated over dt maps to the diagonal variance slots of Velocity and Bias.
        // P += Q
        P[Self::VV] += self.Q_velocity * dt;
        P[Self::BB] += self.Q_bias * dt;

        // Time propagation is highly sensitive to asymmetric shearing, so enforce symmetry on the covariance matrix.
        P.enforce_symmetry();
    }
}

// **** Correct ***

#[allow(non_snake_case)]
impl KalmanFilterZ {
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
        let S = self.P[Self::PP] + R;

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

        // Calculate the Kalman Gain (K)
        let P_pos = self.P[Self::PP];
        let P_vel = self.P[Self::VP];
        let P_acc_bias = self.P[Self::BP];

        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // The innovation(y) is a scalar.
        let innovation_m = altitude_m - self.state.pos_m;

        // NIS = yᵀ * S⁻¹ * y.
        let nis = (innovation_m * innovation_m) * S_inv;
        let diagnostics = KalmanFilterDiagnostics { position_trace: self.P[Self::PP], nis };
        if diagnostics.nis > KalmanFilterDiagnostics::NIS_THRESHOLD_1D
            && self.consecutive_rejection_count < KalmanFilterDiagnostics::MAX_CONSECUTIVE_REJECTION_COUNT
        {
            // If nis > NIS_THRESHOLD_1D, there is less than a 5% chance this is normal sensor noise,
            // so reject this measurement as an outlier.
            self.consecutive_rejection_count += 1;
            return diagnostics;
        }
        self.consecutive_rejection_count = 0;

        // Update state using vector scaling
        self.state.pos_m += K_pos * innovation_m;
        self.state.vel_mps += K_vel * innovation_m;
        self.state.acc_bias_mps2 += K_acc_bias * innovation_m;

        // Covariance update
        //
        // P -= (P Hᵀ)(H P) / S
        //
        self.P[Self::PP] -= K_pos * P_pos;
        self.P[Self::PV] -= K_pos * P_vel;
        self.P[Self::PB] -= K_pos * P_acc_bias;

        self.P[Self::VP] -= K_vel * P_pos;
        self.P[Self::VV] -= K_vel * P_vel;
        self.P[Self::VB] -= K_vel * P_acc_bias;

        self.P[Self::BP] -= K_acc_bias * P_pos;
        self.P[Self::BV] -= K_acc_bias * P_vel;
        self.P[Self::BB] -= K_acc_bias * P_acc_bias;

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();

        diagnostics
    }

    pub fn correct_position(
        &mut self,
        measurement_m: f32,
        predicted_measurement_m: f32,
        R: f32,
    ) -> KalmanFilterDiagnostics {
        // S = H·P·Hᵀ + R
        // K = P·Hᵀ·S⁻¹
        // y = z - H·x
        // x = x + K·y
        // P = P - K·H·P

        // Extract the submatrices representing H·P.
        let P_pos = self.P[Self::PP];
        let P_vel = self.P[Self::PV];
        let P_acc_bias = self.P[Self::PB];

        // Calculate S, the Residual Covariance matrix:
        // S = H·P·Hᵀ + R
        let S = P_pos + R;
        if S == 0.0 {
            return KalmanFilterDiagnostics::default();
        }
        let S_inv = 1.0 / S;

        // Calculate the innovation.
        // y = z - H·x
        let innovation_m = measurement_m - predicted_measurement_m;

        // Normalized Innovation Squared (NIS) = yᵀ·S⁻¹·y  (aka Mahalanobis Distance)
        let nis = (innovation_m * innovation_m) * S_inv;
        let diagnostics = KalmanFilterDiagnostics { position_trace: self.P[Self::PP], nis };
        if diagnostics.nis > KalmanFilterDiagnostics::NIS_THRESHOLD_1D
            && self.consecutive_rejection_count < KalmanFilterDiagnostics::MAX_CONSECUTIVE_REJECTION_COUNT
        {
            // If nis > NIS_THRESHOLD_1D, there is less than a 5% chance this is normal sensor noise,
            // so reject this measurement as an outlier.
            self.consecutive_rejection_count += 1;
            return diagnostics;
        }
        self.consecutive_rejection_count = 0;

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

        diagnostics
    }
}
#[allow(unused)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStateZ {
    // 3D Kinematic State Vectors
    /// Position, meters.
    pub pos_m: f32,
    /// Velocity, meters per second.
    pub vel_mps: f32,
    /// Accelerometer Bias, meters per second squared.
    pub acc_bias_mps2: f32,
}

impl Default for KalmanStateZ {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanStateZ {
    /// Constructor.
    #[must_use]
    pub const fn new() -> Self {
        Self { pos_m: 0.0, vel_mps: 0.0, acc_bias_mps2: 0.0 }
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterZ>();
        is_full::<KalmanStateZ>();
    }
}
