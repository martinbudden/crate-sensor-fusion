use vqm::Matrix3x3f32;

use super::KalmanFilterDiagnostics;

/// `f32` variant of `KalmanFilterZ`.
pub type KalmanFilterZf32 = KalmanFilterZ;

/// `f32` variant of `KalmanStateXyz`.
pub type KalmanStateZf32 = KalmanStateZ;

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterZ {
    /// 3D Kinematic State Vectors, in the NED frame.
    state: KalmanStateZ,

    // Predicted System Uncertainty Covariance Matrix (P).
    /// `P`: Prediction error covariance (aka state covariance matrix, the system's internal uncertainty).
    P: Matrix3x3f32,

    // state transition noise covariance Matrix `Q`
    /// Process Noise spectral density mapping to Velocity variance.
    Q_velocity: f32,
    /// Process Noise spectral density mapping to Sensor Drift variance.
    /// Units: (m/s³)² / Hz.
    Q_bias: f32,
    /// Count to avoid filter latching.
    consecutive_rejection_count: u32,
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

    /// Set the state covariance matrix of a newly constructed `KalmanFilterZ`.
    #[must_use]
    pub fn with_p(mut self, p: Matrix3x3f32) -> Self {
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

    pub fn set_position(&mut self, pos_ned_m: f32) {
        self.state.pos_ned_m = pos_ned_m;
    }

    pub fn set_initial_position(&mut self, initial_position_m: f32) {
        self.state.pos_ned_m = initial_position_m;

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
        self.state.pos_ned_m
    }

    #[inline]
    #[must_use]
    pub fn vel_mps(&self) -> f32 {
        self.state.vel_ned_mps
    }

    #[inline]
    #[must_use]
    pub fn acc_bias_mps2(&self) -> f32 {
        self.state.acc_bias_ned_mps2
    }

    #[inline]
    #[must_use]
    pub fn p(&self) -> &Matrix3x3f32 {
        &self.P
    }
}

// **** Predict ****

impl KalmanFilterZ {
    /// Predicts the state vectors using `delta_vel_mps` and `delta_pos_m` from the IMU integrator.
    /// ```text
    ///     ┌          ┐
    ///     │  ½·dt    │
    /// B = │    1     │
    ///     │    0     │
    ///     └          ┘.
    /// ```
    /// Assumptions:
    /// - `delta_pos_ned_m` is the integrated displacement contribution from acceleration in NED,
    ///   including gravity compensation and correction using the estimated accelerometer bias.
    /// - `delta_vel_ned_mps` is the corresponding integrated velocity increment, with the same corrections.
    /// - Both increments are expressed in NED.
    /// - `dt` equals the total integration duration represented by these increments.
    pub fn predict_state(&mut self, delta_vel_ned_mps: f32, delta_pos_ned_m: f32, dt: f32) {
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
    ///     │  1       dt       -½·dt²   │
    /// F = │  0        1        -dt     │
    ///     │  0        0          1     │
    ///     └                            ┘
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
        P[PP] += dt * (P[PV] + P[VP] + dt * (P[VV] - 0.5 * (P[PB] + P[BP] + dt * (P[VB] + P[BV] - dt * 0.5 * P[BB]))));
        // Prevent P_new[PP] being accidentally used in subsequent calculations by retiring PP.
        retire!(PP);

        P[VP] += dt * (P[VV] - P[BP] - dt * (P[BV] + 0.5 * (P[VB] - dt * P[BB])));
        retire!(VP);

        P[BP] += dt * (P[BV] - dt * 0.5 * P[BB]);
        retire!(BP);

        // --- VELOCITY COLUMNS (PV, VV, BV) ---
        P[PV] += dt * (P[VV] - P[PB] - dt * (P[VB] + 0.5 * (P[BV] - dt * P[BB])));
        retire!(PV);

        P[VV] -= dt * (P[BV] + P[VB] + dt * P[BB]);
        retire!(VV);

        P[BV] -= dt * P[BB];
        retire!(BV);

        // --- BIAS COLUMNS (PB, VB, BB) ---
        P[PB] += dt * (P[VB] - dt * 0.5 * P[BB]);
        retire!(PB);

        P[VB] -= dt * P[BB];
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

        P[Self::PP] += q_pos;
        P[Self::PV] += q_cross;
        P[Self::VP] += q_cross;
        P[Self::VV] += q_vel; // Random walk bias drift integration
        // Bias random-walk noise is added only to the bias covariance (BB).
        // Its indirect contribution to position, velocity, and their cross-covariances during this interval is neglected as an approximation,
        // since these are second order in dt.
        // This is reasonable when dt is small and the bias random-walk noise is sufficiently low.
        P[Self::BB] += q_bias;

        // Time propagation is highly sensitive to asymmetric shearing, so enforce symmetry on the covariance matrix.
        P.enforce_symmetry();
    }
}

// **** Correct ***

#[allow(non_snake_case)]
impl KalmanFilterZ {
    pub fn correct_altitude(&mut self, altitude_m: f32, R: f32) -> KalmanFilterDiagnostics {
        self.correct_position(altitude_m, self.state.pos_ned_m, R)
    }

    pub fn correct_position(
        &mut self,
        measurement_m: f32,
        predicted_measurement_m: f32,
        R: f32,
    ) -> KalmanFilterDiagnostics {
        // H = [ 1   0   0 ]
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
        let S = P_pos + R;
        if S.abs() <= f32::EPSILON {
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
        measurement_m: f32,
        predicted_measurement_m: f32,
        R: f32,
    ) -> KalmanFilterDiagnostics {
        // H = [ I   0   0 ]
        // S = H·P·Hᵀ + R
        // y = z - H·x
        // K = P·Hᵀ·S⁻¹
        // x = x + K·y
        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ); equivalent to P - K·H·P

        // Extract the submatrices representing H·P.
        let P_pos = self.P[Self::PP];
        let P_vel = self.P[Self::PV];
        let P_acc_bias = self.P[Self::PB];

        // Calculate S, the Innovation Covariance matrix:
        // S = H·P·Hᵀ + R
        let S = P_pos + R;
        if S.abs() <= f32::EPSILON {
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

        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ

        // Calculate the columns of intermediate matrix A = (I - KH)·P
        let A = Matrix3x3f32::from_column_array([
            // Column 0, position column submatrices.
            self.P[Self::PP] - K_pos * P_pos,
            self.P[Self::VP] - K_vel * P_pos,
            self.P[Self::BP] - K_acc_bias * P_pos,
            // Column 1, velocity column submatrices.
            self.P[Self::PV] - K_pos * P_vel,
            self.P[Self::VV] - K_vel * P_vel,
            self.P[Self::BV] - K_acc_bias * P_vel,
            // Column 2, bias column submatrices.
            self.P[Self::PB] - K_pos * P_acc_bias,
            self.P[Self::VB] - K_vel * P_acc_bias,
            self.P[Self::BB] - K_acc_bias * P_acc_bias,
        ]);

        // Precompute and helper term.
        let I_minus_K_pos = 1.0 - K_pos;
        // Calculate J = A·(I - KH)ᵀ.
        // Only calculate the values that are different from A.
        let J = [
            A[Self::PP] * I_minus_K_pos - A[Self::PV] * K_vel - A[Self::PB] * K_acc_bias,
            A[Self::VP] * I_minus_K_pos - A[Self::VV] * K_vel - A[Self::VB] * K_acc_bias,
            A[Self::BP] * I_minus_K_pos - A[Self::BV] * K_vel - A[Self::BB] * K_acc_bias,
        ];

        // Calculate K·R·Kᵀ blocks
        let K_pos_R = K_pos * R;
        let K_vel_R = K_vel * R;
        let K_acc_bias_R = K_acc_bias * R;

        // Reassemble J and A into P.
        self.P[Self::PP] = J[Self::PP] + K_pos_R * K_pos;
        self.P[Self::VP] = J[Self::VP] + K_vel_R * K_pos;
        self.P[Self::BP] = J[Self::BP] + K_acc_bias_R * K_pos;

        self.P[Self::PV] = A[Self::PV] + K_pos_R * K_vel;
        self.P[Self::VV] = A[Self::VV] + K_vel_R * K_vel;
        self.P[Self::BV] = A[Self::BV] + K_acc_bias_R * K_vel;

        self.P[Self::PB] = A[Self::PB] + K_pos_R * K_acc_bias;
        self.P[Self::VB] = A[Self::VB] + K_vel_R * K_acc_bias;
        self.P[Self::BB] = A[Self::BB] + K_acc_bias_R * K_acc_bias;

        // Ensure numerical stability by enforcing symmetry on the covariance matrix.
        self.P.enforce_symmetry();

        diagnostics
    }
}
#[allow(unused)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanStateZ {
    // 3D Kinematic State Vectors
    /// Position, meters, in the NED frame.
    pub pos_ned_m: f32,
    /// Velocity, meters per second, in the NED frame.
    pub vel_ned_mps: f32,
    /// Accelerometer Bias, meters per second squared, in the NED frame.
    pub acc_bias_ned_mps2: f32,
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
        Self { pos_ned_m: 0.0, vel_ned_mps: 0.0, acc_bias_ned_mps2: 0.0 }
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
