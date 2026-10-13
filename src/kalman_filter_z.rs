use vqm::Matrix3x3f32;

use crate::KalmanFilterDiagnostics;

pub type KalmanFilterZf32 = KalmanFilterZ;

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterZ {
    /// 3D Kinematic State Vectors.
    pub state: KalmanStateZ,

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
    #[allow(non_snake_case)]
    #[must_use]
    pub fn new() -> Self {
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

    /// Set `Q_velocity`, the spectral density of the acceleration noise.
    /// Input is accelerometer noise density in g per root-Hertz.
    /// Datasheets typically specify accelerometer noise density in micro-g per root-Hertz  or mg per root-Hertz.
    pub fn set_q_velocity(&mut self, acc_noise_density_g_per_root_hz: f32, denominator: u32) {
        const G: f32 = 9.80665;
        let acc_noise_density_mps2_per_root_hz = acc_noise_density_g_per_root_hz * G;
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
        const G: f32 = 9.80665;
        let bias_instability_mps2 = bias_instability_g * G;
        self.Q_velocity = bias_instability_mps2 * bias_instability_mps2 / tau;
    }

    pub fn set_initial_position(&mut self, initial_position: f32) {
        self.state.pos = initial_position;

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
    pub fn pos(&self) -> f32 {
        self.P[Self::PP]
    }

    #[inline]
    #[must_use]
    pub fn vel(&self) -> f32 {
        self.P[Self::VV]
    }

    #[inline]
    #[must_use]
    pub fn acc_bias(&self) -> f32 {
        self.P[Self::BB]
    }
}

// **** Predict ****

impl KalmanFilterZ {
    // Propagates the state vector forward using IMU acceleration inputs.
    /// Integrates raw IMU accelerometer data to predict new position and velocity vectors.
    ///
    /// This forms the continuous dead reckoning pipeline driving vehicle kinetics forward.
    ///
    /// ### Physical Mechanics
    /// ```math
    ///
    /// pos_k = pos_k₋₁ + vel_k₋₁ * dt + 0.5 * acc * dt²
    /// vel_k = vel_k₋₁ + acc * dt
    /// ```
    pub fn predict_state(&mut self, acc_measurement: f32, dt: f32) {
        // In NED, positive Z is down, so gravity is a positive vector
        let gravity = 9.80665;

        // Calculate true physical acceleration by removing bias and adding gravity
        // Note: in a NED frame, gravity is a positive vector pointing downwards, so it is added to the accelerometer measurement.
        let acc = acc_measurement - self.state.acc_bias + gravity;

        // Physical mechanics
        // s = ut + 0.5 * a * t²
        self.state.pos += (self.state.vel + 0.5 * acc * dt) * dt;

        // v = u + a * t
        self.state.vel += acc * dt;

        // Bias remains constant during prediction, it is modeled as a random walk in covariance.
        // So `self.acc_bias` is not updated.
    }

    /// Predict covariance:
    ///
    /// `P = F * P * Fᵀ + Q`.
    ///
    /// `P`: State covariance matrix
    /// `F`: State transition matrix
    /// `Q`: Process noise covariance.
    ///
    /// A full 9x9 matrix multiplication involves 729 individual arithmetic operations.
    /// Instead the matrix is divided into 9 separate 3x3 sub-matrices (blocks),
    /// and each block is processed separately.
    ///
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
    ///     │ │    I     ││   dt*I   ││    0     │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    /// F = │ │    0     ││    I     ││  -dt*I   │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     │ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
    ///     │ │    0     ││    0     ││    I     │ │
    ///     │ └          ┘└          ┘└          ┘ │
    ///     └                                      ┘
    ///
    /// Since this is a sparse matrix, we can calculate F * P * Fᵀ without doing full 9x9 matrix multiplications.
    /// Instead we perform the multiplication block by block.
    ///
    /// Expanding F * P * Fᵀ algebraically gives the following block updates:
    ///
    /// For the position columns (PP, VP, BP)
    ///     PP_new = PP + dt * (VP + PV) + dt^2 * VV
    ///     VP_new = VP + dt * VV - dt * BP - dt^2 * BV
    ///     BP_new = BP + dt * BV
    /// For the velocity columns (PV, VV, BV)
    ///     PV_new = PV + dt * VV - dt * PB - dt^2 * VB
    ///     VV_new = VV - dt * (BV + VB) + dt^2 * BB
    ///     BV_new = BV - dt * BB
    /// For the bias columns (PB, VB, BB)
    ///     PB_new = PB + dt * VB
    ///     VB_new = VB - dt * BB
    ///     BB_new = BB
    /// ```
    #[allow(unused)]
    #[allow(non_snake_case)]
    pub fn predict_covariance(&mut self, dt: f32) {
        // =====================================================================
        // PROPAGATE THE COVARIANCE P_new = (F * P_old * Fᵀ)
        // =====================================================================

        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt2 * dt2;

        // Avoid taking a copy of self.P by using temporary indices and shadowing them out of scope when they have been used.
        // This ensures we cannot use an updated value in a subsequent update.
        let (PP, VP, BP, PV, VV, BV, PB, VB, BB) =
            (Self::PP, Self::VP, Self::BP, Self::PV, Self::VV, Self::BV, Self::PB, Self::VB, Self::BB);

        // --- POSITION COLUMNS (PP, VP, BP) ---
        self.P[PP] = (self.P[VP] + self.P[PV]) * dt + self.P[VV] * dt2;
        let PP = (); // Stop P_new[PP] being used in subsequent calculations.
        self.P[VP] = self.P[VP] + (self.P[VV] - self.P[BP]) * dt - self.P[BV] * dt2;
        let VP = (); // Stop P_new[VP] being used in subsequent calculations.
        self.P[BP] += self.P[BV] * dt;
        let BP = (); // Stop P_new[BP] being used in subsequent calculations.

        // --- VELOCITY COLUMNS (PV, VV, BV) ---
        self.P[PV] = self.P[PV] + (self.P[VV] - self.P[PB]) * dt - self.P[VB] * dt2;
        let PV = ();
        self.P[VV] = self.P[VV] - (self.P[BV] + self.P[VB]) * dt + self.P[BB] * dt2;
        let VV = ();
        self.P[BV] -= self.P[BB] * dt;
        let BV = ();

        // --- BIAS COLUMNS (PB, VB, BB) ---
        self.P[PB] += self.P[VB] * dt;
        let PB = ();
        self.P[VB] -= self.P[BB] * dt;
        // self.P[Self::BB] = self.P[Self::BB]; don't need to update BB.

        // =====================================================================
        // APPLY PROCESS NOISE (Q)
        // =====================================================================

        // Continuous process noise integrated over dt maps to the diagonal variance slots of Velocity and Bias.
        // P += Q
        self.P[Self::VV] += self.Q_velocity * dt;
        self.P[Self::BB] += self.Q_bias * dt;

        // Time propagation is highly sensitive to asymmetric shearing, so enforce symmetry on the covariance matrix.
        self.P.enforce_symmetry();
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
    pub fn correct_altitude(&mut self, altitude_measurement: f32, R: f32) -> KalmanFilterDiagnostics {
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
        let innovation = altitude_measurement - self.state.pos;

        // NIS = yᵀ * S⁻¹ * y.
        let nis = (innovation * innovation) * S_inv;
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
        self.state.pos += K_pos * innovation;
        self.state.vel += K_vel * innovation;
        self.state.acc_bias += K_acc_bias * innovation;

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
        measurement: f32,
        predicted_measurement: f32,
        R: f32,
    ) -> KalmanFilterDiagnostics {
        // S = H * P * Hᵀ + R
        // K = P * Hᵀ * S⁻¹
        // y = z - H * x
        // x += K * y
        // P -= (K * H) * P

        // Extract the submatrices representing H * P.
        let P_pos = self.P[Self::PP];
        let P_vel = self.P[Self::PV];
        let P_acc_bias = self.P[Self::PB];

        // Calculate S, the Residual Covariance matrix:
        // S = H * P * Hᵀ + R
        let S = P_pos + R;
        if S == 0.0 {
            return KalmanFilterDiagnostics::default();
        }
        let S_inv = 1.0 / S;

        // Calculate the innovation.
        // y = z - H * x
        let innovation = measurement - predicted_measurement;

        // NIS = yᵀ * S⁻¹ * y.
        let nis = (innovation * innovation) * S_inv;
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
        // K = P * Hᵀ * S⁻¹
        //
        // H selects the position column block stack (Column 0: PP, VP, BP)
        // H selects the position states, so P * Hᵀ is simply the first three columns of P,
        // represented by the first three Matrix3x3 blocks.
        let K_pos = P_pos * S_inv;
        let K_vel = P_vel * S_inv;
        let K_acc_bias = P_acc_bias * S_inv;

        // Update the physical state vectors
        // x += K * y
        self.state.pos += K_pos * innovation;
        self.state.vel += K_vel * innovation;
        self.state.acc_bias += K_acc_bias * innovation;

        // P -= K * (H * P)
        // Calculate by subtracting the K * (H * P) submatrices one by one.

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
    /// Position.
    pub pos: f32,
    /// Velocity.
    pub vel: f32,
    /// Accelerometer Bias.
    pub acc_bias: f32,
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
        Self { pos: 0.0, vel: 0.0, acc_bias: 0.0 }
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
