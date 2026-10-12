/// `KalmanFilterDiagnostics`.
///
/// |Symptom                           | Cause                                                             | Mitigation|
/// |NIS stays very high (>10.0)       | The filter's state estimate is far away from what the sensor sees | `Q_velocity` might be set too low (the filter is ignoring reality), or your sensor has a permanent structural calibration shift|
/// |NIS stays very close to 0 (<0.05) | The filter is tracking the sensor too perfectly | The R noise matrix parameter is set too high. You are telling the filter the sensor is noisy when it is actually clean|
/// |Position Trace drops to zero      | The covariance matrix has collapsed, the filter thinks it is 100% perfect | Increase your process noise `Q_velocity` to force uncertainty back into the matrix over time|.
///
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterDiagnostics {
    /// Normalized Innovation Squared (NIS).
    /// If this sits consistently above your threshold, the filter is diverging.
    pub nis: f32,
    /// Total positional uncertainty (sum of X, Y, Z variances).
    /// If this goes to 0, the filter is overconfident. If it goes to infinity, it's unstable.
    pub position_trace: f32,
}

impl Default for KalmanFilterDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterDiagnostics {
    // For a 1D measurement, a Chi-Squared 95% confidence threshold is 3.84.
    // If NIS exceeds this consistently, the sensor and physics model disagree.
    pub const NIS_THRESHOLD_1D: f32 = 3.84;
    pub const NIS_THRESHOLD_3D: f32 = 7.81;
    pub const MAX_CONSECUTIVE_REJECTION_COUNT: u32 = 20;

    #[must_use]
    pub fn new() -> Self {
        Self { nis: 0.0, position_trace: 0.0 }
    }
}
