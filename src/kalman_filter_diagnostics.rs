use vqm::Vector3f32;

/// Status of estimator, for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum EstimatorHealth {
    #[default]
    Initializing,
    Healthy,
    Degraded,
    Recovering,
    Failed,
}

/// Status of the Kalman filter, for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum FusionStatus {
    #[default]
    Accepted,
    RejectedOutlier,
    SensorUnavailable,
    InvalidMeasurement,
    MeasurementTooSmall,
    NumericalFailure,
    IllConditionedInnovationCovarianceMatrix,
    SnapshotNotFound,
}

/// `KalmanFilterDiagnostics`.
///
/// |Symptom                           | Cause                                                             | Mitigation|
/// |NIS stays very high (>10.0)       | The filter's state estimate is far away from what the sensor sees | `Q_velocity` might be set too low (the filter is ignoring reality), or your sensor has a permanent structural calibration shift|
/// |NIS stays very close to 0 (<0.05) | The filter is tracking the sensor too perfectly | The R noise matrix parameter is set too high. You are telling the filter the sensor is noisy when it is actually clean|
/// |Position Trace drops to zero      | The covariance matrix has collapsed, the filter thinks it is 100% perfect | Increase your process noise `Q_velocity` to force uncertainty back into the matrix over time|.
///
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterDiagnostics {
    pub status: FusionStatus,
    /// Normalized Innovation Squared (NIS).
    /// If this sits consistently above your threshold, the filter is diverging.
    pub nis: f32,
    /// Total positional uncertainty (sum of X, Y, Z variances).
    /// If this goes to 0, the filter is overconfident. If it goes to infinity, it's unstable.
    pub covariance_trace: f32,
    /// Reciprocal condition of the covariance matrix, 1.0 represents perfectly conditions, 0.0 means singular. Ideally >= 0.01.
    pub covariance_reciprocal_condition: Option<f32>,
    pub innovation_m: Vector3f32,
}

impl Default for KalmanFilterDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterDiagnostics {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            status: FusionStatus::Accepted,
            nis: 0.0,
            covariance_trace: 0.0,
            covariance_reciprocal_condition: None,
            innovation_m: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 },
        }
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<EstimatorHealth>();
        is_full::<FusionStatus>();
        is_full::<KalmanFilterDiagnostics>();
    }
}
