#![cfg(feature = "oooc")]
use num_traits::ConstZero;
use vqm::Vector2f32;

use super::KalmanStateXy;

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapshotXy {
    pub timestamp_us: u64,
    pub delta_vel_ned_mps: Vector2f32,
    pub delta_pos_ned_m: Vector2f32,
    pub state: KalmanStateXy,
    pub gps_pos_m: Option<Vector2f32>,
    pub baro_alt_m: Option<f32>,
}

impl Default for SnapshotXy {
    fn default() -> Self {
        Self::new()
    }
}

impl SnapshotXy {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            timestamp_us: 0,
            delta_vel_ned_mps: Vector2f32::ZERO,
            delta_pos_ned_m: Vector2f32::ZERO,
            state: KalmanStateXy::new(),
            gps_pos_m: None,
            baro_alt_m: None,
        }
    }
}
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXyHistory {
    /// Fixed circular window (e.g., handles up to 640ms of latency at 100Hz).
    pub snapshot: [SnapshotXy; Self::SNAPSHOT_SIZE],
    /// Ring buffer index.
    pub head_idx: usize,
}

impl Default for KalmanFilterXyHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterXyHistory {
    pub const SNAPSHOT_SIZE: usize = 128; // 256ms of history at 500Hz

    #[must_use]
    pub const fn new() -> Self {
        Self { snapshot: [SnapshotXy::new(); Self::SNAPSHOT_SIZE], head_idx: 0 }
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterXyHistory>();
        is_full::<SnapshotXy>();
    }
}
