use num_traits::ConstZero;
use vqm::Vector2f32;

use super::{FusionStatus, KalmanFilterDiagnostics, KalmanFilterXy, KalmanStateXy};

/// `f32` variant of `PositionKalmanFilterExtended`.
pub type KalmanFilterXyWithSensorsf32 = KalmanFilterXyWithSensors;

/*
Desired Physical Noise (σ)            Equivalent R Variance Code Value (σ²)
5 cm (Ultra-accurate UWB / Laser)     0.0025
10 cm (High-end Barometer)            0.01
20 cm (Standard Barometer)            0.04
50 cm (Precision RTK GPS)             0.25
1.0 meter (Good Commercial GPS)       1.0
2.0 meters (Standard Multi-path GPS)  4.0
*/
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXyWithSensors {
    base: KalmanFilterXy,
    // measurement noise covariance Matrix `R`
    /// Absolute Measurement Noise variance for horizontal GPS position.
    pub R_gps_position_horizontal: f32,
    /// Absolute Measurement Noise variance for vertical GPS position.
    pub R_gps_position_vertical: f32,
    /// Absolute Measurement Noise variance for horizontal GPS velocity.
    pub R_gps_velocity_horizontal: f32,
    /// Absolute Measurement Noise variance for vertical GPS velocity.
    pub R_gps_velocity_vertical: f32,
    /// Absolute Measurement Noise variance for barometric pressure altimeter.
    pub R_barometer: f32,
    /// Absolute Measurement Noise variance for rangefinder.
    pub R_rangefinder: f32,
    /// Absolute Measurement Noise variance for optical flow.
    pub R_optical_flow: Vector2f32,
    /// Absolute Measurement Noise variance for ultra-wideband indoor positioning system..
    pub R_ultra_wideband: f32,
    /// Delayed State Buffer for Retrodictive Updates.
    pub history: KalmanFilterXyHistory,
}

impl Default for KalmanFilterXyWithSensors {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterXyWithSensors {
    /// For a 1D measurement, a Chi-Squared 95% confidence threshold is 3.84.
    /// If NIS (Normalized Innovation Squared aka Mahalanobis distance squared) exceeds this consistently, the sensor and physics model disagree.
    pub const NIS_THRESHOLD_1D: f32 = 3.84;
    /// For a 2D measurement, a Chi-Squared 95% confidence threshold is 5.99.
    /// If NIS (Normalized Innovation Squared aka Mahalanobis distance squared) exceeds this consistently, the sensor and physics model disagree.
    pub const NIS_THRESHOLD_2D: f32 = 5.991;

    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: KalmanFilterXy::new(),
            // Commercial GPS modules are typically accurate to within 1.5 to 3.0 meters horizontally, less vertically.
            R_gps_position_horizontal: 2.25,
            R_gps_position_vertical: 9.0,
            // GPS velocity measurements are often considerably more accurate than GPS position measurements, sometimes by an order of magnitude or more.
            // This is because the measurements are done by measuring Doppler shift.
            R_gps_velocity_horizontal: 0.05, // m/s
            R_gps_velocity_vertical: 0.08,   // m/s
            // Vertical variance in meters squared
            // A standard barometric pressure sensor (e.g., BMP280, MS5611) typically has an RMS noise of around 10 to 20 centimeters
            // which corresponds to a variance of 0.01 to 0.04 m². However, the actual noise can vary based on environmental conditions and sensor quality.
            R_barometer: 0.03,
            R_rangefinder: 0.03,
            R_optical_flow: Vector2f32 { x: 0.04, y: 0.04 },
            // 1cm to 10cm typical accuracy, use 10cm = 0.10m, = 0.01 variance for cheap modules.
            R_ultra_wideband: 0.01,
            history: KalmanFilterXyHistory::new(),
        }
    }
}

impl KalmanFilterXyWithSensors {
    #[inline]
    #[must_use]
    pub fn state(&self) -> KalmanStateXy {
        self.base.state()
    }

    #[inline]
    #[must_use]
    pub fn pos_m(&self) -> Vector2f32 {
        self.base.pos_m()
    }

    #[inline]
    #[must_use]
    pub fn vel_mps(&self) -> Vector2f32 {
        self.base.vel_mps()
    }

    #[inline]
    #[must_use]
    pub fn acc_bias_mps2(&self) -> Vector2f32 {
        self.base.acc_bias_mps2()
    }

    /// Called inside the main IMU execution loop (eg `gyro_pid` loop).
    pub fn predict(&mut self, timestamp_us: u64, delta_vel_ned_mps: Vector2f32, delta_pos_ned_m: Vector2f32, dt: f32) {
        // Integrate physical kinematics forward (deterministic)
        self.predict_state(delta_vel_ned_mps, delta_pos_ned_m, dt);

        // Propagate covariance uncertainty blocks forward (stochastic)
        self.predict_covariance(dt);

        self.push_snapshot(timestamp_us, delta_pos_ned_m, delta_vel_ned_mps);
    }

    pub fn predict_state(&mut self, delta_vel_ned_mps: Vector2f32, delta_pos_ned_m: Vector2f32, dt: f32) {
        self.base.predict_state(delta_vel_ned_mps, delta_pos_ned_m, dt);
    }

    pub fn predict_covariance(&mut self, dt: f32) {
        self.base.predict_covariance(dt);
    }

    /// Correct position using GPS position measurement (typically at a 1Hz to 10Hz rate).
    pub fn correct_position_using_gps(&mut self, timestamp_us: u64, position_m: Vector2f32) -> KalmanFilterDiagnostics {
        _ = timestamp_us;
        let r_gps_position = Vector2f32 { x: self.R_gps_position_horizontal, y: self.R_gps_position_horizontal };
        self.base.correct_position(position_m, r_gps_position, Self::NIS_THRESHOLD_2D)
    }

    /// Use the GPS velocity to predict the GPS position forward to the current time,
    /// And use that to correct the position estimate.
    pub fn correct_position_using_gps_with_delay_compensation(
        &mut self,
        time_us: u64,
        timestamp_us: u64,
        position_m: Vector2f32,
        velocity_mps: Vector2f32,
    ) -> KalmanFilterDiagnostics {
        let r_gps = Vector2f32 { x: self.R_gps_position_horizontal, y: self.R_gps_position_horizontal };
        let dt_us = time_us - timestamp_us;
        #[allow(clippy::cast_precision_loss)]
        let dt = (dt_us as f32) * 1e-6;
        // Estimate the current position and use that for the correction.
        let position = position_m + velocity_mps * dt;
        self.base.correct_position(position, r_gps, Self::NIS_THRESHOLD_2D)
    }

    /// Correct velocity using GPS velocity measurement (typically at a 1Hz to 10Hz rate).
    /// GPS velocity measurements are often considerably more accurate than GPS position measurements, sometimes by an order of magnitude or more.
    /// This is because the measurements are done by measuring Doppler shift.
    pub fn correct_position_and_velocity_using_gps(
        &mut self,
        timestamp_us: u64,
        position_m: Vector2f32,
        velocity_mps: Vector2f32,
    ) -> KalmanFilterDiagnostics {
        _ = timestamp_us;
        let r_gps_position = Vector2f32 { x: self.R_gps_position_horizontal, y: self.R_gps_position_horizontal };
        let r_gps_velocity = Vector2f32 { x: self.R_gps_velocity_horizontal, y: self.R_gps_velocity_horizontal };
        self.base.correct_position_and_velocity(
            position_m,
            r_gps_position,
            velocity_mps,
            r_gps_velocity,
            Self::NIS_THRESHOLD_2D,
        )
    }

    /// Correct position using optical flow position measurement.
    /// Placeholder function, currently unimplemented.
    pub fn correct_position_using_optical_flow(
        &mut self,
        timestamp_us: u64,
        position_m: Vector2f32,
    ) -> KalmanFilterDiagnostics {
        _ = self;
        _ = timestamp_us;
        _ = position_m;
        todo!("implement correct_position_using_optical_flow for xy");
        //self.base.correct_position(position_m, self.R_optical_flow)
    }
}

#[allow(non_snake_case)]
impl KalmanFilterXyWithSensors {
    /// Correct position using delayed measurement without rewinding `P`.
    /// When a delayed measurement arrives, we calculate the innovation error using the past state vector (`past.pos`).
    /// We calculate the Kalman Gain vectors using the current active P matrix.
    /// We correct the current state vector directly.
    pub fn correct_position_delayed(
        &mut self,
        timestamp_us: u64,
        measurement_m: Vector2f32,
        R: Vector2f32,
    ) -> KalmanFilterDiagnostics {
        // Find the past state matching the sensor timestamp
        let Some((past, _)) = self.find_snapshot(timestamp_us) else {
            return KalmanFilterDiagnostics { status: FusionStatus::SnapshotNotFound, ..Default::default() };
        };
        let predicted_measurement_m = past.state.pos_ned_m;
        self.base.correct_position_joseph_stabilized_form(
            measurement_m,
            predicted_measurement_m,
            R,
            Self::NIS_THRESHOLD_2D,
        )
    }
}

impl KalmanFilterXyWithSensors {
    /// Saves the current state of the filter into our history ring buffer.
    /// Run this at the very end of EVERY IMU time update step.
    pub fn push_snapshot(&mut self, timestamp_us: u64, delta_vel_ned_mps: Vector2f32, delta_pos_ned_m: Vector2f32) {
        let snapshot = SnapshotXy {
            timestamp_us,
            delta_vel_ned_mps,
            delta_pos_ned_m,
            state: self.base.state,
            gps_pos_m: None,
            baro_alt_m: None,
        };

        self.history.head_idx = (self.history.head_idx + 1) % self.history.snapshot.len();
        self.history.snapshot[self.history.head_idx] = snapshot;
    }

    pub fn register_gps_event(&mut self, timestamp_us: u64, gps_pos_m: Vector2f32) {
        _ = timestamp_us;
        self.history.snapshot[self.history.head_idx].gps_pos_m = Some(gps_pos_m);
    }

    /// Retrieves a historical snapshot of the filter state at a given index.
    /// The index is relative to the most recent snapshot, with 0 being the latest.
    #[must_use]
    pub fn find_snapshot(&self, sensor_time_us: u64) -> Option<(SnapshotXy, usize)> {
        let mut match_idx = None;

        for ii in 0..self.history.snapshot.len() {
            let timestamp_us = self.history.snapshot[ii].timestamp_us;
            // Skip uninitialized buffer slots at system startup
            if timestamp_us == 0 {
                continue;
            }
            if sensor_time_us > timestamp_us {
                match_idx = Some(ii);
                break;
            }
        }

        let start_idx = match_idx?;

        Some((self.history.snapshot[start_idx], start_idx))
    }
}

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
        is_full::<KalmanFilterXyWithSensors>();
        is_full::<KalmanFilterXyWithSensorsf32>();
        is_full::<KalmanFilterXyHistory>();
        is_full::<SnapshotXy>();
    }
}
