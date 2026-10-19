use num_traits::ConstZero;
use vqm::Vector2f32;

use super::{KalmanFilterDiagnostics, KalmanFilterXy, KalmanStateXy};

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
    /// Absolute Measurement Noise variance for horizontal GPS channels.
    pub R_gps_horizontal: f32,
    /// Absolute Measurement Noise variance for vertical GPS channels.
    pub R_gps_vertical: f32,
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
    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: KalmanFilterXy::new(),
            // Commercial GPS modules are typically accurate to within 1.5 to 3.0 meters horizontally, less vertically.
            R_gps_horizontal: 2.25,
            R_gps_vertical: 9.0,
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
    pub fn predict(&mut self, delta_vel_ned_mps: Vector2f32, delta_pos_ned_m: Vector2f32, dt: f32) {
        // Integrate physical kinematics forward (deterministic)
        self.predict_state(delta_vel_ned_mps, delta_pos_ned_m, dt);

        // Propagate covariance uncertainty blocks forward (stochastic)
        self.predict_covariance(dt);

        self.push_snapshot(dt);
    }

    pub fn predict_state(&mut self, delta_vel_ned_mps: Vector2f32, delta_pos_ned_m: Vector2f32, dt: f32) {
        self.base.predict_state(delta_vel_ned_mps, delta_pos_ned_m, dt);
    }

    pub fn predict_covariance(&mut self, dt: f32) {
        self.base.predict_covariance(dt);
    }

    /// Correct position using GPS position measurement (typically at a 1Hz to 10Hz rate).
    pub fn correct_position_using_gps(&mut self, position_m: Vector2f32) -> KalmanFilterDiagnostics {
        let r_gps = Vector2f32 { x: self.R_gps_horizontal, y: self.R_gps_horizontal };
        self.base.correct_position(position_m, r_gps)
    }

    /// Correct position using optical flow position measurement.
    pub fn correct_position_using_optical_flow(&mut self, position_m: Vector2f32) -> KalmanFilterDiagnostics {
        self.base.correct_position(position_m, self.R_optical_flow)
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
        measurement_m: Vector2f32,
        R: Vector2f32,
        sensor_time: f32,
        dt: f32,
    ) -> KalmanFilterDiagnostics {
        // Find the past state matching the sensor timestamp
        let Some((past, _)) = self.find_snapshot(sensor_time, dt) else {
            return KalmanFilterDiagnostics::default();
        };
        let predicted_measurement_m = past.pos_m;
        self.base.correct_position_short_form(measurement_m, predicted_measurement_m, R)
    }
}

impl KalmanFilterXyWithSensors {
    /// Caches the current state of the filter into our history ring buffer.
    /// Run this at the very end of EVERY IMU time update step.
    pub fn push_snapshot(&mut self, dt: f32) {
        // Cache the fully synchronized state + covariance into history
        // Call this here so it captures the results of BOTH predictions
        // Accumulate the raw measurements to form a true average over the window
        self.history.tick_counter += 1;

        self.history.system_time += dt;

        let snapshot =
            Snapshot { time_stamp: self.history.system_time, pos_m: self.pos_m(), gps_pos_m: None, baro_alt_m: None };

        self.history.head_idx = (self.history.head_idx + 1) % self.history.snapshot.len();
        self.history.snapshot[self.history.head_idx] = snapshot;

        // Reset accumulators
        self.history.tick_counter = 0;
    }

    pub fn register_gps_event(&mut self, gps_pos_m: Vector2f32) {
        self.history.snapshot[self.history.head_idx].gps_pos_m = Some(gps_pos_m);
    }

    /// Retrieves a historical snapshot of the filter state at a given index.
    /// The index is relative to the most recent snapshot, with 0 being the latest.
    #[must_use]
    pub fn find_snapshot(&self, sensor_time: f32, dt: f32) -> Option<(Snapshot, usize)> {
        let mut match_idx = None;
        let mut min_diff = f32::MAX;
        #[allow(clippy::cast_precision_loss)]
        let diff_threshold = dt * 0.5; // Allowable time difference threshold

        for ii in 0..self.history.snapshot.len() {
            let time_stamp = self.history.snapshot[ii].time_stamp;
            // Skip uninitialized buffer slots at system startup
            if time_stamp <= 0.0 {
                continue;
            }
            let diff = (time_stamp - sensor_time).abs();
            if diff < min_diff && diff < diff_threshold {
                min_diff = diff;
                match_idx = Some(ii);
            }
        }

        let start_idx = match_idx?;

        Some((self.history.snapshot[start_idx], start_idx))
    }
}

#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snapshot {
    pub time_stamp: f32,

    pub pos_m: Vector2f32,
    pub gps_pos_m: Option<Vector2f32>,
    pub baro_alt_m: Option<f32>,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::new()
    }
}

impl Snapshot {
    pub const fn new() -> Self {
        Self { time_stamp: 0.0, pos_m: Vector2f32::ZERO, gps_pos_m: None, baro_alt_m: None }
    }
}
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXyHistory {
    pub snapshot: [Snapshot; Self::SNAPSHOT_SIZE], // Fixed circular window (e.g., handles up to 640ms of latency at 100Hz)
    /// Ring buffer index.
    pub head_idx: usize,
    pub system_time: f32,
    pub tick_counter: usize,
}

impl Default for KalmanFilterXyHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterXyHistory {
    pub const SNAPSHOT_SIZE: usize = 128; // 256ms of history at 500Hz

    pub const fn new() -> Self {
        Self { snapshot: [Snapshot::new(); Self::SNAPSHOT_SIZE], head_idx: 0, system_time: 0.0, tick_counter: 0 }
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
        is_full::<Snapshot>();
    }
}
