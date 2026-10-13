use num_traits::ConstZero;
use vqm::{Vector2f32, Vector3f32};

use super::{KalmanFilterDiagnostics, KalmanFilterXyz};

/// `f32` variant of `KalmanFilterXyzWithSensors`.
pub type KalmanFilterXyzWithSensorsf32 = KalmanFilterXyzWithSensors;

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
pub struct KalmanFilterXyzWithSensors {
    pub base: KalmanFilterXyz,
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
    pub history: KalmanFilterXyzHistory,
}

impl Default for KalmanFilterXyzWithSensors {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterXyzWithSensors {
    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: KalmanFilterXyz::new(),
            // Commercial GPS modules are typically accurate to within 1.5 to 3.0 meters horizontally, less vertically.
            R_gps_horizontal: 2.25,
            R_gps_vertical: 9.0,
            // Vertical variance in meters squared
            // A standard barometric pressure sensor (e.g., BMP280, MS5611) typically has an RMS noise of around 10 to 20 centimeters
            // which corresponds to a variance of 0.01 to 0.04 m². However, the actual noise can vary based on environmental conditions and sensor quality.
            // If the barometer is placed near spinning propellers or inside an enclosure with turbulent wind forces,
            // its real-world noise will be much higher than the datasheet specification.
            // A common practice is to multiply the datasheet variance by 4 to 10 to account for environmental pressure spikes.
            R_barometer: 0.1, // Slightly detuned to account for wind buffeting
            R_rangefinder: 0.03,
            R_optical_flow: Vector2f32 { x: 0.04, y: 0.04 },
            // 1cm to 10cm typical accuracy, use 10cm = 0.10m, = 0.01 variance for cheap modules.
            R_ultra_wideband: 0.01,
            history: KalmanFilterXyzHistory::new(),
        }
    }
}

impl KalmanFilterXyzWithSensors {
    #[inline]
    #[must_use]
    pub fn pos(&self) -> Vector3f32 {
        self.base.state.pos
    }

    #[inline]
    #[must_use]
    pub fn vel(&self) -> Vector3f32 {
        self.base.state.vel
    }

    #[inline]
    #[must_use]
    pub fn acc_bias(&self) -> Vector3f32 {
        self.base.state.acc_bias
    }

    /// Called inside the main IMU execution loop (eg `gyro_pid` loop).
    pub fn predict(&mut self, acc: Vector3f32, dt: f32) {
        // Integrate physical kinematics forward (deterministic)
        self.predict_state(acc, dt);

        // Propagate covariance uncertainty blocks forward (stochastic)
        self.predict_covariance(dt);

        self.push_snapshot(dt);
    }

    pub fn predict_state(&mut self, acc_measurement: Vector3f32, dt: f32) {
        self.base.predict_state(acc_measurement, dt);
    }

    pub fn predict_covariance(&mut self, dt: f32) {
        self.base.predict_covariance(dt);
    }

    /// Correct altitude using supplied R.
    #[inline]
    pub fn correct_altitude(&mut self, altitude: f32, r: f32) -> KalmanFilterDiagnostics {
        self.base.correct_altitude(altitude, r)
    }

    /// Correct altitude using the barometer measurement.
    #[inline]
    pub fn correct_altitude_using_barometer(&mut self, altitude: f32) -> KalmanFilterDiagnostics {
        self.base.correct_altitude(altitude, self.R_barometer)
    }

    /// Correct altitude using the rangefinder measurement.
    #[inline]
    pub fn correct_altitude_using_rangefinder(&mut self, altitude: f32) -> KalmanFilterDiagnostics {
        self.base.correct_altitude(altitude, self.R_rangefinder)
    }

    /// Correct altitude using GPS vertical measurement.
    #[inline]
    pub fn correct_altitude_using_gps(&mut self, altitude: f32) -> KalmanFilterDiagnostics {
        self.base.correct_altitude(altitude, self.R_gps_vertical)
    }

    /// Correct position using GPS position measurement (typically at a 1Hz to 10Hz rate).
    pub fn correct_position_using_gps(&mut self, position: Vector3f32) -> KalmanFilterDiagnostics {
        let r_gps = Vector3f32 { x: self.R_gps_horizontal, y: self.R_gps_horizontal, z: self.R_gps_vertical };
        self.base.correct_position(position, r_gps)
    }

    /// Correct position using optical flow position measurement.
    pub fn correct_position_using_optical_flow(&mut self, position: Vector2f32) -> KalmanFilterDiagnostics {
        self.base.correct_position_xy(position, self.R_optical_flow)
    }
}

#[allow(non_snake_case)]
impl KalmanFilterXyzWithSensors {
    /*
    Strategy: "State-Only Rewind"

    In a Kalman filter, the Kalman Gain `K` scales down over time as the filter collects measurements.
    Because K changes very slowly, we can make an accurate engineering trade-off:
    Assume the covariance matrix at the current time is close enough to use for a measurement that happened 100ms ago.


    By adopting this strategy, we remove `P` and `acc_raw` from the snapshot entirely.
    The history snapshot drops from 90 floats down to just 10 floats.
    We no longer need a while loop to fast-forward predictions.
    The delayed update becomes a fast operation executed at the present time step.
    */
    /// Correct position using delayed measurement without rewinding `P`.
    /// When a delayed measurement arrives, we calculate the innovation error using the past state vector (`past.pos`).
    /// We calculate the Kalman Gain vectors using the current active P matrix.
    /// We correct the current state vector directly.
    pub fn correct_position_delayed(
        &mut self,
        measurement: Vector3f32,
        R: Vector3f32,
        sensor_time: f32,
        dt: f32,
    ) -> KalmanFilterDiagnostics {
        // Find the past state matching the sensor timestamp
        let Some((past, _)) = self.find_snapshot(sensor_time, dt) else {
            return KalmanFilterDiagnostics::default();
        };
        let predicted_measurement = past.pos;
        self.base.correct_position_short_form(measurement, predicted_measurement, R)
    }
}

impl KalmanFilterXyzWithSensors {
    /// Caches the current state of the filter into our history ring buffer.
    /// Run this at the very end of EVERY IMU time update step.
    pub fn push_snapshot(&mut self, dt: f32) {
        // Cache the fully synchronized state + covariance into history
        // Call this here so it captures the results of BOTH predictions
        // Accumulate the raw measurements to form a true average over the window
        self.history.tick_counter += 1;

        self.history.system_time += dt;

        let snapshot =
            Snapshot { time_stamp: self.history.system_time, pos: self.pos(), gps_pos: None, baro_alt: None };

        self.history.head_idx = (self.history.head_idx + 1) % self.history.snapshot.len();
        self.history.snapshot[self.history.head_idx] = snapshot;

        // Reset accumulators
        self.history.tick_counter = 0;
    }

    pub fn register_gps_event(&mut self, gps_pos: Vector3f32) {
        self.history.snapshot[self.history.head_idx].gps_pos = Some(gps_pos);
    }

    /// Flags that a fresh Barometric Altitude measurement occurred at the current step.
    pub fn register_baro_event(&mut self, altitude: f32) {
        self.history.snapshot[self.history.head_idx].baro_alt = Some(altitude);
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

    pub pos: Vector3f32,
    pub gps_pos: Option<Vector3f32>,
    pub baro_alt: Option<f32>,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self::new()
    }
}

impl Snapshot {
    pub const fn new() -> Self {
        Self { time_stamp: 0.0, pos: Vector3f32::ZERO, gps_pos: None, baro_alt: None }
    }
}
#[allow(non_snake_case)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanFilterXyzHistory {
    pub snapshot: [Snapshot; Self::SNAPSHOT_SIZE], // Fixed circular window (e.g., handles up to 640ms of latency at 100Hz)
    /// Ring buffer index.
    pub head_idx: usize,
    pub system_time: f32,
    pub tick_counter: usize,
}

impl Default for KalmanFilterXyzHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterXyzHistory {
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
        is_full::<KalmanFilterXyzWithSensors>();
        is_full::<KalmanFilterXyzWithSensorsf32>();
        is_full::<KalmanFilterXyzHistory>();
        is_full::<Snapshot>();
    }
}
