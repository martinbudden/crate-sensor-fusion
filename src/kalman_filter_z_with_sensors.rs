use super::{KalmanFilterDiagnostics, KalmanFilterZ, KalmanStateZ};

/// `f32` variant of `PositionKalmanFilterExtended`.
pub type KalmanFilterZWithSensorsf32 = KalmanFilterZWithSensors;

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
pub struct KalmanFilterZWithSensors {
    base: KalmanFilterZ,
    // measurement noise covariance Matrix `R`
    /// Absolute Measurement Noise variance for vertical GPS channels.
    pub R_gps_vertical: f32,
    /// Absolute Measurement Noise variance for barometric pressure altimeter.
    pub R_barometer: f32,
    /// Absolute Measurement Noise variance for rangefinder.
    pub R_rangefinder: f32,
    /// Absolute Measurement Noise variance for optical flow.
    pub R_ultra_wideband: f32,
}

impl Default for KalmanFilterZWithSensors {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilterZWithSensors {
    /// Constructor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: KalmanFilterZ::new(),
            // Commercial GPS modules are typically accurate to within 1.5 to 3.0 meters horizontally, less vertically.
            R_gps_vertical: 9.0,
            // Vertical variance in meters squared
            // A standard barometric pressure sensor (e.g., BMP280, MS5611) typically has an RMS noise of around 10 to 20 centimeters
            // which corresponds to a variance of 0.01 to 0.04 m². However, the actual noise can vary based on environmental conditions and sensor quality.
            R_barometer: 0.03,
            R_rangefinder: 0.03,
            // 1cm to 10cm typical accuracy, use 10cm = 0.10m, = 0.01 variance for cheap modules.
            R_ultra_wideband: 0.01,
        }
    }
}

impl KalmanFilterZWithSensors {
    #[inline]
    #[must_use]
    pub fn state(&self) -> KalmanStateZ {
        self.base.state()
    }

    #[inline]
    #[must_use]
    pub fn pos_m(&self) -> f32 {
        self.base.pos_m()
    }

    #[inline]
    #[must_use]
    pub fn vel_mps(&self) -> f32 {
        self.base.vel_mps()
    }

    #[inline]
    #[must_use]
    pub fn acc_bias_mps2(&self) -> f32 {
        self.base.acc_bias_mps2()
    }

    /// Called inside the main IMU execution loop (eg `gyro_pid` loop).
    pub fn predict(&mut self, delta_vel_mps: f32, delta_pos_m: f32, dt: f32) {
        // Integrate physical kinematics forward (deterministic)
        self.predict_state(delta_vel_mps, delta_pos_m, dt);

        // Propagate covariance uncertainty blocks forward (stochastic)
        self.predict_covariance(dt);
    }

    pub fn predict_state(&mut self, delta_vel_mps: f32, delta_pos_m: f32, dt: f32) {
        self.base.predict_state(delta_vel_mps, delta_pos_m, dt);
    }

    pub fn predict_covariance(&mut self, dt: f32) {
        self.base.predict_covariance(dt);
    }

    /// Correct altitude using supplied R.
    #[inline]
    pub fn correct_altitude(&mut self, altitude_m: f32, r: f32) -> KalmanFilterDiagnostics {
        self.base.correct_altitude(altitude_m, r)
    }

    /// Correct altitude using the barometer measurement.
    #[inline]
    pub fn correct_altitude_using_barometer(&mut self, altitude_m: f32) -> KalmanFilterDiagnostics {
        self.base.correct_altitude(altitude_m, self.R_barometer)
    }

    /// Correct altitude using the rangefinder measurement.
    #[inline]
    pub fn correct_altitude_using_rangefinder(&mut self, altitude_m: f32) -> KalmanFilterDiagnostics {
        self.base.correct_altitude(altitude_m, self.R_rangefinder)
    }

    /// Correct altitude using GPS vertical measurement.
    #[inline]
    pub fn correct_altitude_using_gps(&mut self, altitude_m: f32) -> KalmanFilterDiagnostics {
        self.base.correct_altitude(altitude_m, self.R_gps_vertical)
    }
}

#[cfg(test)]
mod test_traits {
    use super::*;

    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanFilterZWithSensors>();
        is_full::<KalmanFilterZWithSensorsf32>();
    }
}
