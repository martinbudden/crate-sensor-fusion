use sensor_fusion::{ImuIntegrator, KalmanFilterXyz};
use vqm::{Matrix3x3f32, Matrix3x3xM3x3f32, Quaternionf32, Vector3f32};

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_3d_position_convergence() {
        let mut filter = KalmanFilterXyz::new();
        let mut integrator = ImuIntegrator::new();
        let orientation = Quaternionf32::identity();

        // Capture initial uncertainty variance values from the main diagonal blocks
        let init_p_pos_x = filter.pp()[KalmanFilterXyz::M11];
        //let init_p_vel_x = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M11];

        // Setup simulation pacing parameters
        let dt = 0.01; // 100 Hz tracking thread loop

        // Define realistic GPS/UWB measurement noise variances (R)
        // Tracks ±1.5m horizontal variance and ±3.0m vertical variance
        let r_gps = Vector3f32 { x: 2.25, y: 2.25, z: 9.0 };

        // Define the ground-truth physical state parameters
        let mut true_pos_m = Vector3f32 { x: 10.0, y: -5.0, z: 0.0 }; // Start at an offset
        let mut true_vel_mps = Vector3f32::default();
        let true_bias_mps2 = Vector3f32 { x: -0.03, y: 0.04, z: 0.01 }; // System accelerometer bias
        let gravity_mps2 = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 }; // NED gravity

        // Prime the filter's initial position guess to match our starting point
        filter.set_position(true_pos_m);

        println!("\n--- PHASE 1: STATIONARY GPS LOCK (1 SECOND) ---");
        #[allow(clippy::cast_possible_truncation)]
        let loops_stationary = (1.0 / dt) as i32;
        for _ in 0..loops_stationary {
            // Raw IMU reading = Kinematic Acc (0) + Bias - Gravity Reaction Force
            #[allow(unused)]
            let acc_body_mps2 = Vector3f32::default() + true_bias_mps2 - gravity_mps2;
            //let delta_vel_ned_mps = acc_measurement_mps2 * dt;
            integrator.integrate(acc_body_mps2, orientation, dt);
            let integral = integrator.imu_integral();
            integrator.reset();

            filter.predict_state(integral.delta_vel_ned_mps, integral.delta_pos_ned_m, dt);
            filter.predict_covariance(dt);

            //filter.correct_position(true_pos, r_gps);
            _ = filter.correct_position(true_pos_m, r_gps);
        }

        // --- PHASE 1 AUDIT LOGS ---
        println!("Stationary -> True Pos X: {:.4}, Estimated Pos X: {:.4}", true_pos_m.x, filter.pos_m().x);
        println!("Stationary -> True Vel X: {:.4}, Estimated Vel X: {:.4}", true_vel_mps.x, filter.vel_mps().x);
        println!(
            "Stationary -> True Bias X: {:.4}, Estimated Bias X: {:.4}",
            true_bias_mps2.x,
            filter.acc_bias_mps2().x
        );

        // Verification assertions for Phase 1
        assert!((filter.pos_m().x - true_pos_m.x).abs() < 0.05, "Position drifted while stationary!");
        assert!((filter.vel_mps().x - true_vel_mps.x).abs() < 0.05, "Velocity accumulated noise while stationary!");

        // Ensure covariance has shrunk significantly below the starting bounds
        let post_stat_p_pos_x = filter.pp()[KalmanFilterXyz::M11];
        assert!(post_stat_p_pos_x < init_p_pos_x, "Position covariance failed to contract under GPS track!");

        println!("\n--- PHASE 2: 3D DYNAMIC SLIDE MANEUVER (2 SECONDS) ---");
        // Simulate a diagonal lateral acceleration profile (+X, +Y)
        #[allow(clippy::cast_possible_truncation)]
        let loops_maneuver = (2.0 / dt) as i32;
        let true_acc_kinematic = Vector3f32 { x: 1.5, y: 2.0, z: 0.0 };

        for _ in 0..loops_maneuver {
            // Update truth kinematics using the exact model equations (Trapezoidal Rule)
            true_pos_m += (true_vel_mps + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel_mps += true_acc_kinematic * dt;

            // Raw IMU specific force generation
            #[allow(unused)]
            let acc_body_mps2 = true_acc_kinematic + true_bias_mps2 - gravity_mps2;
            integrator.integrate(acc_body_mps2, orientation, dt);
            let integral = integrator.imu_integral();
            integrator.reset();

            filter.predict_state(integral.delta_vel_ned_mps, integral.delta_pos_ned_m, dt);
            filter.predict_covariance(dt);

            _ = filter.correct_position(true_pos_m, r_gps);
        }

        // --- PHASE 2 AUDIT LOGS ---
        println!("Maneuver -> True Final Pos X: {:.4}, Estimated Final Pos X: {:.4}", true_pos_m.x, filter.pos_m().x);
        println!("Maneuver -> True Final Pos Y: {:.4}, Estimated Final Pos Y: {:.4}", true_pos_m.y, filter.pos_m().y);
        println!(
            "Maneuver -> True Final Vel X: {:.4}, Estimated Final Vel X: {:.4}",
            true_vel_mps.x,
            filter.vel_mps().x
        );
        println!(
            "Maneuver -> True Final Vel Y: {:.4}, Estimated Final Vel Y: {:.4}",
            true_vel_mps.y,
            filter.vel_mps().y
        );

        // Tracking precision assertions
        assert!((filter.pos_m().x - true_pos_m.x).abs() < 0.1, "Filter missed true physical X position track!");
        assert!((filter.pos_m().y - true_pos_m.y).abs() < 0.1, "Filter missed true physical Y position track!");
        assert!(
            (filter.vel_mps().x - true_vel_mps.x).abs() < 0.1,
            "Filter velocity tracking tracking failed on X axis!"
        );
        assert!(
            (filter.vel_mps().y - true_vel_mps.y).abs() < 0.1,
            "Filter velocity tracking tracking failed on Y axis!"
        );

        println!("\n✅ 3D position state propagation and measurement updates verified successfully!");
    }

    #[test]
    fn test_altitude_convergence() {
        let mut filter = KalmanFilterXyz::new();
        let mut integrator = ImuIntegrator::new();
        let orientation = Quaternionf32::identity();

        // Capture the initial uncertainty bounds
        let initial_p_pos_m = filter.pp()[KalmanFilterXyz::M33];
        // let initial_p_vel_mps = filter.pv()[PositionKalmanFilter::M33];

        // Define simulation parameters
        let dt = 0.01; // 100 Hz simulation step
        let r_baro = 0.04; // 20cm variance for our barometer

        // True Physical States (Sim truths to compare against)
        let mut true_pos_m = Vector3f32::default();
        let mut true_vel_mps = Vector3f32::default();

        // Explicitly track a constant real accelerometer bias in the hardware
        let true_bias_mps2 = Vector3f32 { x: 0.02, y: -0.01, z: 0.05 };

        // Aerospace NED Gravity vector (Positive down)
        let gravity_mps2 = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 };

        println!("\n--- PHASE 1: STATIONARY BENCH TEST (2 SECONDS) ---");
        // During this phase, an IMU resting horizontally on the ground in an NED frame
        // measures an upward reaction force against gravity.
        #[allow(clippy::cast_possible_truncation)]
        let sim_loops_stationary = (2.0 / dt) as i32;

        for step in 0..sim_loops_stationary {
            // Raw sensor reading = True Kinematic Acc (0) + Bias + Upward Reaction Force (-gravity)
            #[allow(unused)]
            let acc_body_mps2 = Vector3f32::default() + true_bias_mps2 - gravity_mps2;

            // Step A: Time Update
            integrator.integrate(acc_body_mps2, orientation, dt);
            let integral = integrator.imu_integral();
            integrator.reset();

            filter.predict_state(integral.delta_vel_ned_mps, integral.delta_pos_ned_m, dt);
            filter.predict_covariance(dt);

            // Step B: Measurement Update (Simulate stationary altitude updates)
            // Injecting a fixed R variance value
            _ = filter.correct_altitude(true_pos_m.z, r_baro);
            // Print out the Z-variance tracking elements every 50 steps
            if step % 50 == 0 {
                println!(
                    "Step {:03} -> P_pos_z: {:e}, P_vel_z: {:e}, P_bias_z: {:e}",
                    step,
                    filter.pp()[KalmanFilterXyz::M33],
                    filter.pv()[KalmanFilterXyz::M33],
                    filter.bb()[KalmanFilterXyz::M33]
                );
            }
        }

        // --- VERIFICATIONS FOR PHASE 1 ---
        println!("True Pos Z: {:.4}, Estimated Pos Z: {:.4}", true_pos_m.z, filter.pos_m().z);
        println!("True Vel Z: {:.4}, Estimated Vel Z: {:.4}", true_vel_mps.z, filter.vel_mps().z);
        println!("True Bias Z: {:.4}, Estimated Bias Z: {:.4}", true_bias_mps2.z, filter.acc_bias_mps2().z);

        // Assertions: The filter should remain near zero despite raw measurements reading ~ -9.81
        assert!((filter.pos_m().z - true_pos_m.z).abs() < 0.05, "Position drifted significantly while resting!");
        assert!((filter.vel_mps().z - true_vel_mps.z).abs() < 0.05, "Velocity accumulated phantom motion!");

        // Assertions: Covariance bounds MUST contract if the sensor math is operating properly
        let post_stationary_p_pos = filter.pp()[KalmanFilterXyz::M33];
        assert!(post_stationary_p_pos < initial_p_pos_m, "Covariance failed to contract with sensor updates!");

        println!("\n--- PHASE 2: UNIFORM CLIMB TEST (3 SECONDS) ---");
        // Vehicle climbs downward/upward. Let's simulate a downward acceleration in NED (+Z)
        // of 1.0 m/s^2 for a clean mathematical trajectory.
        #[allow(clippy::cast_possible_truncation)]
        let sim_loops_ascent = (3.0 / dt) as i32;
        let true_acc_kinematic = Vector3f32 { x: 0.0, y: 0.0, z: 1.0 };

        for _ in 0..sim_loops_ascent {
            // Update physical truths using the exact kinematic laws inside our test harness
            true_pos_m += (true_vel_mps + 0.5 * true_acc_kinematic * dt) * dt;
            true_vel_mps += true_acc_kinematic * dt;

            // Generate sensor reading: Kinematic Acc + Bias - Upward Reaction Force
            #[allow(unused)]
            let acc_body_mps2 = true_acc_kinematic + true_bias_mps2 - gravity_mps2;

            // Step A: Time Update
            integrator.integrate(acc_body_mps2, orientation, dt);
            let integral = integrator.imu_integral();
            integrator.reset();

            filter.predict_state(integral.delta_vel_ned_mps, integral.delta_pos_ned_m, dt);
            filter.predict_covariance(dt);

            // Step B: Measurement Update with simulated moving barometer data
            _ = filter.correct_altitude(true_pos_m.z, r_baro);
        }

        // --- VERIFICATIONS FOR PHASE 2 ---
        println!("True Final Pos Z: {:.4}, Estimated Final Pos Z: {:.4}", true_pos_m.z, filter.pos_m().z);
        println!("True Final Vel Z: {:.4}, Estimated Final Vel Z: {:.4}", true_vel_mps.z, filter.vel_mps().z);

        // The filter tracking error should be tight
        assert!((filter.pos_m().z - true_pos_m.z).abs() < 0.1, "Filter failed to track dynamic trajectory accurately!");
        assert!((filter.vel_mps().z - true_vel_mps.z).abs() < 0.1, "Filter velocity tracking diverged during climb!");

        println!("\n✅ All Kalman Filter math, indices, and coordinate signs verified successfully!");
    }
}

#[cfg(test)]
mod test_anchors {
    use super::*;
    use num_traits::Zero;
    use sensor_fusion::{KalmanFilterXyz, KalmanStateXyz};

    fn v(x: f32, y: f32, z: f32) -> Vector3f32 {
        Vector3f32 { x, y, z }
    }

    fn assert_near(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() < epsilon,
            "expected {}, got {}, error {}",
            expected,
            actual,
            (actual - expected).abs(),
        );
    }

    fn make_filter() -> KalmanFilterXyz {
        KalmanFilterXyz::new().with_state(KalmanStateXyz {
            pos_ned_m: v(3.0, 4.0, 0.0),
            vel_ned_mps: v(0.0, 0.0, 0.0),
            acc_bias_ned_mps2: v(0.0, 0.0, 0.0),
        })
    }

    #[test]
    fn range_update_basic() {
        const EPS: f32 = 2.0e-3;
        let mut kf = make_filter();

        // Position = (3,4,0), anchor = (0,0,0)
        // Predicted range = 5
        //
        // H = [0.6, 0.8, 0]
        //
        // Measurement = 6
        // innovation = +1
        //
        // S = 1 + 0.01 = 1.01

        _ = kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 6.0, 100.0, 0.01);

        let scale = 1.0 / 1.01;

        assert!(kf.pos_m().is_near(v(3.0 + 0.6 * scale, 4.0 + 0.8 * scale, 0.0), EPS));

        // No cross covariance, so velocity and bias should not change.
        assert!(kf.vel_mps().is_near(v(0.0, 0.0, 0.0), EPS));

        assert!(kf.acc_bias_mps2().is_near(v(0.0, 0.0, 0.0), EPS));
    }

    #[test]
    fn range_update_corrects_velocity_through_cross_covariance() {
        const EPS: f32 = 2.0e-3;

        let identity = Matrix3x3f32::identity();
        let zero = Matrix3x3f32::zero();

        let cross = Matrix3x3f32::from_diagonal_element(0.5);

        let mut kf = KalmanFilterXyz::new()
            .with_state(KalmanStateXyz {
                pos_ned_m: v(3.0, 4.0, 0.0),
                vel_ned_mps: v(0.0, 0.0, 0.0),
                acc_bias_ned_mps2: v(0.0, 0.0, 0.0),
            })
            .with_p(Matrix3x3xM3x3f32::from_column_array([
                identity, cross, zero, // PP, VP, BP
                cross, identity, zero, // PV, VV, BV
                zero, zero, identity, // PB, VB, BB
            ]));

        _ = kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 6.0, 100.0, 0.01);

        // P_VP * H^T:
        //
        // [0.5 0   0]   [0.6]   [0.3]
        // [0   0.5 0] * [0.8] = [0.4]
        // [0   0   0.5] [0  ]   [0.0]
        //
        // S = 1.01
        //
        // innovation = 1

        let scale = 1.0 / 1.01;

        assert!(kf.vel_mps().is_near(v(0.3 * scale, 0.4 * scale, 0.0), EPS));
    }

    #[test]
    fn range_update_corrects_accelerometer_bias_through_cross_covariance() {
        const EPS: f32 = 2.0e-3;

        let identity = Matrix3x3f32::identity();
        let zero = Matrix3x3f32::zero();

        let bp = Matrix3x3f32::from_diagonal_array([0.2, 0.3, 0.4]);
        let pb = bp;

        let mut kf = KalmanFilterXyz::new()
            .with_state(KalmanStateXyz {
                pos_ned_m: v(3.0, 4.0, 0.0),
                vel_ned_mps: v(0.0, 0.0, 0.0),
                acc_bias_ned_mps2: v(0.0, 0.0, 0.0),
            })
            .with_p(Matrix3x3xM3x3f32::from_column_array([
                identity, zero, bp, // PP, VP, BP
                zero, identity, zero, // PV, VV, BV
                pb, zero, identity, // PB, VB, BB
            ]));

        _ = kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 6.0, 100.0, 0.01);

        // P_BP * H^T =
        //
        // [0.2 * 0.6] = 0.12
        // [0.3 * 0.8] = 0.24
        // [0.4 * 0.0] = 0.00

        let scale = 1.0 / 1.01;

        assert!(kf.acc_bias_mps2().is_near(v(0.12 * scale, 0.24 * scale, 0.0), EPS));
    }

    #[test]
    fn perfect_range_does_not_change_state() {
        const EPS: f32 = 2.4e-3;

        let mut kf = make_filter();

        let original_pos = kf.pos_m();
        let original_vel = kf.vel_mps();
        let original_bias = kf.acc_bias_mps2();

        // Predicted range is exactly 5.
        _ = kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 5.0, 100.0, 0.01);

        assert!(kf.pos_m().is_near(original_pos, EPS));
        assert!(kf.vel_mps().is_near(original_vel, EPS));
        assert!(kf.acc_bias_mps2().is_near(original_bias, EPS));
    }

    #[test]
    fn outlier_is_rejected_without_changing_filter() {
        const EPS: f32 = 2.0e-3;

        let mut kf = make_filter();

        let original_pos = kf.pos_m();
        let original_vel = kf.vel_mps();
        let original_bias = kf.acc_bias_mps2();
        let original_p = *kf.p();

        // Predicted range = 5.
        // 100 m is obviously an outlier.
        _ = kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 100.0, 6.63, 0.01);

        assert!(kf.pos_m().is_near(original_pos, EPS));
        assert!(kf.vel_mps().is_near(original_vel, EPS));
        assert!(kf.acc_bias_mps2().is_near(original_bias, EPS));

        assert!(kf.p()[KalmanFilterXyz::PP].is_near(original_p[KalmanFilterXyz::PP], EPS));
        assert!(kf.p()[KalmanFilterXyz::PV].is_near(original_p[KalmanFilterXyz::PV], EPS));
        assert!(kf.p()[KalmanFilterXyz::PB].is_near(original_p[KalmanFilterXyz::PB], EPS));
        assert!(kf.p()[KalmanFilterXyz::VP].is_near(original_p[KalmanFilterXyz::VP], EPS));
        assert!(kf.p()[KalmanFilterXyz::VV].is_near(original_p[KalmanFilterXyz::VV], EPS));
        assert!(kf.p()[KalmanFilterXyz::VB].is_near(original_p[KalmanFilterXyz::VB], EPS));
        assert!(kf.p()[KalmanFilterXyz::BP].is_near(original_p[KalmanFilterXyz::BP], EPS));
        assert!(kf.p()[KalmanFilterXyz::BV].is_near(original_p[KalmanFilterXyz::BV], EPS));
        assert!(kf.p()[KalmanFilterXyz::BB].is_near(original_p[KalmanFilterXyz::BB], EPS));
    }
    #[test]
    fn range_update_reduces_radial_position_variance() {
        let mut kf = make_filter();

        let h = v(0.6, 0.8, 0.0);

        let before = h.dot(kf.pp() * h);

        _ = kf.correct_position_using_anchor_distance(v(0.0, 0.0, 0.0), 5.0, 100.0, 0.01);

        let after = h.dot(kf.pp() * h);

        assert!(after < before, "radial variance should decrease: before={before}, after={after}");

        // With identity P and a range-only measurement,
        // the direction perpendicular to H should retain its
        // original variance.
        let perpendicular = v(-0.8, 0.6, 0.0);

        let before_perp = perpendicular.dot(kf.pp() * perpendicular);

        assert_near(before_perp, 1.0, 2.0e-3);

        // Z is completely unobserved by this range measurement.
        let z = v(0.0, 0.0, 1.0);

        let z_variance = z.dot(kf.pp() * z);

        assert_near(z_variance, 1.0, 2.0e-3);
    }
}
