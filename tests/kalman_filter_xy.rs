use sensor_fusion::{ImuIntegrator, KalmanFilterXy};
use vqm::{Quaternionf32, Vector2f32, Vector3f32};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_position_convergence() {
        let mut filter = KalmanFilterXy::new();
        let mut integrator = ImuIntegrator::new();
        let orientation = Quaternionf32::identity();

        // Capture initial uncertainty variance values from the main diagonal blocks
        let init_p_pos_x = filter.pp()[KalmanFilterXy::M11];
        //let init_p_vel_x = filter.P[PositionKalmanFilter::VV][PositionKalmanFilter::M11];

        // Setup simulation pacing parameters
        let dt = 0.01; // 100 Hz tracking thread loop

        // Define realistic GPS/UWB measurement noise variances (R)
        // Tracks ±1.5m horizontal variance
        let r_gps = Vector2f32 { x: 2.25, y: 2.25 };

        // Define the ground-truth physical state parameters
        let mut true_pos_m = Vector3f32 { x: 10.0, y: -5.0, z: 0.0 }; // Start at an offset
        let mut true_vel_mps = Vector3f32::default();
        let true_bias_mps2 = Vector3f32 { x: -0.03, y: 0.04, z: 0.0 }; // System accelerometer bias
        let gravity_mps2 = Vector3f32 { x: 0.0, y: 0.0, z: 9.80665 }; // NED gravity

        // Prime the filter's initial position guess to match our starting point
        filter.set_position(true_pos_m.into());

        println!("\n--- PHASE 1: STATIONARY GPS LOCK (1 SECOND) ---");
        #[allow(clippy::cast_possible_truncation)]
        let loops_stationary = (1.0 / dt) as i32;
        for _ in 0..loops_stationary {
            // Raw IMU reading = Kinematic Acc (0) + Bias
            #[allow(unused)]
            let acc_body_mps2 = Vector3f32::default() + true_bias_mps2 - gravity_mps2;
            //let delta_vel_ned_mps = acc_measurement_mps2 * dt;
            integrator.integrate(acc_body_mps2, orientation, dt);
            let integral = integrator.imu_integral();
            integrator.reset();

            filter.predict_state(integral.delta_vel_ned_mps.into(), integral.delta_pos_ned_m.into(), dt);
            filter.predict_covariance(dt);

            //filter.correct_position(true_pos, r_gps);
            _ = filter.correct_position(true_pos_m.into(), r_gps);
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
        let post_stat_p_pos_x = filter.pp()[KalmanFilterXy::M11];
        assert!(post_stat_p_pos_x < init_p_pos_x, "Position covariance failed to contract under GPS track!");

        println!("\n--- PHASE 2: 2D DYNAMIC SLIDE MANEUVER (2 SECONDS) ---");
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

            filter.predict_state(integral.delta_vel_ned_mps.into(), integral.delta_pos_ned_m.into(), dt);
            filter.predict_covariance(dt);

            _ = filter.correct_position(true_pos_m.into(), r_gps);
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

        println!("\n✅ 2D position state propagation and measurement updates verified successfully!");
    }
}
