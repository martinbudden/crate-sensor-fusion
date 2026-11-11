#![allow(non_snake_case)]
#![allow(clippy::clone_on_copy)]

use sensor_fusion::{FusionStatus, KalmanFilterXyz, KalmanStateXyz};
use vqm::{Matrix3x3f32, Matrix3x3xM3x3f32, Vector3f32};

use nalgebra::{DMatrix, DVector};

// Adjust the import paths above to match your crate.

// ------------------------------------------------------------
// Test tolerances.
// ------------------------------------------------------------

const STATE_TOLERANCE: f64 = 2e-4;
const COVARIANCE_TOLERANCE: f64 = 2e-4;
const NIS_TOLERANCE: f64 = 2e-4;
const SYMMETRY_TOLERANCE: f64 = 1e-5;

// ------------------------------------------------------------
// Measurement types.
// ------------------------------------------------------------

#[derive(Clone, Copy)]
enum MeasurementType {
    Position,
    Velocity,
    Both,
}

// ------------------------------------------------------------
// Conversion helpers.
// ------------------------------------------------------------

fn to_na9(P: &Matrix3x3xM3x3f32) -> DMatrix<f64> {
    let mut result = DMatrix::zeros(9, 9);

    for block_row in 0..3 {
        for block_col in 0..3 {
            let block = &P[block_row + block_col * 3];

            for r in 0..3 {
                for c in 0..3 {
                    result[(block_row * 3 + r, block_col * 3 + c)] = block[r + c * 3] as f64;
                }
            }
        }
    }

    result
}

fn to_na_state(state: &KalmanStateXyz) -> DVector<f64> {
    DVector::from_vec(vec![
        state.pos_ned_m.x as f64,
        state.pos_ned_m.y as f64,
        state.pos_ned_m.z as f64,
        state.vel_ned_mps.x as f64,
        state.vel_ned_mps.y as f64,
        state.vel_ned_mps.z as f64,
        state.acc_bias_ned_mps2.x as f64,
        state.acc_bias_ned_mps2.y as f64,
        state.acc_bias_ned_mps2.z as f64,
    ])
}

#[allow(unused)]
fn calculate_position_update_intermediate_a(
    p: &nalgebra::SMatrix<f64, 9, 9>,
    k: &nalgebra::SMatrix<f64, 9, 3>,
) -> nalgebra::SMatrix<f64, 9, 9> {
    let mut identity = nalgebra::SMatrix::<f64, 9, 9>::identity();

    // H = [ I  0  0 ]
    let mut h = nalgebra::SMatrix::<f64, 3, 9>::zeros();
    for i in 0..3 {
        h[(i, i)] = 1.0;
    }

    identity -= k * h;
    identity * p
}

#[allow(unused)]
fn reference_position_intermediate_a(P: &Matrix3x3xM3x3f32, R_position: Vector3f32) -> (DMatrix<f64>, DMatrix<f64>) {
    let P = to_na9(P);

    // H = [ I₃  0₃  0₃ ]
    let mut H = DMatrix::<f64>::zeros(3, 9);

    for i in 0..3 {
        H[(i, i)] = 1.0;
    }

    // Position measurement covariance.
    let r = [R_position.x, R_position.y, R_position.z];
    let mut R = DMatrix::<f64>::zeros(3, 3);

    for i in 0..3 {
        R[(i, i)] = r[i] as f64;
    }

    // Innovation covariance.
    let S = &H * &P * H.transpose() + &R;

    // Cholesky decomposition.
    let cholesky = S.cholesky().expect("Reference innovation covariance is not positive definite");

    // Kalman gain.
    let HP = &H * &P;
    let K = cholesky.solve(&HP).transpose();

    // Joseph intermediate matrix:
    // A = (I - K·H)·P
    let I = DMatrix::<f64>::identity(9, 9);
    let A_intermediate = (&I - &K * &H) * &P;

    (A_intermediate, K)
}

#[allow(unused)]
fn print_a_block_errors(actual: &nalgebra::SMatrix<f64, 9, 9>, expected: &nalgebra::SMatrix<f64, 9, 9>) {
    let names = ["PP", "PV", "PB", "VP", "VV", "VB", "BP", "BV", "BB"];

    for block_row in 0..3 {
        for block_col in 0..3 {
            let mut max_error: f64 = 0.0;

            for i in 0..3 {
                for j in 0..3 {
                    let row = block_row * 3 + i;
                    let col = block_col * 3 + j;

                    max_error = max_error.max((actual[(row, col)] - expected[(row, col)]).abs());
                }
            }

            println!("{}: {:.12e}", names[block_row * 3 + block_col], max_error);
        }
    }
}

fn reference_9x9_predict(P: &Matrix3x3xM3x3f32, dt: f32, q_velocity: f32, q_bias: f32) -> DMatrix<f64> {
    let P = to_na9(P);

    let dt = dt as f64;
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;

    // Construct the full 9x9 state transition matrix.
    let mut F = DMatrix::<f64>::identity(9, 9);

    for i in 0..3 {
        F[(i, i + 3)] = dt;
        F[(i, i + 6)] = -0.5 * dt2;
        F[(i + 3, i + 6)] = -dt;
    }

    // Construct the full 9x9 process noise matrix.
    let mut Q = DMatrix::<f64>::zeros(9, 9);

    let qv = q_velocity as f64;
    let qb = q_bias as f64;

    for i in 0..3 {
        // Position noise.
        Q[(i, i)] = qv * dt3 / 3.0;

        // Position / velocity cross-covariance.
        Q[(i, i + 3)] = 0.5 * qv * dt2;
        Q[(i + 3, i)] = 0.5 * qv * dt2;

        // Velocity noise.
        Q[(i + 3, i + 3)] = qv * dt;

        // Bias random walk.
        Q[(i + 6, i + 6)] = qb * dt;
    }

    // Dense reference covariance prediction:
    //
    // P_new = F * P * Fᵀ + Q
    &F * &P * F.transpose() + Q
}
// ------------------------------------------------------------
// 9x9 reference Kalman update.
//
// Uses f64 and Cholesky decomposition.
// ------------------------------------------------------------

fn reference_9x9_update(
    state: &KalmanStateXyz,
    P: &Matrix3x3xM3x3f32,
    measurement_m: Vector3f32,
    velocity_mps: Vector3f32,
    R_position: Vector3f32,
    R_velocity: Vector3f32,
    measurement_type: MeasurementType,
) -> (DVector<f64>, DMatrix<f64>, f64) {
    let P = to_na9(P);
    let x = to_na_state(state);

    let (H, R, z) = match measurement_type {
        MeasurementType::Position => {
            let mut H = DMatrix::<f64>::zeros(3, 9);
            let mut R = DMatrix::<f64>::zeros(3, 3);

            for i in 0..3 {
                H[(i, i)] = 1.0;
            }

            let r = [R_position.x, R_position.y, R_position.z];
            for i in 0..3 {
                R[(i, i)] = r[i] as f64;
            }

            let z = DVector::from_vec(vec![measurement_m.x as f64, measurement_m.y as f64, measurement_m.z as f64]);

            (H, R, z)
        }

        MeasurementType::Velocity => {
            let mut H = DMatrix::<f64>::zeros(3, 9);
            let mut R = DMatrix::<f64>::zeros(3, 3);

            for i in 0..3 {
                H[(i, i + 3)] = 1.0;
            }

            let r = [R_velocity.x, R_velocity.y, R_velocity.z];
            for i in 0..3 {
                R[(i, i)] = r[i] as f64;
            }

            let z = DVector::from_vec(vec![velocity_mps.x as f64, velocity_mps.y as f64, velocity_mps.z as f64]);

            (H, R, z)
        }

        MeasurementType::Both => {
            let mut H = DMatrix::<f64>::zeros(6, 9);
            let mut R = DMatrix::<f64>::zeros(6, 6);

            for i in 0..3 {
                H[(i, i)] = 1.0;
                H[(i + 3, i + 3)] = 1.0;
            }

            let rp = [R_position.x, R_position.y, R_position.z];
            let rv = [R_velocity.x, R_velocity.y, R_velocity.z];
            for i in 0..3 {
                R[(i, i)] = rp[i] as f64;
                R[(i + 3, i + 3)] = rv[i] as f64;
            }

            let z = DVector::from_vec(vec![
                measurement_m.x as f64,
                measurement_m.y as f64,
                measurement_m.z as f64,
                velocity_mps.x as f64,
                velocity_mps.y as f64,
                velocity_mps.z as f64,
            ]);

            (H, R, z)
        }
    };

    // Innovation.
    let y = &z - &H * &x;

    // Innovation covariance.
    let S = &H * &P * H.transpose() + &R;

    // Cholesky decomposition.
    let cholesky = S.clone().cholesky().expect("Reference innovation covariance is not positive definite");

    // NIS = yᵀ S⁻¹ y.
    let s_inv_times_y = cholesky.solve(&y);
    let nis = y.dot(&s_inv_times_y);

    // K = P Hᵀ S⁻¹.
    let HP = &H * &P;
    let K = cholesky.solve(&HP).transpose();

    // State update.
    let x_new = &x + &K * &y;

    // Joseph covariance update.
    let I = DMatrix::<f64>::identity(9, 9);
    let A = &I - &K * &H;

    let P_new = &A * &P * A.transpose() + &K * &R * K.transpose();

    (x_new, P_new, nis)
}

// ------------------------------------------------------------
// Test filter factory.
//
// The covariance includes non-zero cross-correlations.
// ------------------------------------------------------------

fn make_test_filter() -> KalmanFilterXyz {
    // This constructor assumes the filter has a public constructor
    // accepting the initial state and covariance.
    //
    // If it does not, expose a test-only factory from your library.

    let I = Matrix3x3f32::identity();

    let P = Matrix3x3xM3x3f32::from_column_array([
        I * 4.0,  // PP
        I * 0.6,  // VP
        I * 0.2,  // BP
        I * 0.6,  // PV
        I * 1.0,  // VV
        I * 0.1,  // BV
        I * 0.2,  // PB
        I * 0.1,  // VB
        I * 0.25, // BB
    ]);

    let state = KalmanStateXyz {
        pos_ned_m: Vector3f32 { x: 100.0, y: 200.0, z: 30.0 },
        vel_ned_mps: Vector3f32 { x: 10.0, y: -5.0, z: 2.0 },
        acc_bias_ned_mps2: Vector3f32 { x: 0.01, y: -0.02, z: 0.005 },
    };

    KalmanFilterXyz::new_for_test(state, P, 0.01, 0.001)
}

// ------------------------------------------------------------
// Comparison helpers.
// ------------------------------------------------------------

fn max_matrix_error(actual: &DMatrix<f64>, expected: &DMatrix<f64>) -> f64 {
    assert_eq!(actual.shape(), expected.shape());
    (actual - expected).iter().map(|x| x.abs()).fold(0.0, f64::max)
}

fn compare_results(
    name: &str,
    filter: &KalmanFilterXyz,
    expected_state: &DVector<f64>,
    expected_P: &DMatrix<f64>,
    actual_nis: f64,
    expected_nis: f64,
) {
    let actual_state = to_na_state(&filter.state_for_test());
    let actual_P = to_na9(&filter.covariance_for_test());
    //println!("\nCovariance block errors:");

    //let block_names = ["PP", "VP", "BP", "PV", "VV", "BV", "PB", "VB", "BB"];

    for block_col in 0..3 {
        for block_row in 0..3 {
            let mut max_error = 0.0_f64;
            for r in 0..3 {
                for c in 0..3 {
                    let row = block_row * 3 + r;
                    let col = block_col * 3 + c;

                    let error = (actual_P[(row, col)] - expected_P[(row, col)]).abs();

                    max_error = max_error.max(error);
                }
            }
            //println!("{}: {:.12e}", block_names[block_row + block_col * 3], max_error);
        }
    }

    let state_error = (&actual_state - expected_state).iter().map(|x| x.abs()).fold(0.0, f64::max);
    let covariance_error = max_matrix_error(&actual_P, expected_P);
    let nis_error = (actual_nis - expected_nis).abs();

    /*println!("\n========== {} ==========", name);
    println!("Maximum state error:      {:.12e}", state_error);
    println!("Maximum covariance error: {:.12e}", covariance_error);
    println!("Actual NIS:               {:.12}", actual_nis);
    println!("Expected NIS:             {:.12}", expected_nis);
    println!("NIS error:                {:.12e}", nis_error);

    for i in 0..9 {
        println!(
            "State[{}]: actual={:.10}, expected={:.10}, error={:.10}",
            i,
            actual_state[i],
            expected_state[i],
            actual_state[i] - expected_state[i]
        );
    }*/

    assert!(state_error < STATE_TOLERANCE, "{}: state mismatch: {}", name, state_error);
    assert!(covariance_error < COVARIANCE_TOLERANCE, "{}: covariance mismatch: {}", name, covariance_error);
    assert!(nis_error < NIS_TOLERANCE, "{}: NIS mismatch: {}", name, nis_error);

    // Check covariance symmetry.
    let symmetry_error = max_matrix_error(&actual_P, &actual_P.transpose());
    assert!(symmetry_error < SYMMETRY_TOLERANCE, "{}: covariance is not symmetric: {}", name, symmetry_error);

    // Check positive definiteness.
    let eigenvalues = actual_P.clone().symmetric_eigen().eigenvalues;
    let minimum_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    //println!("Minimum covariance eigenvalue: {:.12e}", minimum_eigenvalue);
    assert!(minimum_eigenvalue > 0.0, "{}: covariance is not positive definite: {}", name, minimum_eigenvalue);
}

// ------------------------------------------------------------
// Shared measurements.
// ------------------------------------------------------------

fn test_measurements() -> (Vector3f32, Vector3f32, Vector3f32, Vector3f32) {
    (
        Vector3f32 { x: 101.2, y: 198.5, z: 31.0 },
        Vector3f32 { x: 10.4, y: -4.7, z: 1.8 },
        Vector3f32 { x: 0.5, y: 0.5, z: 1.0 },
        Vector3f32 { x: 0.04, y: 0.04, z: 0.09 },
    )
}

#[cfg(test)]
mod test_gps_reference {
    use super::*;

    // ------------------------------------------------------------
    // TEST 1: Position-only correction.
    // ------------------------------------------------------------
    #[test]
    fn position_only_matches_9x9_reference() {
        let mut filter = make_test_filter();
        let initial_filter = filter.clone();

        let (measurement_m, velocity_mps, R_position, R_velocity) = test_measurements();

        let initial_state = initial_filter.state_for_test();
        let initial_P = initial_filter.covariance_for_test();

        let predicted_measurement_m = initial_state.pos_ned_m;

        println!("\n========== POSITION UPDATE REFERENCE ==========");

        /*println!("Reference Kalman gain K:");
        println!("{:#?}", K_reference);

        println!("Reference intermediate A = (I - KH)P:");
        // Calculate the reference intermediate A matrix.
        let (A_reference, K_reference) = reference_position_intermediate_a(&initial_P, R_position);
        println!("{:#?}", A_reference);*/

        // Calculate the complete 9x9 reference update.
        let (expected_state, expected_P, expected_nis) = reference_9x9_update(
            &initial_state,
            &initial_P,
            measurement_m,
            velocity_mps,
            R_position,
            R_velocity,
            MeasurementType::Position,
        );

        // Run the production implementation.
        let diagnostics = filter.correct_position_joseph_stabilized_form(
            measurement_m,
            predicted_measurement_m,
            R_position,
            f32::MAX,
        );

        assert_eq!(diagnostics.status, FusionStatus::Accepted);

        compare_results("POSITION ONLY", &filter, &expected_state, &expected_P, diagnostics.nis as f64, expected_nis);
    }

    // ------------------------------------------------------------
    // TEST 2: Velocity-only correction.
    // ------------------------------------------------------------
    #[test]
    fn velocity_only_matches_9x9_reference() {
        let mut filter = make_test_filter();
        let initial_filter = filter.clone();

        let (measurement_m, velocity_mps, R_position, R_velocity) = test_measurements();

        let predicted_velocity_mps = initial_filter.state_for_test().vel_ned_mps;

        let (expected_state, expected_P, expected_nis) = reference_9x9_update(
            &initial_filter.state_for_test(),
            &initial_filter.covariance_for_test(),
            measurement_m,
            velocity_mps,
            R_position,
            R_velocity,
            MeasurementType::Velocity,
        );

        let diagnostics =
            filter.correct_velocity_joseph_stabilized_form(velocity_mps, predicted_velocity_mps, R_velocity, f32::MAX);

        assert_eq!(diagnostics.status, FusionStatus::Accepted);

        compare_results("VELOCITY ONLY", &filter, &expected_state, &expected_P, diagnostics.nis as f64, expected_nis);
    }

    // ------------------------------------------------------------
    // TEST 3: Combined position then velocity.
    // ------------------------------------------------------------
    #[test]
    fn combined_position_then_velocity_matches_9x9_reference() {
        let mut filter = make_test_filter();
        let initial_filter = filter.clone();

        let (measurement_m, velocity_mps, R_position, R_velocity) = test_measurements();

        let predicted_measurement_m = initial_filter.state_for_test().pos_ned_m;

        let predicted_velocity_mps = initial_filter.state_for_test().vel_ned_mps;

        let (expected_state, expected_P, expected_nis) = reference_9x9_update(
            &initial_filter.state_for_test(),
            &initial_filter.covariance_for_test(),
            measurement_m,
            velocity_mps,
            R_position,
            R_velocity,
            MeasurementType::Both,
        );

        let diagnostics = filter.correct_position_and_velocity_joseph_stabilized_form(
            measurement_m,
            predicted_measurement_m,
            R_position,
            velocity_mps,
            predicted_velocity_mps,
            R_velocity,
            f32::MAX,
        );

        assert_eq!(diagnostics.status, FusionStatus::Accepted);

        compare_results(
            "COMBINED POSITION THEN VELOCITY",
            &filter,
            &expected_state,
            &expected_P,
            diagnostics.nis as f64,
            expected_nis,
        );
    }

    // ------------------------------------------------------------
    // TEST 4: Combined velocity then position.
    // ------------------------------------------------------------
    #[test]
    fn combined_velocity_then_position_matches_9x9_reference() {
        let mut filter = make_test_filter();
        let initial_filter = filter.clone();

        let (measurement_m, velocity_mps, R_position, R_velocity) = test_measurements();

        let (expected_state, expected_P, expected_nis) = reference_9x9_update(
            &initial_filter.state_for_test(),
            &initial_filter.covariance_for_test(),
            measurement_m,
            velocity_mps,
            R_position,
            R_velocity,
            MeasurementType::Both,
        );

        // First correct velocity.
        let diagnostics_velocity = filter.correct_velocity_joseph_stabilized_form(
            velocity_mps,
            initial_filter.state_for_test().vel_ned_mps,
            R_velocity,
            f32::MAX,
        );

        assert_eq!(diagnostics_velocity.status, FusionStatus::Accepted);

        // Then correct position using the updated predicted position.
        let predicted_measurement_m = filter.state_for_test().pos_ned_m;

        let diagnostics_position = filter.correct_position_joseph_stabilized_form(
            measurement_m,
            predicted_measurement_m,
            R_position,
            f32::MAX,
        );

        assert_eq!(diagnostics_position.status, FusionStatus::Accepted);

        let combined_nis = diagnostics_velocity.nis + diagnostics_position.nis;

        compare_results(
            "COMBINED VELOCITY THEN POSITION",
            &filter,
            &expected_state,
            &expected_P,
            combined_nis as f64,
            expected_nis,
        );
    }
}

#[cfg(test)]
mod test_predict_covariance {
    use super::*;

    #[test]
    fn predict_covariance_matches_9x9_reference() {
        let mut filter = make_test_filter();
        let initial_filter = filter.clone();

        let dt = 0.25_f32;

        // IMPORTANT:
        // Use the same process noise values configured in make_test_filter().
        let q_velocity = 0.01;
        let q_bias = 0.001;

        let expected_P = reference_9x9_predict(&initial_filter.covariance_for_test(), dt, q_velocity, q_bias);
        println!("\nPrediction diagnostic:");
        println!("dt:             {:.9}", dt);
        println!("Reference Qv:   {:.9}", q_velocity);
        println!("Reference Qb:   {:.9}", q_bias);
        println!("Expected VV:    {:.9}", expected_P[(3, 3)]);
        println!("Initial VV:     {:.9}", to_na9(&initial_filter.covariance_for_test())[(3, 3)]);
        let P0 = to_na9(&initial_filter.covariance_for_test());

        println!("\nInitial velocity covariance diagnostic:");
        println!("VV: {:.9}", P0[(3, 3)]);
        println!("BV: {:.9}", P0[(6, 3)]);
        println!("VB: {:.9}", P0[(3, 6)]);
        println!("BB: {:.9}", P0[(6, 6)]);

        println!(
            "Expected VV propagation: {:.9}",
            P0[(3, 3)] - dt as f64 * (P0[(6, 3)] + P0[(3, 6)])
                + (dt as f64).powi(2) * P0[(6, 6)]
                + q_velocity as f64 * dt as f64
        );
        // Run the actual production implementation.
        filter.predict_covariance(dt);

        println!("\nProduction covariance after prediction:");
        println!("VV after prediction: {:?}", filter.covariance_for_test()[KalmanFilterXyz::VV]);
        let actual_P = to_na9(&filter.covariance_for_test());

        // Compare all 81 covariance elements.
        let mut max_error = 0.0_f64;
        let mut max_error_row = 0;
        let mut max_error_col = 0;

        println!("\n========== COVARIANCE PREDICTION ==========");

        for row in 0..9 {
            for col in 0..9 {
                let error = (actual_P[(row, col)] - expected_P[(row, col)]).abs();

                if error > max_error {
                    max_error = error;
                    max_error_row = row;
                    max_error_col = col;
                }
            }
        }

        println!("Maximum covariance error: {:.12e}", max_error);
        println!(
            "Maximum error at [{}, {}]: actual={:.12e}, expected={:.12e}",
            max_error_row,
            max_error_col,
            actual_P[(max_error_row, max_error_col)],
            expected_P[(max_error_row, max_error_col)],
        );

        // Report errors for each 3x3 covariance block.
        //let block_names = [["PP", "PV", "PB"], ["VP", "VV", "VB"], ["BP", "BV", "BB"]];

        //println!("\nCovariance block errors:");
        for block_row in 0..3 {
            for block_col in 0..3 {
                let mut block_error = 0.0_f64;
                for i in 0..3 {
                    for j in 0..3 {
                        let row = block_row * 3 + i;
                        let col = block_col * 3 + j;
                        let error = (actual_P[(row, col)] - expected_P[(row, col)]).abs();
                        block_error = block_error.max(error);
                    }
                }
                //println!("{}: {:.12e}", block_names[block_row][block_col], block_error);
            }
        }

        // Check symmetry.
        let mut symmetry_error = 0.0_f64;

        for row in 0..9 {
            for col in 0..9 {
                symmetry_error = symmetry_error.max((actual_P[(row, col)] - actual_P[(col, row)]).abs());
            }
        }

        println!("Maximum symmetry error: {:.12e}", symmetry_error);

        assert!(
            max_error < 1.0e-5,
            "COVARIANCE PREDICTION: covariance mismatch: {:.12e} at [{}, {}]",
            max_error,
            max_error_row,
            max_error_col,
        );

        assert!(symmetry_error < 1.0e-6, "COVARIANCE PREDICTION: covariance is not symmetric: {:.12e}", symmetry_error,);
    }
}
