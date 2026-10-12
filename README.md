# `sensor-fusion` Rust Crate<br>[![Crates.io](https://img.shields.io/crates/v/sensor-fusion.svg)](https://crates.io/crates/sensor-fusion) [![Documentation](https://docs.rs/sensor-fusion/badge.svg)](https://docs.rs/sensor-fusion) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![open source](https://badgen.net/badge/open/source/blue?icon=github)

This crate contains [sensor fusion](https://en.wikipedia.org/wiki/Sensor_fusion) algorithms to combine
output from a gyroscope, accelerometer, and optionally a magnetometer to give output that has less uncertainty
than the output of the individual sensors.

This crate is `no_std`, `no alloc`, and the Minimum Supported Rust Version (MSRV) is `Rust 1.89`.

Six sensor fusion implementations are available:

1. Complementary Filter
2. Mahony Filter
3. Madgwick Filter
4. Altitude Kalman Filter
5. 2D Position Kalman Filter.
6. 3D Position Kalman Filter.

The Madgwick filter and Kalman filters have been optimized for computational efficiency.

## Simple example

Here's a simple example that uses a Madgwick filter to calculate the orientation by fusing accelerometer and gyro values:

```rust
use sensor_fusion::{MadgwickFilterf32, SensorFusion};
use vqm::Vector3f32;

fn main() {
    let mut madgwick = MadgwickFilterf32::new();
    let dt = 0.001; // 1 millisecond

    // Mock sensor values, gyro converted to rps (values normally read from IMU).
    let gyro_rps = Vector3f32::new(90.0, 8.0, 10.0).to_radians();
    let acc = Vector3f32::new(0.1, 0.2, 0.9);

    // Fuse acc and gyro values
    let orientation = madgwick.fuse_acc_gyro(acc, gyro_rps, dt);

    let (roll, pitch, yaw) = orientation.calculate_euler_angles_degrees();

    // Print out the Euler angles.
    println!("pitch={}, roll={}, yaw={}", pitch, roll, yaw);
}
```

## Madgwick Filter

The Madgwick filter has been refactored to be more computationally efficient (and so faster) than
the standard version used in many implementations, see [Optimization](#opt) below.

The Madgwick filter also supports three-way fusing of accelerometer, gyroscope, and magnetometer readings:

```text
let orientation = madgwick.fuse_acc_gyro_mag(acc, gyro_rps, mag, dt);
```

## Mahony filter

The Mahony filter has the same interface as the Madgwick filter:

```text
use crate::sensor_fusion::{MahonyFilterf32,SensorFusion};

let mut mahony = MahonyFilterf32::default();

let orientation = mahony.fuse_acc_gyro(acc, gyro_rps, dt);
```

The Mahony filter does not support 3-way fusion using a magnetometer.

## Complementary filter

This has been implemented for reference. It is not recommended for general use:
both the Mahony filter and the Madgwick filter are faster and don't suffer from gimbal lock.

## Method call interface

The `FuseAccGyro` and `FuseAccGyroMag` traits allow method-call syntax to be used:

```rust
use {crate::sensor_fusion::{FuseAccGyro, FuseAccGyroMag, MadgwickFilterf32}, vqm::Vector3f32};

let dt = 0.001;
let mut madgwick = MadgwickFilterf32::default();
let acc = Vector3f32::default();
let gyro = Vector3f32::default();
let mag = Vector3f32::default();

let orientation = (acc, gyro).fuse_acc_gyro_using(&mut madgwick, dt);
// or
let orientation = (acc, gyro, mag).fuse_acc_gyro_mag_using(&mut madgwick, dt);
```

## Kalman filter

Three Kalman filters are provided:

1. `KalmanFilterZ` - a filter to estimate altitude and vertical speed.
2. `KalmanFilterXY` - a filter to estimate 2D position and velocity.
3. `KalmanFilterXYZ` - a filter to estimate 3D position and velocity.

## 3D Kalman filter

When using a Kalman filter for 3D there are two possible implementation choices:

1. Use a `KalmanFilterXYZ`.
2. Use a spilt architecture: use a `KalmanFilterXY` for horizontal position, and a `KalmanFilterZ` for vertical position.

| Feature            | Split Architecture (1D + 2D)                                     | Unified 3D Architecture                             |
| ------------------ | ---------------------------------------------------------------- | --------------------------------------------------- |
| Computational Cost | Lower (2×2 and 4×4 matrices )                                    | Higher (6×6 and 9×9 matrices)                       |
| Sensor Coupling    | Independent (barometer noise cannot corrupt horizontal accuracy) | Coupled (vertical errors can bleed into horizontal) |
| Tuning Complexity  | Easier (can tune vertical and horizontal separately)             | Harder                                              |
| Fault Tolerance    | Higher                                                           | Lower                                               |

For vehicles that stay upright, or tilt minimally (eg a walker, a car, or a cinematic quadcopter) the split approach is a good one.

For vehicles such as acrobatic quadcopters or rockets, where pitching and rolling movements are common,
the combined approach is better.

## Position Kalman filter overview

A Kalman filter works by predicting an object's state using a physical model of the object and a control input.

So, for example, using acceleration (`a`) as the control input it would predict an object's position and velocity using kinematic equations:

```text
s = ut + ½at²
v = u + at`
```

However, if you keep predicting position an velocity using integration like that, noise in the system
will cause the predicted values to drift away from the actual values.

A Kalman filter periodically corrects this prediction with an actual measurement, for example a
position from a GPS, or an altitude from a barometer.

A Kalman filter can accept measurements from multiple sources to make its corrections,
for example: a GPS, a barometer, a rangefinder, and an optical flow device.

(Note: The control input is normally denoted as **u**, the predicted measurement as **x**, and the actual measurement as **z**.)

The correction step doesn't simply replace the predicted value **x** with the measured value **z**,
instead it (simplistically speaking) uses a weighted mean of the two values.

That mean is weighted according to how confident we are in the values from our sensors. So, for example,
if we know the accelerometer is very noisy and the GPS is highly accurate, we would put more weight on the GPS reading.

The confidence in a value is given by a "noise matrix".
The confidence in control input **u** (accelerometer in our case) is given by the matrix **Q** (called the process noise covariance matrix).
The confidence in a measurement **z**, is given by the matrix **R** (called the measurement noise covariance matrix).
Each sensor will have its matrix **R**. So there will be separate matrices **R gps**, **R barometer**, etc.

## Use of Kalman filter for vehicle control

It would be possible to create a Kalman filter that takes accelerometer and gyroscope values as control inputs
and from those estimates orientation, position and velocity.

However this would be computationally very expensive: in 3D the Kalman filter would need a massive 15x15 covariance matrix,
even in 2D a 10x10 covariance matrix would be needed.

Then the Kalman filter would need to calculate [Jacobians](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
as well as multiplying and inverting these large matrices.

A better approach is to remove the responsibility for calculating the orientation from the Kalman filter
and use a another sensor fusion filter (ie a Mahony or Madgwick) filter to do this.

```text
                    ┌──────────┐  Orientation
 ┌─────┐  Acc/Gyro  | MADGWICK |  Quaternion        ┌────────────┐
 │ IMU ├───────────►│ FILTER   ├───────────────────►| FLIGHT     |
 └─────┘  Acc       └──────────┘                    | CONTROLLER |
     │─────────────────────────────────────────────►|            |
┌────────────┐                                      └────────────┘
│ INTEGRATOR |
└────────────┘
     |              ┌──────────┐  Position Vector
  delta V           | POSITION |  Velocity Vector   ┌───────────┐
     └────────────► │ KALMAN   ├───────────────────►| AUTOPILOT |
                    | FILTER   |                    └───────────┘
                    └──────────┘
                         │
  GPS, Barometer etc ──►─┘
```

This reduces the covariance matrix to more manageable size of 9x9 for 3D (and 6x6 for 2D) and avoids the need to calculate Jacobians.

Note that we integrate the accelerometer output from the IMU to give a `delta_v` value since
the Kalman filter runs at a lower rate than the IMU (the IMU can run at up to 8kHz, the Kalman
filter typically runs at 1kHz or less).

## Kalman Filter Optimization

A naïve implementation of a 3D position Kalman filter is computationally very expensive:

|     | Description                     | Type         | Notes                                       |
| --- | ------------------------------- | ------------ | ------------------------------------------- |
|     | **predict**<br>`x = F·x`<br>`P = F·P·Fᵀ + Q` | |                                             |
| `x` | estimated state vector          | 3 x vector3  | position x,y,z; velocity x,y,z; bias x,y,z  |
| `F` | state transition matrix         | 9 x 9 matrix | derived from system dynamics                |
| `P` | state covariance matrix         | 9 x 9 matrix | estimated uncertainty                       |
| `Q` | process noise covariance matrix | 9 x 9 matrix | how noisy is control vector (eg IMU)        |
|     | **correct**<br>`S = H·P·Hᵀ + R`<br>`K = P·Hᵀ·S⁻¹`<br>`y = z - H·x`<br>`x = x + K·y` | |      |
| `y` | measurement innovation          | 3 x vector3  | difference between measurement and estimate |
| `z` | measurement vector              | vector3      | input                                       |
| `H` | measurement matrix              | 3 x 9 matrix | derived from system dynamics                |
| `R` | measurement noise covariance    | 3 x 3 matrix | how noisy/accurate is sensor (eg GPS)       |
| `S` | innovation covariance matrix    | 9 x 9 matrix |                                             |
| `K` | Kalman gain matrix              | 9 x 3        |                                             |

As you can see the **predict** step requires multiplying 9x9 matrices,
and the **correct** step requires 9x9 matrix multiplication and inversion.

Multiplying two 9x9 matrices requires 729 individual arithmetic operations, inversion even more.

Fortunately, these matrices are sparse and so we can avoid doing the full multiplications and inversions.

Firstly we split the 9x9 matrix **P** into nine 3x3 submatrices:

```text
┌                 9x9                  ┐
│ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
│ │ Position ││ Position ││ Position │ │
│ │ Position ││ Velocity ││ Bias     │ │
│ └          ┘└          ┘└          ┘ │
│ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
│ │ Velocity ││ Velocity ││ Velocity │ │
│ │ Position ││ Velocity ││ Bias     │ │
│ └          ┘└          ┘└          ┘ │
│ ┌   3x3    ┐┌   3x3    ┐┌   3x3    ┐ │
│ │ Bias     ││ Bias     ││ Bias     │ │
│ │ Position ││ Velocity ││ Bias     │ │
│ └          ┘└          ┘└          ┘ │
└                                      ┘
```

Then instead of performing operations on the entire 9x9 matrix, we can perform sub-operations on the submatrices.
So, for example, the calculation`P = F·P·Fᵀ` is transformed into a number of 3x3 matrix multiplications and additions,
rather than two 9x9 matrix multiplications and a transposition.

In the correct phase we do not need to invert the full 9x9 matrix `S`, instead we invert the relevant submatrix.
For a 3D measurement correction (eg position from GPS) this means inverting a 3x3 submatrix.
For a 1D measurement correction (eg altitude from a barometer) this involves simply taking the reciprocal of a scalar.

Further optimizations include:

1. recognizing `R` is diagonal matrix, so multiplication by `R` is simplified.
2. collapsing `Q` from a 9x9 matrix into the two scalars `Q_velocity` and `Q_bias`.
3. don't explicitly create the matrix `H`: rather directly select the relevant submatrices when calculating `H·P·Hᵀ`, `P·Hᵀ·S⁻¹` and `H·x`.

## Madgwick Filter Optimization {#opt}

Classically, the calculation of the Madgwick gradient descent corrective step involves multiplication of a vector by a matrix,
this involves a total of 54 arithmetic operations for the acc/gyro case.

However, because both the matrix and vector contain zero elements, and because there is some symmetry in the matrix,
the calculation can be refactored to use fewer arithmetic operations.

Indeed it can be reduced to a total of 31 arithmetic operations.

I haven't yet benchmarked this Rust implementation,
but the original C++ version of `MadgwickFilter::update_orientation` (equivalent to `madgwick.fuse_acc_gyro`)
took or under 20 microseconds on a 240 MHz ESP32 S3.

The aim is to be able to run sensor fusion a part of a Gyro/PID loop running at 8kHz. That means everything
(including reading the IMU, filtering the output, performing sensor fusion and calculation the motor outputs
using a PID controller) needs to run in 125 microseconds. This is currently looking achievable.

```rust
// Classic version
//
// total:
//      54 arithmetic operations (35 multiplications, 19 additions/subtractions)
//
# use vqm::{Quaternionf32, Vector3f32, Vector4f32, Matrix4x4f32};
fn madgwick_step(q: Quaternionf32, a: Vector3f32) -> Vector4f32 {
    let M = Matrix4x4f32::new([ // 10 multiplications
        -2.0*q.x, 2.0*q.w,      0.0, 0.0,
         2.0*q.y, 2.0*q.z, -4.0*q.w, 0.0,
        -2.0*q.z, 2.0*q.y, -4.0*q.x, 0.0,
         2.0*q.w, 2.0*q.x,      0.0, 0.0
    ]);

    let v = Vector4f32::new( // 9 multiplications, 7 additions/subtractions
        2.0*(      q.w*q.y - q.z*q.x) - a.x,
        2.0*(      q.z*q.w + q.x*q.y) - a.y,
        2.0*(0.5 - q.w*q.w - q.x*q.x) - a.z,
        0.0
    );

    M * v // 16 multiplications, 12 additions
}
```

```rust
// Refactored version
//
// total:
//      31 arithmetic operations (19 multiplications, 12 additions/subtractions)
// when converted to SIMD this becomes:
//      16 operations (7 multiplications, 4 additions, 3 vector multiplications, 2 vector additions)
//
# use vqm::{Quaternionf32, Vector3f32};
fn madgwick_step(q: Quaternionf32, a: Vector3f32) -> Quaternionf32 {
    let wz_common = 2.0 * (q.x * q.x + q.y * q.y); // 3 multiplications, 1 addition
    let xy_common = 2.0 * (q.w * q.w + q.z * q.z - 1.0 + 2.0 * wz_common + a.z); // 4 multiplications, 3 additions/subtractions

    Quaternionf32 { // 12 multiplications, 8 additions/subtractions
        w: q.w * wz_common + q.y * a.x - q.x * a.y,
        x: q.x * xy_common - q.z * a.x - q.w * a.y,
        y: q.y * xy_common + q.w * a.x - q.z * a.y,
        z: q.z * wz_common - q.x * a.x - q.y * a.y,
    }
}
```

## Original implementation

I originally implemented this crate as a C++ library:
[Library-SensorFusion](https://github.com/martinbudden/Library-SensorFusion).

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0)>
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT)>

at your option.
