# `sensor-fusion` Rust Crate<br>[![Crates.io](https://img.shields.io/crates/v/sensor-fusion.svg)](https://crates.io/crates/sensor-fusion) [![Documentation](https://docs.rs/sensor-fusion/badge.svg)](https://docs.rs/sensor-fusion) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![open source](https://badgen.net/badge/open/source/blue?icon=github)

**sensor-fusion** is a lightweight, allocation-free, Rust math library for [sensor fusion](https://en.wikipedia.org/wiki/Sensor_fusion).
It is designed for embedded systems, robotics, and real-time applications.

Sensor fusion algorithms to combine output from a gyroscope, accelerometer, and other sensors to give output that has less uncertainty
than the output of the individual sensors.

This crate is `no_std`, `no alloc`, `no panic`, `no unwrap`, `no expect`, and the Minimum Supported Rust Version (MSRV) is `Rust 1.89`.

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

## Navigation Kalman filters

Three Navigation Kalman filters are provided:

1. `KalmanFilterZ` - a filter to estimate altitude and vertical speed.
2. `KalmanFilterXy` - a filter to estimate 2D position and velocity.
3. `KalmanFilterXyz` - a filter to estimate 3D position and velocity.

These are not general purpose Kalman filters (where you provide the `F`, `B`, and `H` matrices according to your problem domain).
Rather these are Kalman filters specific to and optimized for the Navigation domain - that is they are for predicting position and velocity
given inputs from relevant sensors (ie IMU, GPS, barometer etc).

### 3D Kalman filter

When using a Kalman filter for 3D there are two possible implementation choices:

1. Use a `KalmanFilterXyz`.
2. Use a spilt architecture: use a `KalmanFilterXy` for horizontal position, and a `KalmanFilterZ` for vertical position.

| Feature            | Split Architecture (1D + 2D)                                     | Unified 3D Architecture                             |
| ------------------ | ---------------------------------------------------------------- | --------------------------------------------------- |
| Computational Cost | Lower (2×2 and 6×6 matrices)                                     | Higher (6×6 and 9×9 matrices)                       |
| Sensor Coupling    | Independent (barometer noise cannot corrupt horizontal accuracy) | Coupled (vertical errors can bleed into horizontal) |
| Tuning Complexity  | Easier (can tune vertical and horizontal separately)             | Harder                                              |
| Fault Tolerance    | Higher                                                           | Lower                                               |
| Banked movement    | Produces inaccuracies                                            | Handled by model                                    |

For vehicles that stay upright, or tilt minimally (eg a walker, a car, or a cinematic quadcopter) the split approach is a good one.

For vehicles such as acrobatic quadcopters or rockets, where pitching and rolling movements are common, the combined approach is better.

### Position Kalman filter overview

A Kalman filter works by predicting an object's state using a physical model of the object and a control input.

So, for example, using acceleration (`a`) as the control input it would predict an object's position and velocity using kinematic equations:

```text
s = ut + ½at²
v = u + at
```

However, if you keep predicting position and velocity using integration like that, noise in the system
will cause the predicted values to drift away from the actual values.

A Kalman filter periodically corrects this prediction with an actual measurement, for example a
position from a GPS, or an altitude from a barometer.

A Kalman filter can accept measurements from multiple sources to make its corrections,
for example: a GPS, a barometer, a rangefinder, and an optical flow device.

(Note: The control input is normally denoted as `u`, the predicted measurement as `x`, and the actual measurement as `z`.)

The correction step doesn't simply replace the predicted value `x` with the measured value `z`,
instead it (simplistically speaking) uses a weighted mean of the two values.

That mean is weighted according to how confident we are in the values from our sensors. So, for example,
if we know the accelerometer is very noisy and the GPS is highly accurate, we would put more weight on the GPS reading.

The confidence in a value is given by a "noise matrix".
The confidence in control input `u` (accelerometer in our case) is given by the matrix `Q` (called the process noise covariance matrix).
The confidence in a measurement `z`, is given by the matrix `R` (called the measurement noise covariance matrix).
Each sensor will have its matrix `R`. So there will be separate matrices `R_gps`, `R_barometer`, etc.

### Use of Kalman filter for vehicle control

It is possible to create a Kalman filter that takes accelerometer and gyroscope values as control inputs
and from those estimates orientation, position and velocity.

However this is computationally very expensive: in 3D the Kalman filter would need massive 15x15
state transition (`F`) and state covariance (`P`) matrices.
(Position, velocity, attitude, accelerometer bias, and gyroscope bias: each of those being 3D vectors.)
Even in 2D, 10x10 matrices are needed.

Since the calculation of attitude is non-linear, the Kalman filter needs to calculate
the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
of the state transition matrix
`F = ∂f(x,u )/∂x`
as well as multiplying and inverting these large matrices.

A computationally less expensive approach is to remove the responsibility for calculating the orientation from
the Kalman filter and use a another sensor fusion filter (ie a Mahony or Madgwick) filter to do this:

```text
╭─────╮  ┌────────┐            Accₛ/Gyroₛ          ┌────────────┐
│     │  │ SIGNAL ├────────────────▶───────────────┤   FLIGHT   │
│ Imu ├─▶┤ FILTER │ Accₛ  ┌──────────┐             │ CONTROLLER │
│     │  │        ├───▶───┤ MADGWICK ├─────▶───────┤   (PIDS)   │
╰──┬──╯  └────────┘ Gyroₛ │  FILTER  │ Orientation └────────────┘
   │                      └─┬──┬───┬─┘
   ▼ Accₛ(unfiltered)       │  │   │ 
   │                        │  │   │ Yaw ╭──────────────╮
   │            Orientation ▼  ▲   └──◀──┤ Magnetometer │
┌──┴─────────┐              │  │         ╰──────────────╯
│    IMU     ├─────◀────────┘  │
│ INTEGRATOR │                 │ Yaw(calculated)
└──┬─────────┘     Positionₑ ╭─┴───╮
   │                    ┌──◀─┤ Gps │
 delta_posₑ             │    ╰─────╯
 delta_velₑ             │
   │                ┌───┴────┐       Positionₑ      ┌───────────┐
   └──────▶─────────┤ KALMAN ├────────────▶─────────┤ AUTOPILOT │
                    │ FILTER │       Velocityₑ      └───────────┘
                    └────┬───┘
                         │
       Barometer, etc ─▶─┘

╭───────╮
│ Boxes │ with rounded corners and mixed case letters represent hardware.
╰───────╯
The subscript ₛ denotes a value in the sensor frame (body frame).
The subscript ₑ denotes a value in the earth frame (inertial frame).
```

This reduces the covariance matrix to more manageable size of 9x9 for 3D (and 6x6 for 2D) and avoids the need
to calculate the Jacobian of the state transition matrix (`F`).

In this setup:

The Madgwick filter is entirely responsible for the calculation of orientation.
Since an accelerometer cannot determine absolute yaw, and gyro bias introduces yaw drift,
the Madgwick filter needs yaw correction inputs to maintain a correct yaw orientation.
These corrections can come from two sources:

1. a magnetometer for an absolute yaw value
2. when in motion, the yaw value can be determined from the GPS velocity.

The IMU integrator serves two purposes:

1. It uses the orientation from the Madgwick filter to convert the acceleration `Accₛ` from the sensor
   frame to the earth frame (`Accₑ`).
2. It integrates this `Accₑ` to give `delta_posₑ` and `delta_velₑ` values.
   These are needed because the Kalman filter runs at a lower rate than the IMU
   (the IMU can run at up to 8kHz, the Kalman filter typically runs at 500Hz or less).

### Kalman Filter Optimization

A naïve implementation of a 3D position Kalman filter is computationally very expensive:

|     | Description                     | Type         | Notes                                       |
| --- | ------------------------------- | ------------ | ------------------------------------------- |
|     | **predict**<br>`x = F·x + B·u`<br>`P = F·P·Fᵀ + Q` | |                                       |
| `u` | control input vector            | vector3      | acceleration x,y,z (from IMU)               |
| `x` | estimated state vector          | 3 x vector3  | position x,y,z; velocity x,y,z; bias x,y,z  |
| `F` | state transition matrix         | 9 x 9 matrix | derived from system dynamics                |
| `B` | control input matrix            | 9 x 3 matrix | derived from system dynamics                |
| `P` | state covariance matrix         | 9 x 9 matrix | estimated uncertainty (calculated)          |
| `Q` | process noise covariance matrix | 9 x 9 matrix | how noisy is `u`. obtained from IMU datasheet and/or measurement |
|     | **correct**<br>`y = z - H·x`<br>`S = H·P·Hᵀ + R`<br>`K = P·Hᵀ·S⁻¹`<br>`x = x + K·y`<br>`P = P - K·H·P` | | |
| `z` | measurement vector              | vector3      | input (eg from GPS)                         |
| `y` | measurement innovation          | 3 x vector3  | difference between measurement and estimate |
| `H` | measurement matrix              | 3 x 9 matrix | derived from system dynamics                |
| `R` | measurement noise covariance    | 3 x 3 matrix | how noisy/accurate is sensor (eg GPS). obtained from datasheet |
| `S` | innovation covariance matrix    | 9 x 9 matrix | calculated                                  |
| `K` | Kalman gain matrix              | 9 x 3        | calculated                                  |

As you can see the **predict** step requires multiplying 9x9 matrices,
and the **correct** step requires 9x9 matrix multiplication and inversion.

9×9 matrix multiplication requires 729 scalar multiplications and 648 additions, inversion even more.

Fortunately, these matrices are sparse and so we can avoid doing the full multiplications and inversions.

For a 3D position Kalman filter with acceleration as the control input, we have:

```text
    ┌                                           ┐
    │  1  0  0   dt   0   0   -½dt²    0     0  |
    │  0  1  0    0  dt   0      0  -½dt²    0  |
    │  0  0  1    0   0  dt      0     0  -½dt² |
    |                                           |
    │  0  0  0    1   0   0    -dt     0     0  |
F = │  0  0  0    0   1   0      0   -dt     0  |
    │  0  0  0    0   0   1      0     0   -dt  |
    |                                           |
    │  0  0  0    0   0   0      1     0     0  |
    │  0  0  0    0   0   0      0     1     0  |
    │  0  0  0    0   0   0      0     0     1  |
    └                                           ┘
```

Calculating `P = F·P·Fᵀ` directly by matrix multiplication and transposition is clearly extremely wasteful.

So, instead of performing operations on the entire 9x9 matrix, we can split the 9x9 matrix `P` into nine 3x3 submatrices:

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

The calculation `P = F·P·Fᵀ` in the predict phase becomes a sequence of operations on the 3x3 submatrices.

In the correct phase we also use these submatrices.
In particular we do not need to invert the full 9x9 matrix `S`, instead we invert the relevant submatrix.

- for a 3D measurement correction (eg position from GPS) this means inverting a 3x3 submatrix.
- for a 1D measurement correction (eg altitude from a barometer) this involves simply taking the reciprocal of a scalar.

Further optimizations include:

1. recognizing `R` is diagonal matrix, so multiplication by `R` is simplified.
2. collapsing `Q` from a 9x9 matrix into the two scalars `Q_velocity` and `Q_bias`.
3. don't explicitly create the matrix `H`: rather directly select the relevant submatrices when calculating `H·P·Hᵀ`, `P·Hᵀ·S⁻¹` and `H·x`.

### Handling corrections using delayed GPS measurements

GPS measurements are inherently delayed. This delay has a number of causes including the time required for
signal processing by the GPS receiver and the time to transmit the GPS solution over the UART.

This means the Kalman filter state needs to be corrected at the time the GPS measurement was taken, rather than
the time that it was received. There are a number of ways to handle this, including:

1. Rewind and fast-forward: Maintain a history of the filter states and inputs, and when a GPS measurement is received, rewind it to the
   time of the GPS measurement, apply the GPS correction, and the fast-forward the measurements to the current time.
2. Delay compensation: Compensate for the delay by using the GPS velocity to predict the historical position forward to the current time.

In the current release `0.0.14` delay compensation is implemented and rewind and fast-forward is under development.
The aim is to eventually support both options.

### Testing optimizations against reference implementation

`KalmanFilterXyz` has been tested against a reference implementation written using 9x9 matrices, `nalgebra`, and `f64`.
See: [reference test](tests/kalman_filter_reference.rs).

### Kalman filter numerical stability

The Kalman filter equations themselves are mathematically well behaved, however errors can accumulate when implemented with `f32` precision.
To mitigate these errors this implementation:

1. Explicitly restores symmetry to the covariance matrix after each state prediction (by setting `P = ½(P + Pᵀ)`).
2. Uses the numerically stable Joseph form for covariance predictions ( `P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ`
   rather than `P = P - K·H·P + K·R·Kᵀ`)

Additionally scaling the states and measurements so their scales are broadly comparable improves numerical stability.
Scaling brings state values and uncertainties to closer to O(1) (order one), improving numerical conditioning
without changing the underlying physical problem.
(Floating-point arithmetic may lose precision when operations combine quantities with very different scales.)

Initial tests indicate the filter is numerically stable without scaling, but scaling may, nevertheless, be added in a future
release.

### Reference

See [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/tree/master)
for a really good introductory text for Kalman and Bayesian filters.
All code is written in Python, and the book itself is written using Jupyter Notebook so that you can run and modify the code in your browser.

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

- Apache License, Version 2.0, see: [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
- MIT license, see: [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.
