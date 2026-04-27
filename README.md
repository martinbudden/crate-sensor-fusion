# `sensor-fusion` Rust Crate ![license](https://img.shields.io/badge/license-MIT-green) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![open source](https://badgen.net/badge/open/source/blue?icon=github)

This crate contains [sensor fusion](https://en.wikipedia.org/wiki/Sensor_fusion) algorithms to combine
output from a gyroscope, accelerometer, and optionally a magnetometer to give output that has less uncertainty
than the output of the individual sensors.

Three sensor fusion implementations are available:

1. Complementary Filter
2. Mahony Filter
3. Madgwick Filter

The Madgwick filter has been refactored to be more computationally efficient (and so faster) than
the standard version used in many implementations, see [Optimization](#opt) below.

## Simple example

Here's a simple example that calculates the orientation by fusing accelerometer and gyro values:

```rust
use sensor_fusion::{MadgwickFilterf32, SensorFusion};
use vqm::Vector3df32;

fn main() {
    let mut madgwick = MadgwickFilterf32::new();
    let dt = 0.001; // 1 millisecond

    // Mock sensor values, normally read from IMU
    let gyro_dps = Vector3df32::new(100.0, 20.0, 10.0);
    let acc = Vector3df32::new(0.1, 0.2, 0.9);

    // Fuse acc and gyro values, after converting gyro to radians/s
    let orientation = madgwick.fuse_acc_gyro(acc, gyro_dps.to_radians(), dt);

    let roll = orientation.calculate_roll_degrees();
    let pitch = orientation.calculate_pitch_degrees();
    let yaw = orientation.calculate_yaw_degrees();

    // Print out the Euler angles.
    println!("pitch={}, roll={}, yaw={}", pitch, roll, yaw);
}
```

The Madgwick filter also supports three-way fusing of accelerometer, gyroscope, and magnetometer readings:

```rust
# use vqm::{Quaternionf32, Vector3df32};
# use crate::sensor_fusion::{MadgwickFilterf32,SensorFusion};
# let mut madgwick = MadgwickFilterf32::default();
# let dt: f32 = 0.001;
# let acc = Vector3df32::default();
# let gyro_rps = Vector3df32::default();
# let mag = Vector3df32::default();
let orientation = madgwick.fuse_acc_gyro_mag(acc, gyro_rps, mag, dt);
```

## Mahony filter

The Mahony filter has the same interface as the Madgwick filter:

```rust
# use vqm::{Quaternionf32, Vector3df32};
use crate::sensor_fusion::{MahonyFilterf32,SensorFusion};

let mut mahony = MahonyFilterf32::default();
# let dt: f32 = 0.001;
# let acc = Vector3df32::default();
# let gyro_rps = Vector3df32::default();

let orientation = mahony.fuse_acc_gyro(acc, gyro_rps, dt);
```

The Mahony filter does not support 3-way fusion using a magnetometer.

## Method call interface

The `FuseAccGyro` and `FuseAccGyroMag` traits allow method-call syntax to be used:

```rust
# use vqm::{Quaternionf32, Vector3df32};
use crate::sensor_fusion::{FuseAccGyro,FuseAccGyroMag,MadgwickFilterf32};

let mut madgwick = MadgwickFilterf32::default();
# let dt: f32 = 0.001;
# let acc = Vector3df32::default();
# let gyro_rps = Vector3df32::default();
# let mag = Vector3df32::default();

let orientation = (acc, gyro_rps).fuse_acc_gyro_using(&mut madgwick, dt);
// or
let orientation = (acc, gyro_rps, mag).fuse_acc_gyro_mag_using(&mut madgwick, dt);
```

## SIMD support

**SIMD** support (for the `f32` variants) can be enabled with the `simd` feature.

It is currently experimental, so if you used SIMD make sure you benchmark to show that you are indeed getting
a performance improvement over the non-SIMD version.

This uses [portable simd](https://doc.rust-lang.org/core/simd/index.html), which requires the nightly compiler, since it is still
unstable in rust.

This can be invoked using `rustup`, eg:

```sh
rustup run nightly cargo build --features simd --target thumbv8m.main-none-eabi
```

## Optimization {#opt}

Classically, the calculation of the Madgwick gradient descent corrective step involves multiplication of a vector by a matrix,
this involves a total of 54 arithmetic operations for the acc/gyro case.

However, because both the matrix and vector contain zero elements, and because there is some symmetry in the matrix,
the calculation can be refactored to use fewer arithmetic operations.

Indeed it can be reduced to a total of 31 arithmetic operations.
By using SIMD this can be further reduced to 16 operations (11 scalar and 5 SIMD operations).

See below:

```text
// Classic version
//
// total:
//      54 arithmetic operations (35 multiplications, 19 additions/subtractions)
//
fn madgwick_step(q: Quaternionf32, a: Vector3df32) -> Quaternionf32 {
    let M = Matrix4x4f32::new( // 10 multiplications
        -2.0*q.x, 2.0*q.w,      0.0, 0.0,
         2.0*q.y, 2.0*q.z, -4.0*q.w, 0.0,
        -2.0*q.z, 2.0*q.y, -4.0*q.x, 0.0,
         2.0*q.w, 2.0*q.x,      0.0, 0.0
    );

    let v = Vector4df32::new( // 9 multiplications, 7 additions/subtractions
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
# use vqm::{Quaternionf32, Vector3df32};
fn madgwick_step(q: Quaternionf32, a: Vector3df32) -> Quaternionf32 {
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
