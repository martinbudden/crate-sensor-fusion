# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Releases of the form `0.1.n` do not adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
that is each release may contain incompatible API changes.

Once the API has stabilized this project will adopt semantic versioning, the first release to do so will be `0.2.0`.

## [Possible_future]

### Added

- May port  **VQF** (Versatile Quaternion Filter) from my C++ library [Library-SensorFusion](https://github.com/martinbudden/Library-SensorFusion).

## [0.1.13] - 2026-10-28

### Changed

- Updated to `vqm` version `0.1.19`.

## [0.1.12] - 2026-10-23

### Added

- `KalmanFilterXyzHistory` struct.
- `_ned` and `_body` qualifiers to state vectors.
- `-mps2` suffix for accelerations.
- `-mps` suffix for velocities.
- `retire!` macro.
- `ImuIntegrator`

### Changed

- Updated to `vqm` version `0.1.18`.
- Renamed `KalmanFilterXY` and related to `KalmanFilterXy`.
- Renamed `KalmanFilterXYZ` and related to `KalmanFilterXyz`.
- Renamed `correct_position_standard_form` to `correct_position_short_form`.
- Moved Kalman filter tests into tests directory.
- Optimized `predict_covariance`.
- Changed trilateration to use Cholesky solver.
- Tidied accessor functions.

### Removed

- fast-forward from `KalmanFilterXYZWithSensors`.
- `kalman_state_vector9.rs`.

## [0.1.11] - 2026-10-01

### Added

- support for continuous integration.

### Changed

- Updated to Rust version 1.89.
- Improved `with` form of `MadgwickFilter` constructors.

## [0.1.10] - 2026-09-05

### Added

- first cut of Delayed State Buffer.

### Changed

- Renamed `AltitudeKalmanFilter` to `KalmanFilterX`
- Renamed `AltitudeKalmanFilterExtended` to `KalmanFilterXWithR`
- Renamed `PositionKalmanFilter` to `KalmanFilterXYZ`
- Renamed `PositionKalmanFilterExtended` to `KalmanFilterXYZWithR`
- Updated to `vqm` version `0.1.16`.

### Removed

- `KalmanStateVector9`.

### Fixed

### Security

## [0.1.9] - 2026-09-04

### Changed

- Updated to `vqm` version `0.1.14`.

## [0.1.8] - 2026-07-04

### Added

- `KalmanStateVector9`.
- `trilaterate_2d` and `trilaterate_3d_weighted` functions.
- `#[must_use] attributes where appropriate.
- `deny`s in `lib.rs`.

### Changed

- Updated to `vqm` version `0.1.13`.
- `AltitudeKalmanFilter` changed to use generics.

## [0.1.7] - 2026-06-08

### Added

- Correct using rangefinder and using barometer functions to `AltitudeKalmanFilter`.
- Correct using barometer and using gps to `PositionKalmanFilter`.
- `correct_yaw` to Madgwick filter.

### Changed

- Updated to `vqm` version `0.1.12`.
- Changed AltitudeKalman filter to have predict and correct functions.
- Renamed update to correct for `PositionKalmanFilter`.
- Code tidy.

## [0.1.6] - 2026-05-24

### Changed

- Updated to `vqm` version `0.1.9`.
- Improved efficiency of Kalman filters.
- Improved Kalman documentation.

## [0.1.5] - 2026-05-16

### Added

- `AltitudeKalmanFilter`
- `PositionKalmanFilter`

### Removed

- `katex-header.html`

## [0.1.4] - 2026-05-16

### Changed

- Updated to `vqm` version `0.1.5`.

## [0.1.3] - 2026-05-06

### Changed

- Updated to `vqm` version `0.1.3`.

## [0.1.2] - 2026-05-02

### Changed

- Improved README.md.
- Updated to `vqm` version `0.1.2`.

## [0.1.1] - 2026-04-27

### Added

- `madgwick.rs` in `examples` directory.
- This changelog.
- CONTRIBUTING.md

### Changed

- Improved README.md.
- Improved documentation.
- Updated to `vqm` version `0.1.1`.

## [0.1.0] - 2026-04-12

Initial release.
