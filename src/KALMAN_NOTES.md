# Kalman filter notes

Unabbreviated forms of the Kalman equations and explanations.

## Notation

In estimation theory and signal processing, the following notation is used:

The "hat" symbol ( ^ ) placed over a variable denotes an estimate of that variable, rather than its true, exact value.
`x̂` reads as "x-hat".

The subscript `k|k-1` denotes a conditional relationship,
reads as "at time step `k`, given all information up to time step `k-1`".

It highlights the difference between a prediction and an updated estimate:

- The Left Side `k`: the specific point in time you are currently predicting or analyzing.
- The Vertical Bar `|`: means "given" or "conditioned upon" (exactly like conditional probability notation).
- The Right Side `k-1`: the point in time representing the last available measurement data.

| Notation    | Type                        | Plain English Meaning                                                                                    |
| :---------- | :-------------------------- | :------------------------------------------------------------------------------------------------------- |
| `x̂ₖ\|ₖ₋₁` | *A Priori* (Prediction)     | Our best guess for the state at step `k`, based solely on what we knew up to step `k-1`              |
| `x̂ₖ\|ₖ`   | *A Posteriori* (Correction) | Our corrected state estimate at step `k`, now that we have factored in the measurement from step `k` |

- `x̂ₖ|ₖ₋₁` reads as "x-hat of k, given k minus one" or "the a priori state estimate at step k".
- `x̂ₖ|ₖ` reads as "x-hat of k, given k" or "the a posteriori state estimate at step k".

## Predict Equations (Time Update)

| Step | Equation | Description |
| :--- | :--- | :--- |
| **Predicted State** | x̂<sub>k\|k-1</sub> = F<sub>k</sub> x̂<sub>k-1\|k-1</sub> + B<sub>k</sub> u<sub>k</sub> | Projects the state ahead |
| **Predicted Covariance** | P<sub>k\|k-1</sub> = F<sub>k</sub> P<sub>k-1\|k-1</sub> F<sub>k</sub><sup>T</sup> + Q<sub>k</sub> | Projects the error covariance ahead |

## Correct Equations (Measurement Update)

| Step | Equation | Description |
| :--- | :--- | :--- |
| **Innovation** | ỹ<sub>k</sub> = z<sub>k</sub> - H<sub>k</sub> x̂<sub>k\|k-1</sub> | Computes the measurement innovation |
| **Innovation Covariance** | S<sub>k</sub> = H<sub>k</sub> P<sub>k\|k-1</sub> H<sub>k</sub><sup>T</sup> + R<sub>k</sub> | Computes the innovation covariance |
| **Kalman Gain** | K<sub>k</sub> = P<sub>k\|k-1</sub> H<sub>k</sub><sup>T</sup> S<sub>k</sub><sup>-1</sup> | Computes the optimal gain matrix |
| **Updated State** | x̂<sub>k\|k</sub> = x̂<sub>k\|k-1</sub> + K<sub>k</sub> ỹ<sub>k</sub> | Updates state estimate with measurement |
| **Updated Covariance** | P<sub>k\|k</sub> = (I - K<sub>k</sub> H<sub>k</sub>) P<sub>k\|k-1</sub> | Updates the error covariance matrix |

## Variable Definitions

- `u`: Control vector
- `x̂`: Estimated state vector
- `z`: Measurement vector
- `y`: Measurement innovation

- `P`: State covariance matrix (estimated uncertainty)
- `F`: State transition matrix (derived from system dynamics)
- `B`: Control input matrix
- `Q`: Process noise covariance matrix
- `H`: for LFK Measurement matrix; for EKF Jacobian matrix, ∂h(x)/∂x, evaluated at the current state estimate.
- `R`: Measurement noise covariance matrix
- `K`: Kalman gain matrix
- `I`: Identity matrix

## Matrix representations

```text
x = [x, y, z, vx, vy, vz, bx, by, bz]

for acceleration as u
    ┌            ┐
    │  ½·dt²·I₃  │
B = │    dt·I₃   │
    │     0₃     │
    └            ┘

for delta_v as u
    ┌           ┐
    │  ½·dt·I₃  │
B = │    I₃     │
    │    0₃     │
    └           ┘

for delta_v as u:

    ┌                                           ┐
    │  1  0  0   dt   0   0   -½dt     0     0  |
    │  0  1  0    0  dt   0      0  -½dt     0  |
    │  0  0  1    0   0  dt      0     0  -½dt  |
    |                                           |
    │  0  0  0    1   0   0    -dt     0     0  |
F = │  0  0  0    0   1   0      0   -dt     0  |
    │  0  0  0    0   0   1      0     0   -dt  |
    |                                           |
    │  0  0  0    0   0   0      1     0     0   |
    │  0  0  0    0   0   0      0     1     0  |
    │  0  0  0    0   0   0      0     0     1  |
    └                                           ┘
    ┌                            ┐
    │  I₃     dt·I₃   -½·dt²·I₃  │
F = │  0₃       I₃      -dt·I₃   │
    │  0₃       0₃         I₃    │
    └                            ┘
```

```text
H = ∂h(x)/∂x

    [ ∂h₁/∂x₁  ∂h₁/∂x₂  …  ∂h₁/∂xₙ ]
    [ ∂h₂/∂x₁  ∂h₂/∂x₂  …  ∂h₂/∂xₙ ]
H = [    ⋮        ⋮     ⋱     ⋮    ]
    [ ∂hₘ/∂x₁  ∂hₘ/∂x₂  …  ∂hₘ/∂xₙ ]
```

## Using acc and gyro

```text
    [   I₃      I₃·Δt       0₃         ]
F = [   0₃        I₃     -R(q)·[a]ₓ·Δt ]
    [   0₃        0₃     I₃ - [ω]ₓ·Δt  ]

        [   0   -a_z   a_y ]
[a]ₓ = [  a_z    0   -a_x ]
        [ -a_y   a_x    0  ]

[ω]ₓ: The 3×3 skew-symmetric cross-product matrix of your gyroscope reading [ωₓ, ]

         [   0   -ω₟   ω₞ ]
  [ω]ₓ = [  ω₟    0   -ωₓ ]
         [ -ω₞   ωₓ    0  ]

        [   0   -ω_z   ω_y ]
[ω]ₓ = [  ω_z    0   -ω_x ]
        [ -ω_y   ω_x    0  ]
```

## Using acc and gyro with bias

```text
    [   I₃      I₃·Δt          0₃            0₃          0₃    ]  <- δpos
    [   0₃        I₃     -R(q)·[a-bₐ]ₓ·Δt  -R(q)·Δt      0₃    ]  <- δvel
F = [   0₃        0₃       I₃ - [ω-b_g]ₓ·Δt  0₃       -I₃·Δt   ]  <- δθ
    [   0₃        0₃           0₃            I₃          0₃    ]  <- δbₐ
    [   0₃        0₃           0₃            0₃          I₃    ]  <- δb_g
```

1. Bias Compensation: The skew-symmetric matrices for acceleration and angular velocity now subtract the current estimated biases:
   [a - bₐ]ₓ and [ω - b_g]ₓ.
2. Velocity Drift: The term -R(q)·Δt maps how an error in your accelerometer bias directly causes an error in your velocity.
3. Attitude Drift: The term -I₃·Δt maps how an error in your gyroscope bias directly causes your orientation to drift over time.
4. Bias Stability: The bottom right quadrant contains I₃ blocks, modeling the biases as a random walk
   (their change from step to step is purely driven by noise, not the state itself).

In an IMU-driven filter, the process noise is injected through four distinct physical sources:

- σ_v²: Accelerometer measurement noise (Velocity random walk) [2]
- σ_θ²: Gyroscope measurement noise (Angular random walk) [2]
- σ_ba²: Accelerometer bias instability noise (Bias random walk) [2]
- σ_bg²: Gyroscope bias instability noise (Bias random walk) [2]

```text
    [ Q_pos    Q_pv       0₃        0₃        0₃    ]  <- δpos
    [ Q_pvᵀ    Q_vel      0₃        0₃        0₃    ]  <- δvel
Q = [   0₃      0₃      Q_att       0₃        0₃    ]  <- δθ
    [   0₃      0₃        0₃      Q_b_acc     0₃    ]  <- δbₐ
    [   0₃      0₃        0₃        0₃      Q_b_gyr ]  <- δb_g
```

Because the orientation changes dynamically, the velocity noise sub-blocks must be rotated by your current attitude rotation matrix
 R(q) so that the noise aligns with the global frame:

- Position Noise:
  Q_pos = (1/3) · I₃ · σ_v² · Δt³
- Velocity Noise:
  Q_vel = I₃ · σ_v² · Δt
- Cross Position-Velocity Noise:
  Q_pv = (1/2) · I₃ · σ_v² · Δt²
- Attitude Noise:
  Q_att = I₃ · σ_θ² · Δt
- Accelerometer Bias Noise:
  Q_b_acc = I₃ · σ_ba² · Δt
- Gyroscope Bias Noise:
  Q_b_gyr = I₃ · σ_bg² · Δt
