# Kinematic State-Space Sensor Fusion

This crate implements a high-performance, asynchronous, 9-state 3D Position Kalman Filter
optimized for embedded systems (`no_std` and no alloc). It targets hardware platforms like the RP2040 and
RP2350 (Cortex-M33).

## 1. The State Vector (x)

The filter tracks a 9-dimensional kinematic state vector split into three 3D spatial domains:

```text
         ⎡ Position  (x, y, z) ⎤
     x = ⎢ Velocity  (x, y, z) ⎥
         ⎣ Acc bias  (x, y, z) ⎦
```

In code, this mapping is managed via zero-cost abstractions by destructuring `Vector3f32` structures.

## 2. System Architecture: Asynchronous Multi-Sensor Execution Loop

In flight controller firmware, sensors execute at different rates (asynchronous execution).
This filter splits the prediction and correction cycles into modular functions:

```text
┌──────────────────────────────────────────────────────────────┐
│  IMU INTERRUPT TIMELOOP (~1000 Hz)                           │
│  1. Read Accel ──► predict_states()      [State Propagation] │
│  2. Execute    ──► predict_covariance()  [P = F·E·Fᵀ + Q]    │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
    Asynchronous Data Packets Available?
    ├── YES (Baro SPI @ ~40Hz)  ──► correct_barometer(z)  [S = P₂₂ + R]
    └── YES (GPS UART @ ~5Hz)   ──► correct_gps(x, y, z)  [S = H·P·Hᵀ + R]
```

## 3. Architectural Rule: The Accelerometer is NOT a Measurement Update

A common point of confusion is searching for an `update_accelerometer` function.
In a kinematic navigation filter, the accelerometer acts as the **Control Input (`u`)**
during the time-propagation step, pushing the physics model forward. Absolute reference sensors
(**GPS, Barometer**) act as **Measurements (`z`)** to strip away the integration drift.

## 4. Matrix Transformations

### Covariance Time Propagation: `P = F·E·Fᵀ + Q`

```text
    ⎡  1    0   -dt    ⎤          ⎡ -dt²·q_vel    0          0      ⎤
F = ⎢ dt    1     0    ⎥      Q = ⎢      0        0          0      ⎥
    ⎣  0    0   1+β·dt ⎦          ⎣      0        0      dt²·q_bias ⎦
```

### Multi-Dimensional GPS Innovation Matrix: `S = H·P·Hᵀ + R`

```text
                 ⎡ P₁₁  P₁₂  P₁₃ ⎤   ⎡ R_horiz     0         0   ⎤
S = H·P·Hᵀ + R = ⎢ P₂₁  P₂₂  P₂₃ ⎥ + ⎢    0     R_horiz      0   ⎥
                 ⎣ P₃₁  P₃₂  P₃₃ ⎦   ⎣    0        0      R_vert ⎦
```

---

## 🛠️ API & Core Operator Mechanics

1. **Pass-by-Value API:** Data structures arrive via registers (`FPU s0-s31`), bypassing stack pointer overhead.
2. **Analytic Inversion:** Multi-dimensional GPS steps leverage a zero-loop 3x3 determinant inverse (Cramer's Rule).
3. **Non-Snake Case:** Methods use `#[allow(non_snake_case)]` to match textbook terminology.

## See also

<https://thekalmanfilter.com/kalman-filter-explained-simply/>

|     | Description                     | Type         | Notes                                       |
| :-- | :------------------------------ | :----------- | :------------------------------------------ |
|     | **predict**<br>`x = F·x + B·u`<br>`P = F·P·Fᵀ + Q` | |                                       |
| `x` | estimated state vector          | n x 1 vector | Output                                      |
| `F` | state transition matrix         | n x n matrix | derived from system dynamics                |
| `P` | state covariance matrix         | n x n matrix | estimated uncertainty                       |
| `Q` | process noise covariance matrix | n x n matrix | how noisy is control vector (eg IMU)        |
|     | **correct**<br>`S = H·P·Hᵀ + R`<br>`K = P·Hᵀ·S⁻¹`<br>`y = z - H·x`<br>`x = x + K·y` | |      |
| `y` | measurement innovation          | n x 1 vector | difference between measurement and estimate |
| `z` | measurement vector              | m x 1 vector | input                                       |
| `H` | measurement matrix              | m x n matrix | derived from system dynamics                |
| `R` | measurement noise covariance    | m x m matrix | how noisy/accurate is sensor (eg GPS)       |
| `S` | innovation covariance matrix    | n x n matrix |                                             |
| `K` | Kalman gain matrix              | n x m        |                                             |

Alternate terms used in the literature and implementations

`F` - sometimes represented as `A`
`H` - observation matrix
`y` - pre-fit residual (or simply residual), error, observation vector

- innovation (or pre_fit_residual): the measurement minus the prediction (\(z_k - H\hat{x}_{k\vert{}k-1}\)).
- residual (or post_fit_residual): the measurement minus the corrected state (\(z_k - H\hat{x}_{k\vert{}k}\)).

For a 1D KalmanFilter n = 3, m = 1,
For a 2D KalmanFilter n = 6, m = 2,
For a 3D KalmanFilter n = 9, m = 3,

Note n = 3*m, since we have {pos, velocity, acc_bias} for each dimension.

|     | type         | 1D  | 2D  | 2D opt    | 3D  | 3D opt    | explanation |
| :-- | :----------- | :-- | :-- | :---------| :-- | :-------- | :---------- |
|     | **predict**  |     |     |           |     |           |             |
| `x` | n x 1 vector | 1x1 | 2x1 | V2        | 3x1 | V3        |             |
| `F` | n x n matrix | 3x3 | 6x6 | M3x3xM2x2 | 9x9 | M3x3xM3x3 | sparse      |
| `P` | n x n matrix | 3x3 | 6x6 | M3x3xM2x2 | 9x9 | M3x3xM3x3 | sparse      |
| `Q` | n x n matrix | 3x3 | 6x6 | 2x1       | 9x9 | 2x1       | vel, bias   |
|     | **correct**  |     |     |           |     |           |             |
| `y` | n x 1 vector | 1x1 | 2x1 | V2        | 3x1 | V3        |             |
| `z` | m x 1 vector | 1x1 | 2x1 | V2        | 3x1 | V3        |             |
| `H` | m x n matrix | 1x3 | 2x6 | virtual   | 3x9 | virtual   | sparse      |
| `R` | m x m matrix | 1x1 | 2x2 | 2x1       | 3x3 | V3        | diagonal    |
| `S` | n x n matrix | 3x3 | 6x6 | M2x2      | 9x9 | M3x3      |             |
| `K` | n x m matrix | 3x1 | 6x2 | 3xM2x2    | 9x3 | 3xM3x3    |             |

In the optimized form the following is used
For 3D:

- `P` and `F` are `Matrix3x3xM3x3` rather than `Matrix9x9` and are treated as sparse matrices for calculation.
- `R` is `Vector3` rather than `Matrix3x3`.
- `Q` is the two scalars `Q_velocity` and `Q_bias` rather than `Matrix9x9`.

`S` is `Matrix3x3` rather than `Matrix9x9`, since `S` must be inverted this is a considerable optimization.
`H` is never explicitly created, but 3x3 submatrices of `H` are used in calculations.
`K` is implemented as three `Matrix3x3`s.

### Predict Equations (Time Update)

| Step                     | Equation         | Description                         |
| :----------------------- | :--------------- | :---------------------------------- |
| **Predicted State**      | `x = F·x + B·u`  | Projects the state estimate ahead   |
| **Predicted Covariance** | `P = F·P·Fᵀ + Q` | Projects the error covariance ahead |

### Correct Equations (Measurement Update)

| Step                      | Equation          | Description                               |
| :------------------------ | :---------------- | :---------------------------------------- |
| **Innovation**            | `y = z - H·x`     | Calculates the measurement residual       |
| **Innovation Covariance** | `S = H·P·Hᵀ + R`  | Calculates the residual covariance        |
| **Kalman Gain**           | `K = P·Hᵀ·S⁻¹`    | Calculates the optimal gain matrix        |
| **Updated State**         | `x = x + K·y`     | Updates state estimate with measurement   |
| **Updated Covariance**    | `P = (I - K·H)·P` | Updates the error covariance matrix       |

The measurement residual is the difference between the actual sensor reading and what your filter expected the sensor to read.

- The Equation: `y = z - H·x`
- In Plain English: Residual = (Actual Sensor Measurement) - (Predicted Sensor Measurement)
- If the residual is zero, the sensor read exactly what the physics model predicted. The filter is perfectly on track.
If the residual is large, something unexpected happened (eg the vehicle hit a pothole, or the sensor experienced a sudden glitch).

The residual covariance (denoted as S) measures the total expected uncertainty of that measurement residual.
It is the sum of all errors coming from both your physics model and your physical sensor.

- The Equation: `S = H·P·Hᵀ + R`
- In plain English: Total Uncertainty = (Uncertainty of our internal prediction) + (Uncertainty/Noise of the sensor itself).
  It defines a boundary for what a "reasonable" surprise is.
- If your sensor is known to be very noisy (R is large), or your physics prediction is highly uncertain (P is large), S will be large.
  This means the filter expects the residual (y) to bounce around quite a bit naturally.

How They Work Together: The Kalman Gain Matrix

The filter divides the residual by the residual covariance to calculate the Kalman Gain (K):
`K = P·Hᵀ·S⁻¹`
By doing this, the Kalman filter calculates a ratio: "Is this surprise (y) caused by a real change in the world, or is it just random sensor noise (S)?"

- If S is dominated by sensor noise (R), the filter assumes the surprise is just a glitch.
  The Kalman gain drops, and the filter mostly ignores the measurement.
- If S is low (meaning the sensor is highly accurate), the filter assumes the surprise is real.
  The Kalman gain rises, and it heavily corrects the state estimate using the residual.

## Variable Definitions

- `x̂`: Estimated state vector
- `P`: State covariance matrix (estimated uncertainty)
- `F`: State transition matrix (derived from system dynamics)
- `B`: Control input matrix
- `u`: Control input vector
- `Q`: Process noise covariance matrix
- `z`: Measurement vector
- `H`: Measurement matrix
- `R`: Measurement noise covariance matrix
- `K`: Kalman gain matrix
- `I`: Identity matrix

### B - the Control Input Matrix

The `B` matrix is formally called the Control Input Matrix.
In the predict equation `(x̂ₖ|ₖ₋₁ = Fₖ x̂ₖ₋₁|ₖ₋₁ + Bₖ uₖ)`, its job is to map known external forces or control inputs onto the system's state variables.
It translates the real-world actions you are taking into the changes they cause in your system.

#### How it Works (A Simple Example)

Imagine you are tracking a moving vehicle. Your state vector x tracks [position, velocity].
Your control vector u is how hard the driver hits the gas pedal, which adds a known acceleration (\(a\)).
According to basic physics:

- New Position = Old Position + (Velocity × time) + (0.5 × time² × Acceleration)
- New Velocity = Old Velocity + (time × Acceleration)

The B matrix holds those bold coefficients to map the acceleration to the position and velocity:

```text
       ┌          ┐
       │ 0.5×dt²  │
 B =   │          │
       │    dt    │
       └          ┘
```

Core Traits

- Optional: If the system is passive and has no external steering, motors, or brakes acting on it (like a drifting asteroid or a passive temperature sensor),
  the control vector u is zero, and the B matrix is simply left out.
- Deterministic: Unlike the process noise matrix (Q), B handles known, intentional changes that you are actively commanding the system to perform.

### F - the State Transition Matrix

The F matrix is formally called the State Transition Matrix (or system matrix).
In the predict equation (x̂ₖ|ₖ₋₁ = Fₖ x̂ₖ₋₁|ₖ₋₁ + Bₖ uₖ), its job is to model the natural laws of physics or system dynamics.
It defines how the system changes from the previous time step (k-1) to the current time step (k) if no external control forces are applied and no noise is present.

#### How it Works (Continuing the Vehicle Example)

Using the same vehicle tracking scenario where your state vector x is [position, velocity],
think about what happens if a car is drifting at a constant speed with no one hitting the gas pedal or the brakes.

According to basic physics:

- New Position = (1 × Old Position) + (time_delta × Old Velocity)
- New Velocity = (0 × Old Position) + (1 × Old Velocity)

The F matrix captures these exact relationships:

```text
       ┌          ┐
       │  1   dt  │
 F =   │          │
       │  0    1  │
       └          ┘
```

#### Core Traits

- Mandatory: Unlike the B matrix, every Kalman filter requires an F matrix because it represents how the system evolves over time.
- Linear Model: It assumes a linear transition. If your system dynamics are non-linear (like tracking a turning vehicle using angles),
  you have to transition to an Extended Kalman Filter (EKF), where F is replaced by a Jacobian matrix of partial derivatives.

### H - the Measurement Matrix

The `H` matrix is formally called the Measurement Matrix (or Observation Matrix).
In the subscript-less correct equation (`y = z - H·x`), its job is to bridge the gap between your internal state variables and your physical sensor readings.
It maps the hidden variables you are tracking inside your code into the exact dimensions and units that your hardware sensors can actually see.

#### How it Works (Using the Same Vehicle Example)

Continuing with our vehicle tracking system, your internal state vector x tracks 9 variables inside your 3x3 block layout:

x = [Position_XYZ, Velocity_XYZ, Acc_Bias_XYZ]ᵀ

Now, imagine your Barometer sensor fires an update.
It doesn't know anything about velocity, X/Y positioning, or accelerometer biases.
It only outputs a single scalar value: Z-axis position (altitude).
To calculate the measurement residual (how much the barometer reading surprises the filter),
the math needs to subtract your 9-variable state vector from a 1-variable sensor reading.
This is impossible without the H matrix.
The H matrix acts as an extraction filter.
For the barometer, it is a single row of 9 elements containing eight zeros and a single 1 pointing directly at the Z-position slot:

```text
H = [ 0  0  1   0  0  0   0  0  0 ]
```

When you perform the matrix multiplication H @ x, it unrolls to do something incredibly simple:

- (0 × X_pos) + (0 × Y_pos) + (1 × Z_pos) + (0 × X_vel) ... = Z_pos

It extracts just the Z-position from your complex internal state.
Now, the math can easily compute the innovation: y = altitude_measurement - Z_pos.

#### Core Traits

- The Multilingual Translator: Your internal state might track values you have absolutely no way of measuring directly (like your accelerometer biases).
  The `H` matrix tells the filter how those hidden internal variables are expected to manifest themselves in the sensors you do have.
- Shape Dynamics: The dimensions of H are completely flexible. Its number of rows always matches the number of incoming sensor channels,
  and its number of columns always matches your internal state size. (eg for a 3D GPS measuring X, Y, and Z, H expands to 3 rows and 9 columns [1]).
- Bypassed in Code: As you implemented in your correct_altitude function, because H is so sparse (mostly zeros and ones),
  we rarely ever allocate a literal H matrix in embedded code.
  Instead, we use its mathematical structure as a blueprint to extract columns and lines directly via hardcoded array indices, saving massive amounts of CPU bandwidth.

### R - the Measurement Noise Covariance Matrix

In the correct phase equation (`S = H·P·Hᵀ + R`), its job is to quantify the trustworthiness and accuracy of your physical sensors.
It explicitly tells the Kalman filter how much random noise, electrical interference, or environmental vibration is polluting your incoming sensor readings.

#### How it Works (Using the Multi-Sensor Example)

Let's look at your two sensors—the GPS and the Barometer—and see how the R matrix alters its shape and values to match them.

##### Scenario 1: The Barometer (1D Sensor)

When your barometer sends an update to your correct_altitude function, it passes a single scalar value (altitude in meters).
Because there is only one sensor channel, the R matrix collapses down into a 1×1 matrix (a single scalar variance):

`R_baro = sigma_baro²`

If your barometer has an RMS noise of 0.1 meters, `R_baro` = [0.01].

##### Scenario 2: The GPS (3D Sensor)

When your GPS updates the system, it passes a full 3D position vector [X, Y, Z].
Because you have 3 distinct measurement channels, your R matrix expands to a 3×3 matrix.
If the GPS reports a horizontal tracking uncertainty of 2.0 meters and a vertical tracking uncertainty of 5.0 meters,
your R matrix maps out as a diagonal matrix:

```text
         ┌                ┐
         │ 4.0   0.0  0.0 │  <- X-axis variance (2.0²)
R_gps =  │ 0.0   4.0  0.0 │  <- Y-axis variance (2.0²)
         │ 0.0   0.0 25.0 │  <- Z-axis variance (5.0²)
         └                ┘
```

The zeros on the off-diagonals tell the filter an important assumption:
noise corrupting the X-axis reading is statistically independent of noise corrupting the Z-axis reading.

### Core Traits

- The Filter's Brake Pedal: R acts as a tuning knob to control how smoothly your filter reacts.
  If you increase the values in R, you are telling the filter that your sensors are garbage.
  The filter will respond by ignoring the jerky sensor jumps, choosing instead to rely heavily on your smooth predict_state physics propagation.
- Dynamically Adaptable: As you saw with the GPS configuration, R does not have to stay constant.
  If your hardware telemetry tells you your sensor signal quality is degrading (e.g., lower satellite counts),
  you can inflate the elements of R at runtime to protect the filter from tracking corrupt data.
- Dimension Symmetry: The dimensions of R are always M × M (Rows × Columns),
  where M is the exact number of data channels the active sensor provides.
  It always matches the number of rows in your H matrix.

### The standard Kalman filter equations

```text
Predict: 
    x = F·x 
    P_new = F·P·Fᵀ + Q

Correct: 
    S = H·P·Hᵀ + R
    K = P·Hᵀ·S⁻¹
    y = z - H·x
    x = x + K·y

    covariance update:
    P = P - (K·H)·P
    or
    Joseph's Stabilized Form for the covariance update step, computationally more expensive, numerically stabler
    P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ
```

```text
Predict: 
    state = F·state, 
    P_new = F·P·Fᵀ + Q

Correct: 
    S = H·P·Hᵀ + R
    K = P·Hᵀ·S⁻¹
    innovation = measurement - H·state 
    state += K·innovation
    P -= (K·H)·P
```

### Rule of Thumb for Tuning

If the filter output looks laggy or sluggish (clinging too closely to the old path and trailing behind rapid movements),
you need to trust the physics model less and the sensors more:
Action: Increase `Q` or Decrease `R`.

If the filter output looks jittery or nervous (shaking whenever the sensor outputs minor noise spikes),
you are trusting the noisy sensors too heavily over the smooth physics predictions:
Action: Decrease `Q` or Increase `R`.
