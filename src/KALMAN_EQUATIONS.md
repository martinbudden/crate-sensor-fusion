# Kalman filter equations

These are the unabbreviated forms of the Kalman equations.

## Notation

In estimation theory and signal processing, the following notation is used:

The "hat" symbol ( ^ ) placed over a variable denotes an estimate of that variable, rather than its true, exact value.
**x̂** reads as "x-hat".

The subscript **k|k-1** denotes a conditional relationship,
reads as "at time step **k**, given all information up to time step **k-1**".

It highlights the difference between a prediction and an updated estimate:

- The Left Side **k**: the specific point in time you are currently predicting or analyzing.
- The Vertical Bar **|**: means "given" or "conditioned upon" (exactly like conditional probability notation).
- The Right Side **k-1**: the point in time representing the last available measurement data.

| Notation    | Type                        | Plain English Meaning                                                                                    |
| :---------- | :-------------------------- | :------------------------------------------------------------------------------------------------------- |
| **x̂ₖ\|ₖ₋₁** | *A Priori* (Prediction)     | Our best guess for the state at step **k**, based solely on what we knew up to step **k-1**              |
| **x̂ₖ\|ₖ**   | *A Posteriori* (Correction) | Our corrected state estimate at step **k**, now that we have factored in the measurement from step **k** |

- **x̂ₖ|ₖ₋₁** reads as "x-hat of k, given k minus one" or "the a priori state estimate at step k".
- **x̂ₖ|ₖ** reads as "x-hat of k, given k" or "the a posteriori state estimate at step k".

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

- **u**: Control vector
- **x̂**: Estimated state vector
- **z**: Measurement vector
- **y**: Measurement innovation

- **P**: State covariance matrix (estimated uncertainty)
- **F**: State transition matrix (derived from system dynamics)
- **B**: Control input matrix
- **Q**: Process noise covariance matrix
- **H**: Measurement matrix
- **R**: Measurement noise covariance matrix
- **K**: Kalman gain matrix
- **I**: Identity matrix
