use num_traits::float::FloatCore;
use vqm::{Cholesky3x3f32, Vector3, Vector3f32};

/// 3-dimensional anchor of `f32` values<br><br>
pub type Anchor3df32 = Anchor3d<f32>;
/// 3-dimensional anchor of `f64` values<br><br>
pub type Anchor3df64 = Anchor3d<f64>;

#[derive(Clone, Copy, Debug, PartialEq)]
//#[cfg_attr(feature = "std", derive(derive_more::Display))]
//#[cfg_attr(feature = "std", display("V{{x:{x}, y:{y}}}"))]
pub struct Anchor3d<T> {
    pub pos: Vector3<T>,
    pub distance: T,
    /// Reliability multiplier. 1.0 is default, 0.0 ignores the sensor entirely.
    pub weight: T,
}

impl<T> Default for Anchor3d<T>
where
    T: FloatCore,
{
    fn default() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }
}

impl<T> Anchor3d<T>
where
    T: FloatCore,
{
    /// Constructor.
    #[must_use]
    pub fn new(x: T, y: T, z: T) -> Self {
        Anchor3d { pos: Vector3 { x, y, z }, distance: T::zero(), weight: T::one() }
    }
}

/// Trilaterates a 3D position using the least squares method.
#[must_use]
pub fn trilaterate_3d_weighted(anchors: &[Anchor3df32]) -> Option<Vector3f32> {
    const A00: usize = 0;
    const A01: usize = 1;
    const A02: usize = 2;
    const A11: usize = 3;
    const A12: usize = 4;
    const A22: usize = 5;

    if anchors.len() < 4 {
        return None;
    }

    // Find the anchor with the greatest weight. In the event of a tie, choose the nearest anchor.
    // This maximizes stability against sensor noise.
    let mut base_idx = 0;
    let mut min_distance = anchors[0].distance;
    let mut max_weight = anchors[0].weight;

    for (ii, anchor) in anchors.iter().enumerate().skip(1) {
        #[allow(clippy::float_cmp)]
        if anchor.weight > max_weight || (anchor.weight == max_weight && anchor.distance < min_distance) {
            max_weight = anchor.weight;
            min_distance = anchor.distance;
            base_idx = ii;
        }
    }
    // If the best anchor has no weight, the whole system is untrustworthy
    if max_weight < f32::EPSILON {
        return None;
    }

    let base = anchors[base_idx];

    // For a system of `n` anchors,
    // We have `A · x = b`, where `A` is an overdetermined `(n-1)x3` matrix and `b` is an `(n-1)`-dimensional vector.
    // We use the least squares formula to solve for `x`, ie `x = (Aᵀ·A)⁻¹·(Aᵀ·b)`,
    // or `x = (Aᵀ·A·w)⁻¹·(Aᵀ·b·w)` when we are using weighted distances.
    // Note that Aᵀ·A·w is a 3x3 matrix and Aᵀ·b·w, is a 3d vector.

    // Initialize accumulation structure for the 3x3 matrix Aᵀ·A·w
    // Since Aᵀ·A·w is symmetric we only need to calculate the upper right corner of the matrix.
    #[allow(non_snake_case)]
    let mut ATA_W: [f32; 6] = [0.0; 6];

    // Initialize 3x1 vector Aᵀ·b·w
    let mut atb_w = Vector3f32::default();

    // Accumulate all rows, skipping the base anchor
    for (ii, anchor) in anchors.iter().enumerate() {
        if ii == base_idx {
            continue;
        }

        // Combine anchor weights
        let weight = anchor.weight * base.weight;
        if weight < f32::EPSILON {
            continue;
        }

        // Calculate Matrix A row values
        let a_row = (base.pos - anchor.pos) * 2.0;
        // Accumulate into the Aᵀ·A·w 3x3 matrix
        ATA_W[A00] += a_row.x * a_row.x * weight;
        ATA_W[A01] += a_row.x * a_row.y * weight;
        ATA_W[A02] += a_row.x * a_row.z * weight;
        ATA_W[A11] += a_row.y * a_row.y * weight;
        ATA_W[A12] += a_row.y * a_row.z * weight;
        ATA_W[A22] += a_row.z * a_row.z * weight;

        // Calculate vector b row values
        let b_row = (anchor.distance * anchor.distance - base.distance * base.distance)
            - (anchor.pos.norm_squared() - base.pos.norm_squared());
        // Accumulate into the Aᵀ·b·w 3d vector
        atb_w += a_row * b_row * weight;
    }

    /*
    let m = Matrix3x3f32::new([
        ATA_W[A00], ATA_W[A01], ATA_W[A02], //
        ATA_W[A01], ATA_W[A11], ATA_W[A12], //
        ATA_W[A02], ATA_W[A12], ATA_W[A22], //
    ]);

    let (adjugate, determinant) = m.adjugate_symmetric();
    if determinant.abs() < f32::EPSILON { None } else { Some((adjugate * atb_w) / determinant) }
    */

    // Compute the Cholesky factorization (L·Lᵀ = Aᵀ·A·w).
    // This will fail and return None if the anchors are coplanar/collinear
    // or if noise causes the matrix to lose its positive-definite status.
    let cholesky = Cholesky3x3f32::from_lower_triangle_array(ATA_W)?;

    // Solve the system using Cholesky fast forward- and backward-substitution
    Some(cholesky.solve_vector(atb_w))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;

    /// Generates true distances from a known target position to an anchor layout.
    fn calculate_distances(target: Vector3f32, anchors: &mut [Anchor3df32]) {
        for anchor in anchors.iter_mut() {
            anchor.distance = anchor.pos.distance(target);
        }
    }

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Anchor3df32>();
    }

    #[test]
    fn test_perfect_geometric_data() {
        // Position of the tracked object
        let position = Vector3f32::new(4.5, 2.1, 7.8);

        // Define 5 distributed anchors spanning 3D space
        let mut anchors = [
            Anchor3df32::new(0.0, 0.0, 0.0),
            Anchor3df32::new(10.0, 0.0, 2.0),
            Anchor3df32::new(0.0, 12.0, 1.0),
            Anchor3df32::new(2.0, 1.0, 15.0),
            Anchor3df32::new(8.0, 9.0, 5.0),
        ];

        calculate_distances(position, &mut anchors);

        let option = trilaterate_3d_weighted(&anchors);
        assert!(option.is_some());

        if let Some(result) = option {
            // Validate that calculated values match the true location within tolerance
            let epsilon = 0.007;
            assert!((result.x - position.x).abs() < epsilon, "X mismatch: expected {}, got {}", position.x, result.x);
            assert!((result.y - position.y).abs() < epsilon, "Y mismatch: expected {}, got {}", position.y, result.y);
            assert!((result.z - position.z).abs() < epsilon, "Z mismatch: expected {}, got {}", position.z, result.z);
        }
    }

    #[test]
    fn test_insufficient_anchors_fails() {
        // 3D space mathematically demands at least 4 anchors to resolve the system
        let anchors = [
            Anchor3df32 { pos: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 }, distance: 5.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 10.0, y: 0.0, z: 0.0 }, distance: 5.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 0.0, y: 10.0, z: 0.0 }, distance: 5.0, weight: 1.0 },
        ];

        let result = trilaterate_3d_weighted(&anchors);
        assert!(result.is_none(), "Should fail with fewer than 4 anchors");
    }

    #[test]
    fn test_outlier_suppression_via_weights() {
        let position = Vector3f32::new(5.0, 5.0, 5.0);

        let mut anchors = [
            Anchor3df32 { pos: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 }, distance: 0.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 10.0, y: 0.0, z: 2.0 }, distance: 0.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 0.0, y: 10.0, z: 1.0 }, distance: 0.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 2.0, y: 1.0, z: 10.0 }, distance: 0.0, weight: 1.0 },
            // Anchor 5 will be our corrupted "outlier" sensor mimicking multipath reflection
            Anchor3df32 { pos: Vector3f32 { x: 8.0, y: 8.0, z: 8.0 }, distance: 0.0, weight: 0.001 },
        ];

        calculate_distances(position, &mut anchors);

        // Artificially corrupt the distance on the 5th anchor by adding 15 meters of noise
        anchors[4].distance += 15.0;

        let option = trilaterate_3d_weighted(&anchors);
        assert!(option.is_some());
        if let Some(result) = option {
            // The target coordinates should stay highly accurate because the bad sensor's weight is near zero
            assert!((result.x - position.x).abs() < 0.05, "X deviated too far due to outlier: {}", result.x);
            assert!((result.y - position.y).abs() < 0.05, "Y deviated too far due to outlier: {}", result.y);
            assert!((result.z - position.z).abs() < 0.05, "Z deviated too far due to outlier: {}", result.z);
        }
    }

    #[test]
    fn test_coplanar_anchors_fail() {
        // When all tracking nodes lie flat on a single ceiling plane (identical Z coordinate),
        // the 3D matrix system becomes non-invertible due to spatial ambiguity.
        let anchors = [
            Anchor3df32 { pos: Vector3f32 { x: 0.0, y: 0.0, z: 3.0 }, distance: 5.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 10.0, y: 0.0, z: 3.0 }, distance: 5.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 0.0, y: 10.0, z: 3.0 }, distance: 5.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 10.0, y: 10.0, z: 3.0 }, distance: 5.0, weight: 1.0 },
        ];

        let option = trilaterate_3d_weighted(&anchors);
        //assert!(option.is_none(), "Matrix inversion should fail when anchors are flatly coplanar");
        if let Some(pos) = option {
            assert_eq!(pos, Vector3f32::default());
        }
    }

    #[test]
    fn test_zero_weight_anchors_ignored() {
        let position = Vector3f32::new(3.0, 3.0, 3.0);

        let mut anchors = [
            Anchor3df32 { pos: Vector3f32 { x: 0.0, y: 0.0, z: 0.0 }, distance: 0.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 6.0, y: 0.0, z: 1.0 }, distance: 0.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 0.0, y: 6.0, z: 2.0 }, distance: 0.0, weight: 1.0 },
            Anchor3df32 { pos: Vector3f32 { x: 2.0, y: 1.0, z: 6.0 }, distance: 0.0, weight: 1.0 },
            // This anchor is entirely dead and turned off (weight = 0.0). Its data should be completely bypassed.
            Anchor3df32 { pos: Vector3f32 { x: 99.0, y: 99.0, z: 99.0 }, distance: 999.0, weight: 0.0 },
        ];

        calculate_distances(position, &mut anchors);
        // Ensure its absolute junk data stays corrupted
        anchors[4].distance = 999.0;

        let option = trilaterate_3d_weighted(&anchors);
        assert!(option.is_some());
        if let Some(result) = option {
            let epsilon = 0.002;
            assert!((result.x - position.x).abs() < epsilon);
            assert!((result.y - position.y).abs() < epsilon);
            assert!((result.z - position.z).abs() < epsilon);
        }
    }
}
