use core::ops::{Add, AddAssign};
use num_traits::{ConstZero, float::FloatCore};
use vqm::{Matrix2x2, Vector2};

/// 2-dimensional anchor of `f32` values<br><br>
pub type Anchor2df32 = Anchor2d<f32>;
/// 2-dimensional anchor of `f64` values<br><br>
pub type Anchor2df64 = Anchor2d<f64>;

#[derive(Clone, Copy, Debug, PartialEq)]
//#[cfg_attr(feature = "std", derive(derive_more::Display))]
//#[cfg_attr(feature = "std", display("V{{x:{x}, y:{y}}}"))]
pub struct Anchor2d<T> {
    pub pos: Vector2<T>,
    pub distance: T,
}

impl<T> Default for Anchor2d<T>
where
    T: FloatCore,
{
    fn default() -> Self {
        Self::new(T::zero(), T::zero())
    }
}

impl<T> Anchor2d<T>
where
    T: FloatCore,
{
    /// Constructor.
    #[must_use]
    pub fn new(x: T, y: T) -> Self {
        Anchor2d { pos: Vector2 { x, y }, distance: T::zero() }
    }
}

/// Trilaterates a 2d position using the least squares method.
#[must_use]
pub fn trilaterate_2d<T>(anchors: &[Anchor2d<T>]) -> Option<Vector2<T>>
where
    T: ConstZero + Default + FloatCore + Add<T, Output = T> + AddAssign + vqm::Matrix2x2Math,
{
    const A00: usize = 0;
    const A01: usize = 1;
    const A11: usize = 2;

    if anchors.len() < 3 {
        return None;
    }

    // Find the closest anchor and use it as the base anchor.
    let mut base_idx = 0;
    let mut min_distance = anchors[0].distance;

    for (ii, anchor) in anchors.iter().enumerate().skip(1) {
        if anchor.distance < min_distance {
            min_distance = anchor.distance;
            base_idx = ii;
        }
    }

    let base = anchors[base_idx];

    // Initialize the accumulation structure for the 2x2 matrix Aᵀ·A
    // Since Aᵀ·A is symmetric we only need to calculate the upper right corner of the matrix.
    // [ ata_00  ata_01 ]
    // [         ata_11 ]
    #[allow(non_snake_case)]
    let mut ATA: [T; 3] = [T::zero(); 3];

    // Initialize the 2x1 vector (atb)
    let mut atb = Vector2::default();

    // Accumulate all rows, skipping the base anchor
    for (ii, anchor) in anchors.iter().enumerate() {
        if ii == base_idx {
            continue;
        }
        // Calculate row components for Matrix A
        let a_row = (base.pos - anchor.pos) * (T::one() + T::one());

        // Accumulate into the Aᵀ·A matrix
        ATA[A00] += a_row.x * a_row.x;
        ATA[A01] += a_row.x * a_row.y;
        ATA[A11] += a_row.y * a_row.y;

        // Calculate vector b row values
        let b_row = (anchor.distance * anchor.distance - base.distance * base.distance)
            - (anchor.pos.norm_squared() - base.pos.norm_squared());
        // Accumulate into the 3d vector A^T * B elements
        atb += a_row * b_row;
    }

    let m = Matrix2x2::from_lower_triangular_array(ATA);

    let (adjugate, determinant) = m.adjugate();
    if determinant.abs() < T::min_positive_value() { None } else { Some((adjugate * atb) / determinant) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<Anchor2d<f32>>();
    }
}
