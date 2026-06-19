use core::ops::Mul;

use num_traits::Zero;

use vqm::{Matrix9x9, Matrix9x9Math, Vector3d};

/// Kalman state vector of `f32` values<br>
pub type KalmanStateVector9f32 = KalmanStateVector9<f32>;
/// Kalman tate vector of `f64` values<br><br>
pub type KalmanStateVector9f64 = KalmanStateVector9<f64>;

/// Flattened representation of a 9-element state vector for Kalman filter matrix math.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C, align(64))]
pub struct KalmanStateVector9<T> {
    pub pos: Vector3d<T>,
    pub vel: Vector3d<T>,
    pub bias: Vector3d<T>,
}

impl<T> KalmanStateVector9<T>
where
    T: Copy,
{
    #[inline]
    pub const fn new(v: (Vector3d<T>, Vector3d<T>, Vector3d<T>)) -> Self {
        Self { pos: v.0, vel: v.1, bias: v.2 }
    }
}

impl<T> From<(Vector3d<T>, Vector3d<T>, Vector3d<T>)> for KalmanStateVector9<T> {
    #[inline]
    fn from(v: (Vector3d<T>, Vector3d<T>, Vector3d<T>)) -> Self {
        Self { pos: v.0, vel: v.1, bias: v.2 }
    }
}

/// Implement vector-by-scalar multiplication to scale the Kalman Gain.
impl Mul<f32> for KalmanStateVector9<f32> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self { pos: self.pos * rhs, vel: self.vel * rhs, bias: self.bias * rhs }
    }
}

impl Mul<f64> for KalmanStateVector9<f64> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self { pos: self.pos * rhs, vel: self.vel * rhs, bias: self.bias * rhs }
    }
}

impl<T> KalmanStateVector9<T>
where
    T: Copy + Zero + Matrix9x9Math + Mul<T, Output = T> + PartialEq,
{
    /// Calculates the outer product of two 9-element states in a compiler-friendly manner.
    #[inline]
    pub fn outer_product(self, row: KalmanStateVector9<T>) -> Matrix9x9<T> {
        Self::outer_product_associated(self, row)
    }

    /// Calculates the outer product of two 9-element states in a compiler-friendly manner.
    #[inline]
    fn outer_product_associated(col: KalmanStateVector9<T>, row: KalmanStateVector9<T>) -> Matrix9x9<T> {
        // Convert the 9 elements into three 4-lane logically padded chunks.
        // The 4th element is padding to fulfill 128-bit SIMD constraints.
        let r1 = [row.pos.x, row.pos.y, row.pos.z, T::zero()];
        let r2 = [row.vel.x, row.vel.y, row.vel.z, T::zero()];
        let r3 = [row.bias.x, row.bias.y, row.bias.z, T::zero()];

        // Flatten `col` into an array  easily indexable via registers
        let c = [col.pos.x, col.pos.y, col.pos.z, col.vel.x, col.vel.y, col.vel.z, col.bias.x, col.bias.y, col.bias.z];

        let mut ret = <Matrix9x9<T>>::zero();

        // Process each row. LLVM sees fixed loops of 4 elements and emits optimized parallel hardware vector instructions.
        for (r_idx, &scalar) in c.iter().enumerate().take(9) {
            // array::map creates a clean, fixed-size loop that LLVM easily unrolls
            // and vectorizes since the size (4) and operation are perfectly predictable.
            let chunk1 = r1.map(|val| scalar * val);
            let chunk2 = r2.map(|val| scalar * val);
            let chunk3 = r3.map(|val| scalar * val);

            // Get a mutable reference to the row (9 elements)
            let ret_row = &mut ret[r_idx * 9..(r_idx + 1) * 9];

            // Write back to the matrix, dropping the 4th padding lane of each chunk.
            // Since the sizes are matched the compiler can generate raw, grouped memory stream instructions.
            ret_row[0..3].copy_from_slice(&chunk1[0..3]);
            ret_row[3..6].copy_from_slice(&chunk2[0..3]);
            ret_row[6..9].copy_from_slice(&chunk3[0..3]);
        }

        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<KalmanStateVector9f32>();
    }
}
