#![allow(clippy::inline_always)]
#![allow(clippy::many_single_char_names)]

#[cfg(feature = "simd")]
use core::simd::{f32x4, simd_swizzle};

#[allow(unused)]
use vqm::{MathMethods, Quaternion, Vector3};

/// Math functions for `SensorFusion`, using **SIMD** accelerations for `f32`.
pub trait SensorFusionMath: Sized {
    /// Estimate the gravity vector, ie direction of vertical in the sensor frame.
    fn gravity(q: Quaternion<Self>) -> Vector3<Self>;
    /// Time derivative of `q`, given angular velocity `omega` in radians per second.
    fn derivative(q: Quaternion<Self>, omega: Vector3<Self>) -> Quaternion<Self>;
}

impl SensorFusionMath for f32 {
    #[inline(always)]
    fn gravity(q: Quaternion<Self>) -> Vector3<Self> {
        Vector3 {
            x: 2.0 * (q.x * q.z - q.w * q.y),
            y: 2.0 * (q.y * q.z + q.w * q.x),
            z: 2.0 * (q.w * q.w + q.z * q.z) - 1.0, // = w*w - x*x - y*y + z*z
        }
    }

    #[inline(always)]
    fn derivative(q: Quaternion<Self>, omega: Vector3<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            /*// Load q: [w, x, y, z]
            let q_v: f32x4 = unsafe { core::mem::transmute(q) };

            // Load gyro: [x, y, z, padding] -> Swizzle to [0, x, y, z]
            let g_raw: f32x4 = unsafe { core::mem::transmute(gyro) };
            //let g_v = simd_swizzle!(g_raw, [f32x4::splat(0.0)], [4, 0, 1, 2]);

            // Efficiently shift: [x, y, z, 0] -> [0, x, y, z]
            // We rotate right and then zero out the 'w' (index 0)
            let g_v = simd_swizzle!(g_raw, [3, 0, 1, 2]) * f32x4::from_array([0.0, 1.0, 1.0, 1.0]);

            // Hamilton Product (q * g) for [w, x, y, z] layout:
            // w_out = -x1*gx - y1*gy - z1*gz
            // x_out =  w1*gx + y1*gz - z1*gy
            // y_out =  w1*gy - x1*gz + z1*gx
            // z_out =  w1*gz + x1*gy - y1*gx

            // Row A: [w, w, w, w] * [0, gx, gy, gz]
            let res = simd_swizzle!(q_v, [0, 0, 0, 0]) * g_v;

            // Row B: [x, -x, -x, -x] * [gx, 0, gz, gy]
            // (Using signs and swizzles to match the Hamilton rows)
            let x_part = simd_swizzle!(q_v, [1, 1, 1, 1])
                * simd_swizzle!(g_v, [1, 0, 3, 2])
                * f32x4::from_array([-1.0, 1.0, -1.0, 1.0]);

            // Row C: [y, y, -y, y] * [gy, gz, 0, gx]
            let y_part = simd_swizzle!(q_v, [2, 2, 2, 2])
                * simd_swizzle!(g_v, [2, 3, 0, 1])
                * f32x4::from_array([-1.0, 1.0, 1.0, -1.0]);

            // Row D: [z, z, z, -z] * [gz, gy, gx, 0]
            let z_part = simd_swizzle!(q_v, [3, 3, 3, 3])
                * simd_swizzle!(g_v, [3, 2, 1, 0])
                * f32x4::from_array([-1.0, -1.0, 1.0, 1.0]);

            let q_dot = (res + x_part + y_part + z_part) * f32x4::splat(0.5);

            unsafe { core::mem::transmute(q_dot) }*/
            let q_v = f32x4::from(q);
            let g_raw = f32x4::from(omega);

            // Shift [x, y, z, pad] to [0, x, y, z] and zero the w lane
            let g_v = simd_swizzle!(g_raw, [3, 0, 1, 2]) * f32x4::from_array([0.0, 1.0, 1.0, 1.0]);

            // Parallel Hamilton Calculation
            let w1 = simd_swizzle!(q_v, [0, 0, 0, 0]);
            let x1 = simd_swizzle!(q_v, [1, 1, 1, 1]);
            let y1 = simd_swizzle!(q_v, [2, 2, 2, 2]);
            let z1 = simd_swizzle!(q_v, [3, 3, 3, 3]);

            let g_w = g_v; // [0, gx, gy, gz]
            let g_x = simd_swizzle!(g_v, [1, 0, 3, 2]); // [gx, 0, gz, gy]
            let g_y = simd_swizzle!(g_v, [2, 3, 0, 1]); // [gy, gz, 0, gx]
            let g_z = simd_swizzle!(g_v, [3, 2, 1, 0]); // [gz, gy, gx, 0]

            let res = (w1 * g_w)
                + (x1 * g_x * f32x4::from_array([-1.0, 1.0, -1.0, 1.0]))
                + (y1 * g_y * f32x4::from_array([-1.0, 1.0, 1.0, -1.0]))
                + (z1 * g_z * f32x4::from_array([-1.0, -1.0, 1.0, 1.0]));

            let q_dot = res * f32x4::splat(0.5);
            q_dot.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            Quaternion {
                w: (-q.x * omega.x - q.y * omega.y - q.z * omega.z) * 0.5,
                x: f32::midpoint(q.w * omega.x - q.z * omega.y, q.y * omega.z),
                y: (q.z * omega.x + q.w * omega.y - q.x * omega.z) * 0.5,
                z: (-q.y * omega.x + q.x * omega.y + q.w * omega.z) * 0.5,
            }
        }
    }
}

impl SensorFusionMath for f64 {
    #[inline(always)]
    fn gravity(q: Quaternion<Self>) -> Vector3<Self> {
        Vector3 {
            x: 2.0 * (q.x * q.z - q.w * q.y),
            y: 2.0 * (q.y * q.z + q.w * q.x),
            z: 2.0 * (q.w * q.w + q.z * q.z) - 1.0, // = w*w - x*x - y*y + z*z
        }
    }

    #[inline(always)]
    fn derivative(q: Quaternion<Self>, omega: Vector3<Self>) -> Quaternion<Self> {
        Quaternion {
            w: (-q.x * omega.x - q.y * omega.y - q.z * omega.z) * 0.5,
            x: (q.w * omega.x + q.y * omega.z - q.z * omega.y) * 0.5,
            y: f64::midpoint(q.w * omega.y - q.x * omega.z, q.z * omega.x),
            z: (q.w * omega.z + q.x * omega.y - q.y * omega.x) * 0.5,
        }
    }
}
