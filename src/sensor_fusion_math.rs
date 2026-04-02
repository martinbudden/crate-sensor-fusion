#[cfg(feature = "simd")]
use cfg_if::cfg_if;
cfg_if! {
    if #[cfg(feature = "simd")] {
        use core::{simd::f32x4};
        use core::simd::{simd_swizzle};
    }
}

use vector_quaternion_matrix::{Quaternion, Vector3d};

pub trait SensorFusionMath: Sized {
    fn derivative(q: Quaternion<Self>, gyro: Vector3d<Self>) -> Quaternion<Self>;
    fn madgwick_step(q: Quaternion<Self>, a: Vector3d<f32>) -> Quaternion<Self>;
    fn estimate_gravity(q: Quaternion<Self>) -> Vector3d<Self>;
}

impl SensorFusionMath for f64 {
    #[inline(always)]
    fn madgwick_step(_q: Quaternion<Self>, _a: Vector3d<f32>) -> Quaternion<Self> {
        Quaternion::default()
    }
    #[inline(always)]
    fn derivative(q: Quaternion<Self>, gyro_rps: Vector3d<Self>) -> Quaternion<Self> {
        Quaternion {
            w: (q.x * gyro_rps.x - q.y * gyro_rps.y - q.z * gyro_rps.z) * 0.5,
            x: (q.w * gyro_rps.x + q.y * gyro_rps.z - q.z * gyro_rps.y) * 0.5,
            y: (q.w * gyro_rps.y - q.x * gyro_rps.z + q.z * gyro_rps.x) * 0.5,
            z: (q.w * gyro_rps.z + q.x * gyro_rps.y - q.y * gyro_rps.x) * 0.5,
        }
    }
    #[inline(always)]
    fn estimate_gravity(q: Quaternion<Self>) -> Vector3d<Self> {
        Vector3d {
            x: 2.0 * (q.x * q.z - q.w * q.y),
            y: 2.0 * (q.y * q.z + q.w * q.x),
            z: q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
        }
    }
}

impl SensorFusionMath for f32 {
    #[inline(always)]
    fn derivative(q: Quaternion<Self>, gyro: Vector3d<Self>) -> Quaternion<Self> {
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
            let g_raw = f32x4::from(gyro);

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
            Self {
                w: 0.5 * (-q.x * gyro.x - q.y * gyro.y - q.z * gyro.z),
                x: 0.5 * (q.w * gyro.x + q.y * gyro.z - q.z * gyro.y),
                y: 0.5 * (q.w * gyro.y - q.x * gyro.z + q.z * gyro.x),
                z: 0.5 * (q.w * gyro.z + q.x * gyro.y - q.y * gyro.x),
            }
        }
    }

    /// Features of this implementation
    ///
    /// 1. Parallel Throughput: Instead of 12 separate floating-point multiplications and 8 additions,
    ///    the SIMD unit performs 3 vector multiplications and 2 vector additions.
    ///
    /// 2. Instruction Density: The simd_swizzle! maps directly to the VREV or VMOV instructions on the M33.
    ///
    /// 3. Register Reuse: q_v stays in its SIMD register the entire time.
    ///    The compiler will likely use VFMA (Vector Fused Multiply-Add) to combine the terms,
    ///    meaning this whole function could resolve in under 15 clock cycles.
    ///
    #[inline(always)]
    fn madgwick_step(q: Quaternion<Self>, a: Vector3d<Self>) -> Quaternion<Self> {
        #[cfg(feature = "simd")]
        {
            let q_v = f32x4::from(q);

            // 1. Calculate wz_common
            let wz_common = 2.0 * (q.x * q.x + q.y * q.y);

            // 2. Calculate xy_common (w*w + z*z - 1.0 + 2.0*wz_common + a.z)
            let xy_common = 2.0 * ((q.w * q.w) + (q.z * q.z) - 1.0 + (2.0 * wz_common) + a.z);

            // 3. Calculate Term 1: [w, x, y, z] * [qq, xy_common, xy_common, qq]
            let term1_scalars = f32x4::from_array([wz_common, xy_common, xy_common, wz_common]);
            let term1 = q_v * term1_scalars;

            // 4. Calculate Term 2: [y, -z, w, -x] * ax
            // Term 2: [y, -z, w, -x] * ax
            // Indices needed: y=2, z=3, w=0, x=1
            let term2_q = simd_swizzle!(q_v, [2, 3, 0, 1]);
            let term2_signs = f32x4::from_array([1.0, -1.0, 1.0, -1.0]);
            let term2 = (term2_q * term2_signs) * f32x4::splat(a.x);

            // 5. Calculate Term 3: [-x, -w, -z, -y] * ay
            // Term 3: [-x, -w, -z, -y] * ay
            // Indices needed: x=1, w=0, z=3, y=2
            let term3_q = simd_swizzle!(q_v, [1, 0, 3, 2]);
            let term3 = (term3_q * f32x4::splat(-1.0)) * f32x4::splat(a.y);

            // 6. Combine: step = term1 + term2 + term3
            let ret_v = term1 + term2 + term3;

            ret_v.into()
        }
        #[cfg(not(feature = "simd"))]
        {
            let wz_common = 2.0 * (q.x * q.x + q.y * q.y);
            let xy_common = 2.0 * (q.w * q.w + q.z * q.z - 1.0 + 2.0 * wz_common + a.z);
            Self {
                w: q.w * wz_common + q.y * a.x - q.x * a.y,
                x: q.x * xy_common - q.z * a.x - q.w * a.y,
                y: q.y * xy_common + q.w * a.x - q.z * a.y,
                z: q.z * wz_common - q.x * a.x - q.y * a.y,
            }
        }
    }

    #[inline(always)]
    fn estimate_gravity(q: Quaternion<Self>) -> Vector3d<Self> {
        Vector3d {
            x: 2.0 * (q.x * q.z - q.w * q.y),
            y: 2.0 * (q.y * q.z + q.w * q.x),
            z: q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
        }
    }
}
