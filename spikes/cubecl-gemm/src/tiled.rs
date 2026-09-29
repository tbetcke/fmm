//! Hand-written tiled GEMMs C = A X specialised at compile time on p.
//!
//! A is nc x nc with nc = (p+1)^2, X and C are nc x b, all row-major and contiguous.
//! Two kernels, both with nc comptime:
//!
//! - `smem`: GPU-shaped, shared-memory tiles and cube-wide barriers.
//! - `reg`: one unit per register tile of C, columns vectorised, operands read straight
//!   from global memory (relying on caches). No shared memory or barriers, so it also
//!   suits the CubeCL CPU runtime, where `smem` is very slow.
//!
//! `smem`: a cube of 16 x 16 units computes a block of `16 * tm` rows and `BN` columns of C;
//! each unit accumulates `tm x TN` outputs in registers, at rows `ty + 16 i` and columns
//! `tx + 16 j`. Tiles of A (BK columns) and X (BK rows) are staged in shared memory.
//! `nc` and `tm` are comptime, so the k loop trip count and the row guards are fixed
//! when the kernel is compiled for a given p. The rows per unit `tm` is chosen so that
//! one or two row blocks cover all nc rows: X is then read from memory at most twice.

use cubecl::prelude::*;

use crate::reference::Real;

/// Units per cube in each direction.
const UNITS: usize = 16;
/// Columns of C per unit.
const TN: usize = 4;
/// Columns of C per cube.
const BN: usize = UNITS * TN;
/// Depth of one k tile.
const BK: usize = 16;
/// Upper bound on the rows of C per unit, to bound register use.
const MAX_TM: usize = 12;

#[cube(launch_unchecked)]
fn tiled_gemm_kernel<F: Float>(
    a: &Array<F>,
    x: &Array<F>,
    c: &mut Array<F>,
    b: u32,
    #[comptime] nc: usize,
    #[comptime] tm: usize,
) {
    let bm = UNITS * tm;
    let b = b as usize;
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let tid = ty * UNITS + tx;
    let col0 = CUBE_POS_X as usize * BN;
    let row0 = CUBE_POS_Y as usize * bm;

    // a_s[k * bm + m] holds A[row0 + m, k0 + k]; x_s[k * BN + n] holds X[k0 + k, col0 + n].
    let mut a_s = SharedMemory::<F>::new(BK * bm);
    let mut x_s = SharedMemory::<F>::new(BK * BN);
    let mut acc = Array::<F>::new(tm * TN);
    #[unroll]
    for i in 0..tm * TN {
        acc[i] = F::new(0.0f32);
    }
    let mut a_reg = Array::<F>::new(tm);
    let mut x_reg = Array::<F>::new(TN);

    for kt in 0..nc.div_ceil(BK) {
        let k0 = kt * BK;
        // The A tile has BK * bm = 256 * tm entries: tm loads per unit.
        #[unroll]
        for l in 0..tm {
            let idx = tid + l * UNITS * UNITS;
            let m = idx / BK;
            let k = idx % BK;
            let (gi, gk) = (row0 + m, k0 + k);
            let mut v = F::new(0.0f32);
            if gi < nc && gk < nc {
                v = a[gi * nc + gk];
            }
            a_s[k * bm + m] = v;
        }
        // The X tile has BK * BN = 256 * TN entries: TN loads per unit.
        #[unroll]
        for l in 0..TN {
            let idx = tid + l * UNITS * UNITS;
            let k = idx / BN;
            let n = idx % BN;
            let (gk, gj) = (k0 + k, col0 + n);
            let mut v = F::new(0.0f32);
            if gk < nc && gj < b {
                v = x[gk * b + gj];
            }
            x_s[k * BN + n] = v;
        }
        sync_cube();

        #[unroll]
        for k in 0..BK {
            #[unroll]
            for i in 0..tm {
                a_reg[i] = a_s[k * bm + ty + UNITS * i];
            }
            #[unroll]
            for j in 0..TN {
                x_reg[j] = x_s[k * BN + tx + UNITS * j];
            }
            #[unroll]
            for i in 0..tm {
                #[unroll]
                for j in 0..TN {
                    acc[i * TN + j] += a_reg[i] * x_reg[j];
                }
            }
        }
        sync_cube();
    }

    #[unroll]
    for i in 0..tm {
        let gi = row0 + ty + UNITS * i;
        #[unroll]
        for j in 0..TN {
            let gj = col0 + tx + UNITS * j;
            if gi < nc && gj < b {
                c[gi * b + gj] = acc[i * TN + j];
            }
        }
    }
}

/// Rows of C per unit for a given nc: the smallest tm <= MAX_TM such that
/// ceil(nc / (16 tm)) row blocks cover nc with as little padding as possible.
pub fn rows_per_unit(nc: usize) -> usize {
    let row_groups = nc.div_ceil(UNITS);
    let blocks = row_groups.div_ceil(MAX_TM);
    row_groups.div_ceil(blocks)
}

/// Launches the shared-memory kernel for C = A X, with nc = (p+1)^2 fixed at compile time.
///
/// `a`, `x` and `c` must hold nc * nc, nc * b and nc * b elements of `F`.
pub fn launch_smem<R: Runtime, F: Real>(
    client: &ComputeClient<R>,
    p: usize,
    b: usize,
    a: &cubecl::server::Handle,
    x: &cubecl::server::Handle,
    c: &cubecl::server::Handle,
) {
    let nc = (p + 1) * (p + 1);
    let tm = rows_per_unit(nc);
    let cubes_x = b.div_ceil(BN) as u32;
    let cubes_y = nc.div_ceil(UNITS * tm) as u32;
    // SAFETY: the array lengths match the allocations made by the caller, and every
    // access in the kernel is guarded by the comptime nc and the runtime b.
    unsafe {
        tiled_gemm_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(cubes_x, cubes_y, 1),
            CubeDim::new_2d(UNITS as u32, UNITS as u32),
            ArrayArg::from_raw_parts(a.clone(), nc * nc),
            ArrayArg::from_raw_parts(x.clone(), nc * b),
            ArrayArg::from_raw_parts(c.clone(), nc * b),
            b as u32,
            nc,
            tm,
        );
    }
}

/// Rows of C per unit in the register kernel.
pub const REG_TM: usize = 4;
/// Column vectors of C per unit in the register kernel.
pub const REG_TNV: usize = 2;
/// Vector width (elements) of the register kernel.
pub const REG_VEC: usize = 4;
/// Units per cube of the register kernel, along the columns.
const REG_UNITS: usize = 32;

#[cube(launch_unchecked)]
fn reg_gemm_kernel<F: Float, N: Size>(
    a: &Array<F>,
    x: &Array<Vector<F, N>>,
    c: &mut Array<Vector<F, N>>,
    bv: u32,
    #[comptime] nc: usize,
    #[comptime] tm: usize,
    #[comptime] tnv: usize,
    #[comptime] rows_fast: bool,
) {
    let bv = bv as usize;
    // rows_fast: the units of a cube are the row blocks of one column strip, so the
    // strip of X is reused from cache by the whole cube. Otherwise the units of a cube
    // are adjacent column blocks of one row block (coalesced loads of X on a GPU).
    let (row_block, col_block) = if rows_fast {
        (ABSOLUTE_POS_X as usize, ABSOLUTE_POS_Y as usize)
    } else {
        (ABSOLUTE_POS_Y as usize, ABSOLUTE_POS_X as usize)
    };
    let j0 = col_block * tnv;
    let i0 = row_block * tm;
    if j0 >= bv || i0 >= nc {
        terminate!();
    }
    let mut acc = Array::<Vector<F, N>>::new(tm * tnv);
    #[unroll]
    for i in 0..tm * tnv {
        acc[i] = Vector::new(F::new(0.0f32));
    }
    let mut xv = Array::<Vector<F, N>>::new(tnv);
    for k in 0..nc {
        #[unroll]
        for j in 0..tnv {
            let mut v = Vector::new(F::new(0.0f32));
            if j0 + j < bv {
                v = x[k * bv + j0 + j];
            }
            xv[j] = v;
        }
        #[unroll]
        for i in 0..tm {
            let mut aik = F::new(0.0f32);
            if i0 + i < nc {
                aik = a[(i0 + i) * nc + k];
            }
            let av = Vector::new(aik);
            #[unroll]
            for j in 0..tnv {
                acc[i * tnv + j] += av * xv[j];
            }
        }
    }
    #[unroll]
    for i in 0..tm {
        #[unroll]
        for j in 0..tnv {
            if i0 + i < nc && j0 + j < bv {
                c[(i0 + i) * bv + j0 + j] = acc[i * tnv + j];
            }
        }
    }
}

/// Launches the register kernel for C = A X, with nc = (p+1)^2 fixed at compile time.
///
/// `a`, `x` and `c` must hold nc * nc, nc * b and nc * b elements of `F`, and b must be a
/// multiple of [`REG_VEC`]. With `rows_fast`, one cube covers all row blocks of a column
/// strip; otherwise one cube covers [`REG_UNITS`] column blocks of a row block.
#[allow(clippy::too_many_arguments)]
pub fn launch_reg<R: Runtime, F: Real>(
    client: &ComputeClient<R>,
    p: usize,
    b: usize,
    rows_fast: bool,
    a: &cubecl::server::Handle,
    x: &cubecl::server::Handle,
    c: &cubecl::server::Handle,
) {
    assert_eq!(b % REG_VEC, 0, "B must be a multiple of {REG_VEC}");
    let nc = (p + 1) * (p + 1);
    let bv = b / REG_VEC;
    let col_blocks = bv.div_ceil(REG_TNV);
    let row_blocks = nc.div_ceil(REG_TM);
    let (count, dim) = if rows_fast {
        (
            CubeCount::Static(1, col_blocks as u32, 1),
            CubeDim::new_2d(row_blocks as u32, 1),
        )
    } else {
        (
            CubeCount::Static(col_blocks.div_ceil(REG_UNITS) as u32, row_blocks as u32, 1),
            CubeDim::new_2d(REG_UNITS as u32, 1),
        )
    };
    // SAFETY: the array lengths (in vectors for x and c) match the caller's allocations,
    // and every access is guarded by the comptime nc and the runtime bv.
    unsafe {
        reg_gemm_kernel::launch_unchecked::<F, R>(
            client,
            count,
            dim,
            REG_VEC,
            ArrayArg::from_raw_parts(a.clone(), nc * nc),
            ArrayArg::from_raw_parts(x.clone(), nc * bv),
            ArrayArg::from_raw_parts(c.clone(), nc * bv),
            bv as u32,
            nc,
            REG_TM,
            REG_TNV,
            rows_fast,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `launch` on the CubeCL CPU runtime and returns the error against the reference.
    #[cfg(feature = "cpu")]
    fn cpu_error<F: Real>(
        p: usize,
        b: usize,
        launch: impl Fn(
            &ComputeClient<cubecl::cpu::CpuRuntime>,
            &cubecl::server::Handle,
            &cubecl::server::Handle,
            &cubecl::server::Handle,
        ),
    ) -> f64 {
        use crate::reference::{gemm_f64, rel_max_error, uniform};
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let client = CpuRuntime::client(&CpuDevice);
        let nc = (p + 1) * (p + 1);
        let a_host = uniform::<F>(nc * nc, 7);
        let x_host = uniform::<F>(nc * b, 8);
        let a = client.create_from_slice(F::as_bytes(&a_host));
        let x = client.create_from_slice(F::as_bytes(&x_host));
        let c = client.empty(nc * b * size_of::<F>());
        launch(&client, &a, &x, &c);
        let bytes = client.read_one(c).expect("read back C");
        rel_max_error(F::from_bytes(&bytes), &gemm_f64(&a_host, &x_host, nc, b))
    }

    /// Both hand-written kernels match the f64 reference on the CPU runtime, for shapes
    /// that exercise the row, column and k guards (B is not a multiple of any tile).
    #[cfg(feature = "cpu")]
    #[test]
    fn kernels_match_reference_on_cpu_runtime() {
        for (p, b) in [(1, 12), (4, 68), (8, 100)] {
            let e = cpu_error::<f64>(p, b, |cl, a, x, c| launch_smem::<_, f64>(cl, p, b, a, x, c));
            assert!(e < 1e-12, "smem f64 p = {p}, B = {b}: {e:e}");
            let e = cpu_error::<f32>(p, b, |cl, a, x, c| launch_smem::<_, f32>(cl, p, b, a, x, c));
            assert!(e < 1e-5, "smem f32 p = {p}, B = {b}: {e:e}");
            for rows_fast in [false, true] {
                let e = cpu_error::<f64>(p, b, |cl, a, x, c| {
                    launch_reg::<_, f64>(cl, p, b, rows_fast, a, x, c)
                });
                assert!(
                    e < 1e-12,
                    "reg f64 p = {p}, B = {b}, rows_fast = {rows_fast}: {e:e}"
                );
                let e = cpu_error::<f32>(p, b, |cl, a, x, c| {
                    launch_reg::<_, f32>(cl, p, b, rows_fast, a, x, c)
                });
                assert!(
                    e < 1e-5,
                    "reg f32 p = {p}, B = {b}, rows_fast = {rows_fast}: {e:e}"
                );
            }
        }
    }

    /// Row blocks cover nc for the spike's degrees, and stay within the register bound.
    #[test]
    fn row_blocking_covers_nc() {
        for p in [4, 8, 12, 16] {
            let nc = (p + 1) * (p + 1);
            let tm = rows_per_unit(nc);
            assert!(tm <= MAX_TM);
            let blocks = nc.div_ceil(UNITS * tm);
            assert!(blocks * UNITS * tm >= nc);
            assert!(blocks <= 2, "p = {p}: {blocks} row blocks");
        }
    }
}
