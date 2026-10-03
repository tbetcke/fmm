//! Backends, the capability query, the report line and the counters.

use nd_fmm_kernels::{BackendKind, CUBECL_VERSION, Device, KernelError, Precision};

use crate::common::tests_on;

#[test]
fn uncompiled_backends_give_their_error() {
    let mut checked = Vec::new();
    for kind in BackendKind::ALL.into_iter().filter(|k| !k.is_compiled()) {
        let err = Device::open(kind).unwrap_err();
        assert_eq!(err, KernelError::NotCompiled { backend: kind });
        println!("{kind}: {err}");
        checked.push(kind.name());
    }
    println!(
        "uncompiled backends checked: {}",
        if checked.is_empty() {
            "none (every backend is compiled in)".into()
        } else {
            checked.join(", ")
        }
    );
    println!("{}", crate::common::backends_line(&[]));
}

#[test]
fn backend_names_round_trip() {
    for kind in BackendKind::ALL {
        assert_eq!(BackendKind::from_name(kind.name()), Some(kind));
        assert_eq!(kind.to_string(), kind.name());
    }
    assert_eq!(BackendKind::from_name("host"), None);
    assert_eq!(Precision::F64.to_string(), "f64");
}

/// `CUBECL_VERSION` is the workspace pin of `cubecl` in the root `Cargo.toml`.
#[test]
fn cubecl_version_is_the_workspace_pin() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../Cargo.toml"))
        .expect("read the root Cargo.toml");
    let pin = format!("cubecl = {{ version = \"={CUBECL_VERSION}\"");
    assert!(
        manifest.contains(&pin),
        "root Cargo.toml does not pin {pin}"
    );
}

/// Every backend reports f32, and its report line names it.
fn reports_f32(device: &mut Device) {
    let info = device.info().clone();
    assert!(device.supports(Precision::F32));
    device.require(Precision::F32).unwrap();
    let line = info.to_string();
    assert!(line.starts_with(&format!("{} (", info.backend)), "{line}");
    assert!(line.contains(&format!("CubeCL {CUBECL_VERSION}")), "{line}");
    println!(
        "  plane size {:?}, shared memory {} B, units per cube {}, cubes {:?}, memory {:?}",
        info.plane_size,
        info.max_shared_memory,
        info.max_units_per_cube,
        info.max_cube_count,
        info.max_memory
    );
}

/// The CPU runtime reports f64 and allocates f64 buffers.
fn reports_f64(device: &mut Device) {
    assert!(device.supports(Precision::F64));
    device.require(Precision::F64).unwrap();
    assert_eq!(device.info().precisions(), [Precision::F32, Precision::F64]);
    let buffer = device.upload(&[1.5f64, -2.0]).unwrap();
    let mut out = [0.0f64; 2];
    device.download(buffer.as_slice(), &mut out).unwrap();
    assert_eq!(out, [1.5, -2.0]);
}

/// Metal came up with the MSL compiler, reports f32 only, and refuses f64 with the
/// typed error wherever it is asked for, never by a panic.
fn refuses_f64(device: &mut Device) {
    assert_eq!(device.info().compiler, "wgpu<msl>");
    assert!(!device.supports(Precision::F64));
    assert_eq!(device.info().precisions(), [Precision::F32]);
    let refused = KernelError::UnsupportedPrecision {
        backend: BackendKind::Metal,
        precision: Precision::F64,
    };
    assert_eq!(device.require(Precision::F64).unwrap_err(), refused);
    assert_eq!(device.alloc::<f64>(4).unwrap_err(), refused);
    assert_eq!(device.upload(&[1.0f64]).unwrap_err(), refused);
    println!("  f64 refused: {refused}");
}

/// A buffer of one device is refused by another.
fn refuses_foreign_buffers(device: &mut Device) {
    let mut other = Device::open(device.backend()).unwrap();
    let mut buffer = device.upload(&[1.0f32, 2.0]).unwrap();
    let mut out = [0.0f32; 2];
    assert_eq!(
        other.download(buffer.as_slice(), &mut out),
        Err(KernelError::WrongDevice)
    );
    assert_eq!(
        other.write(buffer.as_slice_mut(), &[0.0, 0.0]),
        Err(KernelError::WrongDevice)
    );
    assert_eq!(
        nd_fmm_kernels::movement::zero(&mut other, buffer.as_slice_mut()),
        Err(KernelError::WrongDevice)
    );
}

/// The counters count calls and bytes of transfers, launches and syncs.
fn counts_transfers_launches_and_syncs(device: &mut Device) {
    device.reset_counters();
    let mut a = device.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let indices = device.upload_indices_widened(&[2u16, 0]).unwrap();
    let mut b = device.alloc::<f32>(2).unwrap();
    nd_fmm_kernels::movement::gather_columns(
        device,
        1,
        a.as_slice(),
        indices.as_slice(),
        b.as_slice_mut(),
    )
    .unwrap();
    nd_fmm_kernels::movement::zero(device, a.slice_mut(1..1)).unwrap();
    device.write(a.slice_mut(..1), &[4.0]).unwrap();
    let mut out = [0.0f32; 2];
    device.download(b.as_slice(), &mut out).unwrap();
    assert_eq!(out, [3.0, 1.0]);
    device.sync().unwrap();
    let c = device.counters();
    println!("  {c:?}");
    assert_eq!((c.uploads, c.upload_bytes), (3, 12 + 8 + 4));
    assert_eq!((c.downloads, c.download_bytes), (1, 8));
    // The zeroing of `alloc` and the gather; the empty zero launches nothing.
    assert_eq!(c.launches, 2);
    assert_eq!(c.syncs, 2);
    device.reset_counters();
    assert_eq!(device.counters(), Default::default());
}

tests_on!(cpu: reports_f32, reports_f64, refuses_foreign_buffers, counts_transfers_launches_and_syncs);
tests_on!(metal: reports_f32, refuses_f64, refuses_foreign_buffers, counts_transfers_launches_and_syncs);
