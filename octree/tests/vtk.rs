use nd_octree::{PhysicalBox, morton, vtk};
fn values(xml: &str, name: &str) -> Vec<String> {
    xml.split(&format!("Name=\"{name}\""))
        .nth(1)
        .unwrap()
        .split_once('>')
        .unwrap()
        .1
        .split("</DataArray>")
        .next()
        .unwrap()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}
#[test]
fn geometry_attributes_and_empty() {
    let domain = PhysicalBox::new([-2., 3., 4., 6., 7., 16.]);
    let keys = [morton::root(), morton::from_index_and_level([65535; 3], 16)];
    let mut bytes = Vec::new();
    vtk::write_vtu(&mut bytes, &keys, &domain, 7).unwrap();
    let xml = String::from_utf8(bytes).unwrap();
    assert!(xml.contains("NumberOfPoints=\"16\" NumberOfCells=\"2\""));
    let points: Vec<f64> = xml
        .split("format=\"ascii\">")
        .nth(1)
        .unwrap()
        .split("</DataArray>")
        .next()
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let expected: Vec<f64> = keys
        .iter()
        .flat_map(|&k| {
            morton::physical_box(k, &domain)
                .corners()
                .into_iter()
                .flatten()
        })
        .collect();
    assert_eq!(points, expected);
    assert_eq!(
        &points[..12],
        &[-2., 3., 4., 6., 3., 4., 6., 7., 4., -2., 7., 4.]
    );
    assert_eq!(
        values(&xml, "connectivity"),
        (0..16).map(|v| v.to_string()).collect::<Vec<_>>()
    );
    assert_eq!(values(&xml, "offsets"), ["8", "16"]);
    assert_eq!(values(&xml, "types"), ["12", "12"]);
    assert_eq!(values(&xml, "refinement_level"), ["0", "16"]);
    assert_eq!(values(&xml, "owner_rank"), ["7", "7"]);
    assert!(keys[1] > (1 << 53));
    assert!(xml.contains("type=\"UInt64\" Name=\"morton_key\""));
    assert_eq!(values(&xml, "morton_key"), keys.map(|k| k.to_string()));
    let mut bytes = Vec::new();
    vtk::write_vtu(&mut bytes, &[], &domain, 0).unwrap();
    let xml = String::from_utf8(bytes).unwrap();
    assert!(xml.contains("NumberOfPoints=\"0\" NumberOfCells=\"0\""));
    for name in [
        "connectivity",
        "offsets",
        "types",
        "refinement_level",
        "owner_rank",
        "morton_key",
    ] {
        assert!(values(&xml, name).is_empty());
    }
}
#[test]
fn io_and_invalid_input() {
    let domain = PhysicalBox::new([0., 0., 0., 1., 1., 1.]);
    assert!(vtk::write_vtu(&mut [0u8; 0][..], &[0], &domain, 0).is_err());
    assert!(vtk::write_vtu(Vec::new(), &[morton::invalid_key()], &domain, 0).is_err());
}

#[test]
fn rejects_degenerate_and_nonfinite_domains() {
    let keys = [morton::root()];
    for coords in [
        [0., 0., 0., 0., 1., 1.],              // zero extent along x
        [0., 2., 0., 1., 1., 1.],              // ymax below ymin
        [0., 0., 0., f64::INFINITY, 1., 1.],   // non-finite bound
        [f64::NAN, 0., 0., 1., 1., 1.],        // NaN bound
        [-f64::MAX, 0., 0., f64::MAX, 1., 1.], // finite bounds, non-finite extent
    ] {
        assert!(vtk::write_vtu(Vec::new(), &keys, &PhysicalBox::new(coords), 0).is_err());
    }

    let domain = PhysicalBox::new([0., 0., 0., 1., 1., 1.]);
    // A key below the deepest supported level cannot be given a box.
    assert!(vtk::write_vtu(Vec::new(), &[17], &domain, 0).is_err());
    // One bad key rejects the whole batch, without writing a partial document.
    let mut bytes = Vec::new();
    assert!(
        vtk::write_vtu(
            &mut bytes,
            &[morton::root(), morton::invalid_key()],
            &domain,
            0
        )
        .is_err()
    );
    assert!(bytes.is_empty());
    // The same call with a sound domain and keys succeeds.
    assert!(vtk::write_vtu(Vec::new(), &keys, &domain, 0).is_ok());
}
