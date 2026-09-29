//! `CONVENTION_VERSION` must match the version stated in `docs/CONVENTIONS.md`.

use nd_fmm_math::CONVENTION_VERSION;

#[test]
fn convention_version_matches_conventions_file() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/CONVENTIONS.md");
    let text = std::fs::read_to_string(path).expect("docs/CONVENTIONS.md must be readable");

    let marker = "`CONVENTION_VERSION = ";
    let versions: Vec<u32> = text
        .match_indices(marker)
        .map(|(start, _)| {
            let rest = &text[start + marker.len()..];
            let end = rest
                .find('`')
                .expect("version statement must end with a backtick");
            rest[..end]
                .parse()
                .expect("version must be an unsigned integer")
        })
        .collect();

    assert!(
        !versions.is_empty(),
        "docs/CONVENTIONS.md states no CONVENTION_VERSION"
    );
    for version in versions {
        assert_eq!(version, CONVENTION_VERSION);
    }
}
