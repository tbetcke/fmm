#!/usr/bin/env bash
# Runs the examples registered for the weekly run-examples job.
#
# Usage: run-examples.sh [nprocesses]   (default 3)
#
# An example is registered when its package's Cargo.toml has an [[example]] section and
#
#     [package.metadata.example.<name>.templated-examples]
#     command = "mpirun -n {{NPROCESSES}}"
#
# (root CLAUDE.md, "MPI"). For each, found through `cargo metadata` under the package's
# real name (packages are nd-*, their directories not), this builds the example in
# release with its required-features and runs `<command> <binary>`, {{NPROCESSES}}
# replaced by the argument, each run under a 10-minute timeout. Unregistered examples are
# not run. It reports every example and exits non-zero if any failed.
#
# It replaces `cargo templated-examples`, which passes a member's directory as its
# package name and so found none of this workspace's packages (and ran every example,
# registered or not), and `cargo mpirun`, which the job did not install.
set -euo pipefail

nprocesses=${1:-3}

metadata=$(cargo metadata --format-version 1 --no-deps)
target_dir=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])' <<<"$metadata")

# One line per registered example: package, example, required features (comma-separated),
# command template, separated by the unit separator (a tab, being whitespace to `read`,
# would merge an empty features field into its neighbours). A registration naming no
# [[example]] target is an error.
examples=$(python3 -c '
import json, sys
meta = json.load(sys.stdin)
for package in meta["packages"]:
    registered = ((package.get("metadata") or {}).get("example") or {})
    targets = {t["name"]: t for t in package["targets"] if "example" in t["kind"]}
    for name, settings in sorted(registered.items()):
        command = (settings or {}).get("templated-examples", {}).get("command")
        if command is None:
            continue
        if name not in targets:
            sys.exit(package["name"] + ": example " + name + " is registered but has no [[example]] target")
        features = ",".join(targets[name].get("required-features", []))
        print("\x1f".join([package["name"], name, features, command]))
' <<<"$metadata")

if [ -z "$examples" ]; then
    echo "No registered examples found." >&2
    exit 1
fi

passed=0
failed=()
while IFS=$'\x1f' read -r package name features command; do
    command=${command//\{\{NPROCESSES\}\}/$nprocesses}
    echo
    echo "=== $package: example $name (${command})"
    build=(cargo build --release -p "$package" --example "$name")
    if [ -n "$features" ]; then
        build+=(--features "$features")
    fi
    # The command is a template from Cargo.toml, split into words as the shell would.
    # shellcheck disable=SC2206
    run=(timeout 600 $command "$target_dir/release/examples/$name")
    # Both read nothing: mpirun would otherwise swallow the rest of the list on stdin.
    if "${build[@]}" </dev/null && "${run[@]}" </dev/null; then
        passed=$((passed + 1))
    else
        failed+=("$package/$name")
    fi
done <<<"$examples"

echo
echo "SUMMARY: ${passed} passed, ${#failed[@]} failed"
if [ "${#failed[@]}" -gt 0 ]; then
    printf '  failed: %s\n' "${failed[@]}"
    exit 1
fi
