#!/usr/bin/env python3
"""Generate site-specific YAMLs without changing archived scientific settings."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HISTORIC_PROJECTS = (
    "/gpfs/userdata/ldeng/Speech-BCI/biomag2026/0907/meg_fmri_clip_cc_0907",
    "/gpfs/userdata/ldeng/Speech-BCI/biomag_ldeng/meg_fmri_clip_cc",
)
HISTORIC_OUTPUT = "/gpfs/userdata/ldeng/Speech-BCI/biomag2026"
HISTORIC_DATA = "/gpfs/userdata/ldeng/dataset/speech"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Parent of MEG/lpp_data_challenge and fMRI/ds003643-download")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root for new checkpoints, results, and logs")
    parser.add_argument("--dest", type=Path, default=REPO / ".generated" / "configs")
    args = parser.parse_args()
    source_root = (REPO / "source").resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dest = args.dest.expanduser().resolve()
    inputs = sorted((REPO / "source" / "configs").rglob("*.yaml"))
    inputs += sorted((REPO / "experiments").rglob("*.yaml"))
    unresolved = []
    for path in inputs:
        text = path.read_text(encoding="utf-8")
        for old in HISTORIC_PROJECTS:
            text = text.replace(old, str(source_root))
        text = text.replace(HISTORIC_DATA, str(data_root))
        text = text.replace(HISTORIC_OUTPUT, str(output_root))
        if "/gpfs/userdata/ldeng" in text or "/share/home/ldeng" in text:
            unresolved.append(str(path.relative_to(REPO)))
        target = dest / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(f"Wrote {len(inputs)} YAML files to {dest}")
    if unresolved:
        print("Unresolved historical paths in:")
        for path in unresolved:
            print(f"  {path}")
        return 1
    print("Only paths were changed; archived YAMLs are untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
