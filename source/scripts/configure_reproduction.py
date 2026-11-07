#!/usr/bin/env python3
"""Configure a copied repository for a new project and dataset location."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path


CONFIG_PATH_KEYS = {
    "fmri_root": lambda root: root / "fMRI" / "ds003643-download",
    "meg_root": lambda root: root / "MEG" / "lpp_data_challenge",
    "text_information_csv": lambda root: (
        root
        / "fMRI"
        / "ds003643-download"
        / "annotation"
        / "CN"
        / "lppCN_word_information.csv"
    ),
    "text_embedding_csv": lambda root: (
        root
        / "fMRI"
        / "ds003643-download"
        / "annotation"
        / "CN"
        / "lppCN_word_embeddings_BERT.csv"
    ),
}


def replace_yaml_scalar(text: str, key: str, value: Path) -> tuple[str, int]:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}:\s*).*$")
    return pattern.subn(lambda match: f"{match.group(1)}{value}", text)


def configure_yaml(path: Path, project_dir: Path, dataset_root: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated, _ = replace_yaml_scalar(original, "project_dir", project_dir)
    for key, builder in CONFIG_PATH_KEYS.items():
        updated, _ = replace_yaml_scalar(updated, key, builder(dataset_root))
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def configure_sbatch(
    path: Path, project_dir: Path, fmri_root: Path, conda_env: str
) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = re.sub(
        r"(?m)^cd\s+\S*meg_fmri_[^\s]+$",
        lambda _match: f"cd {shlex.quote(str(project_dir))}",
        original,
    )
    updated = re.sub(
        r"(?m)(ls -d )\S+(/derivatives/sub-CN\*)",
        lambda match: f"{match.group(1)}{fmri_root}{match.group(2)}",
        updated,
    )
    updated = re.sub(
        r"(?m)^conda activate\s+\S+\s*$",
        lambda _match: f"conda activate {shlex.quote(conda_env)}",
        updated,
    )
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def validate(
    project_dir: Path, dataset_root: Path, config_paths: list[Path], sbatch_paths: list[Path]
) -> list[str]:
    expected = {
        "project_dir": project_dir,
        **{key: builder(dataset_root) for key, builder in CONFIG_PATH_KEYS.items()},
    }
    problems: list[str] = []
    for path in config_paths:
        text = path.read_text(encoding="utf-8")
        for key, value in expected.items():
            match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*(.+?)\s*$", text)
            if match and match.group(1) != str(value):
                problems.append(f"{path}: {key}={match.group(1)!r}, expected {str(value)!r}")
    for path in sbatch_paths:
        text = path.read_text(encoding="utf-8")
        project_lines = re.findall(r"(?m)^cd\s+(.+?)\s*$", text)
        if project_lines and shlex.split(project_lines[0])[0] != str(project_dir):
            problems.append(
                f"{path}: project cd={project_lines[0]!r}, expected {str(project_dir)!r}"
            )
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Point all experiment configs and Slurm scripts at one dataset root. "
            "Expected layout: DATASET_ROOT/MEG/lpp_data_challenge and "
            "DATASET_ROOT/fMRI/ds003643-download."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="copied repository root; defaults to the parent of scripts/",
    )
    parser.add_argument("--conda-env", default="biomag_0530")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="write paths even when the dataset is not mounted yet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    fmri_root = CONFIG_PATH_KEYS["fmri_root"](dataset_root)

    required = [
        CONFIG_PATH_KEYS["meg_root"](dataset_root),
        fmri_root,
        CONFIG_PATH_KEYS["text_information_csv"](dataset_root),
        CONFIG_PATH_KEYS["text_embedding_csv"](dataset_root),
    ]
    missing = [path for path in required if not path.exists()]
    if missing and not args.allow_missing:
        print("Dataset layout check failed. Missing:")
        for path in missing:
            print(f"  - {path}")
        print("Use --allow-missing only when these paths will be mounted later.")
        return 2

    config_paths = sorted((project_dir / "configs").rglob("*.yaml"))
    sbatch_paths = sorted((project_dir / "scripts").glob("*.sbatch"))
    if not config_paths:
        print(f"No YAML configs found under {project_dir / 'configs'}")
        return 2

    changed_configs = 0
    changed_sbatch = 0
    if not args.check_only:
        changed_configs = sum(
            configure_yaml(path, project_dir, dataset_root) for path in config_paths
        )
        changed_sbatch = sum(
            configure_sbatch(path, project_dir, fmri_root, args.conda_env)
            for path in sbatch_paths
        )

    problems = validate(project_dir, dataset_root, config_paths, sbatch_paths)
    print(f"project_dir={project_dir}")
    print(f"dataset_root={dataset_root}")
    print(f"configs={len(config_paths)} changed={changed_configs}")
    print(f"sbatch={len(sbatch_paths)} changed={changed_sbatch}")
    if problems:
        print("Configuration validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Configuration validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
