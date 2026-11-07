"""Generate only split/path variants; verify against the live original configs."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from exhaustive_subject_pair import build_all_subject_datasets
from train_encoder import build_unified_datasets

ROOT = Path.cwd()
OUT = ROOT / "section_matched_0908"
SOURCES = {
    "encoder": ROOT / "configs/unified_benchmark_task1529545_matched.yaml",
    "mlp": ROOT / "configs/unified_fixed_encoder_legacy_mlp.yaml",
    "attention": ROOT / "configs/unified_fixed_encoder_legacy_attention.yaml",
}


def differences(a, b, prefix=""):
    found = []
    for key in sorted(set(a) | set(b)):
        name = f"{prefix}.{key}" if prefix else key
        if key not in a or key not in b:
            found.append(name)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            found.extend(differences(a[key], b[key], name))
        elif a[key] != b[key]:
            found.append(name)
    return found


def main():
    report = {"config_differences": {}, "dataset_checks": {}, "source_sha256": {}}
    # Everything upstream of main is identical except split validation/logging.
    original = ast.parse((ROOT / "unified_benchmark.py").read_text())
    variant = ast.parse((OUT / "train_encoder.py").read_text())
    functions = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
    checked = []
    for node in variant.body:
        if isinstance(node, ast.FunctionDef) and node.name not in {"main", "build_unified_datasets"}:
            assert ast.dump(node) == ast.dump(functions[node.name]), node.name
            checked.append(node.name)
    # Main follows the same statements through the encoder training call.
    original_main = functions["main"].body
    variant_main = next(n for n in variant.body if isinstance(n, ast.FunctionDef) and n.name == "main").body
    def training_prefix(body):
        selected = []
        for node in body:
            # The only changed pre-training statement is selection logging.
            if isinstance(node, ast.If) and "best epoch on sections" in ast.unparse(node):
                continue
            selected.append(ast.dump(node))
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "encoder_checkpoint" for t in node.targets):
                return selected
        raise AssertionError("encoder training call missing")
    assert training_prefix(original_main) == training_prefix(variant_main)
    report["unchanged_encoder_functions"] = checked
    report["encoder_execution_prefix_matches"] = True
    for name, heldout in [("holdout_45", [4, 5]), ("holdout_12", [1, 2])]:
        train = [s for s in range(1, 10) if s not in heldout]
        for stage, source in SOURCES.items():
            old = yaml.safe_load(source.read_text())
            cfg = copy.deepcopy(old)
            if stage == "encoder":
                cfg["protocol"]["train_sections"] = train
                cfg["protocol"]["test_sections"] = heldout
                for key, parent in [("checkpoint_root", "checkpoints"), ("result_root", "results")]:
                    cfg["output"][key] = f"{parent}/section_matched_0908/{name}/encoder"
                allowed = {"protocol.train_sections", "protocol.test_sections", "output.checkpoint_root", "output.result_root"}
                ds, info, dims = build_unified_datasets(cfg, "all_subjects", "exhaustive", None, seed=42)
                tr, ev = ds["train"].window_indices, ds["test"].window_indices
                assert np.array_equal(ev, ds["selection"].window_indices)
                assert ds["train"].data_mode == ds["test"].data_mode == "exhaustive"
                # The original selection function averages the subject axis
                # inside evaluate_selection_mean; the dataset stays exhaustive.
                assert ds["selection"].data_mode == "exhaustive"
            else:
                cfg["alignment"]["train_sections"] = train
                cfg["alignment"]["val_sections"] = heldout
                cfg["model"]["init_from"] = f"checkpoints/section_matched_0908/{name}/encoder/all_subjects/exhaustive/seed_42/encoder_best.pt"
                cfg["training"]["checkpoint_dir"] = f"checkpoints/section_matched_0908/{name}/{stage}/s42"
                allowed = {"alignment.train_sections", "alignment.val_sections", "model.init_from", "training.checkpoint_dir"}
                train_ds, eval_ds, info, dims = build_all_subject_datasets(cfg)
                tr, ev = train_ds.indices, eval_ds.indices
                assert train_ds.meg.shape[1] == 20 and train_ds.fmri.shape[1] == 35
                assert cfg["model"]["joint_hidden_dim"] == (512 if stage == "mlp" else 767)
                assert cfg["model"]["freeze_encoders"] is True
            changed = differences(old, cfg)
            assert set(changed) == allowed, (name, stage, changed)
            assert not set(tr) & set(ev)
            assert len(tr) + len(ev) == 2931
            key = f"{name}_{stage}"
            report["config_differences"][key] = changed
            report["dataset_checks"][key] = {"train_windows": len(tr), "evaluation_windows": len(ev), "dims": dims}
            (OUT / f"{key}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    paths = list(SOURCES.values()) + [ROOT / p for p in [
        "unified_benchmark.py", "train_subject_pair_exhaustive.py", "exhaustive_subject_pair.py",
        "cross_subject_generalization.py", "window_selection.py", "models/brain_clip.py", "models/encoders.py",
    ]]
    for path in paths:
        report["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["config_differences"], indent=2))
    print("PREFLIGHT PASSED: only split and output/checkpoint paths differ; original encoder execution preserved.")


if __name__ == "__main__":
    main()
