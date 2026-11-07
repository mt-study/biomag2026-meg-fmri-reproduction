"""Check encoder split provenance; export original trainer's saved metrics."""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--seed", required=True, type=int)
p.add_argument("--check-lineage", action="store_true")
p.add_argument("--checkpoint")
p.add_argument("--output")
a = p.parse_args()
cfg = yaml.safe_load(Path(a.config).read_text())
name = Path(a.config).stem.rsplit("_", 1)[0]
enc_path = Path(f"results/section_matched_0908/{name}/encoder/all_subjects/exhaustive/seed_42/encoder_result.json")
enc = json.loads(enc_path.read_text())
assert enc["split"]["train_sections"] == cfg["alignment"]["train_sections"]
assert enc["split"]["test_sections"] == cfg["alignment"]["val_sections"]
assert enc["split"]["data_mode"] == "exhaustive"
assert enc["seed"] == 42
assert Path(enc["checkpoint"]).resolve() == Path(cfg["model"]["init_from"]).resolve()
assert Path(enc["checkpoint"]).is_file()
if a.check_lineage:
    print("[lineage check] frozen encoder trained on the matching seven sections")
else:
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["config"]["training"]["seed"] == a.seed
    assert ckpt["config"]["alignment"] == cfg["alignment"]
    result = {
        "split": name, "seed": a.seed, "mode": cfg["model"]["fusion_mode"],
        "config": ckpt["config"], "selected_epoch": ckpt["epoch"],
        "encoder_result": enc, "checkpoint": a.checkpoint,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "metrics": ckpt["val_metrics"],
        "evaluation_role": "historical selection/reporting protocol; no extra validation set",
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[result] {out}")
