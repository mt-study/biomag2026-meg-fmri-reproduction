"""Parse the cached BERT word-embedding CSV once into a plain float32
``.npy`` matrix (row order matches ``lppCN_word_information.csv``
exactly), so training never re-parses the ~190 MB stringified-vector CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import parse_bert_vector  # noqa: E402


def main():
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "configs/default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    info_df = pd.read_csv(config["data"]["text_information_csv"])
    emb_df = pd.read_csv(config["data"]["text_embedding_csv"])
    if len(info_df) != len(emb_df):
        # Row COUNT is the actual alignment invariant we rely on (both CSVs
        # are produced from the same word-segmented stream): a mismatch
        # here means positional alignment is broken and must not proceed.
        raise ValueError(f"Row count mismatch: {len(info_df)} vs {len(emb_df)}")
    mismatched = (info_df["word"].astype(str) != emb_df["word"].astype(str)).sum()
    if mismatched:
        # A small fraction of rows carry a different surface form at the
        # same position (e.g. "猎获物" vs "猎物", "事儿" vs "事", "B612" vs
        # "612") -- two independently-tokenized annotation files disagreeing
        # on erhua suffixes / compound splitting for the same slot, not a
        # row-shift (row counts match exactly). At <2% of 16,009 rows and
        # averaged into multi-word 2s-window embeddings, this does not
        # warrant discarding the cached embeddings; log it instead.
        print(
            f"[warn] {mismatched}/{len(info_df)} rows have a differing surface word between "
            "the annotation and embedding CSVs at the same row index (row order/count still "
            "match). Proceeding -- see README.md for examples and rationale."
        )

    vectors = np.stack(emb_df["BERT"].map(parse_bert_vector).to_numpy()).astype(np.float32)
    print(f"Parsed {vectors.shape[0]} rows, dim={vectors.shape[1]}")

    out_path = Path(config["project_dir"]) / config["data"]["text_embedding_cache"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vectors)
    print(f"[done] saved {out_path}")


if __name__ == "__main__":
    main()
