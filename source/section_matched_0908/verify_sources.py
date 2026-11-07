import hashlib
import json
from pathlib import Path

report = json.loads(Path("section_matched_0908/preflight.json").read_text())
for name, expected in report["source_sha256"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, f"Original source changed: {name}"
print("[source check] original code/config hashes match preflight")
