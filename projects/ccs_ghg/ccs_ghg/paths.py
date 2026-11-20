"""Project path resolution.

Two directories can vary per user:

* `DEV_DIR` — the repo root (containing this `ccs_ghg/` package). Auto-derived from this
  file's location (``parents[1]``); no configuration needed.

* `DATA_DIR` — the external data root (large shapefiles, pipeline input
  CSVs, etc.). Override with the ``CCS_GHG_DATA_DIR`` environment variable;
  defaults to ``<DEV_DIR>/data`` if not set.

Importing this module also registers both paths as environment variables
(``CCS_GHG_DEV_DIR`` and ``CCS_GHG_DATA_DIR``) so YAML configs loaded via
`ccs_ghg._vendored.io.yaml_to_dict` can reference them as ``${CCS_GHG_DEV_DIR}`` /
``${CCS_GHG_DATA_DIR}`` and have the placeholders expanded at load time.
`os.environ.setdefault` is used, so any user-supplied env-var values take
precedence over the defaults.
"""
import os
from pathlib import Path

DEV_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = Path(os.environ.get("CCS_GHG_DATA_DIR") or (DEV_DIR / "data"))

os.environ.setdefault("CCS_GHG_DEV_DIR", str(DEV_DIR))
os.environ.setdefault("CCS_GHG_DATA_DIR", str(DATA_DIR))
