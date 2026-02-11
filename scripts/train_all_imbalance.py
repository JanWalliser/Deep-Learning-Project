# scripts/train_all_imbalances_cnn.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

IMBALANCES = ["balanced", "severe", "strong", "extreme"]

BASE_CFG = Path("configs/base.yaml")
MODEL_CFG = Path("configs/model_fc.yaml")  # ggf. anpassen, falls anders benannt

def main() -> int:
    missing = [p for p in [BASE_CFG, MODEL_CFG] if not p.exists()]
    if missing:
        print(f"[ERR] Missing config(s): {missing}")
        return 1

    for imb in IMBALANCES:
        imb_cfg = Path(f"configs/imbalance_{imb}.yaml")
        if not imb_cfg.exists():
            print(f"[WARN] Skip missing: {imb_cfg}")
            continue

        cmd = [
            sys.executable, "-m", "src.main",
            "--config", str(BASE_CFG),
            "--config", "configs/task_multiclass.yaml",
            "--config", str(MODEL_CFG),
            "--config", str(imb_cfg),
            "--config", str(imb_cfg),
            "--config", "configs/opt_adamw.yaml",
        ]

        print("\n" + "=" * 80)
        print(f"[RUN] CNN + imbalance_{imb}")
        print(" ".join(cmd))
        print("=" * 80)

        # stoppt beim ersten Fehler (check=True)
        subprocess.run(cmd, check=True)

    print("\n[DONE] All imbalance configs finished.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
