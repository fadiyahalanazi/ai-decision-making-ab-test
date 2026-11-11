# -*- coding: utf-8 -*-
"""المرحلة الثانية: حساب حجم العينة ومدة التجربة وMDE."""
from pathlib import Path
import json
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent

def n_per_group_proportions(p_control, p_treatment, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    return int(np.ceil((z_a + z_b) ** 2 * var / (p_treatment - p_control) ** 2))

def mde_proportions(p_control, n_per_group, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((z_a + z_b) * np.sqrt(2 * p_control * (1 - p_control) / n_per_group))

def duration_days(n_per_group, daily_new_users, groups=2):
    days = int(np.ceil(n_per_group * groups / daily_new_users))
    return int(np.ceil(days / 7) * 7)

def main():
    cfg = json.loads((ROOT / "outputs/results/experiment_config.json").read_text(encoding="utf-8"))["config"]
    n = n_per_group_proportions(cfg["baseline_conversion"], cfg["baseline_conversion"] + cfg["mde_abs"], cfg["alpha"], cfg["power"])
    days = duration_days(n, cfg["daily_new_users"])
    planned = days * cfg["daily_new_users"] // 2
    result = {
        "n_per_group_required": n, "n_total_required": 2*n,
        "duration_days": days, "planned_users": 2*planned,
        "planned_per_group": planned,
        "mde_planned": mde_proportions(cfg["baseline_conversion"], planned, cfg["alpha"], cfg["power"]),
        "mde_one_week": mde_proportions(cfg["baseline_conversion"], 7*cfg["daily_new_users"]//2, cfg["alpha"], cfg["power"])
    }
    (ROOT / "outputs/results/sample_size.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("02 | Sample size and duration:", result)
if __name__ == "__main__":
    main()
