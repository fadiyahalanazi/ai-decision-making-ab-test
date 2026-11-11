# -*- coding: utf-8 -*-
"""المرحلة الأولى: السؤال والفرضية وتصميم تجربة A/B."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
CONFIG = {
    "project": "AI Recommendations for Business Decision Accuracy",
    "project_ar": "أثر توصيات الذكاء الاصطناعي على دقة قرارات الأعمال",
    "baseline_conversion": 0.60,
    "mde_abs": 0.05,
    "alpha": 0.05,
    "power": 0.80,
    "daily_new_users": 200,
    "treatment_share": 0.5,
    "true_lift_abs": 0.05,
    "activity_strength": 0.8,
    "novelty_boost": 0.0,
    "srm_bug_share": 0.0,
    "seed": 2026,
}
DESIGN = {
    "question": "هل ترفع توصيات الذكاء الاصطناعي دقة قرارات الأعمال؟",
    "hypothesis": "دقة القرارات في B أعلى من A",
    "A": "لوحة بيانات دون توصيات AI",
    "B": "لوحة بيانات مع توصيات AI",
    "randomization_unit": "user",
    "allocation": "50/50",
    "primary_metric": "correct_decision",
    "secondary_metrics": ["decision_time", "satisfaction"],
    "guardrail": "high_impact_error",
    "SUTVA": "لا يتأثر مستخدم بتخصيص أو قرارات المستخدمين الآخرين",
    "data_type": "synthetic simulation",
}
def main():
    dest = ROOT / "outputs" / "results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "experiment_config.json").write_text(
        json.dumps({"config": CONFIG, "design": DESIGN}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("01 | Experiment question and design saved")
if __name__ == "__main__":
    main()
