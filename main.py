
# -*- coding: utf-8 -*-
"""
AI Recommendations for Business Decision Accuracy
SDAIA Academy - A/B Testing and Causal Inference
CMD-compatible main.py
"""

from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ============================================================
# 0. PROJECT SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHARTS_DIR = ROOT / "outputs" / "charts"
RESULTS_DIR = ROOT / "outputs" / "results"
REPORTS_DIR = ROOT / "reports"

for folder in [DATA_DIR, CHARTS_DIR, RESULTS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def save_json(filename, content):
    path = RESULTS_DIR / filename
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def say(message):
    print(message, flush=True)


def section(number, title):
    print("\n" + "=" * 60)
    print(f"{number} | {title}")
    print("=" * 60)


# ============================================================
# 1. QUESTION AND EXPERIMENT DESIGN
# ============================================================

section("01", "Question and Experiment Design")

CONFIG = {
    "project": "AI Recommendations for Business Decision Accuracy",
    "baseline_conversion": 0.60,
    "mde_abs": 0.05,
    "alpha": 0.05,
    "power": 0.80,
    "daily_new_users": 200,
    "treatment_share": 0.50,
    "true_lift_abs": 0.05,
    "activity_strength": 0.8,
    "novelty_boost": 0.0,
    "srm_bug_share": 0.0,
    "seed": 2026
}

DESIGN = {
    "question": (
        "Do AI recommendations improve business decision accuracy?"
    ),
    "hypothesis": (
        "AI recommendations improve decision accuracy."
    ),
    "randomization_unit": "User",
    "control_group": "A - Dashboard without AI recommendations",
    "treatment_group": "B - Dashboard with AI recommendations",
    "allocation": "50/50",
    "primary_metric": "correct_decision",
    "secondary_metrics": [
        "decision_time",
        "satisfaction"
    ],
    "guardrail_metric": "high_impact_error",
    "sutva": (
        "Users' decisions are not influenced by "
        "other users' assignments or recommendations."
    )
}

save_json(
    "experiment_config.json",
    {"config": CONFIG, "design": DESIGN}
)

say("Experiment question and design saved.")


# ============================================================
# 2. SAMPLE SIZE AND DURATION
# ============================================================

section("02", "Sample Size and Duration")


def n_per_group_proportions(p0, p1, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    variance = p0 * (1 - p0) + p1 * (1 - p1)

    n = ((z_alpha + z_beta) ** 2 * variance) / ((p1 - p0) ** 2)

    return int(np.ceil(n))


def mde_proportions(p0, n, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    return (
        (z_alpha + z_beta)
        * np.sqrt(2 * p0 * (1 - p0) / n)
    )


def duration_days(n, daily_users):
    days = int(np.ceil(2 * n / daily_users))
    return int(np.ceil(days / 7) * 7)


n_required = n_per_group_proportions(
    CONFIG["baseline_conversion"],
    CONFIG["baseline_conversion"] + CONFIG["mde_abs"],
    CONFIG["alpha"],
    CONFIG["power"]
)

days = duration_days(
    n_required,
    CONFIG["daily_new_users"]
)

n_total_required = n_required * 2
n_simulated = days * CONFIG["daily_new_users"]
n_planned_group = n_simulated // 2

sample_results = {
    "n_per_group_required": n_required,
    "n_total_required": n_total_required,
    "duration_days": days,
    "simulated_users": n_simulated,
    "expected_users_per_group": n_planned_group,
    "mde": CONFIG["mde_abs"],
    "alpha": CONFIG["alpha"],
    "power": CONFIG["power"]
}

save_json("sample_size.json", sample_results)

say(f"Required sample per group: {n_required:,}")
say(f"Required total sample: {n_total_required:,}")
say(f"Experiment duration: {days} days")
say(f"Simulated users: {n_simulated:,}")

planned_mde = mde_proportions(
    CONFIG["baseline_conversion"],
    n_planned_group,
    CONFIG["alpha"],
    CONFIG["power"]
)

say(f"Approximate detectable effect: {planned_mde:.2%}")


# ============================================================
# 3. ANALYSIS PLAN
# ============================================================

section("03", "Statistical Analysis Plan")

ANALYSIS_PLAN = {
    "primary_metric": "Decision accuracy",
    "primary_test": "Difference in proportions",
    "alpha": 0.05,
    "confidence_level": 0.95,
    "srm_test": "Chi-square goodness-of-fit",
    "srm_threshold": 0.001,
    "variance_reduction": "CUPED",
    "secondary_metrics": [
        "Decision time",
        "User satisfaction"
    ],
    "multiple_testing": "Benjamini-Hochberg",
    "guardrail": "High-impact decision errors",
    "analysis_timing": "After 21 days",
    "decision_rule": (
        "Launch only with statistically significant "
        "positive primary impact and acceptable guardrails."
    )
}

save_json("analysis_plan.json", ANALYSIS_PLAN)
say("Pre-specified analysis plan saved.")


# ============================================================
# 4. SIMULATE EXPERIMENT DATA
# ============================================================

section("04", "Simulate Experiment Data")


def simulate_experiment(cfg, experiment_days):
    r = np.random.default_rng(cfg["seed"])

    n = experiment_days * cfg["daily_new_users"]

    is_b = (
        r.random(n) < cfg["treatment_share"]
    ).astype(int)

    day = r.integers(1, experiment_days + 1, n)

    experience = r.normal(0, 1, n)

    baseline = cfg["baseline_conversion"]

    logit = (
        np.log(baseline / (1 - baseline))
        + cfg["activity_strength"] * experience
    )

    p0 = 1 / (1 + np.exp(-logit))
    p0 = np.clip(p0 * baseline / p0.mean(), 0, 1)

    pre_decisions_90d = r.poisson(
        5 * p0 / baseline
    )

    lift = cfg["true_lift_abs"] * (
        1
        + cfg["novelty_boost"]
        * np.exp(-(day - 1) / 3)
    )

    correct_decision = r.binomial(
        1,
        np.clip(p0 + lift * is_b, 0, 1)
    )

    decision_time = np.maximum(
        10,
        r.normal(90 - 15 * is_b, 20, n)
    )

    satisfaction = np.clip(
        np.rint(
            r.normal(3.5 + 0.3 * is_b, 0.8, n)
        ),
        1,
        5
    ).astype(int)

    high_impact_error = (
        (correct_decision == 0)
        & (r.random(n) < 0.20)
    ).astype(int)

    df = pd.DataFrame({
        "variant": np.where(is_b == 1, "B", "A"),
        "day": day,
        "pre_decisions_90d": pre_decisions_90d,
        "correct_decision": correct_decision,
        "decision_time": decision_time,
        "satisfaction": satisfaction,
        "high_impact_error": high_impact_error
    })

    if cfg["srm_bug_share"] > 0:
        lost = (
            (df["variant"] == "B")
            & (experience < 0)
            & (
                r.random(n)
                < cfg["srm_bug_share"] * 2
            )
        )
        df = df[~lost]

    return df.reset_index(drop=True)


data = simulate_experiment(CONFIG, days)

data.to_csv(
    DATA_DIR / "dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

say(f"Dataset saved: {len(data):,} users")


# ============================================================
# 5. DATA QUALITY AND SRM
# ============================================================

section("05", "Data Quality and SRM")

counts = data["variant"].value_counts()

n_a = int(counts.get("A", 0))
n_b = int(counts.get("B", 0))

chi2, srm_p = stats.chisquare(
    [n_a, n_b],
    f_exp=[
        len(data) * (1 - CONFIG["treatment_share"]),
        len(data) * CONFIG["treatment_share"]
    ]
)

srm_ok = bool(srm_p >= 0.001)

say(f"Group A: {n_a:,}")
say(f"Group B: {n_b:,}")
say(f"SRM p-value: {srm_p:.6f}")
say(f"SRM status: {'PASS' if srm_ok else 'FAIL'}")


# ============================================================
# 6. STATISTICAL ANALYSIS
# ============================================================

section("06", "Statistical Analysis")


def diff_ci(a, b, alpha=0.05):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    diff = b.mean() - a.mean()

    se = np.sqrt(
        a.var(ddof=1) / len(a)
        + b.var(ddof=1) / len(b)
    )

    z = stats.norm.ppf(1 - alpha / 2)

    low = diff - z * se
    high = diff + z * se

    p = 2 * stats.norm.sf(abs(diff / se))

    return {
        "difference": float(diff),
        "ci_low": float(low),
        "ci_high": float(high),
        "standard_error": float(se),
        "p_value": float(p)
    }


A = data[data["variant"] == "A"]
B = data[data["variant"] == "B"]

primary = diff_ci(
    A["correct_decision"],
    B["correct_decision"]
)

accuracy_a = float(A["correct_decision"].mean())
accuracy_b = float(B["correct_decision"].mean())

say(f"Accuracy A: {accuracy_a:.2%}")
say(f"Accuracy B: {accuracy_b:.2%}")
say(f"Absolute lift: {primary['difference'] * 100:.2f} pp")
say(f"Primary p-value: {primary['p_value']:.4f}")

say(
    "95% CI: "
    f"[{primary['ci_low'] * 100:.2f}, "
    f"{primary['ci_high'] * 100:.2f}] pp"
)


# ============================================================
# 7. CUPED VARIANCE REDUCTION
# ============================================================

section("07", "CUPED Analysis")

theta = (
    np.cov(
        data["correct_decision"],
        data["pre_decisions_90d"]
    )[0, 1]
    / data["pre_decisions_90d"].var(ddof=1)
)

adjusted = (
    data["correct_decision"]
    - theta
    * (
        data["pre_decisions_90d"]
        - data["pre_decisions_90d"].mean()
    )
)

cuped = diff_ci(
    adjusted[data["variant"] == "A"],
    adjusted[data["variant"] == "B"]
)

variance_reduction = (
    1
    - (
        cuped["standard_error"]
        / primary["standard_error"]
    ) ** 2
)

say(f"CUPED lift: {cuped['difference'] * 100:.2f} pp")
say(f"CUPED p-value: {cuped['p_value']:.4f}")
say(f"Variance reduction: {variance_reduction:.2%}")


# ============================================================
# 8. SECONDARY METRICS AND BH CORRECTION
# ============================================================

section("08", "Secondary Metrics")

secondary = {
    "decision_time": diff_ci(
        A["decision_time"],
        B["decision_time"]
    ),
    "satisfaction": diff_ci(
        A["satisfaction"],
        B["satisfaction"]
    )
}

secondary_names = list(secondary.keys())

p_values = [
    secondary[name]["p_value"]
    for name in secondary_names
]

rejected, corrected_p, _, _ = multipletests(
    p_values,
    alpha=CONFIG["alpha"],
    method="fdr_bh"
)

for i, name in enumerate(secondary_names):
    secondary[name]["adjusted_p_value"] = float(corrected_p[i])
    secondary[name]["significant_after_bh"] = bool(rejected[i])

    say(
        f"{name}: "
        f"Difference={secondary[name]['difference']:.3f}, "
        f"Adjusted p={corrected_p[i]:.4f}"
    )


# ============================================================
# 9. GUARDRAIL ANALYSIS
# ============================================================

section("09", "Guardrail Analysis")

guardrail = diff_ci(
    A["high_impact_error"],
    B["high_impact_error"]
)

guardrail_a = float(A["high_impact_error"].mean())
guardrail_b = float(B["high_impact_error"].mean())

guardrail_harm = guardrail["ci_low"] > 0
guardrails_ok = not guardrail_harm

say(f"High-impact errors A: {guardrail_a:.2%}")
say(f"High-impact errors B: {guardrail_b:.2%}")
say(f"Guardrail status: {'PASS' if guardrails_ok else 'FAIL'}")


# ============================================================
# 10. NOVELTY EFFECT AND CHARTS
# ============================================================

section("10", "Charts and Novelty Effect")

daily = (
    data.groupby(["day", "variant"])["correct_decision"]
    .mean()
    .unstack()
)

daily_lift = (daily["B"] - daily["A"]) * 100

plt.figure(figsize=(10, 5))
plt.plot(
    daily_lift.index,
    daily_lift.values,
    marker="o",
    color="#0E7490",
    label="Daily treatment effect"
)

plt.axhline(
    primary["difference"] * 100,
    color="#E58A1F",
    linestyle="--",
    label="Overall effect"
)

plt.axhline(0, color="gray", linewidth=1)

plt.title("AI Recommendations - Effect Over Time")
plt.xlabel("Experiment Day")
plt.ylabel("Effect (Percentage Points)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "novelty_effect.png",
    dpi=180
)
plt.close()

# Decision accuracy comparison
plt.figure(figsize=(7, 5))
plt.bar(
    ["Group A", "Group B"],
    [accuracy_a * 100, accuracy_b * 100],
    color=["#0E7490", "#E58A1F"]
)
plt.ylabel("Decision Accuracy (%)")
plt.title("Decision Accuracy by Experiment Group")
plt.tight_layout()
plt.savefig(
    CHARTS_DIR / "decision_accuracy.png",
    dpi=180
)
plt.close()

# Confidence interval
plt.figure(figsize=(8, 4))

effect = primary["difference"] * 100
low = primary["ci_low"] * 100
high = primary["ci_high"] * 100

plt.errorbar(
    effect,
    0,
    xerr=[[effect - low], [high - effect]],
    fmt="o",
    capsize=6,
    color="#0E7490"
)

plt.axvline(0, color="gray", linestyle="--")
plt.axvline(5, color="#E58A1F", linestyle=":")
plt.yticks([0], ["Primary Effect"])
plt.xlabel("Effect (Percentage Points)")
plt.title("95% Confidence Interval")
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "confidence_intervals.png",
    dpi=180
)
plt.close()

say("Three charts saved in outputs/charts/")


# ============================================================
# 11. OPTIONAL DIFFERENCE-IN-DIFFERENCES EXAMPLE
# ============================================================

section("11", "Optional Difference-in-Differences")


def simulate_policy_panel(
    n_units=20,
    n_treated=6,
    n_periods=24,
    policy_period=13,
    effect=8.0,
    seed=32
):
    r = np.random.default_rng(seed)
    level = r.normal(100, 15, n_units)
    shock = r.normal(0, 2, n_periods)

    rows = []

    for i in range(n_units):
        for t in range(1, n_periods + 1):
            treated = int(i < n_treated)
            post = int(t >= policy_period)

            y = (
                level[i]
                + 0.8 * t
                + shock[t - 1]
                + effect * treated * post
                + r.normal(0, 3)
            )

            rows.append({
                "unit": i,
                "period": t,
                "treated": treated,
                "post": post,
                "y": y
            })

    df = pd.DataFrame(rows)
    df["treated_post"] = df["treated"] * df["post"]

    return df


policy = simulate_policy_panel()

twfe = smf.ols(
    "y ~ treated_post + C(unit) + C(period)",
    data=policy
).fit(
    cov_type="cluster",
    cov_kwds={"groups": policy["unit"]}
)

did_low, did_high = twfe.conf_int().loc["treated_post"]

say(f"Optional DiD effect: {twfe.params['treated_post']:.2f}")
say(f"Optional DiD 95% CI: [{did_low:.2f}, {did_high:.2f}]")


# ============================================================
# 12. FINAL DECISION
# ============================================================

section("12", "Final Decision")


def decide(ci_low, ci_high, mde, guardrails_ok=True):
    if ci_high < 0:
        return "Do not launch: negative primary effect."

    if ci_low > 0 and not guardrails_ok:
        return "Do not launch: guardrail deterioration."

    if ci_low > 0:
        return "Launch: statistically significant positive effect."

    if ci_high < mde:
        return "Do not launch: target effect is not supported."

    return "Inconclusive: consider a redesigned experiment."


decision = decide(
    primary["ci_low"],
    primary["ci_high"],
    CONFIG["mde_abs"],
    guardrails_ok
)

if not srm_ok:
    decision = "Do not make a launch decision: SRM detected."

analysis_results = {
    "group_a_size": n_a,
    "group_b_size": n_b,
    "accuracy_a": accuracy_a,
    "accuracy_b": accuracy_b,
    "srm_p_value": float(srm_p),
    "srm_passed": srm_ok,
    "primary": primary,
    "cuped": cuped,
    "cuped_variance_reduction": float(variance_reduction),
    "secondary": secondary,
    "guardrail": {
        "group_a": guardrail_a,
        "group_b": guardrail_b,
        "test": guardrail,
        "passed": bool(guardrails_ok)
    },
    "optional_did": {
        "effect": float(twfe.params["treated_post"]),
        "ci_low": float(did_low),
        "ci_high": float(did_high)
    }
}

save_json("analysis_results.json", analysis_results)

final_decision = {
    "decision": decision,
    "launch": decision.startswith("Launch:"),
    "primary_significant": bool(
        primary["p_value"] < CONFIG["alpha"]
    ),
    "guardrails_ok": bool(guardrails_ok),
    "srm_ok": bool(srm_ok),
    "recommendation": (
        "Improve AI recommendations and run another A/B test."
    ),
    "limitations": (
        "Results are based on simulated data, not real users."
    )
}

save_json("final_decision.json", final_decision)

say(f"Decision: {decision}")


# ============================================================
# 13. ONE-PAGE REPORT
# ============================================================

section("13", "Final Report")

annual_users = 50_000
annual_impact = primary["difference"] * annual_users

report = f"""# AI Recommendations for Business Decision Accuracy

## Experiment Overview

This project evaluates whether AI recommendations improve
business decision accuracy compared with a dashboard alone.

The experiment uses simulated data.

## Experiment Design

- Group A: Dashboard without AI recommendations
- Group B: Dashboard with AI recommendations
- Randomization unit: User
- Allocation: 50/50
- Primary metric: Decision accuracy
- Secondary metrics: Decision time and satisfaction
- Guardrail: High-impact decision errors
- Duration: {days} days
- Total simulated users: {len(data):,}

## Results

| Metric | Result |
|---|---|
| Accuracy A | {accuracy_a:.2%} |
| Accuracy B | {accuracy_b:.2%} |
| Primary effect | {primary['difference'] * 100:.2f} pp |
| 95% CI | [{primary['ci_low'] * 100:.2f}, {primary['ci_high'] * 100:.2f}] pp |
| Primary p-value | {primary['p_value']:.4f} |
| CUPED effect | {cuped['difference'] * 100:.2f} pp |
| CUPED p-value | {cuped['p_value']:.4f} |
| CUPED variance reduction | {variance_reduction:.2%} |
| High-impact errors A | {guardrail_a:.2%} |
| High-impact errors B | {guardrail_b:.2%} |
| SRM check | {'PASS' if srm_ok else 'FAIL'} |

## Final Decision

**{decision}**

The observed improvement is not statistically significant
at the 5% significance level.

The confidence interval includes zero, so the experiment
does not establish a positive effect on decision accuracy.

## Recommendation

Improve the AI recommendation system and evaluate it
through another properly designed A/B experiment.

## Estimated Annual Impact

The observed difference corresponds to approximately
{annual_impact:,.0f} additional correct decisions per
50,000 decisions, if the observed effect were sustained.

This is an illustrative extrapolation, not a confirmed benefit.

## Limitations

All results are based on simulated data and do not
demonstrate real-world effectiveness.
"""

(REPORTS_DIR / "One_Page_Report.md").write_text(
    report,
    encoding="utf-8"
)

say("Report saved: reports/One_Page_Report.md")

print("\n" + "=" * 60)
print("Main project completed successfully.")
print("=" * 60)

print("\nGenerated files:")
print("- data/dataset.csv")
print("- outputs/charts/")
print("- outputs/results/")
print("- reports/One_Page_Report.md")

