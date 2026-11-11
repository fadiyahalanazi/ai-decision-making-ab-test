# -*- coding: utf-8 -*-
"""المرحلة الثالثة: خطة التحليل، المحاكاة، SRM، CUPED، المقاييس والرسوم."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent

def diff_ci(a, b, alpha=0.05):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = float(b.mean() - a.mean())
    se = float(np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b)))
    z = stats.norm.ppf(1-alpha/2)
    return {"difference": diff, "ci_low": float(diff-z*se), "ci_high": float(diff+z*se),
            "se": se, "p": float(2*stats.norm.sf(abs(diff/se)))}

def simulate_experiment(cfg, duration_days):
    # مطابق لمنطق المحاكاة في الملف المرجعي
    r = np.random.default_rng(cfg["seed"])
    n = duration_days * cfg["daily_new_users"]
    is_b = (r.random(n) < cfg["treatment_share"]).astype(int)
    day = r.integers(1, duration_days+1, n)
    experience = r.normal(0, 1, n)
    base = cfg["baseline_conversion"]
    logit = np.log(base/(1-base)) + cfg["activity_strength"]*experience
    p0 = 1/(1+np.exp(-logit))
    p0 = np.clip(p0*base/p0.mean(), 0, 1)
    pre_decisions_90d = r.poisson(5*p0/base)
    lift = cfg["true_lift_abs"]*(1+cfg["novelty_boost"]*np.exp(-(day-1)/3))
    correct_decision = r.binomial(1, np.clip(p0+lift*is_b, 0, 1))
    decision_time = np.maximum(10, r.normal(90-15*is_b, 20, n))
    satisfaction = np.clip(np.rint(r.normal(3.5+0.3*is_b, 0.8, n)), 1, 5).astype(int)
    high_impact_error = ((correct_decision == 0)&(r.random(n)<0.20)).astype(int)
    df = pd.DataFrame({
        "variant": np.where(is_b==1,"B","A"), "day": day,
        "pre_decisions_90d": pre_decisions_90d,
        "correct_decision": correct_decision, "decision_time": decision_time,
        "satisfaction": satisfaction, "high_impact_error": high_impact_error
    })
    if cfg["srm_bug_share"] > 0:
        lost = ((df["variant"]=="B") & (experience<0)
                & (r.random(n)<cfg["srm_bug_share"]*2))
        df = df[~lost]
    return df.reset_index(drop=True)

def main():
    cfg = json.loads((ROOT/"outputs/results/experiment_config.json").read_text(encoding="utf-8"))["config"]
    duration = json.loads((ROOT/"outputs/results/sample_size.json").read_text(encoding="utf-8"))["duration_days"]
    data_dir, result_dir, chart_dir = ROOT/"data", ROOT/"outputs/results", ROOT/"outputs/charts"
    for d in (data_dir, result_dir, chart_dir): d.mkdir(parents=True, exist_ok=True)
    plan = {
        "primary": "Two-sided comparison of correct_decision between A and B at alpha 0.05",
        "srm": "Chi-square test, alert threshold 0.001",
        "cuped": "Pre-experiment decisions as covariate",
        "secondary": ["decision_time", "satisfaction"],
        "multiple_testing": "Benjamini-Hochberg FDR for secondary outcomes",
        "guardrail": "high_impact_error",
        "analysis_time": "After the planned 21-day experiment"
    }
    (result_dir/"analysis_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    data = simulate_experiment(cfg, duration)
    data.to_csv(data_dir/"dataset.csv", index=False)
    A, B = data[data.variant=="A"], data[data.variant=="B"]
    counts = data.variant.value_counts()
    total = len(data)
    chi2, srm_p = stats.chisquare([len(A),len(B)], f_exp=[total*(1-cfg["treatment_share"]),total*cfg["treatment_share"]])
    primary = diff_ci(A.correct_decision, B.correct_decision)
    theta = np.cov(data.correct_decision, data.pre_decisions_90d)[0,1]/data.pre_decisions_90d.var(ddof=1)
    adjusted = data.correct_decision - theta*(data.pre_decisions_90d-data.pre_decisions_90d.mean())
    cuped = diff_ci(adjusted[data.variant=="A"], adjusted[data.variant=="B"])
    secondary = {k: diff_ci(A[k],B[k]) for k in ("decision_time","satisfaction")}
    significant, p_adj, _, _ = multipletests([secondary[k]["p"] for k in secondary], alpha=0.05, method="fdr_bh")
    for k, sig, padj in zip(secondary, significant, p_adj):
        secondary[k]["bh_significant"] = bool(sig)
        secondary[k]["bh_adjusted_p"] = float(padj)
    guardrail = diff_ci(A.high_impact_error, B.high_impact_error)
    guardrail["status"] = "significant_harm" if guardrail["ci_low"]>0 else "ok"
    results = {
        "n": total, "counts": {"A":len(A),"B":len(B)},
        "accuracy": {"A":float(A.correct_decision.mean()),"B":float(B.correct_decision.mean())},
        "srm": {"chi2":float(chi2),"p":float(srm_p),"ok":bool(srm_p>=0.001)},
        "primary": primary, "cuped": cuped,
        "cuped_variance_reduction": float(1-(cuped["se"]/primary["se"])**2),
        "secondary": secondary,
        "guardrail": {"A":float(A.high_impact_error.mean()),"B":float(B.high_impact_error.mean()),**guardrail},
        "simulation_only": True
    }
    (result_dir/"analysis_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # الرسوم: مقارنة الدقة، فترة الثقة، وأثر الحداثة
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(["A: dashboard","B: dashboard + AI"],[results["accuracy"]["A"]*100,results["accuracy"]["B"]*100])
    ax.set_ylabel("Correct decisions (%)"); ax.set_title("Decision accuracy by group")
    fig.tight_layout(); fig.savefig(chart_dir/"decision_accuracy.png",dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,3))
    for i,(name,r) in enumerate([("Unadjusted",primary),("CUPED",cuped)]):
        ax.errorbar(r["difference"]*100,i,xerr=[[100*(r["difference"]-r["ci_low"])],[100*(r["ci_high"]-r["difference"])]],fmt="o",capsize=5)
    ax.axvline(0,color="gray",linestyle="--")
    ax.set_yticks([0,1],["Unadjusted","CUPED"]); ax.set_xlabel("B - A (percentage points)")
    ax.set_title("95% confidence intervals"); fig.tight_layout()
    fig.savefig(chart_dir/"confidence_intervals.png",dpi=160); plt.close(fig)
    daily = data.groupby(["day","variant"]).correct_decision.mean().unstack()
    daily_lift = (daily["B"]-daily["A"])*100
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(daily_lift.index,daily_lift.values,marker="o",label="Daily effect")
    ax.axhline(primary["difference"]*100,linestyle="--",label="Overall effect")
    ax.axhline(0,color="gray",linewidth=0.8)
    ax.set_xlabel("Day"); ax.set_ylabel("B - A (percentage points)")
    ax.set_title("Novelty effect check"); ax.legend(); fig.tight_layout()
    fig.savefig(chart_dir/"novelty_effect.png",dpi=160); plt.close(fig)
    print(f"03 | Analysis complete: A {results['accuracy']['A']:.2%}, B {results['accuracy']['B']:.2%}, p {primary['p']:.4f}")
if __name__ == "__main__":
    main()
