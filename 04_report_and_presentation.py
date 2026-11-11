# -*- coding: utf-8 -*-
"""المرحلة الرابعة: التقرير والتوصية ومخطط العرض (Markdown، وليس PDF/PPTX)."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

def decide(ci_low, ci_high, mde, guardrails_ok=True):
    if ci_high < 0:
        return "لا نطلق، لأن النسخة الجديدة أضرت بالمقياس الرئيسي."
    if ci_low > 0 and not guardrails_ok:
        return "لا نطلق الآن، لأن المقياس الرئيسي تحسن لكن أحد مقاييس الحماية ساء."
    if ci_low > 0:
        return "نطلق، لأن الأثر إيجابي بثقة ومقاييس الحماية سليمة."
    if ci_high < mde:
        return "لا نطلق، لأننا استبعدنا وجود أثر بالحجم الذي يهمنا."
    return "لا نقرر بعد. نمدد التجربة، أو نعيد تصميمها بقوة أعلى."

def main():
    results_dir = ROOT/"outputs/results"
    cfg = json.loads((results_dir/"experiment_config.json").read_text(encoding="utf-8"))["config"]
    r = json.loads((results_dir/"analysis_results.json").read_text(encoding="utf-8"))
    sample = json.loads((results_dir/"sample_size.json").read_text(encoding="utf-8"))
    primary, cuped = r["primary"], r["cuped"]
    decision = decide(primary["ci_low"], primary["ci_high"], cfg["mde_abs"],
                      r["guardrail"]["status"]=="ok")
    annual_users = 50_000
    # مجرد إسقاط افتراضي، وليس أثرًا مثبتًا في الواقع.
    annual_impact = primary["difference"]*annual_users
    final = {"decision":decision,"annual_impact_estimate_simulated":annual_impact,
             "note":"Synthetic simulation only; not real user evidence"}
    (results_dir/"final_decision.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    reports = ROOT/"reports"; reports.mkdir(exist_ok=True)
    report = f"""# تقرير تجربة A/B: {cfg['project_ar']}

**البيانات محاكاة وليست نتائج مستخدمين حقيقيين.**

**القرار:** {decision}

| المؤشر | النتيجة |
|---|---|
| حجم العينة | {r['n']:,} |
| مدة التجربة | {sample['duration_days']} يومًا |
| دقة A | {r['accuracy']['A']:.2%} |
| دقة B | {r['accuracy']['B']:.2%} |
| الفرق | {primary['difference']*100:.2f} نقطة مئوية |
| فترة الثقة 95% | {primary['ci_low']*100:.2f} إلى {primary['ci_high']*100:.2f} نقطة |
| قيمة p | {primary['p']:.4f} |
| الفرق بعد CUPED | {cuped['difference']*100:.2f} نقطة |
| مقياس الحماية | {r['guardrail']['status']} |

**التوصية:** {decision}

**القيود:** هذه نتائج محاكاة، ولا تثبت أثرًا حقيقيًا في بيئة تشغيلية.
"""
    (reports/"One_Page_Report.md").write_text(report,encoding="utf-8")
    slides = f"""# مخطط عرض التجربة (اختياري، وليس ملف PowerPoint)

1. عنوان المشروع وسؤال التجربة.
2. الفرضية وتصميم A/B ووحدة العشوأة.
3. حجم العينة والمدة: {sample['n_total_required']} مستخدمًا على الأقل، {sample['duration_days']} يومًا.
4. خطة التحليل وفحص SRM.
5. النتائج: A={r['accuracy']['A']:.2%}، B={r['accuracy']['B']:.2%}.
6. CUPED والمقاييس الثانوية والحماية.
7. القرار النهائي: {decision}
8. القيود: بيانات محاكاة فقط.

الرسوم في outputs/charts/.
"""
    (reports/"Presentation_Outline.md").write_text(slides,encoding="utf-8")
    print("04 | Final decision, one-page report and optional presentation outline saved")
if __name__ == "__main__":
    main()
