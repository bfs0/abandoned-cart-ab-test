"""
قياس أثر تذكير السلة المتروكة على معدل إتمام الشراء
دورة: التجارب واختبارات A/B والاستدلال السببي - أكاديمية سدايا

التشغيل:
    pip install -r requirements.txt
    python main.py

ما يفعله السكربت بالترتيب:
    1. حساب حجم العينة ومدة التجربة
    2. إنشاء بيانات المحاكاة (أو قراءتها من data/dataset.csv إن وُجدت)
    3. فحوص الجودة (اكتمال البيانات، التكرار، SRM)
    4. التحليل: المقياس الرئيسي، CUPED، المقياس الثانوي، مقاييس الحماية، أثر الحداثة
    5. حفظ الرسوم في outputs/charts والنتائج في outputs/results
    6. القرار والتوصية
"""

import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # حفظ الرسوم دون الحاجة إلى نافذة عرض
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# المسارات (نسبية بالنسبة لمكان الملف، فلا تحتاج أي تعديل على أي جهاز)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

# ---------------------------------------------------------------------------
# الإعدادات والافتراضات
# ---------------------------------------------------------------------------
CONFIG = {
    "project": "قياس أثر تذكير السلة المتروكة على معدل إتمام الشراء",
    "seed": 2026,
    "salt": "abandoned-cart-reminder-v1",  # ملح التوزيع الثابت (Deterministic Hashing)
    "treatment_share": 0.5,
    # تصميم التجربة
    "alpha": 0.05,
    "power": 0.80,
    "baseline_conversion": 0.10,  # نسبة الإتمام المتوقعة بدون تذكير (افتراض)
    "mde_abs": 0.01,              # أصغر أثر نهتم به: نقطة مئوية واحدة (افتراض)
    "daily_new_users": 1500,      # مستخدمون مؤهلون يدخلون التجربة يومياً (افتراض)
    # افتراضات المحاكاة (القيم "الحقيقية" المخفية داخل البيانات المحاكاة)
    "true_lift_abs": 0.015,       # أثر التذكير الحقيقي: +1.5 نقطة مئوية
    "novelty_boost": 0.30,        # زيادة مؤقتة في الأثر في الأيام الأولى
    "activity_strength": 0.8,     # قوة ارتباط نشاط المستخدم باحتمال الشراء
    "order_value_mean": 180,      # متوسط قيمة الطلب بالريال
    "order_value_sd": 60,
    "unsub_base": 0.010,          # إلغاء الاشتراك من رسائل المتجر (أساس)
    "unsub_effect": 0.002,        # أثر التذكير على إلغاء الاشتراك
    "support_base": 0.030,        # التواصل مع خدمة العملاء (أساس)
    "support_effect": 0.0005,     # أثر التذكير على التواصل مع الدعم
}

TEAL, ORANGE, NAVY, GRAY = "#0E7490", "#E58A1F", "#0F2A3F", "#A9B4BC"
plt.rcParams.update({
    "figure.figsize": (8, 4.5), "figure.dpi": 110,
    "axes.edgecolor": "#56656F", "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": "#3D4F5C", "ytick.color": "#3D4F5C",
    "axes.grid": True, "grid.color": "#D8E1E7", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "font.size": 11,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# 1. حجم العينة ومدة التجربة
# ---------------------------------------------------------------------------
def n_per_group_proportions(p_control, p_treatment, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    return int(np.ceil((z_a + z_b) ** 2 * var / (p_treatment - p_control) ** 2))


def mde_proportions(p_control, n_per_group, alpha=0.05, power=0.80):
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (z_a + z_b) * np.sqrt(2 * p_control * (1 - p_control) / n_per_group)


def duration_days(n_per_group, daily_new_users, groups=2, round_to_weeks=True):
    days = int(np.ceil(n_per_group * groups / daily_new_users))
    return int(np.ceil(days / 7) * 7) if round_to_weeks else days


# ---------------------------------------------------------------------------
# 2. البيانات: التوزيع الثابت + المحاكاة
# ---------------------------------------------------------------------------
def assign_variant(user_id, salt, treatment_share):
    """توزيع ثابت: نفس المستخدم يقع دائماً في نفس المجموعة."""
    digest = hashlib.md5(f"{salt}:{user_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "B" if bucket < treatment_share else "A"


def simulate_experiment(cfg, days):
    """يحاكي المستخدمين الذين دخلوا التجربة خلال مدتها (صف لكل مستخدم)."""
    r = np.random.default_rng(cfg["seed"])
    n = days * cfg["daily_new_users"]
    user_id = np.arange(1, n + 1)
    variant = np.array([assign_variant(u, cfg["salt"], cfg["treatment_share"]) for u in user_id])
    is_b = (variant == "B").astype(int)
    day = np.repeat(np.arange(1, days + 1), cfg["daily_new_users"])

    # نشاط المستخدم (لا نقيسه مباشرة) يؤثر في احتمال الشراء وفي طلباته السابقة
    activity = r.normal(0, 1, n)
    base = cfg["baseline_conversion"]
    logit = np.log(base / (1 - base)) + cfg["activity_strength"] * activity
    p0 = 1 / (1 + np.exp(-logit))
    p0 = np.clip(p0 * base / p0.mean(), 0, 1)  # ضبط المتوسط ليساوي خط الأساس
    pre_orders_90d = r.poisson(5 * p0 / base)   # طلبات آخر 90 يوماً (قبل التجربة)

    # الأثر الحقيقي مع زيادة مؤقتة في الأيام الأولى (أثر الحداثة)
    lift = cfg["true_lift_abs"] * (1 + cfg["novelty_boost"] * np.exp(-(day - 1) / 3))
    ordered = r.binomial(1, np.clip(p0 + lift * is_b, 0, 1))

    value = np.maximum(10, r.normal(cfg["order_value_mean"], cfg["order_value_sd"], n))
    order_value = np.where(ordered == 1, np.round(value, 2), np.nan)
    revenue = np.where(ordered == 1, np.round(value, 2), 0.0)

    unsub = r.binomial(1, np.clip(cfg["unsub_base"] + cfg["unsub_effect"] * is_b, 0, 1))
    support = r.binomial(1, np.clip(cfg["support_base"] + cfg["support_effect"] * is_b, 0, 1))

    return pd.DataFrame({
        "user_id": user_id, "variant": variant, "day": day,
        "pre_orders_90d": pre_orders_90d, "ordered": ordered,
        "order_value": order_value, "revenue": revenue,
        "unsubscribed": unsub, "contacted_support": support,
    })


def load_or_create_data(cfg, days):
    if DATA_PATH.exists():
        print("تم تحميل البيانات من: data/dataset.csv")
        return pd.read_csv(DATA_PATH)
    df = simulate_experiment(cfg, days)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print("تم إنشاء بيانات المحاكاة وحفظها في: data/dataset.csv")
    return df


# ---------------------------------------------------------------------------
# 3. فحوص الجودة
# ---------------------------------------------------------------------------
def srm_check(n_a, n_b, expected_share_b=0.5, threshold=0.001):
    total = n_a + n_b
    chi2, p = stats.chisquare([n_a, n_b], f_exp=[total * (1 - expected_share_b), total * expected_share_b])
    return chi2, p, p < threshold


def quality_checks(df, cfg):
    counts = df["variant"].value_counts()
    chi2, p, srm_bug = srm_check(counts.get("A", 0), counts.get("B", 0), cfg["treatment_share"])
    inconsistent = int(((df["ordered"] == 0) & df["order_value"].notna()).sum()
                       + ((df["ordered"] == 1) & df["order_value"].isna()).sum())
    return {
        "rows": len(df),
        "n_a": int(counts.get("A", 0)), "n_b": int(counts.get("B", 0)),
        "duplicate_users": int(df["user_id"].duplicated().sum()),
        "missing_values": int(df.drop(columns=["order_value"]).isna().sum().sum()),
        "order_value_inconsistent": inconsistent,
        "srm_chi2": float(chi2), "srm_p": float(p), "srm_problem": bool(srm_bug),
    }


# ---------------------------------------------------------------------------
# 4. التحليل
# ---------------------------------------------------------------------------
def diff_ci(a, b, alpha=0.05):
    """الفرق بين B وA مع فترة الثقة وقيمة p (بالتقريب الطبيعي)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"diff": diff, "ci_low": diff - z * se, "ci_high": diff + z * se, "se": se,
            "p": 2 * stats.norm.sf(abs(diff / se))}


def cuped_adjust(y, x):
    """CUPED: تقليل التباين باستخدام الطلبات السابقة x."""
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean())


def decide(ci_low, ci_high, mde, guardrails_ok=True):
    if ci_high < 0:
        return "لا نطلق، لأن التذكير أضرّ بالمقياس الرئيسي."
    if ci_low > 0 and not guardrails_ok:
        return ("لا نطلق الآن، لأن المقياس الرئيسي تحسن لكن أحد مقاييس الحماية ساء. "
                "نعدّل التغيير ثم نعيد التجربة.")
    if ci_low > 0:
        return "نطلق، لأن الأثر إيجابي بثقة ومقاييس الحماية سليمة."
    if ci_high < mde:
        return "لا نطلق، لأننا استبعدنا وجود أثر بالحجم الذي يهمنا."
    return "لا نقرر بعد. نمدد التجربة، أو نعيد تصميمها بقوة أعلى."


def run_analysis(df, cfg):
    A, B = df[df["variant"] == "A"], df[df["variant"] == "B"]
    res = {"n": {"A": len(A), "B": len(B)}}

    # المقياس الرئيسي: نسبة إتمام الشراء
    res["primary"] = diff_ci(A["ordered"], B["ordered"])
    res["primary"].update(mean_a=A["ordered"].mean(), mean_b=B["ordered"].mean())

    # CUPED
    y = df["ordered"].to_numpy(dtype=float)
    x = df["pre_orders_90d"].to_numpy(dtype=float)
    adj = cuped_adjust(y, x)
    is_a = (df["variant"] == "A").to_numpy()
    res["cuped"] = diff_ci(adj[is_a], adj[~is_a])
    res["cuped"]["variance_reduction"] = 1 - (res["cuped"]["se"] / res["primary"]["se"]) ** 2

    # مقياس ثانوي: الإيراد لكل مستخدم
    res["revenue"] = diff_ci(A["revenue"], B["revenue"])
    res["revenue"].update(mean_a=A["revenue"].mean(), mean_b=B["revenue"].mean())

    # مقاييس الحماية (الاتجاه السيئ: الارتفاع)
    guardrails = {}
    for col, label in [("unsubscribed", "إلغاء الاشتراك من رسائل المتجر"),
                       ("contacted_support", "التواصل مع خدمة العملاء")]:
        g = diff_ci(A[col], B[col])
        g.update(label=label, mean_a=A[col].mean(), mean_b=B[col].mean(),
                 status="تدهور دال" if g["ci_low"] > 0 else "سليم")
        guardrails[col] = g
    res["guardrails"] = guardrails

    # أثر الحداثة: الأثر اليومي، والأسبوع الأول مقابل الأخير
    daily = df.groupby(["day", "variant"])["ordered"].mean().unstack()
    res["daily_lift"] = daily["B"] - daily["A"]
    last_day = int(df["day"].max())
    first_week = df[df["day"] <= 7]
    last_week = df[df["day"] > last_day - 7]
    res["first_week"] = diff_ci(first_week[first_week["variant"] == "A"]["ordered"],
                                first_week[first_week["variant"] == "B"]["ordered"])
    res["last_week"] = diff_ci(last_week[last_week["variant"] == "A"]["ordered"],
                               last_week[last_week["variant"] == "B"]["ordered"])

    # القرار
    res["guardrails_ok"] = all(g["status"] == "سليم" for g in guardrails.values())
    res["decision"] = decide(res["primary"]["ci_low"], res["primary"]["ci_high"],
                             cfg["mde_abs"], res["guardrails_ok"])
    annual_users = cfg["daily_new_users"] * 365
    res["annual_users"] = annual_users
    res["annual_revenue"] = {k: res["revenue"][k] * annual_users for k in ("diff", "ci_low", "ci_high")}
    return res


# ---------------------------------------------------------------------------
# 5. الرسوم (العناوين بالإنجليزية حتى تظهر صحيحة على أي جهاز دون مكتبات إضافية)
# ---------------------------------------------------------------------------
def make_charts(res, cfg):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    p = res["primary"]
    z = stats.norm.ppf(0.975)

    # 1) نسبة إتمام الشراء لكل مجموعة
    fig, ax = plt.subplots()
    n_a, n_b = res["n"]["A"], res["n"]["B"]
    vals = [p["mean_a"] * 100, p["mean_b"] * 100]
    errs = [z * np.sqrt(p["mean_a"] * (1 - p["mean_a"]) / n_a) * 100,
            z * np.sqrt(p["mean_b"] * (1 - p["mean_b"]) / n_b) * 100]
    ax.bar(["A: no reminder (control)", "B: reminder (treatment)"], vals, yerr=errs,
           color=[TEAL, ORANGE], capsize=6, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + errs[i] + 0.2, f"{v:.2f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Purchase completion rate (%)")
    ax.set_title("Purchase completion rate by group (95% CI)")
    ax.set_ylim(0, max(vals) + max(errs) + 2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "conversion_by_group.png")
    plt.close(fig)

    # 2) الأثر المقدَّر وفترة الثقة مقابل الصفر وأصغر أثر مهم
    fig, ax = plt.subplots(figsize=(8, 3.5))
    rows = [("Without CUPED", res["primary"]), ("With CUPED", res["cuped"])]
    for i, (_, r_) in enumerate(rows):
        ax.errorbar(r_["diff"] * 100, i,
                    xerr=[[(r_["diff"] - r_["ci_low"]) * 100], [(r_["ci_high"] - r_["diff"]) * 100]],
                    fmt="o", color=TEAL, capsize=6, markersize=8)
    ax.axvline(0, color=GRAY, linewidth=1.2, label="No effect")
    ax.axvline(cfg["mde_abs"] * 100, color=ORANGE, linestyle="--", label="Minimum effect of interest")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r_[0] for r_ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Effect on completion rate (percentage points)")
    ax.set_title("Estimated effect with 95% confidence interval")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "effect_ci.png")
    plt.close(fig)

    # 3) أثر الحداثة: الأثر في كل يوم
    fig, ax = plt.subplots()
    daily = res["daily_lift"] * 100
    ax.plot(daily.index, daily.values, marker="o", color=TEAL, label="Daily effect")
    ax.axhline(p["diff"] * 100, color=ORANGE, linestyle="--", label="Overall effect")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("Day the user entered the experiment")
    ax.set_ylabel("Effect (percentage points)")
    ax.set_title("Is the effect stable over time? (novelty check)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "daily_lift.png")
    plt.close(fig)

    # 4) مقاييس الحماية
    fig, ax = plt.subplots(figsize=(8, 4))
    gs = list(res["guardrails"].values())
    names = ["Unsubscribe rate", "Contacted support"]
    xpos = np.arange(len(gs))
    ax.bar(xpos - 0.2, [g["mean_a"] * 100 for g in gs], width=0.4, color=TEAL, label="A (control)")
    ax.bar(xpos + 0.2, [g["mean_b"] * 100 for g in gs], width=0.4, color=ORANGE, label="B (treatment)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Guardrail metrics by group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "guardrails.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. التقرير
# ---------------------------------------------------------------------------
def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def pp(x, d=2):
    return f"{x * 100:+.{d}f}"


def build_report(cfg, sizing, quality, res):
    p, c, rev = res["primary"], res["cuped"], res["revenue"]
    fw, lw = res["first_week"], res["last_week"]
    persists = "ما زال الأثر موجوداً في الأسبوع الأخير" if lw["ci_low"] > 0 else \
        "الأثر في الأسبوع الأخير غير واضح، وقد يكون جزء منه أثر حداثة"
    lines = [
        f"# تقرير التجربة: {cfg['project']}", "",
        "> البيانات محاكاة. الأثر الحقيقي مُدخل في الإعدادات (CONFIG)، والنتائج تختبر خط التحليل لا متجراً حقيقياً.", "",
        "## حجم العينة والمدة", "",
        f"- مستوى الدلالة α = {cfg['alpha']}، والقوة المستهدفة = {cfg['power']:.0%}",
        f"- نسبة الإتمام الأساسية = {pct(cfg['baseline_conversion'], 1)}، وأصغر أثر مهم = {cfg['mde_abs'] * 100:.1f} نقطة مئوية",
        f"- حجم العينة المطلوب لكل مجموعة = {sizing['n_per_group']:,}",
        f"- المستخدمون المتوقعون يومياً = {cfg['daily_new_users']:,}، والمدة المقدّرة = {sizing['days']} يوماً",
        f"- أصغر أثر قابل للاكتشاف بالعينة الفعلية = {sizing['mde_actual'] * 100:.2f} نقطة مئوية", "",
        "## فحوص الجودة", "",
        f"- عدد المستخدمين = {quality['rows']:,} (A = {quality['n_a']:,}، B = {quality['n_b']:,})",
        f"- سجلات مكررة (user_id) = {quality['duplicate_users']}",
        f"- قيم مفقودة (خارج قيمة الطلب) = {quality['missing_values']}",
        f"- تناقض بين الطلب وقيمته = {quality['order_value_inconsistent']}",
        f"- SRM: χ² = {quality['srm_chi2']:.2f}، p = {quality['srm_p']:.3g}، النتيجة: "
        f"{'خلل في التوزيع' if quality['srm_problem'] else 'التوزيع سليم'}", "",
        "## النتائج", "",
        "| المقياس | A (مقارنة) | B (معالجة) | الأثر | فترة الثقة 95% | قيمة p |",
        "|---|---|---|---|---|---|",
        f"| نسبة إتمام الشراء | {pct(p['mean_a'])} | {pct(p['mean_b'])} | {pp(p['diff'])} نقطة | "
        f"{pp(p['ci_low'])} إلى {pp(p['ci_high'])} | {p['p']:.3g} |",
        f"| نسبة الإتمام مع CUPED | - | - | {pp(c['diff'])} نقطة | {pp(c['ci_low'])} إلى {pp(c['ci_high'])} | {c['p']:.3g} |",
        f"| الإيراد لكل مستخدم (ريال) | {rev['mean_a']:.2f} | {rev['mean_b']:.2f} | {rev['diff']:+.2f} | "
        f"{rev['ci_low']:+.2f} إلى {rev['ci_high']:+.2f} | {rev['p']:.3g} |",
    ]
    for g in res["guardrails"].values():
        lines.append(f"| {g['label']} (حماية: {g['status']}) | {pct(g['mean_a'])} | {pct(g['mean_b'])} | "
                     f"{pp(g['diff'])} نقطة | {pp(g['ci_low'])} إلى {pp(g['ci_high'])} | {g['p']:.3g} |")
    lines += [
        "",
        f"- الأثر النسبي على نسبة الإتمام = {p['diff'] / p['mean_a']:+.1%}",
        f"- CUPED قلّل التباين بنسبة {c['variance_reduction']:.1%}",
        f"- أثر الحداثة: الأسبوع الأول {pp(fw['diff'])} نقطة، والأسبوع الأخير {pp(lw['diff'])} نقطة. {persists}.",
        f"- الأثر التقديري على الإيراد سنوياً (بافتراض {res['annual_users']:,} مستخدم مؤهل): "
        f"{res['annual_revenue']['diff'] / 1e6:.2f} مليون ريال "
        f"(من {res['annual_revenue']['ci_low'] / 1e6:.2f} إلى {res['annual_revenue']['ci_high'] / 1e6:.2f})", "",
        "## القرار", "", f"**{res['decision']}**", "",
    ]
    return "\n".join(lines)


def save_results_csv(res):
    rows = []
    for name, r_ in [("conversion_rate", res["primary"]), ("conversion_rate_cuped", res["cuped"]),
                     ("revenue_per_user", res["revenue"])]:
        rows.append({"metric": name, "mean_A": r_.get("mean_a"), "mean_B": r_.get("mean_b"),
                     "effect": r_["diff"], "ci_low_95": r_["ci_low"], "ci_high_95": r_["ci_high"],
                     "p_value": r_["p"], "guardrail_status": ""})
    for col, g in res["guardrails"].items():
        rows.append({"metric": col, "mean_A": g["mean_a"], "mean_B": g["mean_b"],
                     "effect": g["diff"], "ci_low_95": g["ci_low"], "ci_high_95": g["ci_high"],
                     "p_value": g["p"], "guardrail_status": "degraded" if g["status"] != "سليم" else "ok"})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "results.csv", index=False)


# ---------------------------------------------------------------------------
# التشغيل
# ---------------------------------------------------------------------------
def main():
    cfg = CONFIG
    print("=" * 60)
    print(cfg["project"])
    print("=" * 60)

    # 1) حجم العينة
    n_group = n_per_group_proportions(cfg["baseline_conversion"],
                                      cfg["baseline_conversion"] + cfg["mde_abs"],
                                      cfg["alpha"], cfg["power"])
    days = duration_days(n_group, cfg["daily_new_users"])
    actual_n = days * cfg["daily_new_users"] // 2
    sizing = {"n_per_group": n_group, "days": days,
              "mde_actual": mde_proportions(cfg["baseline_conversion"], actual_n, cfg["alpha"], cfg["power"])}
    print(f"\n[1] حجم العينة: {n_group:,} لكل مجموعة، والمدة المقدّرة {days} يوماً")

    # 2) البيانات
    print("\n[2] البيانات")
    df = load_or_create_data(cfg, days)

    # 3) فحوص الجودة
    quality = quality_checks(df, cfg)
    print("\n[3] فحوص الجودة")
    print(f"    المستخدمون: {quality['rows']:,} (A={quality['n_a']:,}, B={quality['n_b']:,})")
    print(f"    سجلات مكررة: {quality['duplicate_users']} | قيم مفقودة: {quality['missing_values']}")
    print(f"    SRM: chi2={quality['srm_chi2']:.2f}, p={quality['srm_p']:.3g} -> "
          f"{'خلل في التوزيع، لا تثق بالنتائج' if quality['srm_problem'] else 'التوزيع سليم'}")
    if quality["srm_problem"] or quality["duplicate_users"] > 0:
        print("    تم إيقاف التحليل بسبب مشكلة في جودة البيانات.")
        sys.exit(1)

    # 4) التحليل
    res = run_analysis(df, cfg)

    # 5) الرسوم والنتائج
    make_charts(res, cfg)
    save_results_csv(res)
    report = build_report(cfg, sizing, quality, res)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")

    # 6) طباعة التقرير
    print("\n" + report)
    print("تم حفظ الرسوم في: outputs/charts")
    print("تم حفظ النتائج في: outputs/results")


if __name__ == "__main__":
    main()
