"""
A/B-тест нового блока рекомендаций в телеком-приложении (симуляция).

Пайплайн: генерация данных -> загрузка в SQLite -> метрики через SQL ->
SRM-проверка -> расчёт мощности -> z-тест / Welch t-test -> CUPED ->
bootstrap -> A/A-тест -> поправка Холма на множественные сравнения.
"""
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)
N = 200_000
ALPHA = 0.05

# ---------------------------------------------------------------- 1. Данные
platform = rng.choice(["android", "ios"], N, p=[0.7, 0.3])
tenure = rng.exponential(30, N).round(1)
pre_sessions = rng.poisson(rng.lognormal(1.6, 0.6, N))
group = rng.integers(0, 2, N)  # 0 = control, 1 = treatment

# Метрика 1: клик по рекомендациям (бинарная). Истинный эффект ~ +0.15 лог-шанса
logit = -1.4 + 0.18 * np.log1p(pre_sessions) + 0.1 * (platform == "ios") + 0.15 * group
p_click = 1 / (1 + np.exp(-logit))
clicked = rng.binomial(1, p_click)

# Метрика 2: сессии за период теста (сильно коррелирует с pre_sessions). Эффект ~ +1.5%
lam = pre_sessions * 0.95 + 1.0
lam = lam * (1 + 0.015 * group)
sessions = rng.poisson(lam)

# Guardrail: обращение в поддержку (эффекта нет)
support = rng.binomial(1, 0.04, N)

df = pd.DataFrame(dict(user_id=np.arange(N), grp=group, platform=platform,
                       tenure_months=tenure, pre_sessions=pre_sessions,
                       clicked=clicked, sessions=sessions, support_call=support))

# ------------------------------------------------------- 2. Метрики через SQL
con = sqlite3.connect(":memory:")
df.to_sql("users", con, index=False)
sql = """
SELECT grp,
       COUNT(*)                          AS n,
       AVG(clicked)                      AS ctr_user,
       AVG(sessions)                     AS sessions_mean,
       AVG(support_call)                 AS support_rate
FROM users
GROUP BY grp
ORDER BY grp;
"""
agg = pd.read_sql(sql, con)
print(agg, "\n")

c, t = df[df.grp == 0], df[df.grp == 1]
res = {"n_control": len(c), "n_treatment": len(t)}

# --------------------------------------------------------------- 3. SRM-check
chi = stats.chisquare([len(c), len(t)])
res["srm_p"] = float(chi.pvalue)

# ------------------------------------------------------ 4. Мощность / размер
def n_per_group(p, rel_mde, alpha=0.05, power=0.8):
    p2 = p * (1 + rel_mde)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = p * (1 - p) + p2 * (1 - p2)
    return int(np.ceil((za + zb) ** 2 * var / (p2 - p) ** 2))

base = c.clicked.mean()
res["baseline_ctr"] = float(base)
res["n_needed_mde_3pct"] = n_per_group(base, 0.03)
res["n_needed_mde_5pct"] = n_per_group(base, 0.05)

# ------------------------------------------------- 5. Тест по долям (CTR)
p1, p2 = t.clicked.mean(), c.clicked.mean()
pp = df.clicked.mean()
se_pool = np.sqrt(pp * (1 - pp) * (1 / len(t) + 1 / len(c)))
z = (p1 - p2) / se_pool
p_ctr = 2 * (1 - stats.norm.cdf(abs(z)))
se = np.sqrt(p1 * (1 - p1) / len(t) + p2 * (1 - p2) / len(c))
res["ctr_diff_abs"] = float(p1 - p2)
res["ctr_rel_lift_pct"] = float((p1 / p2 - 1) * 100)
res["ctr_ci95"] = [float(p1 - p2 - 1.96 * se), float(p1 - p2 + 1.96 * se)]
res["ctr_p"] = float(p_ctr)

# ------------------------------------ 6. Сессии: Welch и CUPED
tt = stats.ttest_ind(t.sessions, c.sessions, equal_var=False)
res["sessions_welch_p"] = float(tt.pvalue)
res["sessions_lift_pct"] = float((t.sessions.mean() / c.sessions.mean() - 1) * 100)

theta = np.cov(df.sessions, df.pre_sessions)[0, 1] / df.pre_sessions.var()
y_cuped = df.sessions - theta * (df.pre_sessions - df.pre_sessions.mean())
tc = stats.ttest_ind(y_cuped[df.grp == 1], y_cuped[df.grp == 0], equal_var=False)
res["sessions_cuped_p"] = float(tc.pvalue)
res["cuped_var_reduction_pct"] = float((1 - y_cuped.var() / df.sessions.var()) * 100)
res["cuped_corr"] = float(np.corrcoef(df.sessions, df.pre_sessions)[0, 1])

# Bootstrap (CUPED-метрика)
yt, yc = y_cuped[df.grp == 1].values, y_cuped[df.grp == 0].values
boots = [rng.choice(yt, len(yt)).mean() - rng.choice(yc, len(yc)).mean() for _ in range(1000)]
res["sessions_cuped_boot_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

# Guardrail
sp = stats.chi2_contingency(pd.crosstab(df.grp, df.support_call))[1]
res["support_p"] = float(sp)

# ------------------------------------------ 7. Поправка Холма (3 метрики)
pvals = {"ctr": p_ctr, "sessions_cuped": tc.pvalue, "support": sp}
order = sorted(pvals, key=pvals.get)
m, holm, running = len(order), {}, 0.0
for i, k in enumerate(order):
    running = max(running, min(1, (m - i) * pvals[k]))
    holm[k] = float(running)
res["holm_adjusted_p"] = holm

# -------------------------------------------------- 8. A/A-тест (валидация)
fp = 0
runs = 500
for _ in range(runs):
    fake = rng.integers(0, 2, N)
    pv = stats.ttest_ind(df.sessions[fake == 1], df.sessions[fake == 0], equal_var=False).pvalue
    fp += pv < ALPHA
res["aa_false_positive_rate"] = fp / runs

print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open("results.json", "w"), indent=2, ensure_ascii=False)
