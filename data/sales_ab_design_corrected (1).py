# -*- coding: utf-8 -*-
"""Диагностика дизайна AB-теста сейлзов с перебором верхних отсечений.

Зависимости: pandas, numpy, scipy, openpyxl — как в исходном скрипте.
Запуск: python sales_ab_design_corrected.py
В ноутбуке можно вызвать tables = run_analysis(dgr, ab), затем display(...).

Сохранены продуктовые списки, деление дохода между соисполнителями,
фильтр Y > 0 по умолчанию и перебор исключений сверху по Y.
В AB-популяции одинаковая доля удаляется отдельно в каждой группе;
при одинаковых исходных n группы остаются равными на каждом шаге.
Основной скорректированный эффект: ANCOVA с робастными ошибками HC3.
Классический CUPED + Welch также сохранён в диагностических колонках.
Метки статистического сигнала относятся к строке диагностического перебора.
"""

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nct, t as t_dist


# ============================================================
# 0. Настройки: пути, периоды, параметры диагностики
# ============================================================
DATA_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\data\441774c5-7986-43a6-8269-5d5ce8627e9b.xlsx")
AB_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\Телефоны УГР 29.06.2026.xlsx")
AB_SHEET = "Штатная расстановка"
AB_NAME = "Ф.И.О. сотрудника"
AB_GROUP = "AB-тест"
OUT_DIR = Path("ab sales")

# Оба окна заданы явно и не сдвигаются при добавлении новых данных.
PRE_MONTHS = ("2026-05", "2026-06")
TEST_MONTHS = ("2026-09", "2026-10")
POSITIVE_Y_ONLY = True       # Сохранён исходный фильтр диагностической выборки.
TRIM_WITHIN_GROUP = True     # Одинаковая доля удалений внутри T/C.
TRIM_COLUMN = "Y"           # "Y" — доход теста; "X" — доход пре-периода.
MIN_N_GROUP = 10             # Ниже этого n p-value/MDE не рассчитываются.
ALPHA = 0.05
POWER = 0.80
EFFECT_THRESHOLD = 200_000_000
EFFECT_TARGET = "rollout"    # "rollout" — все оставшиеся; "treatment" — только T.
DESIGN_POPULATION = "ab"     # "ab" — участники AB; "all" — все сейлзы из сделок.
REQUIRE_ALL_MONTHS = True    # Остановка, если целый ожидаемый месяц отсутствует.

# Один базовый предиктор: доход до теста. Взаимодействие позволяет наклону
# зависимости от X различаться между T/C; X центрируется по всему kept.
ANCOVA_INTERACTION = True

FLOW_COL = "КМ Flow (исторический)"
FICC_COL = "КМ Ficc (исторический)"
flow_products = {
    "Brokerage", "Brokerage PAO", "Com_Der_Agro", "Com_Der_Base",
    "Com_OMS_PM", "Com_Phy_PM", "Commodity", "Commodity Energy",
    "Commodity Metals", "DCD", "Deposit COGR", "DFA", "FX Forward",
    "FX Futures", "FX NDF", "FX Option", "FX Spot", "FX Spot IF",
    "FX Spot PB", "FX Swap", "Loans & Deposits", "Repo", "RepoPB",
    "Structured Deposit", "Treasury",
}
ficc_products = {
    "BHS", "Caps & Floors", "Cash Bonds", "Com_Der_CVA", "Com_Fin",
    "Com_Fin_Oil", "Com_Prepay_PM", "Com_PrepFWD_PM", "CPI Option",
    "Currency swaps", "Derivatives", "FRA", "Interest rate swaps",
    "IOS", "PM Loans", "SberStrBond", "Structured Note", "TARS", "TRS",
}
excluded_products = {
    "Cash Equities", "DCM commission", "ECM commission", "OMS PB", "PM PB",
}


# ============================================================
# 1. Очистка имён и дат
# ============================================================
def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def name_key(value):
    name = clean_text(value).casefold().replace("ё", "е")
    return "" if name in {"", "-", "—", "nan", "none", "null"} else name


def parse_month(value):
    """Русские месяцы, даты Excel, datetime, Period, ISO и dd.mm.yyyy."""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Period):
        return value.asfreq("M").to_timestamp()
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).to_period("M").to_timestamp()
    if isinstance(value, (int, float, np.number)):
        if not np.isfinite(value) or not 1 <= value <= 100000:
            return pd.NaT
        return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(value))).to_period("M").to_timestamp()
    s = clean_text(value).lower()
    match = re.fullmatch(r"([а-яё]+)\.?\s+(\d{4})(?:\s*г\.?)?", s)
    if match:
        month_word, year = match.groups()
        prefixes = (("янв", 1), ("фев", 2), ("мар", 3), ("апр", 4),
                    ("май", 5), ("мая", 5), ("июн", 6), ("июл", 7),
                    ("авг", 8), ("сен", 9), ("окт", 10), ("ноя", 11), ("дек", 12))
        month = next((m for prefix, m in prefixes if month_word.startswith(prefix)), None)
        try:
            return pd.Timestamp(year=int(year), month=month, day=1) if month else pd.NaT
        except (ValueError, OverflowError):
            return pd.NaT
    try:
        if re.match(r"^\d{4}-\d{2}(?:-|$)", s):
            parsed = pd.Timestamp(s)
        else:
            parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return parsed.to_period("M").to_timestamp() if pd.notna(parsed) else pd.NaT
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def split_names(value):
    if pd.isna(value):
        return []
    # Удаление повторов ДО деления дохода: «Иванов, Иванов, Петров» → 1/2 + 1/2.
    names = {}
    for item in re.split(r"[,;\n]+", str(value)):
        key = name_key(item)
        if key:
            names.setdefault(key, clean_text(item))
    return list(names.items())


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"В {label} отсутствуют колонки: {missing}")


def prepare_ab(ab):
    require_columns(ab, [AB_NAME, AB_GROUP], "AB-файле")
    out = pd.DataFrame({
        "sale_key": ab[AB_NAME].map(name_key),
        "sale_name": ab[AB_NAME].map(clean_text),
        "group": ab[AB_GROUP].map(clean_text).str.casefold(),
    })
    out = out[out["group"].isin(["treatment", "control"])].copy()
    if out.empty:
        raise ValueError("В AB-файле нет строк treatment/control")
    if out["sale_key"].eq("").any():
        raise ValueError("В AB-файле у участника treatment/control отсутствует ФИО")
    # Проверка обязательно ПЕРЕД drop_duplicates.
    conflict = out.groupby("sale_key")["group"].nunique()
    bad = conflict[conflict > 1].index
    if len(bad):
        raise ValueError("Один сотрудник записан в обе группы: " + "; ".join(bad))
    return out.drop_duplicates("sale_key").reset_index(drop=True)


# ============================================================
# 2. Распределение дохода по продуктам + отчёт о пропущенных строках
# ============================================================
def prepare_deals(dgr):
    require_columns(dgr, ["Продукт", FLOW_COL, FICC_COL, "Доход", "Месяц-Год"], "dgr")
    if flow_products & ficc_products or (flow_products | ficc_products) & excluded_products:
        raise ValueError("Справочники продуктов пересекаются")
    records, audit = [], []
    for excel_row, (_, row) in enumerate(dgr.iterrows(), start=2):
        product = clean_text(row["Продукт"])
        month = parse_month(row["Месяц-Год"])
        income = pd.to_numeric(row["Доход"], errors="coerce")
        reason = ""
        if product in excluded_products:
            reason = "исключённый продукт"
        elif product not in flow_products | ficc_products:
            reason = "неизвестный продукт"
        elif pd.isna(month):
            reason = "не распознана дата"
        elif pd.isna(income) or not np.isfinite(income):
            # Пропуск дохода в строке сделки не равен отсутствию сделок у сейлза.
            raise ValueError(f"Строка Excel {excel_row}: некорректный/пустой Доход={row['Доход']!r}")
        else:
            col = FLOW_COL if product in flow_products else FICC_COL
            names = split_names(row[col])
            if not names:
                reason = "нет ФИО в выбранной колонке"
        if reason:
            audit.append({"строка Excel": excel_row, "причина": reason, "Продукт": product,
                          "Месяц-Год": clean_text(row["Месяц-Год"]), "Доход": income})
            continue
        share = float(income) / len(names)
        for key, name in names:
            records.append({"sale_key": key, "sale_name": name, "month_p": month.to_period("M"),
                            "income": share, "строка Excel": excel_row, "Продукт": product})
    if not records:
        raise ValueError("После распределения не осталось валидных сделок")
    long = pd.DataFrame(records)
    audit_df = pd.DataFrame(audit, columns=["строка Excel", "причина", "Продукт", "Месяц-Год", "Доход"])
    return long, audit_df


def aggregate_sales(long, pre_months, test_months):
    pre_set = {pd.Period(m, freq="M") for m in pre_months}
    test_set = {pd.Period(m, freq="M") for m in test_months}
    if not pre_set or not test_set or max(pre_set) >= min(test_set):
        raise ValueError("Непустой пре-период должен целиком предшествовать тестовому периоду")
    present = set(long["month_p"])
    missing = sorted((pre_set | test_set) - present)
    if REQUIRE_ALL_MONTHS and missing:
        raise ValueError(f"В распределённых сделках отсутствуют месяцы: {[str(m) for m in missing]}")
    relevant = long[long["month_p"].isin(pre_set | test_set)]
    cur = relevant[relevant["month_p"].isin(test_set)].groupby("sale_key")["income"].sum().rename("Y")
    pre = relevant[relevant["month_p"].isin(pre_set)].groupby("sale_key")["income"].sum().rename("X")
    totals = pd.concat([cur, pre], axis=1).fillna(0.0).reset_index()
    names = long[["sale_key", "sale_name"]].drop_duplicates("sale_key")
    return totals.merge(names, on="sale_key", how="left", validate="one_to_one")


# ============================================================
# 3. CUPED, Welch, ANCOVA-HC3, MDE
# ============================================================
def cuped_values(X, Y):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    xc, yc = X - X.mean(), Y - Y.mean()
    ssx = float(xc @ xc)
    theta = float(xc @ yc / ssx) if ssx > 0 else 0.0
    return Y - theta * xc, theta, (2 if ssx > 0 else 1)


def welch_summary(treatment, control, alpha=ALPHA):
    nt, nc = len(treatment), len(control)
    delta = float(np.mean(treatment) - np.mean(control))
    vt, vc = float(np.var(treatment, ddof=1)), float(np.var(control, ddof=1))
    se2 = vt / nt + vc / nc
    if not np.isfinite(se2) or se2 <= 0:
        return {"delta": delta, "se": np.nan, "df": np.nan, "p": np.nan}
    df = se2 ** 2 / ((vt / nt) ** 2 / (nt - 1) + (vc / nc) ** 2 / (nc - 1))
    se = np.sqrt(se2)
    return {"delta": delta, "se": se, "df": df,
            "p": float(2 * t_dist.sf(abs(delta / se), df))}


def ancova_hc3(X, Y, treatment, interaction=ANCOVA_INTERACTION, alpha=ALPHA):
    """OLS на [1,T,Xc,T*Xc] и HC3; t-аппроксимация с df=n-rank(D).

    Это аппроксимация для конечной выборки, а не точный t-тест.
    Коэффициент T — эффект в средней X анализируемой популяции.
    Масштабирование X защищает матрицу от плохой обусловленности в рублях.
    """
    X, Y, treatment = np.asarray(X, float), np.asarray(Y, float), np.asarray(treatment, float)
    columns = [np.ones(len(Y)), treatment]
    xc = X - X.mean()
    scale = float(np.sqrt(np.mean(xc ** 2)))
    if scale > 0:
        z = xc / scale
        columns.append(z)
        if interaction:
            columns.append(treatment * z)
    D = np.column_stack(columns)
    n, p = D.shape
    if n <= p or np.linalg.matrix_rank(D) < p:
        return None
    beta = np.linalg.lstsq(D, Y, rcond=None)[0]
    residual = Y - D @ beta
    # Нулевой шум/точная подгонка не превращаются в MDE=0 и p=0.
    if np.linalg.norm(residual) <= 1e-12 * max(np.linalg.norm(Y), 1.0):
        return None
    bread = np.linalg.inv(D.T @ D)
    leverage = np.einsum("ij,jk,ik->i", D, bread, D)
    if np.any(1.0 - leverage <= 1e-10):
        return None
    weights = (residual / (1.0 - leverage)) ** 2
    cov = bread @ (D.T @ (weights[:, None] * D)) @ bread
    se2 = float(cov[1, 1])
    if not np.isfinite(se2) or se2 <= 0:
        return None
    delta, se, df = float(beta[1]), float(np.sqrt(se2)), n - p
    critical = float(t_dist.ppf(1 - alpha / 2, df))
    return {"delta": delta, "se": se, "df": df,
            "p": float(2 * t_dist.sf(abs(delta / se), df)),
            "lo": delta - critical * se, "hi": delta + critical * se}


@lru_cache(maxsize=4096)
def mde_multiplier(df, alpha=ALPHA, power=POWER):
    """Множитель MDE/SE из мощности двустороннего noncentral-t теста.

    Для HC3 это планировочное приближение при фиксированном SE и df.
    Для design используется оценка шума из текущей диагностической выборки.
    """
    if not np.isfinite(df) or df <= 0 or not 0 < alpha < power < 1:
        return np.nan
    critical = t_dist.ppf(1 - alpha / 2, df)
    def achieved(noncentrality):
        return nct.sf(critical, df, noncentrality) + nct.cdf(-critical, df, noncentrality)
    upper = 1.0
    while achieved(upper) < power and upper < 1024:
        upper *= 2
    return float(brentq(lambda a: achieved(a) - power, 0.0, upper))


def signal_label(delta, p_value):
    if not np.isfinite(p_value):
        return "НЕТ ОЦЕНКИ"
    if p_value >= ALPHA:
        return "НЕТ СИГНАЛА"
    return "ПОЛОЖИТЕЛЬНЫЙ СИГНАЛ" if delta > 0 else "ОТРИЦАТЕЛЬНЫЙ СИГНАЛ"


# ============================================================
# 4. Диагностический перебор порогов
# ============================================================
def ordered_sales(frame):
    return frame.sort_values([TRIM_COLUMN, "sale_key"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def step_info(frame, k):
    kept = frame.iloc[k:]
    return {"удалено": k, "удалено, %": 100.0 * k / len(frame),
            "осталось": len(kept), "осталось, %": 100.0 * len(kept) / len(frame),
            "удалён на шаге": frame.iloc[k - 1]["sale_name"] if k else "—",
            "Y удалённого": float(frame.iloc[k - 1]["Y"]) if k else np.nan,
            "максимальный Y оставшихся": float(kept["Y"].max())}


def trimmed_samples(frame):
    """Общие подвыборки для таблиц дизайна и AB.

    При равных исходных n удаляем по одному из каждой группы за шаг.
    При разных n удаляем одну долю, округляя число удалений вниз.
    Исходный дисбаланс после фильтра Y>0 автоматически не исправляется.
    Без меток групп (DESIGN_POPULATION='all') используется общий порог.
    """
    frame = ordered_sales(frame)
    if not TRIM_WITHIN_GROUP or "group" not in frame.columns:
        for k in range(len(frame)):
            kept = frame.iloc[k:]
            yield kept, {**step_info(frame, k), "поле отсечения": TRIM_COLUMN,
                         "режим отсечения": "общий список"}
        return
    treatment = ordered_sales(frame[frame["group"] == "treatment"])
    control = ordered_sales(frame[frame["group"] == "control"])
    nt0, nc0 = len(treatment), len(control)
    base = min(nt0, nc0)
    if base < MIN_N_GROUP:
        return
    previous_t = previous_c = 0
    for step in range(base + 1):
        # Целочисленная арифметика исключает ошибки округления float.
        kt, kc = step * nt0 // base, step * nc0 // base
        kept_t, kept_c = treatment.iloc[kt:], control.iloc[kc:]
        if min(len(kept_t), len(kept_c)) < MIN_N_GROUP:
            break
        kept = pd.concat([kept_t, kept_c], ignore_index=True)
        newly_removed = pd.concat([treatment.iloc[previous_t:kt], control.iloc[previous_c:kc]])
        n, removed = len(kept), kt + kc
        info = {"удалено": removed, "удалено, %": 100.0 * removed / len(frame),
                "осталось": n, "осталось, %": 100.0 * n / len(frame),
                "исходно n_treatment": nt0, "исходно n_control": nc0,
                "удалено treatment": kt, "удалено control": kc,
                "удалено treatment, %": 100.0 * kt / nt0,
                "удалено control, %": 100.0 * kc / nc0,
                "удалён на шаге": "; ".join(newly_removed["sale_name"]) if removed else "—",
                "Y удалённого": float(newly_removed["Y"].iloc[0]) if len(newly_removed) == 1 else np.nan,
                "сумма Y удалённых на шаге": float(newly_removed["Y"].sum()),
                "максимальный Y оставшихся": float(kept["Y"].max()),
                "поле отсечения": TRIM_COLUMN, "режим отсечения": "доля внутри каждой группы",
                "порог отсечения treatment": float(kept_t[TRIM_COLUMN].max()),
                "порог отсечения control": float(kept_c[TRIM_COLUMN].max())}
        yield kept, info
        previous_t, previous_c = kt, kc


def design_scan(frame):
    frame = ordered_sales(frame)
    rows = []
    # Нет подгонки по двум точкам: расчёт лишь для n >= 2*MIN_N_GROUP.
    for kept, info in trimmed_samples(frame):
        Y, X = kept["Y"].to_numpy(float), kept["X"].to_numpy(float)
        n = len(Y)
        if n < 2 * MIN_N_GROUP:
            break
        adjusted, theta, n_params = cuped_values(X, Y)
        var_y = float(np.var(Y, ddof=1))
        var_adjusted = float(np.var(adjusted, ddof=1))
        sse = float(np.sum((adjusted - adjusted.mean()) ** 2))
        # Поправка на число оценённых параметров: intercept и, если есть, theta.
        sigma2 = sse / (n - n_params)
        nt, nc = n // 2, n - n // 2
        degenerate = sse <= 1e-24 * max(float(Y @ Y), 1.0)
        se = np.sqrt(sigma2 * (1 / nt + 1 / nc))
        mde = mde_multiplier(float(n - n_params)) * se if not degenerate else np.nan
        target_size = n if EFFECT_TARGET == "rollout" else nt
        rows.append({**info, "план n_treatment": nt, "план n_control": nc,
                     "theta CUPED": theta, "дисперсия Y": var_y,
                     "описательная дисперсия CUPED": var_adjusted,
                     "оценка остаточной дисперсии": sigma2,
                     "снижение описательной дисперсии, %": 100 * (1 - var_adjusted / var_y) if var_y > 0 else np.nan,
                     "MDE на сейлза, прибл., руб.": mde,
                     "MDE treatment, прибл., руб.": mde * nt,
                     "MDE rollout, прибл., руб.": mde * n,
                     "MDE выбранного масштаба, руб.": mde * target_size,
                     "статус": "ВЫРОЖДЕННАЯ ДИСПЕРСИЯ" if degenerate else "OK"})
    return pd.DataFrame(rows)


def ab_scan(frame):
    frame = ordered_sales(frame)
    rows = []
    for kept, info in trimmed_samples(frame):
        mask = kept["group"].eq("treatment").to_numpy()
        nt, nc = int(mask.sum()), int((~mask).sum())
        # При удалениях численности групп могут только убывать.
        if min(nt, nc) < MIN_N_GROUP:
            break
        Y, X = kept["Y"].to_numpy(float), kept["X"].to_numpy(float)
        adjusted, theta, _ = cuped_values(X, Y)
        welch = welch_summary(adjusted[mask], adjusted[~mask])
        raw = welch_summary(Y[mask], Y[~mask])
        model = ancova_hc3(X, Y, mask)
        delta = model["delta"] if model else np.nan
        p_value = model["p"] if model else np.nan
        mde = mde_multiplier(float(model["df"])) * model["se"] if model else np.nan
        lo, hi = (model["lo"], model["hi"]) if model else (np.nan, np.nan)
        var_y, var_adj = float(np.var(Y, ddof=1)), float(np.var(adjusted, ddof=1))
        target_size = len(Y) if EFFECT_TARGET == "rollout" else nt
        rows.append({**info, "n_treatment": nt, "n_control": nc,
                     "средний RAW доход T, руб.": float(Y[mask].mean()),
                     "средний RAW доход C, руб.": float(Y[~mask].mean()),
                     "разница RAW средних, руб.": raw["delta"],
                     "разница RAW сумм T-C, руб.": float(Y[mask].sum() - Y[~mask].sum()),
                     "разница CUPED средних, руб.": welch["delta"],
                     "p Welch CUPED, прибл.": welch["p"],
                     "эффект ANCOVA на сейлза, руб.": delta,
                     "SE ANCOVA HC3, руб.": model["se"] if model else np.nan,
                     "df ANCOVA": model["df"] if model else np.nan,
                     "p ANCOVA HC3, прибл.": p_value,
                     "ДИ на сейлза, нижний": lo, "ДИ на сейлза, верхний": hi,
                     "эффект treatment, руб.": delta * nt,
                     "эффект rollout, руб.": delta * len(Y),
                     "ДИ treatment, нижний": lo * nt, "ДИ treatment, верхний": hi * nt,
                     "ДИ rollout, нижний": lo * len(Y), "ДИ rollout, верхний": hi * len(Y),
                     "MDE на сейлза, прибл., руб.": mde,
                     "MDE treatment, прибл., руб.": mde * nt,
                     "MDE rollout, прибл., руб.": mde * len(Y),
                     "MDE выбранного масштаба, руб.": mde * target_size,
                     "theta CUPED": theta, "дисперсия Y": var_y,
                     "описательная дисперсия CUPED": var_adj,
                     "снижение описательной дисперсии, %": 100 * (1 - var_adj / var_y) if var_y > 0 else np.nan,
                     "диагностический сигнал": signal_label(delta, p_value)})
    return pd.DataFrame(rows)


# ============================================================
# 5. Общая сборка; можно вызвать напрямую с уже загруженными DataFrame
# ============================================================
def run_analysis(dgr, ab):
    if EFFECT_TARGET not in {"rollout", "treatment"} or DESIGN_POPULATION not in {"ab", "all"}:
        raise ValueError("Проверьте EFFECT_TARGET и DESIGN_POPULATION")
    if TRIM_COLUMN not in {"X", "Y"}:
        raise ValueError("TRIM_COLUMN должен быть X или Y")
    if MIN_N_GROUP < 3 or not 0 < ALPHA < POWER < 1:
        raise ValueError("Проверьте MIN_N_GROUP, ALPHA и POWER")
    roster = prepare_ab(ab)
    long, audit = prepare_deals(dgr)
    sales = aggregate_sales(long, PRE_MONTHS, TEST_MONTHS)
    ab_keys, deal_keys = set(roster["sale_key"]), set(long["sale_key"])
    name_lookup = long[["sale_key", "sale_name"]].drop_duplicates("sale_key").set_index("sale_key")["sale_name"]
    diff_rows = ([{"ФИО": row.sale_name, "источник": "только в AB-файле"}
                  for row in roster.itertuples() if row.sale_key not in deal_keys]
                 + [{"ФИО": name_lookup.loc[key], "источник": "только в сделках"}
                    for key in sorted(deal_keys - ab_keys)])
    names_diff = pd.DataFrame(diff_rows, columns=["ФИО", "источник"])
    # Явный состав roster; отсутствие строк дохода видно отдельным флагом.
    cohort = roster.merge(sales[["sale_key", "X", "Y"]], on="sale_key", how="left", validate="one_to_one")
    cohort["есть_данные_в_окнах"] = cohort["Y"].notna()
    cohort[["X", "Y"]] = cohort[["X", "Y"]].fillna(0.0)
    cohort["в диагностической выборке"] = cohort["Y"].gt(0) if POSITIVE_Y_ONLY else True
    selected = cohort[cohort["в диагностической выборке"]].copy()
    selected = ordered_sales(selected)
    design_data = selected if DESIGN_POPULATION == "ab" else sales.copy()
    if POSITIVE_Y_ONLY:
        design_data = design_data[design_data["Y"] > 0]
    design = design_scan(design_data)
    steps = ab_scan(selected)
    if steps.empty:
        raise ValueError(f"Недостаточно участников: нужны минимум {MIN_N_GROUP} в каждой группе. "
                         f"После фильтра: {selected['group'].value_counts().to_dict()}")
    if design.empty:
        raise ValueError("Недостаточно сейлзов для таблицы чувствительности дизайна")
    shared_rows = long.groupby("строка Excel").agg(n_sales=("sale_key", "nunique"))
    multi = set(shared_rows.index[shared_rows["n_sales"] > 1])
    shared = long[long["строка Excel"].isin(multi)].merge(
        roster[["sale_key", "group"]], on="sale_key", how="left", validate="many_to_one")
    shared = shared.copy()
    shared["month_p"] = shared["month_p"].astype(str)
    metadata = pd.DataFrame([
        ("Пре-период", ", ".join(PRE_MONTHS)), ("Тестовый период", ", ".join(TEST_MONTHS)),
        ("Фильтр Y > 0", POSITIVE_Y_ONLY), ("MIN_N_GROUP", MIN_N_GROUP),
        ("alpha", ALPHA), ("power", POWER), ("Доверительная вероятность, %", 100 * (1 - ALPHA)),
        ("Масштаб цели MDE", EFFECT_TARGET),
        ("Целевой суммарный MDE, руб.", EFFECT_THRESHOLD), ("Популяция дизайна", DESIGN_POPULATION),
        ("Основная модель", "ANCOVA + HC3, t-аппроксимация"),
        ("Взаимодействие T*X", ANCOVA_INTERACTION),
        ("MDE", "приближение noncentral-t при фиксированных SE и df"),
        ("Поле отсечения", TRIM_COLUMN),
        ("Отсечения внутри T/C", TRIM_WITHIN_GROUP),
        ("Перебор", "одинаковая доля внутри T/C; без меток групп — общий список" if TRIM_WITHIN_GROUP
         else "общий список; последовательное удаление по убыванию выбранного дохода"),
        ("Состав AB до фильтра", len(cohort)), ("Состав AB после фильтра", len(selected)),
        ("Совместные сделки", len(multi)),
        ("Исходные строки с пропуском Доход", "ошибка; в ноль не заменяются"),
        ("Совместные сделки и взаимное влияние", "отдельная диагностика; независимость модель не проверяет"),
    ], columns=["настройка", "значение"])
    audit_summary = (audit.groupby("причина", as_index=False).agg(
        **{"число строк": ("строка Excel", "size"), "известный доход, руб.": ("Доход", "sum")}))
    return {"design_steps": design, "ab_steps": steps, "sales": sales, "design_participants": design_data,
            "participants": selected, "roster": cohort, "names_diff": names_diff,
            "audit": audit, "audit_summary": audit_summary, "shared_deals": shared, "metadata": metadata}


def save_reports(tables, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    design_path, ab_path = out_dir / "cuped_variance_steps.xlsx", out_dir / "ab_test_steps.xlsx"
    with pd.ExcelWriter(design_path, engine="openpyxl") as writer:
        tables["design_steps"].to_excel(writer, index=False, sheet_name="CUPED_steps")
        tables["design_participants"].to_excel(writer, index=False, sheet_name="Данные_сейлзов")
        tables["metadata"].to_excel(writer, index=False, sheet_name="Настройки")
    with pd.ExcelWriter(ab_path, engine="openpyxl") as writer:
        for key, title in [("ab_steps", "AB_тест_шаги"), ("participants", "Данные_участников"),
                           ("roster", "Полный_список_AB"), ("names_diff", "Имена_расхождения"),
                           ("audit", "Пропущенные_строки"), ("audit_summary", "Сводка_пропусков"),
                           ("shared_deals", "Совместные_сделки"), ("metadata", "Настройки")]:
            tables[key].to_excel(writer, index=False, sheet_name=title)
    return design_path, ab_path


def print_summary(tables):
    cohort, selected = tables["roster"], tables["participants"]
    print(f"Пре-период: {', '.join(PRE_MONTHS)} | тест: {', '.join(TEST_MONTHS)}")
    print(f"AB до фильтра: {len(cohort)}; после: {len(selected)}; "
          f"группы: {selected['group'].value_counts().to_dict()}")
    if not tables["audit_summary"].empty:
        print("\nСтроки, не вошедшие в распределённый доход:")
        print(tables["audit_summary"].to_string(index=False))
    print(f"\nРасхождения имён: {len(tables['names_diff'])}; подробности в Excel.")
    for key, title in [("design_steps", "Планируемое равное разбиение"), ("ab_steps", "Текущие группы")]:
        frame = tables[key]
        below = frame[frame["MDE выбранного масштаба, руб."] <= EFFECT_THRESHOLD]
        print(f"\n{title}: MDE <= {EFFECT_THRESHOLD:,.0f} руб., масштаб={EFFECT_TARGET}")
        if below.empty:
            print("Подходящих порогов нет.")
        else:
            row = below.iloc[0]
            cols = ["удалено", "осталось", "осталось, %", "MDE на сейлза, прибл., руб.",
                    "MDE treatment, прибл., руб.", "MDE rollout, прибл., руб."]
            print(pd.DataFrame([row[cols]]).to_string(index=False))
    steps = tables["ab_steps"]
    positive = steps[steps["диагностический сигнал"] == "ПОЛОЖИТЕЛЬНЫЙ СИГНАЛ"]
    print("\nПервый порог с положительным диагностическим сигналом ANCOVA:")
    if positive.empty:
        print("Не найден.")
    else:
        cols = ["удалено", "осталось", "n_treatment", "n_control", "эффект ANCOVA на сейлза, руб.",
                "ДИ на сейлза, нижний", "ДИ на сейлза, верхний", "p ANCOVA HC3, прибл.",
                "эффект treatment, руб.", "эффект rollout, руб."]
        print(positive.iloc[[0]][cols].to_string(index=False))


def main():
    dgr = pd.read_excel(DATA_FILE)
    ab = pd.read_excel(AB_FILE, sheet_name=AB_SHEET)
    tables = run_analysis(dgr, ab)
    paths = save_reports(tables)
    print_summary(tables)
    for path in paths:
        print(f"Файл сохранён: {path.resolve()}")
    return tables


if __name__ == "__main__":
    main()
