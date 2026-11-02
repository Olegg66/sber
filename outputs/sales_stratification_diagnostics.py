# -*- coding: utf-8 -*-
"""Выбор признаков для стратификации AB-теста сейлзов.

Зависимости: pandas, numpy, scipy, openpyxl. Новых ML-библиотек нет.
Запуск: python sales_stratification_diagnostics.py
В Jupyter: tables = run_analysis(dgr, ab); display(tables['categorical_ranking']).

Единица анализа — сейлз. Все признаки строятся по PRE_MONTHS; целевая
переменная — суммарный доход того же сейлза за TARGET_MONTHS.
Пересчёт FICC/Flow и деление дохода между соисполнителями такие же,
как в исправленном коде AB. ФИО используются только как идентификаторы.

Категориальные поля оцениваются тремя способами:
1) объяснённая доля разброса дохода eta^2 — описательная величина;
2) прогноз дохода на отложенных сейлзах: repeated K-fold CV R^2;
3) аналитическая дисперсия оценки эффекта при полном случайном разбиении
   и разбиении внутри страт, отдельно на RAW и на CUPED-скорректированном Y.

CV-модели — ridge с фиксированной регуляризацией; результат зависит от
модели. Отрицательные CV R^2 и отрицательные выигрыши не обрезаются.
Снижение дисперсии по историческому Y — диагностика дизайна для этой
популяции, а не гарантия величины снижения в новом периоде.

Кол-во клиентов суммируется по строкам и делится между соисполнителями.
Это НЕ число уникальных клиентов: по агрегированной выгрузке его нет.
"""

from datetime import date, datetime
from itertools import combinations
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# 0. Настройки
# ============================================================
DATA_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\data\441774c5-7986-43a6-8269-5d5ce8627e9b.xlsx")
AB_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\Телефоны УГР 29.06.2026.xlsx")
# AB_FILE = None  # Если нужен анализ всех сейлзов, без списка участников AB.
AB_SHEET = "Штатная расстановка"
AB_NAME = "Ф.И.О. сотрудника"
AB_GROUP = "AB-тест"
OUT_DIR = Path("ab sales/stratification")

PRE_MONTHS = ("2026-05", "2026-06")
TARGET_MONTHS = ("2026-09", "2026-10")
POSITIVE_Y_ONLY = True  # Для сопоставимости с предыдущим диагностическим скриптом.
MIN_SALES = 20
MIN_STRATUM_SIZE = 10
MAX_PROFILE_LEVELS = 30
MIN_PROFILE_SUPPORT = 3
CHECK_PAIRS = True
NUMERIC_QUANTILES = (3, 4, 5)
CV_FOLDS = 5
CV_REPEATS = 10
RIDGE_ALPHA = 10.0
RANDOM_SEED = 20261102

CATEGORY_FIELDS = ("ТБ", "Сегмент", "Дэск", "ДКК", "Отрасль", "Продукт")
ROLE_FIELD = "тип сейлза"
DEALS_COL = "Кол-во сделок"
CLIENTS_COL = "Кол-во клиентов"
NO_HISTORY = "__НЕТ_ДАННЫХ_ДО_ТЕСТА__"
OTHER = "__ОСТАЛЬНЫЕ__"

# Общие продуктовые справочники и функции очистки вставлены ниже.
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


# ============================================================
# 1. Длинная таблица с атрибуцией по продуктам
# ============================================================
def prepare_deals(dgr):
    dgr = dgr.copy()
    dgr.columns = [clean_text(c) for c in dgr.columns]
    if not dgr.columns.is_unique:
        raise ValueError("После очистки заголовков появились одинаковые колонки")
    require_columns(dgr, list(CATEGORY_FIELDS) + [FLOW_COL, FICC_COL, "Доход", DEALS_COL,
                                                CLIENTS_COL, "Месяц-Год"], "dgr")
    if flow_products & ficc_products or (flow_products | ficc_products) & excluded_products:
        raise ValueError("Справочники продуктов пересекаются")
    wanted = {pd.Period(m, freq="M") for m in PRE_MONTHS + TARGET_MONTHS}
    records, audit = [], []
    for excel_row, (_, row) in enumerate(dgr.iterrows(), start=2):
        product = clean_text(row["Продукт"])
        month = parse_month(row["Месяц-Год"])
        reason = ""
        if product in excluded_products:
            reason = "исключённый продукт"
        elif product not in flow_products | ficc_products:
            reason = "неизвестный продукт"
        elif pd.isna(month):
            reason = "не распознана дата"
        elif month.to_period("M") not in wanted:
            continue
        else:
            role = "Flow" if product in flow_products else "FICC"
            names = split_names(row[FLOW_COL if role == "Flow" else FICC_COL])
            if not names:
                reason = "нет ФИО в выбранной колонке"
        if reason:
            audit.append({"строка Excel": excel_row, "причина": reason, "Продукт": product,
                          "Доход": pd.to_numeric(row["Доход"], errors="coerce")})
            continue
        values = {}
        for column in ["Доход", DEALS_COL, CLIENTS_COL]:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value) or not np.isfinite(value) or (column != "Доход" and value < 0):
                raise ValueError(f"Строка Excel {excel_row}: некорректное {column}={row[column]!r}")
            values[column] = float(value)
        for key, name in names:
            rec = {"sale_key": key, "sale_name": name, "month_p": month.to_period("M"),
                   "income": values["Доход"] / len(names),
                   "deals": values[DEALS_COL] / len(names),
                   "clients_entries": values[CLIENTS_COL] / len(names),
                   "row_share": 1 / len(names), ROLE_FIELD: role,
                   "строка Excel": excel_row}
            rec.update({field: clean_text(row[field]).casefold() or "__ПРОПУСК__"
                        for field in CATEGORY_FIELDS})
            records.append(rec)
    if not records:
        raise ValueError("В выбранных периодах нет подходящих сделок")
    long = pd.DataFrame(records)
    missing = wanted - set(long["month_p"])
    if missing:
        raise ValueError(f"Отсутствуют целые месяцы: {sorted(str(m) for m in missing)}")
    return long, pd.DataFrame(audit, columns=["строка Excel", "причина", "Продукт", "Доход"])


def prepare_roster(long, ab):
    if ab is None:
        return long[["sale_key", "sale_name"]].drop_duplicates("sale_key").reset_index(drop=True)
    ab = ab.copy()
    ab.columns = [clean_text(c) for c in ab.columns]
    require_columns(ab, [AB_NAME], "AB-файле")
    # Метки групп нужны только для отбора списка участников; как признаки не используются.
    if AB_GROUP in ab.columns:
        groups = ab[AB_GROUP].map(clean_text).str.casefold()
        ab = ab[groups.isin(["treatment", "control"])]
    roster = pd.DataFrame({"sale_key": ab[AB_NAME].map(name_key),
                           "sale_name": ab[AB_NAME].map(clean_text)})
    if roster.empty or roster["sale_key"].eq("").any():
        raise ValueError("Пустой список участников или пропущено ФИО")
    return roster.drop_duplicates("sale_key").reset_index(drop=True)


# ============================================================
# 2. Одна строка на сейлза; признаки только из PRE_MONTHS
# ============================================================
def manager_features(long, ab=None):
    roster = prepare_roster(long, ab)
    pre_set = {pd.Period(m, freq="M") for m in PRE_MONTHS}
    target_set = {pd.Period(m, freq="M") for m in TARGET_MONTHS}
    pre = long[long["month_p"].isin(pre_set)].copy()
    target = long[long["month_p"].isin(target_set)]
    summary = pre.groupby("sale_key").agg(
        income_pre=("income", "sum"), deals_pre=("deals", "sum"),
        clients_entries_pre=("clients_entries", "sum"), rows_pre=("row_share", "sum"))
    Y = target.groupby("sale_key")["income"].sum().rename("Y")
    managers = roster.merge(summary, on="sale_key", how="left", validate="one_to_one")
    managers = managers.merge(Y, on="sale_key", how="left", validate="one_to_one")
    managers["нет истории"] = managers["rows_pre"].isna()
    managers["нет целевых строк"] = managers["Y"].isna()
    numeric = ["income_pre", "deals_pre", "clients_entries_pre", "rows_pre", "Y"]
    managers[numeric] = managers[numeric].fillna(0.0)
    # Не пропускаем незамеченным ФИО, не сопоставившееся вообще ни с одной строкой.
    unmatched = managers[~managers["sale_key"].isin(set(long["sale_key"]))][["sale_name"]].copy()
    if POSITIVE_Y_ONLY:
        managers = managers[managers["Y"] > 0].copy()
    managers = managers.sort_values("sale_key").reset_index(drop=True)
    if len(managers) < MIN_SALES:
        raise ValueError(f"Недостаточно сейлзов: {len(managers)}; нужно минимум {MIN_SALES}")
    keys = managers["sale_key"].tolist()
    # Вес категории — доля сделок до теста. Если у человека все counts=0,
    # используем число распределённых строк; доходом категории не взвешиваются.
    pre["weight"] = pre["deals"]
    no_deals = pre.groupby("sale_key")["deals"].transform("sum").eq(0)
    pre.loc[no_deals, "weight"] = pre.loc[no_deals, "row_share"]
    profiles, dominant = {}, pd.DataFrame(index=managers.index)
    profile_rows = []
    for field in CATEGORY_FIELDS + (ROLE_FIELD,):
        exposures = pre.groupby(["sale_key", field])["weight"].sum().unstack(fill_value=0)
        exposures = exposures.reindex(keys, fill_value=0).fillna(0.0)
        denominator = exposures.sum(axis=1)
        shares = exposures.div(denominator.replace(0, np.nan), axis=0).fillna(0.0)
        shares[NO_HISTORY] = denominator.eq(0).astype(float)
        # Порядок столбцов задаёт детерминированное разрешение равных долей.
        shares = shares.reindex(sorted(shares.columns), axis=1)
        dominant[field] = shares.idxmax(axis=1).to_numpy()
        managers[field + " | доминирует"] = dominant[field]
        managers[field + " | доля доминирующей"] = shares.max(axis=1).to_numpy()
        support = shares.gt(0).sum().sort_values(ascending=False, kind="mergesort")
        keep = support[support >= MIN_PROFILE_SUPPORT].head(MAX_PROFILE_LEVELS).index
        if NO_HISTORY in shares.columns and shares[NO_HISTORY].sum() > 0:
            keep = keep.union(pd.Index([NO_HISTORY]), sort=False)
        matrix = shares.loc[:, keep].copy()
        dropped = shares.columns.difference(keep)
        if len(dropped):
            matrix[OTHER] = shares[dropped].sum(axis=1)
        profiles[field] = matrix.to_numpy(float)
        for key, values in shares.iterrows():
            for category, fraction in values[values > 0].items():
                profile_rows.append({"sale_key": key, "поле": field, "категория": category,
                                     "доля сделок до теста": fraction})
    return managers, dominant, profiles, pd.DataFrame(profile_rows), unmatched


# ============================================================
# 3. Прогноз на отложенных сейлзах, без sklearn
# ============================================================
def make_splits(n):
    folds = min(CV_FOLDS, n // 4)
    if folds < 2:
        raise ValueError("Недостаточно наблюдений для CV")
    rng = np.random.default_rng(RANDOM_SEED)
    return [[np.asarray(chunk, int) for chunk in np.array_split(rng.permutation(n), folds)]
            for _ in range(CV_REPEATS)]


def ridge_predict(X, Y, train, test):
    if X.shape[1] == 0:
        return np.full(len(test), np.mean(Y[train]))
    means = X[train].mean(axis=0)
    scales = X[train].std(axis=0)
    usable = scales > 1e-12
    if not usable.any():
        return np.full(len(test), np.mean(Y[train]))
    A = (X[train][:, usable] - means[usable]) / scales[usable]
    B = (X[test][:, usable] - means[usable]) / scales[usable]
    yc = Y[train] - Y[train].mean()
    # Dual-вариант при большом числе категорий: не обращаем огромную p*p матрицу.
    if A.shape[1] <= len(train):
        beta = np.linalg.solve(A.T @ A + RIDGE_ALPHA * np.eye(A.shape[1]), A.T @ yc)
    else:
        beta = A.T @ np.linalg.solve(A @ A.T + RIDGE_ALPHA * np.eye(len(train)), yc)
    return Y[train].mean() + B @ beta


def cv_scores(X, Y, splits):
    denominator = float(np.sum((Y - Y.mean()) ** 2))
    scores = []
    for repeat in splits:
        prediction = np.empty(len(Y))
        for test in repeat:
            train = np.setdiff1d(np.arange(len(Y)), test, assume_unique=True)
            prediction[test] = ridge_predict(X, Y, train, test)
        scores.append(1 - np.sum((Y - prediction) ** 2) / denominator if denominator > 0 else np.nan)
    return np.asarray(scores)


def numeric_block(values):
    values = np.asarray(values, float)
    # Целевая метрика остаётся в рублях; log — только дополнительный предиктор.
    return np.column_stack([values, np.sign(values) * np.log1p(np.abs(values))])


# ============================================================
# 4. Категориальные страты и дисперсия оценки эффекта
# ============================================================
def collapse_small_strata(labels):
    labels = pd.Series(list(labels), dtype=object)
    counts = labels.value_counts()
    rare = set(counts[counts < MIN_STRATUM_SIZE].index)
    mapped = labels.map(lambda x: OTHER if x in rare else x)
    # Маленький остаточный OTHER объединяем с самой маленькой большой стратой.
    counts = mapped.value_counts()
    if OTHER in counts and counts[OTHER] < MIN_STRATUM_SIZE and len(counts) > 1:
        candidates = [(count, str(label), label) for label, count in counts.items() if label != OTHER]
        merge_label = min(candidates)[2]
        mapped = mapped.map(lambda x: OTHER if x == merge_label else x)
    merged_fraction = float(np.mean([a != b for a, b in zip(labels, mapped)]))
    return mapped.to_numpy(), merged_fraction


def eta_squared(Y, labels):
    denominator = float(np.sum((Y - Y.mean()) ** 2))
    if denominator <= 0:
        return np.nan
    frame = pd.DataFrame({"Y": Y, "label": labels})
    groups = frame.groupby("label", sort=False)["Y"]
    between = sum(len(g) * (g.mean() - Y.mean()) ** 2 for _, g in groups)
    return float(between / denominator)


def randomization_variance(Y, labels):
    """Точная design-based дисперсия на фиксированных Y при нулевом эффекте.

    Внутри каждой страты фиксируем nT близкое к n/2, суммы nT дают floor(N/2).
    Итоговая оценка = sum_h (Nh/N)*(mean_T,h-mean_C,h), а не простая разница
    общих средних, если доли назначения в нечётных стратах различаются.
    """
    n = len(Y)
    nt = n // 2
    complete = float(np.var(Y, ddof=1) * (1 / nt + 1 / (n - nt)))
    unique = sorted(set(labels), key=str)
    groups = [np.flatnonzero(labels == label) for label in unique]
    allocation = [len(g) // 2 for g in groups]
    extras = nt - sum(allocation)
    for j, group in enumerate(groups):
        if extras and len(group) % 2:
            allocation[j] += 1
            extras -= 1
    variance = 0.0
    detail = []
    for label, idx, nt_h in zip(unique, groups, allocation):
        nh, nc_h = len(idx), len(idx) - nt_h
        if min(nt_h, nc_h) == 0:
            raise ValueError("Страта слишком мала для назначения в обе группы")
        variance += (nh / n) ** 2 * np.var(Y[idx], ddof=1) * (1 / nt_h + 1 / nc_h)
        detail.append({"страта": str(label), "n": nh, "план nT": nt_h, "план nC": nc_h,
                       "средний Y, руб.": float(np.mean(Y[idx])), "SD Y, руб.": float(np.std(Y[idx], ddof=1))})
    gain = 100 * (1 - variance / complete) if complete > 0 else np.nan
    return complete, float(variance), gain, detail


def describe_score(values):
    finite = values[np.isfinite(values)]
    return (float(finite.mean()), float(finite.std(ddof=1)) if len(finite) > 1 else np.nan) if len(finite) else (np.nan, np.nan)


# ============================================================
# 5. Рейтинги отдельных полей, пар полей и числовых признаков
# ============================================================
def run_analysis(dgr, ab=None):
    if MIN_STRATUM_SIZE < 2 or CV_REPEATS < 1 or RIDGE_ALPHA <= 0 or CV_FOLDS < 2:
        raise ValueError("Проверьте MIN_STRATUM_SIZE, CV_REPEATS, CV_FOLDS, RIDGE_ALPHA")
    if not PRE_MONTHS or not TARGET_MONTHS or max(PRE_MONTHS) >= min(TARGET_MONTHS):
        raise ValueError("PRE_MONTHS должен целиком предшествовать TARGET_MONTHS")
    long, audit = prepare_deals(dgr)
    managers, dominant, profiles, profile_table, unmatched = manager_features(long, ab)
    Y, X = managers["Y"].to_numpy(float), managers["income_pre"].to_numpy(float)
    splits = make_splits(len(Y))
    baseline_block = numeric_block(X)
    base_scores = cv_scores(baseline_block, Y, splits)
    constant_scores = cv_scores(np.empty((len(Y), 0)), Y, splits)
    # CUPED здесь описательный: одно общее theta для исторической популяции.
    xc = X - X.mean()
    theta = float(xc @ (Y - Y.mean()) / (xc @ xc)) if xc @ xc > 0 else 0.0
    residual_Y = Y - theta * xc
    base_r2, base_sd = describe_score(base_scores)
    business_fields = list(dominant.columns)
    candidates = [(field,) for field in business_fields]
    if CHECK_PAIRS:
        candidates += list(combinations(business_fields, 2))
    quantile_fields = set()
    numerical_values = [("Доход до теста", X),
                        ("Кол-во сделок до теста", managers["deals_pre"].to_numpy(float)),
                        ("Сумма клиентских счётчиков до теста", managers["clients_entries_pre"].to_numpy(float))]
    effective_q = sorted({min(int(q), max(1, len(Y) // MIN_STRATUM_SIZE)) for q in NUMERIC_QUANTILES})
    for field, values in numerical_values:
        for q in effective_q:
            title = f"{field}: q={q}"
            codes = pd.Series(pd.qcut(values, q=q, labels=False, duplicates="drop"))
            labels = codes.map(lambda v: f"квантиль {int(v) + 1}" if pd.notna(v) else "все значения")
            dominant[title] = labels.to_numpy()
            profiles[title] = pd.get_dummies(labels, dtype=float).to_numpy()
            quantile_fields.add(title)
            candidates.append((title,))
    rows, stratum_rows, mapping_rows = [], [], []
    for fields in candidates:
        title = " + ".join(fields)
        is_quantile = fields[0] in quantile_fields
        # Редкие КОМБИНАЦИИ объединяются повторно: количество страт не взрывается.
        original = dominant[list(fields)].astype(str).agg(" | ".join, axis=1).to_numpy()
        strata, merged = collapse_small_strata(original)
        dummy = pd.get_dummies(pd.Series(strata), dtype=float).to_numpy()
        profile = np.column_stack([profiles[field] for field in fields])
        cat_scores = cv_scores(dummy, Y, splits)
        profile_scores = cv_scores(profile, Y, splits)
        added_cat = cv_scores(np.column_stack([baseline_block, dummy]), Y, splits)
        added_profile = cv_scores(np.column_stack([baseline_block, profile]), Y, splits)
        cat_r2, cat_sd = describe_score(cat_scores)
        profile_r2, _ = describe_score(profile_scores)
        add_cat_r2, _ = describe_score(added_cat)
        add_profile_r2, _ = describe_score(added_profile)
        incremental, incremental_sd = describe_score(added_cat - base_scores)
        cr, blocked, gain, detail = randomization_variance(Y, strata)
        cr_c, blocked_c, gain_c, _ = randomization_variance(residual_Y, strata)
        rows.append({"кандидат": title, "число полей": len(fields),
                     "тип кандидата": "числовые квантили" if is_quantile else "категориальные поля",
                     "категорий до объединения": len(set(original)), "страт после объединения": len(set(strata)),
                     "объединено сейлзов, %": 100 * merged,
                     "минимальный размер страты": min(pd.Series(strata).value_counts()),
                     "eta2 до объединения, %": 100 * eta_squared(Y, original),
                     "eta2 после объединения, %": 100 * eta_squared(Y, strata),
                     "CV R2 страт": cat_r2, "CV R2 SD повторов": cat_sd,
                     "CV R2 состава портфеля": profile_r2 if not is_quantile else np.nan,
                     "CV R2 прошлый доход": base_r2,
                     "CV R2 доход + категории": add_cat_r2,
                     "CV прирост R2 сверх дохода": incremental,
                     "SD прироста по повторам": incremental_sd,
                     "CV R2 доход + портфель": add_profile_r2 if not is_quantile else np.nan,
                     "CV прирост R2 портфеля сверх дохода": add_profile_r2 - base_r2 if not is_quantile else np.nan,
                     "снижение дисперсии RAW, %": gain,
                     "снижение сверх CUPED, %": gain_c,
                     "дисперсия эффекта RAW без страт": cr,
                     "дисперсия эффекта RAW со стратами": blocked,
                     "дисперсия эффекта CUPED без страт": cr_c,
                     "дисперсия эффекта CUPED со стратами": blocked_c,
                     "отношение SE со стратами к без, RAW": np.sqrt(blocked / cr) if cr > 0 else np.nan,
                     "средняя доля доминирующей": float(np.mean([managers[f + " | доля доминирующей"].mean() for f in fields])) if not is_quantile else np.nan})
        stratum_rows.extend({"кандидат": title, **item} for item in detail)
        mapping_rows.extend({"кандидат": title, "sale_key": key, "исходная категория": old,
                             "страта": str(new)} for key, old, new in zip(managers["sale_key"], original, strata))
    ranking = pd.DataFrame(rows).sort_values(
        ["CV прирост R2 сверх дохода", "снижение сверх CUPED, %"], ascending=False).reset_index(drop=True)
    numerical_rows = []
    for field, values in numerical_values:
        scores = cv_scores(numeric_block(values), Y, splits)
        r2, sd = describe_score(scores)
        nonconstant = np.std(values) > 0 and np.std(Y) > 0
        numerical_rows.append({"признак": field,
                               "Pearson с Y": float(pearsonr(values, Y)[0]) if nonconstant else np.nan,
                               "Spearman с Y": float(spearmanr(values, Y)[0]) if nonconstant else np.nan,
                               "CV R2": r2, "CV R2 SD повторов": sd,
                               "комментарий": "Не уникальные клиенты: сумма счётчиков по строкам" if "клиент" in field else ""})
    metadata = pd.DataFrame([
        ("Единица анализа", "сейлз"), ("Признаки", ", ".join(PRE_MONTHS)),
        ("Целевой доход", ", ".join(TARGET_MONTHS)), ("Сейлзов", len(managers)),
        ("Только Y > 0", POSITIVE_Y_ONLY), ("MIN_STRATUM_SIZE", MIN_STRATUM_SIZE),
        ("Квантильные разбиения числовых признаков", str(NUMERIC_QUANTILES)),
        ("Доминирующая категория", "по распределённому количеству сделок до теста"),
        ("Состав портфеля", "доли распределённого количества сделок по категориям до теста"),
        ("CV folds / repeats", f"{CV_FOLDS} / {CV_REPEATS}"), ("ridge alpha", RIDGE_ALPHA),
        ("CV R2 прошлого дохода", base_r2), ("SD повторов CV R2 дохода", base_sd),
        ("CV R2 прогноз средним train", describe_score(constant_scores)[0]),
        ("theta описательного CUPED", theta),
        ("Дисперсия RAW Y", float(np.var(Y, ddof=1))),
        ("Дисперсия описательного CUPED Y", float(np.var(residual_Y, ddof=1))),
        ("SD повторов", "разброс по разбиениям, не доверительный интервал"),
        ("Признаки-идентификаторы", "КМ и ФИО не используются как предикторы"),
        ("Месяц-Год", "только для выбора временных окон"),
        ("Клиенты", "сумма счётчиков не является числом уникальных клиентов"),
        ("Оценка эффекта со стратами", "взвешенная сумма разниц средних по стратам с весами Nh/N"),
        ("Предположение для дисперсии", "нулевой/одинаковый аддитивный эффект; Y фиксирован"),
        ("Совместные сделки", "сохраняется равное деление; зависимость между сейлзами не моделируется"),
        ("Валидация", "сейлз целиком в train или test; признаки только из пре-периода"),
    ], columns=["параметр", "значение"])
    return {"categorical_ranking": ranking,
            "numeric_ranking": pd.DataFrame(numerical_rows).sort_values("CV R2", ascending=False),
            "managers": managers, "profiles": profile_table, "strata": pd.DataFrame(stratum_rows),
            "assignments": pd.DataFrame(mapping_rows), "unmatched": unmatched,
            "audit": audit, "metadata": metadata}


def save_report(tables, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stratification_diagnostics.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for key, title in [("categorical_ranking", "Рейтинг_стратификаций"), ("numeric_ranking", "Рейтинг_числовых"),
                           ("managers", "Данные_сейлзов"), ("profiles", "Состав_портфелей"),
                           ("strata", "Размеры_страт"), ("assignments", "Страты_сейлзов"),
                           ("unmatched", "ФИО_без_сделок"), ("audit", "Пропущенные_строки"),
                           ("metadata", "Настройки_и_метод")]:
            tables[key].to_excel(writer, sheet_name=title, index=False)
    return path


def show_results(tables):
    cols = ["кандидат", "страт после объединения", "CV R2 страт",
            "CV прирост R2 сверх дохода", "CV R2 состава портфеля",
            "снижение дисперсии RAW, %", "снижение сверх CUPED, %"]
    singles = tables["categorical_ranking"].query("`число полей` == 1")
    print("Отдельные поля и квантильные группы, ранжированные по добавочной прогнозной пользе:")
    try:
        from IPython.display import display
    except ImportError:
        display = lambda x: print(x.to_string(index=False))
    display(singles[cols].round(4))
    if CHECK_PAIRS:
        print("\nЛучшие сочетания полей:")
        display(tables["categorical_ranking"].query("`число полей` == 2")[cols].head(10).round(4))
    print("\nЧисловые признаки:")
    display(tables["numeric_ranking"].round(4))
    useful = tables["categorical_ranking"]
    useful = useful[(useful["страт после объединения"] > 1)
                    & (useful["CV прирост R2 сверх дохода"] > 0)
                    & (useful["снижение сверх CUPED, %"] > 0)]
    if useful.empty:
        print("\nДобавочной пользы категориальной стратификации сверх прошлого дохода не найдено.")
    else:
        print("\nПервый кандидат для проверки нового дизайна:", useful.iloc[0]["кандидат"])


def main():
    dgr = pd.read_excel(DATA_FILE)
    ab = pd.read_excel(AB_FILE, sheet_name=AB_SHEET) if AB_FILE is not None else None
    tables = run_analysis(dgr, ab)
    show_results(tables)
    path = save_report(tables)
    print("\nФайл сохранён:", path.resolve())
    return tables


if __name__ == "__main__":
    main()
