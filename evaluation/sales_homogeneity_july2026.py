# -*- coding: utf-8 -*-
"""Гомогенность групп сейлзов на 01.07.2026.

Standalone: pandas, numpy, scipy, matplotlib, openpyxl (для чтения Excel).
Ноутбук:
    import sales_homogeneity_july2026 as h
    tables = h.run_analysis(dgr, ab)
    files = h.save_report(tables)
    display(tables["numeric"])

Берётся исходный состав AB и только PRE_MONTHS до AS_OF.
Каждый сейлз — одно наблюдение. Нет фильтров по будущей активности/доходу,
нет отсечений выбросов. Клиентский счётчик — сумма, не уникальные клиенты.
KS: все известные значения и отдельно ненулевые. Chi-square: доли NA,
доли нулей и доминирующие категории. При повторах значений KS калибруется
перестановками; для редких категорий — перестановочный p статистики Pearson.
Сохраняются и стандартные p SciPy, и выбранный p, и глобальный Holm.
PNG 300 dpi: белое оформление, прозрачный фон, палитра tab10.
"""
from datetime import date, datetime
from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2_contingency, gaussian_kde

DATA_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\data\441774c5-7986-43a6-8269-5d5ce8627e9b.xlsx")
AB_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\Телефоны УГР 29.06.2026.xlsx")
AB_SHEET = "Штатная расстановка"
AB_NAME = "Ф.И.О. сотрудника"
AB_GROUP = "AB-тест"
AS_OF = "2026-07-01"                  # Правая граница не включается.
PRE_MONTHS = ("2026-05", "2026-06")   # Можно увеличить, все месяцы должны быть < AS_OF.
CATEGORY_FIELDS = ("ТБ", "Сегмент", "Дэск", "ДКК", "Отрасль", "Продукт", "тип сейлза")
DEALS_COL, CLIENTS_COL = "Кол-во сделок", "Кол-во клиентов"
ABSENT_PRE_IS_ZERO = False  # Нет строк: по умолчанию NA; True, если гарантирован нулевой бизнес.
INCLUDE_ACTIVITY_FEATURES = True
CHECK_PORTFOLIO_SHARES = True
DEALS_QUANTILE_Q = 5        # Дополнительный chi-square по PRE-стратам сделок; None отключает.
PERMUTATIONS = 9999
RANDOM_SEED = 20260701
ALPHA = 0.05
OUT_DIR = Path("ab sales/homogeneity_2026_07_01")
DPI = 300
SHOW = False               # Для Agg сохраняем PNG без plt.show().
DENSITY_KIND = "kde"       # "hist" — step-гистограммы, общие 40 bins.
GROUPS = ("control", "treatment")
COLORS = {"control": "#1f77b4", "treatment": "#ff7f0e"}
NO_HISTORY, MISSING = "⟨нет истории⟩", "⟨пропуск⟩"

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
    name = clean_text(clean_text(value).replace("\u200b", "").replace("\ufeff", "")).casefold().replace("ё", "е")
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
    month_text = re.sub(r"[\s.\u200b\ufeff]+", "", s)
    match = re.fullmatch(r"([а-яё]+)(\d{4})(?:г)?", month_text)
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


def validate_settings():
    as_of = pd.Timestamp(AS_OF)
    if as_of != as_of.to_period("M").to_timestamp():
        raise ValueError("Для помесячных данных AS_OF должен быть первым числом месяца")
    periods = [pd.Period(m, freq="M") for m in PRE_MONTHS]
    if not periods or len(set(periods)) != len(periods) or max(periods) >= as_of.to_period("M"):
        raise ValueError("PRE_MONTHS: нужны уникальные месяцы строго до AS_OF")
    if PERMUTATIONS < 99 or not 0 < ALPHA < 1 or DENSITY_KIND not in {"kde", "hist"}:
        raise ValueError("Проверьте PERMUTATIONS >= 99, ALPHA и DENSITY_KIND")
    if len(set(CATEGORY_FIELDS)) != len(CATEGORY_FIELDS):
        raise ValueError("Повторные CATEGORY_FIELDS")
    if not set(CATEGORY_FIELDS) <= {"ТБ", "Сегмент", "Дэск", "ДКК", "Отрасль", "Продукт", "тип сейлза"}:
        raise ValueError("Неизвестное категориальное поле")
    if DEALS_QUANTILE_Q is not None and (not isinstance(DEALS_QUANTILE_Q, int) or DEALS_QUANTILE_Q < 2):
        raise ValueError("DEALS_QUANTILE_Q: целое >= 2 либо None")
    if flow_products & ficc_products or (flow_products | ficc_products) & excluded_products:
        raise ValueError("Списки продуктов пересекаются")


def numeric_cell(value, column, row_number):
    if pd.isna(value) or clean_text(value) in {"", "-", "—"}:
        return np.nan
    result = pd.to_numeric(value, errors="coerce")
    if pd.isna(result) or not np.isfinite(result):
        raise ValueError(f"Строка Excel {row_number}: {column}={value!r} не является числом")
    if column != "Доход" and result < 0:
        raise ValueError(f"Строка Excel {row_number}: отрицательное {column}={value!r}")
    return float(result)


def prepare_pre_deals(dgr):
    data = dgr.copy()
    data.columns = [clean_text(c) for c in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("После очистки названий получились повторные колонки dgr")
    needed = {"Месяц-Год", "Продукт", FLOW_COL, FICC_COL, "Доход", DEALS_COL, CLIENTS_COL}
    require_columns(data, needed | (set(CATEGORY_FIELDS) - {"тип сейлза"}), "dgr")
    months = data["Месяц-Год"].map(parse_month)
    dates = pd.DataFrame({"исходная дата": data["Месяц-Год"].map(clean_text),
                          "месяц": months.map(lambda x: str(x.to_period("M")) if pd.notna(x) else "NA")})
    dates = dates.value_counts(dropna=False).rename("число строк").reset_index()
    pre_set = {pd.Period(m, freq="M") for m in PRE_MONTHS}
    records, audit = [], []
    for row_number, ((_, row), month) in enumerate(zip(data.iterrows(), months), start=2):
        product = clean_text(row["Продукт"])
        if pd.isna(month):
            audit.append({"строка Excel": row_number, "причина": "не распознана дата", "поле": "Месяц-Год"})
            continue
        if month.to_period("M") not in pre_set:
            continue  # Будущие и прочие месяцы вообще не участвуют в агрегациях.
        role = "Flow" if product in flow_products else "FICC" if product in ficc_products else None
        if role is None:
            reason = "исключённый продукт" if product in excluded_products else "неизвестный продукт"
            audit.append({"строка Excel": row_number, "причина": reason, "поле": product})
            continue
        names = split_names(row[FLOW_COL if role == "Flow" else FICC_COL])
        if not names:
            audit.append({"строка Excel": row_number, "причина": "нет ФИО в выбранной колонке", "поле": role})
            continue
        values = {c: numeric_cell(row[c], c, row_number) for c in ("Доход", DEALS_COL, CLIENTS_COL)}
        for column, value in values.items():
            if pd.isna(value):
                audit.append({"строка Excel": row_number, "причина": "пустое числовое поле", "поле": column})
        categories = {f: (role if f == "тип сейлза" else clean_text(row[f]).casefold() or MISSING)
                      for f in CATEGORY_FIELDS}
        for key, name in names:
            records.append({"sale_key": key, "sale_name": name, "month": month.to_period("M"),
                            "income_pre": values["Доход"] / len(names),
                            "deals_pre": values[DEALS_COL] / len(names),
                            "clients_pre": values[CLIENTS_COL] / len(names),
                            "rows_pre": 1.0 / len(names), **categories})
    long = pd.DataFrame(records)
    observed = set(long["month"]) if len(long) else set()
    missing = pre_set - observed
    if missing:
        raise ValueError("В распределённых сделках отсутствуют PRE-месяцы: " + str(sorted(map(str, missing)))
                         + ". Распознанные даты: " + str(dates.to_dict("records"))
                         + ". Причины пропусков: " + str(pd.Series([r["причина"] for r in audit], dtype=object).value_counts().to_dict()))
    return long, pd.DataFrame(audit, columns=["строка Excel", "причина", "поле"]), dates


def complete_sum(series):
    # Частично неизвестный показатель не подменяем известной частью суммы.
    return series.sum() if series.notna().all() else np.nan


def build_features(long, roster):
    aggregation = long.groupby("sale_key").agg(
        income_pre=("income_pre", complete_sum), deals_pre=("deals_pre", complete_sum),
        clients_pre=("clients_pre", complete_sum), rows_pre=("rows_pre", "sum"),
        months_pre=("month", "nunique"))
    people = roster.merge(aggregation, on="sale_key", how="left", validate="one_to_one")
    people["нет истории PRE"] = people["rows_pre"].isna()
    numeric = {"income_pre": "Доход за пре-период", "deals_pre": "Количество сделок за пре-период",
               "clients_pre": "Клиентский счётчик за пре-период"}
    if INCLUDE_ACTIVITY_FEATURES:
        numeric.update({"rows_pre": "Количество строк с учётом долей", "months_pre": "Месяцев с данными"})
    if ABSENT_PRE_IS_ZERO:
        people.loc[people["нет истории PRE"], list(numeric)] = 0.0
    pre = long[long["sale_key"].isin(people["sale_key"])].copy()
    counts = pre.groupby("sale_key")["deals_pre"]
    valid_weights = counts.transform("count").eq(counts.transform("size")) & counts.transform("sum").gt(0)
    pre["weight"] = np.where(valid_weights, pre["deals_pre"], pre["rows_pre"])
    profiles = {}
    for field in CATEGORY_FIELDS:
        matrix = pre.groupby(["sale_key", field])["weight"].sum().unstack(fill_value=0)
        matrix = matrix.reindex(people["sale_key"], fill_value=0).fillna(0.0)
        total = matrix.sum(axis=1)
        matrix = matrix.div(total.replace(0, np.nan), axis=0)
        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        matrix.index = people.index
        dominant = matrix.fillna(0).idxmax(axis=1) if len(matrix.columns) else pd.Series(index=people.index, dtype=object)
        people[field + " | доминирует"] = dominant.where(total.to_numpy() > 0, NO_HISTORY).fillna(NO_HISTORY)
        profiles[field] = matrix
    bins = []
    if DEALS_QUANTILE_Q is not None:
        values = people["deals_pre"].dropna()
        label = f"Сделки | q={DEALS_QUANTILE_Q}"
        people[label] = MISSING
        if len(values) and values.nunique() > 1:
            codes, edges = pd.qcut(values, DEALS_QUANTILE_Q, labels=False, retbins=True, duplicates="drop")
            people.loc[values.index, label] = codes.map(lambda x: f"Q{int(x) + 1}")
            bins = [{"страта": f"Q{j + 1}", "нижняя граница": edges[j], "верхняя граница": edges[j + 1]}
                    for j in range(len(edges) - 1)]
        elif len(values):
            people.loc[values.index, label] = "все известные значения одинаковы"
    return people, numeric, profiles, pd.DataFrame(bins)


def simulated_counts(codes, levels, n_control, rng, batch):
    selected = np.argsort(rng.random((batch, len(codes))), axis=1)[:, :n_control]
    counts = np.zeros((batch, levels), dtype=float)
    np.add.at(counts, (np.arange(batch)[:, None], codes[selected]), 1)
    return counts


def ks_check(c, t, rng):
    c, t = np.asarray(c, float), np.asarray(t, float)
    c, t = c[np.isfinite(c)], t[np.isfinite(t)]
    if not len(c) or not len(t):
        return {"D": np.nan, "p": np.nan, "p SciPy": np.nan, "метод": "нет значений в одной из групп"}
    with warnings.catch_warnings():
        # SciPy сам возвращает asymp-результат, если exact не сошёлся.
        warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful.*", category=RuntimeWarning)
        result = ks_2samp(c, t, method="auto")
    values, codes, totals = np.unique(np.r_[c, t], return_inverse=True, return_counts=True)
    if len(values) == len(c) + len(t):
        return {"D": float(result.statistic), "p": float(result.pvalue), "p SciPy": float(result.pvalue), "метод": "KS SciPy auto"}
    if result.statistic == 0:
        p = 1.0
    else:
        extreme = 0
        for start in range(0, PERMUTATIONS, 250):
            a = simulated_counts(codes, len(values), len(c), rng, min(250, PERMUTATIONS - start))
            d = np.max(np.abs(np.cumsum(a, axis=1) / len(c) - np.cumsum(totals - a, axis=1) / len(t)), axis=1)
            extreme += int(np.count_nonzero(d >= result.statistic - 1e-12))
        p = (extreme + 1) / (PERMUTATIONS + 1)
    return {"D": float(result.statistic), "p": p, "p SciPy": float(result.pvalue), "метод": "KS, перестановки (есть повторы)"}


def chi_check(c, t, rng):
    c, t = np.asarray(c, str), np.asarray(t, str)
    if not len(c) or not len(t):
        return {"chi2": np.nan, "p": np.nan, "p Pearson": np.nan, "min expected": np.nan, "метод": "нет наблюдений"}, pd.DataFrame()
    levels, codes, totals = np.unique(np.r_[c, t], return_inverse=True, return_counts=True)
    obs_c = np.bincount(codes[:len(c)], minlength=len(levels))
    observed = np.array([obs_c, totals - obs_c], dtype=float)
    detail = pd.DataFrame({"категория": levels, "n control": observed[0].astype(int), "n treatment": observed[1].astype(int),
                           "доля control": observed[0] / len(c), "доля treatment": observed[1] / len(t)})
    if len(levels) == 1:
        return {"chi2": 0.0, "p": 1.0, "p Pearson": np.nan, "min expected": np.nan, "метод": "нет вариации"}, detail
    result = chi2_contingency(observed, correction=False)
    p, method = float(result.pvalue), "Pearson chi-square"
    if np.min(result.expected_freq) < 5:
        extreme = 0
        for start in range(0, PERMUTATIONS, 250):
            a = simulated_counts(codes, len(levels), len(c), rng, min(250, PERMUTATIONS - start))
            stat = np.sum((a - result.expected_freq[0]) ** 2 / result.expected_freq[0]
                          + (totals - a - result.expected_freq[1]) ** 2 / result.expected_freq[1], axis=1)
            extreme += int(np.count_nonzero(stat >= result.statistic - 1e-12))
        p, method = (extreme + 1) / (PERMUTATIONS + 1), "Pearson chi-square, перестановки (редкие ячейки)"
    return {"chi2": float(result.statistic), "p": p, "p Pearson": float(result.pvalue),
            "min expected": float(np.min(result.expected_freq)), "метод": method}, detail


def numeric_check(c, t, rng):
    c, t = np.asarray(c, float), np.asarray(t, float)
    known_c, known_t = c[np.isfinite(c)], t[np.isfinite(t)]
    ks_all, ks_nonzero = ks_check(known_c, known_t, rng), ks_check(known_c[known_c != 0], known_t[known_t != 0], rng)
    na, _ = chi_check(np.where(np.isfinite(c), "known", "NA"), np.where(np.isfinite(t), "known", "NA"), rng)
    zero, _ = chi_check(np.where(known_c == 0, "zero", "nonzero"), np.where(known_t == 0, "zero", "nonzero"), rng)
    mean_c, mean_t = (np.mean(x) if len(x) else np.nan for x in (known_c, known_t))
    sd = np.sqrt((np.var(known_c, ddof=1) + np.var(known_t, ddof=1)) / 2) if min(len(known_c), len(known_t)) > 1 else np.nan
    delta = mean_t - mean_c
    smd = delta / sd if sd > 0 else (0.0 if sd == 0 and delta == 0 else np.sign(delta) * np.inf if sd == 0 else np.nan)
    out = {"N control": len(c), "N treatment": len(t), "NA control": len(c) - len(known_c), "NA treatment": len(t) - len(known_t),
           "Zero control": int((known_c == 0).sum()), "Zero treatment": int((known_t == 0).sum()),
           "Nonzero control": int((known_c != 0).sum()), "Nonzero treatment": int((known_t != 0).sum()),
           "Mean control": mean_c, "Mean treatment": mean_t, "SMD T-C": smd}
    for name, check in (("KS all", ks_all), ("KS nonzero", ks_nonzero), ("chi2 NA", na), ("chi2 zero", zero)):
        out.update({f"{name} | {key}": value for key, value in check.items()})
    return out


def holm(values):
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate((len(order) - np.arange(len(order))) * values[order]))
    return result


def run_analysis(dgr, ab):
    validate_settings()
    roster = prepare_ab(ab)
    if set(roster["group"]) != set(GROUPS):
        raise ValueError("Нужны обе группы control/treatment")
    long, audit, dates = prepare_pre_deals(dgr)
    people, labels, profiles, bins = build_features(long, roster)
    mask = people["group"].eq("control")
    rng = np.random.default_rng(RANDOM_SEED)
    numeric = pd.DataFrame([{"column": column, "признак": label,
                            **numeric_check(people.loc[mask, column], people.loc[~mask, column], rng)}
                           for column, label in labels.items()])
    categories, details, shares = [], [], []
    fields = list(CATEGORY_FIELDS)
    category_columns = {f: f + " | доминирует" for f in fields}
    if DEALS_QUANTILE_Q is not None:
        field = f"Сделки | q={DEALS_QUANTILE_Q}"
        fields.append(field)
        category_columns[field] = field
    for field in fields:
        column = category_columns[field]
        check, detail = chi_check(people.loc[mask, column], people.loc[~mask, column], rng)
        categories.append({"поле": field, **check})
        detail.insert(0, "поле", field)
        details.append(detail)
    if CHECK_PORTFOLIO_SHARES:
        for field, matrix in profiles.items():
            for level in matrix.columns:
                shares.append({"поле": field, "категория": level,
                               **numeric_check(matrix.loc[mask, level], matrix.loc[~mask, level], rng)})
    tables = {"numeric": numeric, "categorical": pd.DataFrame(categories), "portfolio": pd.DataFrame(shares),
              "category_levels": pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
              "roster": people, "audit": audit, "dates": dates, "bins": bins}
    # Одна семья всех рассчитанных проверок; необработанные p также сохранены.
    positions, pvalues = [], []
    for table_name in ("numeric", "categorical", "portfolio"):
        table = tables[table_name]
        for column in [c for c in table.columns if c == "p" or c.endswith(" | p")]:
            for index, value in table[column].items():
                positions.append((table_name, index, column))
                pvalues.append(value)
    for (name, index, column), corrected in zip(positions, holm(pvalues)):
        tables[name].loc[index, column + " Holm"] = corrected
    tables["parameters"] = pd.DataFrame([
        ("Дата оценки, не включительно", AS_OF), ("PRE_MONTHS", ", ".join(PRE_MONTHS)),
        ("Состав", "весь переданный AB-список: одна строка на сейлза"),
        ("Нет истории PRE", "нули" if ABSENT_PRE_IS_ZERO else "NA, участник остаётся в составе"),
        ("Частичный NA в метрике", "вся сумма этой метрики у сейлза = NA"),
        ("Клиенты", "сумма распределённых клиентских счётчиков, не уникальные клиенты"),
        ("Категория сейлза", "максимальная доля PRE-сделок; равенство разрешается алфавитом"),
        ("Портфель", "доли PRE-сделок; если счётчик NA либо сумма=0, веса — доли строк"),
        ("KS", "полная выборка и ненулевые; повторы => перестановочный p"),
        ("Chi-square", "Pearson, correction=False; min expected < 5 => перестановочный p"),
        ("Holm", "одна семья всех конечных p: numeric + categorical + portfolio"),
        ("Перестановок", PERMUTATIONS), ("Seed", RANDOM_SEED), ("Alpha", ALPHA),
        ("Интерпретация", "p > alpha не доказывает гомогенность; дополнительно смотрите SMD и графики"),
    ], columns=["настройка", "значение"])
    tables["profiles"] = profiles
    return tables


def fmt_p(value):
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2e}" if value < 1e-4 else f"{value:.4f}"


def style_axis(ax):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(axis="both", colors="white", labelsize=11)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.get_offset_text().set_color("white")
    ax.yaxis.get_offset_text().set_color("white")
    ax.title.set_color("white")
    ax.grid(True, color="white", alpha=0.20, linewidth=0.7)
    ax.set_axisbelow(True)


def style_legend(ax):
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_alpha(0)
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_edgecolor("white")
        for text in legend.get_texts():
            text.set_color("white")
        legend.get_title().set_color("white")


def plot_status(ax, c, t, title, row):
    for j, (group, values) in enumerate(zip(GROUPS, (c, t))):
        values = np.asarray(values, float)
        known = values[np.isfinite(values)]
        fractions = [np.mean(~np.isfinite(values)), np.mean(known == 0) if len(known) else np.nan]
        ax.bar(np.arange(2) + (j - 0.5) * 0.34, fractions, width=0.34,
               label=f"{group} (n={len(values)})", color=COLORS[group])
    ax.set_xticks([0, 1], ["NA / все сейлзы", "Нули / известные значения"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Доля сейлзов", fontsize=12)
    ax.set_title(f"{title}: пропуски и нули\nχ² NA p={fmt_p(row['chi2 NA | p'])} | χ² Zero p={fmt_p(row['chi2 zero | p'])}", fontsize=13)
    ax.legend(title="Группа")
    style_axis(ax)
    style_legend(ax)


def plot_density(ax, c, t, title, row):
    groups = {g: np.asarray(x, float) for g, x in zip(GROUPS, (c, t))}
    groups = {g: x[np.isfinite(x) & (x != 0)] for g, x in groups.items()}
    combined = np.concatenate(list(groups.values()))
    scale = float(np.median(np.abs(combined))) if len(combined) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    transformed = {g: np.arcsinh(x / scale) for g, x in groups.items()}
    z_all = np.arcsinh(combined / scale)
    bins = np.histogram_bin_edges(z_all, bins=40) if len(z_all) else np.array([-0.5, 0.5])
    span = float(np.ptp(z_all)) if len(z_all) else 1.0
    grid = np.linspace(float(z_all.min()) - max(0.1 * span, 0.15),
                       float(z_all.max()) + max(0.1 * span, 0.15), 400) if len(z_all) else np.linspace(-1, 1, 400)
    for group, z in transformed.items():
        label = f"{group} (nonzero n={len(z)})"
        if not len(z):
            ax.plot([], [], color=COLORS[group], label=label)
        elif DENSITY_KIND == "hist":
            ax.hist(z, bins=bins, density=True, histtype="step", linewidth=1.8, color=COLORS[group], label=label)
        elif len(z) < 2 or np.unique(z).size < 2:
            ax.axvline(z[0], color=COLORS[group], linewidth=2, label=label + ": константа")
        else:
            try:
                ax.plot(grid, gaussian_kde(z)(grid), color=COLORS[group], linewidth=2, label=label)
            except (ValueError, np.linalg.LinAlgError):
                ax.hist(z, bins=bins, density=True, histtype="step", linewidth=1.8, color=COLORS[group], label=label)
    ax.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.35)
    ax.set_title(f"{title}: ненулевые значения\nKS nonzero p={fmt_p(row['KS nonzero | p'])} | KS all p={fmt_p(row['KS all | p'])}", fontsize=13)
    ax.set_xlabel(f"asinh(x / {scale:.3g})", fontsize=12)
    ax.set_ylabel("Плотность", fontsize=12)
    ax.legend(title="Группа")
    if not len(combined):
        ax.text(0.5, 0.5, "Нет известных ненулевых значений", transform=ax.transAxes,
                ha="center", va="center", color="white")
    style_axis(ax)
    style_legend(ax)


def export_figure(fig, path, dpi, show):
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", facecolor="none", edgecolor="none")
    if show:
        plt.show()
    plt.close(fig)


def plot_numeric(tables, out_dir, dpi, show):
    table, people = tables["numeric"], tables["roster"]
    paths = []
    for start in range(0, len(table), 2):
        fig, axes = plt.subplots(2, 2, figsize=(21, 12))
        fig.patch.set_alpha(0)
        fig.suptitle(f"Гомогенность control / treatment на {pd.Timestamp(AS_OF):%d.%m.%Y}\nPRE: {', '.join(PRE_MONTHS)}",
                     fontsize=19, color="white")
        for j in range(2):
            if start + j >= len(table):
                axes[0, j].set_visible(False)
                axes[1, j].set_visible(False)
                continue
            row = table.iloc[start + j]
            c = people.loc[people["group"].eq("control"), row["column"]]
            t = people.loc[people["group"].eq("treatment"), row["column"]]
            plot_status(axes[0, j], c, t, row["признак"], row)
            plot_density(axes[1, j], c, t, row["признак"], row)
        fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=3.0, w_pad=2.0)
        path = out_dir / f"numeric_{start // 2 + 1:02d}_2x2.png"
        export_figure(fig, path, dpi, show)
        paths.append(path)
    return paths


def plot_categories(tables, out_dir, dpi, show):
    paths = []
    for index, row in tables["categorical"].iterrows():
        field = row["поле"]
        detail = tables["category_levels"].query("поле == @field").copy()
        profile = tables["portfolio"]
        profile = profile[profile["поле"].eq(field)].copy() if len(profile) else pd.DataFrame()
        levels = set(detail["категория"])
        if len(profile):
            levels |= set(profile["категория"])
        levels = sorted(levels)
        height = max(8.0, 0.36 * len(levels) + 3.5)
        fig, axes = plt.subplots(1, 2 if len(profile) else 1, figsize=(21, height), squeeze=False)
        fig.patch.set_alpha(0)
        fig.suptitle(f"{field}: control / treatment на {pd.Timestamp(AS_OF):%d.%m.%Y}\nPRE: {', '.join(PRE_MONTHS)}",
                     fontsize=19, color="white")
        indexed = detail.set_index("категория").reindex(levels).fillna(0)
        y = np.arange(len(levels))
        ax = axes[0, 0]
        for j, group in enumerate(GROUPS):
            ax.barh(y + (j - 0.5) * 0.36, indexed[f"доля {group}"], height=0.36, color=COLORS[group], label=group)
        ax.set_yticks(y, levels)
        ax.invert_yaxis()
        ax.set_xlabel("Доля сейлзов")
        ax.set_title(f"Доминирующая категория / страта\nχ² p={fmt_p(row['p'])} | Holm p={fmt_p(row['p Holm'])}", fontsize=13)
        ax.legend(title="Группа")
        style_axis(ax)
        style_legend(ax)
        if len(profile):
            ax = axes[0, 1]
            p = profile.set_index("категория").reindex(levels)
            for j, group in enumerate(GROUPS):
                ax.barh(y + (j - 0.5) * 0.36, p[f"Mean {group}"], height=0.36, color=COLORS[group], label=group)
            tick_labels = [f"{level} | KS p={fmt_p(p.loc[level, 'KS all | p'])}" for level in levels]
            ax.set_yticks(y, tick_labels)
            ax.invert_yaxis()
            ax.set_xlabel("Средняя доля в портфеле сейлза с PRE-историей")
            ax.set_title("Портфель по долям сделок\nKS сравнивает распределения долей по сейлзам", fontsize=13)
            ax.legend(title="Группа")
            style_axis(ax)
            style_legend(ax)
        fig.tight_layout(rect=[0, 0, 1, 0.9], w_pad=3.0)
        path = out_dir / f"category_{index + 1:02d}.png"
        export_figure(fig, path, dpi, show)
        paths.append(path)
    return paths


def save_report(tables, out_dir=OUT_DIR, dpi=DPI, show=SHOW):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, value in tables.items():
        if isinstance(value, pd.DataFrame):
            path = out_dir / f"{key}.csv"
            value.to_csv(path, index=False, encoding="utf-8-sig")
            paths.append(path)
    paths += plot_numeric(tables, out_dir, dpi, show)
    paths += plot_categories(tables, out_dir, dpi, show)
    return paths


def main(dgr=None, ab=None):
    if dgr is None:
        dgr = pd.read_excel(DATA_FILE)
    if ab is None:
        ab = pd.read_excel(AB_FILE, sheet_name=AB_SHEET)
    tables = run_analysis(dgr, ab)
    summary = ["признак", "N control", "N treatment", "NA control", "NA treatment", "SMD T-C",
               "KS all | p", "KS nonzero | p", "chi2 zero | p", "chi2 NA | p"]
    print(tables["numeric"][summary].to_string(index=False))
    print("\nКатегории:")
    print(tables["categorical"].to_string(index=False))
    paths = save_report(tables)
    print(f"\nСохранено файлов: {len(paths)}. Папка: {OUT_DIR.resolve()}")
    print("p > 0.05 не доказывает гомогенность. Смотрите также SMD и графики.")
    return tables


if __name__ == "__main__":
    tables = main()
