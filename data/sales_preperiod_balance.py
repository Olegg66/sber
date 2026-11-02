# -*- coding: utf-8 -*-
"""Баланс control/treatment по пре-периоду после сопоставления списков.

Зависимости: pandas, numpy, scipy, openpyxl.
Одна строка анализа — один сейлз. ФИО и атрибуция FICC/Flow такие же,
как в основном коде AB. Нулевой и отрицательный доход сохраняются.
Фильтр Y>0 и отсечения выбросов здесь не применяются.

Сравниваются исходный AB, AB ∩ распределённые сделки за любые месяцы,
и (если все месяцы доступны) AB ∩ сделки в PRESENCE_MONTHS.
Опционально можно передать актуальный кадровый список current_staff.
Наличие в сделках — отдельный признак; это не подтверждение трудоустройства.

Отсутствие всех строк в пре-периоде отмечается как отсутствие истории.
Такие числовые признаки остаются NaN и не превращаются в доход 0.
Если история есть хотя бы в одном пре-месяце, отсутствие строк в другом
пре-месяце считается нулевой активностью при полной выгрузке этого месяца.

SMD, разница средних/долей и распределения — основные диагностики.
p-value — дополнительные; p>0.05 не означает доказанную однородность.
Holm применяется отдельно внутри каждого состава и семейства тестов.
Клиентские счётчики по строкам не являются числом уникальных клиентов.

Jupyter: tables = run_analysis(dgr, ab); display(tables['numeric']).
Самостоятельный запуск: python sales_preperiod_balance.py
"""
from datetime import date, datetime
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ks_2samp, ttest_ind

DATA_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\data\441774c5-7986-43a6-8269-5d5ce8627e9b.xlsx")
AB_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\Телефоны УГР 29.06.2026.xlsx")
AB_SHEET = "Штатная расстановка"
AB_NAME = "Ф.И.О. сотрудника"
AB_GROUP = "AB-тест"
PRE_MONTHS = ("2026-05", "2026-06")
PRESENCE_MONTHS = ("2026-09", "2026-10")
CURRENT_STAFF_FILE = None  # Опционально: путь к действительно актуальному кадровому списку.
CURRENT_STAFF_SHEET = 0
STAFF_NAME_COL = "Ф.И.О. сотрудника"
OUT_DIR = Path("ab sales/balance")
CATEGORY_FIELDS = ("ТБ", "Сегмент", "Дэск", "ДКК", "Отрасль", "Продукт")
ROLE_FIELD = "тип сейлза"
DEALS_COL = "Кол-во сделок"
CLIENTS_COL = "Кол-во клиентов"
SMD_ATTENTION = 0.10  # Ориентир внимания, не статистический порог доказанной однородности.
PERMUTATIONS = 4999
RANDOM_SEED = 20261102

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


def prepare_deals(dgr):
    dgr = dgr.copy()
    dgr.columns = [clean_text(c) for c in dgr.columns]
    if not dgr.columns.is_unique:
        raise ValueError("Дублирующиеся заголовки после очистки пробелов")
    require_columns(dgr, list(CATEGORY_FIELDS) + [FLOW_COL, FICC_COL, "Доход", DEALS_COL,
                                                CLIENTS_COL, "Месяц-Год"], "dgr")
    records, skipped, date_rows = [], [], []
    for excel_row, (_, row) in enumerate(dgr.iterrows(), start=2):
        product = clean_text(row["Продукт"])
        month = parse_month(row["Месяц-Год"])
        date_rows.append({"исходная дата": repr(row["Месяц-Год"]),
                          "месяц": str(month.to_period("M")) if pd.notna(month) else "НЕ РАСПОЗНАНО"})
        if product in excluded_products:
            reason = "исключённый продукт"
        elif product not in flow_products | ficc_products:
            reason = "неизвестный продукт"
        elif pd.isna(month):
            reason = "не распознана дата"
        else:
            role = "Flow" if product in flow_products else "FICC"
            names = split_names(row[FLOW_COL if role == "Flow" else FICC_COL])
            reason = "нет ФИО в выбранной колонке" if not names else ""
        if reason:
            skipped.append({"строка Excel": excel_row, "причина": reason,
                            "Продукт": product, "Месяц-Год": clean_text(row["Месяц-Год"]),
                            "Доход": pd.to_numeric(row["Доход"], errors="coerce")})
            continue
        values = {}
        for field in ["Доход", DEALS_COL, CLIENTS_COL]:
            value = pd.to_numeric(row[field], errors="coerce")
            if pd.isna(value) or not np.isfinite(value) or (field != "Доход" and value < 0):
                raise ValueError(f"Строка Excel {excel_row}: некорректное {field}={row[field]!r}")
            values[field] = float(value) / len(names)
        for key, name in names:
            records.append({"sale_key": key, "sale_name": name, "month_p": month.to_period("M"),
                            "income": values["Доход"], "deals": values[DEALS_COL],
                            "clients_entries": values[CLIENTS_COL], ROLE_FIELD: role,
                            "row_share": 1 / len(names), "строка Excel": excel_row,
                            **{f: clean_text(row[f]).casefold() or "__ПРОПУСК__" for f in CATEGORY_FIELDS}})
    if not records:
        raise ValueError("Нет распределённых сделок. Проверьте даты, продукты и ФИО")
    long = pd.DataFrame(records)
    dates = pd.DataFrame(date_rows).value_counts().rename("число строк").reset_index()
    audit = pd.DataFrame(skipped, columns=["строка Excel", "причина", "Продукт", "Месяц-Год", "Доход"])
    missing_pre = {pd.Period(m, freq="M") for m in PRE_MONTHS} - set(long["month_p"])
    if missing_pre:
        raise ValueError(f"В распределённых сделках нет пре-месяцев {sorted(str(x) for x in missing_pre)}. "
                         f"Распознанные даты: {dates.to_dict('records')}. "
                         f"Причины пропусков: {audit['причина'].value_counts().to_dict()}")
    return long, audit, dates


def build_features(long, ab, current_staff=None):
    roster = prepare_ab(ab)
    roster["найден в сделках"] = roster["sale_key"].isin(set(long["sale_key"]))
    presence = long[long["month_p"].isin({pd.Period(m, freq="M") for m in PRESENCE_MONTHS})]
    roster["есть строки в окне присутствия"] = roster["sale_key"].isin(set(presence["sale_key"]))
    if current_staff is not None:
        require_columns(current_staff, [STAFF_NAME_COL], "актуальном кадровом списке")
        staff_keys = set(current_staff[STAFF_NAME_COL].map(name_key)) - {""}
        roster["есть в актуальном штате"] = roster["sale_key"].isin(staff_keys)
    pre = long[long["month_p"].isin({pd.Period(m, freq="M") for m in PRE_MONTHS})].copy()
    aggregation = pre.groupby("sale_key").agg(
        income_pre=("income", "sum"), deals_pre=("deals", "sum"),
        clients_entries_pre=("clients_entries", "sum"), months_pre=("month_p", "nunique"))
    managers = roster.merge(aggregation, on="sale_key", how="left", validate="one_to_one")
    managers["есть история до теста"] = managers["months_pre"].notna().astype(float)
    labels = {"income_pre": "Доход за пре-период", "deals_pre": "Кол-во сделок за пре-период",
              "clients_entries_pre": "Сумма клиентских счётчиков", "months_pre": "Месяцев со строками",
              "есть история до теста": "Есть история до теста"}
    for month in PRE_MONTHS:
        column = f"income_{month}"
        income = pre[pre["month_p"].eq(pd.Period(month, freq="M"))].groupby("sale_key")["income"].sum()
        managers[column] = managers["sale_key"].map(income)
        has_history = managers["есть история до теста"].eq(1)
        managers.loc[has_history, column] = managers.loc[has_history, column].fillna(0.0)
        labels[column] = f"Доход {month}"
    pre["weight"] = pre["deals"]
    zero_weights = pre.groupby("sale_key")["deals"].transform("sum").eq(0)
    pre.loc[zero_weights, "weight"] = pre.loc[zero_weights, "row_share"]
    profiles, dominant = {}, {}
    for field in CATEGORY_FIELDS + (ROLE_FIELD,):
        matrix = pre.groupby(["sale_key", field])["weight"].sum().unstack(fill_value=0)
        matrix = matrix.reindex(managers["sale_key"], fill_value=0).fillna(0.0)
        total = matrix.sum(axis=1)
        matrix = matrix.div(total.replace(0, np.nan), axis=0).fillna(0.0)
        matrix["__НЕТ_ИСТОРИИ__"] = total.eq(0).astype(float)
        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        matrix.index = managers.index
        profiles[field] = matrix
        dominant[field] = matrix.idxmax(axis=1)
        managers[field + " | доминирует"] = dominant[field]
    return managers, labels, profiles, dominant


def holm(values):
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        corrected = np.maximum.accumulate((len(order) - np.arange(len(order))) * values[order])
        result[order] = np.minimum(corrected, 1.0)
    return result


def standardized(delta, denominator):
    if not np.isfinite(denominator):
        return np.nan
    if denominator == 0:
        return 0.0 if delta == 0 else float(np.sign(delta) * np.inf)
    return float(delta / denominator)


def numeric_comparison(T, C, reference_sd=np.nan):
    T, C = np.asarray(T, float), np.asarray(C, float)
    missing_t, missing_c = int(np.isnan(T).sum()), int(np.isnan(C).sum())
    T, C = T[np.isfinite(T)], C[np.isfinite(C)]
    if len(T) == 0 or len(C) == 0:
        return {"nT": len(T), "nC": len(C), "нет данных T": missing_t,
                "нет данных C": missing_c, "SMD": np.nan, "p Welch": np.nan, "p KS": np.nan}
    mean_t, mean_c = float(T.mean()), float(C.mean())
    vt, vc = (float(np.var(T, ddof=1)) if len(T) > 1 else np.nan,
              float(np.var(C, ddof=1)) if len(C) > 1 else np.nan)
    sd = np.sqrt((vt + vc) / 2)
    smd = standardized(mean_t - mean_c, sd)
    ks = ks_2samp(T, C, method="auto")
    # Постоянные группы не создают искусственное p=0 или отрицание различий.
    p_welch = float(ttest_ind(T, C, equal_var=False).pvalue) if len(T) > 1 and len(C) > 1 and vt + vc > 0 else np.nan
    return {"nT": len(T), "nC": len(C), "нет данных T": missing_t, "нет данных C": missing_c,
            "среднее T": mean_t, "среднее C": mean_c, "разница T-C": mean_t - mean_c,
            "медиана T": float(np.median(T)), "медиана C": float(np.median(C)),
            "SD T": np.sqrt(vt), "SD C": np.sqrt(vc),
            "доля нулей T": float(np.mean(T == 0)), "доля нулей C": float(np.mean(C == 0)),
            "SMD": smd, "abs SMD": abs(smd),
            "SMD в масштабе исходного AB": standardized(mean_t - mean_c, reference_sd),
            "дисперсия T/C": vt / vc if vc > 0 else (1.0 if vt == vc == 0 else np.inf),
            "KS D": float(ks.statistic), "p Welch": p_welch, "p KS": float(ks.pvalue),
            "ориентир SMD": "нет оценки" if np.isnan(smd) else
            ("обратить внимание" if abs(smd) >= SMD_ATTENTION else "ниже ориентира")}


def category_comparison(values, treatment, seed):
    categories, codes = np.unique(np.asarray(values, str), return_inverse=True)
    nt, nc, n = int(treatment.sum()), int((~treatment).sum()), len(values)
    counts = np.bincount(codes, minlength=len(categories)).astype(float)
    t_counts = np.bincount(codes[treatment], minlength=len(categories)).astype(float)
    c_counts = counts - t_counts
    pt, pc = t_counts / nt, c_counts / nc
    denominator = np.sqrt((pt * (1 - pt) + pc * (1 - pc)) / 2)
    smd = [standardized(a - b, d) for a, b, d in zip(pt, pc, denominator)]
    et, ec = counts * nt / n, counts * nc / n
    def statistics(t):
        return np.sum((t - et) ** 2 / et + (counts - t - ec) ** 2 / ec, axis=-1)
    observed = float(statistics(t_counts))
    if len(categories) == 1:
        p = 1.0
    else:
        rng = np.random.default_rng(seed)
        extreme = 0
        onehot = np.eye(len(categories), dtype=float)[codes]
        # Перестановка меток сейлзов, а не отдельных сделок/клиентов.
        for start in range(0, PERMUTATIONS, 250):
            b = min(250, PERMUTATIONS - start)
            selected = np.argsort(rng.random((b, n)), axis=1)[:, :nt]
            shuffled_t = onehot[selected].sum(axis=1)
            extreme += int(np.sum(statistics(shuffled_t) >= observed - 1e-12))
        p = (extreme + 1) / (PERMUTATIONS + 1)
    detail = pd.DataFrame({"категория": categories, "nT": t_counts.astype(int), "nC": c_counts.astype(int),
                           "доля T": pt, "доля C": pc, "разница долей, п.п.": 100 * (pt - pc), "SMD": smd})
    return {"число категорий": len(categories), "TV расстояние": float(np.abs(pt - pc).sum() / 2),
            "Cramers V": np.sqrt(observed / n), "максимальный abs SMD": float(np.max(np.abs(smd))),
            "p перестановочный": p}, detail


def balance_tables(frame, labels, profiles, dominant, reference_sds, title):
    treatment = frame["group"].eq("treatment")
    if min(int(treatment.sum()), int((~treatment).sum())) < 2:
        return None
    numeric, categories, levels, share_rows = [], [], [], []
    for column, label in labels.items():
        numeric.append({"состав": title, "признак": label,
                        **numeric_comparison(frame.loc[treatment, column], frame.loc[~treatment, column], reference_sds[column])})
    numeric = pd.DataFrame(numeric)
    numeric["p Welch Holm"] = holm(numeric["p Welch"])
    numeric["p KS Holm"] = holm(numeric["p KS"])
    for j, field in enumerate(dominant):
        values = dominant[field].loc[frame.index].to_numpy()
        result, detail = category_comparison(values, treatment.to_numpy(), RANDOM_SEED + j)
        categories.append({"состав": title, "поле": field, **result})
        detail.insert(0, "поле", field)
        detail.insert(0, "состав", title)
        levels.append(detail)
        for category in profiles[field].columns:
            values = profiles[field].loc[frame.index, category]
            # Доли портфеля сравниваются по сейлзам, не как независимые сделки.
            share_rows.append({"состав": title, "поле": field, "категория": category,
                               **numeric_comparison(values[treatment], values[~treatment])})
    categories = pd.DataFrame(categories)
    categories["p перестановочный Holm"] = holm(categories["p перестановочный"])
    shares = pd.DataFrame(share_rows)
    shares["p Welch Holm"] = holm(shares["p Welch"])
    return numeric, categories, pd.concat(levels, ignore_index=True), shares


def run_analysis(dgr, ab, current_staff=None):
    if not PRE_MONTHS or PERMUTATIONS < 99:
        raise ValueError("Проверьте PRE_MONTHS и PERMUTATIONS (минимум 99)")
    long, audit, dates = prepare_deals(dgr)
    managers, labels, profiles, dominant = build_features(long, ab, current_staff)
    if set(managers["group"]) != {"treatment", "control"}:
        raise ValueError("Для проверки нужны обе группы")
    reference_sds = {}
    for column in labels:
        v = [managers.loc[managers["group"].eq(g), column].var(ddof=1) for g in ["treatment", "control"]]
        reference_sds[column] = np.sqrt(sum(v) / 2)
    masks = {"Исходный AB": pd.Series(True, index=managers.index),
             "AB ∩ сделки, все месяцы": managers["найден в сделках"]}
    present_months = set(long["month_p"])
    missing_presence = {pd.Period(m, freq="M") for m in PRESENCE_MONTHS} - present_months
    if PRESENCE_MONTHS and not missing_presence:
        masks["AB ∩ сделки, окно присутствия"] = managers["есть строки в окне присутствия"]
    if current_staff is not None:
        masks["AB ∩ актуальный штат"] = managers["есть в актуальном штате"]
    counts, numeric, categorical, levels, shares = [], [], [], [], []
    for title, mask in masks.items():
        selected = managers[mask]
        for group in ["treatment", "control"]:
            n0 = int(managers["group"].eq(group).sum())
            n = int(selected["group"].eq(group).sum())
            counts.append({"состав": title, "группа": group, "исходно": n0, "осталось": n,
                           "не вошло": n0 - n, "осталось, %": 100 * n / n0,
                           "с историей пре-периода": int(selected.loc[selected["group"].eq(group), "есть история до теста"].sum())})
        result = balance_tables(selected, labels, profiles, dominant, reference_sds, title)
        if result:
            for target, value in zip([numeric, categorical, levels, shares], result):
                target.append(value)
    comparison = []
    for title, mask in masks.items():
        if title == "Исходный AB":
            continue
        table = [[int((managers["group"].eq(g) & mask).sum()),
                  int((managers["group"].eq(g) & ~mask).sum())] for g in ["treatment", "control"]]
        rates = [row[0] / sum(row) for row in table]
        comparison.append({"состав": title, "вошло T": table[0][0], "не вошло T": table[0][1],
                           "вошло C": table[1][0], "не вошло C": table[1][1],
                           "разница сохранения T-C, п.п.": 100 * (rates[0] - rates[1]),
                           "p Fisher, сохранение состава": float(fisher_exact(table).pvalue)})
    if not numeric:
        raise ValueError("Ни в одном составе нет минимум двух сейлзов в каждой группе")
    numeric = pd.concat(numeric, ignore_index=True)
    baseline = numeric[numeric["состав"].eq("Исходный AB")][["признак", "abs SMD", "SMD в масштабе исходного AB"]]
    changes = numeric.merge(baseline, on="признак", how="left", suffixes=("", " исходный"))
    changes["изменение abs SMD"] = changes["abs SMD"] - changes["abs SMD исходный"]
    changes["изменение abs SMD, фиксированный масштаб"] = (changes["SMD в масштабе исходного AB"].abs()
        - changes["SMD в масштабе исходного AB исходный"].abs())
    metadata = pd.DataFrame([
        ("Пре-период", ", ".join(PRE_MONTHS)), ("Окно проверки присутствия", ", ".join(PRESENCE_MONTHS)),
        ("Недостающие месяцы присутствия", ", ".join(str(x) for x in sorted(missing_presence))),
        ("Пересечение", "AB и валидные распределённые сделки; роли определяются продуктом"),
        ("Отсутствие в сделках", "не равно увольнению; возможны отсутствие активности и ошибки сопоставления"),
        ("Фильтр дохода", "не применяется: нулевой и отрицательный доход сохранены"),
        ("Отсечения выбросов", "не применяются"),
        ("Нет истории в пре-периоде", "числовые метрики NaN; отдельный индикатор отсутствия истории"),
        ("Категории", "доминирующие категории по доле сделок до теста и средние доли портфелей по сейлзам"),
        ("Клиенты", "сумма счётчиков, не уникальные клиенты"),
        ("SMD ориентир", SMD_ATTENTION), ("Ориентир SMD", "не критерий доказанной однородности"),
        ("p-value", "диагностические; p>0.05 не подтверждает однородность"),
        ("Перестановки для категорий", PERMUTATIONS),
        ("Holm", "отдельно для Welch, KS, категориальных тестов и долей портфелей в каждом составе"),
    ], columns=["настройка", "значение"])
    return {"counts": pd.DataFrame(counts), "retention": pd.DataFrame(comparison),
            "numeric": numeric, "categorical": pd.concat(categorical, ignore_index=True),
            "category_levels": pd.concat(levels, ignore_index=True),
            "portfolio_shares": pd.concat(shares, ignore_index=True),
            "changes": changes, "roster": managers, "audit": audit, "dates": dates, "metadata": metadata}


def save_report(tables, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preperiod_balance.xlsx"
    titles = {"counts": "Численности", "retention": "Сохранение_состава", "numeric": "Баланс_числовых",
              "categorical": "Баланс_категорий", "category_levels": "Категории_подробно",
              "portfolio_shares": "Доли_портфелей", "changes": "Изменение_баланса", "roster": "Список_сейлзов",
              "audit": "Пропущенные_строки", "dates": "Диагностика_дат", "metadata": "Настройки"}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for key, title in titles.items():
            tables[key].to_excel(writer, sheet_name=title, index=False)
    return path


def show_results(tables):
    print("Численности исходных и сопоставленных групп:")
    print(tables["counts"].to_string(index=False))
    print("\nРазличия в сохранении состава:")
    print(tables["retention"].to_string(index=False))
    cols = ["состав", "признак", "nT", "nC", "среднее T", "среднее C", "SMD", "KS D", "p Welch Holm"]
    print("\nБаланс числовых признаков за пре-период:")
    frame = tables["numeric"].reindex(columns=cols)
    try:
        from IPython.display import display
        display(frame.round(4))
    except ImportError:
        print(frame.round(4).to_string(index=False))
    missing = tables["metadata"].set_index("настройка").loc["Недостающие месяцы присутствия", "значение"]
    if missing:
        print("\nПроверка состава по окну присутствия пропущена: нет месяцев", missing)


def main():
    dgr = pd.read_excel(DATA_FILE)
    ab = pd.read_excel(AB_FILE, sheet_name=AB_SHEET)
    staff = pd.read_excel(CURRENT_STAFF_FILE, sheet_name=CURRENT_STAFF_SHEET) if CURRENT_STAFF_FILE else None
    tables = run_analysis(dgr, ab, staff)
    show_results(tables)
    path = save_report(tables)
    print("\nФайл сохранён:", path.resolve())
    return tables


if __name__ == "__main__":
    main()
