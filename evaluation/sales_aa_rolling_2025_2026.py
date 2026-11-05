# -*- coding: utf-8 -*-
"""Историческая гомогенность фиксированных групп в скользящих окнах.

Один самостоятельный файл; старый модуль не требуется.
Зависимости: pandas, numpy, scipy, matplotlib, openpyxl (чтение Excel).
    import sales_aa_rolling_2025_2026 as aa
    result = aa.run_analysis(dgr, ab)
    files = aa.save_report(result)
    display(result["numeric"])

Те же группы во всех трёхмесячных окнах; повторной рандомизации нет.
До запуска — ретроспективная A/A-диагностика. Во время воздействия —
диагностика A/B. Перекрывающиеся окна зависимы; Holm не требует независимости.
Сохраняются KS, chi-square, SMD, пропуски, размеры групп и графики.
Сравнение изменений средних с опорным окном учитывает, что это те же люди.
Отбор по active, будущему доходу, увольнениям и обрезка выбросов не применяются.
"""
from datetime import date, datetime
from pathlib import Path
import re
import unicodedata
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import ks_2samp, chi2_contingency, gaussian_kde
from scipy.stats import t as t_dist

START_MONTH = "2025-01"
END_MONTH = "2026-12"
WINDOW_MONTHS = 3
STEP_MONTHS = 1
# None: до предыдущего календарного месяца включительно. Если файл заканчивается
# раньше, используем его последний месяц. Полнота выгрузки внутри месяца не известна.
LAST_COMPLETE_MONTH = None  # Например "2026-10"; можно явно задать позднее.
EXPERIMENT_START = "2026-07-01"
REFERENCE_END_MONTH = "2026-06"  # Иначе последнее полное окно до запуска, затем первое доступное.
MIN_KNOWN_PER_GROUP = 2   # При меньшем n показываются описания, p=NA.
SAVE_WINDOW_PLOTS = True
SAVE_WINDOW_CATEGORY_PLOTS = True
OUT_DIR = Path("ab sales/aa_rolling_2025_2026")
CORE_COLUMNS = ("income_pre", "deals_pre", "clients_pre")


DATA_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\data\441774c5-7986-43a6-8269-5d5ce8627e9b.xlsx")

AB_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\Телефоны УГР 29.06.2026.xlsx")

AB_SHEET = "Штатная расстановка"

AB_NAME = "Ф.И.О. сотрудника"

AB_GROUP = "AB-тест"

CATEGORY_FIELDS = ("ТБ", "Сегмент", "Дэск", "ДКК", "Отрасль", "Продукт", "тип сейлза")

DEALS_COL, CLIENTS_COL = "Кол-во сделок", "Кол-во клиентов"

ABSENT_PRE_IS_ZERO = False

INCLUDE_ACTIVITY_FEATURES = True

CHECK_PORTFOLIO_SHARES = True

DEALS_QUANTILE_Q = 5

PERMUTATIONS = 9999

RANDOM_SEED = 20260701

ALPHA = 0.05

DPI = 300

SHOW = False

DENSITY_KIND = "kde"

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

def numeric_cell(value, column, row_number):
    if pd.isna(value) or clean_text(value) in {"", "-", "—"}:
        return np.nan
    result = pd.to_numeric(value, errors="coerce")
    if pd.isna(result) or not np.isfinite(result):
        raise ValueError(f"Строка Excel {row_number}: {column}={value!r} не является числом")
    if column != "Доход" and result < 0:
        raise ValueError(f"Строка Excel {row_number}: отрицательное {column}={value!r}")
    return float(result)

def prepare_history_deals(dgr):
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
    complete_to = (pd.Period(LAST_COMPLETE_MONTH, "M") if LAST_COMPLETE_MONTH is not None
                   else pd.Timestamp.today().to_period("M") - 1)
    history_last = min(pd.Period(END_MONTH, "M"), complete_to)
    pre_set = set(pd.period_range(START_MONTH, history_last, freq="M"))
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
    if long.empty:
        raise ValueError("В заданных годах нет распределённых сделок; проверьте даты, продукты и ФИО")
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
    numeric = {"income_pre": "Доход за окно", "deals_pre": "Количество сделок за окно",
               "clients_pre": "Клиентский счётчик за окно"}
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
        fig.suptitle(f"Гомогенность control / treatment — {tables['window_label']}\n{tables['period_type']}",
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
        fig.suptitle(f"{field}: control / treatment — {tables['window_label']}\n{tables['period_type']}",
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


def validate_settings():
    first, last = pd.Period(START_MONTH, "M"), pd.Period(END_MONTH, "M")
    if first > last or not isinstance(WINDOW_MONTHS, int) or WINDOW_MONTHS < 1:
        raise ValueError("Проверьте START_MONTH, END_MONTH и WINDOW_MONTHS")
    if not isinstance(STEP_MONTHS, int) or STEP_MONTHS < 1 or MIN_KNOWN_PER_GROUP < 2:
        raise ValueError("STEP_MONTHS >= 1, MIN_KNOWN_PER_GROUP >= 2")
    if PERMUTATIONS < 99 or not 0 < ALPHA < 1 or DENSITY_KIND not in {"kde", "hist"}:
        raise ValueError("Проверьте PERMUTATIONS, ALPHA и DENSITY_KIND")
    if not set(CATEGORY_FIELDS) <= {"ТБ", "Сегмент", "Дэск", "ДКК", "Отрасль", "Продукт", "тип сейлза"}:
        raise ValueError("Неизвестное поле CATEGORY_FIELDS")
    if len(set(CATEGORY_FIELDS)) != len(CATEGORY_FIELDS):
        raise ValueError("Повторные CATEGORY_FIELDS")
    if DEALS_QUANTILE_Q is not None and (not isinstance(DEALS_QUANTILE_Q, int) or DEALS_QUANTILE_Q < 2):
        raise ValueError("DEALS_QUANTILE_Q: целое >= 2 либо None")
    if flow_products & ficc_products or (flow_products | ficc_products) & excluded_products:
        raise ValueError("Списки продуктов пересекаются")


def period_type(start, end):
    launch = pd.Timestamp(EXPERIMENT_START).to_period("M")
    if end < launch:
        return "До запуска: историческая A/A-проверка"
    if start < launch:
        return "Окно пересекает запуск A/B"
    return "После запуска: диагностика A/B"


def checked_numeric(c, t, rng):
    result = numeric_check(c, t, rng)
    known_c, known_t = (np.isfinite(np.asarray(x, float)).sum() for x in (c, t))
    checks = {"KS all": min(known_c, known_t),
              "KS nonzero": min(result["Nonzero control"], result["Nonzero treatment"]),
              "chi2 zero": min(known_c, known_t), "chi2 NA": min(len(c), len(t))}
    for prefix, n in checks.items():
        if n < MIN_KNOWN_PER_GROUP:
            for column in ("p", "p SciPy", "p Pearson"):
                if prefix + " | " + column in result:
                    result[prefix + " | " + column] = np.nan
            result[prefix + " | метод"] = f"недостаточно наблюдений: min n={n}"
    result["Known control"], result["Known treatment"] = int(known_c), int(known_t)
    return result


def primary_p_columns(table):
    return [c for c in table.columns if c == "p" or c.endswith(" | p")]


def correct_tables(tables, names, suffix):
    positions, values = [], []
    for name in names:
        for column in primary_p_columns(tables[name]):
            for index, value in tables[name][column].items():
                positions.append((name, index, column))
                values.append(value)
    for (name, index, column), adjusted in zip(positions, holm(values)):
        tables[name].loc[index, column + suffix] = adjusted


def correct_over_time(tables):
    families = {"numeric": ["column"], "categorical": ["поле"],
                "portfolio": ["поле", "категория"], "mean_gap_changes": ["column"]}
    for name, keys in families.items():
        table = tables[name]
        if table.empty:
            continue
        for column in primary_p_columns(table):
            for indices in table.groupby(keys, sort=False).groups.values():
                table.loc[indices, column + " Holm over time"] = holm(table.loc[indices, column])


def analyze_window(long, roster, start, end):
    people, labels, profiles, bins = build_features(long, roster)
    control = people["group"].eq("control")
    # Результат конкретного окна не зависит от числа предыдущих окон.
    rng = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, start.ordinal, end.ordinal]))
    numeric = pd.DataFrame([{"column": column, "признак": label,
                            **checked_numeric(people.loc[control, column], people.loc[~control, column], rng)}
                           for column, label in labels.items()])
    categories, details, shares = [], [], []
    category_columns = {f: f + " | доминирует" for f in CATEGORY_FIELDS}
    if DEALS_QUANTILE_Q is not None:
        field = f"Сделки | q={DEALS_QUANTILE_Q}"
        category_columns[field] = field
    for field, column in category_columns.items():
        check, detail = chi_check(people.loc[control, column], people.loc[~control, column], rng)
        if min(int(control.sum()), int((~control).sum())) < MIN_KNOWN_PER_GROUP:
            check["p"] = check["p Pearson"] = np.nan
            check["метод"] = "недостаточно сейлзов"
        categories.append({"поле": field, **check})
        detail.insert(0, "поле", field)
        details.append(detail)
    if CHECK_PORTFOLIO_SHARES:
        for field, matrix in profiles.items():
            for level in matrix.columns:
                shares.append({"поле": field, "категория": level,
                               **checked_numeric(matrix.loc[control, level], matrix.loc[~control, level], rng)})
    tables = {"numeric": numeric, "categorical": pd.DataFrame(categories), "portfolio": pd.DataFrame(shares),
              "category_levels": pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
              "roster": people, "bins": bins, "profiles": profiles,
              "window_label": f"{start} — {end}", "period_type": period_type(start, end)}
    correct_tables(tables, ("numeric", "categorical", "portfolio"), " Holm")
    return tables


def add_window_columns(table, info):
    table = table.copy()
    for position, column in enumerate(("window", "window_start", "window_end", "period_type")):
        table.insert(position, column, info[column])
    return table


def compare_mean_changes(people, reference, labels):
    paired = people.merge(reference[["sale_key", *CORE_COLUMNS]], on="sale_key", how="left",
                          validate="one_to_one", suffixes=("", "_reference"))
    rows = []
    for column in CORE_COLUMNS:
        valid = paired[column].notna() & paired[column + "_reference"].notna()
        frame = paired.loc[valid].copy()
        frame["change"] = frame[column] - frame[column + "_reference"]
        c, t = (frame[frame["group"].eq(g)] for g in GROUPS)
        nc, nt = len(c), len(t)
        gap_ref = t[column + "_reference"].mean() - c[column + "_reference"].mean()
        gap_now = t[column].mean() - c[column].mean()
        delta = t["change"].mean() - c["change"].mean()
        se = df = p = lower = upper = np.nan
        status = "недостаточно сейлзов с данными в обоих окнах"
        if min(nc, nt) >= MIN_KNOWN_PER_GROUP:
            vc, vt = c["change"].var(ddof=1) / nc, t["change"].var(ddof=1) / nt
            se = np.sqrt(vc + vt)
            if se > 0:
                df = (vc + vt) ** 2 / (vc ** 2 / (nc - 1) + vt ** 2 / (nt - 1))
                p = float(2 * t_dist.sf(abs(delta / se), df))
                margin = t_dist.ppf(1 - ALPHA / 2, df) * se
                lower, upper = delta - margin, delta + margin
                status = "Welch по изменениям каждого сейлза"
            elif delta == 0:
                p, lower, upper, status = 1.0, 0.0, 0.0, "нулевое изменение разницы, нет вариации"
            else:
                status = "разные постоянные изменения, SE=0; t-инференс не определён"
        rows.append({"column": column, "признак": labels[column], "paired n control": nc, "paired n treatment": nt,
                     "gap reference, same sales": gap_ref, "gap window, same sales": gap_now,
                     "change T-C": delta, "SE": se, "df": df, "CI low": lower, "CI high": upper,
                     "p": p, "метод": status})
    return pd.DataFrame(rows)


def run_analysis(dgr, ab):
    validate_settings()
    roster = prepare_ab(ab).sort_values("sale_key").reset_index(drop=True)
    if set(roster["group"]) != set(GROUPS):
        raise ValueError("Нужны исходные группы control/treatment")
    long, audit, dates = prepare_history_deals(dgr)
    raw_months = {pd.Period(m, "M") for m in dates["месяц"] if m != "NA"}
    selected = {m for m in raw_months if pd.Period(START_MONTH, "M") <= m <= pd.Period(END_MONTH, "M")}
    if not selected:
        raise ValueError("Нет распознанных месяцев в заданном диапазоне")
    complete_to = (pd.Period(LAST_COMPLETE_MONTH, "M") if LAST_COMPLETE_MONTH is not None
                   else pd.Timestamp.today().to_period("M") - 1)
    last = min(pd.Period(END_MONTH, "M"), max(selected), complete_to)
    first = pd.Period(START_MONTH, "M")
    if last - (WINDOW_MONTHS - 1) < first:
        raise ValueError("Нет ни одного полного календарного окна; проверьте даты и LAST_COMPLETE_MONTH")
    routed_months = set(long["month"])
    coverage, windows, window_info = [], {}, {}
    for start in pd.period_range(first, last - (WINDOW_MONTHS - 1), freq="M")[::STEP_MONTHS]:
        end = start + WINDOW_MONTHS - 1
        months = set(pd.period_range(start, end, freq="M"))
        missing_raw, missing_routed = months - raw_months, months - routed_months
        key = f"{start}__{end}"
        sub = long[long["month"].isin(months)]
        info = {"window": key, "window_start": str(start), "window_end": str(end), "period_type": period_type(start, end)}
        coverage.append({**info, "status": "пропущено" if missing_routed else "рассчитано",
                         "missing raw months": ", ".join(map(str, sorted(missing_raw))),
                         "missing routed months": ", ".join(map(str, sorted(missing_routed))),
                         "allocated rows, all sales": len(sub)})
        if missing_routed:
            continue  # Пропущенный целиком месяц не превращается в нулевой бизнес.
        tables = analyze_window(sub, roster, start, end)
        windows[key], window_info[key] = tables, info
    if not windows:
        raise ValueError("Нет полных окон после распределения сделок: " + str(coverage))
    desired_end = pd.Period(REFERENCE_END_MONTH, "M") if REFERENCE_END_MONTH else None
    reference_keys = [k for k, v in window_info.items() if desired_end is not None and pd.Period(v["window_end"], "M") == desired_end]
    if not reference_keys:
        pre = [k for k, v in window_info.items() if pd.Period(v["window_end"], "M") < pd.Timestamp(EXPERIMENT_START).to_period("M")]
        reference_keys = [pre[-1] if pre else next(iter(windows))]
    reference_key = reference_keys[0]
    reference = windows[reference_key]["roster"]
    labels = windows[reference_key]["numeric"].set_index("column")["признак"].to_dict()
    scales = {}
    for column in labels:
        values = [reference.loc[reference["group"].eq(g), column].var(ddof=1) for g in GROUPS]
        scales[column] = np.sqrt(sum(values) / 2)
    names = ("numeric", "categorical", "portfolio", "category_levels", "roster", "bins")
    combined = {name: [] for name in names}
    changes = []
    for key, tables in windows.items():
        numeric = tables["numeric"]
        numeric["reference window"] = reference_key
        numeric["reference SD"] = numeric["column"].map(scales)
        delta = numeric["Mean treatment"] - numeric["Mean control"]
        numeric["gap T-C"] = delta
        scale = numeric["reference SD"]
        numeric["SMD fixed reference"] = np.where(scale > 0, delta / scale, np.where(scale.eq(0) & delta.eq(0), 0.0, np.nan))
        for name in names:
            combined[name].append(add_window_columns(tables[name], window_info[key]))
        change = compare_mean_changes(tables["roster"], reference, labels)
        change["reference window"] = reference_key
        changes.append(add_window_columns(change, window_info[key]))
    result = {name: pd.concat(frames, ignore_index=True) for name, frames in combined.items()}
    result["mean_gap_changes"] = pd.concat(changes, ignore_index=True)
    # Для динамики каждого конкретного теста — отдельная поправка по времени.
    # Она не контролирует общую ошибку по всем различным признакам.
    correct_over_time(result)
    # Одна семья всех рассчитанных тестов по всем окнам и признакам.
    # Коррекция Холма применима и к зависимым, перекрывающимся окнам.
    correct_tables(result, ("numeric", "categorical", "portfolio", "mean_gap_changes"), " Holm all windows")
    for key, tables in windows.items():
        for name in ("numeric", "categorical", "portfolio"):
            selected_rows = result[name][result[name]["window"].eq(key)].reset_index(drop=True)
            tables[name] = selected_rows
    result.update({"coverage": pd.DataFrame(coverage), "audit": audit, "dates": dates,
                   "windows": windows, "reference_window": reference_key})
    result["parameters"] = pd.DataFrame([
        ("Диапазон", f"{START_MONTH} — {END_MONTH}"), ("Длина окна, мес.", WINDOW_MONTHS), ("Шаг, мес.", STEP_MONTHS),
        ("Последний разрешённый месяц", str(complete_to)), ("Последний доступный месяц", str(last)),
        ("Полнота месяца", "нужно подтвердить по выгрузке; наличие строк не гарантирует полноту"),
        ("Запуск A/B", EXPERIMENT_START), ("Опорное окно", reference_key),
        ("Фиксированный состав", "весь переданный AB; состав и имена групп одинаковы во всех окнах"),
        ("Нет истории в окне", "ноль" if ABSENT_PRE_IS_ZERO else "NA, сейлз остаётся в составе"),
        ("Частичный NA", "вся сумма соответствующей метрики у сейлза = NA"),
        ("Клиентский счётчик", "сумма, не число уникальных клиентов"),
        ("Фильтры по исходам / выбросы", "не применяются"),
        ("Отсутствующий целиком месяц", "окно пропускается с записью в coverage.csv"),
        ("Категории", "доминирующая по доле сделок; при NA или нулевой сумме сделок — веса по долям строк"),
        ("Квантили по сделкам", "общие для T/C в каждом окне, пересчитываются; границы в bins.csv"),
        ("KS", "все известные и отдельно ненулевые; повторы => перестановочный p"),
        ("Chi-square", "Pearson correction=False; ожидаемая частота <5 => перестановочный p"),
        ("p Holm", "поправка по признакам внутри окна"),
        ("p Holm over time", "одна семья = один признак и один тест во всех окнах; используется на графиках динамики"),
        ("p Holm all windows", "поправка по всем окнам, признакам и тестам, включая изменение средних"),
        ("Минимальное перестановочное p", 1 / (PERMUTATIONS + 1)),
        ("Разрешение общей поправки", "при большом числе тестов общая Holm консервативна и может потребовать больше перестановок"),
        ("Изменение разницы средних", "Welch по индивидуальным изменениям, только сейлзы с данными в обоих окнах"),
        ("SMD fixed reference", "разница средних окна / фиксированное pooled SD опорного окна"),
        ("Минимум известных в группе", MIN_KNOWN_PER_GROUP), ("Перестановок", PERMUTATIONS), ("Seed", RANDOM_SEED),
        ("Интерпретация", "p>0.05 не доказывает однородность; динамику оцениваем также по SMD, KS D и пропускам"),
        ("A/A", "только окна до запуска агента; остальные — диагностика во время A/B"),
    ], columns=["настройка", "значение"])
    return result


def trajectory_frame(result, table_name, column):
    base = result["coverage"][["window", "window_end"]].copy()
    values = result[table_name]
    values = values[values["column"].eq(column)].drop(columns=["window_end"])
    return base.merge(values, on="window", how="left", validate="one_to_one")


def style_time_axis(ax, dates):
    style_axis(ax)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    launch = pd.Timestamp(EXPERIMENT_START)
    if dates.min() <= launch <= dates.max():
        ax.axvline(launch, color="white", linestyle=":", alpha=0.75)
        ax.axvspan(launch, dates.max(), color="white", alpha=0.035)
    ax.legend(fontsize=10)
    style_legend(ax)


def core_dynamics_plot(result, out_dir, dpi, show):
    fig, axes = plt.subplots(3, 3, figsize=(23, 15), squeeze=False)
    fig.patch.set_alpha(0)
    fig.suptitle(f"Баланс групп: скользящее окно {WINDOW_MONTHS} мес.\nОпорное окно: {result['reference_window']} | пунктир — запуск A/B",
                 color="white", fontsize=19)
    for i, column in enumerate(CORE_COLUMNS):
        table = trajectory_frame(result, "numeric", column)
        dates = pd.to_datetime(table["window_end"]) + pd.offsets.MonthEnd(0)
        label = result["numeric"].loc[result["numeric"]["column"].eq(column), "признак"].iloc[0]
        ax = axes[i, 0]
        for group in GROUPS:
            ax.plot(dates, table[f"Mean {group}"], marker="o", ms=3, color=COLORS[group], label=group)
        ax.set_title(label + ": среднее", fontsize=13)
        ax.set_ylabel("На одного сейлза с известной метрикой")
        ax = axes[i, 1]
        ax.plot(dates, table["SMD T-C"].replace([np.inf, -np.inf], np.nan), color=COLORS["control"], marker="o", ms=3, label="SMD внутри окна")
        ax.plot(dates, table["SMD fixed reference"], color=COLORS["treatment"], marker="o", ms=3, label="В масштабе опорного окна")
        ax.axhline(0, color="white", linewidth=0.8)
        for value in (-0.1, 0.1):
            ax.axhline(value, color="white", linestyle="--", alpha=0.35)
        ax.set_title("Разница T−C в стандартных отклонениях", fontsize=13)
        ax.set_ylabel("SMD; ±0.1 — описательный ориентир")
        ax = axes[i, 2]
        ax.plot(dates, table["KS all | p"], color=COLORS["control"], marker="o", ms=3, label="KS all: p")
        ax.plot(dates, table["KS all | p Holm over time"], color=COLORS["treatment"], marker="o", ms=3, label="KS all: Holm по окнам этого признака")
        ax.axhline(ALPHA, color="white", linestyle="--", alpha=0.7, label=f"α={ALPHA:g}")
        ax.set_ylim(-0.03, 1.04)
        ax.set_title("KS: все известные значения", fontsize=13)
        ax.set_ylabel("p-value")
        for ax in axes[i]:
            style_time_axis(ax, dates)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.5, w_pad=2)
    path = out_dir / "numeric_dynamics.png"
    export_figure(fig, path, dpi, show)
    return path


def coverage_and_distance_plot(result, out_dir, dpi, show):
    fig, axes = plt.subplots(3, 2, figsize=(21, 15), squeeze=False)
    fig.patch.set_alpha(0)
    fig.suptitle("Расстояние распределений и доступность данных\nСостав AB фиксирован; число известных значений может меняться",
                 color="white", fontsize=19)
    for i, column in enumerate(CORE_COLUMNS):
        table = trajectory_frame(result, "numeric", column)
        dates = pd.to_datetime(table["window_end"]) + pd.offsets.MonthEnd(0)
        label = result["numeric"].loc[result["numeric"]["column"].eq(column), "признак"].iloc[0]
        ax = axes[i, 0]
        ax.plot(dates, table["KS all | D"], marker="o", ms=3, color=COLORS["control"], label="KS D: все значения")
        ax.plot(dates, table["KS nonzero | D"], marker="o", ms=3, color=COLORS["treatment"], label="KS D: ненулевые")
        ax.set_ylim(-0.03, 1.04)
        ax.set_title(label, fontsize=13)
        ax.set_ylabel("KS D; 0 — одинаковые эмпирические распределения")
        ax = axes[i, 1]
        for group in GROUPS:
            ax.plot(dates, table[f"Known {group}"], marker="o", ms=3, color=COLORS[group], label=f"{group}: известные")
            ax.plot(dates, table[f"N {group}"], linestyle="--", alpha=0.5, color=COLORS[group], label=f"{group}: весь состав")
        ax.set_title("Число сейлзов с известным показателем", fontsize=13)
        ax.set_ylabel("Сейлзы")
        for ax in axes[i]:
            style_time_axis(ax, dates)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.5, w_pad=2)
    path = out_dir / "distribution_and_coverage.png"
    export_figure(fig, path, dpi, show)
    return path


def mean_changes_plot(result, out_dir, dpi, show):
    fig, axes = plt.subplots(3, 2, figsize=(21, 15), squeeze=False)
    fig.patch.set_alpha(0)
    fig.suptitle(f"Как изменилась разница средних относительно {result['reference_window']}\nРасчёт по индивидуальным изменениям тех же сейлзов",
                 fontsize=19, color="white")
    for i, column in enumerate(CORE_COLUMNS):
        table = trajectory_frame(result, "mean_gap_changes", column)
        dates = pd.to_datetime(table["window_end"]) + pd.offsets.MonthEnd(0)
        label = result["numeric"].loc[result["numeric"]["column"].eq(column), "признак"].iloc[0]
        ax = axes[i, 0]
        ax.plot(dates, table["change T-C"], marker="o", ms=3, color=COLORS["control"], label="Изменение разницы T−C")
        ax.fill_between(dates, table["CI low"], table["CI high"], color=COLORS["control"], alpha=0.2,
                        label=f"{100 * (1 - ALPHA):g}% CI, без поправки")
        ax.axhline(0, color="white", linestyle="--", alpha=0.6)
        ax.set_title(label, fontsize=13)
        ax.set_ylabel("Изменение разницы средних")
        ax = axes[i, 1]
        ax.plot(dates, table["p"], color=COLORS["control"], marker="o", ms=3, label="Welch по изменениям: p")
        ax.plot(dates, table["p Holm over time"], color=COLORS["treatment"], marker="o", ms=3, label="Holm по окнам этого признака")
        ax.axhline(ALPHA, color="white", linestyle="--", alpha=0.6, label=f"α={ALPHA:g}")
        ax.set_ylim(-0.03, 1.04)
        ax.set_ylabel("p-value")
        ax.set_title("Тест изменения разницы средних", fontsize=13)
        for ax in axes[i]:
            style_time_axis(ax, dates)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.5, w_pad=2)
    path = out_dir / "mean_gap_changes.png"
    export_figure(fig, path, dpi, show)
    return path


def pvalues_heatmap(result, out_dir, dpi, show, corrected=False):
    rows, labels = [], []
    order = result["coverage"]["window"].tolist()
    numeric = result["numeric"]
    short = {"income_pre": "Доход", "deals_pre": "Сделки", "clients_pre": "Клиентский счётчик",
             "rows_pre": "Строки с долями", "months_pre": "Месяцы с данными"}
    suffix = " Holm over time" if corrected == "time" else " Holm all windows" if corrected else ""
    for column in numeric["column"].drop_duplicates():
        part = numeric[numeric["column"].eq(column)].set_index("window")
        for test in ("KS all", "KS nonzero", "chi2 zero", "chi2 NA"):
            labels.append(f"{short.get(column, column)} | {test}")
            rows.append(part[test + " | p" + suffix].reindex(order).to_numpy(float))
    categories = result["categorical"]
    if len(categories):
        for field in categories["поле"].drop_duplicates():
            part = categories[categories["поле"].eq(field)].set_index("window")
            labels.append(f"{field} | χ²")
            rows.append(part["p" + suffix].reindex(order).to_numpy(float))
    values = np.asarray(rows, float)
    transformed = np.ma.masked_invalid(-np.log10(np.maximum(values, 1e-300)))
    fig, ax = plt.subplots(figsize=(max(16, len(order) * 0.8), max(9, len(rows) * 0.42 + 3)))
    fig.patch.set_alpha(0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((1, 1, 1, 0.12))
    artist = ax.imshow(transformed, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=8)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(order)), result["coverage"]["window_end"].tolist(), rotation=55, ha="right")
    for i in range(len(rows)):
        for j in range(len(order)):
            value = values[i, j]
            text = ("—" if not np.isfinite(value) else f"{value:.2f}" if value >= 0.01 else f"{value:.0e}")
            color = "black" if np.isfinite(value) and -np.log10(max(value, 1e-300)) > 5 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel(f"Последний месяц окна ({WINDOW_MONTHS} мес.)")
    ax.set_title("p-value: " + ("Holm по времени отдельно для каждого теста" if corrected == "time" else "Holm по всем окнам и признакам" if corrected else "исходные")
                 + "\n— означает пропущенное окно или недостаток данных", fontsize=19)
    style_axis(ax)
    ax.grid(False)
    colorbar = fig.colorbar(artist, ax=ax, pad=0.015)
    colorbar.set_label("−log10(p), цвет ограничен сверху 8", color="white")
    colorbar.ax.tick_params(colors="white")
    colorbar.outline.set_edgecolor("white")
    fig.tight_layout()
    path = out_dir / ("pvalues_holm_over_time.png" if corrected == "time" else "pvalues_holm_all_windows.png" if corrected else "pvalues_raw.png")
    export_figure(fig, path, dpi, show)
    return path


def save_report(result, out_dir=None, dpi=None, show=None):
    out_dir = Path(OUT_DIR if out_dir is None else out_dir)
    dpi, show = DPI if dpi is None else dpi, SHOW if show is None else show
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in result.items():
        if isinstance(table, pd.DataFrame):
            path = out_dir / f"{name}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            paths.append(path)
    paths.append(core_dynamics_plot(result, out_dir, dpi, show))
    paths.append(coverage_and_distance_plot(result, out_dir, dpi, show))
    paths.append(mean_changes_plot(result, out_dir, dpi, show))
    paths.append(pvalues_heatmap(result, out_dir, dpi, show, corrected=False))
    paths.append(pvalues_heatmap(result, out_dir, dpi, show, corrected="time"))
    paths.append(pvalues_heatmap(result, out_dir, dpi, show, corrected=True))
    if SAVE_WINDOW_PLOTS:
        for key, tables in result["windows"].items():
            folder = out_dir / "windows" / key
            folder.mkdir(parents=True, exist_ok=True)
            paths += plot_numeric(tables, folder, dpi, show)
            if SAVE_WINDOW_CATEGORY_PLOTS:
                paths += plot_categories(tables, folder, dpi, show)
    return paths


def main(dgr=None, ab=None):
    if dgr is None:
        dgr = pd.read_excel(DATA_FILE)
    if ab is None:
        ab = pd.read_excel(AB_FILE, sheet_name=AB_SHEET)
    result = run_analysis(dgr, ab)
    print(result["coverage"].to_string(index=False))
    print("\nОпорное окно:", result["reference_window"])
    print("Рассчитано полных окон:", len(result["windows"]))
    paths = save_report(result)
    print(f"Сохранено файлов: {len(paths)}. Папка: {Path(OUT_DIR).resolve()}")
    return result


if __name__ == "__main__":
    result = main()
