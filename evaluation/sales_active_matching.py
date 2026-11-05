# -*- coding: utf-8 -*-
"""Matching пользователей агента и сканирование порога interactions > k.

Зависимости: pandas, numpy, scipy, openpyxl.
Jupyter:
    import sales_active_matching as am
    tables = am.run_analysis(dgr, ab, active)
    display(tables['effects'])
    am.save_report(tables)
Или tables = am.main(active=active): сделки и AB читаются из настроек ниже.

Оригинальные группы никогда не меняются. active должен содержать количество
взаимодействий за тот же TEST_MONTHS: временных меток в нём нет, отфильтровать
иной период автоматически невозможно.
Matching только по PRE_MONTHS, один control на пользователя, без повторений.
По умолчанию глобально минимизируется сумма евклидовых расстояний, пары
фиксированы на всех порогах. Пары — единица статистического расчёта.

RAW: парный t-тест разниц D_Y. CUPED: D_Y = tau + beta*D_X, ошибки HC3,
tau оценивается при D_X=0. D_X масштабируется без центрирования!
Эффект total = tau * число пар, MDE total = MDE per user * число пар.
Не применяется фильтр положительного дохода; нулевые и отрицательные Y
сохраняются. Результаты — сравнение выбранных пользователей с matched control,
а не восстановленный ITT/причинный эффект исходного рандомизированного AB.
Номинальные p и MDE диагностические; Holm дополнительно по уникальным составам.
"""
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, brentq
from scipy.spatial.distance import cdist
from scipy.stats import t as t_dist, nct

DATA_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\data\441774c5-7986-43a6-8269-5d5ce8627e9b.xlsx")
AB_FILE = Path(r"C:\Users\24006183\Documents\ab\ab sales\Телефоны УГР 29.06.2026.xlsx")
AB_SHEET = "Штатная расстановка"
AB_NAME = "Ф.И.О. сотрудника"
AB_GROUP = "AB-тест"
FINAL_GROUPS_FILE = None  # Опционально: three_controls_to_remove.xlsx, лист Итоговый_состав.
ACTIVE_FILE = None        # Не нужен при передаче DataFrame active.
OUT_DIR = Path("ab sales/active_matching")
PRE_MONTHS = ("2026-05", "2026-06")
TEST_MONTHS = ("2026-09", "2026-10")

MATCH_FEATURES = [
    {"column": "Доход", "transform": "signed_log1p", "weight": 1.0},
    {"column": "Кол-во сделок", "transform": "log1p", "weight": 1.0},
    # {"column": "Кол-во клиентов", "transform": "log1p", "weight": 1.0},
    # {"column": "Сегмент", "weight": 1.0},
    # {"column": "Отрасль", "weight": 1.0},
    # {"column": "Продукт", "weight": 1.0},
    # {"column": "тип сейлза", "weight": 1.0},
]
# Для категорий используются доли портфеля по количеству сделок до теста.
# После стандартизации блок поля нормируется на sqrt(числа его координат).
EXACT_MATCH_ON = []           # Например ["тип сейлза"] или ["Сегмент"].
REMATCH_EACH_THRESHOLD = False
MAX_MATCH_DISTANCE = None     # Опциональный предел; неполный matching вызывает ошибку.
EXCLUDE_ACTIVE_CONTROLS = True
ACTIVE_DUPLICATE_RULE = "sum" # "sum", "max" или "error" после нормализации ФИО.
MIN_PAIRS = 10                # Меньше: численности показываются, p/MDE не вычисляются.
THRESHOLD_MODE = "all_integers" # Или "changes_only": интервалы одинаковых составов.
MAX_EXPANDED_THRESHOLDS = 100000 # При больших счётчиках автоматически интервальная запись.
ALPHA = 0.05
POWER = 0.80
DEALS_COL = "Кол-во сделок"
CLIENTS_COL = "Кол-во клиентов"
CATEGORY_FIELDS = {"ТБ", "Сегмент", "Дэск", "ДКК", "Отрасль", "Продукт", "тип сейлза"}
NUMERIC_FIELDS = {"Доход": "X", "Кол-во сделок": "deals_pre", "Кол-во клиентов": "clients_pre", "Месяцев с данными": "months_pre"}

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
    if not MATCH_FEATURES or MIN_PAIRS < 3 or not 0 < ALPHA < POWER < 1:
        raise ValueError("Проверьте MATCH_FEATURES, MIN_PAIRS, ALPHA, POWER")
    if ACTIVE_DUPLICATE_RULE not in {"sum", "max", "error"} or THRESHOLD_MODE not in {"all_integers", "changes_only"}:
        raise ValueError("Неизвестный режим дубликатов active или порогов")
    if MAX_EXPANDED_THRESHOLDS < 1 or MAX_MATCH_DISTANCE is not None and MAX_MATCH_DISTANCE < 0:
        raise ValueError("Проверьте ограничения порогов/расстояния")
    if not PRE_MONTHS or not TEST_MONTHS or max(pd.Period(m) for m in PRE_MONTHS) >= min(pd.Period(m) for m in TEST_MONTHS):
        raise ValueError("Пре-период должен предшествовать тестовому")
    if len({s.get('column') for s in MATCH_FEATURES}) != len(MATCH_FEATURES):
        raise ValueError("Повторное поле в MATCH_FEATURES")
    for spec in MATCH_FEATURES:
        if spec.get("column") not in CATEGORY_FIELDS | set(NUMERIC_FIELDS):
            raise ValueError(f"Неизвестное поле matching: {spec}")
        if not np.isfinite(spec.get("weight", 1.0)) or spec.get("weight", 1.0) <= 0:
            raise ValueError("Веса matching должны быть положительными")
        if spec.get("transform", "raw") not in {"raw", "log1p", "signed_log1p"}:
            raise ValueError("Неизвестное преобразование признака")
    if not set(EXACT_MATCH_ON) <= CATEGORY_FIELDS:
        raise ValueError("EXACT_MATCH_ON поддерживает категориальные поля")


def prepare_activity(active, roster):
    require_columns(active, ["sales", "interactions"], "active")
    activity = pd.DataFrame({"sale_key": active["sales"].map(name_key),
                             "sales": active["sales"].map(clean_text),
                             "interactions": pd.to_numeric(active["interactions"], errors="coerce")})
    value = activity["interactions"].to_numpy(float)
    if np.any(~np.isfinite(value)) or np.any(value < 0) or np.any(np.abs(value-np.round(value)) > 1e-9):
        raise ValueError("active.interactions должен содержать конечные неотрицательные целые числа")
    if activity["sale_key"].eq("").any():
        raise ValueError("В active есть пустое ФИО")
    activity["interactions"] = activity["interactions"].astype('int64')
    counts = activity.groupby("sale_key").size()
    if ACTIVE_DUPLICATE_RULE == "error" and counts.gt(1).any():
        raise ValueError(f"Повторные ФИО в active: {counts[counts.gt(1)].index.tolist()}")
    rule = "max" if ACTIVE_DUPLICATE_RULE == "max" else "sum"
    merged = activity.groupby("sale_key", as_index=False).agg(
        sales=("sales", "first"), interactions=("interactions", rule), active_rows=("sales", "size"))
    merged = merged.merge(roster[["sale_key", "group"]], on="sale_key", how="left", validate="one_to_one")
    merged["категория active"] = np.select(
        [merged["group"].isna(), merged["interactions"].eq(0), merged["group"].eq("control")],
        ["вне исходного AB", "нулевое использование", "control с использованием"], default="пользователь treatment")
    return merged


def prepare_population(dgr, ab):
    dgr = dgr.copy()
    dgr.columns = [clean_text(c) for c in dgr.columns]
    if not dgr.columns.is_unique:
        raise ValueError("Дубли заголовков после очистки пробелов")
    categories = ({s["column"] for s in MATCH_FEATURES} | set(EXACT_MATCH_ON)) & CATEGORY_FIELDS
    columns = ["Продукт", FLOW_COL, FICC_COL, "Доход", DEALS_COL, "Месяц-Год"]
    columns += list(categories - {"тип сейлза", "Продукт"})
    need_clients = any(s["column"] == CLIENTS_COL for s in MATCH_FEATURES)
    if need_clients:
        columns.append(CLIENTS_COL)
    require_columns(dgr, columns, "dgr")
    roster = prepare_ab(ab)
    pre_set, test_set = ({pd.Period(m, freq="M") for m in window} for window in [PRE_MONTHS, TEST_MONTHS])
    wanted = pre_set | test_set
    records, audit = [], []
    for row_number, (_, row) in enumerate(dgr.iterrows(), start=2):
        product = clean_text(row["Продукт"])
        month = parse_month(row["Месяц-Год"])
        if product in excluded_products:
            reason = "исключённый продукт"
        elif product not in flow_products | ficc_products:
            reason = "неизвестный продукт"
        elif pd.isna(month):
            reason = "не распознана дата"
        elif month.to_period("M") not in wanted:
            continue
        else:
            names = split_names(row[FLOW_COL if product in flow_products else FICC_COL])
            reason = "нет ФИО в нужной колонке" if not names else ""
        if reason:
            audit.append({"строка Excel": row_number, "причина": reason, "Продукт": product,
                          "Месяц-Год": clean_text(row["Месяц-Год"])})
            continue
        numeric = {}
        for source, target in [("Доход", "income"), (DEALS_COL, "deals_pre")] + ([(CLIENTS_COL, "clients_pre")] if need_clients else []):
            if source != "Доход" and month.to_period("M") not in pre_set:
                numeric[target] = np.nan
                continue
            v = pd.to_numeric(row[source], errors="coerce")
            if pd.isna(v) or not np.isfinite(v) or source != "Доход" and v < 0:
                raise ValueError(f"Строка Excel {row_number}: некорректное {source}={row[source]!r}")
            numeric[target] = float(v)/len(names)
        cats = {f: (("Flow" if product in flow_products else "FICC") if f == "тип сейлза"
                    else clean_text(row[f]).casefold() or "__ПРОПУСК__") for f in categories}
        for key, name in names:
            records.append({"sale_key": key, "sale_name": name, "month_p": month.to_period("M"),
                            "row_share": 1/len(names), **numeric, **cats})
    if not records:
        raise ValueError("Нет валидных распределённых сделок")
    long = pd.DataFrame(records)
    missing = wanted - set(long["month_p"])
    if missing:
        raise ValueError(f"В распределённых сделках отсутствуют месяцы: {sorted(str(m) for m in missing)}. "
                         "Проверьте исходные даты, продуктовые списки и ФИО.")
    pre, current = long[long["month_p"].isin(pre_set)].copy(), long[long["month_p"].isin(test_set)]
    aggregation = pre.groupby("sale_key").agg(X=("income", "sum"), deals_pre=("deals_pre", "sum"), months_pre=("month_p", "nunique"))
    if need_clients:
        aggregation["clients_pre"] = pre.groupby("sale_key")["clients_pre"].sum()
    out = roster.merge(aggregation, on="sale_key", how="left", validate="one_to_one")
    out["has_pre"] = out["X"].notna()
    out["Y"] = out["sale_key"].map(current.groupby("sale_key")["income"].sum()).fillna(0.0)
    out["has_test_rows"] = out["sale_key"].isin(set(current["sale_key"]))
    pre["weight"] = pre["deals_pre"]
    zero = pre.groupby("sale_key")["weight"].transform("sum").eq(0)
    pre.loc[zero, "weight"] = pre.loc[zero, "row_share"]
    profiles = {}
    for field in sorted(categories):
        table = pre.groupby(["sale_key", field])["weight"].sum().unstack(fill_value=0)
        table = table.reindex(out["sale_key"], fill_value=0).fillna(0.0)
        total = table.sum(axis=1)
        table = table.div(total.replace(0, np.nan), axis=0).fillna(0.0)
        table["__НЕТ_ИСТОРИИ__"] = total.eq(0).astype(float)
        table = table.reindex(sorted(table.columns), axis=1)
        table.index = out["sale_key"]
        profiles[field] = table
        out[field] = table.idxmax(axis=1).to_numpy()
    return out, profiles, pd.DataFrame(audit, columns=["строка Excel", "причина", "Продукт", "Месяц-Год"])


def matching_space(reference, profiles):
    arrays, raw_arrays, metadata = [], [], []
    keys = reference["sale_key"]
    for spec in MATCH_FEATURES:
        field = spec["column"]
        if field in NUMERIC_FIELDS:
            v = reference[NUMERIC_FIELDS[field]].to_numpy(float)
            transform = spec.get("transform", "raw")
            if transform == "log1p":
                if np.any(v < 0):
                    raise ValueError(f"Для отрицательных значений {field} используйте signed_log1p")
                v = np.log1p(v)
            elif transform == "signed_log1p":
                v = np.sign(v)*np.log1p(np.abs(v))
            block, names = v[:, None], [f"{field} | {transform}"]
        else:
            table = profiles[field].loc[keys]
            block, names = table.to_numpy(float), [f"{field} | {c}" for c in table.columns]
        if not np.isfinite(block).all():
            raise ValueError(f"Неизвестные предэкспериментальные значения для matching: {field}")
        scale = block.std(axis=0, ddof=1)
        keep = scale > 0
        if not keep.any():
            continue
        block, scale = block[:, keep], scale[keep]
        names = [n for n,k in zip(names,keep) if k]
        center = block.mean(axis=0)
        factor = np.sqrt(spec.get("weight",1.0)/block.shape[1])
        arrays.append((block-center)/scale*factor)
        raw_arrays.append(block)
        metadata.extend({"поле": field, "координата": name, "среднее reference": mu,
                         "SD reference": sd, "вес поля": spec.get("weight",1.),
                         "множитель координаты": factor} for name,mu,sd in zip(names,center,scale))
    if not arrays:
        # Все одинаковы: все расстояния равны нулю, порядок ФИО детерминирует matching.
        return np.zeros((len(reference),1)), np.zeros((len(reference),1)), pd.DataFrame(
            [{"поле":"все признаки постоянны","координата":"константа","SD reference":0.}])
    return np.column_stack(arrays), np.column_stack(raw_arrays), pd.DataFrame(metadata)


def match_users(users, controls, space):
    if users.empty:
        return pd.DataFrame()
    if len(users)>len(controls):
        raise ValueError(f"Для matching без повторов не хватает control: пользователей {len(users)}, control {len(controls)}")
    cost = cdist(space.loc[users["sale_key"]],space.loc[controls["sale_key"]],metric="euclidean")
    for field in EXACT_MATCH_ON:
        cost[users[field].to_numpy()[:,None] != controls[field].to_numpy()[None,:]] = np.inf
    if MAX_MATCH_DISTANCE is not None:
        cost[cost > MAX_MATCH_DISTANCE] = np.inf
    try:
        row,col = linear_sum_assignment(cost)
    except ValueError as exc:
        raise ValueError("Не удалось подобрать уникальный control каждому пользователю при заданных ограничениях matching") from exc
    if len(row)!=len(users) or not np.isfinite(cost[row,col]).all():
        raise ValueError("Неполный matching: проверьте EXACT_MATCH_ON и MAX_MATCH_DISTANCE")
    matched_t,matched_c = users.iloc[row],controls.iloc[col]
    pairs = pd.DataFrame({
        "t_key":matched_t["sale_key"].to_numpy(),"treatment":matched_t["sale_name"].to_numpy(),
        "c_key":matched_c["sale_key"].to_numpy(),"control":matched_c["sale_name"].to_numpy(),
        "interactions":matched_t["interactions"].to_numpy(),"distance":cost[row,col],
        "X_T":matched_t["X"].to_numpy(),"X_C":matched_c["X"].to_numpy(),
        "Y_T":matched_t["Y"].to_numpy(),"Y_C":matched_c["Y"].to_numpy()})
    pairs["D_X"] = pairs["X_T"]-pairs["X_C"]
    pairs["D_Y"] = pairs["Y_T"]-pairs["Y_C"]
    return pairs


@lru_cache(maxsize=4096)
def mde_multiplier(df,alpha,power):
    critical=t_dist.ppf(1-alpha/2,df)
    def probability(a):
        return nct.sf(critical,df,a)+nct.cdf(-critical,df,a)
    upper=1.
    while probability(upper)<power and upper<1024:
        upper*=2
    return float(brentq(lambda a:probability(a)-power,0.,upper))


def paired_raw(pairs):
    n=len(pairs)
    if n==0:
        return {"status":"нет пар"}
    D=pairs["D_Y"].to_numpy(float)
    result={"delta":float(D.mean()),"status":"мало пар"}
    if n<MIN_PAIRS:
        return result
    variance=float(np.var(D,ddof=1))
    if variance <= 1e-24*max(float(np.mean(D**2)),1.):
        return {**result,"status":"нулевая дисперсия разниц: p/MDE не оцениваются"}
    se=np.sqrt(variance/n)
    return {"delta":float(D.mean()),"se":se,"df":n-1,"theta":0.,"status":"OK"}


def paired_cuped(pairs,pre_scale):
    n=len(pairs)
    if n<MIN_PAIRS:
        return {"status":"нет пар" if n==0 else "мало пар"}
    dx=pairs["D_X"].to_numpy(float)
    Y=pairs["D_Y"].to_numpy(float)
    if np.all(np.abs(dx)<=1e-12*max(pre_scale,1.)):
        raw = paired_raw(pairs)
        if raw['status'] == 'OK':
            raw['status'] = 'OK: D_X=0, CUPED совпадает с RAW'
        return {**raw,"theta":0.}
    scale=max(pre_scale,1.)
    D=np.column_stack([np.ones(n),dx/scale])
    if np.linalg.matrix_rank(D)<2:
        return {"status":"CUPED не идентифицируется: одинаковая ненулевая D_X"}
    beta=np.linalg.lstsq(D,Y,rcond=None)[0]
    residual=Y-D@beta
    if np.linalg.norm(residual)<=1e-12*max(np.linalg.norm(Y),1.):
        return {"delta":float(beta[0]),"theta":float(beta[1]/scale),"status":"нулевой остаточный шум: p/MDE не оцениваются"}
    bread=np.linalg.inv(D.T@D)
    leverage=np.einsum('ij,jk,ik->i',D,bread,D)
    if np.any(1-leverage<=1e-10):
        return {"status":"вырожденный leverage CUPED"}
    weights=(residual/(1-leverage))**2
    covariance=bread@(D.T@(weights[:,None]*D))@bread
    variance=float(covariance[0,0])
    if not np.isfinite(variance) or variance<=0:
        return {"status":"нет оценки SE CUPED"}
    return {"delta":float(beta[0]),"se":np.sqrt(variance),"df":n-2,
            "theta":float(beta[1]/scale),"status":"OK"}


def effect_row(estimate,pairs,method,threshold,upper):
    n=len(pairs)
    delta=estimate.get("delta",np.nan)
    se=estimate.get("se",np.nan)
    df=estimate.get("df",np.nan)
    if np.isfinite(se) and se>0 and np.isfinite(df) and df>0:
        critical=t_dist.ppf(1-ALPHA/2,df)
        p=float(2*t_dist.sf(abs(delta/se),df))
        mde=mde_multiplier(float(df),ALPHA,POWER)*se
        lo,hi=delta-critical*se,delta+critical*se
    else:
        p=mde=lo=hi=np.nan
    signal="НЕТ ОЦЕНКИ" if not np.isfinite(p) else ("НЕТ СИГНАЛА" if p>=ALPHA else
            ("ПОЛОЖИТЕЛЬНЫЙ СИГНАЛ" if delta>0 else "ОТРИЦАТЕЛЬНЫЙ СИГНАЛ"))
    return {"interactions >":threshold,"до порога включительно":upper,"метод":method,
            "n_pairs":n,"n_treatment":n,"n_control":n,
            "средний доход T":float(pairs["Y_T"].mean()) if n else np.nan,
            "средний доход C":float(pairs["Y_C"].mean()) if n else np.nan,
            "эффект на пользователя, руб.":delta,"SE на пользователя, руб.":se,
            "эффект total treatment, руб.":delta*n if n else np.nan,
            "SE total treatment, руб.":se*n if n else np.nan,
            "ДИ total, нижний":lo*n if n else np.nan,"ДИ total, верхний":hi*n if n else np.nan,
            "MDE на пользователя, прибл.":mde,"MDE total treatment, прибл.":mde*n if n else np.nan,
            "эффект / MDE":delta/mde if mde>0 else np.nan,
            "p-value nominal":p,"df":df,"theta CUPED":estimate.get('theta',np.nan),
            "сигнал nominal":signal,"статус":estimate['status'],
            "среднее расстояние":float(pairs['distance'].mean()) if n else np.nan,
            "max расстояние":float(pairs['distance'].max()) if n else np.nan}


def holm(values):
    values=np.asarray(values,float)
    out=np.full(len(values),np.nan)
    finite=np.flatnonzero(np.isfinite(values))
    order=finite[np.argsort(values[finite])]
    if len(order):
        out[order]=np.minimum(np.maximum.accumulate((len(order)-np.arange(len(order)))*values[order]),1.)
    return out


def balance_rows(pairs,reference,raw_space,feature_info,threshold,upper):
    if pairs.empty:
        return []
    info=feature_info.copy()
    rows=[]
    keyed=reference.set_index('sale_key')
    blocks=[('Доход до теста, RAW',keyed['X']),('Кол-во сделок до теста, RAW',keyed['deals_pre'])]
    for j,row in info.iterrows():
        blocks.append((row['координата'],raw_space.iloc[:,j]))
    for name,values in blocks:
        t=values.loc[pairs['t_key']].to_numpy(float)
        c=values.loc[pairs['c_key']].to_numpy(float)
        sd=float(values.std(ddof=1))
        diff=float(t.mean()-c.mean())
        smd=diff/sd if sd>0 else (0. if diff==0 else np.sign(diff)*np.inf)
        rows.append({'interactions >':threshold,'до порога включительно':upper,'признак':name,
                     'среднее T':t.mean(),'среднее C':c.mean(),'разница T-C':diff,
                     'SD reference':sd,'SMD reference':smd,'abs SMD reference':abs(smd)})
    return rows


def run_analysis(dgr,ab,active):
    validate_settings()
    population,profiles,deal_audit=prepare_population(dgr,ab)
    activity=prepare_activity(active,population)
    population=population.merge(activity[['sale_key','interactions']],on='sale_key',how='left',validate='one_to_one')
    population['interactions']=population['interactions'].fillna(0).astype('int64')
    users=population[population['group'].eq('treatment') & population['interactions'].gt(0)].copy()
    if users.empty:
        raise ValueError("В active нет пользователей из исходного treatment")
    no_history=users.loc[~users['has_pre'],'sale_name'].tolist()
    if no_history:
        raise ValueError(f"У пользователей treatment отсутствует история пре-периода: {no_history}. "
                         "Проверьте сопоставление ФИО/период; нельзя незаметно исключить этих пользователей.")
    eligible_control=population['group'].eq('control') & population['has_pre']
    if EXCLUDE_ACTIVE_CONTROLS:
        eligible_control &= population['interactions'].eq(0)
    controls=population[eligible_control].copy()
    population['использован как активный T']=population['sale_key'].isin(set(users['sale_key']))
    population['допустимый control']=population['sale_key'].isin(set(controls['sale_key']))
    users=users.sort_values('sale_key').reset_index(drop=True)
    controls=controls.sort_values('sale_key').reset_index(drop=True)
    if len(controls)<len(users):
        raise ValueError(f"Пользователей treatment {len(users)}, допустимых control {len(controls)}: "
                         "для полного matching 1:1 без повторов не хватает control.")
    reference=pd.concat([users,controls],ignore_index=True)
    matrix,raw,feature_info=matching_space(reference,profiles)
    space=pd.DataFrame(matrix,index=reference['sale_key'])
    raw_space=pd.DataFrame(raw,index=reference['sale_key'])
    pre_scale=float(reference['X'].std(ddof=1))
    initial_pairs=match_users(users,controls,space)
    endpoints=[0]+sorted(set(int(x) for x in users['interactions']))
    max_count=endpoints[-1]
    expanded=THRESHOLD_MODE=='all_integers' and max_count+1<=MAX_EXPANDED_THRESHOLDS
    records,balance,membership=[],[],[]
    for j,lower in enumerate(endpoints):
        upper=endpoints[j+1]-1 if j+1<len(endpoints) else max_count
        if REMATCH_EACH_THRESHOLD:
            pairs=match_users(users[users['interactions']>lower],controls,space)
        else:
            pairs=initial_pairs[initial_pairs['interactions']>lower].copy()
        if not pairs.empty:
            kept=pairs.copy()
            kept.insert(0,'до порога включительно',upper)
            kept.insert(0,'interactions >',lower)
            membership.append(kept)
        balance.extend(balance_rows(pairs,reference,raw_space,feature_info,lower,upper))
        for method,estimate in [('RAW paired',paired_raw(pairs)),('CUPED paired HC3',paired_cuped(pairs,pre_scale))]:
            records.append(effect_row(estimate,pairs,method,lower,upper))
    effects_unique=pd.DataFrame(records)
    effects_unique['p Holm по уникальным составам']=np.nan
    for method,group in effects_unique.groupby('метод',sort=False):
        effects_unique.loc[group.index,'p Holm по уникальным составам']=holm(group['p-value nominal'])
    if expanded:
        repeated=effects_unique.loc[effects_unique.index.repeat(
            effects_unique['до порога включительно']-effects_unique['interactions >']+1)].copy()
        offset=repeated.groupby(level=0).cumcount()
        repeated['interactions >']=repeated['interactions >']+offset
        repeated['до порога включительно']=repeated['interactions >']
        effects=repeated.sort_values(['interactions >','метод']).reset_index(drop=True)
    else:
        effects=effects_unique.copy()
    summary=pd.DataFrame([
        ('Пре-период',', '.join(PRE_MONTHS)),('Период дохода',', '.join(TEST_MONTHS)),
        ('active должен относиться к периоду дохода','Временных меток в active нет; автоматическая проверка невозможна'),
        ('Пользователей из treatment',len(users)),('Доступных control',len(controls)),
        ('Людей active вне AB',int(activity['group'].isna().sum())),
        ('Control с interactions>0',int((population['group'].eq('control') & population['interactions'].gt(0)).sum())),
        ('Исключать использовавших control',EXCLUDE_ACTIVE_CONTROLS),
        ('Пользователи treatment без pre','ошибка; пользователи незаметно не исключаются'),
        ('Control без pre',int((population['group'].eq('control') & ~population['has_pre']).sum())),
        ('Дубликаты active',ACTIVE_DUPLICATE_RULE),('Признаки matching',json.dumps(MATCH_FEATURES,ensure_ascii=False)),
        ('Точные ограничения matching',json.dumps(EXACT_MATCH_ON,ensure_ascii=False)),
        ('matching','минимальная сумма евклидовых расстояний; 1:1 без повторений'),
        ('Перематчивать каждый состав',REMATCH_EACH_THRESHOLD),('MAX_MATCH_DISTANCE',str(MAX_MATCH_DISTANCE)),
        ('Стандартизация','фиксирована на pre всех пользователей и допустимых control'),
        ('Запись порогов','все целые' if expanded else 'интервалы неизменных составов'),
        ('Порог','interactions > k; upper — последний целый порог с тем же составом'),
        ('RAW','парный t-тест разниц доходов'),('CUPED','регрессия разниц D_Y на НЕцентрированную D_X, HC3'),
        ('MIN_PAIRS для p/MDE',MIN_PAIRS),('alpha',ALPHA),('power',POWER),
        ('Масштаб суммы','число пользователей = число пар; не суммарное число T+C'),
        ('MDE','noncentral-t при фиксированных SE и df; для HC3 приближение'),
        ('Holm','отдельно по методам, по разным составам; повторяющиеся целые пороги не дублируют проверки'),
        ('Интерпретация','диагностическое сравнение выбранных пользователей; matching не устраняет отбор по использованию'),
    ],columns=['параметр','значение'])
    return {'effects':effects,'effects_unique':effects_unique,'pairs':initial_pairs,
            'membership':pd.concat(membership,ignore_index=True) if membership else pd.DataFrame(),
            'balance':pd.DataFrame(balance),'features':feature_info,'roster':population,
            'active_audit':activity,'deal_audit':deal_audit,'parameters':summary}


def save_report(tables,out_dir=OUT_DIR):
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/'active_matching_effects.xlsx'
    sheets={'effects':'Эффект_и_MDE','effects_unique':'Уникальные_составы','pairs':'Исходные_пары',
            'membership':'Пары_по_порогам','balance':'Баланс_после_matching','features':'Масштабы_признаков',
            'roster':'Список_AB','active_audit':'Сверка_active_и_AB','deal_audit':'Пропуски_сделок','parameters':'Настройки'}
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for key,title in sheets.items():
            tables[key].to_excel(writer,sheet_name=title,index=False)
    return path


def main(active=None,dgr=None,ab=None):
    if active is None:
        if ACTIVE_FILE is None:
            raise ValueError("Передайте DataFrame: tables = main(active=active), либо задайте ACTIVE_FILE")
        active=pd.read_excel(ACTIVE_FILE)
    if dgr is None:
        dgr=pd.read_excel(DATA_FILE)
    if ab is None:
        if FINAL_GROUPS_FILE:
            ab=pd.read_excel(FINAL_GROUPS_FILE,sheet_name='Итоговый_состав').rename(columns={'sale_name':AB_NAME,'group':AB_GROUP})
        else:
            ab=pd.read_excel(AB_FILE,sheet_name=AB_SHEET)
    tables=run_analysis(dgr,ab,active)
    cols=['interactions >','до порога включительно','метод','n_pairs','эффект total treatment, руб.',
          'MDE total treatment, прибл.','эффект / MDE','p-value nominal','статус']
    try:
        from IPython.display import display
        display(tables['effects'][cols])
    except ImportError:
        print(tables['effects'][cols].to_string(index=False))
    print('Файл сохранён:',save_report(tables).resolve())
    return tables


if __name__=='__main__':
    tables=main(active=globals().get('active'))
