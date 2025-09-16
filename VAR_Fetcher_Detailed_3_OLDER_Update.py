import os, time, shutil, math, hashlib
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
LOGFILE   = "kpi_log_3.csv"
GAPS_OUT  = "kpi_log_3_gaps.csv"
STATS_OUT = "kpi_log_3_stats_by_wip.csv"

# Machines per WS (your setup)
M_LIST = [3, 2, 3]  # WS1, WS2, WS3

# Columns whose values are in seconds and should be converted to hours
SEC_COLS = ["WS1","WS1_F","WS2","WS2_F","WS3","WS3_F","AGV1","AGV1_F","AGV2","AGV2_F"]

# KPI columns we compute stats for (we’ll auto-filter to the ones that exist)
CT_COLS_WANTED = [
    "CT_WS1", "WS1", "WS1_F", "WS2", "WS2_F", "CT_WS2", "CT_WS3", "WS3", "WS3_F",
    "CT_P_WS1", "CT_P_WS2", "CT_P_WS3", "CT_AGV1", "AGV1", "AGV1_F",
    "CT_AGV2", "AGV2", "AGV2_F", "CT_P_AGV1", "CT_P_AGV2"
]

# --- NEW: make theory columns less empty ---
# 1) Use fallback utils when UtilizationStation* are missing
USE_FALLBACK_UTILS = True
FALLBACK_UTILS = [0.70, 0.85, 0.60]  # [u1,u2,u3] if not present in CSV

# 2) Proxy-fill theory for cycle time metrics (CT_*) using corresponding service-time CVs
FILL_CT_WITH_SERVICE_CV_PROXY = True

# Optional AGV service-time theory
#   None -> leave AGV service theory as NaN
#   0.0  -> deterministic AGV service
#   1.0  -> exponential AGV service
AGV_SCV = 0.0

PROPAGATION_ROUNDS = 30

# =========================
# UTIL: wait & snapshot
# =========================
def wait_for_stable_file(path, min_bytes=1, stable_checks=3, interval=0.5, timeout=30):
    start = time.time()
    last_size = -1
    steady = 0
    while True:
        if time.time() - start > timeout:
            print(f"[WARN] Timeout waiting for stable CSV: {path}")
            return False
        if not os.path.exists(path):
            time.sleep(interval); continue
        sz = os.path.getsize(path)
        if sz >= min_bytes and sz == last_size:
            steady += 1
            if steady >= stable_checks:
                return True
        else:
            steady = 0
            last_size = sz
        time.sleep(interval)

def snapshot_csv(path):
    base, ext = os.path.splitext(path)
    snap = f"{base}.__snapshot__{ext or '.csv'}"
    shutil.copyfile(path, snap)
    return snap

def safe_read_csv(path):
    if not wait_for_stable_file(path, min_bytes=2):
        raise FileNotFoundError(f"CSV not stable or missing: {path}")
    snap = snapshot_csv(path)
    try:
        return pd.read_csv(snap, engine="python", on_bad_lines="skip")
    finally:
        try: os.remove(snap)
        except Exception: pass

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def short_id(text):
    if pd.isna(text): return "none"
    s = str(text)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def norm_util(u):
    """Clamp/util normalize: if looks like % (>1.5), convert to fraction."""
    try:
        u = float(u)
    except Exception:
        return 0.0
    if u > 1.5:  # likely percent
        u = u / 100.0
    return min(max(u, 0.0), 0.999)

# =========================
# THEORY: service-time SCV
# =========================
def scv_normal(mu_s, sigma_s):  return (sigma_s / mu_s) ** 2
def scv_exponential():           return 1.0
def scv_uniform(a_s, b_s):
    mean = 0.5 * (a_s + b_s)
    std  = (b_s - a_s) / math.sqrt(12.0)
    return (std / mean) ** 2

# textbook (times in seconds; unit cancels)
WS1_scv = scv_normal(mu_s=20.0, sigma_s=5.0)   # ≈0.0625 (CV≈0.25)
WS2_scv = scv_exponential()                    # 1.0
WS3_scv = scv_uniform(a_s=10.0, b_s=35.0)      # ≈0.1029 (CV≈0.321)
WS1_cv, WS2_cv, WS3_cv = math.sqrt(WS1_scv), math.sqrt(WS2_scv), math.sqrt(WS3_scv)

# =========================
# PROPAGATION WS1->WS2->WS3
# =========================
def _cd2_single(ca2, ce2, rho, m, g_fun=lambda mm: 1.0/np.sqrt(mm)):
    ca2 = max(0.0, float(ca2))
    ce2 = max(0.0, float(ce2))
    rho = norm_util(rho)
    m   = max(1, int(m))
    cd2 = 1.0 + (1.0 - rho**2) * (ca2 - 1.0) + (rho**2) * g_fun(m) * (ce2 - 1.0)
    return max(0.0, cd2)

def propagate_rounds_ws1_ws2_ws3(util_ws, service_scv, m_list, n_rounds=30, ca1_init=0.0):
    u1, u2, u3 = [norm_util(x) for x in util_ws]
    ce1, ce2, ce3 = service_scv
    m1, m2, m3 = m_list
    ca1 = float(ca1_init)
    final = None
    for _ in range(int(n_rounds)):
        cd1 = _cd2_single(ca1, ce1, u1, m1)
        ca2 = cd1
        cd2 = _cd2_single(ca2, ce2, u2, m2)
        ca3 = cd2
        cd3 = _cd2_single(ca3, ce3, u3, m3)
        final = {"WS1":{"ca2":ca1,"cd2":cd1},
                 "WS2":{"ca2":ca2,"cd2":cd2},
                 "WS3":{"ca2":ca3,"cd2":cd3}}
        ca1 = cd3
    return final

def filter_with_queue(ca2, ce2, rho, m=1):
    return _cd2_single(ca2, ce2, rho, m)

# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] CWD: {os.getcwd()}")
    print(f"[INFO] Reading: {LOGFILE}")

    df = safe_read_csv(LOGFILE)
    df.columns = df.columns.str.strip()
    print("[INFO] Columns:", df.columns.tolist())

    if df.empty:
        print("[WARN] Input CSV has no rows. Writing empty outputs.")
        pd.DataFrame().to_csv(GAPS_OUT, index=False)
        pd.DataFrame().to_csv(STATS_OUT, index=False)
        return

    # seconds -> hours
    for col in SEC_COLS:
        if col in df.columns:
            df[col] = to_num(df[col]) / 3600.0

    # Grouping keys
    group_cols = [c for c in ["WIP", "NLP_D"] if c in df.columns]
    if not group_cols:
        df["_DUMMY_GROUP"] = 0
        group_cols = ["_DUMMY_GROUP"]
        print("[WARN] No WIP/NLP_D found; using single group.")

    # Which KPI columns exist?
    ct_cols = [c for c in CT_COLS_WANTED if c in df.columns]
    if not ct_cols:
        print("[WARN] None of the desired KPI columns are present.")
        df.to_csv(GAPS_OUT, index=False)
        pd.DataFrame().to_csv(STATS_OUT, index=False)
        return

    # Non-CT columns for gap transform
    ct_cols_auto = [c for c in df.columns if c.startswith("CT")]
    non_ct_cols = [c for c in df.columns if c not in ct_cols_auto + group_cols + ["Products_Finished"]]

    # Gap transform
    def gap_transform(group):
        if "Products_Finished" in group.columns:
            group = group.sort_values("Products_Finished").reset_index(drop=True)
        else:
            group = group.reset_index(drop=True)
        for col in non_ct_cols:
            if col in group.columns:
                series = to_num(group[col])
                group[col] = series.diff().fillna(series)
        return group

    df_gap = df.groupby(group_cols, group_keys=False).apply(gap_transform)
    df_gap.to_csv(GAPS_OUT, index=False)
    print(f"[OK] Saved gaps: {os.path.abspath(GAPS_OUT)}")

    # Stats + Theory
    all_blocks = []

    have_utils_cols = all(col in df_gap.columns for col in ["UtilizationStation1","UtilizationStation2","UtilizationStation3"])

    for keys, group in df_gap.groupby(group_cols):
        key_dict = {}
        if isinstance(keys, tuple):
            for k,v in zip(group_cols, keys): key_dict[k]=v
        else:
            key_dict[group_cols[0]] = keys

        # Base theory arrays aligned to ct_cols
        theo_cv  = [np.nan]*len(ct_cols)
        theo_scv = [np.nan]*len(ct_cols)

        # Service-time theory (workstations)
        service_map = {
            "CT_P_WS1": (WS1_cv, WS1_scv),
            "CT_P_WS2": (WS2_cv, WS2_scv),
            "CT_P_WS3": (WS3_cv, WS3_scv),
        }
        for name,(cv_val, scv_val) in service_map.items():
            if name in ct_cols:
                i = ct_cols.index(name)
                theo_cv[i]  = cv_val
                theo_scv[i] = scv_val

        # Service-time theory (AGV)
        if AGV_SCV is not None:
            agv_cv = math.sqrt(AGV_SCV)
            for name in ["CT_P_AGV1","CT_P_AGV2"]:
                if name in ct_cols:
                    i = ct_cols.index(name)
                    theo_cv[i]  = agv_cv
                    theo_scv[i] = AGV_SCV

        # Utilizations (from CSV or fallback)
        if have_utils_cols:
            u1 = norm_util(to_num(group["UtilizationStation1"]).mean())
            u2 = norm_util(to_num(group["UtilizationStation2"]).mean())
            u3 = norm_util(to_num(group["UtilizationStation3"]).mean())
        else:
            if USE_FALLBACK_UTILS:
                u1,u2,u3 = FALLBACK_UTILS
            else:
                u1=u2=u3 = 0.7  # harmless default

        # Propagate WS arrivals/departures
        final = propagate_rounds_ws1_ws2_ws3(
            util_ws=[u1,u2,u3],
            service_scv=[WS1_scv,WS2_scv,WS3_scv],
            m_list=M_LIST,
            n_rounds=PROPAGATION_ROUNDS,
            ca1_init=0.0
        )
        # Map arrivals/departures
        mapping = {
            "WS1":("WS1","ca2"), "WS1_F":("WS1","cd2"),
            "WS2":("WS2","ca2"), "WS2_F":("WS2","cd2"),
            "WS3":("WS3","ca2"), "WS3_F":("WS3","cd2"),
        }
        for col,(st,fld) in mapping.items():
            if col in ct_cols:
                i = ct_cols.index(col)
                scv_val = float(final[st][fld])
                theo_scv[i] = scv_val
                theo_cv[i]  = math.sqrt(scv_val)

        # AGV arrival/dep theory (rough, using WS deps as arrivals + AGV queue filter)
        if AGV_SCV is not None:
            agv_util = norm_util(to_num(group.get("AGVs_Utilization", pd.Series([0.5]))).mean())
            approx_in = {
                "AGV1":   ("WS1","cd2"), "AGV1_F": ("WS1","cd2"),
                "AGV2":   ("WS2","cd2"), "AGV2_F": ("WS2","cd2"),
            }
            for col,(from_ws,fld) in approx_in.items():
                if col in ct_cols:
                    i = ct_cols.index(col)
                    ca_in = float(final[from_ws][fld])
                    scv_out = ca_in if not col.endswith("_F") else filter_with_queue(ca_in, AGV_SCV, agv_util, m=1)
                    theo_scv[i] = scv_out
                    theo_cv[i]  = math.sqrt(scv_out)

        # Optional: proxy-fill cycle-time metrics CT_WS* and CT_AGV* with service CVs
        if FILL_CT_WITH_SERVICE_CV_PROXY:
            proxy_map = {
                "CT_WS1": WS1_scv, "CT_WS2": WS2_scv, "CT_WS3": WS3_scv,
            }
            for name, scv_ in proxy_map.items():
                if name in ct_cols:
                    i = ct_cols.index(name)
                    if pd.isna(theo_scv[i]):
                        theo_scv[i] = scv_
                        theo_cv[i]  = math.sqrt(scv_)
            if AGV_SCV is not None:
                for name in ["CT_AGV1","CT_AGV2"]:
                    if name in ct_cols:
                        i = ct_cols.index(name)
                        if pd.isna(theo_scv[i]):
                            theo_scv[i] = AGV_SCV
                            theo_cv[i]  = math.sqrt(AGV_SCV)

        # Empirical stats
        rows = []
        for c in ct_cols:
            series = to_num(group[c]) if c in group.columns else pd.Series(dtype=float)
            mean_v = float(series.mean()) if len(series)>0 else np.nan
            std_v  = float(series.std())  if len(series)>0 else np.nan
            if not (pd.isna(mean_v) or mean_v == 0):
                cv_v  = std_v / mean_v
                scv_v = cv_v**2
            else:
                cv_v = scv_v = 0.0
            rows.append({**key_dict, "metric": c, "mean": mean_v, "std": std_v, "cv": cv_v, "scv": scv_v})

        block = pd.DataFrame(rows)
        block["theory_cv"]  = theo_cv
        block["theory_scv"] = theo_scv

        # compact label (keep NLP_D as its own column)
        def _mk_metric_label(row):
            parts = [row.get("metric","")]
            if "WIP" in row: parts.append(f"WIP={row['WIP']}")
            if "NLP_D" in row: parts.append(f"NLP={short_id(row['NLP_D'])}")
            return "_".join(parts)
        block["metric_label"] = block.apply(_mk_metric_label, axis=1)

        all_blocks.append(block)

    result = pd.concat(all_blocks, ignore_index=True)

    # Diagnostics: how many theory cells were filled?
    total = len(result)
    filled = total - int(result["theory_scv"].isna().sum())
    print(f"[diag] theory cells filled: {filled}/{total} rows "
          f"({filled/total*100:.1f}%).  NaNs remain where theory is undefined or explicitly disabled.")

    # Write outputs
    df_gap.to_csv(GAPS_OUT, index=False)
    result.to_csv(STATS_OUT, index=False, float_format="%.6f")
    print(f"[OK] Saved gaps -> {os.path.abspath(GAPS_OUT)}")
    print(f"[OK] Saved stats -> {os.path.abspath(STATS_OUT)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[FATAL] Unhandled error:", e)
        traceback.print_exc()
