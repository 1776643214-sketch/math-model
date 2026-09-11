from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pulp

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter
from scipy import stats


# =============================================================================
# 路径配置
# =============================================================================
ATTACH1 = r"C:\Users\asus\Desktop\附件\附件1.xlsx"
# 输出目录：默认写到脚本同级的 outputs/；若需要覆盖桌面的交付目录，改这里即可。
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")


# =============================================================================
# 第 0 部分  全局参数
# =============================================================================
DT = 1.0 / 6.0          # 时间步长 (h)，10 min
T = 144                 # 一天 144 个时段
P_MAX = 5000.0          # 最大充放电功率 (kW)  —— 附录1
E_MAX = 12000.0         # 储能最大容量 (kWh) —— 附录1
E_LO = 1200.0           # SOC 运行下限 (kWh)
E_HI = 10800.0          # SOC 运行上限 (kWh)
S0 = 6000.0             # 0:00 初始储电量 (kWh)
ETA_C = 0.90            # 充电效率
ETA_D = 0.90            # 放电效率
S_TERMINAL = 6000.0     # 题目要求：0:00 与 24:00 储电量相同（Q1 硬约束）

# ── 全篇统一的下采样口径（Q1/Q2/Q3/Q4 共用，禁止各自另立标准）──────────────
# 'zoh' 零阶保持：忠实于「小时平均功率」的预报语义，不引入原始数据中不存在的
# 平滑取值，因而不会人为改善 Q2 回测的预测精度。正式结果一律以此口径生成。
DOWNSAMPLE_METHOD = "zoh"


def default_params(**over):
    """返回一份参数字典，over 中的键覆盖默认值。"""
    p = dict(
        dt=DT, T=T,
        p_max=P_MAX,
        e_max=E_MAX,
        e_lo=E_LO,
        e_hi=E_HI,
        eta_c=ETA_C,
        eta_d=ETA_D,
        s0=S0,
        s_terminal=S_TERMINAL,   # 题目要求 24:00 储电量 = 0:00 储电量 = 6000
        k_bat=0.0,               # 电池折旧系数 元/kWh；0 = 题目口径，0.15 = 工程口径
        downsample=DOWNSAMPLE_METHOD,
    )
    p.update(over)
    return p


# =============================================================================
# 第 1 部分  数据读取与预处理
# =============================================================================
def load_attachment1(path: str) -> pd.DataFrame:
    """
    读取附件1：时间 / 电价(元/kWh) / 小区负载(kW) / 光伏发电预测功率(kW)。
    返回 144 行，索引 0..143 对应 [0:00,0:10) ... [23:50,24:00)。
    附件1 的 '0:00+1' 行是当天 24:00 瞬时断面，不作为调度时段。
    """
    df = pd.read_excel(path, header=None)
    df = df.iloc[1:, :4].reset_index(drop=True)
    df.columns = ["time", "price", "load", "pv"]
    for c in ["price", "load", "pv"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["price", "load"]).reset_index(drop=True)
    if len(df) > T:
        df = df.iloc[:T].copy()
    assert len(df) == T, f"附件1 时段数应为 {T}，实际 {len(df)}"
    return df


def load_attachment2(path: str, day_index: int = 0):
    """读取附件2 某一天的小区负载(kW) 与 光伏实际功率(kW)，各 144 点（Q2 用）。"""
    ld = pd.read_excel(path, sheet_name="小区负载", header=None)
    pv = pd.read_excel(path, sheet_name="光伏发电实际功率", header=None)
    ld = ld.iloc[1:, 1:].astype(float)
    pv = pv.iloc[1:, 1:].astype(float)
    return ld.iloc[day_index, :T].to_numpy(float), pv.iloc[day_index, :T].to_numpy(float)


def hour_to_10min(hourly, method=None):
    """
    小时级预报(24整点) → 10 min 网格(144点) 的通用下采样函数。

    'zoh'   零阶保持／阶梯保持 —— 第 h 小时的值覆盖 [h:00, h+1:00) 共 6 个槽。
            这是「预报语义」最忠实的做法：预报给出的是该小时的平均功率。
    'linear' 线性插值。
    'spline' 三次样条插值（自然边界），曲线光滑，适合可视化。

    说明：Q1 主模型直接使用附件1 的 144 点完美数据；本函数是为 Q2/Q3 与
          10 分钟离散化框架对齐而准备的统一入口，保证全篇口径一致。
    """
    if method is None:
        method = DOWNSAMPLE_METHOD
    hourly = np.asarray(hourly, float).reshape(-1)
    assert hourly.size == 24, "小时级预报应为 24 个整点"
    if method == "zoh":
        return np.repeat(hourly, 6)
    xh = np.arange(24, dtype=float)
    xg = np.arange(T, dtype=float) / 6.0
    if method == "linear":
        return np.interp(xg, xh, hourly)
    if method == "spline":
        from scipy.interpolate import CubicSpline
        ext_x = np.concatenate([[-1.0], xh, [24.0]])
        ext_y = np.concatenate([[hourly[-1]], hourly, [hourly[0]]])
        cs = CubicSpline(ext_x, ext_y, bc_type="natural")
        return np.clip(cs(xg), 0.0, None)
    raise ValueError(f"未知 method: {method}")


# =============================================================================
# 第 2 部分  统一底层 MILP：solve_day（Q1~Q4 共用）
# =============================================================================
def solve_day(price, load, pv, params=None, solver_msg=False,
              objective="cost", eps_cost=None):
    """
    单日微网日前调度模型。

    min  J = Σ_t p_t * G_t * Δt                      （购电费, 元）
    s.t. G_t + PV_t + D_t = L_t + C_t + E_t          （功率平衡, kW）
         S_{t+1} = S_t + η_c C_t Δt − D_t Δt / η_d   （SOC 递推, kWh）
         1200 ≤ S_t ≤ 10800, 0 ≤ C_t,D_t ≤ P_max
         C_t ≤ P_max z_t, D_t ≤ P_max (1−z_t), z_t ∈ {0,1}
         E_t ≥ 0（弃光）, S_0 = 6000, S_T = S_0（题目要求首末储电量相同）

    ── 关于弃光 E_t 的说明 ─────────────────────────────────────────────────
    PV_t 是外部给定的常数数组，**不是决策变量**，不可被模型削减；
    E_t 是纯粹的松弛变量（slack），仅当「光伏盈余功率 > 可用吸收能力」时才被迫为正。
    本算例光伏盈余峰值仅 2117.5 kW，故当 P_max=5000 kW 时弃光必然为 0——
    若把 P_max 降到 2000/1500/800 kW，弃光立刻增至 61/714/2765 kWh，
    验证松弛逻辑正常（见 diagnose_curtailment() 的解析校验）。

    ── 电池折旧（工程口径，可选）────────────────────────────────────────
    k_bat > 0 时目标函数追加 Σ k_bat·D_t·Δt（放电量计提折旧）。
    提交用的 result1.xlsx 一律采用 k_bat = 0（题目口径，零合规风险）；
    k_bat = 0.15 仅用于论文正文的工程口径对比。

    objective='curtail' 时启用 ε-约束：min Σ E_t Δt  s.t.  J ≤ eps_cost
    """
    params = default_params() if params is None else params
    dt, Tn = params["dt"], params["T"]
    pmax, e_max = params["p_max"], params["e_max"]
    e_lo, e_hi = params["e_lo"], params["e_hi"]
    eta_c, eta_d = params["eta_c"], params["eta_d"]
    s0, s_T = params["s0"], params["s_terminal"]
    k_bat = float(params.get("k_bat", 0.0))

    price = np.asarray(price, float)
    load = np.asarray(load, float)
    pv = np.asarray(pv, float)          # ← 常数参数，不可调控
    idx = range(Tn)

    m = pulp.LpProblem("Microgrid_Day_Ahead", pulp.LpMinimize)
    G = pulp.LpVariable.dicts("G", idx, lowBound=0)                   # 购电功率 kW
    C = pulp.LpVariable.dicts("C", idx, lowBound=0, upBound=pmax)     # 充电功率 kW
    D = pulp.LpVariable.dicts("D", idx, lowBound=0, upBound=pmax)     # 放电功率 kW
    Ed = pulp.LpVariable.dicts("E", idx, lowBound=0)                  # 弃光松弛 kW
    S = pulp.LpVariable.dicts("S", range(Tn + 1),
                              lowBound=e_lo, upBound=e_hi)            # 储电量 kWh
    z = pulp.LpVariable.dicts("z", idx, cat="Binary")                 # 充放互斥

    buy_cost = pulp.lpSum(price[t] * G[t] * dt for t in idx)
    wear_cost = pulp.lpSum(k_bat * D[t] * dt for t in idx)            # 电池折旧
    if objective == "cost":
        m += buy_cost + wear_cost
    elif objective == "curtail":
        m += pulp.lpSum(Ed[t] * dt for t in idx)
        if eps_cost is not None:
            m += buy_cost <= eps_cost
    else:
        raise ValueError(f"未知 objective: {objective}")

    m += S[0] == s0
    for t in idx:
        m += G[t] + pv[t] + D[t] == load[t] + C[t] + Ed[t]
        m += C[t] <= pmax * z[t]
        m += D[t] <= pmax * (1 - z[t])
        m += S[t + 1] == S[t] + eta_c * C[t] * dt - D[t] * dt / eta_d
    if s_T is not None:
        m += S[Tn] == s_T

    status = m.solve(pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=120))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"求解失败: {pulp.LpStatus[status]}")

    Gv = np.array([G[t].value() for t in idx], float)
    Cv = np.array([C[t].value() for t in idx], float)
    Dv = np.array([D[t].value() for t in idx], float)
    Ev = np.array([Ed[t].value() for t in idx], float)
    Sv = np.array([S[i].value() for i in range(Tn + 1)], float)

    return dict(
        G=Gv, C=Cv, D=Dv, E=Ev, S=Sv, price=price, load=load, pv=pv,
        J=float(np.sum(price * Gv * dt)),          # 纯购电费（题目口径）
        J_wear=float(np.sum(k_bat * Dv * dt)),     # 折旧成本
        J_total=float(np.sum(price * Gv * dt) + np.sum(k_bat * Dv * dt)),
        E_buy=float(np.sum(Gv * dt)),
        E_charge=float(np.sum(Cv * dt)),
        E_discharge=float(np.sum(Dv * dt)),
        E_curtail=float(np.sum(Ev * dt)),
        S0=float(Sv[0]), S_end=float(Sv[-1]),
        balance_residual=float(abs(
            np.sum((Gv + pv + Dv - load - Cv - Ev) * dt))),
        # 等效循环次数：标准定义 = 全天放电量 / 额定容量（E_max）
        n_cycles=float(np.sum(Dv * dt) / params["e_max"]),
        # 携带求解参数，供 export_result1() 做口径隔离断言（K=0 校验）
        params=dict(params),
    )


def diagnose_curtailment(df, pmax):
    """
    弃光的解析校验：逐一列出「盈余功率超过 P_max」的瞬时尖峰，
    给出理论弃光量，用于与模型输出比对（证明 Ed 松弛变量工作正常）。
    返回 (理论弃光 kWh, 明细 DataFrame)
    """
    pv = df["pv"].to_numpy(float)
    load = df["load"].to_numpy(float)
    surplus = pv - load
    rows = []
    for i in np.where(surplus > pmax)[0]:
        over = surplus[i] - pmax
        rows.append(dict(idx=int(i),
                         time=f"{i//6}:{(i%6)*10:02d}",
                         surplus_kW=round(float(surplus[i]), 2),
                         over_kW=round(float(over), 2),
                         curtail_kWh=round(float(over * DT), 4)))
    return float(sum(r["curtail_kWh"] for r in rows)), pd.DataFrame(rows)


def solve_q1_baseline(df, params=None, eps=0.005):
    """Q1 主模型：① 最小购电费  ② ε-约束下最小弃光。"""
    params = default_params() if params is None else params
    p, l, v = (df["price"].to_numpy(float), df["load"].to_numpy(float),
               df["pv"].to_numpy(float))
    r1 = solve_day(p, l, v, params, objective="cost")
    r2 = solve_day(p, l, v, params, objective="curtail",
                   eps_cost=(1 + eps) * r1["J"])
    return r1, r2, r1["J"]


# =============================================================================
# 第 3 部分  灵敏度分析
# =============================================================================
def sensitivity_emax(df, values=(6000, 9000, 12000, 15000, 18000), base_params=None):
    """
    ① 储能容量敏感性。
    附录1 给出容量 12000 kWh、运行区间 1200~10800（上限 = 0.9×容量），
    故容量变化时同步令 e_hi = 0.9·E_max，并保证 e_hi ≥ S_0（否则初始 SOC 越限不可行）。
    """
    rows = []
    s0 = (base_params or default_params())["s0"]
    for em in values:
        p = default_params() if base_params is None else dict(base_params)
        em = float(em)
        e_hi = min(max(0.9 * em, s0, 1500.0), em)
        p["e_max"], p["e_hi"] = em, e_hi
        r = solve_day(df["price"], df["load"], df["pv"], p)
        rows.append(dict(e_max=em, e_hi=e_hi, J=r["J"], E_buy=r["E_buy"],
                         E_curtail=r["E_curtail"], E_charge=r["E_charge"],
                         E_discharge=r["E_discharge"], S_end=r["S_end"]))
    return pd.DataFrame(rows)


def sensitivity_pmax(df, values=(2000, 3000, 4000, 5000, 6000, 8000), base_params=None):
    """② 最大充放电功率敏感性。"""
    rows = []
    for pm in values:
        p = default_params() if base_params is None else dict(base_params)
        p["p_max"] = float(pm)
        r = solve_day(df["price"], df["load"], df["pv"], p)
        rows.append(dict(p_max=pm, J=r["J"], E_buy=r["E_buy"],
                         E_curtail=r["E_curtail"], E_charge=r["E_charge"],
                         E_discharge=r["E_discharge"]))
    return pd.DataFrame(rows)


def sensitivity_eta(df, values=(0.70, 0.80, 0.90, 0.95, 1.00), base_params=None):
    """③ 储能充放电效率敏感性（充放同效率）。"""
    rows = []
    for e in values:
        p = default_params() if base_params is None else dict(base_params)
        p["eta_c"] = p["eta_d"] = float(e)
        r = solve_day(df["price"], df["load"], df["pv"], p)
        rows.append(dict(eta=e, J=r["J"], E_buy=r["E_buy"],
                         E_curtail=r["E_curtail"], E_charge=r["E_charge"],
                         E_discharge=r["E_discharge"]))
    return pd.DataFrame(rows)


def sensitivity_price(df, alphas=(0.7, 0.85, 1.0, 1.15, 1.3), base_params=None):
    """④ 电价水平敏感性  p'_t = α·p_t（正文以文字描述，不单独出图）。"""
    rows = []
    for a in alphas:
        p = default_params() if base_params is None else dict(base_params)
        r = solve_day(a * df["price"], df["load"], df["pv"], p)
        rows.append(dict(alpha=a, J=r["J"], E_buy=r["E_buy"],
                         E_curtail=r["E_curtail"], E_charge=r["E_charge"],
                         E_discharge=r["E_discharge"]))
    return pd.DataFrame(rows)


def storage_value(df, base_params=None):
    """
    储能经济价值  V_ESS = J_noESS − J_ESS
    及边际收益 ΔJ/ΔE_max 的离散估计。
    """
    p = default_params() if base_params is None else dict(base_params)
    p_no = dict(p); p_no["p_max"] = 0.0          # 无储能：充放电功率上限置 0
    r_no = solve_day(df["price"], df["load"], df["pv"], p_no)
    r_yes = solve_day(df["price"], df["load"], df["pv"], p)
    tab = sensitivity_emax(df, (6000, 9000, 12000, 15000, 18000), p)
    dJ = -np.diff(tab["J"].to_numpy())
    dE = np.diff(tab["e_max"].to_numpy())
    return dict(
        J_noESS=r_no["J"], J_ESS=r_yes["J"],
        V_ESS=r_no["J"] - r_yes["J"],
        V_ESS_ratio=(r_no["J"] - r_yes["J"]) / r_no["J"] if r_no["J"] else np.nan,
        marginal_df=pd.DataFrame(dict(e_max=tab["e_max"][1:], dJ=dJ, dE=dE,
                                      marginal=dJ / dE)),
        no_ess=r_no, ess=r_yes,
    )


def compare_depreciation(df, k_values=(0.0, 0.15), base_params=None):
    """
    电池折旧系数对比（题目口径 K=0 vs 工程口径 K=0.15）。
    折旧计入目标后，模型会自发减少浅充浅放，等效循环次数下降。
    """
    rows = []
    for k in k_values:
        p = default_params() if base_params is None else dict(base_params)
        p["k_bat"] = float(k)
        r = solve_day(df["price"], df["load"], df["pv"], p)
        rows.append(dict(
            口径=("题目口径 K=0" if k == 0 else f"工程口径 K={k}"),
            K=k,
            J_buy=r["J"], J_wear=r["J_wear"], J_total=r["J_total"],
            E_charge=r["E_charge"], E_discharge=r["E_discharge"],
            E_curtail=r["E_curtail"], n_cycles=r["n_cycles"],
            S_end=r["S_end"]))
    return pd.DataFrame(rows)


def curtailment_scan(df, values=(800, 1200, 1500, 1800, 2000, 2500, 3000, 4000, 5000)):
    """
    弃光—功率上限扫描：验证 Ed 松弛变量随 P_max 单调变化（物理自洽性检验）。
    同时给出解析理论弃光量作对照。
    """
    rows = []
    for pm in values:
        p = default_params(); p["p_max"] = float(pm)
        r = solve_day(df["price"], df["load"], df["pv"], p)
        theo, _ = diagnose_curtailment(df, pm)
        rows.append(dict(p_max=pm, E_curtail_model=r["E_curtail"],
                         E_curtail_theory=theo, J=r["J"]))
    return pd.DataFrame(rows)


def sensitivity_terminal_soc(df, values=(None, 6000.0), base_params=None):
    """
    终端 SOC 对照：
      · 题目口径：S_T = S_0 = 6000 kWh（问题1 明文要求，主模型硬约束）
      · 对照情景：不约束终端 SOC，允许储能跨日套利
    返回表用于指出「释放该约束可额外节省多少成本」。
    """
    rows = []
    for v in values:
        p = default_params() if base_params is None else dict(base_params)
        p["s_terminal"] = None if v is None else float(v)
        r = solve_day(df["price"], df["load"], df["pv"], p)
        rows.append(dict(
            scenario=("不约束终端SOC\n(对照)" if v is None
                      else f"$S_T=S_0={p['s_terminal']:.0f}$\n(题目要求)"),
            J=r["J"], E_buy=r["E_buy"], E_curtail=r["E_curtail"], S_end=r["S_end"]))
    return pd.DataFrame(rows)


# =============================================================================
# 第 4 部分  结果导出
# =============================================================================
SEG_DEFS = [("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
            ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120),
            ("20:00-24:00", 120, 144)]


def _fmt_time_point(i: int) -> str:
    """第 i 个时段起点（i=0 → 0:00；i=144 → 0:00+1）。"""
    h, m = divmod(i * 10, 60)
    return "0:00+1" if h == 24 else f"{h}:{m:02d}"


def _fmt_time_point_end(i: int) -> str:
    """第 i 个时段终点（i=143 → 0:00+1）。"""
    h, m = divmod((i + 1) * 10, 60)
    return "0:00+1" if h == 24 else f"{h}:{m:02d}"


def export_result1(res, out_path):
    """
    按附件5 模板写出『计划购电量』与『充放电量』两个工作表。

    ⚠ 口径隔离（硬约束）：提交给竞赛的 result1.xlsx 必须且只能是题目口径，
      即 k_bat = 0（不计电池折旧）。K=0.15 的工程口径结论仅存在于论文正文，
      绝不写入本文件。此处用断言从运行时层面封死，而非仅靠注释约束。
    """
    kb = float(res.get("params", {}).get("k_bat", 0.0))
    assert abs(kb) < 1e-12, (
        f"口径隔离校验失败：result1.xlsx 只允许 K=0（题目口径），"
        f"但传入的 res 来自 k_bat={kb} 的解。工程口径结果请勿导出到提交文件。")
    if abs(float(res.get("J_wear", 0.0))) > 1e-6:
        raise AssertionError("口径隔离校验失败：K=0 口径下不应存在折旧成本 J_wear。")

    rows = [[f"{_fmt_time_point(i)}-{_fmt_time_point_end(i)}",
             round(res["G"][i] * DT, 2)] for i in range(T)]
    rows += [["全天购电量", round(res["E_buy"], 2)],
             ["全天购电费", round(res["J"], 2)]]
    sh1 = pd.DataFrame(rows, columns=["时间段", "购电量"])

    rows2 = []
    for name, a, b in SEG_DEFS:
        rows2.append([name, round(float(res["C"][a:b].sum()) * DT, 2),
                      round(float(res["D"][a:b].sum()) * DT, 2), "", ""])
    rows2[0][3], rows2[0][4] = "0:00", round(res["S"][0], 2)
    rows2[1][3], rows2[1][4] = "24:00", round(res["S"][-1], 2)
    sh2 = pd.DataFrame(rows2, columns=["时间段", "充电量", "放电量", "时刻", "储电量"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        sh1.to_excel(w, sheet_name="计划购电量", index=False)
        sh2.to_excel(w, sheet_name="充放电量", index=False)
    return out_path


def segment_table(res):
    """6 个时段的充电量/放电量/购电量/购电费。"""
    return pd.DataFrame([dict(
        时间段=n,
        充电量=round(float(res["C"][a:b].sum()) * DT, 2),
        放电量=round(float(res["D"][a:b].sum()) * DT, 2),
        购电量=round(float(res["G"][a:b].sum()) * DT, 2),
        购电费=round(float((res["G"][a:b] * res["price"][a:b]).sum()) * DT, 2))
        for n, a, b in SEG_DEFS])


def verify_energy_conservation(res):
    """
    电量守恒等式核算：
        负载总耗电量 = 光伏发电量 + 外网购电量 − 储能净充电量 − 弃光电量
    返回 (左端, 右端, 绝对误差, 相对误差)
    """
    dt = DT
    load_tot = float(res["load"].sum() * dt)
    pv_tot = float(res["pv"].sum() * dt)
    buy_tot = float(res["G"].sum() * dt)
    net_chg = float((res["C"].sum() - res["D"].sum()) * dt)   # 正=净充电
    curt = float(res["E"].sum() * dt)
    rhs = pv_tot + buy_tot - net_chg - curt
    return load_tot, rhs, abs(load_tot - rhs), abs(load_tot - rhs) / load_tot


# =============================================================================
# 第 5 部分  学术风绘图样式
# =============================================================================
FIG_DPI = 300
C1 = "#2E5C8A"   # 深蓝（主）
C2 = "#B5443A"   # 砖红（强调）
C3 = "#C88A2E"   # 土黄
C4 = "#3E7A5E"   # 墨绿
C5 = "#6B5B95"   # 紫灰
CG = "#757575"   # 灰
PV_C = "#E8B33C"  # 光伏金
MIN_FS = 9       # 打印兼容：全图最小字号

# ── 被引用的关键数值（由 main() 在运行时填充；绘图时用于标注，保证图文一致）──
KPI: dict = {}


def set_academic_style():
    """
    统一学术风 rcParams。
    中文字体：serif 族遇中文易回退成方框，故把 SimSun 放 sans-serif 首位并显式注册；
    SimSun 无 U+2212，须 axes.unicode_minus=False 用 ASCII 连字符显示负号。

    粗体问题：SimSun / SimHei 只有 weight=400 一个字重，一旦使用
    fontweight="bold" 就会触发 "Failed to find font weight bold" 并静默回退。
    解决办法：把带真粗体（msyhbd.ttc, weight=700）的 Microsoft YaHei 注册进
    matplotlib，并在 sans-serif 栈中放在 SimSun 之后作为「粗体供给者」。
    """
    for name, path in [("SimSun", r"C:\Windows\Fonts\simsun.ttc"),
                       ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
                       ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttc"),
                       ("Microsoft YaHei", r"C:\Windows\Fonts\msyhbd.ttc")]:
        if os.path.exists(path):
            try:
                mpl.font_manager.fontManager.addfont(path)
            except Exception:
                pass

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimSun", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.65",
        "axes.grid": True,
        "grid.color": "#E4E4E4",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.4, "ytick.major.size": 3.4,
        "xtick.minor.size": 1.9, "ytick.minor.size": 1.9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5.0,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.dpi": 110,
    })


def _tag(ax, s, dx=-0.085, dy=1.045):
    """
    子图编号统一置于左上角。
    显式指定 fontfamily="Microsoft YaHei"（其粗体 msyhbd 已注册），
    避免落到无粗体的 SimSun 上导致编号不加粗或触发缺失字重告警。
    """
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", fontfamily="Microsoft YaHei",
            va="top", ha="left")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(png, dpi=FIG_DPI, facecolor="white")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), facecolor="white")
    plt.close(fig)
    print(f"  [图] {png}")
    return png


def _hour_ticks(step=12):
    idxs = np.arange(0, 145, step)
    return idxs, [_fmt_time_point(i).replace("+1", "") for i in idxs]


def _shade_price(ax, price):
    """电价分位背景着色（低=浅米，中=浅黄，高=浅红），略提高可辨识度。"""
    q1, q2 = np.quantile(price, [0.33, 0.66])
    for i, p in enumerate(price):
        c = "#F6E9D5" if p < q1 else ("#FBF3D5" if p < q2 else "#F7DEDB")
        ax.axvspan(i, i + 1, color=c, alpha=0.75, lw=0, zorder=0)


def _time_axis(ax, step=12, xlim=(0, T)):
    """统一的 10-min 时间轴：主刻度每 2h，次刻度每 10min。"""
    xt, xl = _hour_ticks(step)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_xlim(*xlim)
    ax.xaxis.set_minor_locator(MultipleLocator(6))


def _annot(ax, xy, text, xytext, color=C2, fs=MIN_FS, ha="left", va="top"):
    """统一的箭头注释样式。"""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=color,
                ha=ha, va=va, fontweight="bold", fontfamily="Microsoft YaHei",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                shrinkA=1, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=color, lw=0.85, alpha=0.94), zorder=6)


# =============================================================================
# 【排版原则】正文看结论，附录看过程 —— 一图一主题，绝不拼大图。
# 正文 3 张：F1 调度总览 / F2 弃光自洽性 / F3 终端SOC稳健性
# 附录 3 张：FA1 灵敏度 / FA2 价值与边际 / FA3 全时段热力图
# =============================================================================

# -----------------------------------------------------------------------------
# 正文 图 1  供需功率平衡与储能 SOC 轨迹（双联，X 轴严格对齐）
# -----------------------------------------------------------------------------
def fig1_balance_soc(res, df):
    """
    论文可用图题：图 1  基于分时电价的储能日前调度功率平衡与 SOC 轨迹
    建议正文解读：中午光伏峰值被储能全额消纳，晚高峰储能放电削减购电尖峰，
                  全天购电成本由无储能的 48,052 元降至 32,910 元（下降 31.51%）。
    """
    set_academic_style()
    price = df["price"].to_numpy(float)
    load = df["load"].to_numpy(float)
    G, C, D, S = res["G"], res["C"], res["D"], res["S"]
    x = np.arange(T)
    xs = np.arange(T + 1)

    fig = plt.figure(figsize=(10.6, 7.0))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.30, 1.0], hspace=0.155,
                           left=0.085, right=0.975, top=0.945, bottom=0.078)

    # ── (a) 供给侧功率平衡：堆叠电源 vs 负荷曲线 ─────────────────────────
    ax = fig.add_subplot(gs[0, 0]); _shade_price(ax, price)
    ax.stackplot(x, G, pv_flow := df["pv"].to_numpy(float), D,
                 labels=["外网购电 $G_t$", "光伏 $PV_t$", "储能放电 $D_t$"],
                 colors=[C1, PV_C, C4], alpha=0.93,
                 edgecolor="white", lw=0.3)
    ax.plot(x, load, color="k", lw=1.9, ls="--", label="小区负载 $L_t$")
    _time_axis(ax)
    ax.set_ylabel("功率 (kW)")
    ax.set_xlabel("")
    ax.set_ylim(0, 11500)
    ax.legend(loc="upper left", ncol=4, fontsize=MIN_FS)
    ax.set_title("(a) 供给侧功率平衡：购电 + 光伏 + 储能放电 = 负载", pad=6)
    _tag(ax, "(a)", dx=-0.075, dy=1.055)

    # 关键机理解读标注：正午光伏消纳 + 晚高峰放电顶峰
    ax.annotate("正午光伏峰值\n优先给储能充电",
                xy=(72, 6200), xytext=(40, 9900),
                fontsize=MIN_FS, color=C4, ha="left", va="top",
                fontweight="bold", fontfamily="Microsoft YaHei",
                arrowprops=dict(arrowstyle="->", color=C4, lw=1.2,
                                shrinkA=1, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=C4, lw=0.85, alpha=0.94), zorder=6)
    ax.annotate("晚高峰储能放电\n削减购电尖峰",
                xy=(126, 6400), xytext=(104, 9900),
                fontsize=MIN_FS, color=C3, ha="left", va="top",
                fontweight="bold", fontfamily="Microsoft YaHei",
                arrowprops=dict(arrowstyle="->", color=C3, lw=1.2,
                                shrinkA=1, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=C3, lw=0.85, alpha=0.94), zorder=6)
    ax.text(0.995, 0.955,
            f"全天购电量 {KPI.get('E_buy', np.nan):,.0f} kWh  |  "
            f"购电费 {KPI.get('J', np.nan):,.0f} 元",
            transform=ax.transAxes, ha="right", va="top", fontsize=MIN_FS,
            fontweight="bold", fontfamily="Microsoft YaHei",
            bbox=dict(boxstyle="round,pad=0.32", fc="#F7F9FB",
                      ec=C1, lw=0.9, alpha=0.95), zorder=6)

    # ── (b) 储能 SOC 轨迹（同一时间轴，便于上下对照）──────────────────────
    ax = fig.add_subplot(gs[1, 0]); _shade_price(ax, price)
    ax.fill_between(xs, 1200, 10800, color="#EAF2F8", alpha=0.9,
                    label="允许运行区间 [1200, 10800] kWh")
    ax.plot(xs, S, color=C1, lw=2.1, label="储电量 $S_t$", zorder=4)
    ax.axhline(6000, color=CG, ls=":", lw=1.2,
               label="初始/末端储电量 $S_0=S_T=6000$ kWh")
    ax.axhline(10800, color=C2, ls="--", lw=1.0, alpha=0.85)
    ax.axhline(1200, color=C2, ls="--", lw=1.0, alpha=0.85)
    sc = ax.scatter(xs, S, c=price[np.clip(xs, 0, T - 1)], cmap="RdYlBu_r",
                    s=10, zorder=5)
    _time_axis(ax)
    ax.set_ylabel("储电量 $S_t$ (kWh)")
    ax.set_xlabel("时间")
    ax.set_ylim(0, 13000)
    cb = fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.040)
    cb.set_label("电价 (元/kWh)", fontsize=MIN_FS)
    cb.ax.tick_params(labelsize=MIN_FS)
    ax.legend(loc="lower left", fontsize=MIN_FS - 0.3, ncol=3)
    ax.set_title("(b) 储能储电量 $S_t$ 最优轨迹（散点按电价着色）", pad=6)
    _tag(ax, "(b)", dx=-0.075, dy=1.055)
    # 充放电区间直标，避免读者自行推断
    ax.text(0.005, 0.055, "低电价时段：充电蓄能", transform=ax.transAxes,
            fontsize=MIN_FS, color=C4, fontweight="bold",
            fontfamily="Microsoft YaHei", va="bottom", ha="left")
    ax.text(0.995, 0.055, "高电价时段：放电套利", transform=ax.transAxes,
            fontsize=MIN_FS, color=C3, fontweight="bold",
            fontfamily="Microsoft YaHei", va="bottom", ha="right")

    fig.suptitle("图 1  基于分时电价的储能日前调度功率平衡与 SOC 轨迹",
                 fontsize=12.5, y=0.988)
    return _save(fig, "F1_balance_soc")


# -----------------------------------------------------------------------------
# 正文 图 2  弃光自洽性检验：弃光电量随功率上限的下降曲线（单主题，纯净折线图）
# -----------------------------------------------------------------------------
def fig2_curtailment_scan(t_scan, pmax_design=5000.0, pmax_demo=2000.0):
    """
    论文可用图题：图 2  弃光电量随储能充放电功率上限的下降曲线（松弛变量自洽性检验）
    建议正文解读：随 P_max 提升弃光由 2765 kWh 单调降至 61 kWh 并最终归零，
                  证明弃光松弛变量未被架空；基准 P_max=5000 kW 时弃光为 0 属真实结论。
    """
    set_academic_style()
    fig = plt.figure(figsize=(7.6, 5.0))
    ax = fig.add_axes([0.115, 0.135, 0.845, 0.775])

    xp = t_scan["p_max"].to_numpy(float)
    ym = t_scan["E_curtail_model"].to_numpy(float)

    ax.plot(xp, ym, color=C1, ls="-", marker="o", mfc="white", mew=1.4,
            ms=6.2, lw=1.7, zorder=4, label="模型求解弃光电量 $\\sum E_t\\Delta t$")
    ax.fill_between(xp, 0, ym, color=C1, alpha=0.13, lw=0, zorder=2)

    # 零弃光分界线
    zero_x = xp[ym <= 1e-9]
    if zero_x.size:
        ax.axvspan(zero_x.min(), xp.max() * 1.02, color="#EAF6EE",
                   alpha=0.85, lw=0, zorder=1, label="弃光 = 0（光伏全额消纳）")

    # 设计值参考线
    ax.axvline(pmax_design, color=CG, ls=":", lw=1.3, zorder=3)
    ax.text(pmax_design * 1.015, ax.get_ylim()[1] * 0.965,
            f"设计值 $P_{{max}}$={pmax_design:.0f} kW\n弃光 = 0",
            color=CG, fontsize=MIN_FS, va="top", ha="left")

    # 关键点数值直标（首点、演示点、归零点）
    def _lab(i, dx=0.0, dy=0.0, ha="center", va="bottom", c=C2):
        ax.annotate(f"{ym[i]:,.0f}", xy=(xp[i], ym[i]),
                    xytext=(xp[i] + dx, ym[i] + dy), fontsize=MIN_FS,
                    color=c, ha=ha, va=va, fontweight="bold",
                    fontfamily="Microsoft YaHei", zorder=6)

    _lab(0)
    i_demo = int(np.argmin(np.abs(xp - pmax_demo)))
    if xp[i_demo] != xp[0]:
        _lab(i_demo, dy=55)
    i_zero = int(np.argmax(ym <= 1e-9)) if np.any(ym <= 1e-9) else len(xp) - 1
    ax.annotate(f"{xp[i_zero]:.0f} kW 起弃光归零",
                xy=(xp[i_zero], 0), xytext=(xp[i_zero] + 250, ym.max() * 0.22),
                fontsize=MIN_FS, color=C4, ha="left", va="center",
                fontweight="bold", fontfamily="Microsoft YaHei",
                arrowprops=dict(arrowstyle="->", color=C4, lw=1.2,
                                shrinkA=1, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec=C4, lw=0.85, alpha=0.94), zorder=6)

    ax.set_xlabel("储能最大充放电功率 $P_{max}$ (kW)")
    ax.set_ylabel("全天弃光电量 $\\sum E_t\\Delta t$ (kWh)")
    ax.set_xlim(xp.min() * 0.93, xp.max() * 1.06)
    ax.set_ylim(-ym.max() * 0.06, ym.max() * 1.20)
    ax.legend(loc="upper right", fontsize=MIN_FS)

    max_dev = float(np.max(np.abs(t_scan["E_curtail_model"]
                                  - t_scan["E_curtail_theory"])))
    ax.text(0.015, 0.035,
            "物理含义：弃光仅在「光伏盈余功率 > $P_{max}$」的瞬时尖峰处被迫产生。\n"
            f"本算例光伏盈余峰值 2,117.5 kW；模型解与解析理论值最大偏差 {max_dev:.1e} kWh。",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=MIN_FS - 0.5,
            bbox=dict(boxstyle="round,pad=0.32", fc="#F7F9FB",
                      ec="#5B7FA6", lw=0.9, alpha=0.95), zorder=6)

    ax.set_title("图 2  弃光电量随储能充放电功率上限的下降曲线", pad=8)
    return _save(fig, "F2_curtailment_scan")


# -----------------------------------------------------------------------------
# 正文 图 3  终端 SOC 约束稳健性对比（单主题，纯净柱状图）
# -----------------------------------------------------------------------------
def fig3_terminal_soc(t_term):
    """
    论文可用图题：图 3  终端 SOC 约束稳健性对比
    建议正文解读：题目要求的 S_T=S_0 相比允许跨日套利多支出约 2,217 元（+6.74%），
                  该差额即为「日内能量平衡约束」的显式经济代价。
    """
    set_academic_style()
    fig = plt.figure(figsize=(7.6, 5.4))
    ax = fig.add_axes([0.13, 0.13, 0.84, 0.78])

    names_raw = t_term["scenario"].tolist()
    Js = t_term["J"].to_numpy(float)

    # 简化后的图例标签：去掉换行，便于阅读
    short_names = ["自由末端\n(无终端约束)", "强制归零\n($S_T=S_0=6000$)"]
    # 与 t_term 行序对齐：values=(None, 6000.0) → 第一行自由、第二行强制
    assert len(Js) == 2, "正文图3 仅展示两个情景"
    order = np.argsort(Js)  # 自由末端(便宜)在下，强制归零(贵)在上，更直观
    Js_p = Js[order]
    names_p = [short_names[i] for i in order]
    cs = [C1, C3][:len(Js_p)]

    bars = ax.bar(range(len(Js_p)), Js_p, width=0.55, color=cs,
                  alpha=0.93, edgecolor="black", lw=0.75, zorder=3)

    # 数值标签
    for r, v in zip(bars, Js_p):
        ax.text(r.get_x() + r.get_width() / 2, v * 1.012, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=MIN_FS + 1.0,
                fontweight="bold", fontfamily="Microsoft YaHei", zorder=6)

    # 差额双箭头 + 标签
    d = float(Js_p.max() - Js_p.min())
    ax.annotate("", xy=(1, Js_p[1]), xytext=(0, Js_p[0]),
                arrowprops=dict(arrowstyle="<->", color=C2, lw=1.8))
    ax.text(0.5, (Js_p[0] + Js_p[1]) / 2,
            f"差额 = {d:,.0f} 元\n(+{(Js_p[1]/Js_p[0]-1)*100:.2f}%)",
            ha="center", va="center", fontsize=MIN_FS + 0.5, color=C2,
            fontweight="bold", fontfamily="Microsoft YaHei",
            bbox=dict(boxstyle="round,pad=0.34", fc="white",
                      ec=C2, lw=0.95), zorder=6)

    ax.set_xticks(range(len(Js_p)))
    ax.set_xticklabels(names_p, fontsize=MIN_FS + 0.5)
    ax.set_ylabel("全天购电费 $J$ (元)")
    ax.set_ylim(0, Js_p.max() * 1.20)

    # 顶部解读
    ax.text(0.5, 0.965,
            "释放 S_T=S_0 约束 → 允许跨日套利 → 全天成本下降",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=MIN_FS, color=CG, style="italic",
            fontfamily="Microsoft YaHei")

    ax.set_title("图 3  终端 SOC 约束的稳健性对比", pad=10)
    return _save(fig, "F3_terminal_soc")


# -----------------------------------------------------------------------------
# 附录图 A-1  关键参数灵敏度综合分析（三联，扫参证据）
# -----------------------------------------------------------------------------
def figA1_sensitivity(t_emax, t_pmax, t_eta):
    """附录 A-1：储能容量 / 最大充放电功率 / 效率 三联灵敏度。"""
    set_academic_style()
    fig = plt.figure(figsize=(11.5, 4.0))
    gs = gridspec.GridSpec(1, 3, wspace=0.38, left=0.072, right=0.978,
                           top=0.795, bottom=0.170)

    # ---- (a) 储能容量 ----
    ax = fig.add_subplot(gs[0, 0])
    xe = t_emax["e_max"].to_numpy(float)
    l1, = ax.plot(xe, t_emax["J"], color=C1, ls="-", marker="o",
                  mfc="white", mew=1.3, ms=5.5, label="最优购电费 $J^*$（实线○）")
    ax.set_xlabel("储能容量上限 $E_{max}$ (kWh)")
    ax.set_ylabel("购电费 (元)", color=C1)
    ax.tick_params(axis="y", colors=C1, labelsize=MIN_FS)
    ax.axvline(12000, color=CG, ls=":", lw=1.3)
    ax.text(12000, ax.get_ylim()[1] * 0.995, " 设计值", color=CG,
            fontsize=MIN_FS, va="top")
    ax3 = ax.twinx(); ax3.grid(False)
    l2, = ax3.plot(xe, t_emax["E_curtail"], color=C3, ls="--", marker="s",
                   mfc="white", mew=1.3, ms=5.0, label="弃光电量（虚线□）")
    ax3.set_ylabel("弃光电量 (kWh)", color=C3)
    ax3.tick_params(axis="y", colors=C3, labelsize=MIN_FS)
    ax.legend(handles=[l1, l2], loc="upper right", fontsize=MIN_FS)
    ax.set_title("(a) 储能容量灵敏度", pad=6); _tag(ax, "(a)", dx=-0.185)

    # ---- (b) 最大充放电功率 ----
    ax = fig.add_subplot(gs[0, 1])
    xp = t_pmax["p_max"].to_numpy(float)
    ax.plot(xp, t_pmax["J"], color=C1, ls="-", marker="^", mfc="white",
            mew=1.3, ms=6.0, label="最优购电费 $J^*$")
    ax.set_xlabel("最大充放电功率 $P_{max}$ (kW)")
    ax.set_ylabel("购电费 (元)")
    ax.axvline(5000, color=CG, ls=":", lw=1.3)
    ax.text(5000, ax.get_ylim()[0], " 设计值", color=CG, fontsize=MIN_FS,
            va="bottom", ha="left")
    ax.annotate("功率饱和区\n边际收益$\\approx$0", xy=(7000, t_pmax["J"].iloc[-1]),
                xytext=(5150, t_pmax["J"].max() * 0.972),
                fontsize=MIN_FS, color=C2, ha="left",
                arrowprops=dict(arrowstyle="->", color=C2, lw=1.1))
    ax.legend(loc="lower left", fontsize=MIN_FS)
    ax.set_title("(b) 最大充放电功率灵敏度", pad=6); _tag(ax, "(b)", dx=-0.185)

    # ---- (c) 效率 ----
    ax = fig.add_subplot(gs[0, 2])
    xe2 = t_eta["eta"].to_numpy(float)
    ax.plot(xe2, t_eta["J"], color=C1, ls="-", marker="D", mfc="white",
            mew=1.3, ms=5.5, label="最优购电费 $J^*$")
    ax.set_xlabel("充放电效率 $\\eta$")
    ax.set_ylabel("购电费 (元)", color=C1)
    ax.tick_params(axis="y", colors=C1, labelsize=MIN_FS)
    ax.set_ylim(t_eta["J"].min() * 0.965, t_eta["J"].max() * 1.10)
    ax.axvline(0.90, color=CG, ls=":", lw=1.3)
    ax.text(0.90, ax.get_ylim()[0], " 设计值 0.9", color=CG,
            fontsize=MIN_FS, va="bottom", ha="left")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.fill_between(xe2, 0, t_eta["E_charge"], color=C4, alpha=0.32,
                     label="全天充电量 $\\sum C_t\\Delta t$")
    ax2.plot(xe2, t_eta["E_charge"], color=C4, ls="-.", lw=1.4)
    ax2.set_ylabel("全天充电量 (kWh)", color=C4)
    ax2.tick_params(axis="y", colors=C4, labelsize=MIN_FS)
    ax2.set_ylim(0, t_eta["E_charge"].max() * 1.28)
    ax2.text(0.035, 0.30, "阴影 = 全天充电量", transform=ax2.transAxes,
             fontsize=MIN_FS, color=C4, va="center", ha="left",
             fontweight="bold", fontfamily="Microsoft YaHei",
             bbox=dict(boxstyle="round,pad=0.28", fc="white",
                       ec=C4, lw=0.8, alpha=0.92))
    ax.legend(loc="upper left", fontsize=MIN_FS - 0.5)
    ax.set_title("(c) 储能效率灵敏度", pad=6); _tag(ax, "(c)", dx=-0.185)

    fig.suptitle("附录图 A-1  关键参数的灵敏度分析", fontsize=12.5, y=0.978)
    return _save(fig, "FA1_sensitivity")


# -----------------------------------------------------------------------------
# 附录图 A-2  储能经济价值 + 容量扩展边际收益（双联，主题归一）
# -----------------------------------------------------------------------------
def figA2_value_marginal(sv):
    """
    附录 A-2：储能经济价值 + 容量边际收益（双联）。

    说明：原图3 把「经济价值」挤在正文、把「边际收益」和「折旧」挤在附录，
    主题混杂。现重组为「经济价值 + 边际收益」双联——两者都是「储能值不值」这一
    主题下的子问题。电池折旧（工程口径 K=0.15）作为正文文字/表格讨论，不再出图。
    """
    set_academic_style()
    fig = plt.figure(figsize=(11.0, 4.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.12], wspace=0.30,
                           left=0.088, right=0.972, top=0.835, bottom=0.155)

    # ── (a) 储能经济价值：无储能 vs 配置储能 ───────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    vals = [sv["J_noESS"], sv["J_ESS"]]
    b = ax.bar(["无储能\n$P_{max}=0$", "配置储能\n$P_{max}=5000$"], vals,
               width=0.52, color=[CG, C1], alpha=0.93,
               edgecolor="black", lw=0.75, zorder=3)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v * 1.012, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=MIN_FS + 0.5,
                fontweight="bold", fontfamily="Microsoft YaHei", zorder=6)
    ax.annotate("", xy=(1, vals[1]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="<->", color=C2, lw=1.6))
    ax.text(0.5, (vals[0] + vals[1]) / 2,
            f"$V_{{ESS}}$ = {sv['V_ESS']:,.0f} 元\n成本下降 "
            f"{sv['V_ESS_ratio']*100:.2f}%",
            ha="center", va="center", fontsize=MIN_FS, color=C2,
            fontweight="bold", fontfamily="Microsoft YaHei",
            bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=C2, lw=0.95),
            zorder=6)
    ax.set_ylabel("全天购电费 $J$ (元)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("(a) 储能经济价值 $V_{ESS}$", pad=6)
    _tag(ax, "(a)", dx=-0.135, dy=1.045)

    # ── (b) 容量扩展边际收益（递减曲线）───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    md = sv["marginal_df"]
    lbl = [f"{int(a)}→{int(b)}" for a, b in zip(md["e_max"] - md["dE"], md["e_max"])]
    bars = ax.bar(range(len(md)), md["marginal"], width=0.60, color=C4,
                  alpha=0.92, edgecolor="black", lw=0.75, zorder=3)
    ax.set_xticks(range(len(md))); ax.set_xticklabels(lbl, rotation=14,
                                                      fontsize=MIN_FS)
    for r, v in zip(bars, md["marginal"]):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=MIN_FS, zorder=6)
    ax.annotate("边际收益递减", xy=(len(md) - 1, md["marginal"].iloc[-1]),
                xytext=(len(md) - 2.05, md["marginal"].max() * 0.60),
                fontsize=MIN_FS, color=C2, ha="left",
                arrowprops=dict(arrowstyle="->", color=C2, lw=1.1))
    ax.set_xlabel("容量扩展区间 (kWh)")
    ax.set_ylabel("边际收益 $\\Delta J/\\Delta E_{max}$ (元/kWh)")
    ax.set_ylim(0, md["marginal"].max() * 1.22)
    ax.set_title("(b) 容量扩展边际收益递减", pad=6)
    _tag(ax, "(b)", dx=-0.165)

    fig.suptitle("附录图 A-2  储能经济价值与容量扩展边际收益",
                 fontsize=12.5, y=0.972)
    return _save(fig, "FA2_value_marginal")


# -----------------------------------------------------------------------------
# 附录图 A-3  全时段调度变量热力图（单主题，144 个时段全景）
# -----------------------------------------------------------------------------
def figA3_heatmap(res, df):
    """
    附录 A-3：电价/负载/PV/购电/充电/放电/SOC 共 7 条曲线的归一化热力图，
    集中展示 144 个时段下各变量的强度分布。

    说明：原图把「电量守恒校验」条形图也塞在这张附录里。守恒校验是数值核算
    （残差 < 1e-6 kWh），更适合在文字/表格中给出，不适合用图表达。本附录图
    只保留热力图主题；守恒校验在 Q1_report.txt 与打印日志中保留。
    """
    set_academic_style()
    price, load, pv = (df["price"].to_numpy(float), df["load"].to_numpy(float),
                       df["pv"].to_numpy(float))
    G, C, D, S = res["G"], res["C"], res["D"], res["S"]

    fig = plt.figure(figsize=(11.5, 3.6))
    ax = fig.add_axes([0.075, 0.20, 0.90, 0.72])

    mat = np.vstack([price / price.max(), load / load.max(), pv / pv.max(),
                     G / max(G.max(), 1e-9), C / max(C.max(), 1e-9),
                     D / max(D.max(), 1e-9), S[:-1] / 10800.0])
    labels = ["电价 $p_t$", "负载 $L_t$", "光伏 $PV_t$",
              "购电 $G_t$", "充电 $C_t$", "放电 $D_t$", "储电量 $S_t$"]
    cmap = LinearSegmentedColormap.from_list(
        "ac", ["#FFFFFF", "#DCE6F1", "#9DB8D2", "#5B85B0", "#274E77"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=1)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=MIN_FS)
    xt, xl = _hour_ticks()
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=MIN_FS)
    ax.set_xlabel("时间")
    ax.grid(False)
    for k in np.arange(0.5, len(labels) - 0.5, 1):
        ax.axhline(k, color="white", lw=1.4)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
    cb.set_label("归一化幅值", fontsize=MIN_FS); cb.ax.tick_params(labelsize=MIN_FS)

    ax.set_title("附录图 A-3  全时段调度变量时序热力图", pad=10)
    return _save(fig, "FA3_heatmap")


# =============================================================================
# 主流程
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    log = []

    def L(s=""):
        print(s); log.append(str(s))

    L("=" * 78)
    L("C 题 问题一：确定性日前调度 —— 求解、灵敏度分析与图表")
    L("=" * 78)

    if not os.path.exists(ATTACH1):
        L(f"[错误] 未找到附件1：{ATTACH1}")
        L("       请修改文件顶部 ATTACH1 路径后重试。")
        return

    # ---------- 1. 数据 ----------
    df = load_attachment1(ATTACH1)
    L(f"\n[数据] 附件1：{len(df)} 个 10-min 时段 | Δt = 1/6 h")
    L(f"       电价区间 [{df['price'].min():.4f}, {df['price'].max():.4f}] 元/kWh，"
      f"均值 {df['price'].mean():.4f}")
    L(f"       负载合计 {df['load'].sum()*DT:,.2f} kWh | "
      f"光伏合计 {df['pv'].sum()*DT:,.2f} kWh")

    # ---------- 2. 主模型（题目口径：S_T = S_0 = 6000 kWh）----------
    params = default_params()
    res, res_eps, J1 = solve_q1_baseline(df, params, eps=0.005)

    L("\n" + "-" * 78)
    L("[Q1 主模型]  min  J = Σ p_t · G_t · Δt   s.t.  S_T = S_0 = 6000 kWh")
    L("-" * 78)
    L(f"  最优日前购电费 J*        = {res['J']:,.2f} 元")
    L(f"  全天购电量 ΣGΔt          = {res['E_buy']:,.2f} kWh")
    L(f"  全天充电量 ΣCΔt          = {res['E_charge']:,.2f} kWh")
    L(f"  全天放电量 ΣDΔt          = {res['E_discharge']:,.2f} kWh")
    L(f"  全天弃光量 ΣEΔt          = {res['E_curtail']:,.2f} kWh")
    L(f"  0:00 / 24:00 储电量      = {res['S0']:,.2f} / {res['S_end']:,.2f} kWh")
    L(f"  SOC 轨迹范围             = [{res['S'].min():,.2f}, {res['S'].max():,.2f}] kWh")
    L(f"  充电功率峰值 max C       = {res['C'].max():.2f} kW (上限 5000)")
    L(f"  放电功率峰值 max D       = {res['D'].max():.2f} kW (上限 5000)")
    L("  【说明】题目要求 24:00 储电量 = 0:00 储电量 = 6000 kWh，已在主模型中作为硬约束。")
    if res["E_curtail"] < 1e-6:
        L("  【说明】基准配置（P_max=5000 kW）下光伏可全额消纳；弃光约束的有效性"
          "由 P_max 扫描（图 2）另行验证，非程序 Bug。")

    KPI.update(dict(J=res["J"], E_buy=res["E_buy"], E_charge=res["E_charge"],
                    E_discharge=res["E_discharge"], E_curtail=res["E_curtail"],
                    S_end=res["S_end"], C_max=float(res["C"].max()),
                    D_max=float(res["D"].max())))

    # ---------- 3. 守恒校验 ----------
    lt, rt, err, rel = verify_energy_conservation(res)
    L("\n" + "-" * 78)
    L("[电量守恒校验] 负载 = 光伏 + 购电 - 储能净充电 - 弃光")
    L("-" * 78)
    L(f"  负载总耗电量         = {lt:,.4f} kWh")
    L(f"  光伏 {df['pv'].sum()*DT:,.4f} + 购电 {res['E_buy']:,.4f} "
      f"- 净充 {(res['E_charge']-res['E_discharge']):,.4f} "
      f"- 弃光 {res['E_curtail']:,.4f} = {rt:,.4f} kWh")
    L(f"  绝对残差 = {err:.4e} kWh    相对残差 = {rel:.4e}")
    L(f"  逐时段功率平衡残差 = {res['balance_residual']:.4e} kWh")
    L("  → 等式严格成立（残差为浮点/求解器容差量级）。")
    L("  【排版说明】守恒等式为数值核算，不另出图；附录图 A-3 仅保留热力图主题。")

    # ---------- 4. ε-约束 ----------
    L("\n[ε-约束两阶段] 购电费上浮 0.5% 内最小化弃光")
    L(f"  阶段一 经济最优 J* = {J1:,.2f} 元,  弃光 = {res['E_curtail']:,.2f} kWh")
    L(f"  阶段二 弃光最小 J  = {res_eps['J']:,.2f} 元,  弃光 = {res_eps['E_curtail']:,.2f} kWh")
    if abs(res_eps["E_curtail"] - res["E_curtail"]) < 1e-6:
        L("  → 本算例光伏可全额消纳（弃光=0），储能主要承担套利功能，ε-约束退化。")

    # ---------- 4b. 弃光松弛变量自洽性检验 ----------
    L("\n" + "-" * 78)
    L("[弃光松弛变量自洽性检验]  验证 Ed 未被架空、随 P_max 单调变化")
    L("-" * 78)
    L("  说明：PV_t 为外部常数（不可调控），Ed_t ≥ 0 是唯一松弛出口。")
    L("        弃光只在「光伏盈余功率 > P_max」的瞬时尖峰处被迫产生。")
    L(f"  本算例光伏盈余峰值 = {float((df['pv'] - df['load']).max()):,.1f} kW")
    theo2000, detail2000 = diagnose_curtailment(df, 2000)
    L(f"\n  以 P_max = 2000 kW 为例，盈余超过上限的时段明细：")
    if len(detail2000):
        L(detail2000.to_string(index=False))
    L(f"  解析理论弃光合计 = {theo2000:.4f} kWh")
    t_scan = curtailment_scan(df)
    L("\n  弃光—功率上限扫描（模型解 vs 解析理论值）：")
    L(t_scan.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    max_dev = float(np.max(np.abs(t_scan["E_curtail_model"] - t_scan["E_curtail_theory"])))
    L(f"  模型解与解析理论值最大偏差 = {max_dev:.3e} kWh  "
      f"→ {'完全一致，松弛变量工作正常' if max_dev < 1e-3 else '存在偏差，需检查'}")

    # ---------- 5. 灵敏度 ----------
    L("\n" + "-" * 78)
    L("[灵敏度分析] 围绕储能参数展开")
    L("-" * 78)
    t_emax = sensitivity_emax(df)
    t_pmax = sensitivity_pmax(df)
    t_eta = sensitivity_eta(df)
    t_price = sensitivity_price(df)

    L("\n① 储能容量 E_max")
    L(t_emax.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    L("\n② 最大充放电功率 P_max")
    L(t_pmax.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    L("\n③ 储能效率 η")
    L(t_eta.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    L("\n④ 电价水平 α（正文以文字描述，不单独出图）")
    L(t_price.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    _sl, _ic, _rv, _, _ = stats.linregress(t_price["alpha"], t_price["J"])
    dJ = -np.diff(t_pmax["J"].to_numpy()); dP = np.diff(t_pmax["p_max"].to_numpy())
    kg = dJ / dP
    L("\n[结论提炼]")
    L(f"  · 效率最敏感：η 由 0.70→1.00，购电费由 {t_eta['J'].iloc[0]:,.0f} "
      f"降至 {t_eta['J'].iloc[-1]:,.0f} 元（降幅 "
      f"{(1-t_eta['J'].iloc[-1]/t_eta['J'].iloc[0])*100:.2f}%）")
    L(f"  · 容量边际收益递减：{t_emax['J'].iloc[0]:,.0f} → "
      f"{t_emax['J'].iloc[-1]:,.0f} 元，增量收益由 {kg[0]:.4f} 元/kWh 逐步下降")
    L(f"  · 功率存在饱和阈值：P_max>5000 kW 后边际收益 {kg[-1]:.4f} 元/kW ≈ 0")
    L(f"  · 电价近似完全线性：J = {_sl:.2f}·α + {_ic:.2f}，R² = {_rv**2:.5f}")
    L("  【前提】该线性成立仅因调度策略未变，若电价畸高导致策略跳变，则线性失效。")

    # ---------- 6. 储能经济价值 ----------
    sv = storage_value(df)
    L("\n" + "-" * 78)
    L("[储能经济价值]  V_ESS = J_noESS − J_ESS")
    L("-" * 78)
    L(f"  无储能购电费 J_noESS = {sv['J_noESS']:,.2f} 元")
    L(f"  配置储能购电费 J_ESS = {sv['J_ESS']:,.2f} 元")
    L(f"  储能经济价值 V_ESS   = {sv['V_ESS']:,.2f} 元"
      f"（购电成本下降 {sv['V_ESS_ratio']*100:.2f}%）")
    L("  边际收益：")
    L(sv["marginal_df"].to_string(index=False, float_format=lambda v: f"{v:,.5f}"))
    KPI.update(dict(V_ESS=sv["V_ESS"], J_noESS=sv["J_noESS"],
                    V_ESS_ratio=sv["V_ESS_ratio"]))

    # ---------- 7. 终端 SOC 情景 ----------
    t_term = sensitivity_terminal_soc(df, values=(None, 6000.0))
    L("\n" + "-" * 78)
    L("[稳健性对照] 终端 SOC：题目要求 S_T=S_0=6000 vs 放开约束（跨日套利）")
    L("-" * 78)
    L(t_term.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    J_free = float(t_term["J"].iloc[0]); J_tied = float(t_term["J"].iloc[1])
    d_soc = J_free - J_tied
    L(f"  → 题目要求的 S_T=S_0 相对放开终端约束多支出 {-d_soc:,.2f} 元"
      f"（+{(J_tied/J_free-1)*100:.3f}%）。")
    L("")
    L("  【论文可直接引用的表述】")
    L(f"    题目要求储能 24:00 与 0:00 储电量相同。该硬约束使全天购电费为 "
      f"{J_tied:,.2f} 元；")
    L(f"    若允许储能跨日转移能量（不作日末归零要求），购电费可降至 "
      f"{J_free:,.2f} 元，")
    L(f"    即日内能量平衡约束的显式经济代价为 {-d_soc:,.2f} 元/日"
      f"（约 {(J_tied/J_free-1)*100:.2f}%）。")
    L("    本问按题目要求采用 S_T=S_0 作为主模型约束，并将其与放开情景的差额作为"
      "稳健性讨论。")
    KPI["term_gap"] = -d_soc

    # ---------- 7b. 电池折旧口径对比 ----------
    t_dep = compare_depreciation(df, k_values=(0.0, 0.15))
    L("\n" + "-" * 78)
    L("[电池折旧口径对比]  题目口径 K=0（提交用） vs 工程口径 K=0.15（论文加分）")
    L("-" * 78)
    L(t_dep.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    c0, c1 = t_dep.iloc[0], t_dep.iloc[1]
    L(f"  → 计入折旧后，等效循环次数由 {c0['n_cycles']:.3f} 次降至 "
      f"{c1['n_cycles']:.3f} 次（降幅 {(1-c1['n_cycles']/c0['n_cycles'])*100:.1f}%），")
    L(f"     充放电量由 {c0['E_discharge']:,.2f} 降至 {c1['E_discharge']:,.2f} kWh，")
    L(f"     模型自发减少浅充浅放，但全天总成本仅增加 "
      f"{c1['J_total']-c0['J_total']:,.2f} 元（+{(c1['J_total']/c0['J_total']-1)*100:.2f}%）。")
    L("  → 提交 result1.xlsx 采用 K=0（题目口径，零合规风险）；")
    L("     K=0.15 结论仅用于论文正文的工程适用性讨论（不出图）。")
    KPI["dep_k0"] = dict(c0); KPI["dep_k1"] = dict(c1)

    # ---------- 7c. 下采样口径核对 ----------
    L("\n" + "-" * 78)
    L("[小时级预报 → 10 min 网格]  下采样口径核对（全局参数 DOWNSAMPLE_METHOD）")
    L("-" * 78)
    hourly_demo = df["pv"].to_numpy(float).reshape(24, 6).mean(axis=1)
    for mth in ("zoh", "linear", "spline"):
        v = hour_to_10min(hourly_demo, method=mth)
        mark = "  ← 全篇采用" if mth == DOWNSAMPLE_METHOD else ""
        L(f"  {mth:7s}: 144点合计 = {v.sum()*DT:,.2f} kWh, "
          f"峰值 = {v.max():,.2f} kW, 最小 = {v.min():,.2f} kW{mark}")
    L(f"  → 全局 DOWNSAMPLE_METHOD = '{DOWNSAMPLE_METHOD}'，Q1/Q2/Q3/Q4 统一使用；")
    L("     仅可视化或敏感性分析时可显式传入 'spline'，并在论文中注明口径。")
    if DOWNSAMPLE_METHOD != "zoh":
        L("  ⚠ 警告：正式结果建议使用 'zoh'，其他口径会引入预报语义之外的平滑假设。")

    # ---------- 8. 论文表格 ----------
    t_seg = segment_table(res)
    L("\n[论文表2 数据] 储能分时段充放电量")
    L(t_seg.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    pick = ["10:00", "12:00", "14:00", "16:00", "18:00", "20:00"]
    rows1 = []
    for ts in pick:
        h, mi = map(int, ts.split(":"))
        i = h * 6 + mi // 10
        h2, m2 = divmod(h * 60 + mi + 10, 60)
        rows1.append([f"{ts}-{h2}:{m2:02d}", round(res["G"][i] * DT, 2)])
    t1 = pd.DataFrame(rows1, columns=["时间段", "购电量(kWh)"])
    t1.loc[len(t1)] = ["全天购电量", round(res["E_buy"], 2)]
    t1.loc[len(t1)] = ["全天购电费(元)", round(res["J"], 2)]
    t2 = t_seg[["时间段", "充电量", "放电量"]].copy()
    L("\n[论文表1 数据] 指定时段购电量")
    L(t1.to_string(index=False))
    L(f"\n[论文表2 数据] 0:00 储电量 = {res['S0']:,.2f} kWh, "
      f"24:00 储电量 = {res['S_end']:,.2f} kWh")

    # ---------- 9. 导出 ----------
    res_submit, _, _ = solve_q1_baseline(df, default_params(k_bat=0.0), eps=0.005)
    export_result1(res_submit, os.path.join(OUT_DIR, "result1.xlsx"))
    L(f"\n[结果文件] {os.path.join(OUT_DIR, 'result1.xlsx')}")
    L("  【口径】result1.xlsx 仅含题目口径 K=0（无折旧）；"
      "K=0.15 工程口径结论只在论文正文讨论，不写入提交文件。")
    L(f"  【基准】最终成本 {res_submit['J']:,.2f} 元 | "
      f"无储能基准 {sv['J_noESS']:,.2f} 元 | 降本 "
      f"{(1-res_submit['J']/sv['J_noESS'])*100:.2f}% | "
      f"终端电量 {res_submit['S_end']:,.0f} kWh（= 初始值）")

    for nm, tb in [("T_sens_emax", t_emax), ("T_sens_pmax", t_pmax),
                   ("T_sens_eta", t_eta), ("T_sens_price", t_price),
                   ("T_terminal_soc", t_term), ("T_marginal", sv["marginal_df"]),
                   ("T_segment_6", t_seg), ("PaperTable1_buy", t1),
                   ("PaperTable2_soc", t2),
                   ("T_curtail_scan", t_scan), ("T_depreciation", t_dep)]:
        tb.to_csv(os.path.join(TAB_DIR, f"{nm}.csv"), index=False,
                  encoding="utf-8-sig")

    # ---------- 10. 绘图 ----------
    L("\n" + "-" * 78)
    L("[绘图] 正文 3 张单主题图 + 附录 3 张图（PNG@300dpi + PDF 矢量）")
    L("  排版原则：正文只放核心结论，附录放扫参与过程证据；一律不拼大图。")
    L("  正文 3 张：F1 调度总览 / F2 弃光自洽性 / F3 终端SOC稳健性")
    L("  附录 3 张：FA1 灵敏度 / FA2 价值与边际 / FA3 全时段热力图")
    L("-" * 78)
    L("  正文：")
    fig1_balance_soc(res, df)
    fig2_curtailment_scan(t_scan)
    fig3_terminal_soc(t_term)
    L("  附录：")
    figA1_sensitivity(t_emax, t_pmax, t_eta)
    figA2_value_marginal(sv)
    figA3_heatmap(res, df)

    with open(os.path.join(OUT_DIR, "Q1_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    L(f"\n[报告] {os.path.join(OUT_DIR, 'Q1_report.txt')}")
    L("=" * 78)
    L("完成。")
    L("=" * 78)


if __name__ == "__main__":
    main()