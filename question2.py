"""问题二：含预测误差与紧急购电的全年日前储能调度。

运行：python question2.py
依赖：numpy、pandas、pulp、openpyxl。输入文件默认与本脚本同目录。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pulp

DT, T = 1.0 / 6.0, 144
P_MAX, E_LO, E_HI = 5000.0, 1200.0, 10800.0
ETA_C = ETA_D = 0.90
S0 = 6000.0
RISK_QUANTILE = 0.90
WEEKDAY_SAMPLES = 4
START, END = pd.Timestamp("2025-02-01"), pd.Timestamp("2025-12-31")
ROOT = Path(__file__).resolve().parent
ATTACH1, ATTACH2, OUTPUT = ROOT / "附件1.xlsx", ROOT / "附件2.xlsx", ROOT / "result2.xlsx"
TARGET_DATES = tuple(pd.Timestamp(x) for x in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"))
SEGMENTS = (("0:00-4:00", 0, 24), ("4:00-8:00", 24, 48), ("8:00-12:00", 48, 72),
            ("12:00-16:00", 72, 96), ("16:00-20:00", 96, 120), ("20:00-24:00", 120, 144))


@dataclass(frozen=True)
class DayPlan:
    g: np.ndarray; c: np.ndarray; d: np.ndarray; s: np.ndarray
    e_plan: np.ndarray; r_risk_model: np.ndarray


def time_label(t: int, end: bool = False) -> str:
    minutes = (t + int(end)) * 10
    if minutes == 1440:
        return "24:00"
    return f"{minutes // 60}:{minutes % 60:02d}"


def interval_label(t: int) -> str:
    return f"{time_label(t)}-{time_label(t, end=True)}"


def load_data(attach1: Path, attach2: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    price_df = pd.read_excel(attach1, header=0)
    price = pd.to_numeric(price_df.iloc[:T, 1], errors="coerce").to_numpy(float)
    if len(price) != T or not np.isfinite(price).all():
        raise ValueError("附件1必须包含144个有效的10分钟电价。")
    sheets = pd.read_excel(attach2, sheet_name=["小区负载", "光伏发电实际功率"], header=0)
    def parse(sheet: pd.DataFrame, name: str) -> pd.DataFrame:
        dates = pd.to_datetime(sheet.iloc[:, 0], unit="D", origin="1899-12-30", errors="coerce")
        values = sheet.iloc[:, 1:1 + T].apply(pd.to_numeric, errors="coerce")
        if values.shape[1] != T or values.isna().any().any() or dates.isna().any():
            raise ValueError(f"附件2工作表“{name}”必须包含有效日期及144个数值。")
        out = pd.DataFrame(values.to_numpy(float), index=dates.dt.normalize())
        if out.index.duplicated().any():
            raise ValueError(f"附件2工作表“{name}”存在重复日期。")
        return out.sort_index()
    load, pv = parse(sheets["小区负载"], "小区负载"), parse(sheets["光伏发电实际功率"], "光伏发电实际功率")
    if not load.index.equals(pv.index):
        raise ValueError("附件2的负载与光伏日期不一致。")
    required = pd.date_range("2025-01-01", END, freq="D")
    missing = required.difference(load.index)
    if len(missing):
        raise ValueError(f"附件2缺少日期，例如：{missing[0].date()}")
    return price, load, pv


def historical_dates(day: pd.Timestamp, history: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """优先取最近4个同星期几；不足时以最近历史日补足。"""
    prior = history[history < day]
    same = prior[prior.weekday == day.weekday][-WEEKDAY_SAMPLES:]
    if len(same) < WEEKDAY_SAMPLES:
        extra = prior[~prior.isin(same)][-(WEEKDAY_SAMPLES - len(same)):]
        return extra.append(same).sort_values()
    return same


def forecast_and_risk(day: pd.Timestamp, load: pd.DataFrame, pv: pd.DataFrame,
                      alpha: float = 1.0, quantile: float = RISK_QUANTILE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = historical_dates(day, load.index)
    if len(dates) == 0:
        raise ValueError(f"{day.date()}之前没有可用历史数据。")
    lh, ph = load.loc[dates].to_numpy(float), pv.loc[dates].to_numpy(float)
    lhat, phat = lh.mean(axis=0), ph.mean(axis=0)
    # r = (L-Lhat) - (PV-PVhat)，按10分钟时段分别取经验分位数。
    residuals = (lh - lhat) - (ph - phat)
    risk = np.maximum(0.0, np.quantile(alpha * residuals, quantile, axis=0))
    return lhat, phat, risk


def solve_day(price: np.ndarray, lhat: np.ndarray, phat: np.ndarray, risk: np.ndarray, s_initial: float) -> DayPlan:
    """风险变量约束化的日前 MILP；E_plan 是无成本的计划弃电/备用松弛量。"""
    m = pulp.LpProblem("Q2_Risk_Aware_Day_Ahead", pulp.LpMinimize)
    ix = range(T)
    g = pulp.LpVariable.dicts("G_plan", ix, lowBound=0)
    c = pulp.LpVariable.dicts("C_plan", ix, lowBound=0, upBound=P_MAX)
    d = pulp.LpVariable.dicts("D_plan", ix, lowBound=0, upBound=P_MAX)
    e = pulp.LpVariable.dicts("E_plan", ix, lowBound=0)
    rr = pulp.LpVariable.dicts("R_risk", ix, lowBound=0)
    s = pulp.LpVariable.dicts("S_plan", range(T + 1), lowBound=E_LO, upBound=E_HI)
    z = pulp.LpVariable.dicts("mode", ix, cat="Binary")
    m += pulp.lpSum((price[t] * g[t] + 5.0 * price[t] * rr[t]) * DT for t in ix)
    m += s[0] == s_initial
    for t in ix:
        # 额外计划购电必须由 E_plan 显式吸收，不允许能量凭空消失。
        m += g[t] + phat[t] + d[t] == lhat[t] + c[t] + e[t]
        # E_plan 越大，风险缺口越小；普通电与5倍紧急电由目标函数权衡。
        m += rr[t] >= risk[t] - e[t]
        m += c[t] <= P_MAX * z[t]
        m += d[t] <= P_MAX * (1 - z[t])
        m += s[t + 1] == s[t] + ETA_C * c[t] * DT - d[t] * DT / ETA_D
    status = m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=120))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"日前MILP求解失败：{pulp.LpStatus[status]}")
    take = lambda x, n=T: np.array([x[i].value() for i in range(n)], dtype=float)
    return DayPlan(take(g), take(c), take(d), take(s, T + 1), take(e), take(rr))


def execute_day(plan: DayPlan, actual_load: np.ndarray, actual_pv: np.ndarray, price: np.ndarray,
                day: pd.Timestamp, risk: np.ndarray) -> tuple[pd.DataFrame, float]:
    """理解A：只按SOC边界裁剪计划充放电；随后用实际功率平衡计算紧急购电。"""
    soc = plan.s[0]
    rows = []
    for t in range(T):
        c = min(plan.c[t], P_MAX, max(0.0, (E_HI - soc) / (ETA_C * DT)))
        d = min(plan.d[t], P_MAX, max(0.0, (soc - E_LO) / (DT / ETA_D)))
        delta = actual_load[t] + c - actual_pv[t] - plan.g[t] - d
        emergency, curtail = max(0.0, delta), max(0.0, -delta)
        soc_next = soc + ETA_C * c * DT - d * DT / ETA_D
        rows.append(dict(date=day, slot=t, interval=interval_label(t), price=price[t],
                         load_actual=actual_load[t], pv_actual=actual_pv[t],
                         load_forecast=np.nan, pv_forecast=np.nan, risk_margin=risk[t],
                         plan_purchase_kw=plan.g[t], plan_purchase_kwh=plan.g[t] * DT,
                         plan_charge_kw=plan.c[t], plan_discharge_kw=plan.d[t],
                         executed_charge_kw=c, executed_discharge_kw=d,
                         emergency_purchase_kw=emergency, emergency_purchase_kwh=emergency * DT,
                         actual_curtail_kw=curtail, soc_start=soc, soc_end=soc_next,
                         plan_cost=price[t] * plan.g[t] * DT,
                         emergency_cost=5.0 * price[t] * emergency * DT))
        soc = soc_next
    return pd.DataFrame(rows), soc


def validate(detail: pd.DataFrame) -> None:
    tol = 1e-5
    balance = (detail.plan_purchase_kw + detail.emergency_purchase_kw + detail.pv_actual + detail.executed_discharge_kw
               - detail.load_actual - detail.executed_charge_kw - detail.actual_curtail_kw).abs().max()
    checks = [
        (balance <= tol, f"实际功率平衡残差={balance}"),
        ((detail.soc_start >= E_LO-tol).all() and (detail.soc_end <= E_HI+tol).all(), "SOC越界"),
        ((detail.executed_charge_kw <= P_MAX+tol).all() and (detail.executed_discharge_kw <= P_MAX+tol).all(), "充放电功率越界"),
        (((detail.executed_charge_kw * detail.executed_discharge_kw) <= tol).all(), "执行侧充放电互斥被破坏"),
        ((detail.emergency_purchase_kw >= -tol).all(), "紧急购电出现负值"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed: raise AssertionError("；".join(failed))


def emergency_periods(day_detail: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    rows, start, energy = [], None, 0.0
    for r in day_detail.itertuples(index=False):
        if r.emergency_purchase_kwh > 1e-7:
            if start is None: start, energy = r.slot, 0.0
            energy += r.emergency_purchase_kwh
        elif start is not None:
            rows.append([day, f"{time_label(start)}-{time_label(r.slot)}", energy]); start = None
    if start is not None:
        rows.append([day, f"{time_label(start)}-24:00", energy])
    return pd.DataFrame(rows, columns=["日期", "购电时间段", "购电量"])


def export_result(detail: pd.DataFrame, output: Path) -> None:
    # 附件5“计划购电量”模板采用0:10起的显示标签，并要求两个全天汇总列。
    template_slots = [f"{time_label(t + 1)}-{time_label(t + 1, end=True)}" for t in range(T)]
    purchase = detail.pivot(index="date", columns="interval", values="plan_purchase_kwh").reindex(columns=[interval_label(t) for t in range(T)])
    purchase.columns = template_slots
    purchase["全天购电量"] = purchase.sum(axis=1)
    purchase["全天购电费"] = detail.groupby("date").plan_cost.sum()
    purchase.index.name = "日期\\时间"
    purchase = purchase.reset_index().round(4)
    battery_rows = []
    for day, x in detail.groupby("date", sort=True):
        for j, (name, a, b) in enumerate(SEGMENTS):
            y = x.iloc[a:b]
            battery_rows.append([day if j == 0 else "", name,
                                 y.executed_charge_kw.sum()*DT, y.executed_discharge_kw.sum()*DT,
                                 "0:00" if j == 0 else ("24:00" if j == 1 else ""),
                                 y.iloc[0].soc_start if j == 0 else (y.iloc[-1].soc_end if j == 1 else "")])
    battery = pd.DataFrame(battery_rows, columns=["日期", "时间段", "充电量", "放电量", "时刻", "储电量"]).round(4)
    emergency = pd.concat([emergency_periods(x, day) for day, x in detail.groupby("date", sort=True)], ignore_index=True)
    if emergency.empty: emergency = pd.DataFrame(columns=["日期", "购电时间段", "购电量"])
    emergency["购电量"] = emergency["购电量"].round(4)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        purchase.to_excel(writer, sheet_name="计划购电量", index=False)
        battery.to_excel(writer, sheet_name="充放电量", index=False)
        emergency.to_excel(writer, sheet_name="紧急购电量", index=False)


def run(alpha: float = 1.0, quantile: float = RISK_QUANTILE, output: Path = OUTPUT) -> pd.DataFrame:
    price, load, pv = load_data(ATTACH1, ATTACH2)
    operating_days = pd.date_range(START, END, freq="D")
    print(f"电价点数：{len(price)}，负载日期数：{len(load)}，光伏日期数：{len(pv)}")
    print(f"运行区间：{START.date()} ~ {END.date()}，共 {len(operating_days)} 天")
    all_days, soc = [], S0
    for day in operating_days:
        lhat, phat, risk = forecast_and_risk(day, load, pv, alpha, quantile)
        plan = solve_day(price, lhat, phat, risk, soc)
        daily, soc = execute_day(plan, load.loc[day].to_numpy(float), pv.loc[day].to_numpy(float), price, day, risk)
        daily["load_forecast"], daily["pv_forecast"] = lhat, phat
        all_days.append(daily)
    detail = pd.concat(all_days, ignore_index=True)
    if not detail.groupby("date").size().eq(T).all():
        raise AssertionError("存在日期的时段数不等于144")
    validate(detail)
    export_result(detail, output)
    table3 = pd.concat([emergency_periods(detail[detail.date == day], day) for day in TARGET_DATES], ignore_index=True)
    print("全年计划购电费：%.2f 元" % detail.plan_cost.sum())
    print("全年紧急购电费：%.2f 元" % detail.emergency_cost.sum())
    print("全年总成本：%.2f 元" % (detail.plan_cost.sum() + detail.emergency_cost.sum()))
    print("全年紧急购电总次数：%d" % (detail.emergency_purchase_kw > 1e-7).sum())
    print("全年紧急购电量：%.2f kWh" % detail.emergency_purchase_kwh.sum())
    print("全年弃电量：%.2f kWh" % (detail.actual_curtail_kw * DT).sum())
    print("\n表3：指定日期紧急购电时间段与购电量(kWh)")
    print(table3.to_string(index=False) if not table3.empty else "指定日期均无紧急购电")
    print(f"\n已导出：{output}")
    return detail


if __name__ == "__main__":
    run()
