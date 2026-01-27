# -*- coding: utf-8 -*-
"""
Baostock 选股：主板A股 + (可选)ETF
条件：
- 现价 > price_min
- 20日涨幅 > rise_20d_min_pct
- 20日平均成交额 > avg_amount_20d_min
- 收盘价 > MA60
- MA20 > MA60
- 排除 ST / *ST
- 最近5日内 单日最大涨幅 <= max_single_day_rise_5d_pct

并用可配置评分模型选出 Top2
"""

import baostock as bs
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Config:
    # ====== Universe ======
    include_etf: bool = True
    etf_prefixes: tuple = ("51", "56", "58", "15", "16")  # 常见ETF前缀，可改
    include_mainboard_a: bool = True

    # ====== Filters ======
    price_min: float = 5.0
    rise_20d_min_pct: float = 10.0
    avg_amount_20d_min: float = 1e8  # 1亿
    ma_short: int = 20
    ma_long: int = 60
    max_single_day_rise_5d_pct: float = 8.0

    # 需要多少交易日数据：至少 MA_long + 1 + 5
    lookback_days: int = 90  # 给足缓冲，避免停牌/数据缺失

    # ====== Scoring Weights (可改) ======
    w_rise20: float = 0.40        # 20日涨幅
    w_above_ma60: float = 0.20    # (close/ma60 - 1)
    w_ma_trend: float = 0.20      # (ma20/ma60 - 1)
    w_liquidity: float = 0.15     # log(avg_amount_20)
    w_low_vol: float = 0.05       # -vol20 (波动越低越好)

    top_n: int = 2


def bs_to_df(rs: bs.data.resultset.ResultData) -> pd.DataFrame:
    """Baostock ResultData -> DataFrame"""
    data = []
    while rs.error_code == "0" and rs.next():
        data.append(rs.get_row_data())
    return pd.DataFrame(data, columns=rs.fields)


def get_stock_universe(cfg: Config) -> pd.DataFrame:
    """
    返回基础信息 DataFrame: code, code_name
    过滤：主板A股(60/00) + (可选)ETF(前缀)，排除 300/301/688/689/8/4/9 等
    """
    rs = bs.query_stock_basic()
    if rs.error_code != "0":
        raise RuntimeError(f"query_stock_basic failed: {rs.error_code} {rs.error_msg}")
    df = bs_to_df(rs)

    df = df[["code", "code_name"]].copy()

    # 取纯数字代码：sh.600000 -> 600000
    pure = df["code"].str.split(".").str[-1]

    # 排除创业板/科创板/北交所等
    excluded = pure.str.startswith(("300", "301", "688", "689", "8", "4", "9"))

    # 主板A股：60 / 00
    is_mainboard_a = pure.str.startswith(("60", "00"))

    # ETF：按前缀（可配置）
    is_etf = pure.str.startswith(cfg.etf_prefixes)

    keep = (~excluded) & (
        (cfg.include_mainboard_a & is_mainboard_a) |
        (cfg.include_etf & is_etf)
    )

    out = df[keep].reset_index(drop=True)
    return out



def fetch_kdata(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取日线（包含成交额 amount、涨跌幅 pctChg、收盘 close）
    """
    fields = "date,close,pctChg,amount"
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 2: 前复权；0不复权；一般用前复权做均线更稳
    )
    if rs.error_code != "0":
        return pd.DataFrame()

    df = bs_to_df(rs)
    if df.empty:
        return df

    # 类型转换
    df["date"] = pd.to_datetime(df["date"])
    for c in ["close", "pctChg", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df


def compute_features(k: pd.DataFrame, cfg: Config) -> dict | None:
    """
    从k线计算筛选指标与打分需要的特征
    """
    if k is None or k.empty:
        return None

    # 需要足够长度
    need = max(cfg.ma_long + 1, cfg.ma_short + 1) + 5
    if len(k) < need:
        return None

    k = k.copy()
    k["ret"] = k["close"].pct_change()

    k["ma20"] = k["close"].rolling(cfg.ma_short).mean()
    k["ma60"] = k["close"].rolling(cfg.ma_long).mean()

    last = k.iloc[-1]
    if np.isnan(last["ma20"]) or np.isnan(last["ma60"]):
        return None

    close = float(last["close"])
    ma20 = float(last["ma20"])
    ma60 = float(last["ma60"])

    # 20日涨幅：用交易日序列的第-21与最后一天计算（避免用pctChg累计误差）
    c_20ago = float(k.iloc[-(cfg.ma_short + 1)]["close"])
    rise20 = (close / c_20ago - 1.0) * 100.0

    # 20日平均成交额
    avg_amount_20 = float(k["amount"].tail(cfg.ma_short).mean(skipna=True))

    # 最近5日单日最大涨幅（使用pctChg；若缺失则用close变化补）
    last5 = k.tail(5)
    if last5["pctChg"].notna().any():
        max_up_5 = float(last5["pctChg"].max(skipna=True))
    else:
        # fallback
        max_up_5 = float((last5["close"].pct_change() * 100.0).max(skipna=True))

    # 波动（20日收益率标准差）
    vol20 = float(k["ret"].tail(cfg.ma_short).std(skipna=True))

    feats = {
        "close": close,
        "rise20": rise20,
        "avg_amount_20": avg_amount_20,
        "ma20": ma20,
        "ma60": ma60,
        "above_ma60": close / ma60 - 1.0,
        "ma_trend": ma20 / ma60 - 1.0,
        "max_up_5": max_up_5,
        "vol20": vol20 if np.isfinite(vol20) else np.nan,
        "last_date": last["date"].strftime("%Y-%m-%d"),
    }
    return feats


def pass_filters(name: str, feats: dict, cfg: Config) -> tuple[bool, list]:
    """
    返回(是否通过, 未通过原因列表)
    """
    reasons = []

    # 排除 ST / *ST：用名称包含ST判断（简单、直观）
    # 注意：有些基金/ETF名称也可能带ST（极少），若你不希望误杀，可把ETF跳过该规则
    if name is not None and isinstance(name, str):
        if "ST" in name.upper():
            reasons.append("ST/*ST")

    if feats["close"] <= cfg.price_min:
        reasons.append(f"close<= {cfg.price_min}")

    if feats["rise20"] <= cfg.rise_20d_min_pct:
        reasons.append(f"rise20<= {cfg.rise_20d_min_pct}%")

    if feats["avg_amount_20"] <= cfg.avg_amount_20d_min:
        reasons.append(f"avg_amount_20<= {cfg.avg_amount_20d_min:.0f}")

    if not (feats["close"] > feats["ma60"]):
        reasons.append("close<=MA60")

    if not (feats["ma20"] > feats["ma60"]):
        reasons.append("MA20<=MA60")

    if feats["max_up_5"] > cfg.max_single_day_rise_5d_pct:
        reasons.append(f"max_up_5> {cfg.max_single_day_rise_5d_pct}%")

    return (len(reasons) == 0, reasons)


def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def score_candidates(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    df: 已通过过滤的候选，包含特征列
    返回按score降序
    """
    x = df.copy()

    # 评分特征归一化（z-score）
    x["z_rise20"] = zscore(x["rise20"])
    x["z_above_ma60"] = zscore(x["above_ma60"])
    x["z_ma_trend"] = zscore(x["ma_trend"])
    x["z_liq"] = zscore(np.log1p(x["avg_amount_20"]))
    x["z_low_vol"] = zscore(-x["vol20"].fillna(x["vol20"].median()))

    x["score"] = (
        cfg.w_rise20 * x["z_rise20"]
        + cfg.w_above_ma60 * x["z_above_ma60"]
        + cfg.w_ma_trend * x["z_ma_trend"]
        + cfg.w_liquidity * x["z_liq"]
        + cfg.w_low_vol * x["z_low_vol"]
    )

    x = x.sort_values("score", ascending=False).reset_index(drop=True)
    return x


def main():
    cfg = Config()
    print("=== Config ===")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
    print("==============\n")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_code} {lg.error_msg}")
    print("login success!")

    try:
        universe = get_stock_universe(cfg)
        print(f"universe size: {len(universe)}")

        # 计算日期范围：为了简单，使用最近 lookback_days 的自然日范围
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - pd.Timedelta(days=cfg.lookback_days * 2)).strftime("%Y-%m-%d")
        # *2 是自然日冗余，保证交易日数量足够

        rows = []
        total = len(universe)
        for i, r in universe.iterrows():
            code = r["code"]
            name = r["code_name"]

            k = fetch_kdata(code, start, end)
            feats = compute_features(k, cfg)
            if feats is None:
                continue

            ok, reasons = pass_filters(name, feats, cfg)
            if not ok:
                continue

            rows.append({
                "code": code,
                "name": name,
                **feats
            })

            # 适度打印进度（避免刷屏）
            if (i + 1) % 500 == 0:
                print(f"progress: {i+1}/{total}, passed: {len(rows)}")

        if not rows:
            print("No candidates passed filters.")
            return

        cand = pd.DataFrame(rows)
        ranked = score_candidates(cand, cfg)

        print("\n=== Top Candidates (preview) ===")
        show_cols = ["code", "name", "score", "close", "rise20", "avg_amount_20", "ma20", "ma60", "max_up_5", "vol20", "last_date"]
        print(ranked[show_cols].head(20).to_string(index=False))

        top2 = ranked.head(cfg.top_n)
        print(f"\n=== Top {cfg.top_n} (BEST BUY) ===")
        for _, x in top2.iterrows():
            print(
                f"- {x['code']} {x['name']} | score={x['score']:.3f} | close={x['close']:.2f} "
                f"| rise20={x['rise20']:.2f}% | avgAmt20={x['avg_amount_20']/1e8:.2f}亿 "
                f"| MA20={x['ma20']:.2f} MA60={x['ma60']:.2f} "
                f"| maxUp5={x['max_up_5']:.2f}% | vol20={x['vol20']:.4f} | date={x['last_date']}"
            )

    finally:
        bs.logout()
        print("\nlogout success!")


if __name__ == "__main__":
    main()
