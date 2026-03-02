# -*- coding: utf-8 -*-
"""
A股多策略选股脚本 - 直接使用baostock数据源

依赖安装:
    pip install baostock pandas numpy tqdm

注意:
    - 本脚本文件名不要命名为 baostock.py，会与库冲突
    - 确保网络连接正常，baostock需要联网获取数据
    - 首次运行可能需要较长时间（约10-30分钟，取决于股票数量）

运行示例:
    python screen.py
    python screen.py --date 2026-02-27
    python screen.py --max_stocks 200
"""

import argparse
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import baostock as bs
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==================== 配置常量 ====================

# 字段列表（baostock日K数据）
KLINE_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,isST"

# 最小数据长度要求（用于计算MA120等指标）
MIN_KLINE_LEN = 140

# 流动性过滤阈值
MIN_AVG_AMOUNT_20 = 50_000_000  # 5千万

# 上市天数过滤
MIN_IPO_DAYS = 120  # 120个自然日

# 策略标签
STRATEGY_LABELS = {
    "A": "LOW_VAL_BLUECHIP",
    "B": "UP_TREND",
    "C": "OVERSOLD_REBOUND",
    "D": "MACD_KDJ_RESONANCE",
    "E": "VOLUME_BREAKOUT",
    "F": "SMALL_GROWTH",
}


# ==================== 数据获取函数 ====================

def bs_to_df(rs) -> pd.DataFrame:
    """将baostock结果集转换为DataFrame"""
    data = []
    while rs.error_code == "0" and rs.next():
        data.append(rs.get_row_data())
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data, columns=rs.fields)


def get_nearest_trade_date(target_date: Optional[str] = None) -> str:
    """
    获取最近交易日
    如果指定日期，返回该日期之前（含）的最近交易日
    """
    if target_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = target_date

    # 获取前60天的交易日数据
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")

    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != "0":
        raise RuntimeError(f"获取交易日失败: {rs.error_msg}")

    df = bs_to_df(rs)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    df["is_trading_day"] = df["is_trading_day"].astype(int)

    # 获取交易日
    trade_dates = df[df["is_trading_day"] == 1]["calendar_date"]
    if trade_dates.empty:
        raise RuntimeError(f"未找到交易日: {start_date} ~ {end_date}")

    # 如果指定了日期，找该日期或之前的最近交易日
    if target_date:
        target_dt = pd.to_datetime(target_date)
        valid_dates = trade_dates[trade_dates <= target_dt]
        if valid_dates.empty:
            raise RuntimeError(f"未找到 {target_date} 之前的交易日")
        return valid_dates.iloc[-1].strftime("%Y-%m-%d")
    else:
        return trade_dates.iloc[-1].strftime("%Y-%m-%d")


def get_all_stock_list(trade_day: str) -> pd.DataFrame:
    """获取指定交易日的所有股票列表（剔除指数）"""
    rs = bs.query_all_stock(day=trade_day)
    if rs.error_code != "0":
        raise RuntimeError(f"获取股票列表失败: {rs.error_msg}")

    df = bs_to_df(rs)
    if df.empty:
        return df

    # 剔除指数类（通常以指数名称区分，这里通过代码格式过滤）
    # 只保留 sh.xxxxxx 和 sz.xxxxxx 格式
    df = df[df["code"].str.match(r"^(sh|sz)\.", na=False)].copy()

    return df.reset_index(drop=True)


def get_stock_basic_info() -> pd.DataFrame:
    """获取股票基础信息（包含ipoDate）"""
    rs = bs.query_stock_basic()
    if rs.error_code != "0":
        raise RuntimeError(f"获取股票基础信息失败: {rs.error_msg}")

    df = bs_to_df(rs)
    return df


def get_kline_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取单只股票的K线数据"""
    rs = bs.query_history_k_data_plus(
        code,
        KLINE_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 前复权
    )

    if rs.error_code != "0":
        return pd.DataFrame()

    df = bs_to_df(rs)
    if df.empty:
        return df

    # 类型转换
    df["date"] = pd.to_datetime(df["date"])

    # 数值列转换
    numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount",
                    "turn", "tradestatus", "pctChg", "peTTM", "pbMRQ", "isST"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 按日期排序
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ==================== 指标计算函数 ====================

def calculate_ma(series: pd.Series, n: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=n).mean()


def calculate_ema(series: pd.Series, n: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=n, adjust=False).mean()


def calculate_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """计算RSI指标（Wilder方法）"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    # 使用Wilder的平滑方法
    avg_gain = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算MACD指标"""
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal)
    hist = 2 * (dif - dea)
    return dif, dea, hist


def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算KDJ指标"""
    low_min = low.rolling(window=n, min_periods=n).min()
    high_max = high.rolling(window=n, min_periods=n).max()

    # RSV
    rsv = (close - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)  # 处理NaN

    # K值：RSV的m1日EMA
    k = calculate_ema(rsv, m1)
    # D值：K的m2日EMA
    d = calculate_ema(k, m2)
    # J值
    j = 3 * k - 2 * d

    return k, d, j


def calculate_max_drawdown(close: pd.Series) -> float:
    """计算最大回撤"""
    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    return -drawdown.min()  # 返回正值


def calculate_indicators(df: pd.DataFrame) -> Optional[Dict]:
    """
    计算所有技术指标
    返回当日指标字典，如果数据不足返回None
    """
    if len(df) < MIN_KLINE_LEN:
        return None

    # 确保数据按日期排序
    df = df.copy()

    # 基础价格数据
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_price = df["open"]
    volume = df["volume"]
    amount = df["amount"]
    preclose = df["preclose"]

    # 计算MA
    df["MA5"] = calculate_ma(close, 5)
    df["MA10"] = calculate_ma(close, 10)
    df["MA20"] = calculate_ma(close, 20)
    df["MA60"] = calculate_ma(close, 60)
    df["MA120"] = calculate_ma(close, 120)

    # 计算RSI
    df["RSI14"] = calculate_rsi(close, 14)

    # 计算MACD
    df["DIF"], df["DEA"], df["MACD_HIST"] = calculate_macd(close)

    # 计算KDJ
    df["K"], df["D"], df["J"] = calculate_kdj(high, low, close)

    # 计算成交量/成交额均线
    df["vol20"] = calculate_ma(volume, 20)
    df["amount20"] = calculate_ma(amount, 20)
    df["avg_amount_60"] = calculate_ma(amount, 60)

    # 计算20日/60日最高价
    df["high20"] = close.rolling(window=20).max()
    df["high60"] = close.rolling(window=60).max()

    # 计算60日最大回撤
    df["max_dd_60"] = close.rolling(window=60).apply(lambda x: calculate_max_drawdown(x), raw=False)

    # 计算乖离率
    df["bias_ma60"] = (close / df["MA60"] - 1)

    # 计算60日收益
    df["close_60d_ago"] = close.shift(60)
    df["ret60"] = close / df["close_60d_ago"] - 1

    # 获取最新数据
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    prev2 = df.iloc[-3] if len(df) >= 3 else prev
    prev3 = df.iloc[-4] if len(df) >= 4 else prev2

    # 检查关键指标是否有NaN
    key_indicators = ["MA20", "MA60", "MA120", "RSI14", "DIF", "DEA", "K", "D", "J"]
    for key in key_indicators:
        if pd.isna(last[key]):
            return None

    # 获取MA60的20日前值（用于趋势判断）
    ma60_20d_ago = df["MA60"].iloc[-21] if len(df) >= 21 else df["MA60"].iloc[0]

    return {
        # 基础数据
        "code": last["code"],
        "date": last["date"].strftime("%Y-%m-%d") if isinstance(last["date"], pd.Timestamp) else str(last["date"]),
        "open": last["open"],
        "high": last["high"],
        "low": last["low"],
        "close": last["close"],
        "preclose": last["preclose"],
        "volume": last["volume"],
        "amount": last["amount"],
        "turn": last["turn"],
        "pctChg": last["pctChg"],
        "peTTM": last["peTTM"],
        "pbMRQ": last["pbMRQ"],

        # 均线
        "MA5": last["MA5"],
        "MA10": last["MA10"],
        "MA20": last["MA20"],
        "MA60": last["MA60"],
        "MA120": last["MA120"],

        # 技术指标
        "RSI14": last["RSI14"],
        "DIF": last["DIF"],
        "DEA": last["DEA"],
        "MACD_HIST": last["MACD_HIST"],
        "K": last["K"],
        "D": last["D"],
        "J": last["J"],

        # 辅助指标
        "vol20": last["vol20"],
        "amount20": last["amount20"],
        "avg_amount_60": last["avg_amount_60"],
        "high20": last["high20"],
        "high60": last["high60"],
        "max_dd_60": last["max_dd_60"],
        "bias_ma60": last["bias_ma60"],
        "ret60": last["ret60"],

        # 历史数据（用于判断金叉等）
        "DIF_prev": prev["DIF"],
        "DEA_prev": prev["DEA"],
        "K_prev": prev["K"],
        "D_prev": prev["D"],
        "DIF_prev2": prev2["DIF"],
        "DEA_prev2": prev2["DEA"],
        "K_prev2": prev2["K"],
        "D_prev2": prev2["D"],
        "DIF_prev3": prev3["DIF"],
        "DEA_prev3": prev3["DEA"],
        "K_prev3": prev3["K"],
        "D_prev3": prev3["D"],
        "MA60_20d_ago": ma60_20d_ago,

        # 状态标记
        "isST": last["isST"],
        "tradestatus": last["tradestatus"],
    }


# ==================== 策略判断函数 ====================

def check_macd_cross(ind: Dict, lookback: int = 3) -> bool:
    """检查最近lookback天内是否发生MACD金叉"""
    # 今日DIF > DEA
    if ind["DIF"] <= ind["DEA"]:
        return False

    # 检查前lookback天内是否发生过死叉或交叉前状态
    data_points = [
        (ind["DIF_prev"], ind["DEA_prev"]),
        (ind["DIF_prev2"], ind["DEA_prev2"]),
        (ind["DIF_prev3"], ind["DEA_prev3"]),
    ]

    for i in range(min(lookback, len(data_points))):
        dif, dea = data_points[i]
        if pd.notna(dif) and pd.notna(dea) and dif <= dea:
            return True

    return False


def check_kdj_cross(ind: Dict, lookback: int = 3) -> bool:
    """检查最近lookback天内是否发生KDJ金叉"""
    # 今日K > D
    if ind["K"] <= ind["D"]:
        return False

    # 检查前lookback天内是否发生过死叉或交叉前状态
    data_points = [
        (ind["K_prev"], ind["D_prev"]),
        (ind["K_prev2"], ind["D_prev2"]),
        (ind["D_prev3"], ind["D_prev3"]),
    ]

    for i in range(min(lookback, len(data_points))):
        k, d = data_points[i]
        if pd.notna(k) and pd.notna(d) and k <= d:
            return True

    return False


def evaluate_strategies(ind: Dict, market_stats: Dict) -> List[str]:
    """
    评估6个策略，返回命中的策略标签列表
    """
    hit_strategies = []

    # A. 低估值蓝筹策略
    is_big = ind["avg_amount_60"] >= market_stats["big_threshold"]
    is_low_val = (pd.notna(ind["peTTM"]) and ind["peTTM"] > 0 and ind["peTTM"] <= market_stats["pe_low"]) or \
                 (pd.notna(ind["pbMRQ"]) and ind["pbMRQ"] > 0 and ind["pbMRQ"] <= market_stats["pb_low"])
    is_trend_up = ind["close"] >= ind["MA60"] and ind["MA60"] >= ind["MA60_20d_ago"]

    if is_big and is_low_val and is_trend_up:
        hit_strategies.append(STRATEGY_LABELS["A"])

    # B. 趋势向好策略
    ma_aligned = ind["MA20"] > ind["MA60"] > ind["MA120"]
    ma60_rising = ind["MA60"] > ind["MA60_20d_ago"]
    price_above_ma20 = ind["close"] > ind["MA20"]

    if ma_aligned and ma60_rising and price_above_ma20:
        hit_strategies.append(STRATEGY_LABELS["B"])

    # C. 超跌反弹策略
    oversold = pd.notna(ind["max_dd_60"]) and ind["max_dd_60"] >= 0.30
    bias_low = pd.notna(ind["bias_ma60"]) and ind["bias_ma60"] <= -0.10
    is_oversold = oversold or bias_low

    rsi_oversold = pd.notna(ind["RSI14"]) and ind["RSI14"] <= 30
    j_oversold = pd.notna(ind["J"]) and ind["J"] <= 20
    is_super_sold = rsi_oversold or j_oversold

    # 反弹确认
    bullish_candle = ind["close"] > ind["open"] and ind["volume"] >= 1.3 * ind["vol20"]
    above_ma5_stable = ind["close"] >= ind["MA5"] and ind["close"] >= ind["preclose"]
    rebound_confirmed = bullish_candle or above_ma5_stable

    if is_oversold and is_super_sold and rebound_confirmed:
        hit_strategies.append(STRATEGY_LABELS["C"])

    # D. MACD + KDJ 共振策略
    macd_cross = check_macd_cross(ind, lookback=3)
    kdj_cross = check_kdj_cross(ind, lookback=3)
    price_above_ma10 = ind["close"] >= ind["MA10"]

    if macd_cross and kdj_cross and price_above_ma10:
        hit_strategies.append(STRATEGY_LABELS["D"])

    # E. 放量突破策略
    break_high20 = pd.notna(ind["high20"]) and ind["close"] >= 1.01 * ind["high20"]
    break_high60 = pd.notna(ind["high60"]) and ind["close"] >= 1.01 * ind["high60"]
    is_breakout = break_high20 or break_high60

    volume_spike = ind["volume"] >= 1.5 * ind["vol20"]
    amount_spike = ind["amount"] >= 1.5 * ind["amount20"]
    is_volume_surge = volume_spike or amount_spike

    if is_breakout and is_volume_surge:
        hit_strategies.append(STRATEGY_LABELS["E"])

    # F. 小盘成长策略
    is_small = market_stats["small_low"] <= ind["avg_amount_60"] <= market_stats["small_high"]
    is_growth = pd.notna(ind["ret60"]) and ind["ret60"] >= 0.20
    trend_ok = ind["close"] >= ind["MA60"]

    if is_small and is_growth and trend_ok:
        hit_strategies.append(STRATEGY_LABELS["F"])

    return hit_strategies


# ==================== 主程序 ====================

def filter_mainboard_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """过滤主板A股（60/00开头），排除科创板/创业板/北交所等"""
    if df.empty or "code" not in df.columns:
        return df

    def is_mainboard(code: str) -> bool:
        if not isinstance(code, str):
            return False
        parts = code.split(".")
        if len(parts) != 2:
            return False
        exchange, num = parts
        num = str(num)
        # 上海主板: sh.60xxxx
        # 深圳主板: sz.00xxxx
        if exchange == "sh" and num.startswith("60"):
            return True
        if exchange == "sz" and num.startswith("00"):
            return True
        return False

    mask = df["code"].apply(is_mainboard)
    return df[mask].copy()


def calculate_market_stats(indicators_list: List[Dict]) -> Dict:
    """
    计算全市场统计数据（分位阈值等）
    """
    # 提取avg_amount_60
    amounts = [ind["avg_amount_60"] for ind in indicators_list
               if pd.notna(ind["avg_amount_60"]) and ind["avg_amount_60"] > 0]

    # 提取PE/PB（>0的）
    pe_list = []
    pb_list = []
    for ind in indicators_list:
        if pd.notna(ind["peTTM"]) and ind["peTTM"] > 0:
            pe_list.append(ind["peTTM"])
        if pd.notna(ind["pbMRQ"]) and ind["pbMRQ"] > 0:
            pb_list.append(ind["pbMRQ"])

    stats = {}

    if amounts:
        stats["big_threshold"] = np.percentile(amounts, 70)
        stats["small_low"] = np.percentile(amounts, 20)
        stats["small_high"] = np.percentile(amounts, 60)
    else:
        stats["big_threshold"] = 0
        stats["small_low"] = 0
        stats["small_high"] = 0

    if pe_list:
        stats["pe_low"] = np.percentile(pe_list, 30)
    else:
        stats["pe_low"] = 9999

    if pb_list:
        stats["pb_low"] = np.percentile(pb_list, 30)
    else:
        stats["pb_low"] = 9999

    return stats


def main():
    parser = argparse.ArgumentParser(description="A股多策略选股脚本")
    parser.add_argument("--date", type=str, default=None,
                        help="指定交易日，格式YYYY-MM-DD（默认最近交易日）")
    parser.add_argument("--max_stocks", type=int, default=None,
                        help="最大处理股票数量（用于测试，默认全量）")
    args = parser.parse_args()

    print("=" * 60)
    print("A股多策略选股系统")
    print("=" * 60)

    # 登录baostock
    print("\n正在登录baostock...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"登录失败: {lg.error_code} {lg.error_msg}")
    print("登录成功!")

    try:
        # 获取目标交易日
        trade_day = get_nearest_trade_date(args.date)
        print(f"\n目标交易日: {trade_day}")

        # 计算K线数据起始日期（往前450个自然日，确保够260个交易日）
        start_dt = datetime.strptime(trade_day, "%Y-%m-%d") - timedelta(days=450)
        kline_start = start_dt.strftime("%Y-%m-%d")
        print(f"K线数据起始日期: {kline_start}")

        # 获取股票列表
        print("\n正在获取股票列表...")
        stock_list = get_all_stock_list(trade_day)
        print(f"原始股票数量: {len(stock_list)}")

        # 过滤主板A股
        stock_list = filter_mainboard_stocks(stock_list)
        print(f"主板A股数量: {len(stock_list)}")

        if stock_list.empty:
            print(f"\n错误: 未获取到 {trade_day} 的股票数据")
            print("可能原因:")
            print("  1. 该日期是未来日期或非交易日")
            print("  2. 当天数据尚未更新（baostock通常在收盘后更新）")
            print("  3. 网络连接问题")
            print("\n建议: 尝试指定一个过去的交易日，例如:")
            print('  python screen.py --date 2025-02-28')
            return

        # 获取股票基础信息
        print("正在获取股票基础信息...")
        basic_info = get_stock_basic_info()
        basic_info = basic_info[["code", "code_name", "ipoDate"]].copy()
        basic_info["ipoDate"] = pd.to_datetime(basic_info["ipoDate"], errors="coerce")

        # 合并股票列表和基础信息
        stock_list = stock_list.merge(basic_info, on="code", how="left")

        # 重命名code_name为name，便于后续使用
        if "code_name" in stock_list.columns:
            stock_list["name"] = stock_list["code_name"]
        else:
            stock_list["name"] = ""

        # 限制处理数量（用于测试）
        if args.max_stocks and args.max_stocks > 0:
            stock_list = stock_list.head(args.max_stocks)
            print(f"测试模式: 只处理前{args.max_stocks}只股票")

        # 获取交易日历用于计算上市天数
        target_date_dt = pd.to_datetime(trade_day)

        # 处理每只股票
        print("\n正在获取K线数据并计算指标...")
        indicators_list = []
        skipped = 0

        for idx, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="处理进度"):
            code = row["code"]
            name = row.get("code_name", "")
            ipo_date = row.get("ipoDate")

            try:
                # 上市天数过滤
                if pd.notna(ipo_date):
                    days_since_ipo = (target_date_dt - ipo_date).days
                    if days_since_ipo < MIN_IPO_DAYS:
                        skipped += 1
                        continue

                # 获取K线数据
                kdf = get_kline_data(code, kline_start, trade_day)

                if len(kdf) < MIN_KLINE_LEN:
                    skipped += 1
                    continue

                # 获取当日数据
                today_data = kdf[kdf["date"] == trade_day]
                if today_data.empty:
                    skipped += 1
                    continue

                today_row = today_data.iloc[-1]

                # 通用过滤：停牌/ST/流动性
                if today_row["tradestatus"] != 1:  # 停牌
                    skipped += 1
                    continue
                if today_row["isST"] == 1:  # ST股票
                    skipped += 1
                    continue

                # 计算指标
                ind = calculate_indicators(kdf)
                if ind is None:
                    skipped += 1
                    continue

                # 流动性过滤：近20日平均成交额 >= 5千万
                if ind["amount20"] < MIN_AVG_AMOUNT_20:
                    skipped += 1
                    continue

                # 添加股票名称
                ind["name"] = name
                indicators_list.append(ind)

            except Exception as e:
                skipped += 1
                continue

        print(f"\n通过通用过滤的股票数: {len(indicators_list)}")
        print(f"跳过的股票数: {skipped}")

        if not indicators_list:
            print("\n没有股票通过过滤条件，请检查数据或调整参数")
            return

        # 计算市场统计数据（分位阈值）
        print("\n正在计算市场统计数据...")
        market_stats = calculate_market_stats(indicators_list)
        print(f"大盘代理阈值(70%分位): {market_stats['big_threshold']/1e8:.2f}亿")
        print(f"小盘代理区间(20%-60%分位): {market_stats['small_low']/1e8:.2f}亿 ~ {market_stats['small_high']/1e8:.2f}亿")
        print(f"PE低估值阈值(30%分位): {market_stats['pe_low']:.2f}")
        print(f"PB低估值阈值(30%分位): {market_stats['pb_low']:.2f}")

        # 评估策略
        print("\n正在评估策略...")
        results = []
        strategy_counts = {label: 0 for label in STRATEGY_LABELS.values()}

        for ind in indicators_list:
            hit_strategies = evaluate_strategies(ind, market_stats)
            if hit_strategies:
                ind["hit_strategies"] = ",".join(hit_strategies)
                ind["strategy_count"] = len(hit_strategies)
                results.append(ind)

                for s in hit_strategies:
                    strategy_counts[s] += 1

        print(f"\n通过策略筛选的股票数: {len(results)}")

        if not results:
            print("\n没有股票命中任何策略")
            return

        # 创建结果DataFrame
        df_result = pd.DataFrame(results)

        # 选择输出列
        output_cols = [
            "code", "name", "date", "close", "pctChg", "amount", "turn",
            "peTTM", "pbMRQ", "hit_strategies", "strategy_count",
            "MA20", "MA60", "MA120", "RSI14", "DIF", "DEA",
            "K", "D", "J",
            "avg_amount_60", "vol20", "amount20", "max_dd_60", "bias_ma60", "ret60"
        ]

        # 确保所有列都存在
        for col in output_cols:
            if col not in df_result.columns:
                df_result[col] = np.nan

        df_result = df_result[output_cols]

        # 排序：命中策略数降序，成交额降序
        df_result = df_result.sort_values(
            by=["strategy_count", "amount"],
            ascending=[False, False]
        ).reset_index(drop=True)

        # 保存CSV
        output_file = f"screen_result_{trade_day}.csv"
        df_result.to_csv(output_file, index=False, encoding="utf-8-sig")

        # 打印结果
        print("\n" + "=" * 60)
        print("各策略命中数量统计:")
        print("=" * 60)
        for label, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {label}: {count} 只")

        print("\n" + "=" * 60)
        print(f"筛选结果汇总:")
        print("=" * 60)
        print(f"  通过筛选总数: {len(results)}")
        print(f"  输出文件: {output_file}")

        print("\n" + "=" * 60)
        print("前10只命中股票:")
        print("=" * 60)
        display_cols = ["code", "name", "close", "pctChg", "hit_strategies", "strategy_count"]
        print(df_result[display_cols].head(10).to_string(index=False))

    finally:
        bs.logout()
        print("\n已登出baostock")


if __name__ == "__main__":
    main()
