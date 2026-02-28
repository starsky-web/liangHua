# -*- coding: utf-8 -*-
"""
多策略选股系统 - 分层筛选 + 评分聚合模式

实现方式：
1. 分层筛选：基础过滤 → 基本面策略 → 技术面策略 → 风控筛选
2. 评分聚合：每个策略独立打分，按权重汇总后排序

这是专业量化系统常用的多策略实现方式。
"""

import baostock as bs
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from enum import Enum


# ==================== 枚举和类型定义 ====================

class SignalType(Enum):
    """信号类型"""
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    NEUTRAL = "中性"


class StockScore:
    """股票评分结果"""

    def __init__(self, code: str, name: str = ""):
        self.code = code
        self.name = name
        self.signals: Dict[str, SignalType] = {}  # 策略 -> 信号
        self.scores: Dict[str, float] = {}       # 策略 -> 分数(0-100)
        self.details: Dict[str, dict] = {}       # 策略 -> 详细信息
        self.total_score: float = 0.0
        self.passed_filters: List[str] = []     # 通过的筛选层
        self.failed_filters: List[str] = []      # 未通过的筛选层

    def add_signal(self, strategy: str, signal: SignalType, score: float = 0.0, detail: dict = None):
        """添加策略信号"""
        self.signals[strategy] = signal
        self.scores[strategy] = score
        if detail:
            self.details[strategy] = detail

    def add_passed_filter(self, filter_name: str):
        """记录通过的筛选层"""
        self.passed_filters.append(filter_name)

    def add_failed_filter(self, filter_name: str, reason: str = ""):
        """记录未通过的筛选层"""
        self.failed_filters.append(f"{filter_name}: {reason}")

    def calculate_total_score(self, weights: Dict[str, float]) -> float:
        """计算总分"""
        total = 0.0
        for strategy, weight in weights.items():
            if strategy in self.scores:
                total += self.scores[strategy] * weight
        self.total_score = total
        return total


# ==================== 数据源抽象 ====================

class DataSource(ABC):
    """数据源抽象接口"""

    @abstractmethod
    def login(self):
        pass

    @abstractmethod
    def logout(self):
        pass

    @abstractmethod
    def get_stock_kdata(self, code: str, start_date: str, end_date: str,
                       fields: str = "date,open,high,low,close,volume,amount,pctChg") -> pd.DataFrame:
        pass

    @abstractmethod
    def get_stock_basic(self) -> pd.DataFrame:
        pass


class BaostockDataSource(DataSource):
    """baostock 数据源实现"""

    def __init__(self):
        self._logged_in = False

    def login(self):
        lg = bs.login()
        if lg.error_code != '0':
            raise RuntimeError(f"baostock login failed: {lg.error_code} {lg.error_msg}")
        self._logged_in = True
        print("baostock login success!")
        return True

    def logout(self):
        if self._logged_in:
            bs.logout()
            self._logged_in = False
            print("baostock logout success!")
        return True

    def get_stock_kdata(self, code: str, start_date: str, end_date: str,
                       fields: str = "date,open,high,low,close,volume,amount,pctChg") -> pd.DataFrame:
        rs = bs.query_history_k_data_plus(
            code, fields, start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        if rs.error_code != "0":
            return pd.DataFrame()

        data = []
        while rs.error_code == "0" and rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pctChg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        return df

    def get_stock_basic(self) -> pd.DataFrame:
        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            raise RuntimeError(f"get stock basic failed: {rs.error_code} {rs.error_msg}")
        data = []
        while rs.error_code == "0" and rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)
        return df


# ==================== 配置类 ====================

@dataclass
class StrategyConfig:
    """策略配置"""
    # 通用配置
    target_date: str = None
    lookback_days: int = 60      # 减少回看天数（原120）

    # ETF配置
    include_etf: bool = True  # 是否包含ETF
    etf_prefixes: tuple = ("51", "56", "58", "15", "16", "18")  # 常见ETF前缀
    include_mainboard_a: bool = True  # 是否包含主板A股

    # 第0层：基础过滤
    min_price: float = 1.8        # 降低最低价格
    max_price: float = 300.0      # 降低最高价格
    min_avg_amount: float = 3e5   # 降低到30万元
    etf_min_avg_amount: float = 1e5  # ETF降低到10万元

    # 第1层：低估值蓝筹股策略
    low_pe_max: float = 15.0
    low_pb_max: float = 2.0
    low_dividend_min: float = 3.0
    low_volatility_max: float = 3.0

    # 第2层：趋势向好股策略
    trend_ma_short: int = 20
    trend_ma_long: int = 60
    trend_rise_min: float = 10.0
    trend_volume_ratio_min: float = 1.2

    # 第3层：超跌反弹策略
    oversold_max_drop: float = 30.0
    oversold_rsi_min: int = 25

    # 第4层：MACD+KDJ共振策略
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    kdj_n: int = 9

    # 第5层：放量突破策略
    breakout_ma: int = 20
    breakout_volume_ratio: float = 2.0

    # 第6层：小盘成长股策略
    smallcap_growth_min: float = 15.0
    smallcap_growth_max: float = 50.0

    # 策略权重（总和为1.0）
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "低估值蓝筹股": 0.20,
        "趋势向好股": 0.25,
        "超跌反弹": 0.15,
        "MACD+KDJ共振": 0.15,
        "放量突破": 0.15,
        "小盘成长股": 0.10,
    })

    # 风控筛选
    max_single_drop_5d: float = 15.0  # 5日内最大单日跌幅
    exclude_st: bool = True


# ==================== 技术指标计算 ====================

class TechnicalIndicators:
    """技术指标计算工具"""

    @staticmethod
    def ma(data: pd.Series, n: int) -> pd.Series:
        return data.rolling(window=n).mean()

    @staticmethod
    def ema(data: pd.Series, n: int) -> pd.Series:
        return data.ewm(span=n, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, n: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        dif = ema_fast - ema_slow
        dea = TechnicalIndicators.ema(dif, signal)
        macd_hist = (dif - dea) * 2
        return dif, dea, macd_hist

    @staticmethod
    def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9):
        low_min = low.rolling(window=n).min()
        high_max = high.rolling(window=n).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j


# ==================== 策略基类 ====================

class Strategy(ABC):
    """策略基类"""

    def __init__(self, name: str, config: StrategyConfig):
        self.name = name
        self.config = config

    @abstractmethod
    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        """
        评估单只股票
        返回：是否通过该策略
        """
        pass


# ==================== 第0层：基础过滤 ====================

class BasicFilter(Strategy):
    """基础过滤层"""

    def __init__(self, config: StrategyConfig):
        super().__init__("基础过滤", config)

    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        if df.empty or len(df) < 20:
            stock_score.add_failed_filter(self.name, "数据不足")
            return False

        last = df.iloc[-1]

        # 判断是否为ETF
        is_etf_flag = is_etf(code, self.config.etf_prefixes)

        # 价格过滤（ETF价格范围通常较宽）
        min_price = self.config.min_price
        max_price = self.config.max_price

        if last["close"] < min_price:
            stock_score.add_failed_filter(self.name, f"价格过低 {last['close']:.2f}")
            return False
        if last["close"] > max_price:
            stock_score.add_failed_filter(self.name, f"价格过高 {last['close']:.2f}")
            return False

        # 成交额过滤（ETF使用不同的阈值）
        avg_amount = df["amount"].tail(20).mean()
        min_amount = self.config.etf_min_avg_amount if is_etf_flag else self.config.min_avg_amount

        if avg_amount < min_amount:
            stock_score.add_failed_filter(self.name, f"成交额过低 {avg_amount/1e6:.2f}万")
            return False

        # 记录类型信息
        stock_type = "ETF" if is_etf_flag else "股票"
        stock_score.details["stock_type"] = stock_type

        stock_score.add_passed_filter(self.name)
        return True


# ==================== 第1层：低估值蓝筹股策略 ====================

class LowValuationStrategy(Strategy):
    """低估值蓝筹股策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__("低估值蓝筹股", config)

    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        # 计算指标
        volatility = df["close"].pct_change().tail(20).std() * 100
        avg_volume_60 = df["volume"].tail(60).mean()

        # 评分因子
        score = 0.0
        factors = {}

        # 波动率评分（越低越好）
        if volatility < self.config.low_volatility_max:
            vol_score = (self.config.low_volatility_max - volatility) / self.config.low_volatility_max * 40
            score += vol_score
            factors["volatility_score"] = vol_score
        else:
            factors["volatility_score"] = 0

        # 成交量评分
        if avg_volume_60 > 1e6:
            vol_score = min(avg_volume_60 / 1e7 * 30, 30)
            score += vol_score
            factors["volume_score"] = vol_score
        else:
            factors["volume_score"] = 0

        # 价格稳定性评分
        avg_price = df["close"].tail(60).mean()
        if 5.0 <= avg_price <= 100.0:
            price_score = 30
            score += price_score
            factors["price_score"] = 30
        else:
            factors["price_score"] = 0

        # 判断信号
        signal = SignalType.BUY if score > 50 else (SignalType.HOLD if score > 30 else SignalType.NEUTRAL)

        stock_score.add_signal(
            self.name,
            signal,
            score,
            {"volatility": volatility, "avg_volume": avg_volume_60, "avg_price": avg_price}
        )
        stock_score.details[self.name].update(factors)

        return signal != SignalType.NEUTRAL


# ==================== 第2层：趋势向好股策略 ====================

class TrendFollowingStrategy(Strategy):
    """趋势向好股策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__("趋势向好股", config)

    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        # 检查数据长度
        if len(df) < 21:
            stock_score.add_signal(
                self.name,
                SignalType.NEUTRAL,
                0.0,
                {"reason": "数据不足"}
            )
            return False

        df = df.copy()
        df["ma20"] = TechnicalIndicators.ma(df["close"], self.config.trend_ma_short)
        df["ma60"] = TechnicalIndicators.ma(df["close"], self.config.trend_ma_long)
        df["ma120"] = TechnicalIndicators.ma(df["close"], 120)

        last = df.iloc[-1]

        # 安全访问20天前的数据
        idx_20d_ago = max(0, len(df) - 21)
        close_20d_ago = df["close"].iloc[idx_20d_ago]
        close_now = df["close"].iloc[-1]
        rise_20d = (close_now / close_20d_ago - 1) * 100 if close_20d_ago > 0 else 0

        avg_vol_20 = df["volume"].tail(min(20, len(df))).mean()
        avg_vol_60 = df["volume"].tail(min(60, len(df))).mean()
        volume_ratio = avg_vol_20 / avg_vol_60 if pd.notna(avg_vol_60) and avg_vol_60 > 0 else 0

        score = 0.0
        factors = {}

        # 多头排列评分
        if pd.notna(last["ma60"]) and pd.notna(last["ma120"]) and last["close"] > last["ma60"] > last["ma120"]:
            score += 35
            factors["arrangement_score"] = 35
        else:
            factors["arrangement_score"] = 0

        # 均线向上评分
        if len(df) >= 5:
            ma20_last = df["ma20"].iloc[-1]
            ma20_5d_ago = df["ma20"].iloc[-5]
            if pd.notna(ma20_last) and pd.notna(ma20_5d_ago) and ma20_last > ma20_5d_ago:
                score += 25
                factors["ma_direction_score"] = 25
            else:
                factors["ma_direction_score"] = 0
        else:
            factors["ma_direction_score"] = 0

        # 涨幅评分
        if rise_20d > self.config.trend_rise_min:
            rise_score = min((rise_20d - self.config.trend_rise_min) / 10 * 20, 20)
            score += rise_score
            factors["rise_score"] = rise_score
        else:
            factors["rise_score"] = 0

        # 成交量放大评分
        if volume_ratio > self.config.trend_volume_ratio_min:
            vol_score = min((volume_ratio - self.config.trend_volume_ratio_min) * 10, 20)
            score += vol_score
            factors["volume_ratio_score"] = vol_score
        else:
            factors["volume_ratio_score"] = 0

        signal = SignalType.BUY if score > 60 else (SignalType.HOLD if score > 40 else SignalType.NEUTRAL)

        stock_score.add_signal(
            self.name,
            signal,
            score,
            {
                "rise_20d": rise_20d,
                "volume_ratio": volume_ratio,
                "ma20": last["ma20"],
                "ma60": last["ma60"],
                "ma120": last["ma120"]
            }
        )
        stock_score.details[self.name].update(factors)

        return signal != SignalType.NEUTRAL


# ==================== 第3层：超跌反弹策略 ====================

class OversoldBounceStrategy(Strategy):
    """超跌反弹策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__("超跌反弹", config)

    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        df = df.copy()
        df["rsi"] = TechnicalIndicators.rsi(df["close"], 14)

        high_20d = df["high"].tail(20).max()
        max_drop = (high_20d - df["close"].iloc[-1]) / high_20d * 100

        recent_lows = df["low"].tail(5)
        is_stabilizing = recent_lows.iloc[-1] > recent_lows.min()

        last = df.iloc[-1]

        score = 0.0
        factors = {}

        # 跌幅评分（跌幅越大，反弹潜力越大）
        if max_drop > self.config.oversold_max_drop:
            drop_score = min((max_drop - self.config.oversold_max_drop) / 20 * 40, 40)
            score += drop_score
            factors["drop_score"] = drop_score
        else:
            factors["drop_score"] = 0

        # RSI评分（RSI越低，超跌越严重）
        if last["rsi"] < self.config.oversold_rsi_min:
            rsi_score = (self.config.oversold_rsi_min - last["rsi"]) / self.config.oversold_rsi_min * 40
            score += rsi_score
            factors["rsi_score"] = rsi_score
        else:
            factors["rsi_score"] = 0

        # 止跌迹象评分
        if is_stabilizing:
            score += 20
            factors["stabilize_score"] = 20
        else:
            factors["stabilize_score"] = 0

        signal = SignalType.BUY if score > 50 else (SignalType.HOLD if score > 30 else SignalType.NEUTRAL)

        stock_score.add_signal(
            self.name,
            signal,
            score,
            {"max_drop": max_drop, "rsi": last["rsi"], "is_stabilizing": is_stabilizing}
        )
        stock_score.details[self.name].update(factors)

        return signal != SignalType.NEUTRAL


# ==================== 第4层：MACD+KDJ共振策略 ====================

class MacdKdjResonanceStrategy(Strategy):
    """MACD+KDJ共振策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__("MACD+KDJ共振", config)

    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        # 检查数据长度（需要至少3天来比较）
        if len(df) < 3:
            stock_score.add_signal(
                self.name,
                SignalType.NEUTRAL,
                0.0,
                {"reason": "数据不足"}
            )
            return False

        df = df.copy()
        df["dif"], df["dea"], df["macd_hist"] = TechnicalIndicators.macd(
            df["close"], self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )
        df["k"], df["d"], df["j"] = TechnicalIndicators.kdj(
            df["high"], df["low"], df["close"], self.config.kdj_n
        )

        last = df.iloc[-1]

        # MACD金叉判断（安全访问）
        dif_last = df["dif"].iloc[-1]
        dif_prev = df["dif"].iloc[-2] if len(df) >= 2 else df["dif"].iloc[-1]
        dea_last = df["dea"].iloc[-1]
        dea_prev = df["dea"].iloc[-2] if len(df) >= 2 else df["dea"].iloc[-1]

        macd_cross = (
            pd.notna(dif_last) and pd.notna(dea_last) and
            pd.notna(dif_prev) and pd.notna(dea_prev) and
            dif_last > dea_last and dif_prev <= dea_prev
        )

        # KDJ金叉判断（安全访问）
        k_last = df["k"].iloc[-1]
        k_prev = df["k"].iloc[-2] if len(df) >= 2 else df["k"].iloc[-1]
        d_last = df["d"].iloc[-1]
        d_prev = df["d"].iloc[-2] if len(df) >= 2 else df["d"].iloc[-1]

        kdj_cross = (
            pd.notna(k_last) and pd.notna(d_last) and
            pd.notna(k_prev) and pd.notna(d_prev) and
            k_last > d_last and k_prev <= d_prev and
            k_last < 30 and d_last < 30
        )

        score = 0.0
        factors = {}

        # MACD金叉评分
        if pd.notna(dif_last) and pd.notna(dea_last):
            if macd_cross:
                score += 50
                factors["macd_cross_score"] = 50
            elif dif_last > dea_last:  # 已在多头
                score += 30
                factors["macd_cross_score"] = 30
            else:
                factors["macd_cross_score"] = 0
        else:
            factors["macd_cross_score"] = 0

        # KDJ金叉评分
        if pd.notna(k_last) and pd.notna(d_last):
            if kdj_cross:
                score += 50
                factors["kdj_cross_score"] = 50
            elif k_last > d_last:  # 已在多头
                score += 20
                factors["kdj_cross_score"] = 20
            else:
                factors["kdj_cross_score"] = 0
        else:
            factors["kdj_cross_score"] = 0

        signal = SignalType.BUY if score > 60 else (SignalType.HOLD if score > 40 else SignalType.NEUTRAL)

        stock_score.add_signal(
            self.name,
            signal,
            score,
            {
                "macd_cross": macd_cross,
                "kdj_cross": kdj_cross,
                "dif": last["dif"],
                "dea": last["dea"],
                "k": last["k"],
                "d": last["d"]
            }
        )
        stock_score.details[self.name].update(factors)

        return signal != SignalType.NEUTRAL


# ==================== 第5层：放量突破策略 ====================

class VolumeBreakoutStrategy(Strategy):
    """放量突破策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__("放量突破", config)

    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        df = df.copy()
        df["ma20"] = TechnicalIndicators.ma(df["close"], self.config.breakout_ma)
        df["vol_ma20"] = TechnicalIndicators.ma(df["volume"], self.config.breakout_ma)

        last = df.iloc[-1]
        high_20d = df["high"].tail(20).max()

        volume_ratio = last["volume"] / last["vol_ma20"] if last["vol_ma20"] > 0 else 0

        score = 0.0
        factors = {}

        # 价格突破评分
        if last["close"] > last["ma20"]:
            score += 30
            factors["price_break_score"] = 30
            if last["close"] > high_20d * 0.98:  # 创新高
                score += 20
                factors["new_high_score"] = 20
            else:
                factors["new_high_score"] = 0
        else:
            factors["price_break_score"] = 0
            factors["new_high_score"] = 0

        # 放量评分
        if volume_ratio > self.config.breakout_volume_ratio:
            vol_score = min((volume_ratio - self.config.breakout_volume_ratio) / 2 * 50, 50)
            score += vol_score
            factors["volume_break_score"] = vol_score
        else:
            factors["volume_break_score"] = 0

        signal = SignalType.BUY if score > 60 else (SignalType.HOLD if score > 40 else SignalType.NEUTRAL)

        stock_score.add_signal(
            self.name,
            signal,
            score,
            {
                "volume_ratio": volume_ratio,
                "close_vs_ma20": last["close"] / last["ma20"],
                "close_vs_high_20d": last["close"] / high_20d
            }
        )
        stock_score.details[self.name].update(factors)

        return signal != SignalType.NEUTRAL


# ==================== 第6层：小盘成长股策略 ====================

class SmallcapGrowthStrategy(Strategy):
    """小盘成长股策略（不适用于ETF）"""

    def __init__(self, config: StrategyConfig):
        super().__init__("小盘成长股", config)

    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        # ETF不适合用小盘成长股策略，直接给中性信号
        if is_etf(code, self.config.etf_prefixes):
            stock_score.add_signal(
                self.name,
                SignalType.NEUTRAL,
                0.0,
                {"reason": "ETF不适用小盘成长股策略"}
            )
            return False

        # 检查数据长度，避免索引越界
        if len(df) < 20:
            stock_score.add_signal(
                self.name,
                SignalType.NEUTRAL,
                0.0,
                {"reason": "数据不足"}
            )
            return False

        # 安全访问60天前的数据
        idx_60d_ago = max(0, len(df) - 61)
        if idx_60d_ago == 0:
            # 数据不足60天，使用最早的数据
            close_60d_ago = df["close"].iloc[0]
        else:
            close_60d_ago = df["close"].iloc[idx_60d_ago]

        close_now = df["close"].iloc[-1]
        if close_60d_ago > 0:
            rise_60d = (close_now / close_60d_ago - 1) * 100
        else:
            rise_60d = 0

        avg_vol_20 = df["volume"].tail(min(20, len(df))).mean()
        avg_price_60 = df["close"].tail(min(60, len(df))).mean()
        last = df.iloc[-1]

        score = 0.0
        factors = {}

        # 涨幅评分（适中涨幅更好）
        if self.config.smallcap_growth_min < rise_60d < self.config.smallcap_growth_max:
            score += 40
            factors["growth_score"] = 40
        elif rise_60d > self.config.smallcap_growth_max:
            factors["growth_score"] = 10  # 涨幅过大，减分
        else:
            factors["growth_score"] = 0

        # 成交活跃度评分
        if pd.notna(avg_vol_20) and avg_vol_20 > 5e5:
            vol_score = min(avg_vol_20 / 1e6 * 30, 30)
            score += vol_score
            factors["volume_score"] = vol_score
        else:
            factors["volume_score"] = 0

        # 小盘股特征评分
        if 3.0 < avg_price_60 < 50.0:
            score += 30
            factors["size_score"] = 30
        else:
            factors["size_score"] = 0

        signal = SignalType.BUY if score > 60 else (SignalType.HOLD if score > 40 else SignalType.NEUTRAL)

        stock_score.add_signal(
            self.name,
            signal,
            score,
            {"rise_60d": rise_60d, "avg_vol_20": avg_vol_20, "avg_price_60": avg_price_60}
        )
        stock_score.details[self.name].update(factors)

        return signal != SignalType.NEUTRAL


# ==================== 风控策略 ====================

class RiskControlStrategy(Strategy):
    """风控策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__("风控筛选", config)

    def evaluate(self, code: str, df: pd.DataFrame, stock_score: StockScore) -> bool:
        last5 = df.tail(5)
        max_single_drop = last5["pctChg"].min()  # pctChg为负表示跌幅

        if max_single_drop < -self.config.max_single_drop_5d:
            stock_score.add_failed_filter(self.name, f"单日跌幅过大 {max_single_drop:.2f}%")
            return False

        stock_score.add_passed_filter(self.name)
        return True


# ==================== 多策略选股器 ====================

class MultiStrategySelector:
    """多策略选股器 - 分层筛选 + 评分聚合"""

    def __init__(self, data_source: DataSource, config: StrategyConfig):
        self.ds = data_source
        self.cfg = config
        self.strategies: List[Strategy] = []

        # 初始化策略链
        self._init_strategies()

    def _init_strategies(self):
        """初始化策略链"""
        self.strategies = [
            BasicFilter(self.cfg),
            RiskControlStrategy(self.cfg),
            LowValuationStrategy(self.cfg),
            TrendFollowingStrategy(self.cfg),
            OversoldBounceStrategy(self.cfg),
            MacdKdjResonanceStrategy(self.cfg),
            VolumeBreakoutStrategy(self.cfg),
            SmallcapGrowthStrategy(self.cfg),
        ]

    def _get_date_range(self) -> tuple:
        """获取日期范围"""
        end = self.cfg.target_date if self.cfg.target_date else datetime.now().strftime("%Y-%m-%d")
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=self.cfg.lookback_days)).strftime("%Y-%m-%d")
        return start, end

    def _check_data_quality(self, df: pd.DataFrame, min_days: int = 20) -> bool:
        """检查数据质量"""
        return not df.empty and len(df) >= min_days and not df[["open", "high", "low", "close"]].isnull().any().any()

    def evaluate_stock(self, code: str, name: str = "") -> Optional[StockScore]:
        """
        评估单只股票

        分层流程：
        1. 基础过滤 → 2. 风控筛选 → 3. 各策略打分 → 4. 计算总分
        """
        stock_score = StockScore(code, name)

        # 获取K线数据
        start, end = self._get_date_range()
        df = self.ds.get_stock_kdata(code, start, end)
        # 60个自然日约40个交易日，取1/3约20天，降低要求避免因短期停牌被过滤
        min_required_days = max(self.cfg.lookback_days // 3, 20)
        if not self._check_data_quality(df, min_required_days):
            stock_score.add_failed_filter("数据质量", "K线数据不足或异常")
            return stock_score

        # 分层筛选和评分
        for strategy in self.strategies:
            passed = strategy.evaluate(code, df, stock_score)

            # 如果是过滤类策略（BasicFilter, RiskControl），未通过则直接返回
            if isinstance(strategy, (BasicFilter, RiskControlStrategy)) and not passed:
                return stock_score

        # 计算加权总分
        total_score = stock_score.calculate_total_score(self.cfg.strategy_weights)

        return stock_score

    def batch_evaluate(self, stock_codes: List[str], stock_names: Dict[str, str] = None,
                     progress_every: int = 50) -> Tuple[List[StockScore], Dict[str, int]]:
        """
        批量评估股票
        返回：(通过基础过滤的结果, 失败原因统计)
        """
        results = []
        fail_reasons = {}  # 失败原因 -> 数量

        for i, code in enumerate(stock_codes, 1):
            name = stock_names.get(code, "") if stock_names else ""

            stock_score = self.evaluate_stock(code, name)

            # 统计失败原因
            if stock_score.failed_filters:
                for reason in stock_score.failed_filters:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

            if stock_score.passed_filters:  # 至少通过基础过滤
                results.append(stock_score)

            if i % progress_every == 0 or i == len(stock_codes):
                passed = len([r for r in results if not r.failed_filters])
                print(f"进度: [{i}/{len(stock_codes)}] 通过过滤: {passed}/{len(results)}")

        return results, fail_reasons

    def get_ranked_results(self, stock_scores: List[StockScore], top_n: int = 20) -> pd.DataFrame:
        """获取排序后的结果"""
        # 过滤掉未通过基础过滤的
        valid_scores = [s for s in stock_scores if not any("基础过滤" in f or "风控筛选" in f for f in s.failed_filters)]

        # 按总分降序排序
        sorted_scores = sorted(valid_scores, key=lambda x: x.total_score, reverse=True)

        # 转换为DataFrame
        rows = []
        for score in sorted_scores[:top_n]:
            # 汇总信号
            buy_signals = sum(1 for s in score.signals.values() if s == SignalType.BUY)
            hold_signals = sum(1 for s in score.signals.values() if s == SignalType.HOLD)

            row = {
                "code": score.code,
                "name": score.name,
                "total_score": round(score.total_score, 2),
                "buy_signals": buy_signals,
                "hold_signals": hold_signals,
            }

            # 添加各策略得分
            for strategy_name in self.cfg.strategy_weights.keys():
                row[f"{strategy_name}_score"] = round(score.scores.get(strategy_name, 0), 2)
                row[f"{strategy_name}_signal"] = score.signals.get(strategy_name, SignalType.NEUTRAL).value

            rows.append(row)

        return pd.DataFrame(rows)


# ==================== 用户输入处理 ====================

def get_user_input() -> Tuple:
    """获取用户输入"""
    print("\n" + "="*60)
    print("       多策略选股系统 - 分层筛选 + 评分聚合")
    print("="*60)

    print("\n请输入股票代码（用逗号或空格分隔）:")
    print("  示例: sh.600000,sz.000001")
    print("  输入 'auto' 自动获取股票列表")
    stock_input = input("> ").strip()

    if stock_input.lower() == 'auto':
        # 询问是否包含ETF
        print("\n是否包含ETF？（输入 'y' 或 'n'，默认 'y'）:")
        etf_input = input("> ").strip()
        include_etf = etf_input.lower() != 'n'
        return 'auto', None, include_etf

    stock_codes = [item.strip() for item in stock_input.replace(',', ' ').split() if item.strip()]

    print("\n请选择输出数量（前N名）:")
    print("  默认: 20")
    top_n_input = input("> ").strip()
    top_n = int(top_n_input) if top_n_input.isdigit() else 20

    return stock_codes, top_n, None  # 手动输入时不需要ETF选项


def is_mainboard_a_share(code: str) -> bool:
    """判断是否为主板A股"""
    return isinstance(code, str) and (code.startswith("sh.60") or code.startswith("sz.00"))


def is_etf(code: str, etf_prefixes: tuple = ("51", "56", "58", "15", "16", "18")) -> bool:
    """判断是否为ETF"""
    if not isinstance(code, str):
        return False
    # 提取纯数字部分
    parts = code.split(".")
    if len(parts) != 2:
        return False
    pure_code = parts[1]
    return pure_code.startswith(etf_prefixes)


def is_valid_stock(code: str, include_mainboard_a: bool = True,
                  include_etf: bool = True, etf_prefixes: tuple = ("51", "56", "58", "15", "16", "18")) -> bool:
    """判断是否为有效股票（主板A股或ETF）"""
    if not isinstance(code, str):
        return False

    # 排除不需要的板块
    parts = code.split(".")
    if len(parts) != 2:
        return False

    pure_code = parts[1]
    exchange = parts[0].lower()

    # 排除科创板、创业板、北交所等
    excluded_prefixes = ("300", "301", "688", "689", "8", "4", "9")
    if pure_code.startswith(excluded_prefixes):
        return False

    # 排除B股
    if exchange in ("sh", "sz") and pure_code.endswith(("2", "9")):
        return False

    # 检查主板A股
    if include_mainboard_a and is_mainboard_a_share(code):
        return True

    # 检查ETF
    if include_etf and is_etf(code, etf_prefixes):
        return True

    return False


def get_auto_stock_codes(ds: DataSource, include_etf: bool = True,
                         etf_prefixes: tuple = ("51", "56", "58", "15", "16", "18")) -> List[str]:
    """自动获取股票列表（主板A股 + 可选ETF）"""
    df = ds.get_stock_basic()

    # 过滤有效的股票代码
    mask = df["code"].apply(lambda x: is_valid_stock(x, include_mainboard_a=True,
                                                     include_etf=include_etf, etf_prefixes=etf_prefixes))
    codes = df[mask]["code"].tolist()

    # 统计ETF和主板A股数量
    etf_count = sum(1 for code in codes if is_etf(code, etf_prefixes))
    mainboard_count = len(codes) - etf_count

    print(f"  主板A股: {mainboard_count} 只")
    if include_etf:
        print(f"  ETF: {etf_count} 只")

    return codes


# ==================== 主程序 ====================

def main():
    # 创建配置
    config = StrategyConfig()
    # lookback_days 已在配置中设置为60天

    # 获取用户输入
    stock_codes, top_n, include_etf = get_user_input()

    # 初始化
    ds = BaostockDataSource()
    selector = MultiStrategySelector(ds, config)

    try:
        ds.login()

        # 处理股票代码
        if stock_codes == 'auto':
            print("\n正在获取股票列表...")
            stock_codes = get_auto_stock_codes(ds, include_etf=include_etf,
                                               etf_prefixes=config.etf_prefixes)
            print(f"获取到 {len(stock_codes)} 只标的")

        if not stock_codes:
            print("\n错误: 没有有效的股票代码")
            return

        print(f"\n开始评估 {len(stock_codes)} 只标的...\n")

        # 批量评估
        stock_scores, fail_reasons = selector.batch_evaluate(stock_codes, progress_every=100)

        # 显示失败原因统计
        if fail_reasons:
            print("\n" + "="*80)
            print("过滤失败原因统计")
            print("="*80)
            sorted_reasons = sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_reasons:
                print(f"  {reason}: {count} 只 ({count/len(stock_codes)*100:.1f}%)")

        if not stock_scores:
            print("\n没有标的通过基础过滤")
            print("\n建议：")
            print("  1. 降低成交额阈值 (min_avg_amount)")
            print("  2. 扩大价格范围 (min_price, max_price)")
            print("  3. 减少回看天数 (lookback_days)")
            return

        # 获取排序结果
        results_df = selector.get_ranked_results(stock_scores, top_n)

        # 输出结果
        print("\n" + "="*80)
        print(f"                        选股结果 - 前{top_n}名")
        print("="*80)

        # 基础列
        display_cols = ["code", "name", "total_score", "buy_signals", "hold_signals"]
        for strategy in config.strategy_weights.keys():
            display_cols.extend([f"{strategy}_score"])

        print(results_df[display_cols].to_string(index=False))

        # 详细分析第一名
        if len(stock_scores) > 0:
            top_score = sorted(stock_scores, key=lambda x: x.total_score, reverse=True)[0]
            stock_type = top_score.details.get("stock_type", "股票")
            print("\n" + "="*80)
            print(f"第一名详细分析: {top_score.code} {top_score.name} ({stock_type})")
            print("="*80)
            print(f"总分: {top_score.total_score:.2f}")
            print(f"\n通过的筛选层: {', '.join(top_score.passed_filters)}")
            if top_score.failed_filters:
                print(f"未通过的筛选层: {', '.join(top_score.failed_filters)}")

            print("\n各策略信号:")
            for strategy, signal in top_score.signals.items():
                print(f"  {strategy}: {signal.value} (得分: {top_score.scores[strategy]:.2f})")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        ds.logout()


if __name__ == "__main__":
    main()
