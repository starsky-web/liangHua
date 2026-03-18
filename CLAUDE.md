# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

自用股票分析学习项目，使用 baostock 获取A股市场数据进行策略回测和选股。

## 常用命令

```bash
# 运行策略（使用虚拟环境）
.venv/Scripts/python.exe screen.py              # 主策略脚本（推荐）
.venv/Scripts/python.exe screen.py --date 2026-02-27   # 指定日期
.venv/Scripts/python.exe screen.py --max_stocks 200    # 测试模式

.venv/Scripts/python.exe screen_small_cap.py              # 小市值策略脚本（默认10亿以内）
.venv/Scripts/python.exe screen_small_cap.py --date 2026-02-27   # 指定日期
.venv/Scripts/python.exe screen_small_cap.py --market-cap-max 20  # 20亿市值以内
.venv/Scripts/python.exe screen_small_cap.py --market-cap-min 5 --market-cap-max 15  # 5-15亿市值

.venv/Scripts/python.exe test5.py               # 分层筛选+评分聚合模式（交互式）
.venv/Scripts/python.exe test3.py               # 带评分模型的选股策略
.venv/Scripts/python.exe analyze_market.py      # 分析市场参数
.venv/Scripts/python.exe debug_filter.py        # 调试过滤条件
```

## 依赖

- baostock: A股数据源
- pandas: 数据处理
- numpy: 数值计算
- tqdm: 进度条显示

## 核心架构

### 数据源模式
所有脚本使用 `baostock` 作为数据源，必须遵循以下模式：
```python
lg = bs.login()
assert lg.error_code == '0'
# ... 数据处理 ...
bs.logout()
```

### 主策略文件

**screen.py** - 主策略入口（最新版本），支持6种策略：
- A. LOW_VAL_BLUECHIP: 低估值蓝筹
- B. UP_TREND: 趋势向好
- C. OVERSOLD_REBOUND: 超跌反弹
- D. MACD_KDJ_RESONANCE: MACD+KDJ共振
- E. VOLUME_BREAKOUT: 放量突破
- F. SMALL_GROWTH: 小盘成长

**screen_small_cap.py** - 小市值策略入口，基于 screen.py，增加：
- 可配置的市值筛选范围（--market-cap-min, --market-cap-max）
- 默认筛选10亿市值以内股票
- 输出文件: screen_small_cap_result_{date}.csv

**test5.py** - 分层筛选+评分聚合模式：
- `StrategyConfig`: 统一配置类
- `DataSource` / `BaostockDataSource`: 数据源抽象层
- `StockScore`: 评分结果类
- 策略链：BasicFilter → RiskControl → 各策略打分 → 加权汇总

### 技术指标计算
```python
# 均线
MA5, MA10, MA20, MA60, MA120 = close.rolling(n).mean()

# MACD
DIF = EMA(close, 12) - EMA(close, 26)
DEA = EMA(DIF, 9)
MACD_HIST = 2 * (DIF - DEA)

# KDJ
RSV = (close - low_n) / (high_n - low_n) * 100
K = EMA(RSV, 3), D = EMA(K, 3), J = 3K - 2D

# RSI
RSI14 = 100 - 100 / (1 + avg_gain / avg_loss)
```

### 股票代码格式
baostock 股票代码格式：
- `sh.600000`: 上海主板
- `sz.000001`: 深圳主板
- `sh.688xxx`: 科创板（通常被排除）
- `sz.300xxx`: 创业板（通常被排除）
- `bj.`: 北交所

### 股票筛选范围
策略通常限制在"普通账户可交易"的范围：
- 主板A股：`sh.60x` 和 `sz.00x`
- 可选ETF：前缀 `51`, `56`, `58`, `15`, `16`, `18`
- 排除：科创板、创业板、北交所、B股、ST股票

### 数据处理模式
```python
# baostock 结果集转 DataFrame
def bs_to_df(rs) -> pd.DataFrame:
    data = []
    while rs.error_code == "0" and rs.next():
        data.append(rs.get_row_data())
    return pd.DataFrame(data, columns=rs.fields)

# K线数据查询
rs = bs.query_history_k_data_plus(
    code,
    "date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,isST",
    start_date=..., end_date=...,
    frequency="d",
    adjustflag="2"  # 2=前复权
)
```

### 配置模式
每个策略文件使用 dataclass 或顶部常量配置：
- `target_date` / `--date`: 目标日期
- 市值/成交额范围、筛选条件参数
- 策略权重配置