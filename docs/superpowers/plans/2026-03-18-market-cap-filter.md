# 市值筛选功能实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 screen.py 创建新文件，添加可配置的市值筛选功能，默认筛选10亿市值以内的股票。

**Architecture:** 复制 screen.py 为 screen_small_cap.py，添加市值字段获取、市值配置参数和市值过滤逻辑。

**Tech Stack:** Python, baostock, pandas, argparse

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `screen_small_cap.py` | 创建 | 新文件，基于 screen.py |
| `CLAUDE.md` | 修改 | 添加新脚本的运行说明 |

---

## Chunk 1: 创建基础文件并添加市值配置

### Task 1: 创建新文件并添加市值配置常量

**Files:**
- Create: `screen_small_cap.py`
- Modify: 无

- [ ] **Step 1: 复制 screen.py 为 screen_small_cap.py**

复制整个 screen.py 文件内容到新文件。

- [ ] **Step 2: 修改文件头部文档说明**

```python
# -*- coding: utf-8 -*-
"""
A股小市值多策略选股脚本 - 直接使用baostock数据源

基于 screen.py，添加可配置的市值筛选功能。

依赖安装:
    pip install baostock pandas numpy tqdm

注意:
    - 本脚本文件名不要命名为 baostock.py，会与库冲突
    - 确保网络连接正常，baostock需要联网获取数据
    - 首次运行可能需要较长时间（约10-30分钟，取决于股票数量）

运行示例:
    python screen_small_cap.py
    python screen_small_cap.py --date 2026-02-27
    python screen_small_cap.py --market-cap-max 10  # 筛选10亿市值以内
    python screen_small_cap.py --market-cap-min 5 --market-cap-max 20  # 5-20亿市值区间
    python screen_small_cap.py --max_stocks 200  # 测试模式
"""
```

- [ ] **Step 3: 添加市值配置常量**

在配置常量区域（约第31行后）添加：

```python
# 市值筛选配置（单位：亿元人民币）
DEFAULT_MARKET_CAP_MIN = 0       # 默认最小市值（0亿）
DEFAULT_MARKET_CAP_MAX = 10      # 默认最大市值（10亿）
```

- [ ] **Step 4: 修改 K 线字段列表，添加市值字段**

将 `KLINE_FIELDS` 修改为：

```python
# 字段列表（baostock日K数据）
KLINE_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,isST,marketCap"
```

---

## Chunk 2: 添加市值数据处理和过滤逻辑

### Task 2: 修改数据处理函数以支持市值字段

**Files:**
- Modify: `screen_small_cap.py` 中的 `get_kline_data` 函数

- [ ] **Step 1: 修改数值列转换，添加 marketCap**

找到 `get_kline_data` 函数中的 `numeric_cols` 列表（约第154行），添加 `marketCap`：

```python
    # 数值列转换
    numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount",
                    "turn", "tradestatus", "pctChg", "peTTM", "pbMRQ", "isST", "marketCap"]
```

- [ ] **Step 2: 修改 calculate_indicators 函数，添加市值字段到返回字典**

在 `calculate_indicators` 函数的返回字典中添加 `marketCap` 字段（约第313行后）：

```python
        "pbMRQ": last["pbMRQ"],
        "marketCap": last["marketCap"],  # 总市值（单位：万元）
```

---

### Task 3: 添加命令行参数和市值过滤逻辑

**Files:**
- Modify: `screen_small_cap.py` 中的 `main` 函数

- [ ] **Step 1: 添加市值命令行参数**

在 `main` 函数的 argparse 部分（约第546行后）添加：

```python
    parser = argparse.ArgumentParser(description="A股小市值多策略选股脚本")
    parser.add_argument("--date", type=str, default=None,
                        help="指定交易日，格式YYYY-MM-DD（默认最近交易日）")
    parser.add_argument("--max_stocks", type=int, default=None,
                        help="最大处理股票数量（用于测试，默认全量）")
    parser.add_argument("--market-cap-min", type=float, default=DEFAULT_MARKET_CAP_MIN,
                        help=f"最小市值（单位：亿元人民币，默认{DEFAULT_MARKET_CAP_MIN}亿）")
    parser.add_argument("--market-cap-max", type=float, default=DEFAULT_MARKET_CAP_MAX,
                        help=f"最大市值（单位：亿元人民币，默认{DEFAULT_MARKET_CAP_MAX}亿）")
    args = parser.parse_args()
```

- [ ] **Step 2: 在主循环中添加市值过滤逻辑**

在处理每只股票的主循环中，找到流动性过滤之后（约第666行后），添加市值过滤：

```python
                # 流动性过滤：近20日平均成交额 >= 5千万
                if ind["amount20"] < MIN_AVG_AMOUNT_20:
                    skipped += 1
                    continue

                # 市值过滤（marketCap 单位为万元，转换为亿元比较）
                market_cap_yi = ind["marketCap"] / 10000 if pd.notna(ind["marketCap"]) else None
                if market_cap_yi is not None:
                    if market_cap_yi < args.market_cap_min or market_cap_yi > args.market_cap_max:
                        skipped += 1
                        continue
                ind["market_cap_yi"] = market_cap_yi  # 保存市值（亿元）用于输出
```

- [ ] **Step 3: 在输出列中添加市值字段**

修改 `output_cols` 列表（约第716行），在合适位置添加 `market_cap_yi`：

```python
        output_cols = [
            "code", "name", "date", "close", "pctChg", "amount", "turn",
            "market_cap_yi",  # 市值（亿元）
            "peTTM", "pbMRQ", "hit_strategies", "strategy_count",
            "MA20", "MA60", "MA120", "RSI14", "DIF", "DEA",
            "K", "D", "J",
            "avg_amount_60", "vol20", "amount20", "max_dd_60", "bias_ma60", "ret60"
        ]
```

- [ ] **Step 4: 在输出信息中添加市值配置显示**

在打印市场统计数据之前（约第683行后），添加市值配置信息：

```python
        print(f"\n市值筛选范围: {args.market_cap_min}亿 ~ {args.market_cap_max}亿")
```

---

## Chunk 3: 更新文档

### Task 4: 更新 CLAUDE.md 文档

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 在常用命令部分添加新脚本说明**

在 `常用命令` 部分添加：

```bash
.venv/Scripts/python.exe screen_small_cap.py              # 小市值策略脚本（默认10亿以内）
.venv/Scripts/python.exe screen_small_cap.py --date 2026-02-27   # 指定日期
.venv/Scripts/python.exe screen_small_cap.py --market-cap-max 20  # 20亿市值以内
.venv/Scripts/python.exe screen_small_cap.py --market-cap-min 5 --market-cap-max 15  # 5-15亿市值
```

- [ ] **Step 2: 添加新文件描述**

在 `主策略文件` 部分添加：

```markdown
**screen_small_cap.py** - 小市值策略入口，基于 screen.py，增加：
- 可配置的市值筛选范围
- 默认筛选10亿市值以内股票
- 命令行参数：--market-cap-min, --market-cap-max
```

- [ ] **Step 3: 提交更改**

```bash
git add screen_small_cap.py CLAUDE.md
git commit -m "feat: add screen_small_cap.py with configurable market cap filter"
```

---

## 验证测试

创建完成后，运行以下命令验证：

```bash
# 测试基本功能（使用少量股票）
.venv/Scripts/python.exe screen_small_cap.py --max_stocks 50

# 测试指定日期
.venv/Scripts/python.exe screen_small_cap.py --date 2026-02-27 --max_stocks 50
```

---

## 注意事项

1. **市值单位转换**：baostock 返回的 `marketCap` 单位是万元，需要除以10000转换为亿元
2. **市值数据可用性**：部分股票可能没有市值数据（NaN），需要处理这种情况
3. **文件命名**：确保不与现有文件冲突，使用 `screen_small_cap.py` 作为新文件名