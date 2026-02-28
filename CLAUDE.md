# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

自用股票分析学习项目，使用 baostock 获取A股市场数据进行策略回测和选股。

## 常用命令

```bash
# 激活虚拟环境
.venv/Scripts/python.exe

# 运行特定策略（根据需要修改文件）
python test.py      # 拉取特定日期全市场行情
python test2.py     # 小市值策略
python test3.py     # 多条件筛选+评分策略
python main.py      # baostock 基础示例
python test4.py     # 获取股票列表
```

## 依赖

- baostock: A股数据源
- pandas: 数据处理
- numpy: 数值计算

## 代码架构

### 数据源
所有脚本使用 `baostock` 作为数据源，必须遵循以下模式：
```python
lg = bs.login()
assert lg.error_code == '0'
# ... 数据处理 ...
bs.logout()
```

### 文件结构
- `test.py`: 全市场截面数据拉取，支持配置日期和打印设置
- `test2.py`: 小市值策略，筛选主板A股中指定市值范围的股票
- `test3.py`: 综合选股策略，包含多条件过滤和评分模型，主要包含：
  - `Config` 数据类：统一配置所有筛选和评分参数
  - `get_stock_universe()`: 获取股票池（主板A股 + 可选ETF）
  - `fetch_kdata()`: 获取K线数据
  - `compute_features()`: 计算技术指标和特征
  - `pass_filters()`: 多条件筛选（价格、涨幅、成交量、均线、ST等）
  - `score_candidates()`: z-score 归一化评分
- `main.py`: baostock API 基础使用示例
- `test4.py`: 获取当日全部股票列表

### 股票代码格式
baostock 股票代码格式：
- `sh.600000`: 上海主板
- `sz.000001`: 深圳主板
- `sh.688xxx`: 科创板（通常被排除）
- `sz.300xxx`: 创业板（通常被排除）
- `bj.`: 北交所

### 股票筛选范围
策略通常限制在"普通账户可交易"的范围：
- 仅包含沪深主板（`sh.60x` 和 `sz.00x`）
- 排除：科创板（`sh.68x`）、创业板（`sz.3xx`）、北交所（`bj.`）、B股

### 配置模式
每个策略文件顶部都有配置变量，运行前需修改：
- `TARGET_DATE` / `run_date`: 目标日期
- 市值范围、筛选条件参数
- 输出控制参数（打印行数、进度间隔等）

### 数据处理模式
```python
# baostock 结果集转 DataFrame
def bs_to_df(rs) -> pd.DataFrame:
    data = []
    while rs.error_code == "0" and rs.next():
        data.append(rs.get_row_data())
    return pd.DataFrame(data, columns=rs.fields)
```

### K线数据查询
```python
rs = bs.query_history_k_data_plus(
    code,
    "date,open,high,low,close,volume,amount",
    start_date=..., end_date=...,
    frequency="d",  # 日线
    adjustflag="2"    # 2=前复权，3=不复权
)
```
