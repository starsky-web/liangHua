# -*- coding: utf-8 -*-
"""分析A股市场基本情况，确定合理参数"""

import baostock as bs
import pandas as pd
import numpy as np

ds = bs.login()
print(f"登录状态: {ds.error_code} {ds.error_msg}")

# 获取主板A股列表
rs = bs.query_stock_basic()
data = []
while rs.error_code == "0" and rs.next():
    data.append(rs.get_row_data())
df_stocks = pd.DataFrame(data, columns=rs.fields)

def is_mainboard_a_share(code: str) -> bool:
    return isinstance(code, str) and (code.startswith("sh.60") or code.startswith("sz.00"))

df_stocks = df_stocks[df_stocks["code"].apply(is_mainboard_a_share)].reset_index(drop=True)
print(f"\n主板A股总数: {len(df_stocks)}")

# 随机抽取200只股票分析
sample_codes = df_stocks["code"].tolist()[:200]

# 统计数据
prices = []
amounts = []
data_lengths = []

for i, code in enumerate(sample_codes):
    rs = bs.query_history_k_data_plus(
        code,
        "date,close,amount",
        start_date="2025-12-01",
        end_date="2026-02-27",
        frequency="d",
        adjustflag="2"
    )

    data = []
    while rs.error_code == "0" and rs.next():
        data.append(rs.get_row_data())
    df = pd.DataFrame(data, columns=rs.fields)

    if not df.empty:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna()

        if len(df) > 0:
            prices.append(df["close"].iloc[-1])
            avg_amount = df["amount"].tail(20).mean()
            amounts.append(avg_amount)
            data_lengths.append(len(df))

    if (i + 1) % 50 == 0:
        print(f"已处理: {i+1}/{len(sample_codes)}")

# 转为numpy数组便于分析
prices = np.array(prices)
amounts = np.array(amounts)
data_lengths = np.array(data_lengths)

print("\n" + "="*70)
print("A股市场统计分析")
print("="*70)

# 价格分析
print(f"\n【价格分析】")
print(f"  样本数: {len(prices)}")
print(f"  平均价: {np.mean(prices):.2f} 元")
print(f"  中位数: {np.median(prices):.2f} 元")
print(f"  最小值: {np.min(prices):.2f} 元")
print(f"  最大值: {np.max(prices):.2f} 元")
print(f"  25分位: {np.percentile(prices, 25):.2f} 元")
print(f"  75分位: {np.percentile(prices, 75):.2f} 元")
print(f"  价格<3元: {np.sum(prices < 3)} 只 ({np.sum(prices < 3)/len(prices)*100:.1f}%)")
print(f"  价格<2元: {np.sum(prices < 2)} 只 ({np.sum(prices < 2)/len(prices)*100:.1f}%)")
print(f"  价格>100元: {np.sum(prices > 100)} 只 ({np.sum(prices > 100)/len(prices)*100:.1f}%)")

# 成交额分析
print(f"\n【成交额分析】(20日平均)")
print(f"  样本数: {len(amounts)}")
print(f"  平均成交额: {np.mean(amounts)/1e6:.2f} 亿元")
print(f"  中位数: {np.median(amounts)/1e6:.2f} 亿元")
print(f"  最小值: {np.min(amounts)/1e6:.2f} 亿元")
print(f"  最大值: {np.max(amounts)/1e6:.2f} 亿元")
print(f"  25分位: {np.percentile(amounts, 25)/1e6:.2f} 亿元")
print(f"  75分位: {np.percentile(amounts, 75)/1e6:.2f} 亿元")
print(f"\n  成交额<50万: {np.sum(amounts < 50e4)} 只 ({np.sum(amounts < 50e4)/len(amounts)*100:.1f}%)")
print(f"  成交额<100万: {np.sum(amounts < 100e4)} 只 ({np.sum(amounts < 100e4)/len(amounts)*100:.1f}%)")
print(f"  成交额<200万: {np.sum(amounts < 200e4)} 只 ({np.sum(amounts < 200e4)/len(amounts)*100:.1f}%)")
print(f"  成交额<500万: {np.sum(amounts < 500e4)} 只 ({np.sum(amounts < 500e4)/len(amounts)*100:.1f}%)")
print(f"  成交额<1000万: {np.sum(amounts < 1000e4)} 只 ({np.sum(amounts < 1000e4)/len(amounts)*100:.1f}%)")
print(f"  成交额<2000万: {np.sum(amounts < 2000e4)} 只 ({np.sum(amounts < 2000e4)/len(amounts)*100:.1f}%)")
print(f"  成交额>5000万: {np.sum(amounts > 5000e4)} 只 ({np.sum(amounts > 5000e4)/len(amounts)*100:.1f}%)")

# 数据长度分析
print(f"\n【数据长度分析】")
print(f"  样本数: {len(data_lengths)}")
print(f"  平均数据行数: {np.mean(data_lengths):.0f}")
print(f"  中位数: {np.median(data_lengths):.0f}")
print(f"  最小值: {np.min(data_lengths):.0f}")
print(f"  最大值: {np.max(data_lengths):.0f}")
print(f"\n  数据<20条: {np.sum(data_lengths < 20)} 只 ({np.sum(data_lengths < 20)/len(data_lengths)*100:.1f}%)")
print(f"  数据<30条: {np.sum(data_lengths < 30)} 只 ({np.sum(data_lengths < 30)/len(data_lengths)*100:.1f}%)")
print(f"  数据<40条: {np.sum(data_lengths < 40)} 只 ({np.sum(data_lengths < 40)/len(data_lengths)*100:.1f}%)")
print(f"  数据<50条: {np.sum(data_lengths < 50)} 只 ({np.sum(data_lengths < 50)/len(data_lengths)*100:.1f}%)")

# 推荐参数
print("\n" + "="*70)
print("【推荐参数】")
print("="*70)

# 根据中位数和25分位推荐
min_price = np.percentile(prices, 5)  # 排除最低5%
max_price = np.percentile(prices, 95)  # 排除最高5%
min_amount = np.percentile(amounts[amounts > 0], 20)  # 排除最低20%
min_data_days = int(np.percentile(data_lengths, 10))  # 排除最低10%

print(f"\n基于统计分析的推荐：")
print(f"  min_price: {min_price:.2f} 元 (建议: 2.0 元)")
print(f"  max_price: {max_price:.2f} 元 (建议: 300 元)")
print(f"  min_avg_amount: {min_amount/1e6:.2f} 万元 (建议: 30 万元)")
print(f"  最小数据天数: {min_data_days} (建议: 20)")

# 测试不同参数组合的通过率
print("\n" + "="*70)
print("【参数组合测试】")
print("="*70)

test_configs = [
    {"name": "宽松组合", "min_price": 1.0, "max_price": 500, "min_amount": 10e5, "min_days": 15},
    {"name": "中等组合", "min_price": 2.0, "max_price": 300, "min_amount": 30e5, "min_days": 20},
    {"name": "严格组合", "min_price": 3.0, "max_price": 200, "min_amount": 50e5, "min_days": 25},
]

for config in test_configs:
    name = config["name"]
    min_p = config["min_price"]
    max_p = config["max_price"]
    min_a = config["min_amount"]
    min_d = config["min_days"]

    passed = sum(1 for p, a, d in zip(prices, amounts, data_lengths)
                 if min_p <= p <= max_p and a >= min_a and d >= min_d)
    total = len(prices)
    print(f"\n{name}:")
    print(f"  参数: 价格{min_p}-{max_p}元, 成交额>={min_a/1e4:.0f}万, 数据>={min_d}条")
    print(f"  通过率: {passed}/{total} = {passed/total*100:.1f}%")

bs.logout()
