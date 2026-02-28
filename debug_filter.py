# -*- coding: utf-8 -*-
"""调试基础过滤问题"""

import baostock as bs
import pandas as pd
import numpy as np

ds = bs.login()
print(f"登录状态: {ds.error_code} {ds.error_msg}")

# 获取股票列表
rs = bs.query_stock_basic()
data = []
while rs.error_code == "0" and rs.next():
    data.append(rs.get_row_data())
df_stocks = pd.DataFrame(data, columns=rs.fields)

# 过滤主板A股
def is_mainboard_a_share(code: str) -> bool:
    return isinstance(code, str) and (code.startswith("sh.60") or code.startswith("sz.00"))

df_stocks = df_stocks[df_stocks["code"].apply(is_mainboard_a_share)].reset_index(drop=True)
print(f"\n主板A股总数: {len(df_stocks)}")

# 随机抽取50只股票进行测试
sample_codes = df_stocks["code"].tolist()[:50]

# 统计过滤失败原因
fail_reasons = {
    "价格过低": 0,
    "价格过高": 0,
    "成交额过低": 0,
    "数据不足": 0,
    "通过": 0
}

# 详细记录
detailed_records = []

for code in sample_codes:
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount,pctChg",
        start_date="2025-10-01",
        end_date="2026-02-27",
        frequency="d",
        adjustflag="2"
    )

    data = []
    while rs.error_code == "0" and rs.next():
        data.append(rs.get_row_data())
    df = pd.DataFrame(data, columns=rs.fields)

    if df.empty or len(df) < 20:
        fail_reasons["数据不足"] += 1
        detailed_records.append({"code": code, "reason": "数据不足", "details": f"数据行数: {len(df)}"})
        continue

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    last = df.iloc[-1]
    avg_amount = df["amount"].tail(20).mean()

    # 测试不同阈值
    reasons = []

    # 价格过滤
    min_price, max_price = 3.0, 200.0
    if last["close"] < min_price:
        reasons.append(f"价格过低 {last['close']:.2f}")
    if last["close"] > max_price:
        reasons.append(f"价格过高 {last['close']:.2f}")

    # 成交额过滤 (500万)
    min_amount = 5e6
    if avg_amount < min_amount:
        reasons.append(f"成交额过低 {avg_amount/1e6:.2f}万 (<500万)")

    if not reasons:
        fail_reasons["通过"] += 1
        detailed_records.append({
            "code": code,
            "reason": "通过",
            "details": f"价格:{last['close']:.2f} 成交额:{avg_amount/1e6:.2f}万"
        })
    else:
        for r in reasons:
            if "价格过低" in r:
                fail_reasons["价格过低"] += 1
            elif "价格过高" in r:
                fail_reasons["价格过高"] += 1
            elif "成交额过低" in r:
                fail_reasons["成交额过低"] += 1

        detailed_records.append({
            "code": code,
            "reason": ",".join(reasons),
            "details": f"价格:{last['close']:.2f} 成交额:{avg_amount/1e6:.2f}万"
        })

print("\n" + "="*60)
print(f"过滤失败原因统计 (测试{len(sample_codes)}只股票)")
print("="*60)
for reason, count in fail_reasons.items():
    print(f"{reason}: {count}")

print("\n" + "="*60)
print("详细记录 (前20条)")
print("="*60)
for record in detailed_records[:20]:
    print(f"{record['code']:15} | {record['reason']:30} | {record['details']}")

# 测试不同成交额阈值
print("\n" + "="*60)
print("不同成交额阈值下的通过率")
print("="*60)

thresholds = [1e6, 2e6, 3e6, 5e6, 10e6]
valid_df = pd.DataFrame(detailed_records[detailed_records["reason"] == "通过"])
base_count = len(valid_df)

for threshold in thresholds:
    passed = len([r for r in detailed_records
                  if r["reason"] == "通过" or
                     (r["reason"].startswith("成交额过低") and
                      float(r["details"].split("成交额:")[1].split("万")[0]) >= threshold/1e6)])
    print(f"阈值 {threshold/1e6:.0f}万: {passed}/{len(sample_codes)} = {passed/len(sample_codes)*100:.1f}%")

bs.logout()
