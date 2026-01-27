import baostock as bs
import pandas as pd
from datetime import datetime
# 使用示例
# ========== 1. 登录 ==========
lg = bs.login()
assert lg.error_code == '0', lg.error_msg

# ========== 2. 获取今日日期 ==========
# today = datetime.now().strftime("%Y-%m-%d")
today = "2026-01-23"
print("fetch date:", today)

# ========== 3. 获取全部股票列表 ==========
rs = bs.query_stock_basic()
stock_list = []
while rs.next():
    stock_list.append(rs.get_row_data())

stock_df = pd.DataFrame(stock_list, columns=rs.fields)

# 只保留 A 股（沪深）
stock_df = stock_df[
    stock_df["code"].str.startswith(("sh.", "sz."))
]

print(f"total stocks: {len(stock_df)}")

# ========== 4. 拉取当日行情 ==========
all_data = []

for code in stock_df["code"]:
    rs_k = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount",
        start_date=today,
        end_date=today,
        frequency="d",
        adjustflag="3"
    )

    if rs_k.error_code != '0':
        continue

    while rs_k.next():
        row = rs_k.get_row_data()
        row.insert(0, code)  # 加股票代码
        all_data.append(row)

# ========== 5. 汇总 DataFrame ==========
columns = ["code"] + rs_k.fields
df = pd.DataFrame(all_data, columns=columns)

print("valid rows:", len(df))
print(df.head())

# ========== 6. 登出 ==========
bs.logout()
