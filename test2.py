import baostock as bs
import pandas as pd
import time
from typing import List

# 随便拉的数据

# ====== 配置 ======
TARGET_DATE = "2026-01-23"   # 你要拉取的日期：YYYY-MM-DD（改这里）
PRINT_ALL = False           # True=全量都打印（非常多行，会刷屏）
PRINT_HEAD_N = 50           # PRINT_ALL=False 时打印前 N 行
PROGRESS_EVERY = 200        # 每处理多少只股票打印一次进度
SLEEP_EVERY_REQ = 0.0       # 适当加一点(如 0.02)可降低被限流风险


def is_a_share_stock(code: str) -> bool:
    """只保留沪深A股股票：sh.60/sh.68/sz.00/sz.30"""
    if not isinstance(code, str):
        return False
    return (
        code.startswith("sh.60")
        or code.startswith("sh.68")
        or code.startswith("sz.00")
        or code.startswith("sz.30")
    )


def get_stock_list() -> pd.DataFrame:
    rs = bs.query_stock_basic()
    rows: List[List[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    return df


def fetch_one_stock_one_day(code: str, day: str):
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount,turn",
        start_date=day,
        end_date=day,
        frequency="d",
        adjustflag="3",
    )
    if rs.error_code != "0":
        return []

    out = []
    while rs.next():
        row = rs.get_row_data()
        # row fields 对应：date,open,high,low,close,volume,amount,turn
        out.append([code] + row)
    return out


def main():
    # 1) 登录
    lg = bs.login()
    print("login:", lg.error_code, lg.error_msg)
    if lg.error_code != "0":
        return

    # 2) 股票列表
    stock_df = get_stock_list()
    print("raw total:", len(stock_df))

    # 3) 过滤为沪深A股股票
    if "code" not in stock_df.columns:
        raise RuntimeError(f"Unexpected fields from query_stock_basic: {stock_df.columns.tolist()}")

    stock_df = stock_df[stock_df["code"].apply(is_a_share_stock)].reset_index(drop=True)
    codes = stock_df["code"].tolist()
    print("A-share stocks(sh/sz):", len(codes))
    print("target date:", TARGET_DATE)

    # 4) 拉取指定日期日线截面
    t0 = time.time()
    all_rows = []
    empty_cnt = 0
    err_cnt = 0

    for i, code in enumerate(codes, start=1):
        try:
            rows = fetch_one_stock_one_day(code, TARGET_DATE)
            if not rows:
                empty_cnt += 1
            else:
                all_rows.extend(rows)
        except Exception:
            err_cnt += 1

        if SLEEP_EVERY_REQ > 0:
            time.sleep(SLEEP_EVERY_REQ)

        if i % PROGRESS_EVERY == 0 or i == len(codes):
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0
            print(f"[{i}/{len(codes)}] got_rows={len(all_rows)} empty={empty_cnt} err={err_cnt} "
                  f"speed={speed:.1f} stocks/s elapsed={elapsed:.1f}s")

    # 5) 输出 DataFrame
    cols = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turn"]
    df = pd.DataFrame(all_rows, columns=cols)

    print("\n===== RESULT =====")
    print("rows:", len(df))

    # 打印控制台（避免刷屏）
    if PRINT_ALL:
        print(df.to_string(index=False))
    else:
        print(df.head(PRINT_HEAD_N).to_string(index=False))
        if len(df) > PRINT_HEAD_N:
            print(f"... (only printed first {PRINT_HEAD_N} rows; set PRINT_ALL=True to print all)")

    # 6) 登出
    bs.logout()
    print("logout success!")


if __name__ == "__main__":
    main()
