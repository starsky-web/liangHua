import baostock as bs
import pandas as pd
import time

# 改进后的小市值策略

def is_common_a_stock(code, is_st, trade_status):
    """
    严格判断是否为'无门槛'普通账户可交易的股票
    范围：仅限沪深主板
    """
    # 基础状态检查：必须非ST 且 未停牌
    if is_st == "1" or trade_status != "1":
        return False

    # === 严格板块过滤 ===

    # 1. 排除北交所
    if code.startswith("bj."):
        return False

    # 2. 排除科创板 (sh.688, sh.689 等)
    # 上海主板主要是 sh.600, sh.601, sh.603... 只要 sh.68 开头的一律排除
    if code.startswith("sh.68"):
        return False

    # 3. 排除创业板 (sz.300, sz.301...)
    # 深圳主板主要是 sz.000, sz.001, sz.003... 只要 sz.3 开头的一律排除
    if code.startswith("sz.3"):
        return False

    # 4. 排除B股 (sh.9xx, sz.2xx)
    if code.startswith("sh.9") or code.startswith("sz.2"):
        return False

    # 剩下的即为上海主板 (sh.60...) 和 深圳主板 (sz.00..., sz.001..., sz.003...)
    return True


def get_stock_basic_info(date):
    """获取指定日期的所有股票代码列表"""
    print(f"正在获取 {date} 的全市场股票列表...")
    rs = bs.query_all_stock(day=date)

    if rs.error_code != '0':
        print(f"获取股票列表失败 (可能是日期无数据): {rs.error_msg}")
        return []

    stock_list = []
    while (rs.error_code == '0') & rs.next():
        if rs.get_row_data()[1] == '1':  # 仅股票
            stock_list.append(rs.get_row_data()[0])

    if not stock_list:
        print(f"警告: {date} 没有获取到任何股票数据。")

    return stock_list


def get_market_cap(code, date_str, filter_common_tradable=False):
    """获取单只股票的总市值"""
    try:
        # 增加 isST 字段
        rs = bs.query_history_k_data_plus(code, "date,code,close,tradestatus,isST",
                                          start_date=date_str, end_date=date_str,
                                          frequency="d", adjustflag="3")

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return None

        df_k = pd.DataFrame(data_list, columns=rs.fields)
        row = df_k.iloc[0]

        trade_status = row['tradestatus']
        is_st = row['isST']

        # --- 核心筛选逻辑 ---
        if filter_common_tradable:
            if not is_common_a_stock(code, is_st, trade_status):
                return None
        else:
            # 即使不开启严格筛选，也要跳过停牌，否则市值不准
            if trade_status != '1':
                return None

        close_price = float(row['close'])

        # 获取季报股本
        year = int(date_str[:4])
        total_share = 0
        for y in range(year, year - 2, -1):
            for q in [4, 3, 2, 1]:
                rs_p = bs.query_profit_data(code=code, year=y, quarter=q)
                profit_list = []
                while (rs_p.error_code == '0') & rs_p.next():
                    profit_list.append(rs_p.get_row_data())

                if profit_list:
                    df_p = pd.DataFrame(profit_list, columns=rs_p.fields)
                    try:
                        total_share = float(df_p.iloc[0]['totalShare'])
                        if total_share > 0:
                            break
                    except:
                        pass
            if total_share > 0:
                break

        if total_share == 0:
            return None

        market_cap = (close_price * total_share) / 100000000

        return {
            "code": code,
            "close": close_price,
            "market_cap": market_cap
        }

    except Exception as e:
        return None


def run_small_cap_strategy(target_date, min_cap, max_cap, top_n=2, filter_common_tradable=True):
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return

    result_list = []
    all_stocks = get_stock_basic_info(target_date)

    if not all_stocks:
        print("无法获取股票列表，程序结束。")
        bs.logout()
        return

    filter_text = "严格(仅沪深主板)" if filter_common_tradable else "宽松(含创科等)"
    print(f"策略配置: 市值 {min_cap}-{max_cap}亿 | 账户筛选: {filter_text}")
    print(f"开始扫描 {len(all_stocks)} 只股票...")

    count = 0
    for code in all_stocks:
        count += 1
        if count % 500 == 0:
            print(f"已处理 {count}/{len(all_stocks)} 只股票...")

        info = get_market_cap(code, target_date, filter_common_tradable=filter_common_tradable)
        if info:
            result_list.append(info)

    df_result = pd.DataFrame(result_list)

    if not df_result.empty:
        df_filtered = df_result[(df_result['market_cap'] >= min_cap) &
                                (df_result['market_cap'] <= max_cap)]

        print(f"\n筛选完成，符合条件股票共 {len(df_filtered)} 只。")

        if df_filtered.empty:
            print("在该市值范围内未找到符合条件的股票。")
        else:
            df_sorted = df_filtered.sort_values(by='market_cap', ascending=True)
            winners = df_sorted.head(top_n)

            final_winners = []
            for idx, row in winners.iterrows():
                rs_name = bs.query_stock_basic(code=row['code'])
                name = "未知"
                if rs_name.error_code == '0' and rs_name.next():
                    name = rs_name.get_row_data()[1]
                final_winners.append({
                    "code": row['code'],
                    "name": name,
                    "close": row['close'],
                    "market_cap": row['market_cap']
                })

            print("\n" + "=" * 40)
            print(f"策略结果：{target_date} 最终筛选出的 {top_n} 只股票 (仅主板)")
            print("=" * 40)
            for w in final_winners:
                print(f"代码: {w['code']}\t名称: {w['name']}\t"
                      f"收盘: {w['close']}\t市值: {w['market_cap']:.2f}亿")
            print("=" * 40)
    else:
        print("未获取到任何有效数据。")

    bs.logout()


if __name__ == "__main__":
    # ================= 配置区域 =================
    # 1. 日期
    run_date = "2026-01-16"

    # 2. 市值范围 (亿元)
    min_market_cap = 20
    max_market_cap = 30

    # 3. 是否严格筛选"普通账户" (True=仅主板，False=含创业板/科创板)
    enable_common_filter = True

    # 4. 选出数量
    top_n_count = 2
    # ===========================================

    run_small_cap_strategy(
        target_date=run_date,
        min_cap=min_market_cap,
        max_cap=max_market_cap,
        top_n=top_n_count,
        filter_common_tradable=enable_common_filter
    )
