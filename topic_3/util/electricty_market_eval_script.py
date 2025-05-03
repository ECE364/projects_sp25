import pandas as pd

def evaluate_student_strategy(student_csv, dam_csv, rtm_csv):
    # Load all data
    student_df = pd.read_csv(student_csv, parse_dates=['Eastern Date Hour'])
    dam_df = pd.read_csv(dam_csv, parse_dates=['Eastern Date Hour'])
    rtm_df = pd.read_csv(rtm_csv, parse_dates=['Eastern Date Hour'])

    # Merge with actual prices
    merged = student_df.merge(
        dam_df[['Eastern Date Hour', 'Zone Name', 'Zone PTID', 'DAM Zonal LBMP']],
        on=['Eastern Date Hour', 'Zone Name', 'Zone PTID']
    ).merge(
        rtm_df[['Eastern Date Hour', 'Zone Name', 'Zone PTID', 'TWI Zonal LBMP']],
        on=['Eastern Date Hour', 'Zone Name', 'Zone PTID']
    )

    # Convert to numeric types
    numeric_cols = ['DAM Zonal LBMP', 'TWI Zonal LBMP', 'Bid Price', 'DA Lower']
    merged[numeric_cols] = merged[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Extract date and sort by timestamp
    merged['Date'] = merged['Eastern Date Hour'].dt.date
    merged = merged.sort_values('Eastern Date Hour')

    total_profit = 0
    daily_budget = 250000

    # Process each day individually
    for date, daily_group in merged.groupby('Date'):
        daily_profit = 0
        remaining_budget = daily_budget

        # Process each bid in chronological order
        for _, row in daily_group.iterrows():
            if row['DA Lower'] != 1:
                continue

            bid_price = row['Bid Price']
            da_price = row['DAM Zonal LBMP']
            rt_price = row['TWI Zonal LBMP']

            # Check if we can afford this bid
            if (bid_price <= remaining_budget) and bid_price > da_price:
                # Bid on the market
                remaining_budget -= bid_price
                # Check profit conditions
                if (da_price < rt_price):
                    daily_profit += (rt_price - bid_price)

        total_profit += daily_profit

    print(f"\nTotal Annual Profit: ${total_profit:,.2f}")
    return total_profit

if __name__ == "__main__":
    student_csv = "prediction.csv"
    dam_path = "DAM_NYISO_Zonal_LBMP_2016.csv"
    rtm_path = "RTM_NYISO_Zonal_LBMP_2016.csv"

    try:
        profit = evaluate_student_strategy(student_csv, dam_path, rtm_path)
        print(f"Final Profit: {profit:.2f}")
    except Exception as e:
        print("Error during evaluation:", e)
