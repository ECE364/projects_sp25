import pandas as pd
import argparse

def evaluate_student_strategy(student_csv, dam_csv, rtm_csv):
    # Load the student output
    student_df = pd.read_csv(student_csv)
    
    # Load the DAM and RTM data
    dam_df = pd.read_csv(dam_csv)
    rtm_df = pd.read_csv(rtm_csv)
    
    # Merge the student output with the DAM data.
    # DAM data is expected to have the column 'DAM Zonal LBMP'
    merged_df = pd.merge(
        student_df,
        dam_df[['Eastern Date Hour', 'Zone Name', 'Zone PTID', 'DAM Zonal LBMP']],
        on=['Eastern Date Hour', 'Zone Name', 'Zone PTID'],
        how='left'
    )
    
    # Merge the RTM data. RTM data is expected to have the column 'TWI Zonal LBMP'
    merged_df = pd.merge(
        merged_df,
        rtm_df[['Eastern Date Hour', 'Zone Name', 'Zone PTID', 'TWI Zonal LBMP']],
        on=['Eastern Date Hour', 'Zone Name', 'Zone PTID'],
        how='left'
    )
    
    # Rename columns for clarity
    merged_df.rename(columns={'DAM Zonal LBMP': 'DA_Price', 'TWI Zonal LBMP': 'RT_Price'}, inplace=True)
    
    # Calculate raw profit for each trade:
    # A trade is executed only if:
    #   1. Actual DA_Price < RT_Price, AND
    #   2. Student prediction DA Lower == 1, AND
    #   3. Student Bid Price > Actual DA_Price (i.e., bid clears)
    # Profit = RT_Price - Bid Price if trade executed; otherwise, profit = 0.
    def calc_profit(row):
        if (row['DA_Price'] < row['RT_Price'] and 
            row['DA Lower'] == 1 and 
            row['Bid Price'] > row['DA_Price']):
            return row['RT_Price'] - row['Bid Price']
        else:
            return 0

    merged_df['Raw_Profit'] = merged_df.apply(calc_profit, axis=1)
    
    # For enforcing the daily budget, extract the date portion from "Eastern Date Hour".
    # Here we assume "Eastern Date Hour" is in a format where the first 10 characters are the date (e.g., "YYYY-MM-DD").
    merged_df['Date'] = merged_df['Eastern Date Hour'].astype(str).str[:10]
    
    # Now, for each day, enforce that the total Bid Price for executed trades does not exceed $250,000.
    # We process trades in the order they appear (or you may sort them by any priority, e.g., descending profit).
    # For each trade that has a positive raw profit, if adding its Bid Price would exceed the daily budget,
    # we set its profit to 0 (i.e. the trade is not executed).
    def enforce_budget(group, daily_budget=250000):
        remaining = daily_budget
        profits = []
        # Process trades in the order they appear.
        for idx, row in group.iterrows():
            if row['Raw_Profit'] > 0:
                bid = row['Bid Price']
                if bid <= remaining:
                    profits.append(row['Raw_Profit'])
                    remaining -= bid
                else:
                    profits.append(0)
            else:
                profits.append(0)
        return pd.Series(profits, index=group.index)
    
    merged_df['Profit'] = merged_df.groupby('Date', group_keys=False).apply(enforce_budget)
    
    # Sum the profit across all records.
    final_profit = merged_df['Profit'].sum()
    return final_profit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate student's trading strategy for virtual bidding with daily budget constraint.")
    parser.add_argument("student_csv", help="Path to the student's output CSV file (with headers: Eastern Date Hour, Zone Name, Zone PTID, DA Lower, Bid Price).")
    parser.add_argument("dam_csv", help="Path to DAM_NYISO_Zonal_LBMP_2016.csv file.")
    parser.add_argument("rtm_csv", help="Path to RTM_NYISO_Zonal_LBMP_2016.csv file.")
    args = parser.parse_args()
    
    try:
        profit = evaluate_student_strategy(args.student_csv, args.dam_csv, args.rtm_csv)
        print(f"Final Profit: {profit:.2f}")
    except Exception as e:
        print("Error during evaluation:", e)
