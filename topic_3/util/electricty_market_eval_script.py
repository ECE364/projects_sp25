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
        how='inner'
    )
    
    # Merge the RTM data. RTM data is expected to have the column 'TWI Zonal LBMP'
    merged_df = pd.merge(
        merged_df,
        rtm_df[['Eastern Date Hour', 'Zone Name', 'Zone PTID', 'TWI Zonal LBMP']],
        on=['Eastern Date Hour', 'Zone Name', 'Zone PTID'],
        how='inner'
    )
    
    # Rename columns for clarity
    merged_df.rename(columns={'DAM Zonal LBMP': 'DA_Price', 'TWI Zonal LBMP': 'RT_Price'}, inplace=True)

    # Convert relevant columns to numeric types
    merged_df[['DA Lower', 'Bid Price', 'DA_Price', 'RT_Price']] = merged_df[['DA Lower', 'Bid Price', 'DA_Price', 'RT_Price']].apply(pd.to_numeric, errors='coerce')
    
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
    
    # Convert "Eastern Date Hour" to datetime and extract the date.
    merged_df['Eastern Date Hour'] = pd.to_datetime(merged_df['Eastern Date Hour'])
    merged_df['Date'] = merged_df['Eastern Date Hour'].dt.date
    
    # Now, for each day, enforce that the total Bid Price for executed trades does not exceed $250,000.
    # We process trades in the order they appear (or you may sort them by any priority, e.g., descending profit).
    # On every cleared bid (DA Lower==1 AND Bid Price > DA_Price), subtract the bid from remaining budget.
    # If there’s still budget left, record the Raw_Profit (which can be positive, zero, or negative).
    # Otherwise (ran out of budget) set profit to 0.
    def enforce_budget(group, daily_budget=250000):
        remaining = daily_budget
        profits  = []
        for idx, row in group.iterrows():
            # charge budget on every cleared bid (DA Lower=1 AND bid > DA_Price)
            if row['DA Lower'] == 1 and row['Bid Price'] > row['DA_Price']:
                bid = row['Bid Price']
                if bid <= remaining:
                    remaining -= bid
                    # record profit (can be positive, zero, or negative)
                    profits.append(row['Raw_Profit'])
                else:
                    # out of budget
                    profits.append(0)
            else:
                # no clearance → no spend, no profit
                profits.append(0)
        return pd.Series(profits, index=group.index)
    
    # Sort by Eastern Date Hour to ensure we process trades in the correct order.
    merged_df = merged_df.sort_values('Eastern Date Hour')

    # Apply the budget enforcement function to each group of trades for each day.
    merged_df['Profit'] = merged_df.groupby('Date', group_keys=False)[['DA Lower', 'Bid Price', 'DA_Price', 'RT_Price', 'Raw_Profit']].apply(enforce_budget)
    
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
