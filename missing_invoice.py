import re
import pandas as pd
from datetime import datetime
def find_missing_invoices(series):
    a=series.str.split("-").str[-1].str[-7:].str.strip()
    a = a.dropna()
    a=sorted(a.apply(lambda x: re.sub(r'[^0-9]', '', str(x))).replace('', '0').astype(int))
    sorted_invoices = sorted(a)
    missing_invoices = []
    for i in range(min(sorted_invoices), max(sorted_invoices) + 1):
        if i not in sorted_invoices:
            missing_invoices.append(i)
    return missing_invoices

# Create a list to store dictionaries for each group
result_list = []

# Group by 'ntn' and find the missing invoices in each group
def main(df):
    df["date"]=pd.to_datetime(df['created_date_time']).dt.date
    for name, group in df.groupby(['date', 'ntn']):
        date, ntn = name
        result_dict = {'date': date, 'ntn': ntn, 'missing_invoices': find_missing_invoices(group['invoice_no'])}
        result_list.append(result_dict)
    filtered_data = [item for item in result_list if item['missing_invoices']]
    return filtered_data

