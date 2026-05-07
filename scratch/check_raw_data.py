import json
import re
from pathlib import Path

def extract_groww_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Try to find __NEXT_DATA__
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if match:
        data = json.loads(match.group(1))
        # Navigate to fund details (this path might vary, let's look for common ones)
        try:
            fund_data = data['props']['pageProps']['initialState']['mfFundData']
            return {
                'name': fund_data.get('fund_name'),
                'aum': fund_data.get('fund_size'),
                'exit_load': fund_data.get('exit_load'),
                'nav': fund_data.get('nav'),
                'expense_ratio': fund_data.get('expense_ratio')
            }
        except KeyError:
            return "Fund data path not found in JSON"
    return "No __NEXT_DATA__ found"

files = [
    'data/0_raw_html/DOC-012.html', # ICICI Top 100
    'data/0_raw_html/DOC-011.html', # ICICI Nifty Next 50
    'data/0_raw_html/DOC-014.html', # ICICI Infrastructure
    'data/0_raw_html/DOC-005.html', # TWC Gold ETF FoF
    'data/0_raw_html/DOC-015.html', # ICICI Pharma
    'data/0_raw_html/DOC-003.html'  # TWC Ethical
]

for file in files:
    print(f"--- {file} ---")
    print(extract_groww_data(file))
