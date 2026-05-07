
import sqlite3
from pathlib import Path

db_path = Path("data/5_structured_facts/facts.db")
with sqlite3.connect(db_path) as conn:
    row = conn.execute("SELECT value FROM structured_facts WHERE doc_id = 'DOC-019' AND fact_type = 'expense_ratio'").fetchone()
    print(f"Axis Large Cap Expense Ratio in SQLite: {row}")
