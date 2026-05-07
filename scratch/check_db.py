import sqlite3
from pathlib import Path

db_path = Path("data/5_structured_facts/facts.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Distinct Source URLs in facts.db:")
cursor.execute("SELECT DISTINCT source_url FROM structured_facts")
for row in cursor.fetchall():
    print(row[0])

print("\nSample Facts for ICICI:")
cursor.execute("SELECT source_url, fact_name, fact_value FROM structured_facts WHERE source_url LIKE '%icici%' LIMIT 10")
for row in cursor.fetchall():
    print(row)

conn.close()
