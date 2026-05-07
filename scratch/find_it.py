
import os
from pathlib import Path

def find_string(search_str):
    for root, dirs, files in os.walk("./data"):
        for file in files:
            path = Path(root) / file
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if search_str in content:
                        print(f"FOUND in {path}")
            except:
                pass

if __name__ == "__main__":
    find_string("0.65%")
