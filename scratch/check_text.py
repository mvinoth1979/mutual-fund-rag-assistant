
with open("data/2_normalized_text/DOC-019_normalized.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
    idx = text.find("Expense ratio")
    print(f"Index of 'Expense ratio': {idx}")
    if idx != -1:
        snippet = text[idx:idx+200].encode('ascii', 'ignore').decode('ascii')
        print(f"Snippet: {snippet}")
    else:
        print("Not found")

    idx2 = text.find("0.65")
    print(f"Index of '0.65': {idx2}")
    if idx2 != -1:
        snippet2 = text[idx2-100:idx2+100].encode('ascii', 'ignore').decode('ascii')
        print(f"Snippet 2: {snippet2}")
    else:
        print("Not found")
