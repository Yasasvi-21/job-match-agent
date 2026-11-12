import csv, os
from day2_loop import fetch_posting      # reuse your tool

os.makedirs("golden", exist_ok=True)

with open("golden_set.csv") as f:
    for row in csv.DictReader(f):
        text = fetch_posting(row["company"], row["job_id"])
        if text.startswith("Error"):          # hint: what does every failure from your tool start with?
            print("❌", row["company"], row["job_id"], text)
            continue
        path = f"golden/{row['company']}_{row['job_id']}.txt"
        with open(path, "w") as out:
            out.write(text)
        print("✅ saved", path)