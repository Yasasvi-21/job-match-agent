import csv

files = ["eval_results_v5_run1.csv", "eval_results_v5_run2.csv", "eval_results_v5_run3.csv"]
runs = []
for name in files:
    with open(name, encoding="utf-8") as f:
        runs.append(list(csv.DictReader(f)))

# 1. Recall for each run
for name, rows in zip(files, runs):
    fits = [x for x in rows if x["my_label"] == "fit"]
    caught = [x for x in fits if x["agent"] == "fit"]
    print(f"{name}: recall {len(caught) / len(fits):.0%}")

# 2. Which jobs flipped between runs?
print("\nJobs that flipped:")
for i in range(len(runs[0])):
    verdicts = [rows[i]["agent"] for rows in runs]
    scores = [rows[i]["score"] for rows in runs]
    if len(set(verdicts)) > 1:
        print(runs[0][i]["company"], runs[0][i]["job_id"], verdicts, scores)