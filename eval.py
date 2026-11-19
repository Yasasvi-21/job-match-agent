import csv
from day2_loop import load_saved_posting
from graph_agent import run_graph_agent as run_agent

with open("golden_set.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

results = []
runs = []

for row in rows:
    print(f"\n===== {row['company']} {row['job_id']}: {row['title']} =====")
    r = run_agent(row["company"], row["job_id"], fetch_fn=load_saved_posting)
    match = r["verdict"] == row["my_label"]
    results.append({**row, "agent": r["verdict"], "score": r["score"], "match": match,
                    "failed": "; ".join(r["failed_dealbreakers"]), "api_calls": r["api_calls"], "seconds": r["seconds"], "cost": r["cost"]})
    runs.append({
        "job": f"{row['company']}_{row['job_id']}",
        "seconds": r["seconds"],
        "cost": r["cost"],
        "api_calls": r["api_calls"],
    })
    print("✅" if match else "❌", "you:", row["my_label"], "| agent:", r["verdict"])


total_cost = sum(r["cost"] for r in runs)
avg_seconds = sum(r["seconds"] for r in runs) / len(runs)
slowest = max(runs, key=lambda r: r["seconds"])

print(f"Jobs measured: {len(runs)}")
print(f"Total cost:  ${total_cost:.4f}")
print(f"Avg seconds: {avg_seconds:.1f}")
print(f"Slowest job: {slowest['job']} ({slowest['seconds']}s, {slowest['api_calls']} calls)")

errors = [x for x in results if x["agent"] == "error" and x["my_label"] != "error"]
borderline = [x for x in results if x["agent"] == "borderline"]
judged = [x for x in results if x["agent"] not in ("error", "borderline") and x["my_label"] != "error"]

tp = sum(x["my_label"] == "fit"     and x["agent"] == "fit"     for x in judged)
fn = sum(x["my_label"] == "fit"     and x["agent"] == "not fit" for x in judged)
fp = sum(x["my_label"] == "not fit" and x["agent"] == "fit"     for x in judged)
tn = sum(x["my_label"] == "not fit" and x["agent"] == "not fit" for x in judged)

print("\n========== RESULTS ==========")
print(f"Judged: {len(judged)}   Errors: {len(errors)}")
print(f"Borderline (sent to human): {len(borderline)}  {[x['company'] for x in borderline]}")
print(f"Accuracy:         {(tp + tn) / len(judged):.0%}")
print(f"Recall (fit):     {tp / (tp + fn) if tp + fn else 0:.0%}   ← good jobs it caught")
print(f"Precision (fit):  {tp / (tp + fp) if tp + fp else 0:.0%}   ← 'fit' calls that were right")
print(f"Missed good jobs: {[x['company'] for x in judged if x['my_label']=='fit' and x['agent']!='fit']}")
print(f"False alarms:     {[x['company'] for x in judged if x['my_label']=='not fit' and x['agent']=='fit']}")

with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)