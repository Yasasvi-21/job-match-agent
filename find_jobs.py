import requests

boards = ["newsbreak", "gleanwork", "scaleai", "klaviyocampus", "anthropic",
          "avepoint", "c3ascend", "sayari", "databricks", "stripe", "airbnb", "discord"]
keywords = ["machine learning", "ml ", "ai engineer", "new grad", "junior",
            "early career", "university", "data scientist", "data engineer"]

for board in boards:
    r = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs", timeout=10)
    if r.status_code != 200:
        print(f"-- {board}: not on Greenhouse ({r.status_code})")
        continue
    for job in r.json()["jobs"]:
        title = job["title"].lower()
        if any(k in title for k in keywords):
            print(f"{board},{job['id']},{job['title']} | {job['location']['name']}")