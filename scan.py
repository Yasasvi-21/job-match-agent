import os
import re
import sqlite3
import sys
from datetime import date
import html

YEARS = re.compile(r"(\d+)\+?\s*(?:-\s*\d+\s*)?years?[^.\n]{0,40}experience", re.I)

def too_senior(job):
    text = html.unescape(job.get("content", ""))
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    return any(int(n) >= 3 for n in YEARS.findall(text))
import requests

from day2_loop import BASE
from graph_agent import run_graph_agent
from notify import send

MAX_PER_RUN = int(os.getenv("MAX_PER_RUN", "25"))   # cost cap: 25 jobs ≈ $0.08
DRY_RUN = "--dry-run" in sys.argv

TITLE_KEEP = re.compile(
    r"machine learning|\bml\b|\bai\b|artificial intelligence|software engineer|"
    r"data scien|applied scien|research engineer|new grad|early career", re.I)
TITLE_DROP = re.compile(
    r"senior|\bsr\.?\b|staff|principal|\blead\b|manager|director|head of|"
    r"intern|sales|account exec|recruit|marketing|architect|executive|leader|"
    r"practitioner|consultant|transformation|\bdesign(er)?\b|solutions"
    r"|policy|operations|planning|analyst|phd", re.I)
US = re.compile(
    r"united states|\busa?\b|remote|san francisco|new york|seattle|los angeles|boston|"
    r"austin|chicago|mountain view|palo alto|bay area|"
    r", (A[LKZR]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[AT]|W[AIVY])\b",
    re.I)
NON_US = re.compile(
    r"canada|toronto|vancouver|london|\buk\b|dublin|ireland|berlin|germany|paris|france|"
    r"amsterdam|netherlands|sweden|stockholm|denmark|aarhus|spain|poland|switzerland|"
    r"india|bengaluru|bangalore|hyderabad|singapore|tokyo|japan|korea|sydney|australia|"
    r"israel|mexico|brazil|emea|apac", re.I)

EMOJI = {"fit": "🟢", "borderline": "🟡"}


def load_companies():
    lines = (BASE / "companies.txt").read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def list_jobs(company):
    r = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true", timeout=15)
    if r.status_code != 200:
        print(f"  skip {company}: HTTP {r.status_code}")
        return []
    return r.json().get("jobs", [])


def location_of(job):
    return (job.get("location") or {}).get("name", "")


def wanted(job):
    title, loc = job["title"], location_of(job)
    return (TITLE_KEEP.search(title)
            and not TITLE_DROP.search(title)
            and US.search(loc) and not NON_US.search(loc)
            and not too_senior(job))


def main():
    db = sqlite3.connect(BASE / "seen.db")
    db.execute("CREATE TABLE IF NOT EXISTS seen "
               "(job_key TEXT PRIMARY KEY, verdict TEXT, score REAL, checked_on TEXT)")

    # 1. Fetch + cheap filters (free: no Claude calls)
    candidates = []
    for company in load_companies():
        for job in list_jobs(company):
            key = f"{company}_{job['id']}"
            if not wanted(job):
                continue
            if db.execute("SELECT 1 FROM seen WHERE job_key = ?", (key,)).fetchone():
                continue                                   # already checked on an earlier day
            candidates.append((company, job))

    PRIORITY = re.compile(r"new grad|early career|junior|entry", re.I)
    candidates.sort(key=lambda c: (bool(PRIORITY.search(c[1]["title"])),
                               c[1].get("updated_at", "")), reverse=True)
    print(f"{len(candidates)} new matching jobs; checking up to {MAX_PER_RUN}")

    if DRY_RUN:
        for company, job in candidates:
            print(f"  {company:14} {job['title']}  |  {location_of(job)}")
        return

    # 2. Expensive check (Claude) on the survivors only
    pinged = 0
    for company, job in candidates[:MAX_PER_RUN]:
        key = f"{company}_{job['id']}"
        try:
            r = run_graph_agent(company, str(job["id"]))
        except Exception as e:
            print(f"  error on {key}: {e}")              # one bad job must not stop the scan
            continue

        if r["verdict"] != "error":                      # errors get retried tomorrow
            db.execute("INSERT OR REPLACE INTO seen VALUES (?, ?, ?, ?)",
                       (key, r["verdict"], r["score"], date.today().isoformat()))
            db.commit()
        print(f"  {r['verdict']:10} {r['score']}  {company}: {job['title']}")

        # 3. Ping only what's worth your time
        if r["verdict"] in EMOJI:
            text = (f"{EMOJI[r['verdict']]} **{r['verdict']}** · {company}: {job['title']} "
                    f"(score {r['score']})\n{location_of(job)}\nApply: <{job['absolute_url']}>")
            if r["injection_warning"]:
                text += "\n⚠️ Posting contains text aimed at AI tools. Read it yourself."
            send(text)
            pinged += 1

    send(f"📋 Daily scan done: {min(len(candidates), MAX_PER_RUN)} jobs checked, {pinged} worth a look.")


if __name__ == "__main__":
    main()