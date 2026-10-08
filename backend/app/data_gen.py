"""Simulated campus placement data (students, recruiters, drives, offers, history)."""
import numpy as np, pandas as pd
from datetime import date, timedelta

SEED = 42
BRANCH_P = {"CSE": .28, "IT": .18, "ECE": .20, "EEE": .12, "MECH": .13, "CIVIL": .09}
SW = ["python", "java", "sql", "machine learning", "aws", "docker", "react", "node", "dsa", "c++"]
POOL = {"CSE": SW, "IT": SW,
        "ECE": ["embedded c", "vlsi", "matlab", "c++", "python", "dsa", "sql"],
        "EEE": ["matlab", "embedded c", "autocad", "excel", "python", "c++"],
        "MECH": ["autocad", "matlab", "excel", "python", "sql"],
        "CIVIL": ["autocad", "excel", "matlab", "sql", "python"]}
ROLES = {"Software Engineer": ["java", "dsa", "sql", "react"],
         "Data Scientist": ["python", "machine learning", "sql", "excel"],
         "Cloud Engineer": ["aws", "docker", "python", "sql"],
         "Embedded Engineer": ["embedded c", "c++", "matlab"],
         "VLSI Engineer": ["vlsi", "c++", "matlab"],
         "Design / Core Engineer": ["autocad", "matlab", "excel"]}
ALL_B = list(BRANCH_P)
SOFT_B = ["CSE", "IT", "ECE"]

def _co(i, name, skills, min_cgpa, branches, backlogs, bench, ctc, jd):
    return dict(id=f"C{i:02d}", name=name, skills=skills, min_cgpa=min_cgpa, branches=branches,
                max_backlogs=backlogs, mock_bench=bench, ctc=ctc, jd=jd)

COMPANIES = [
    _co(1, "Nimbus Cloud Systems", ["aws", "docker", "python", "sql"], 7.0, ["CSE", "IT"], 0, 60, 14.0, "Cloud engineer building aws and docker based python services with sql data stores and strong dsa"),
    _co(2, "Quantra Analytics", ["python", "machine learning", "sql"], 7.5, ["CSE", "IT", "ECE"], 0, 65, 18.0, "Data scientist using python machine learning and sql to build predictive models"),
    _co(3, "Helix Software", ["java", "sql", "dsa"], 6.5, SOFT_B, 1, 55, 9.0, "Java backend developer with sql and dsa fundamentals for enterprise products"),
    _co(4, "Orbit Embedded", ["embedded c", "c++", "matlab"], 6.8, ["ECE", "EEE"], 1, 58, 8.5, "Embedded software engineer in embedded c and c++ with matlab simulation"),
    _co(5, "Vertex Core Engineering", ["autocad", "matlab", "excel"], 6.5, ["MECH", "CIVIL", "EEE"], 1, 50, 6.0, "Design engineer using autocad matlab and excel for core engineering projects"),
    _co(6, "Pulse FinTech", ["react", "node", "sql", "dsa"], 7.2, ["CSE", "IT"], 0, 62, 16.0, "Full stack engineer with react node sql and dsa for payments platform"),
    _co(7, "Atlas Infra Projects", ["autocad", "excel"], 6.0, ["CIVIL", "MECH"], 2, 45, 5.2, "Site and design engineer proficient in autocad and excel"),
    _co(8, "Zenith AI Labs", ["python", "machine learning", "dsa", "aws"], 8.0, ["CSE", "IT"], 0, 70, 24.0, "ML engineer deploying python machine learning on aws with strong dsa"),
    _co(9, "Cobalt Digital", ["java", "react", "sql"], 6.5, ["CSE", "IT", "ECE", "EEE"], 1, 52, 7.5, "Software engineer with java react and sql for digital delivery"),
    _co(10, "Meridian Semiconductors", ["vlsi", "c++", "matlab"], 7.5, ["ECE", "EEE"], 0, 62, 15.0, "VLSI design engineer using c++ and matlab for chip verification"),
    _co(11, "Stratus Consulting", ["excel", "sql", "python"], 6.5, ALL_B, 1, 50, 6.8, "Analyst role needing excel sql and python for client reporting"),
    _co(12, "Kinetic Motors", ["matlab", "autocad", "c++"], 6.8, ["MECH", "EEE", "ECE"], 1, 55, 7.0, "Automotive controls engineer using matlab autocad and c++"),
]

SLOTS = ["09:00-12:00", "14:00-17:00"]
VENUES = ["Auditorium", "Seminar Hall A", "Seminar Hall B", "Lab Complex"]
# (company idx, day offset from next Monday, slot idx, venue, panel) - deliberately contains clashes
DRIVE_PLAN = [(0, 0, 0, "Auditorium", "P1"), (1, 0, 0, "Auditorium", "P2"), (7, 0, 0, "Seminar Hall A", "P1"),
              (2, 1, 0, "Seminar Hall A", "P3"), (5, 1, 0, "Seminar Hall B", "P4"), (8, 1, 1, "Lab Complex", "P2"),
              (3, 2, 0, "Auditorium", "P5"), (9, 2, 0, "Seminar Hall B", "P5"), (4, 3, 1, "Seminar Hall A", "P4"),
              (10, 4, 0, "Lab Complex", "P1")]

def is_eligible(df, c):
    return (df.cgpa >= c["min_cgpa"]) & df.branch.isin(c["branches"]) & (df.backlogs <= c["max_backlogs"])

def generate():
    rng = np.random.default_rng(SEED)
    n = 600
    ab = rng.normal(0, 1, n)
    branch = rng.choice(ALL_B, n, p=list(BRANCH_P.values()))
    clip = np.clip
    df = pd.DataFrame({
        "id": [f"S{i+1:04d}" for i in range(n)], "branch": branch,
        "cgpa": clip(7.0 + .7 * ab + rng.normal(0, .6, n), 5.0, 9.9).round(2),
        "aptitude": clip(58 + 11 * ab + rng.normal(0, 8, n), 20, 98).round(0),
        "mock": clip(55 + 12 * ab + rng.normal(0, 9, n), 15, 98).round(0),
        "comm": clip(60 + 10 * ab + rng.normal(0, 9, n), 20, 98).round(0),
        "backlogs": clip(rng.poisson(np.exp(-1.2 - .7 * ab)), 0, 6)})
    first = ["Aarav", "Ananya", "Rohan", "Sneha", "Ishita", "Kabir", "Meera", "Arjun", "Priya", "Sanjay", "Tanvi", "Vikram", "Diya", "Rahul", "Nisha", "Aditya", "Pooja", "Siddharth", "Riya", "Manish"]
    last = ["Mishra", "Patra", "Sahoo", "Das", "Nayak", "Behera", "Mohanty", "Rout", "Swain", "Panda", "Sharma", "Singh", "Kumar", "Jena", "Pradhan"]
    df["name"] = [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(n)]
    sk, pr, ce = [], [], []
    for i in range(n):
        pool = POOL[branch[i]]
        k = int(clip(round(4 + 1.2 * ab[i] + rng.normal(0, 1.3)), 2, 9))
        s = [str(x) for x in rng.choice(pool, min(k, len(pool)), replace=False)] + (["communication"] if rng.random() < .3 else [])
        sk.append(s)
        npj = int(clip(round(1.5 + .8 * ab[i] + rng.normal(0, .9)), 0, 5))
        nce = int(clip(round(1 + .7 * ab[i] + rng.normal(0, .9)), 0, 4))
        pr.append([f"{x} project" for x in rng.choice(s, npj, replace=True)])
        ce.append([f"{x} certification" for x in rng.choice(s, min(nce, len(s)), replace=False)])
    df["skills"], df["projects"], df["certs"] = sk, pr, ce
    latent = ab + rng.normal(0, .6, n)
    df["placed"] = latent > np.quantile(latent, .38)
    df["ability"] = ab  # hidden ground truth, never used by models

    # ground-truth recruiter selections for evaluating the matcher (uses hidden ability)
    hist = {}
    for c in COMPANIES:
        req = set(c["skills"])
        cov = df.skills.apply(lambda s: len(req & set(s)) / len(req))
        el = is_eligible(df, c)
        score = (df.ability + 1.0 * cov + rng.normal(0, .4, n))[el]
        hist[c["id"]] = list(df.loc[score.sort_values(ascending=False).index[:10], "id"])

    today = date.today()
    monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
    drives = []
    for k, (ci, off, sl, ven, pan) in enumerate(DRIVE_PLAN):
        drives.append(dict(id=f"D{k+1:02d}", company_id=COMPANIES[ci]["id"], company=COMPANIES[ci]["name"],
                           date=str(monday + timedelta(days=off)), slot=SLOTS[sl], venue=ven, panel=pan,
                           status="Active" if k < 2 else "Upcoming"))

    offers = []
    for _, s in df[df.placed].iterrows():
        opts = [c for c in COMPANIES if s.branch in c["branches"]] or COMPANIES
        c = opts[rng.integers(len(opts))]
        r = rng.random()
        kind = "PPO" if r < .12 else "Internship Conversion" if r < .20 else "Full-time"
        st = str(rng.choice(["Accepted", "Pending", "Deferred", "Withdrawn"], p=[.68, .17, .07, .08]))
        od = today - timedelta(days=int(rng.integers(1, 60)))
        doc = str(rng.choice(["Verified", "Submitted", "Pending"], p=[.4, .3, .3])) if st == "Accepted" else "Pending"
        offers.append(dict(id=f"O{len(offers)+1:04d}", student_id=s.id, student=s["name"], branch=s.branch,
                           company=c["name"], type=kind, ctc=round(c["ctc"] * rng.uniform(.9, 1.15), 2),
                           status=st, doc_status=doc, bond_months=int(rng.choice([0, 12, 24], p=[.5, .3, .2])),
                           offer_date=str(od), doc_deadline=str(od + timedelta(days=21))))
    hrows = []
    for c in COMPANIES:
        base = 12 if c["ctc"] == 9.0 else 6 if c["ctc"] == 18.0 else 5 if c["ctc"] in (14.0, 8.5) else 4
        for y in range(2022, 2026):
            h = int(rng.poisson(base)) if rng.random() > .2 else 0
            if h:
                a = c["ctc"] * (.78 + .06 * (y - 2022)) * rng.uniform(.95, 1.05)
                hrows.append(dict(year=y, company=c["name"], hires=h, avg_ctc=round(a, 2), max_ctc=round(a * rng.uniform(1.4, 1.9), 2)))
    return dict(students=df, companies=COMPANIES, drives=drives, offers=pd.DataFrame(offers),
                history=pd.DataFrame(hrows), hist_selected=hist, roles=ROLES)
