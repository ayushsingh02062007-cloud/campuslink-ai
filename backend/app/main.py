"""CampusLink API - FastAPI app serving the REST API and the dashboard frontend."""
import json, re
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from . import data_gen, ml, scheduler, notifier

FRONT = Path(__file__).resolve().parents[2] / "frontend"
S = {}
js = lambda df: json.loads(df.to_json(orient="records"))

def build():
    d = data_gen.generate()
    st = ml.compute_readiness(d["students"])
    st, risk_metrics = ml.train_risk(st)
    matcher = ml.Matcher(st, d["companies"], d["hist_selected"])
    short = {dr["id"]: matcher.shortlist(dr["company_id"]) for dr in d["drives"]}
    S.update(df=st, companies={c["id"]: c for c in d["companies"]}, drives=d["drives"], original_drives=[dict(x) for x in d["drives"]],
             offers=d["offers"], history=d["history"], roles=d["roles"], matcher=matcher, short=short,
             metrics=dict(risk_model=risk_metrics, matching=matcher.evaluate(),
                          readiness=dict(avg=round(float(st.readiness.mean()), 1))),
             notes=notifier.build_all(d["drives"], short, d["offers"], st))

@asynccontextmanager
async def lifespan(app):
    build(); yield

app = FastAPI(title="CampusLink API", version="1.0", lifespan=lifespan)

def student(sid):
    r = S["df"][S["df"].id == sid]
    if r.empty: raise HTTPException(404, "Student not found")
    return r.iloc[0]

@app.get("/api/summary")
def summary():
    df, o = S["df"], S["offers"]
    live = o[o.status != "Withdrawn"]
    sl = set().union(*S["short"].values())
    risk = df[(~df.placed) & (df.p_place < .5)]
    acc = o[o.status == "Accepted"]
    funnel = [("Registered", len(df)), ("Placement-ready", int((df.readiness >= 60).sum())), ("Shortlisted", len(sl)),
              ("Offers made", len(live)), ("Offers accepted", len(acc)), ("Documents verified", int((acc.doc_status == "Verified").sum()))]
    return dict(registered=len(df), ready=int((df.readiness >= 60).sum()), placed=int(df.placed.sum()),
                placement_rate=round(float(df.placed.mean()) * 100, 1), drives=len(S["drives"]),
                offers=dict(made=len(live), accepted=len(acc), pending=int((o.status == "Pending").sum()),
                            deferred=int((o.status == "Deferred").sum()), withdrawn=int((o.status == "Withdrawn").sum())),
                avg_ctc=round(float(live.ctc.mean()), 2), max_ctc=round(float(live.ctc.max()), 2), at_risk=len(risk),
                docs=o[o.status == "Accepted"].doc_status.value_counts().to_dict(), conflicts=len(scheduler.detect(S["drives"], S["short"])),
                funnel=[dict(stage=a, value=b) for a, b in funnel])

@app.get("/api/analytics")
def analytics():
    df, o, h = S["df"], S["offers"], S["history"]
    br = df.groupby("branch").agg(total=("id", "count"), placed=("placed", "sum"), readiness=("readiness", "mean")).reset_index()
    br["rate"] = (br.placed / br.total * 100).round(1); br["readiness"] = br.readiness.round(1)
    rows = []
    for s in sorted({x for l in df.skills for x in l}):
        m = df.skills.apply(lambda l: s in l)
        rows.append(dict(skill=s, students=int(m.sum()), rate=round(float(df[m].placed.mean()) * 100, 1)))
    live = o[o.status != "Withdrawn"]
    cur = dict(year=2026, hires=len(live), avg_ctc=round(float(live.ctc.mean()), 2), max_ctc=round(float(live.ctc.max()), 2))
    hy = h.groupby("year").apply(lambda g: pd.Series(dict(hires=g.hires.sum(), avg_ctc=round((g.avg_ctc * g.hires).sum() / g.hires.sum(), 2), max_ctc=g.max_ctc.max())), include_groups=False).reset_index()
    rec = live.groupby("company").agg(offers=("id", "count"), avg_ctc=("ctc", "mean"), accepted=("status", lambda s: int((s == "Accepted").sum()))).reset_index()
    rep = h.groupby("company").year.nunique().rename("years_hired").reset_index()
    rec = rec.merge(rep, on="company", how="left").fillna(0)
    rec["avg_ctc"] = rec.avg_ctc.round(2); rec["engagement"] = rec.years_hired.apply(lambda y: "Repeat recruiter" if y >= 3 else "Occasional" if y >= 1 else "New")
    return dict(branch=js(br), skills=sorted(rows, key=lambda r: -r["students"])[:12],
                packages=js(hy) + [cur], recruiters=js(rec.sort_values("offers", ascending=False)),
                readiness_dist=df.level.value_counts().reindex(ml.LABELS, fill_value=0).to_dict())

@app.get("/api/students")
def students(search: str = "", branch: str = "", level: str = "", limit: int = 100):
    df = S["df"]
    if search: df = df[df.name.str.contains(search, case=False) | df.id.str.contains(search, case=False)]
    if branch: df = df[df.branch == branch]
    if level: df = df[df.level == level]
    cols = ["id", "name", "branch", "cgpa", "backlogs", "readiness", "level", "placed", "p_place"]
    return dict(total=len(df), rows=js(df.sort_values("readiness", ascending=False)[cols].head(limit)))

@app.get("/api/students/{sid}")
def student_profile(sid: str):
    s = student(sid)
    gaps = ml.skill_gaps(s.skills, S["roles"])
    p = {k: (v.item() if hasattr(v, "item") else v) for k, v in s.drop(["ability"]).items()}
    p["components"], p["max"] = s.components, dict(Academics=20, Aptitude=20, **{"Mock interview": 20}, Communication=15, Skills=12, **{"Projects & certs": 13})
    p["gaps"], p["recommendations"] = gaps, ml.recommendations(s, gaps)
    p["drives"] = [dict(drive=d["id"], company=d["company"], status=S["matcher"].run(d["company_id"]).set_index("id").loc[sid, "status"],
                        fit=float(S["matcher"].run(d["company_id"]).set_index("id").loc[sid, "fit"])) for d in S["drives"]]
    return json.loads(json.dumps(p, default=str))

@app.get("/api/drives")
def drives():
    return [dict(d, requirements=S["companies"][d["company_id"]], shortlisted=len(S["short"][d["id"]])) for d in S["drives"]]

@app.get("/api/drives/{did}/match")
def match(did: str, view: str = "Shortlisted", limit: int = 50):
    d = next((x for x in S["drives"] if x["id"] == did), None)
    if not d: raise HTTPException(404, "Drive not found")
    m = S["matcher"].run(d["company_id"])
    counts = m.status.value_counts().to_dict()
    sel = m[m.status == view].head(limit) if view != "All" else m.head(limit)
    return dict(drive=d, company=S["companies"][d["company_id"]], counts=counts,
                rows=js(sel[["id", "name", "branch", "cgpa", "fit", "level", "cov", "status", "explanation"]]))

@app.get("/api/scheduler")
def sched():
    return dict(drives=S["drives"], conflicts=scheduler.detect(S["drives"], S["short"]))

@app.post("/api/scheduler/resolve")
def sched_resolve():
    new, changes = scheduler.resolve(S["drives"], S["short"])
    S["drives"] = new
    for c in changes:
        S["notes"].insert(0, notifier.make("Shortlisted students", "Schedule change", f'{c["company"]} drive moved to {c["after"]}.', "Email + WhatsApp", len(S["short"][c["drive"]])))
    return dict(changes=changes, conflicts=scheduler.detect(new, S["short"]), drives=new)

@app.post("/api/scheduler/reset")
def sched_reset():
    S["drives"] = [dict(x) for x in S["original_drives"]]
    return sched()

class OfferUpdate(BaseModel):
    status: str | None = None
    doc_status: str | None = None

@app.get("/api/offers")
def offers(status: str = "", company: str = ""):
    o = S["offers"]
    if status: o = o[o.status == status]
    if company: o = o[o.company == company]
    return dict(total=len(o), rows=js(o.sort_values("offer_date", ascending=False).head(200)),
                companies=sorted(S["offers"].company.unique().tolist()),
                by_type=S["offers"].type.value_counts().to_dict())

@app.post("/api/offers/{oid}")
def update_offer(oid: str, u: OfferUpdate):
    o = S["offers"]; i = o.index[o.id == oid]
    if not len(i): raise HTTPException(404, "Offer not found")
    if u.status:
        if u.status not in ["Accepted", "Pending", "Deferred", "Withdrawn"]: raise HTTPException(400, "Invalid status")
        o.loc[i, "status"] = u.status
    if u.doc_status:
        if u.doc_status not in ["Pending", "Submitted", "Verified"]: raise HTTPException(400, "Invalid document status")
        o.loc[i, "doc_status"] = u.doc_status
    r = o.loc[i[0]]
    S["notes"].insert(0, notifier.make(r.student, "Offer status update", f"{r.company}: offer is {r.status}, documents {r.doc_status}.", "Email + App"))
    return js(o.loc[i])[0]

@app.get("/api/at-risk")
def at_risk(limit: int = 60):
    df = S["df"]; r = df[(~df.placed) & (df.p_place < .5)].sort_values("p_place").head(limit)
    rows = []
    for _, s in r.iterrows():
        comp = {k: v / {"Academics": 20, "Aptitude": 20, "Mock interview": 20, "Communication": 15, "Skills": 12, "Projects & certs": 13}[k] for k, v in s.components.items()}
        weak = sorted(comp, key=comp.get)[:2]
        rows.append(dict(id=s.id, name=s["name"], branch=s.branch, cgpa=s.cgpa, readiness=s.readiness, level=s.level, risk=round(1 - s.p_place, 2),
                         weak_areas=weak, action="Assign mentor + targeted plan for " + " and ".join(w.lower() for w in weak)))
    return dict(total=int(((~df.placed) & (df.p_place < .5)).sum()), rows=rows)

@app.post("/api/at-risk/{sid}/escalate")
def escalate(sid: str):
    s = student(sid)
    n = notifier.make("Mentor panel", "At-risk escalation", f"{s['name']} ({sid}, {s.branch}) escalated for mentoring. Readiness {s.readiness}.", "Email + App")
    S["notes"].insert(0, n); return n

@app.get("/api/notifications")
def notifications(): return S["notes"]

@app.post("/api/notifications/refresh")
def refresh_notes():
    S["notes"] = notifier.build_all(S["drives"], S["short"], S["offers"], S["df"]); return S["notes"]

@app.get("/api/metrics")
def metrics(): return S["metrics"]

class Chat(BaseModel):
    student_id: str
    message: str

@app.post("/api/chat")
def chat(c: Chat):
    s, q = student(c.student_id), c.message.lower()
    for cid, co in S["companies"].items():
        if co["name"].lower().split()[0] in q:
            row = S["matcher"].run(cid).set_index("id").loc[s.id]
            return dict(reply=f"{co['name']}: {row.explanation}")
    if "eligible" in q or "apply" in q:
        ok = [f"{co['name']} (fit {S['matcher'].run(cid).set_index('id').loc[s.id, 'fit']:.0f})" for cid, co in S["companies"].items()
              if data_gen.is_eligible(S["df"][S["df"].id == s.id], co).iloc[0]]
        return dict(reply="You are eligible for: " + (", ".join(ok) if ok else "no drives right now - check the improvement plan."))
    if any(k in q for k in ["improve", "gap", "skill", "prepare"]):
        return dict(reply="Your plan: " + "; ".join(ml.recommendations(s, ml.skill_gaps(s.skills, S["roles"]))))
    if any(k in q for k in ["drive", "schedule", "when", "date"]):
        return dict(reply="Upcoming: " + "; ".join(f"{d['company']} on {d['date']} {d['slot']} ({d['venue']})" for d in S["drives"][:5]))
    if any(k in q for k in ["package", "ctc", "salary"]):
        top = sorted(S["companies"].values(), key=lambda x: -x["ctc"])[:3]
        return dict(reply="Highest packages: " + ", ".join(f"{t['name']} {t['ctc']} LPA" for t in top))
    return dict(reply="Ask me about eligibility, a company name (e.g. 'Nimbus'), skill gaps, drive dates or packages.")

@app.get("/")
def index(): return FileResponse(FRONT / "index.html")
