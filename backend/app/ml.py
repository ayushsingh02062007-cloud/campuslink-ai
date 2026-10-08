"""Readiness scoring, skill-gap analysis, NLP matching, explainability and the at-risk model."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from .data_gen import is_eligible

READY_CUTS, FIT_CUTS = (45, 60, 75), (40, 55, 70)
LABELS = ["Not Ready", "Developing", "Ready", "Highly Employable"]
SHORTLIST_MIN, SHORTLIST_CAP = 55, 40
WEIGHTS = dict(skills=.30, text=.05, readiness=.50, mock=.15)  # tuned on simulated data; configurable per campus
COURSES = {"aws": "AWS Cloud Practitioner learning path", "docker": "Docker & containers crash course", "python": "Python for problem solving",
           "sql": "SQL practice (joins, window functions)", "machine learning": "Applied ML with scikit-learn", "java": "Core Java + OOP",
           "react": "React fundamentals mini-project", "node": "Node.js REST API project", "dsa": "DSA 60-day practice plan", "c++": "C++ STL practice",
           "embedded c": "Embedded C on microcontrollers", "vlsi": "Verilog / VLSI design basics", "matlab": "MATLAB & Simulink onboarding",
           "autocad": "AutoCAD drafting certification", "excel": "Advanced Excel & pivot analytics"}

def level(score, cuts):
    return LABELS[sum(score >= c for c in cuts)]

def compute_readiness(df):
    c = pd.DataFrame(index=df.index)
    c["Academics"] = (df.cgpa / 10 * 20 - np.minimum(df.backlogs * 3, 10)).clip(0, 20)
    c["Aptitude"] = df.aptitude / 100 * 20
    c["Mock interview"] = df.mock / 100 * 20
    c["Communication"] = df.comm / 100 * 15
    c["Skills"] = np.minimum(df.skills.apply(len), 8) / 8 * 12
    c["Projects & certs"] = np.minimum(df.projects.apply(len) * 3 + df.certs.apply(len) * 2, 13)
    out = df.copy()
    out["readiness"] = c.sum(axis=1).round(1)
    out["level"] = out.readiness.apply(lambda s: level(s, READY_CUTS))
    out["components"] = c.round(1).to_dict("records")
    return out

def skill_gaps(skills, roles):
    rows = []
    for role, req in roles.items():
        have = [s for s in req if s in skills]
        rows.append(dict(role=role, coverage=round(len(have) / len(req) * 100), missing=[s for s in req if s not in skills]))
    return sorted(rows, key=lambda r: -r["coverage"])

def recommendations(s, gaps):
    recs, best = [], gaps[0]
    for m in best["missing"][:3]:
        recs.append(f"Close the {m} gap for {best['role']}: {COURSES.get(m, m)}")
    if s["mock"] < 60: recs.append("Book 3 mock interviews this fortnight (current score %d)" % s["mock"])
    if s["comm"] < 60: recs.append("Join the communication & group-discussion workshop")
    if s["aptitude"] < 60: recs.append("Do 30 minutes of aptitude practice daily")
    if s["backlogs"] > 0: recs.append("Clear pending backlogs to unlock more recruiter criteria")
    if len(s["projects"]) < 2: recs.append("Add one more project to your portfolio")
    return recs or ["Profile looks strong - apply to the top-ranked drives."]

def train_risk(df):
    """Gradient boosting: probability a student gets placed. Out-of-fold predictions avoid leakage."""
    X = pd.get_dummies(df[["cgpa", "backlogs", "aptitude", "mock", "comm", "readiness", "branch"]])
    X["n_skills"], X["n_proj"], X["n_cert"] = df.skills.apply(len), df.projects.apply(len), df.certs.apply(len)
    y = df.placed.astype(int)
    m = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
    p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    m.fit(X, y)
    imp = sorted(zip(X.columns, m.feature_importances_), key=lambda t: -t[1])[:6]
    out = df.copy(); out["p_place"] = p.round(3)
    met = dict(model="GradientBoostingClassifier (5-fold out-of-fold)", auc=round(roc_auc_score(y, p), 3),
               accuracy=round(accuracy_score(y, p > .5), 3), f1=round(f1_score(y, p > .5), 3),
               baseline_readiness_auc=round(roc_auc_score(y, df.readiness), 3),
               top_features=[dict(feature=f, importance=round(float(i), 3)) for f, i in imp])
    return out, met

class Matcher:
    """Rule + AI hybrid: eligibility rules, skill coverage, TF-IDF cosine (JD vs profile), readiness, mock vs benchmark."""
    def __init__(self, df, companies, hist):
        self.df, self.cos, self.hist, self.cache = df, {c["id"]: c for c in companies}, hist, {}
        self.text = df.apply(lambda r: " ".join(r.skills + r.certs + r.projects), axis=1)
        self.vec = TfidfVectorizer(ngram_range=(1, 2)).fit(list(self.text) + [c["jd"] for c in companies])
        self.S = self.vec.transform(self.text)

    def run(self, cid):
        if cid in self.cache: return self.cache[cid]
        c, d = self.cos[cid], self.df
        req = set(c["skills"])
        cov = d.skills.apply(lambda s: len(req & set(s)) / len(req))
        cs = cosine_similarity(self.vec.transform([c["jd"]]), self.S)[0]
        cs = cs / (cs.max() or 1)
        w = WEIGHTS
        fit = 100 * (w["skills"] * cov + w["text"] * cs + w["readiness"] * d.readiness / 100 + w["mock"] * np.minimum(d.mock / c["mock_bench"], 1))
        m = d[["id", "name", "branch", "cgpa", "backlogs", "mock", "readiness"]].copy()
        m["fit"], m["cov"], m["eligible"] = fit.round(1), (cov * 100).round(0), is_eligible(d, c)
        m["missing"] = d.skills.apply(lambda s: [x for x in c["skills"] if x not in s])
        m["level"] = m.fit.apply(lambda s: level(s, FIT_CUTS))
        m = m.sort_values(["eligible", "fit"], ascending=False).reset_index(drop=True)
        m["status"] = "Not eligible"
        el = m.eligible
        m.loc[el, "status"] = np.where(m.loc[el, "fit"] >= SHORTLIST_MIN, "Waitlisted", "Below threshold")
        top = m[m.status == "Waitlisted"].index[:SHORTLIST_CAP]
        m.loc[top, "status"] = "Shortlisted"
        m["explanation"] = m.apply(lambda r: explain(r, c), axis=1)
        self.cache[cid] = m
        return m

    def shortlist(self, cid):
        m = self.run(cid)
        return set(m[m.status == "Shortlisted"].id)

    def evaluate(self):
        ps, pc, pr, rc = [], [], [], []
        for cid in self.cos:
            m = self.run(cid); e = m[m.eligible]
            truth = set(self.hist[cid])
            ps.append(len(set(e.head(10).id) & truth) / 10)
            pc.append(len(set(e.sort_values("cgpa", ascending=False).head(10).id) & truth) / 10)
            pr.append(len(set(e.id) & truth) / max(len(e), 1))
            rc.append(len(set(e.head(25).id) & truth) / 10)
        r = lambda x: round(float(np.mean(x)), 3)
        return dict(drives_evaluated=len(self.cos), precision_at_10_fit=r(ps), precision_at_10_cgpa_only=r(pc),
                    precision_at_10_random=r(pr), recall_at_25_fit=r(rc),
                    lift_over_random=round(float(np.mean(ps) / max(np.mean(pr), 1e-9)), 1))

def explain(r, c):
    if not r.eligible:
        why = []
        if r.cgpa < c["min_cgpa"]: why.append(f"CGPA {r.cgpa:.1f} is below the required {c['min_cgpa']}")
        if r.branch not in c["branches"]: why.append(f"{r.branch} is not an eligible branch")
        if r.backlogs > c["max_backlogs"]: why.append(f"{int(r.backlogs)} backlog(s) exceed the limit of {c['max_backlogs']}")
        return "Not eligible: " + "; ".join(why) + "."
    parts = [f"CGPA {r.cgpa:.1f} meets the {c['min_cgpa']} cut-off",
             "covers all required skills" if not r.missing else f"skill gap in {', '.join(r.missing)}",
             f"mock-interview score {r.mock:.0f} is {'at or above' if r.mock >= c['mock_bench'] else 'below'} the recruiter benchmark of {c['mock_bench']}"]
    head = {"Shortlisted": "Shortlisted", "Waitlisted": "Waitlisted (eligible, outside the top slots)", "Below threshold": "Below threshold"}[r.status]
    return f"{head}: " + "; ".join(parts) + f". Fit {r.fit:.0f}/100."
