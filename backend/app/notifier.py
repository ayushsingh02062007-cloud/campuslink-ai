"""Rule-based notification automation (email / WhatsApp / app delivery is simulated by a log)."""
import itertools
from datetime import datetime
_n = itertools.count(1)

def make(audience, kind, msg, channel="Email + App", recipients=1):
    return dict(id=next(_n), time=datetime.now().strftime("%d %b %H:%M"), audience=audience, type=kind,
                channel=channel, recipients=int(recipients), message=msg)

def build_all(drives, shortlists, offers, risk_df):
    out = []
    for d in drives:
        out.append(make("All eligible students", "Drive announcement",
                        f'{d["company"]} drive on {d["date"]} ({d["slot"]}) at {d["venue"]}. Check your eligibility in CampusLink.', "Email + WhatsApp", len(risk_df)))
        out.append(make("Shortlisted students", "Shortlist & schedule",
                        f'You are shortlisted for {d["company"]}. Report to {d["venue"]} on {d["date"]}, {d["slot"]}.', "Email + App", len(shortlists[d["id"]])))
    acc = offers[(offers.status == "Accepted") & (offers.doc_status == "Pending")]
    if len(acc):
        out.append(make("Students with pending documents", "Document deadline",
                        f"{len(acc)} students have not submitted joining documents. Upload them before the deadline.", "Email + WhatsApp", len(acc)))
    pend = offers[offers.status == "Pending"]
    if len(pend):
        out.append(make("Recruiters", "Offer follow-up",
                        f"{len(pend)} offers are still pending acceptance. Please confirm status with candidates.", "Email", pend.company.nunique()))
    risk = risk_df[(~risk_df.placed) & (risk_df.p_place < .5)]
    out.append(make("Mentors", "At-risk escalation", f"{len(risk)} unplaced students are at high risk. Assign mentors this week.", "Email + App", len(risk)))
    return out
