"""Conflict detection (venue / panel / student clashes) and greedy auto-rescheduling."""
from datetime import date, timedelta
from itertools import combinations
from .data_gen import SLOTS, VENUES

def detect(drives, shortlists):
    out = []
    for a, b in combinations(drives, 2):
        if a["date"] != b["date"] or a["slot"] != b["slot"]: continue
        kinds, shared = [], shortlists[a["id"]] & shortlists[b["id"]]
        if a["venue"] == b["venue"]: kinds.append("Venue double-booked")
        if a["panel"] == b["panel"]: kinds.append("Interview panel double-booked")
        if shared: kinds.append(f"{len(shared)} students shortlisted for both")
        if kinds:
            out.append(dict(drive_a=a["id"], drive_b=b["id"], company_a=a["company"], company_b=b["company"],
                            date=a["date"], slot=a["slot"], issues=kinds,
                            severity="High" if len(kinds) > 1 or len(shared) > 5 else "Medium"))
    return out

def _free(d, placed, short, dt, slot, venue):
    for p in placed:
        if p["date"] == dt and p["slot"] == slot:
            if p["venue"] == venue or p["panel"] == d["panel"] or short[p["id"]] & short[d["id"]]:
                return False
    return True

def resolve(drives, short):
    """Earliest-date greedy: keep original slot if free, else try another venue, another slot, then next working day."""
    placed, changes = [], []
    for d in sorted(drives, key=lambda x: (x["date"], x["id"])):
        base, done = date.fromisoformat(d["date"]), False
        for off in range(0, 15):
            dt = base + timedelta(days=off)
            if dt.weekday() > 4: continue
            for slot in [d["slot"]] + [s for s in SLOTS if s != d["slot"]]:
                for venue in [d["venue"]] + [v for v in VENUES if v != d["venue"]]:
                    if _free(d, placed, short, str(dt), slot, venue):
                        n = dict(d, date=str(dt), slot=slot, venue=venue)
                        if (n["date"], n["slot"], n["venue"]) != (d["date"], d["slot"], d["venue"]):
                            changes.append(dict(drive=d["id"], company=d["company"],
                                                before=f'{d["date"]} {d["slot"]} @ {d["venue"]}',
                                                after=f'{n["date"]} {n["slot"]} @ {n["venue"]}'))
                        placed.append(n); done = True; break
                if done: break
            if done: break
    return sorted(placed, key=lambda x: x["id"]), changes
