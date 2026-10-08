# CampusLink - AI-Powered Campus-to-Corporate Placement Platform

Profiling -> Matching -> Scheduling -> Notification -> Offer tracking -> Analytics, in one working prototype.

## Run (2 minutes)
```bash
cd campuslink
./run.sh                 # installs requirements and starts the server
# open http://localhost:8000
```
Or with Docker: `docker build -t campuslink . && docker run -p 8000:8000 campuslink`
Tests: `pip install -r requirements.txt && pytest -q tests`

## Folder structure
```
campuslink/
  backend/app/
    main.py        FastAPI REST API (+ serves the dashboard)
    data_gen.py    Simulated data: 600 students, 12 recruiters, 10 drives, offers, 4-year history
    ml.py          Readiness score, skill-gap, TF-IDF matcher, explanations, at-risk model, evaluation
    scheduler.py   Conflict detection and greedy auto-rescheduling
    notifier.py    Automated notification rules
  frontend/index.html   Placement command dashboard (9 views, Chart.js)
  docs/ARCHITECTURE.md  Architecture, algorithms, scalability
  tests/test_core.py    API tests
```

## Problem-statement coverage
| Requirement | Where |
|---|---|
| Readiness / employability score (Not Ready -> Highly Employable) | `ml.compute_readiness`, Student readiness view |
| Skill-gap analysis vs target roles + prep plan | `ml.skill_gaps`, `ml.recommendations`, student profile |
| Recruiter-student matching, 12 simulated drives, explainable | `ml.Matcher`, `ml.explain`, Recruiter matching view |
| Conflict-aware scheduling (venue, panel, shared students) | `scheduler.py`, Drive scheduling view |
| Offer, CTC, bond, PPO, documents, accept/defer/withdraw | `/api/offers`, Offers view |
| Predictive analytics, at-risk students | `ml.train_risk`, At-risk view |
| Notifications automation | `notifier.py`, Notifications view |
| Chatbot for eligibility FAQ | `/api/chat`, Student assistant view |
| Evaluation of matching and scoring | `/api/metrics`, Models view |

## Use your own data
Replace `data_gen.generate()` with loaders for your CSVs. It must return the same keys: `students`, `companies`, `drives`, `offers`, `history`, `hist_selected`, `roles`.
