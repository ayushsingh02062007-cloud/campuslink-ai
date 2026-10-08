# Architecture

```
 Data sources            Backend (FastAPI)                         Frontend
 student CSV / resume -> Ingestion (data_gen / loaders) --+
 recruiter JDs --------> Readiness engine (rules)          |
 drive calendar -------> Matcher (rules + TF-IDF + score)  +-> REST /api/* -> Dashboard (9 views)
 offers / history -----> Scheduler (conflict + greedy)     |
 assessments ----------> At-risk model (GradientBoosting)  |
                         Notifier (rules) ----------------+
```

## Models and algorithms
1. **Readiness score (0-100)**: weighted rule model - academics 20 (CGPA minus backlog penalty), aptitude 20, mock interview 20, communication 15, skills 12, projects and certifications 13. Levels: <45 Not Ready, <60 Developing, <75 Ready, 75+ Highly Employable. Every component is shown, so the score is explainable.
2. **Skill-gap**: per target role, coverage = required skills held / required skills; missing skills map to a course in `COURSES`.
3. **Matching (rule + AI hybrid)**: hard eligibility (CGPA, branch, backlogs) then
   `fit = 30% skill coverage + 5% TF-IDF cosine (JD vs skills/certs/projects) + 50% readiness + 15% mock vs benchmark`.
   Shortlist = eligible, fit >= 55, top 40. The explanation lists the exact reason (CGPA, skill gaps, mock benchmark).
4. **At-risk model**: GradientBoostingClassifier predicts P(placed) from academics, assessments, skills, branch. Out-of-fold predictions are used, so no leakage. Unplaced students with P < 0.5 are flagged and can be escalated to mentors.
5. **Scheduler**: detects venue, panel and shared-shortlisted-student clashes in the same slot. The greedy resolver keeps a drive where it is if free, else tries another venue, another slot, then the next working day.

## Evaluation (simulated data, seed 42)
- Matching: precision@10 0.43 vs 0.39 for CGPA-only and 0.11 random (4.0x lift); recall@25 0.75.
- At-risk model: AUC 0.83, accuracy 0.74, F1 0.80. The simple readiness score alone reaches AUC 0.86, so it is a strong baseline on this synthetic data; re-evaluate on real history.
- Caveat: ground truth is simulated, so real campus data must be used before trusting these numbers.

## Scalability and multi-campus deployment
- Add `campus_id` to every table (multi-tenant); replace in-memory pandas state with PostgreSQL.
- Move embeddings to sentence-transformers + pgvector for semantic JD matching at scale.
- Run notifications and model retraining through Celery/Redis queues; schedule nightly retraining.
- Containerised (`Dockerfile`); deploy behind a load balancer on Kubernetes, one API deployment shared by all campuses.
- Add SSO (campus Google/Microsoft), role-based access (student, placement officer, recruiter, mentor), and an aggregated cross-campus analytics schema.
- Real channels: SendGrid/SES for email, WhatsApp Business API, FCM push for the mobile app.
