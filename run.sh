#!/usr/bin/env bash
cd "$(dirname "$0")"
pip install -r requirements.txt
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000
