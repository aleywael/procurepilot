#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python seed_demo.py
uvicorn app.main:app --reload
