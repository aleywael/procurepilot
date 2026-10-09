@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install -r requirements.txt
python seed_demo.py
uvicorn app.main:app --reload
