# ProcurePilot Prototype

A runnable graduation-project prototype of an AI-assisted procurement decision-support workbench.

## Included features

- CSV/XLSX procurement import
- Automatic field mapping with confidence
- Data validation and KPI-specific readiness
- Supplier normalization
- Procurement KPI engine
- Priority exception inbox
- Evidence packets with source PO rows
- What Changed comparison between imports
- Supplier scorecards and comparisons
- ML late-delivery risk model when enough labeled data exists
- Grounded AI Analyst demo that only explains stored evidence
- Decision journal
- Outcome review against later imports
- Notifications/activity feed
- Admin view for rules and mappings
- Audit trail
- Synthetic demo dataset generator

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python seed_demo.py
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000

## Prototype note

This is intentionally a prototype. It uses SQLite and an in-process processing pipeline so it is easy to run locally. The production blueprint can replace SQLite with PostgreSQL and move long-running work to Celery + Redis without changing the product workflow.

The AI Analyst included here is evidence-grounded and deterministic by default so the demo works without paid API credentials. The architecture is ready for an external LLM provider later; the deterministic engine remains the source of KPI facts.
