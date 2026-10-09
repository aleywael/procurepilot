from __future__ import annotations

import io
import json
import math
import sqlite3
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
DB_PATH = DATA_DIR / "procurepilot.db"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ProcurePilot Prototype")
app.mount("/static", StaticFiles(directory=BASE / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "app" / "templates")

CANONICAL_FIELDS = {
    "po_id": ["po_id", "po", "purchase_order", "purchase order", "purch.doc.", "purch doc", "order_number"],
    "po_line": ["po_line", "line", "item", "line_item", "po item"],
    "supplier_id": ["supplier_id", "vendor", "vendor_id", "supplier code", "supplier_code"],
    "supplier_name": ["supplier", "supplier_name", "vendor name", "vendor_name"],
    "category": ["category", "item_category", "commodity", "material_group", "material group"],
    "order_date": ["order_date", "po_date", "created date", "document date"],
    "promised_date": ["promised_date", "delivery_date", "deliv. date", "requested_delivery_date", "planned_delivery_date"],
    "actual_receipt_date": ["actual_receipt_date", "gr date", "receipt_date", "goods_receipt_date", "actual_delivery_date"],
    "ordered_quantity": ["ordered_quantity", "quantity", "order_qty", "po qty"],
    "received_quantity": ["received_quantity", "received_qty", "gr_qty", "delivered_qty"],
    "rejected_quantity": ["rejected_quantity", "defective_units", "rejected_qty", "defect_qty"],
    "unit_price": ["unit_price", "net price", "price", "negotiated_price"],
    "currency": ["currency", "curr.", "curr"],
    "uom": ["uom", "unit_of_measure", "unit"],
    "status": ["status", "order_status", "po_status"],
    "compliance": ["compliance", "compliant", "policy_compliance"],
}

DEFAULT_RULES = [
    {"code": "LOW_OTD", "name": "Low supplier on-time delivery", "severity": "HIGH", "threshold": "OTD < 80% and sample >= 5"},
    {"code": "OVERDUE_OPEN", "name": "Overdue open purchase orders", "severity": "HIGH", "threshold": "Promised date passed while line remains open"},
    {"code": "HIGH_REJECTION", "name": "High supplier rejection rate", "severity": "MEDIUM", "threshold": "Rejected / received > 5% and received > 0"},
    {"code": "HIGH_CONCENTRATION", "name": "High spend concentration", "severity": "MEDIUM", "threshold": "Supplier category share > 60%"},
]


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    con = db()
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS imports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            mapping_json TEXT NOT NULL,
            readiness_json TEXT NOT NULL,
            kpi_json TEXT NOT NULL,
            model_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS po_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            import_id INTEGER NOT NULL,
            source_row INTEGER,
            po_id TEXT,
            po_line TEXT,
            supplier_id TEXT,
            supplier_name TEXT,
            category TEXT,
            order_date TEXT,
            promised_date TEXT,
            actual_receipt_date TEXT,
            ordered_quantity REAL,
            received_quantity REAL,
            rejected_quantity REAL,
            unit_price REAL,
            currency TEXT,
            uom TEXT,
            status TEXT,
            compliance TEXT,
            FOREIGN KEY(import_id) REFERENCES imports(id)
        );
        CREATE TABLE IF NOT EXISTS exceptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            import_id INTEGER NOT NULL,
            rule_code TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            explanation TEXT NOT NULL,
            impact_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exception_id INTEGER,
            created_at TEXT NOT NULL,
            owner TEXT NOT NULL,
            decision_text TEXT NOT NULL,
            rationale TEXT NOT NULL,
            target_kpi TEXT,
            target_operator TEXT,
            target_value REAL,
            review_date TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN',
            baseline_value REAL,
            outcome_value REAL,
            outcome_status TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            details TEXT
        );
        """
    )
    con.commit()
    con.close()


def audit(action: str, resource_type: str, resource_id: Any = None, details: str = ""):
    con = db()
    con.execute(
        "INSERT INTO audit_events(created_at,action,resource_type,resource_id,details) VALUES(?,?,?,?,?)",
        (datetime.utcnow().isoformat(), action, resource_type, str(resource_id) if resource_id is not None else None, details),
    )
    con.commit(); con.close()


def norm_header(x: str) -> str:
    return " ".join(str(x).strip().lower().replace("_", " ").replace("-", " ").split())


def auto_mapping(columns: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for col in columns:
        n = norm_header(col)
        best = None
        best_score = 0.0
        for canonical, aliases in CANONICAL_FIELDS.items():
            for alias in aliases:
                a = norm_header(alias)
                if n == a:
                    score = 1.0
                elif a in n or n in a:
                    score = 0.88
                else:
                    nt, at = set(n.split()), set(a.split())
                    score = len(nt & at) / max(1, len(nt | at))
                if score > best_score:
                    best_score = score; best = canonical
        if best_score >= 0.45:
            result[col] = {"field": best, "confidence": round(best_score, 2), "method": "alias/semantic heuristic"}
        else:
            result[col] = {"field": None, "confidence": round(best_score, 2), "method": "unresolved"}
    return result


def canonicalize(df: pd.DataFrame, mapping: dict[str, dict[str, Any]]) -> pd.DataFrame:
    reverse = {src: info["field"] for src, info in mapping.items() if info.get("field")}
    out = pd.DataFrame(index=df.index)
    for f in CANONICAL_FIELDS:
        sources = [s for s, c in reverse.items() if c == f]
        out[f] = df[sources[0]] if sources else np.nan
    out["source_row"] = np.arange(2, len(out) + 2)

    for c in ["order_date", "promised_date", "actual_receipt_date"]:
        out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in ["ordered_quantity", "received_quantity", "rejected_quantity", "unit_price"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["supplier_name"] = out["supplier_name"].astype("string").str.strip().replace({"<NA>": pd.NA})
    out["supplier_id"] = out["supplier_id"].astype("string").str.strip().replace({"<NA>": pd.NA})
    out["category"] = out["category"].astype("string").str.strip().replace({"<NA>": pd.NA})
    out["currency"] = out["currency"].astype("string").str.upper().str.strip().replace({"<NA>": pd.NA})
    out["status"] = out["status"].astype("string").str.upper().str.strip().replace({"<NA>": pd.NA})
    out["uom"] = out["uom"].astype("string").str.upper().str.strip().replace({"<NA>": pd.NA})

    # lightweight supplier normalization for prototype
    if out["supplier_name"].notna().any():
        normalized = out["supplier_name"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        canon = {}
        for name, n in zip(out["supplier_name"], normalized):
            if pd.isna(name) or pd.isna(n):
                continue
            canon.setdefault(n, str(name).strip())
        out["supplier_name"] = normalized.map(canon).fillna(out["supplier_name"])
    return out


def readiness(out: pd.DataFrame) -> dict[str, Any]:
    def pct(cols):
        if len(out) == 0: return 0.0
        return float(out[list(cols)].notna().all(axis=1).mean())
    checks = {}
    specs = {
        "on_time_delivery": ["supplier_name", "promised_date", "actual_receipt_date"],
        "delay": ["supplier_name", "promised_date", "actual_receipt_date"],
        "overdue_open": ["promised_date", "status", "ordered_quantity", "unit_price"],
        "spend_concentration": ["supplier_name", "category", "ordered_quantity", "unit_price", "currency"],
        "quality": ["supplier_name", "received_quantity", "rejected_quantity"],
        "price": ["supplier_name", "category", "unit_price", "currency", "uom"],
    }
    for k, cols in specs.items():
        missing_cols = [c for c in cols if out[c].isna().all()]
        coverage = pct(cols)
        status = "READY" if coverage >= 0.9 and not missing_cols else ("CAUTION" if coverage >= 0.6 and not missing_cols else "NOT_READY")
        checks[k] = {
            "status": status,
            "coverage": round(coverage * 100, 1),
            "required_fields": cols,
            "fully_missing": missing_cols,
            "message": "Analysis is safe to use." if status == "READY" else ("Analysis is possible but limitations may influence the result." if status == "CAUTION" else "Required procurement evidence is missing or insufficient."),
        }
    return checks


def kpis(out: pd.DataFrame, ready: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    # OTD
    if ready["on_time_delivery"]["status"] != "NOT_READY":
        elig = out[out["promised_date"].notna() & out["actual_receipt_date"].notna()].copy()
        if len(elig):
            elig["on_time"] = elig["actual_receipt_date"] <= elig["promised_date"]
            result["overall_otd"] = round(float(elig["on_time"].mean() * 100), 1)
            result["otd_sample"] = int(len(elig))
            result["supplier_otd"] = {
                str(k): {"value": round(float(g["on_time"].mean()*100),1), "sample": int(len(g))}
                for k,g in elig.groupby("supplier_name", dropna=True)
            }
    # delay
    if ready["delay"]["status"] != "NOT_READY":
        elig = out[out["promised_date"].notna() & out["actual_receipt_date"].notna()].copy()
        if len(elig):
            elig["delay_days"] = (elig["actual_receipt_date"] - elig["promised_date"]).dt.days.clip(lower=0)
            result["avg_positive_delay"] = round(float(elig["delay_days"].mean()),1)
    # spend
    spend = out[out["ordered_quantity"].notna() & out["unit_price"].notna()].copy()
    if len(spend):
        spend["line_value"] = spend["ordered_quantity"] * spend["unit_price"]
        result["total_spend"] = round(float(spend["line_value"].sum()),2)
        result["supplier_spend"] = {str(k): round(float(v),2) for k,v in spend.groupby("supplier_name")["line_value"].sum().items()}
    # overdue
    today = pd.Timestamp(date.today())
    open_mask = out["status"].fillna("").str.contains("OPEN|PARTIAL|PENDING", regex=True)
    overdue = out[open_mask & out["promised_date"].notna() & (out["promised_date"] < today)].copy()
    if len(overdue):
        overdue["line_value"] = overdue["ordered_quantity"].fillna(0) * overdue["unit_price"].fillna(0)
        result["overdue_lines"] = int(len(overdue))
        result["overdue_exposure"] = round(float(overdue["line_value"].sum()),2)
    else:
        result["overdue_lines"] = 0; result["overdue_exposure"] = 0.0
    # quality
    q = out[out["received_quantity"].fillna(0) > 0].copy()
    if len(q):
        q["rej_rate"] = q["rejected_quantity"].fillna(0) / q["received_quantity"]
        result["quality_rate"] = round(float(q["rej_rate"].mean()*100),2)
        result["supplier_rejection"] = {str(k): round(float(g["rejected_quantity"].fillna(0).sum()/max(g["received_quantity"].sum(),1)*100),2) for k,g in q.groupby("supplier_name")}
    return result


def generate_exceptions(import_id: int, out: pd.DataFrame, ready: dict[str, Any], kp: dict[str, Any]):
    con = db(); con.execute("DELETE FROM exceptions WHERE import_id=?", (import_id,))
    now = datetime.utcnow().isoformat()
    # low OTD supplier
    for supplier, item in kp.get("supplier_otd", {}).items():
        if item["sample"] >= 5 and item["value"] < 80:
            rows = out[(out["supplier_name"] == supplier) & out["promised_date"].notna() & out["actual_receipt_date"].notna()].copy()
            rows["late"] = rows["actual_receipt_date"] > rows["promised_date"]
            late = rows[rows["late"]]
            evidence = late[["po_id","po_line","promised_date","actual_receipt_date","source_row"]].head(30).astype(str).to_dict("records")
            con.execute("""INSERT INTO exceptions(import_id,rule_code,severity,title,entity_type,entity_id,explanation,impact_json,evidence_json,status,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,'OPEN',?)""",
                (import_id,"LOW_OTD","HIGH",f"{supplier}: low on-time delivery","supplier",supplier,
                 f"Supplier OTD is {item['value']}% across {item['sample']} eligible delivered lines. Threshold is below 80% with sample >= 5.",
                 json.dumps({"otd":item["value"],"sample":item["sample"],"late_lines":len(late),"readiness":ready["on_time_delivery"]["status"]}),json.dumps(evidence),now))
    # overdue lines
    today = pd.Timestamp(date.today())
    open_mask = out["status"].fillna("").str.contains("OPEN|PARTIAL|PENDING", regex=True)
    overdue = out[open_mask & out["promised_date"].notna() & (out["promised_date"] < today)].copy()
    if len(overdue):
        overdue["line_value"] = overdue["ordered_quantity"].fillna(0) * overdue["unit_price"].fillna(0)
        for supplier,g in overdue.groupby("supplier_name"):
            if pd.isna(supplier): continue
            evidence = g[["po_id","po_line","promised_date","status","ordered_quantity","unit_price","source_row"]].head(30).astype(str).to_dict("records")
            con.execute("""INSERT INTO exceptions(import_id,rule_code,severity,title,entity_type,entity_id,explanation,impact_json,evidence_json,status,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,'OPEN',?)""",
                (import_id,"OVERDUE_OPEN","HIGH",f"{supplier}: overdue open PO lines","supplier",str(supplier),
                 f"{len(g)} open line(s) have a promised date before today.",json.dumps({"overdue_lines":len(g),"exposure":round(float(g['line_value'].sum()),2)}),json.dumps(evidence),now))
    # rejection
    for supplier, rate in kp.get("supplier_rejection", {}).items():
        if rate > 5:
            rows = out[out["supplier_name"] == supplier]
            evidence = rows[["po_id","po_line","received_quantity","rejected_quantity","source_row"]].head(30).astype(str).to_dict("records")
            con.execute("""INSERT INTO exceptions(import_id,rule_code,severity,title,entity_type,entity_id,explanation,impact_json,evidence_json,status,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,'OPEN',?)""",
                (import_id,"HIGH_REJECTION","MEDIUM",f"{supplier}: elevated rejection rate","supplier",supplier,
                 f"Supplier rejection rate is {rate}% against a 5% prototype threshold.",json.dumps({"rejection_rate":rate}),json.dumps(evidence),now))
    # concentration per category
    spend = out[out["ordered_quantity"].notna() & out["unit_price"].notna() & out["supplier_name"].notna() & out["category"].notna()].copy()
    if len(spend):
        spend["line_value"] = spend["ordered_quantity"] * spend["unit_price"]
        cat_tot = spend.groupby("category")["line_value"].sum()
        pair = spend.groupby(["category","supplier_name"])["line_value"].sum()
        for (cat,sup), val in pair.items():
            share = val / max(cat_tot.loc[cat], 1)
            if share > 0.60 and cat_tot.loc[cat] > 0:
                evidence = spend[(spend["category"]==cat)&(spend["supplier_name"]==sup)][["po_id","po_line","ordered_quantity","unit_price","currency","source_row"]].head(30).astype(str).to_dict("records")
                con.execute("""INSERT INTO exceptions(import_id,rule_code,severity,title,entity_type,entity_id,explanation,impact_json,evidence_json,status,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,'OPEN',?)""",
                    (import_id,"HIGH_CONCENTRATION","MEDIUM",f"{cat}: high supplier concentration","category",str(cat),
                     f"{sup} represents {round(share*100,1)}% of measured {cat} spend.",json.dumps({"supplier":str(sup),"category":str(cat),"share":round(share*100,1),"spend":round(float(val),2)}),json.dumps(evidence),now))
    con.commit(); con.close()


def train_risk_model(out: pd.DataFrame) -> dict[str, Any]:
    df = out[out["promised_date"].notna() & out["actual_receipt_date"].notna()].copy()
    if len(df) < 40:
        return {"status":"NOT_AVAILABLE","reason":"Need at least 40 delivered records for the prototype ML risk model."}
    df["late"] = (df["actual_receipt_date"] > df["promised_date"]).astype(int)
    if df["late"].nunique() < 2:
        return {"status":"NOT_AVAILABLE","reason":"Historical records need both late and on-time examples."}
    df["lead_days"] = (df["promised_date"] - df["order_date"]).dt.days
    features = ["supplier_name","category","ordered_quantity","unit_price","lead_days"]
    X = df[features].copy(); y = df["late"]
    for c in ["supplier_name","category"]: X[c] = X[c].fillna("UNKNOWN")
    for c in ["ordered_quantity","unit_price","lead_days"]: X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
    cat = ["supplier_name","category"]; num=["ordered_quantity","unit_price","lead_days"]
    prep = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",StandardScaler(),num)])
    model = Pipeline([("prep",prep),("rf",RandomForestClassifier(n_estimators=80,max_depth=6,random_state=42,class_weight="balanced"))])
    model.fit(X,y); pred=model.predict(X)
    acc=float(accuracy_score(y,pred))
    # current/open risk estimates use records that have promised date; if actual missing, score them
    cand = out[out["promised_date"].notna() & out["actual_receipt_date"].isna()].copy()
    risks=[]
    if len(cand):
        cand["lead_days"]=(cand["promised_date"]-cand["order_date"]).dt.days
        Xc=cand[features].copy()
        for c in cat: Xc[c]=Xc[c].fillna("UNKNOWN")
        for c in num: Xc[c]=Xc[c].fillna(df[c].median() if df[c].notna().any() else 0)
        p=model.predict_proba(Xc)[:,1]
        for idx,prob in zip(cand.index,p):
            risks.append({"po_id":str(out.loc[idx,"po_id"]),"po_line":str(out.loc[idx,"po_line"]),"supplier":str(out.loc[idx,"supplier_name"]),"risk":round(float(prob*100),1)})
        risks=sorted(risks,key=lambda x:x["risk"],reverse=True)[:20]
    return {"status":"READY","training_rows":len(df),"training_accuracy":round(acc*100,1),"note":"Prototype in-sample score; not a production validation metric.","top_open_risks":risks}


def store_import(filename: str, content: bytes) -> int:
    ext = Path(filename).suffix.lower()
    if ext not in {".csv", ".xlsx"}: raise HTTPException(400,"Only CSV and XLSX are supported in this prototype.")
    if len(content) > 20 * 1024 * 1024: raise HTTPException(400,"Prototype upload limit is 20 MB.")
    try:
        df = pd.read_csv(io.BytesIO(content)) if ext == ".csv" else pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    if len(df) == 0: raise HTTPException(400,"File has no data rows.")
    mapping=auto_mapping([str(c) for c in df.columns])
    out=canonicalize(df,mapping)
    rd=readiness(out); kp=kpis(out,rd); model=train_risk_model(out)
    h=hashlib.sha256(content).hexdigest()
    safe_name=f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{Path(filename).name}"
    (UPLOAD_DIR/safe_name).write_bytes(content)
    con=db()
    cur=con.execute("INSERT INTO imports(filename,uploaded_at,row_count,status,file_hash,mapping_json,readiness_json,kpi_json,model_json) VALUES(?,?,?,?,?,?,?,?,?)",
        (safe_name,datetime.utcnow().isoformat(),len(out),"PUBLISHED",h,json.dumps(mapping),json.dumps(rd),json.dumps(kp),json.dumps(model)))
    import_id=cur.lastrowid
    cols=list(CANONICAL_FIELDS.keys())
    sql=f"INSERT INTO po_lines(import_id,source_row,{','.join(cols)}) VALUES({','.join(['?']*(2+len(cols)))})"
    for _,r in out.iterrows():
        vals=[]
        for c in cols:
            v=r[c]
            if pd.isna(v): vals.append(None)
            elif isinstance(v,pd.Timestamp): vals.append(v.isoformat())
            elif isinstance(v,np.generic): vals.append(v.item())
            else: vals.append(v)
        con.execute(sql,[import_id,int(r["source_row"]),*vals])
    con.commit(); con.close()
    generate_exceptions(import_id,out,rd,kp)
    audit("IMPORT_PUBLISHED","import",import_id,f"{len(out)} rows from {filename}")
    evaluate_due_decisions(import_id)
    return import_id


def import_record(import_id: int):
    con=db(); row=con.execute("SELECT * FROM imports WHERE id=?",(import_id,)).fetchone(); con.close(); return row


def latest_import_id() -> Optional[int]:
    con=db(); row=con.execute("SELECT id FROM imports ORDER BY id DESC LIMIT 1").fetchone(); con.close(); return int(row[0]) if row else None


def df_for_import(import_id:int)->pd.DataFrame:
    con=db(); df=pd.read_sql_query("SELECT * FROM po_lines WHERE import_id=?",con,params=(import_id,)); con.close()
    for c in ["order_date","promised_date","actual_receipt_date"]: df[c]=pd.to_datetime(df[c],errors="coerce")
    return df


def changes_for(import_id:int)->dict[str,list[dict[str,Any]]]:
    con=db()
    prev=con.execute("SELECT id FROM imports WHERE id < ? ORDER BY id DESC LIMIT 1",(import_id,)).fetchone()
    current=[dict(r) for r in con.execute("SELECT * FROM exceptions WHERE import_id=?",(import_id,)).fetchall()]
    if not prev:
        con.close(); return {"new":current,"resolved":[],"persistent":[]}
    previous=[dict(r) for r in con.execute("SELECT * FROM exceptions WHERE import_id=?",(prev[0],)).fetchall()]
    con.close()
    key=lambda r:(r["rule_code"],r["entity_type"],r["entity_id"])
    cm={key(r):r for r in current}; pm={key(r):r for r in previous}
    return {"new":[cm[k] for k in cm.keys()-pm.keys()],"resolved":[pm[k] for k in pm.keys()-cm.keys()],"persistent":[cm[k] for k in cm.keys()&pm.keys()]}


def supplier_summary(import_id:int)->list[dict[str,Any]]:
    rec=import_record(import_id); kp=json.loads(rec["kpi_json"])
    df=df_for_import(import_id)
    suppliers=sorted([x for x in df["supplier_name"].dropna().unique()])
    out=[]
    for s in suppliers:
        ss=df[df["supplier_name"]==s]
        spend=float((ss["ordered_quantity"].fillna(0)*ss["unit_price"].fillna(0)).sum())
        out.append({"supplier":s,"po_lines":len(ss),"spend":round(spend,2),"otd":kp.get("supplier_otd",{}).get(s,{}).get("value"),"rejection":kp.get("supplier_rejection",{}).get(s),"open_lines":int(ss["status"].fillna("").str.contains("OPEN|PARTIAL|PENDING",regex=True).sum())})
    return out


def metric_value(import_id:int,kpi:str, entity_id:Optional[str]=None)->Optional[float]:
    rec=import_record(import_id)
    if not rec:return None
    kp=json.loads(rec["kpi_json"])
    if kpi=="overall_otd": return kp.get("overall_otd")
    if kpi=="overdue_exposure": return kp.get("overdue_exposure")
    if kpi=="quality_rate": return kp.get("quality_rate")
    if kpi=="supplier_otd" and entity_id: return kp.get("supplier_otd",{}).get(entity_id,{}).get("value")
    return None


def evaluate_due_decisions(import_id:int):
    con=db(); decs=con.execute("SELECT d.*, e.entity_id FROM decisions d LEFT JOIN exceptions e ON d.exception_id=e.id WHERE d.status='OPEN' AND d.target_kpi IS NOT NULL").fetchall()
    for d in decs:
        value=metric_value(import_id,d["target_kpi"],d["entity_id"])
        if value is None: continue
        target=d["target_value"]
        op=d["target_operator"] or ">="
        met = value>=target if op==">=" else value<=target if op=="<=" else value==target
        con.execute("UPDATE decisions SET outcome_value=?, outcome_status=?, status='REVIEWED' WHERE id=?",(value,"ACHIEVED" if met else "NOT_ACHIEVED",d["id"]))
    con.commit(); con.close()


def grounded_ai_answer(question:str, import_id:int, exception_id:Optional[int]=None)->dict[str,Any]:
    q=question.lower(); rec=import_record(import_id)
    if not rec: return {"answer":"No procurement dataset is loaded yet.","evidence":[]}
    kp=json.loads(rec["kpi_json"]); rd=json.loads(rec["readiness_json"])
    evidence=[]; parts=[]; caveats=[]
    con=db()
    if exception_id:
        ex=con.execute("SELECT * FROM exceptions WHERE id=? AND import_id=?",(exception_id,import_id)).fetchone()
        if ex:
            parts.append(ex["explanation"]); evidence.append(f"EXC-{ex['id']}")
            imp=json.loads(ex["impact_json"]); parts.append("Measured impact: "+", ".join(f"{k}={v}" for k,v in imp.items()))
            caveats.append("The system identifies the observed condition; it does not infer a supplier root cause unless that evidence exists in the data.")
    if any(w in q for w in ["overdue","late open","open po"]):
        parts.append(f"There are {kp.get('overdue_lines',0)} overdue open lines with measured exposure {kp.get('overdue_exposure',0):,.2f}.")
        evidence.append(f"IMP-{import_id}-KPI-overdue")
    if any(w in q for w in ["on time","otd","delivery"]):
        if kp.get("overall_otd") is not None:
            parts.append(f"Overall on-time delivery is {kp['overall_otd']}% across {kp.get('otd_sample',0)} eligible delivered lines.")
            parts.append(f"Delivery readiness is {rd['on_time_delivery']['status']} with {rd['on_time_delivery']['coverage']}% coverage.")
            evidence.extend([f"IMP-{import_id}-KPI-otd",f"IMP-{import_id}-READY-otd"])
    if "supplier" in q or "worst" in q or "compare" in q:
        sotd=kp.get("supplier_otd",{})
        if sotd:
            ranked=sorted(sotd.items(),key=lambda x:x[1]["value"])
            low=ranked[:3]
            parts.append("Lowest measured supplier OTD: "+"; ".join(f"{s}: {v['value']}% (n={v['sample']})" for s,v in low)+".")
            evidence.extend([f"SUP-{s}" for s,_ in low])
    if any(w in q for w in ["risk","prediction","predict"]):
        m=json.loads(rec["model_json"])
        if m.get("status")=="READY":
            parts.append(f"The prototype ML model was trained on {m['training_rows']} delivered records. Its in-sample accuracy is {m['training_accuracy']}%, which is not a production validation score.")
            if m.get("top_open_risks"):
                top=m["top_open_risks"][:5]
                parts.append("Highest estimated open-line delay risks: "+"; ".join(f"{x['po_id']}/{x['po_line']} {x['risk']}%" for x in top)+".")
                evidence.append(f"IMP-{import_id}-ML-risk")
        else: parts.append("Predictive late-delivery risk is not available: "+m.get("reason","insufficient data"))
    if not parts:
        exs=con.execute("SELECT * FROM exceptions WHERE import_id=? AND status='OPEN' ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END, id DESC LIMIT 5",(import_id,)).fetchall()
        if exs:
            parts.append("The highest-priority current findings are: "+"; ".join(f"{e['title']} [{e['severity']}]" for e in exs)+".")
            evidence.extend([f"EXC-{e['id']}" for e in exs])
        else: parts.append("No open exceptions are currently detected by the configured prototype rules.")
    con.close()
    parts.append("Suggested next step: open the relevant evidence packet, verify the affected PO lines and data readiness, then record the human procurement decision if action is required.")
    return {"answer":" ".join(parts),"evidence":evidence,"caveats":caveats}


@app.on_event("startup")
def startup(): init_db()

@app.get("/", response_class=HTMLResponse)
def overview(request: Request):
    imp=latest_import_id()
    context={"request":request,"page":"overview","import_id":imp}
    if imp:
        rec=import_record(imp); con=db()
        context.update({"import":dict(rec),"readiness":json.loads(rec["readiness_json"]),"kpi":json.loads(rec["kpi_json"]),"model":json.loads(rec["model_json"]),
            "exceptions":[dict(r) for r in con.execute("SELECT * FROM exceptions WHERE import_id=? ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END,id DESC LIMIT 8",(imp,)).fetchall()],
            "decisions":[dict(r) for r in con.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT 6").fetchall()],"changes":changes_for(imp)})
        con.close()
    return templates.TemplateResponse("index.html",context)

@app.get("/imports", response_class=HTMLResponse)
def imports_page(request:Request):
    con=db(); rows=[dict(r) for r in con.execute("SELECT * FROM imports ORDER BY id DESC").fetchall()]; con.close()
    return templates.TemplateResponse("imports.html",{"request":request,"page":"imports","imports":rows})

@app.post("/imports")
async def upload_import(file:UploadFile=File(...)):
    content=await file.read(); iid=store_import(file.filename or "upload.csv",content)
    return RedirectResponse(url=f"/imports/{iid}",status_code=303)

@app.get("/imports/{import_id}", response_class=HTMLResponse)
def import_detail(request:Request,import_id:int):
    rec=import_record(import_id)
    if not rec: raise HTTPException(404)
    con=db(); sample=[dict(r) for r in con.execute("SELECT * FROM po_lines WHERE import_id=? LIMIT 20",(import_id,)).fetchall()]; con.close()
    return templates.TemplateResponse("import_detail.html",{"request":request,"page":"imports","import":dict(rec),"mapping":json.loads(rec["mapping_json"]),"readiness":json.loads(rec["readiness_json"]),"kpi":json.loads(rec["kpi_json"]),"model":json.loads(rec["model_json"]),"sample":sample})

@app.get("/issues", response_class=HTMLResponse)
def issues(request:Request):
    imp=latest_import_id(); con=db(); rows=[dict(r) for r in con.execute("SELECT * FROM exceptions WHERE import_id=? ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END,id DESC",(imp or -1,)).fetchall()]; con.close()
    return templates.TemplateResponse("issues.html",{"request":request,"page":"issues","exceptions":rows,"import_id":imp})

@app.get("/issues/{exception_id}", response_class=HTMLResponse)
def issue_detail(request:Request,exception_id:int):
    con=db(); ex=con.execute("SELECT * FROM exceptions WHERE id=?",(exception_id,)).fetchone(); con.close()
    if not ex: raise HTTPException(404)
    return templates.TemplateResponse("issue_detail.html",{"request":request,"page":"issues","ex":dict(ex),"impact":json.loads(ex["impact_json"]),"evidence":json.loads(ex["evidence_json"])})

@app.get("/suppliers", response_class=HTMLResponse)
def suppliers(request:Request):
    imp=latest_import_id(); rows=supplier_summary(imp) if imp else []
    return templates.TemplateResponse("suppliers.html",{"request":request,"page":"suppliers","suppliers":rows,"import_id":imp})

@app.get("/decisions", response_class=HTMLResponse)
def decisions_page(request:Request):
    con=db(); rows=[dict(r) for r in con.execute("SELECT d.*,e.title exception_title,e.entity_id FROM decisions d LEFT JOIN exceptions e ON d.exception_id=e.id ORDER BY d.id DESC").fetchall()]; con.close()
    return templates.TemplateResponse("decisions.html",{"request":request,"page":"decisions","decisions":rows})

@app.post("/decisions")
def create_decision(exception_id:int=Form(...),owner:str=Form(...),decision_text:str=Form(...),rationale:str=Form(...),target_kpi:str=Form(""),target_operator:str=Form(">="),target_value:str=Form(""),review_date:str=Form("")):
    con=db(); ex=con.execute("SELECT * FROM exceptions WHERE id=?",(exception_id,)).fetchone()
    if not ex: con.close(); raise HTTPException(404)
    baseline=metric_value(ex["import_id"],target_kpi,ex["entity_id"]) if target_kpi else None
    tv=float(target_value) if target_value.strip() else None
    con.execute("INSERT INTO decisions(exception_id,created_at,owner,decision_text,rationale,target_kpi,target_operator,target_value,review_date,baseline_value) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (exception_id,datetime.utcnow().isoformat(),owner,decision_text,rationale,target_kpi or None,target_operator,tv,review_date or None,baseline))
    con.execute("UPDATE exceptions SET status='DECIDED' WHERE id=?",(exception_id,)); con.commit(); con.close(); audit("DECISION_RECORDED","exception",exception_id)
    return RedirectResponse("/decisions",303)

@app.get("/ai", response_class=HTMLResponse)
def ai_page(request:Request):
    return templates.TemplateResponse("ai.html",{"request":request,"page":"ai","import_id":latest_import_id()})

@app.post("/api/ai")
async def ai_query(request:Request):
    body=await request.json(); imp=int(body.get("import_id") or latest_import_id() or 0)
    ans=grounded_ai_answer(str(body.get("question", "")),imp,body.get("exception_id"))
    audit("AI_QUERY","import",imp,str(body.get("question", ""))[:160])
    return JSONResponse(ans)

@app.get("/admin", response_class=HTMLResponse)
def admin(request:Request):
    con=db(); events=[dict(r) for r in con.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT 40").fetchall()]; con.close()
    return templates.TemplateResponse("admin.html",{"request":request,"page":"admin","rules":DEFAULT_RULES,"events":events})

@app.get("/api/health")
def health(): return {"ok":True,"latest_import_id":latest_import_id()}
