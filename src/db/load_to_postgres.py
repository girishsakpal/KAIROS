"""
Kairos — load the Phase 3 outputs into PostgreSQL (built for Supabase, works on any Postgres 14+).

    python -m src.db.load_to_postgres --dry-run     # validate everything, touch NO database
    python -m src.db.load_to_postgres               # load into an EMPTY database
    python -m src.db.load_to_postgres --reset       # wipe the 8 tables, then reload
    python -m src.db.load_to_postgres --verify-only # just run the health checks

Connection: DATABASE_URL in .env (see docs/SUPABASE_SETUP.md).

What it does, in order
  0. Validates every CSV/JSON (including data/processed/features_*.csv) against db/schema.sql (FKs, duplicates, NOT NULL, VARCHAR lengths,
     JSON shape) BEFORE connecting. A bad file fails here, not halfway through a load.
  1. Ensures the pgvector extension exists (Supabase: installed into the `extensions` schema).
  2. Creates the 8 tables from db/schema.sql if none exist. If an older 7-table database is found,
     applies db/migrations/001_feature_snapshots.sql to add the missing table (then load needs --reset).
  3. Refuses to touch a database that already has rows unless --reset is given.
  4. Loads everything in ONE transaction using COPY (all-or-nothing: any error rolls back).
  5. Resets the serial sequences for tables loaded with explicit ids.
  6. Enables Row Level Security on all 8 tables (see "Security" below).
  7. Verifies counts, sums, JSONB shape, RLS flags and database size.

Design decisions you should know about
  * Both models' predictions are loaded: XGBoost (with SHAP drivers) and the champion (LSTM, which
    has none). The schema makes top_features NOT NULL, so rows without drivers are stored as the
    empty array `[]`, meaning "no per-customer explanation". ANY query that needs drivers must filter
    on model_id (or on jsonb_array_length(top_features) > 0). Use --no-champion-predictions to skip.
  * Security: tables created by SQL are NOT protected by Row Level Security by default, and
    Supabase's public schema can be exposed through its auto-generated Data API. We enable RLS
    with no policies, so the Data API (anon/authenticated roles) sees nothing. The Flask API connects
    with the table owner role through DATABASE_URL, which RLS does not apply to.
  * Customers' ticket/transaction ids are loaded explicitly (they come from the CSVs), so the
    sequences are advanced afterwards; otherwise the first new insert would collide.
"""
import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TABLES = ["customers", "subscriptions", "usage_logs", "support_tickets",
          "transactions", "feature_snapshots", "model_registry", "churn_predictions"]   # FK-safe load order

# Tables added after the original 7: if a database has all the others, the loader creates these
# from the matching migration file instead of refusing.
MIGRATION_FILES = {"feature_snapshots": "001_feature_snapshots.sql"}

COPY_COLUMNS = {
    "customers": ["customer_id", "signup_date", "region", "signup_channel", "age", "gender"],
    "subscriptions": ["customer_id", "plan_type", "contract_type", "billing_cycle", "payment_method",
                      "monthly_charge", "tenure_months", "status", "start_date", "end_date"],
    "usage_logs": ["customer_id", "log_month", "session_count", "avg_session_minutes",
                   "feature_adoption_score", "last_active_date"],
    "support_tickets": ["ticket_id", "customer_id", "created_at", "resolved_at", "category", "priority",
                        "status", "resolution_time_hours", "ticket_text"],
    "transactions": ["transaction_id", "customer_id", "transaction_date", "amount", "transaction_type",
                     "status", "payment_method"],
    "feature_snapshots": ["customer_id", "snapshot_date", "split"],
    "model_registry": ["model_id", "model_name", "algorithm", "trained_at", "feature_list", "roc_auc",
                       "pr_auc", "calibration_error", "drift_psi_score", "drift_ks_pvalue",
                       "drift_detected", "status"],
    "churn_predictions": ["customer_id", "model_id", "predicted_at", "churn_probability", "risk_tier",
                          "top_features", "actual_churn"],
}
EXPLICIT_ID_TABLES = {"support_tickets": "ticket_id", "transactions": "transaction_id"}
BOOL_COLUMNS = {"model_registry": ["drift_detected"], "churn_predictions": ["actual_churn"]}


# ---------------------------------------------------------------------------
# Schema parsing (so validation follows db/schema.sql instead of a hard-coded copy)
# ---------------------------------------------------------------------------

def parse_schema(sql: str):
    """-> {table: {column: {"maxlen": int|None, "notnull": bool}}}"""
    out, current = {}, None
    for raw in sql.splitlines():
        line = raw.split("--")[0].rstrip()
        m = re.match(r"\s*CREATE TABLE (\w+)\s*\(", line, re.I)
        if m:
            current = m.group(1)
            out[current] = {}
            continue
        if current is None:
            continue
        if re.match(r"\s*\)\s*;", line):
            current = None
            continue
        if not line.strip() or re.match(r"\s*(UNIQUE|PRIMARY|FOREIGN|CONSTRAINT)\b", line, re.I):
            continue
        c = re.match(r"\s*(\w+)\s+([A-Za-z]+)(?:\((\d+)(?:,\d+)?\))?(.*)", line)
        if c:
            name, _typ, size, rest = c.groups()
            out[current][name] = {
                "maxlen": int(size) if (size and _typ.upper() == "VARCHAR") else None,
                "notnull": bool(re.search(r"NOT NULL|PRIMARY KEY", rest, re.I)),
            }
    return out


# ---------------------------------------------------------------------------
# Reading + preparing the data (no database needed)
# ---------------------------------------------------------------------------

def _json_file(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_frames(syn_dir: Path, models_dir: Path, include_champion=True, processed_dir=None):
    """Returns (frames, warnings). frames[table] is a DataFrame holding exactly the COPY columns."""
    warnings, frames = [], {}
    for t in ["customers", "subscriptions", "usage_logs", "support_tickets", "transactions"]:
        frames[t] = pd.read_csv(syn_dir / f"{t}.csv")

    # feature snapshots: the as-of date build_features.py used for each customer, and its split
    snaps = []
    if processed_dir is None:
        warnings.append("No processed_dir given - feature_snapshots will be empty.")
    else:
        for split, fname in (("train", "features_train.csv"), ("test", "features_test.csv")):
            f = Path(processed_dir) / fname
            if not f.exists():
                warnings.append(f"{fname} not found in {processed_dir} - those customers get no snapshot.")
                continue
            d = pd.read_csv(f, usecols=["customer_id", "snapshot_date"])
            d["snapshot_date"] = pd.to_datetime(d["snapshot_date"]).dt.strftime("%Y-%m-%d")
            d["split"] = split
            snaps.append(d)
    frames["feature_snapshots"] = (pd.concat(snaps, ignore_index=True) if snaps
                                   else pd.DataFrame(columns=COPY_COLUMNS["feature_snapshots"]))

    # model registry: serving (XGBoost) row first, then champion; de-duplicate by model_id
    rows = []
    for fname in ("xgboost_registry_row.json", "model_registry_row.json"):
        r = _json_file(models_dir / fname)
        if r is None:
            warnings.append(f"{fname} not found - that model will not be registered.")
        elif r["model_id"] not in [x["model_id"] for x in rows]:
            rows.append(r)
    reg = pd.DataFrame(rows)
    if not reg.empty:
        reg["feature_list"] = reg["feature_list"].apply(json.dumps)
    frames["model_registry"] = reg

    # predictions
    sources = [("xgboost_predictions_test.csv", True), ("churn_predictions_test.csv", include_champion)]
    parts = []
    for fname, wanted in sources:
        p = models_dir / fname
        if not wanted:
            continue
        if not p.exists():
            warnings.append(f"{fname} not found - skipped.")
            continue
        # An all-empty top_features column (the LSTM) is read as float64; force text so "[]" can be stored.
        df = pd.read_csv(p, dtype={"top_features": "object"})
        blank = df["top_features"].isna() | (df["top_features"].astype(str).str.strip() == "")
        if blank.any():
            warnings.append(f"{fname}: {int(blank.sum())} rows have no per-customer drivers "
                            f"(model {df['model_id'].iloc[0]}); stored as [] because top_features is NOT NULL.")
            df.loc[blank, "top_features"] = "[]"
        parts.append(df)
    if parts:
        preds = pd.concat(parts, ignore_index=True)
        before = len(preds)
        preds = preds.drop_duplicates(["customer_id", "model_id"], keep="first")
        if len(preds) != before:
            warnings.append(f"Dropped {before - len(preds)} duplicate (customer_id, model_id) prediction rows.")
        frames["churn_predictions"] = preds
    else:
        frames["churn_predictions"] = pd.DataFrame(columns=COPY_COLUMNS["churn_predictions"])

    for t, cols in BOOL_COLUMNS.items():
        for c in cols:
            if c in frames[t].columns:
                frames[t][c] = frames[t][c].map(
                    lambda v: None if pd.isna(v) else ("true" if bool(int(v) if not isinstance(v, bool) else v) else "false"))
    return frames, warnings


def preflight(frames, schema):
    """Returns a list of error strings (empty list = safe to load)."""
    errs = []
    for t in TABLES:
        df = frames[t]
        missing = [c for c in COPY_COLUMNS[t] if c not in df.columns]
        if missing:
            errs.append(f"{t}: missing columns {missing}")
            continue
        for c in COPY_COLUMNS[t]:
            spec = schema.get(t, {}).get(c)
            if spec is None:
                errs.append(f"{t}.{c}: column not found in schema.sql")
                continue
            if spec["notnull"] and df[c].isna().any():
                errs.append(f"{t}.{c}: {int(df[c].isna().sum())} NULL values but the column is NOT NULL")
            if spec["maxlen"] is not None:
                longest = df[c].dropna().astype(str).str.len().max() if len(df) else 0
                if longest > spec["maxlen"]:
                    errs.append(f"{t}.{c}: value of length {longest} exceeds VARCHAR({spec['maxlen']})")
    if errs:
        return errs

    cust = set(frames["customers"]["customer_id"])
    if frames["customers"]["customer_id"].duplicated().any():
        errs.append("customers: duplicate customer_id")
    for t in ["subscriptions", "usage_logs", "support_tickets", "transactions", "feature_snapshots",
              "churn_predictions"]:
        orphans = int((~frames[t]["customer_id"].isin(cust)).sum())
        if orphans:
            errs.append(f"{t}: {orphans} rows reference a customer_id that is not in customers")
    snaps = frames["feature_snapshots"]
    if snaps.empty:
        errs.append("feature_snapshots is empty (data/processed/features_train.csv / features_test.csv missing?)")
    else:
        if snaps["customer_id"].duplicated().any():
            errs.append("feature_snapshots: a customer appears in both train and test (or twice)")
        bad_split = sorted(set(snaps["split"]) - {"train", "test"})
        if bad_split:
            errs.append(f"feature_snapshots: unexpected split value(s) {bad_split}")
    if frames["usage_logs"].duplicated(["customer_id", "log_month"]).any():
        errs.append("usage_logs: duplicate (customer_id, log_month) violates UNIQUE")
    for t, idcol in EXPLICIT_ID_TABLES.items():
        if frames[t][idcol].duplicated().any():
            errs.append(f"{t}: duplicate {idcol}")
    reg_ids = set(frames["model_registry"]["model_id"]) if len(frames["model_registry"]) else set()
    unknown = set(frames["churn_predictions"]["model_id"]) - reg_ids
    if unknown:
        errs.append(f"churn_predictions: model_id(s) {sorted(unknown)} have no model_registry row")
    if len(frames["model_registry"]) and (frames["model_registry"]["status"] == "champion").sum() > 1:
        errs.append("model_registry: more than one 'champion' row")

    preds = frames["churn_predictions"]
    if len(preds):
        if not preds["churn_probability"].between(0, 1).all():
            errs.append("churn_predictions: churn_probability outside [0, 1]")
        for i, raw in preds["top_features"].items():
            try:
                v = json.loads(raw)
                if not isinstance(v, list):
                    raise ValueError("not a list")
            except Exception:
                errs.append(f"churn_predictions row {i}: top_features is not a JSON array")
                break
    return errs


def to_copy_buffer(df: pd.DataFrame, columns):
    """CSV text for COPY ... (FORMAT csv, NULL ''). Empty unquoted field == NULL."""
    buf = io.StringIO()
    df[columns].to_csv(buf, index=False, header=False, na_rep="", lineterminator="\n")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def masked(url: str) -> str:
    parts = urlsplit(url)
    if parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
        return urlunsplit(parts._replace(netloc=netloc))
    return url


def normalize_url(url: str) -> str:
    if not url or "[YOUR-PASSWORD]" in url or "YOUR-PASSWORD" in url.upper():
        raise SystemExit("DATABASE_URL is missing or still contains the [YOUR-PASSWORD] placeholder. "
                         "Edit .env (see docs/SUPABASE_SETUP.md).")
    parts = urlsplit(url)
    if parts.hostname and "supabase" in parts.hostname and "sslmode" not in (parts.query or ""):
        q = (parts.query + "&" if parts.query else "") + "sslmode=require"
        url = urlunsplit(parts._replace(query=q))
    return url


def connect(url: str):
    try:
        import psycopg2
    except ImportError:
        raise SystemExit("psycopg2 is not installed. Run: pip install psycopg2-binary")
    try:
        return psycopg2.connect(url, connect_timeout=15)
    except psycopg2.OperationalError as e:
        msg = str(e)
        hint = ""
        if "could not translate host name" in msg or "Network is unreachable" in msg or "unreachable" in msg:
            hint = ("\nHint: the DIRECT connection string is IPv6-only on Supabase. Use the "
                    "'Session pooler' string from the Connect dialog (works over IPv4).")
        elif "password authentication failed" in msg:
            hint = ("\nHint: wrong password, or the password contains @ : / # ? % and was not URL-encoded. "
                    "Resetting it to letters and digits only avoids this.")
        elif "Tenant or user not found" in msg:
            hint = "\nHint: for the pooler the user must be postgres.<PROJECT-REF>, exactly as shown in the dashboard."
        raise SystemExit(f"Could not connect to {masked(url)}\n{msg.strip()}{hint}")


# ---------------------------------------------------------------------------
# Database steps
# ---------------------------------------------------------------------------

def ensure_vector_extension(conn):
    conn.autocommit = True
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if cur.fetchone():
            return "pgvector already installed"
        last = None
        for stmt in ("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions",
                     "CREATE EXTENSION IF NOT EXISTS vector"):
            try:
                cur.execute(stmt)
                return f"pgvector installed ({stmt.split('WITH')[-1].strip() if 'WITH' in stmt else 'default schema'})"
            except Exception as e:      # noqa: BLE001
                last = e
        raise SystemExit("Could not enable pgvector. In Supabase: Database -> Extensions -> enable 'vector', "
                         f"then re-run.\nLast error: {last}")
    finally:
        conn.autocommit = False


def existing_tables(cur):
    cur.execute("SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY(%s)", (TABLES,))
    return {r[0] for r in cur.fetchall()}


def row_counts(cur, tables):
    out = {}
    for t in tables:
        cur.execute(f"SELECT count(*) FROM {t}")          # t comes from the fixed TABLES list
        out[t] = int(cur.fetchone()[0])
    return out


def run_load(conn, frames, schema_sql, reset=False, assume_yes=False, log=print, migrations=None):
    log(f"[extension] {ensure_vector_extension(conn)}")
    cur = conn.cursor()
    try:
        cur.execute("SET search_path TO public, extensions")   # so VECTOR(384) resolves on Supabase
        cur.execute("SET LOCAL TIME ZONE 'UTC'")               # date-only values load as midnight UTC everywhere
        present = existing_tables(cur)
        missing = set(TABLES) - present
        if present and missing:
            migrations = migrations or {}
            unknown = missing - set(migrations)
            if unknown:
                raise SystemExit(f"Only some tables exist ({sorted(present)}) and there is no migration for "
                                 f"{sorted(unknown)}. Refusing to guess; drop them or fix the database by hand, "
                                 "then re-run.")
            for t in TABLES:                      # keep dependency order
                if t in missing:
                    cur.execute(migrations[t])
                    log(f"[migrate] created missing table {t} from db/migrations/{MIGRATION_FILES[t]}")
            present = existing_tables(cur)
        if not present:
            log(f"[schema] no tables found - creating the {len(TABLES)} tables from db/schema.sql")
            cur.execute(schema_sql)
        else:
            counts = row_counts(cur, TABLES)
            if sum(counts.values()) > 0:
                if not reset:
                    raise SystemExit(f"Database already contains data {counts}.\n"
                                     "Re-run with --reset to wipe and reload, or --verify-only to inspect it.")
                if not assume_yes:
                    ans = input(f"--reset will DELETE all rows in {TABLES}\n  current counts: {counts}\n"
                                "Type RESET to continue: ")
                    if ans.strip() != "RESET":
                        raise SystemExit("Aborted; nothing changed.")
                cur.execute("TRUNCATE TABLE " + ", ".join(reversed(TABLES)) + " RESTART IDENTITY CASCADE")
                log("[reset] tables truncated")

        for t in TABLES:
            df = frames[t]
            cols = COPY_COLUMNS[t]
            cur.copy_expert(f"COPY {t} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '')",
                            to_copy_buffer(df, cols))
            log(f"[load] {t:<18} {len(df):>7} rows")

        for t, idcol in EXPLICIT_ID_TABLES.items():
            cur.execute(f"SELECT setval(pg_get_serial_sequence('{t}', '{idcol}'), "
                        f"(SELECT COALESCE(MAX({idcol}), 1) FROM {t}))")
        log("[sequences] reset for " + ", ".join(EXPLICIT_ID_TABLES))

        for t in TABLES:
            cur.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
        log(f"[security] Row Level Security enabled on all {len(TABLES)} tables (no policies = Data API sees nothing)")
        conn.commit()
        log("[commit] load committed")
    except BaseException:
        conn.rollback()
        log("[rollback] nothing was changed")
        raise


def expected_checks(frames):
    return {"counts": {t: len(frames[t]) for t in TABLES},
            "transactions_amount": round(float(frames["transactions"]["amount"].sum()), 2),
            "monthly_charge": round(float(frames["subscriptions"]["monthly_charge"].sum()), 2)}


def verify(conn, expected=None, log=print):
    """Read-only health checks. Returns True when everything that can be checked passes."""
    cur = conn.cursor()
    ok = True
    counts = row_counts(cur, TABLES)
    for t in TABLES:
        flag = ""
        if expected and counts[t] != expected["counts"][t]:
            flag, ok = f"  <-- expected {expected['counts'][t]}", False
        log(f"[verify] {t:<18} {counts[t]:>7} rows{flag}")
    if expected:
        for label, sql, key in (("transactions.amount sum", "SELECT COALESCE(SUM(amount),0) FROM transactions", "transactions_amount"),
                                ("subscriptions.monthly_charge sum", "SELECT COALESCE(SUM(monthly_charge),0) FROM subscriptions", "monthly_charge")):
            cur.execute(sql)
            got = round(float(cur.fetchone()[0]), 2)
            good = abs(got - expected[key]) < 0.01
            ok &= good
            log(f"[verify] {label}: db {got} vs csv {expected[key]}  {'OK' if good else '<-- MISMATCH'}")
    cur.execute("SELECT count(*) FROM churn_predictions WHERE jsonb_typeof(top_features) <> 'array'")
    bad = int(cur.fetchone()[0])
    ok &= bad == 0
    log(f"[verify] top_features not a JSON array: {bad}  {'OK' if bad == 0 else '<-- PROBLEM'}")
    cur.execute("SELECT model_id, count(*), min(jsonb_array_length(top_features)), max(jsonb_array_length(top_features)) "
                "FROM churn_predictions GROUP BY model_id ORDER BY model_id")
    for model_id, n, lo, hi in cur.fetchall():
        log(f"[verify] predictions {model_id}: {n} rows, drivers per row {lo}-{hi}")
    cur.execute("SELECT count(*) FROM customers c LEFT JOIN feature_snapshots f USING (customer_id) "
                "WHERE f.customer_id IS NULL")
    no_snap = int(cur.fetchone()[0])
    if expected and no_snap:
        ok = False
    log(f"[verify] customers without a feature snapshot: {no_snap}"
        f"{'  <-- PROBLEM' if expected and no_snap else ''}")
    cur.execute("SELECT split, count(*), min(snapshot_date), max(snapshot_date) FROM feature_snapshots "
                "GROUP BY split ORDER BY split")
    for split, n, lo, hi in cur.fetchall():
        log(f"[verify] snapshots {split}: {n} customers, {lo} .. {hi}")
    cur.execute("SELECT relname, relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relname = ANY(%s)", (TABLES,))
    rls = dict(cur.fetchall())
    off = sorted(t for t in TABLES if not rls.get(t))
    ok &= not off
    log(f"[verify] row level security: {'enabled on all ' + str(len(TABLES)) + ' tables' if not off else 'OFF on ' + str(off)}")
    cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
    log(f"[verify] database size: {cur.fetchone()[0]} (Supabase free plan limit: 500 MB)")
    cur.execute("SELECT c.customer_id, p.churn_probability, p.top_features->0->>'feature' "
                "FROM churn_predictions p JOIN customers c USING (customer_id) "
                "WHERE jsonb_array_length(p.top_features) > 0 ORDER BY p.churn_probability DESC LIMIT 3")
    for r in cur.fetchall():
        log(f"[verify] sample join: {r[0]}  p={r[1]}  top driver={r[2]}")
    conn.rollback()      # read-only; end the transaction cleanly
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Load Kairos data into PostgreSQL / Supabase.")
    ap.add_argument("--dry-run", action="store_true", help="validate + show the plan; never connects")
    ap.add_argument("--reset", action="store_true", help="wipe the 8 tables before loading")
    ap.add_argument("--yes", action="store_true", help="skip the RESET confirmation prompt")
    ap.add_argument("--verify-only", action="store_true", help="run the health checks and exit")
    ap.add_argument("--no-champion-predictions", action="store_true",
                    help="load only the XGBoost predictions (skip the LSTM champion's)")
    ap.add_argument("--syn-dir", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--migrations-dir", default=str(ROOT / "db" / "migrations"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--schema", default=str(ROOT / "db" / "schema.sql"))
    args = ap.parse_args(argv)

    schema_sql = Path(args.schema).read_text()
    schema = parse_schema(schema_sql)

    if args.verify_only:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        conn = connect(normalize_url(os.getenv("DATABASE_URL", "")))
        sys.exit(0 if verify(conn) else 1)

    frames, warnings = load_frames(Path(args.syn_dir), Path(args.models_dir),
                                   include_champion=not args.no_champion_predictions,
                                   processed_dir=Path(args.processed_dir))
    for w in warnings:
        print(f"[warn] {w}")
    errors = preflight(frames, schema)
    if errors:
        print("[preflight] FAILED - nothing was sent to the database:")
        for e in errors:
            print("   -", e)
        sys.exit(2)
    print("[preflight] OK: " + ", ".join(f"{t}={len(frames[t])}" for t in TABLES))

    if args.dry_run:
        print("[dry-run] no database contact. A real run would load the tables above in one transaction.")
        return

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    url = normalize_url(os.getenv("DATABASE_URL", ""))
    print(f"[connect] {masked(url)}")
    conn = connect(url)
    try:
        migrations = {t: (Path(args.migrations_dir) / fname).read_text() for t, fname in MIGRATION_FILES.items()
                      if (Path(args.migrations_dir) / fname).exists()}
        run_load(conn, frames, schema_sql, reset=args.reset, assume_yes=args.yes, migrations=migrations)
        good = verify(conn, expected_checks(frames))
    finally:
        conn.close()
    print("[done] load verified" if good else "[done] load finished but a verification check failed - read above")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
