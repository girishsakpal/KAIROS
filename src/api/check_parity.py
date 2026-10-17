"""
Parity check: does the Postgres-backed API return the same answers as the CSV-backed API?

    python -m src.api.check_parity                 # every scored customer + 100 unscored + 1 unknown id
    python -m src.api.check_parity --limit 200     # quicker: first 200 scored customers
    python -m src.api.check_parity --unscored 0

Needs DATABASE_URL (the Postgres side) and the CSV/JSON files (the reference side). It calls POST /predict
and GET /customer-detail for each id plus /health and /model-metrics, through Flask's in-process test
client (no server needed), and reports every field that differs. Exit code 0 = identical.

Tolerances: floats may differ by 1.5e-4 (the database stores probabilities to 5 decimals; the API rounds
to 4, so a double rounding can move the last digit). ISO timestamps are compared as instants.
"""
import argparse
import sys
import time
from datetime import datetime, timezone

FLOAT_TOL = 1.5e-4


def _as_dt(v):
    if isinstance(v, str) and "T" in v and len(v) >= 19:
        try:
            d = datetime.fromisoformat(v)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def diff(a, b, path="$", out=None):
    """Return a list of (path, a, b) for every difference."""
    out = [] if out is None else out
    if isinstance(a, bool) != isinstance(b, bool):          # True must not silently equal 1
        out.append((path, a, b))
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}[len]", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        if abs(a - b) > FLOAT_TOL:
            out.append((path, a, b))
    elif isinstance(a, str) and isinstance(b, str):
        da, db = _as_dt(a), _as_dt(b)
        if (da is not None and db is not None and da != db) or ((da is None or db is None) and a != b):
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))
    return out


def _call(client, method, url, body=None):
    r = client.post(url, json=body) if method == "POST" else client.get(url)
    return r.status_code, r.get_json()


def compare_apps(ref_client, new_client, ids, log=print, every=100):
    """Compare two Flask test clients. Returns a list of (endpoint, id, path, ref_value, new_value)."""
    problems, started = [], time.time()

    def check(label, ident, method, url, body=None):
        s1, j1 = _call(ref_client, method, url, body)
        s2, j2 = _call(new_client, method, url, body)
        if s1 != s2:
            problems.append((label, ident, "status", s1, s2))
        else:
            problems.extend((label, ident, p, x, y) for p, x, y in diff(j1, j2))

    check("health", "-", "GET", "/api/v1/health")
    check("model-metrics", "-", "GET", "/api/v1/model-metrics?include_history=true")
    for n, cid in enumerate(ids, 1):
        check("predict", cid, "POST", "/api/v1/predict", {"customer_id": cid})
        check("customer-detail", cid, "GET", f"/api/v1/customer-detail/{cid}")
        if n % every == 0:
            log(f"  ... {n}/{len(ids)} customers compared ({time.time() - started:.0f}s, {len(problems)} differences so far)")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare the CSV-backed and Postgres-backed APIs.")
    ap.add_argument("--limit", type=int, default=None, help="only the first N scored customers")
    ap.add_argument("--unscored", type=int, default=100, help="how many never-scored customers to include")
    args = ap.parse_args(argv)

    from .app import create_app
    now = datetime.now(timezone.utc)                    # identical clock for both sides
    ref_app = create_app({"REPOSITORY": "csv", "NOW": now})
    new_app = create_app({"REPOSITORY": "postgres", "NOW": now})
    ref_repo = ref_app.extensions["kairos_repo"]

    scored = sorted(ref_repo._pred.index) if ref_repo._pred is not None else []
    if args.limit:
        scored = scored[:args.limit]
    unscored = sorted(set(ref_repo.customers.index) - set(ref_repo._pred.index))[:args.unscored] \
        if ref_repo._pred is not None else []
    ids = scored + unscored + ["NOPE-0000"]
    print(f"[parity] comparing {len(scored)} scored + {len(unscored)} unscored + 1 unknown id "
          f"({len(ids) * 2 + 2} requests per side)")

    problems = compare_apps(ref_app.test_client(), new_app.test_client(), ids)
    if not problems:
        print(f"[parity] IDENTICAL across {len(ids)} customers (floats within {FLOAT_TOL}).")
        return 0
    print(f"[parity] {len(problems)} DIFFERENCES. First 15:")
    for ep, ident, path, ref, new in problems[:15]:
        print(f"   {ep:<16} {ident:<12} {path}: csv={ref!r}  postgres={new!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
