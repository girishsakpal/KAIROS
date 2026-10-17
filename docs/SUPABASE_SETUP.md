# Kairos — Loading the data into Supabase (Postgres)

Time needed: about 15 minutes. Dashboard button names can shift slightly between Supabase releases;
the ideas below stay the same.

## 1. Create the project

1. Sign in at supabase.com and create a new project (the Free plan is enough).
2. Pick the region closest to you.
3. Set the **database password** and **save it somewhere safe**. Use letters and digits only.
   Characters such as `@ : / # ? %` have to be URL-encoded inside a connection string, and a wrong
   encoding gives a confusing "password authentication failed".

## 2. Get the connection string

In the project, click **Connect** and choose the **Session pooler** string. Use this one, not "Direct
connection": the direct host is IPv6-only unless you buy an add-on, and many home networks are
IPv4-only. The session pooler works over IPv4. Do not use "Transaction pooler" (port 6543) for the loader.

It looks like this:

```
postgresql://postgres.<PROJECT-REF>:[YOUR-PASSWORD]@aws-0-<region>.pooler.supabase.com:5432/postgres
```

Note the username is `postgres.<PROJECT-REF>` (with the dot), exactly as the dashboard shows it.

## 3. Put it in `.env`

Open the `.env` file in the repo root (create it from `.env.example` if you don't have one) and set:

```
DATABASE_URL=postgresql://postgres.<PROJECT-REF>:YourRealPassword@aws-0-<region>.pooler.supabase.com:5432/postgres
```

Replace `[YOUR-PASSWORD]` with the real password, with no square brackets.

> **`.env` holds your database password.** It is already in `.gitignore`. Do **not** put it in a zip you
> share, and do not paste it into a chat. If it ever leaks, reset the password in Supabase
> (Project Settings -> Database) and update `.env`.

## 4. (Optional hardening) Turn off the Data API

Supabase can expose tables in the `public` schema through an auto-generated REST API. Kairos does not
use it: the Flask app connects straight to Postgres. In **Project Settings -> Data API** you can switch
it off. The loader also enables Row Level Security on all 8 tables, so even with the Data API on, its
anonymous and authenticated roles see nothing. Doing both is belt and braces.

## 5. Load

From the repo root:

```powershell
pip install -r requirements.txt

python -m src.db.load_to_postgres --dry-run     # validates all data, never connects. Run this first.
python -m src.db.load_to_postgres               # loads into an EMPTY database
```

A successful run prints roughly this (counts depend on your files):

```
[preflight] OK: customers=7043, subscriptions=7043, usage_logs=42258, support_tickets=7723, transactions=44800, feature_snapshots=7043, model_registry=2, churn_predictions=2838
[extension] pgvector installed (extensions)
[schema] no tables found - creating the 8 tables from db/schema.sql
[load] customers ... [load] churn_predictions
[sequences] reset for support_tickets, transactions
[security] Row Level Security enabled on all 8 tables
[commit] load committed
[verify] ... counts match, sums match, RLS enabled on all 8 tables
[done] load verified
```

The load is a single transaction. If anything fails, nothing is written (`[rollback] nothing was changed`).

To rebuild from scratch at any time (this DELETES all rows in the 8 tables):

```powershell
python -m src.db.load_to_postgres --reset
```

Run only the health checks on an existing database:

```powershell
python -m src.db.load_to_postgres --verify-only
```

## 6. Look at it

In the Supabase dashboard open **Table Editor**. You should see the 8 tables, each marked as having
Row Level Security enabled. Or open **SQL Editor** and try:

```sql
SELECT p.customer_id, p.churn_probability, p.risk_tier,
       p.top_features->0->>'feature' AS top_driver
FROM churn_predictions p
WHERE p.model_id = 'xgb_churn_v2'
ORDER BY p.churn_probability DESC
LIMIT 10;
```

## 7. Things to know

- **Two models' predictions are loaded.** `xgb_churn_v2` has SHAP drivers; the LSTM champion
  (`lstm_churn_v1`) has none, and `top_features` is `NOT NULL`, so its rows store `[]`. Any query that
  needs drivers must filter by `model_id` or by `jsonb_array_length(top_features) > 0`. Use
  `--no-champion-predictions` to skip the LSTM rows.
- **Only the 1,419 held-out test customers have predictions.** The other 5,624 customers are in
  `customers` and the behavioural tables but were never scored.
- **Free plan limits.** 500 MB database size, and projects are paused after 7 days of inactivity
  (check current limits at supabase.com/pricing). A paused project can be restored from the dashboard,
  but **un-pause it before your viva, and expect it to pause during the January gap.** Because the loader
  is repeatable, you can always rebuild the database from the repo in a couple of minutes.
- **The database is a copy.** The CSV/JSON files remain the source of truth for now. Re-run with
  `--reset` after you retrain a model.

## 8. Troubleshooting

| Message | Cause and fix |
|---|---|
| `DATABASE_URL is missing or still contains the [YOUR-PASSWORD] placeholder` | Edit `.env`; remove the brackets and put in the real password. |
| `could not translate host name` / `Network is unreachable` | You used the Direct connection. Use the Session pooler string. |
| `password authentication failed` | Wrong password, or special characters not URL-encoded. Reset the password to letters and digits. |
| `Tenant or user not found` | Pooler username must be `postgres.<PROJECT-REF>`, copied from the dashboard. |
| `Database already contains data` | Safety stop. Use `--reset` to wipe and reload, or `--verify-only`. |
| `Only some tables exist` | A previous manual attempt left a partial schema. Drop those tables in the SQL Editor, then re-run. |
| `Could not enable pgvector` | Dashboard -> Database -> Extensions -> enable `vector`, then re-run. |
| `feature_snapshots is empty` | `data/processed/features_train.csv` / `features_test.csv` are missing. Re-run `python src/features/build_features.py`. |
| `[preflight] FAILED` | A data file doesn't fit the schema. The message names the table and column. Nothing was sent to the database. |

## 9. Upgrading an existing 7-table database (adding `feature_snapshots`)

Table 8, `feature_snapshots` (customer_id, snapshot_date, split), stores the as-of date that
`build_features.py` used for each customer. The API needs it to show the same numbers the model saw.
If you loaded the original 7 tables earlier, do this once:

```powershell
python -m src.db.load_to_postgres --dry-run     # now also reads data/processed/features_*.csv
python -m src.db.load_to_postgres --reset       # type RESET when asked
```

The loader notices the missing table and creates it from `db/migrations/001_feature_snapshots.sql`
(inside the same transaction), then wipes and reloads everything. This is safe because the database is a
rebuildable copy of the repo files. If anything fails, nothing is changed, including the new table. You can
also paste the migration into the Supabase SQL Editor yourself; it is safe to run twice.

Then check that the Postgres-backed API gives the same answers as the CSV-backed one:

```powershell
python -m src.api.check_parity --limit 200      # quick check
python -m src.api.check_parity                  # every scored customer; takes a few minutes
```

To run the API on the database instead of the files, set `KAIROS_REPOSITORY=postgres` in `.env`
(the default is `csv`) and start it as usual with `python -m src.api.app`.

