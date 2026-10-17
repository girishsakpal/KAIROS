-- Migration 001: add feature_snapshots (ER diagram table #8).
-- Safe to run more than once. The loader (src/db/load_to_postgres.py) applies this
-- automatically to an existing 7-table database; you can also paste it into the
-- Supabase SQL Editor.
CREATE TABLE IF NOT EXISTS feature_snapshots (
    customer_id         VARCHAR(20)   PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    snapshot_date       DATE          NOT NULL,
    split               VARCHAR(10)   NOT NULL CHECK (split IN ('train', 'test')),
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_split ON feature_snapshots(split);

-- Same protection as the other tables: no Data API access without an explicit policy.
ALTER TABLE feature_snapshots ENABLE ROW LEVEL SECURITY;
