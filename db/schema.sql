-- =====================================================================
-- KAIROS — PostgreSQL Schema (Phase 3)
-- Matches Kairos_ER_Diagram.pdf / .png (7 tables).
-- Color-coding from the ER diagram is preserved as comments below:
--   teal  = core entity        (customers)
--   navy  = behavioral tables  (subscriptions, usage_logs, support_tickets, transactions)
--   purple = ML prediction output (churn_predictions)
--   gray  = model tracking/infra (model_registry)
--
-- Requires: PostgreSQL 14+ and the pgvector extension (for Phase 5 embeddings).
-- Embedding columns are added now (nullable) so no schema migration is needed
-- later — they simply stay NULL until Phase 5 populates them.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- 1. customers  [teal — core entity]
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    customer_id         VARCHAR(20)   PRIMARY KEY,
    signup_date         DATE          NOT NULL,
    region              VARCHAR(50)   NOT NULL,
    signup_channel      VARCHAR(50),
    age                 SMALLINT,
    gender              VARCHAR(20),
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX idx_customers_region ON customers(region);
CREATE INDEX idx_customers_signup_channel ON customers(signup_channel);

-- ---------------------------------------------------------------------
-- 2. subscriptions  [navy — behavioral, one-to-many off customers]
-- ---------------------------------------------------------------------
CREATE TABLE subscriptions (
    subscription_id     SERIAL        PRIMARY KEY,
    customer_id         VARCHAR(20)   NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    plan_type           VARCHAR(50)   NOT NULL,          -- e.g. Basic, Standard, Pro
    contract_type       VARCHAR(20)   NOT NULL,          -- Month-to-month, One year, Two year
    billing_cycle       VARCHAR(20)   NOT NULL,          -- Monthly, Annual
    payment_method      VARCHAR(50)   NOT NULL,
    monthly_charge      NUMERIC(10,2) NOT NULL,
    tenure_months       INTEGER       NOT NULL DEFAULT 0,
    status              VARCHAR(20)   NOT NULL DEFAULT 'active',  -- active, cancelled, paused
    start_date          DATE          NOT NULL,
    end_date            DATE,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX idx_subscriptions_customer_id ON subscriptions(customer_id);
CREATE INDEX idx_subscriptions_status ON subscriptions(status);

-- ---------------------------------------------------------------------
-- 3. usage_logs  [navy — behavioral, one-to-many off customers]
-- Monthly aggregate rows per customer (not raw event-level logs) —
-- keeps volume manageable and matches "session_count_30d"-style features
-- referenced in the SRS / API contracts.
-- ---------------------------------------------------------------------
CREATE TABLE usage_logs (
    usage_id            SERIAL        PRIMARY KEY,
    customer_id         VARCHAR(20)   NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    log_month           DATE          NOT NULL,          -- first-of-month marker for the aggregate period
    session_count       INTEGER       NOT NULL DEFAULT 0,
    avg_session_minutes NUMERIC(6,2)  NOT NULL DEFAULT 0,
    feature_adoption_score NUMERIC(4,3) NOT NULL DEFAULT 0,  -- 0-1, share of features used
    last_active_date    DATE,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now(),
    UNIQUE (customer_id, log_month)
);

CREATE INDEX idx_usage_logs_customer_id ON usage_logs(customer_id);
CREATE INDEX idx_usage_logs_log_month ON usage_logs(log_month);

-- ---------------------------------------------------------------------
-- 4. support_tickets  [navy — behavioral, one-to-many off customers]
-- ticket_text + ticket_embedding support the Phase 5 semantic retrieval
-- layer (pgvector). Embedding dimension (384) matches a typical
-- sentence-transformers MiniLM model — adjust if a different model is
-- locked later.
-- ---------------------------------------------------------------------
CREATE TABLE support_tickets (
    ticket_id           SERIAL        PRIMARY KEY,
    customer_id         VARCHAR(20)   NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now(),
    resolved_at         TIMESTAMPTZ,
    category            VARCHAR(50)   NOT NULL,          -- billing, technical, other
    priority            VARCHAR(20)   NOT NULL DEFAULT 'medium',
    status              VARCHAR(20)   NOT NULL DEFAULT 'open',  -- open, closed
    resolution_time_hours NUMERIC(8,2),
    ticket_text         TEXT,                             -- free-text description
    ticket_embedding    VECTOR(384)                        -- NULL until Phase 5
);

CREATE INDEX idx_support_tickets_customer_id ON support_tickets(customer_id);
CREATE INDEX idx_support_tickets_status ON support_tickets(status);
-- Vector index deferred to Phase 5, after embeddings are populated:
-- CREATE INDEX idx_support_tickets_embedding ON support_tickets
--   USING ivfflat (ticket_embedding vector_cosine_ops) WITH (lists = 100);

-- ---------------------------------------------------------------------
-- 5. transactions  [navy — behavioral, one-to-many off customers]
-- ---------------------------------------------------------------------
CREATE TABLE transactions (
    transaction_id      SERIAL        PRIMARY KEY,
    customer_id         VARCHAR(20)   NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    transaction_date    TIMESTAMPTZ   NOT NULL,
    amount              NUMERIC(10,2) NOT NULL,
    transaction_type    VARCHAR(20)   NOT NULL,          -- payment, refund, failed_payment
    status               VARCHAR(20)   NOT NULL,          -- success, failed
    payment_method       VARCHAR(50)
);

CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);

-- ---------------------------------------------------------------------
-- 6. model_registry  [gray — model tracking / infra]
-- One row per trained model version. churn_predictions.model_id
-- references this table (one-to-many into churn_predictions).
-- ---------------------------------------------------------------------
CREATE TABLE model_registry (
    model_id             VARCHAR(50)   PRIMARY KEY,        -- e.g. 'xgb_churn_v7'
    model_name           VARCHAR(100)  NOT NULL,
    algorithm            VARCHAR(50)   NOT NULL,            -- LogisticRegression, XGBoost, LightGBM
    trained_at           TIMESTAMPTZ   NOT NULL,
    feature_list         JSONB,                             -- list of feature names used
    roc_auc              NUMERIC(5,4),
    pr_auc               NUMERIC(5,4),
    calibration_error    NUMERIC(5,4),
    drift_psi_score      NUMERIC(6,4),
    drift_ks_pvalue      NUMERIC(6,4),
    drift_detected       BOOLEAN       NOT NULL DEFAULT false,
    status                VARCHAR(20)   NOT NULL DEFAULT 'challenger',  -- champion, challenger, retired, active_needs_review
    created_at           TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX idx_model_registry_status ON model_registry(status);

-- ---------------------------------------------------------------------
-- 7. churn_predictions  [purple — ML prediction output]
-- explanation_embedding supports Phase 5's "explanations as retrievable
-- objects" architecture (Section 6.1 of the master doc).
-- ---------------------------------------------------------------------
CREATE TABLE churn_predictions (
    prediction_id        SERIAL        PRIMARY KEY,
    customer_id          VARCHAR(20)   NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_id             VARCHAR(50)   NOT NULL REFERENCES model_registry(model_id),
    predicted_at          TIMESTAMPTZ   NOT NULL DEFAULT now(),
    churn_probability     NUMERIC(6,5)  NOT NULL,
    risk_tier             VARCHAR(20)   NOT NULL,           -- low, medium, high
    top_features          JSONB         NOT NULL,           -- [{feature, shap_value, direction}, ...]
    explanation_text      TEXT,                              -- plain-language rendering of top_features, for embedding
    explanation_embedding VECTOR(384),                       -- NULL until Phase 5
    actual_churn          BOOLEAN                            -- ground-truth label, when known (training/eval only)
);

CREATE INDEX idx_churn_predictions_customer_id ON churn_predictions(customer_id);
CREATE INDEX idx_churn_predictions_model_id ON churn_predictions(model_id);
CREATE INDEX idx_churn_predictions_risk_tier ON churn_predictions(risk_tier);
CREATE INDEX idx_churn_predictions_predicted_at ON churn_predictions(predicted_at);
-- Vector index deferred to Phase 5, same reasoning as support_tickets above.

-- =====================================================================
-- End of schema. Load order for synthetic data: customers -> subscriptions
-- -> usage_logs / support_tickets / transactions -> (model_registry ->
-- churn_predictions, populated by the ML pipeline in Phase 3's later steps,
-- not by the data generator).
-- =====================================================================
