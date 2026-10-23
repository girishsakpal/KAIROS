"""
Retention action catalogue.

Every number here is an ASSUMPTION, not something measured from this project's data (the data has no
record of past interventions, so the effect of an action cannot be learned yet). They are the knobs of
the engine: change them in this file, or scale all save rates at request time with `save_rate_scale`.

  base_save_rate : assumed share of at-risk customers who stay if this action is taken well.
  cost           : cost to run the action for one customer, in currency units (cu). A number, or a
                   function of the customer's context (discounts scale with the monthly charge).
  owner_team     : who would execute it (used for team-capacity limits in campaign planning).
  drivers        : model features (encoder prefixes already stripped) that make this action relevant.
                   A name ending in "_" matches every feature starting with it (e.g. "payment_method_").
  guard / fact   : guard must hold for the action to make sense at all for this customer (e.g. you cannot
                   offer an annual contract to someone already on one); fact is a customer fact that
                   justifies the action even when the model did not rank the matching driver.
"""
from dataclasses import dataclass
from typing import Callable, Tuple, Union

Cost = Union[float, Callable[[dict], float]]


@dataclass(frozen=True)
class Action:
    id: str
    name: str
    owner_team: str
    channel: str
    description: str
    base_save_rate: float
    cost: Cost
    drivers: Tuple[str, ...]
    guard: Callable[[dict], bool]
    fact: Callable[[dict], bool]
    fact_text: Callable[[dict], str]

    def cost_for(self, ctx: dict) -> float:
        return float(self.cost(ctx) if callable(self.cost) else self.cost)


def _auto_pay(ctx):
    return "automatic" in (ctx.get("payment_method") or "").lower()


CATALOG = [
    Action(
        "payment_recovery", "Payment recovery call + switch to auto-pay", "Collections", "Phone + payment link",
        "Resolve failed payments and move the customer to automatic payment so billing stops being a reason to leave.",
        0.35, 4.0,
        ("failed_payment_count_90d", "failed_payment_ratio_90d", "payment_method_", "total_paid_90d",
         "transaction_count_90d"),
        guard=lambda c: (not _auto_pay(c)) or c.get("failed_payments_90d", 0) >= 1,
        fact=lambda c: c.get("failed_payments_90d", 0) >= 1,
        fact_text=lambda c: f"{c.get('failed_payments_90d', 0)} failed payment(s) in the last 90 days",
    ),
    Action(
        "annual_contract_offer", "Offer a discounted annual contract", "Sales", "Phone or in-app offer",
        "Move a month-to-month customer onto a longer contract with an incentive (about half a month's fee).",
        0.30, lambda c: 0.5 * c.get("monthly_charge", 0.0),
        ("contract_type_", "plan_type_"),
        guard=lambda c: c.get("contract_type") == "Month-to-month",
        fact=lambda c: c.get("contract_type") == "Month-to-month",
        fact_text=lambda c: "on a month-to-month contract, so free to leave at any time",
    ),
    Action(
        "onboarding_call", "Early-life onboarding call", "Customer Success", "Phone",
        "A guided call in the first months to confirm the customer is set up and getting value.",
        0.25, 6.0,
        ("tenure_months", "months_of_usage_history"),
        guard=lambda c: c.get("tenure_months", 999) <= 12,
        fact=lambda c: c.get("tenure_months", 999) <= 6,
        fact_text=lambda c: f"only {c.get('tenure_months')} month(s) as a customer",
    ),
    Action(
        "feature_walkthrough", "Proactive feature walkthrough", "Customer Success", "Video call or in-app guide",
        "Show the customer the features they are not using yet, to raise perceived value.",
        0.20, 5.0,
        ("avg_feature_adoption_recent", "avg_session_minutes_recent", "avg_session_count_recent"),
        guard=lambda c: True,
        fact=lambda c: False,
        fact_text=lambda c: "",
    ),
    Action(
        "reengagement_nudge", "Re-engagement check-in", "Customer Success", "Email + short call",
        "Reach out when usage is falling or the customer has gone quiet, before they decide to leave.",
        0.15, 2.0,
        ("session_count_trend", "days_since_last_active"),
        guard=lambda c: True,
        fact=lambda c: c.get("usage_trend") == "declining",
        fact_text=lambda c: "usage has been declining",
    ),
    Action(
        "support_escalation", "Escalate to a retention specialist", "Support", "Senior agent call",
        "Fix open or slow-to-resolve support problems with a senior owner and a follow-up.",
        0.30, 15.0,
        ("ticket_count_90d", "open_ticket_count", "avg_resolution_hours_90d", "billing_ticket_ratio_90d"),
        guard=lambda c: True,
        fact=lambda c: c.get("open_tickets", 0) >= 1,
        fact_text=lambda c: f"{c.get('open_tickets', 0)} support ticket(s) still open",
    ),
    Action(
        "price_fit_review", "Price-fit review / bundle offer", "Sales", "Phone",
        "Review whether the plan fits the customer's usage and offer a better-value bundle or a short discount.",
        0.20, lambda c: 0.9 * c.get("monthly_charge", 0.0),
        ("monthly_charge",),
        guard=lambda c: c.get("monthly_charge", 0.0) > 0,
        fact=lambda c: c.get("monthly_charge", 0.0) >= 80,
        fact_text=lambda c: f"pays a high monthly charge ({c.get('monthly_charge')})",
    ),
    Action(
        "general_checkin", "General retention check-in", "Customer Success", "Phone",
        "A friendly check-in call when no specific problem stands out.",
        0.10, 3.0,
        (),
        guard=lambda c: True,
        fact=lambda c: True,
        fact_text=lambda c: "no single cause stands out",
    ),
]
BY_ID = {a.id: a for a in CATALOG}

# Plain-English names for the model's feature names (used in the "why" text).
FEATURE_LABELS = {
    "tenure_months": "short tenure", "months_of_usage_history": "short usage history",
    "monthly_charge": "monthly charge", "avg_feature_adoption_recent": "low feature adoption",
    "avg_session_minutes_recent": "short sessions", "avg_session_count_recent": "few sessions",
    "session_count_trend": "falling session trend", "days_since_last_active": "days since last active",
    "failed_payment_count_90d": "failed payments (90 days)", "failed_payment_ratio_90d": "failed-payment ratio",
    "total_paid_90d": "amount paid (90 days)", "transaction_count_90d": "transaction count (90 days)",
    "ticket_count_90d": "support tickets (90 days)", "open_ticket_count": "open support tickets",
    "avg_resolution_hours_90d": "slow ticket resolution", "billing_ticket_ratio_90d": "share of billing tickets",
    "contract_type_Month-to-month": "month-to-month contract", "payment_method_Electronic check": "pays by electronic check",
    "payment_method_Mailed check": "pays by mailed check", "plan_type_Basic": "basic plan", "age": "customer age",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


# Neutral names (no direction implied) for charts that show both risk-raising and risk-lowering drivers.
NEUTRAL = {
    "tenure_months": "Tenure", "months_of_usage_history": "Usage history", "monthly_charge": "Monthly charge",
    "avg_feature_adoption_recent": "Feature adoption", "avg_session_minutes_recent": "Session minutes",
    "avg_session_count_recent": "Session count", "session_count_trend": "Session-count trend",
    "days_since_last_active": "Days since last active", "failed_payment_count_90d": "Failed payments (90 d)",
    "failed_payment_ratio_90d": "Failed-payment ratio", "total_paid_90d": "Amount paid (90 d)",
    "transaction_count_90d": "Transactions (90 d)", "ticket_count_90d": "Support tickets (90 d)",
    "open_ticket_count": "Open support tickets", "avg_resolution_hours_90d": "Ticket resolution time",
    "billing_ticket_ratio_90d": "Billing-ticket share", "contract_type_Month-to-month": "Month-to-month contract",
    "contract_type_Two year": "Two-year contract", "contract_type_One year": "One-year contract",
    "payment_method_Electronic check": "Pays by electronic check", "payment_method_Mailed check": "Pays by mailed check",
    "plan_type_Basic": "Basic plan", "age": "Customer age",
}


def neutral(feature: str) -> str:
    return NEUTRAL.get(feature, feature.replace("_", " ").capitalize())
