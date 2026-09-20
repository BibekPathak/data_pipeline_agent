"""Reproducible fixture generator for the demo pipelines.

Run: python datasets/fixtures/_generate.py

Writes healthy baseline sources:
  - orders.csv      : 600 orders, numeric amount, valid customer refs
  - customers.csv   : 200 customers

The drifted/scenario variants are generated at runtime by
``app/evaluation/scenarios.py`` (Phase 7) from these fixtures.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
STATUS = ["completed", "pending", "cancelled", "shipped"]


def gen_customers(rng: random.Random, n: int = 200) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": list(range(1, n + 1)),
            "tier": [rng.choice(["standard", "gold", "platinum"]) for _ in range(n)],
            "country": [rng.choice(["US", "DE", "FR", "UK", "JP"]) for _ in range(n)],
        }
    )


def gen_orders(rng: random.Random, n: int = 600) -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "order_id": list(range(1, n + 1)),
            "customer_id": [rng.randint(1, 200) for _ in range(n)],
            "amount": [round(rng.uniform(5, 500), 2) for _ in range(n)],
            "currency": [rng.choice(CURRENCIES) for _ in range(n)],
            "created_at": [
                (start + timedelta(seconds=rng.randint(0, 120 * 86400))).isoformat()
                for _ in range(n)
            ],
            "status": [rng.choice(STATUS) for _ in range(n)],
        }
    )


def main() -> None:
    rng = random.Random(42)
    out = Path(__file__).parent
    customers = gen_customers(rng)
    orders = gen_orders(rng)
    customers.write_csv(out / "customers.csv")
    orders.write_csv(out / "orders.csv")
    print(f"wrote {out / 'customers.csv'} ({customers.height} rows)")
    print(f"wrote {out / 'orders.csv'} ({orders.height} rows)")


if __name__ == "__main__":
    main()
