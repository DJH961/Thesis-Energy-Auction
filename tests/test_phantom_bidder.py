import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.phantom_bidder import PhantomBidder


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_phantom_anchor_independent_of_ma3():
    """Phantom price draw distribution should not depend on MA3 anchor input."""
    config = _load_config()
    ph_cfg = config.setdefault("phantom_bidder", {})
    ph_cfg["enabled"] = True

    reserve = 45.0
    penalty = 160.0
    supply = 100.0
    n_draws = 400

    rng_floor = np.random.default_rng(2026)
    bidder_floor = PhantomBidder(config, rng_floor)
    prices_floor = np.array([
        bidder_floor.sample_bid(
            price_ma3=45.0,
            reserve_price=reserve,
            penalty_rate=penalty,
            auction_supply=supply,
        )[0]
        for _ in range(n_draws)
    ])

    rng_high = np.random.default_rng(2026)
    bidder_high = PhantomBidder(config, rng_high)
    prices_high = np.array([
        bidder_high.sample_bid(
            price_ma3=200.0,
            reserve_price=reserve,
            penalty_rate=penalty,
            auction_supply=supply,
        )[0]
        for _ in range(n_draws)
    ])

    np.testing.assert_allclose(
        prices_floor,
        prices_high,
        atol=1e-10,
        rtol=1e-10,
        err_msg="Phantom bid prices changed with MA3; anchor should be MA3-independent.",
    )
