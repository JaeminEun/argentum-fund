from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import load_config
from .loaders import load_universe
from .schema import standardize_universe_frame, validate_universe_frame


def get_enabled_universes(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return enabled universe configurations.
    """
    universes = config.get("universes", [])

    return [
        universe
        for universe in universes
        if universe.get("enabled", False)
    ]


def build_universe(config_path: str | Path) -> pd.DataFrame:
    """
    Build a standardized investment universe from configured sources.
    """
    config = load_config(config_path)
    enabled_universes = get_enabled_universes(config)

    if not enabled_universes:
        raise ValueError("No enabled universes found in configuration.")

    frames = []

    for universe_config in enabled_universes:
        frame = load_universe(universe_config)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = standardize_universe_frame(combined)

    # Keep only active rows in final universe.
    combined = combined[combined["active"]].copy()

    validate_universe_frame(combined)

    output_path = Path(config["project"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_path, index=False)

    return combined


if __name__ == "__main__":
    universe = build_universe("config/universe_config.yaml")
    print(f"Built universe with {len(universe)} active securities.")
    print(universe[["ticker", "universe_name", "account_target", "strategy_role"]])
