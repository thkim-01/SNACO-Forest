import json
from pathlib import Path
from typing import Dict

ALGORITHM_PROFILE_DIR = Path("configs") / "algorithms"

ALGORITHM_PROFILES = {
    "id3": "id3.json",
    "c45": "c45.json",
    "cart": "cart.json",
}

_ALIASES = {
    "id3": "id3",
    "information_gain": "id3",
    "c45": "c45",
    "c4.5": "c45",
    "gain_ratio": "c45",
    "cart": "cart",
    "gini": "cart",
}


def normalize_algorithm_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"Unsupported algorithm '{name}'. "
        f"Choose one of: {', '.join(sorted(ALGORITHM_PROFILES.keys()))}"
    )


def load_algorithm_profile(name: str) -> Dict:
    norm = normalize_algorithm_name(name)
    profile_file = ALGORITHM_PROFILE_DIR / ALGORITHM_PROFILES[norm]

    if not profile_file.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_file}")

    with profile_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data["algorithm_key"] = norm
    return data
