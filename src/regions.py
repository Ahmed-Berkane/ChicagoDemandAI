import pandas as pd

from src.config import REGION_MAP_PARQUET

DOWNTOWN = {32, 8, 28, 33, 34, 31, 60, 36, 37, 38, 39, 40}

NORTH_SIDE = {
    1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 77,
}

WEST_SIDE = {
    24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58,
    59, 61, 62, 63, 64, 65,
}

SOUTH_SIDE = {
    41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52,
    53, 54, 55, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
}

FAR_SOUTH = {47, 76, 0}

REGION_MAP = {
    **{area: "DOWNTOWN" for area in DOWNTOWN},
    **{area: "NORTH_SIDE" for area in NORTH_SIDE},
    **{area: "WEST_SIDE" for area in WEST_SIDE},
    **{area: "SOUTH_SIDE" for area in SOUTH_SIDE},
    **{area: "FAR_SOUTH" for area in FAR_SOUTH},
}

REGION_OPTIONS = sorted(set(REGION_MAP.values()) | {"OTHER"})


def save_region_map() -> dict[int, str]:
    df = pd.DataFrame(REGION_MAP.items(), columns=["community_area", "region"])
    df.to_parquet(REGION_MAP_PARQUET, index=False)
    return REGION_MAP
