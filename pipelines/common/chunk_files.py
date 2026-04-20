import os
import re
from typing import List, Optional, Tuple


def parse_chunk_idx(filename: str) -> Optional[int]:
    match = re.match(r"chunk_(\d+)\.parquet$", filename)
    return int(match.group(1)) if match else None


def list_new_chunk_files(directory: str, last_done_idx: int) -> List[Tuple[int, str]]:
    if not os.path.exists(directory):
        return []

    pairs = []
    for filename in os.listdir(directory):
        if not filename.endswith(".parquet"):
            continue
        idx = parse_chunk_idx(filename)
        if idx is not None and idx > last_done_idx:
            pairs.append((idx, filename))
    pairs.sort(key=lambda x: x[0])
    return pairs

