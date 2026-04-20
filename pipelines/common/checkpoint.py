import json
import os


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def load_checkpoint(path: str, default: dict) -> dict:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: str, checkpoint: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def get_last_processed_info(path: str = "output/checkpoint/last_seq.json") -> tuple:
    """마지막으로 처리한 seq 번호와 파티션 날짜를 불러옴"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 초기값이 없으면 넉넉하게 20260101로 설정 (풀스캔 방지)
            return data.get("last_seq", 0), data.get("last_part_basc_dt", "20260101")
    return 0, "20260101"


def save_last_processed_info(seq: int, part_basc_dt: str, path: str = "output/checkpoint/last_seq.json"):
    """마지막으로 처리한 seq 번호와 파티션 날짜를 저장"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"last_seq": seq, "last_part_basc_dt": part_basc_dt}, f)

