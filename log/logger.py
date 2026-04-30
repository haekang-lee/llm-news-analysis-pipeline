import logging
import os
from datetime import datetime, timedelta

LOG_DIR = os.path.join(os.path.dirname(__file__), "files")
RETENTION_DAYS = 14


def _cleanup_old_logs(log_dir: str, retention_days: int = RETENTION_DAYS) -> None:
    cutoff = datetime.now() - timedelta(days=retention_days)
    for filename in os.listdir(log_dir):
        if not filename.startswith("daily_batch_") or not filename.endswith(".log"):
            continue
        filepath = os.path.join(log_dir, filename)
        try:
            date_str = filename.replace("daily_batch_", "").replace(".log", "")
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date < cutoff:
                os.remove(filepath)
        except (ValueError, OSError):
            continue


def cleanup_old_output_dirs(output_dir: str, retention_days: int = RETENTION_DAYS) -> None:
    """output/daily_batch/ 하위의 YYYYMMDD 디렉터리 중 retention_days 초과된 것을 삭제."""
    import shutil

    if not os.path.exists(output_dir):
        return

    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = []
    for dirname in os.listdir(output_dir):
        dirpath = os.path.join(output_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        try:
            dir_date = datetime.strptime(dirname, "%Y%m%d")
            if dir_date < cutoff:
                shutil.rmtree(dirpath)
                removed.append(dirname)
        except (ValueError, OSError):
            continue

    if removed:
        logging.getLogger("news_data").info(
            "오래된 output 디렉터리 %d개 삭제: %s", len(removed), ", ".join(sorted(removed))
        )


def setup_logging(log_dir: str = LOG_DIR, today_str: str = None) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    if today_str is None:
        today_str = datetime.now().strftime("%Y%m%d")

    _cleanup_old_logs(log_dir)

    log_file = os.path.join(log_dir, f"daily_batch_{today_str}.log")

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 중복 핸들러 방지
    if not root_logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root_logger.addHandler(sh)

    return logging.getLogger("news_data")
