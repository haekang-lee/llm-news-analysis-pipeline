import os


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def path_from_cfg(cfg, name: str) -> str:
    """cfg.paths.{name} — 상대 경로면 프로젝트 루트 기준으로 변환."""
    value = getattr(cfg.paths, name)
    if os.path.isabs(str(value)):
        return str(value)
    return os.path.join(project_root(), str(value))
