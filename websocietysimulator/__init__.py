from .simulator import Simulator
import logging

logger = logging.getLogger("websocietysimulator")
logger.setLevel(logging.INFO)  # 默认级别

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

__all__ = ["Simulator"]