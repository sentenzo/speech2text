from .realtime import RealtimeProcessing
from .strategy import IStrategy

DEFAULT_STRATEGY: IStrategy = RealtimeProcessing()

__all__ = [
    "IStrategy",
    "RealtimeProcessing",
    "DEFAULT_STRATEGY",
]
