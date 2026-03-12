from enum import Enum


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RunMode(str, Enum):
    SANDBOX = "sandbox"
    LIVE = "live"
    DRY_RUN = "dry_run"


class PipelineStageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class NewsSentimentLabel(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
