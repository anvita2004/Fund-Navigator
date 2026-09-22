from pydantic import BaseModel
from typing import Optional, List


class FundAnswer(BaseModel):
    """Structure for a single-fund lookup answer."""
    fund_name: str
    answer_summary: str
    risk_level: Optional[str] = None
    expense_ratio: Optional[float] = None
    sources: List[str]


class ComparisonMetric(BaseModel):
    metric_name: str
    values: dict


class ComparisonResult(BaseModel):
    """Structure for a multi-fund comparison answer."""
    funds_compared: List[str]
    comparison: List[ComparisonMetric]
    conclusion: str
    sources: List[str]