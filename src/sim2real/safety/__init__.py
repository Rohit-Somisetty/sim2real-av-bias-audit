"""Safety event detection and rate comparison."""

from .events import summarize_events
from .rate_compare import compute_event_rates, compare_event_rates

__all__ = ["summarize_events", "compute_event_rates", "compare_event_rates"]
