"""Runtime observation modules."""

from tqai.pipeline.registry import register_monitor
from tqai.monitors.stability import StabilityMonitor
from tqai.monitors.lyapunov import LyapunovMonitor

register_monitor("stability", StabilityMonitor)
register_monitor("lyapunov", LyapunovMonitor)
