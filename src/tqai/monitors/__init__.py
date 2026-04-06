"""Runtime observation modules."""

from tqai.monitors.lyapunov import LyapunovMonitor
from tqai.monitors.stability import StabilityMonitor
from tqai.pipeline.registry import register_monitor

register_monitor("stability", StabilityMonitor)
register_monitor("lyapunov", LyapunovMonitor)
