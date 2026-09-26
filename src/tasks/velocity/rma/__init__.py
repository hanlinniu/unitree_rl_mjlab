"""RMA (Rapid Motor Adaptation) training stack.

Supports plane / no-scandot (``num_scan=0``) and rough-with-scandot RMA
(``num_scan>0`` + scan encoder), following extreme-parkour main.
"""

from src.tasks.velocity.rma.runner import RMAOnPolicyRunner

__all__ = ["RMAOnPolicyRunner"]
