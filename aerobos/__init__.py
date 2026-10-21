"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging
"""

from .bos_core import BOSCore
from .plume_metrics import PlumeAnalyzer
from .synthetic_sample import generate_sample_video

__all__ = ["BOSCore", "PlumeAnalyzer", "generate_sample_video"]
