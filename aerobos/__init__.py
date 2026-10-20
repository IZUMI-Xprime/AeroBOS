"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging & Autonomous Valve Isolation
Powered by OpenCV 5, Cloud-Optimized OpenCV (COOL) on AWS Graviton, and Amazon Bedrock.
"""

from .refractometer import SchlierenRefractometer, RefractometerConfig
from .plume_analysis import PlumeAnalyzer, PlumeMetrics
from .dispersion_model import GaussianDispersionModel, LeakSeverity
from .digital_twin_streamer import DigitalTwinStreamer, PlumeTelemetryRecord
from .orchestrator import AeroBOSAgentOrchestrator, AeroBOSMCPServer

try:
    from ai_orchestrator import (
        SageMakerPlumeClassifier,
        BedrockMultimodalDiagnostician,
        ConvLSTMDriftForecaster,
        predict_plume_drift,
        AeroBOSAIOrchestrator,
        GasClassificationLabel,
    )
except ImportError:
    pass

__version__ = "2.7.0"
__all__ = [
    "SchlierenRefractometer",
    "RefractometerConfig",
    "PlumeAnalyzer",
    "PlumeMetrics",
    "GaussianDispersionModel",
    "LeakSeverity",
    "DigitalTwinStreamer",
    "PlumeTelemetryRecord",
    "AeroBOSAgentOrchestrator",
    "AeroBOSMCPServer",
    "SageMakerPlumeClassifier",
    "BedrockMultimodalDiagnostician",
    "ConvLSTMDriftForecaster",
    "predict_plume_drift",
    "AeroBOSAIOrchestrator",
    "GasClassificationLabel",
]
