"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging
Module: digital_twin_streamer.py

Implements DigitalTwinStreamer:
- Streams real-time optical gas imaging telemetry, plume metrics, and valve actuation states
  to Amazon Timestream for LiveAnalytics.
- Broadcasts high-frequency state updates to AWS IoT Core MQTT topics for low-latency digital twin synchronization.
- Fallback in-memory spooling and simulation mode for local development and edge deployment.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception

logger = logging.getLogger("AeroBOS.DigitalTwinStreamer")


@dataclass
class PlumeTelemetryRecord:
    """Individual real-time telemetry observation record."""
    timestamp_ms: int
    facility_id: str
    camera_id: str
    sector_id: str
    gas_type: str
    valve_id: str
    
    # Quantitative measurements
    plume_area_sqm: float
    peak_refractive_anomaly_sigma: float
    centroid_x_norm: float
    centroid_y_norm: float
    mass_flow_rate_kg_h: float
    optical_path_diff_nm: float
    inference_latency_ms: float
    valve_state: str  # OPEN, IN_TRANSIT, ISOLATED
    is_alarm_active: bool


class DigitalTwinStreamer:
    """
    Ingests BOS perception telemetry and streams it to Amazon Timestream and AWS IoT Core.
    """

    def __init__(
        self,
        database_name: str = "AeroBOS_DigitalTwin",
        table_name: str = "OpticalGasPlumes",
        aws_region: str = "us-east-1",
        enable_aws_push: bool = True,
    ) -> None:
        self.database_name = database_name
        self.table_name = table_name
        self.aws_region = aws_region
        self.enable_aws_push = enable_aws_push and BOTO3_AVAILABLE

        self._timestream_client = None
        self._iot_data_client = None
        self._local_history_buffer: List[Dict[str, Any]] = []

        if self.enable_aws_push:
            try:
                boto_config = Config(
                    region_name=self.aws_region,
                    retries={"max_attempts": 3, "mode": "standard"},
                )
                self._timestream_client = boto3.client("timestream-write", config=boto_config)
                self._iot_data_client = boto3.client("iot-data", config=boto_config)
                logger.info("Initialized AWS Timestream and IoT Core clients in %s", self.aws_region)
            except Exception as e:
                logger.warning("AWS client initialization failed (%s); operating in local spool mode.", e)
                self.enable_aws_push = False

    def stream_record(self, record: PlumeTelemetryRecord) -> bool:
        """
        Publishes a telemetry record to Amazon Timestream and AWS IoT Core.
        """
        # Always maintain local telemetry spool for inspection / Grafana fallback API
        record_dict = asdict(record)
        self._local_history_buffer.append(record_dict)
        if len(self._local_history_buffer) > 1000:
            self._local_history_buffer.pop(0)

        if not self.enable_aws_push or self._timestream_client is None:
            # Simulated push
            return True

        # Construct Amazon Timestream multi-measure record
        dimensions = [
            {"Name": "FacilityId", "Value": str(record.facility_id)},
            {"Name": "CameraId", "Value": str(record.camera_id)},
            {"Name": "SectorId", "Value": str(record.sector_id)},
            {"Name": "GasType", "Value": str(record.gas_type)},
            {"Name": "AssociatedValveId", "Value": str(record.valve_id)},
        ]

        measure_values = [
            {"Name": "PlumeArea_sqm", "Value": f"{record.plume_area_sqm:.4f}", "Type": "DOUBLE"},
            {"Name": "PeakRefractiveAnomaly_sigma", "Value": f"{record.peak_refractive_anomaly_sigma:.2f}", "Type": "DOUBLE"},
            {"Name": "CentroidX_norm", "Value": f"{record.centroid_x_norm:.4f}", "Type": "DOUBLE"},
            {"Name": "CentroidY_norm", "Value": f"{record.centroid_y_norm:.4f}", "Type": "DOUBLE"},
            {"Name": "MassFlowRate_kg_h", "Value": f"{record.mass_flow_rate_kg_h:.4f}", "Type": "DOUBLE"},
            {"Name": "OpticalPathDifference_nm", "Value": f"{record.optical_path_diff_nm:.2f}", "Type": "DOUBLE"},
            {"Name": "InferenceLatency_ms", "Value": f"{record.inference_latency_ms:.2f}", "Type": "DOUBLE"},
            {"Name": "ValveState", "Value": str(record.valve_state), "Type": "VARCHAR"},
            {"Name": "IsAlarmActive", "Value": str(record.is_alarm_active).lower(), "Type": "BOOLEAN"},
        ]

        timestream_record = {
            "Dimensions": dimensions,
            "MeasureName": "GasPlumeMetrics",
            "MeasureValueType": "MULTI",
            "MeasureValues": measure_values,
            "Time": str(record.timestamp_ms),
            "TimeUnit": "MILLISECONDS",
        }

        try:
            self._timestream_client.write_records(
                DatabaseName=self.database_name,
                TableName=self.table_name,
                Records=[timestream_record],
            )
        except Exception as ce:
            logger.debug("Timestream write bypassed or failed (offline/mock): %s", ce)
            return False

        # Broadcast real-time MQTT message for live Grafana WebSockets / Digital Twin viewer
        if self._iot_data_client is not None:
            topic = f"aerobos/facilities/{record.facility_id}/cameras/{record.camera_id}/stream"
            try:
                self._iot_data_client.publish(
                    topic=topic,
                    qos=0,
                    payload=json.dumps(record_dict),
                )
            except Exception as ie:
                logger.debug("IoT broadcast skipped: %s", ie)

        return True

    def get_latest_spooled_records(self, count: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent in-memory telemetry records."""
        return self._local_history_buffer[-count:]
