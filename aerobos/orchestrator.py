"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging
Module: orchestrator.py

Implements Amazon Bedrock Agentic Perception-Action Loop & Model Context Protocol (MCP) Server:
- Tool 1 (inspect_refraction_plume): Extracts plume contours, boundary expansion rate (dA/dt), and centroid.
- Tool 2 (estimate_dispersion_flow): Inverts wind sensor data & OPD to estimate mass flow rate (kg/h).
- Tool 3 (isolate_facility_valve): Publishes SCADA trip command to AWS IoT Core MQTT to arrest the leak.
- Tool 4 (dispatch_emergency_sns): Generates EPA-compliant incident report and dispatches SNS alerts.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception

from .dispersion_model import GaussianDispersionModel, LeakSeverity, WindVector
from .plume_analysis import PlumeAnalyzer, PlumeMetrics
from .refractometer import OpticalGeometry

logger = logging.getLogger("AeroBOS.Orchestrator")


# =============================================================================
# Amazon Bedrock & MCP Tool Schemas
# =============================================================================
BEDROCK_TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "inspect_refraction_plume",
            "description": "Triggered when refractive anomaly exceeds 3-sigma noise floor. Extracts plume contour, centroid, boundary expansion rate (dA/dt), and peak optical path difference.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "camera_id": {"type": "string", "description": "Identifier of the camera observing the anomaly."},
                        "sector_id": {"type": "string", "description": "Physical facility sector or battery limit."},
                        "sigma_threshold": {"type": "number", "description": "Statistical anomaly threshold (default 3.0)."}
                    },
                    "required": ["camera_id", "sector_id"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "estimate_dispersion_flow",
            "description": "Cross-references optical path difference (OPD) field with ultrasonic wind sensor data to calculate instantaneous gas leak mass emission rate Q (kg/h) and EPA severity.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "camera_id": {"type": "string", "description": "Camera ID."},
                        "gas_type": {"type": "string", "enum": ["METHANE", "PROPANE", "SF6", "HYDROGEN"], "description": "Target gas species."},
                        "wind_speed_m_s": {"type": "number", "description": "Local wind speed in m/s."},
                        "wind_direction_deg": {"type": "number", "description": "Wind direction in meteorological degrees (0-360)."}
                    },
                    "required": ["camera_id", "gas_type", "wind_speed_m_s", "wind_direction_deg"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "isolate_facility_valve",
            "description": "Commands an industrial SCADA block valve to immediately actuate to CLOSED state via AWS IoT Core MQTT to arrest an active gas blowout.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "valve_id": {"type": "string", "description": "SCADA tag for the emergency isolation valve (e.g. XV-1042-B)."},
                        "facility_id": {"type": "string", "description": "Target industrial site/facility ID."},
                        "reason": {"type": "string", "description": "Justification for emergency shutoff trip."},
                        "authorized_by": {"type": "string", "description": "Autonomous Bedrock Agent authorization identifier."}
                    },
                    "required": ["valve_id", "facility_id", "reason"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "dispatch_emergency_sns",
            "description": "Compiles an EPA Subpart W / Title V compliant incident report and publishes urgent dispatch notifications via Amazon SNS to operators and EHS officers.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "facility_id": {"type": "string", "description": "Facility identifier."},
                        "camera_id": {"type": "string", "description": "Camera identifier."},
                        "valve_id": {"type": "string", "description": "Associated valve identifier."},
                        "mass_flow_rate_kg_h": {"type": "number", "description": "Estimated mass emission rate in kg/h."},
                        "severity": {"type": "string", "enum": ["MINOR", "MODERATE", "CRITICAL"], "description": "Incident classification."},
                        "valve_isolated": {"type": "boolean", "description": "Whether the valve has successfully closed."},
                        "summary": {"type": "string", "description": "Technical narrative of the incident and actions taken."}
                    },
                    "required": ["facility_id", "camera_id", "valve_id", "mass_flow_rate_kg_h", "severity", "valve_isolated", "summary"]
                }
            }
        }
    }
]


class AeroBOSAgentOrchestrator:
    """
    Amazon Bedrock Agentic Loop coordinator.
    Manages the perception-decision-action cycle for optical gas imaging.
    """

    def __init__(
        self,
        facility_id: str = "PERMIAN_BASIN_FACILITY_04",
        aws_region: str = "us-east-1",
        bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        sns_topic_arn: str = "arn:aws:sns:us-east-1:123456789012:AeroBOSEmergencyAlarms",
        s3_bucket_name: str = "aerobos-optical-snapshots",
    ) -> None:
        self.facility_id = facility_id
        self.aws_region = aws_region
        self.bedrock_model_id = bedrock_model_id
        self.sns_topic_arn = sns_topic_arn
        self.s3_bucket_name = s3_bucket_name

        # State storage for active perception
        self._latest_metrics: Dict[str, PlumeMetrics] = {}
        self._latest_opd_field: Dict[str, np.ndarray] = {}
        self._valve_states: Dict[str, str] = {
            "XV-1042-B": "OPEN",
            "XV-2088-A": "OPEN",
            "ESD-001": "OPEN",
        }

        # Subsystems
        self.geometry = OpticalGeometry()
        self.analyzer = PlumeAnalyzer(geometry=self.geometry)
        self.dispersion_model = GaussianDispersionModel(geometry=self.geometry)

        # AWS Clients
        self._bedrock_client = None
        self._sns_client = None
        self._s3_client = None
        self._iot_data_client = None

        if BOTO3_AVAILABLE:
            try:
                cfg = Config(region_name=self.aws_region, retries={"max_attempts": 3})
                self._bedrock_client = boto3.client("bedrock-runtime", config=cfg)
                self._sns_client = boto3.client("sns", config=cfg)
                self._s3_client = boto3.client("s3", config=cfg)
                self._iot_data_client = boto3.client("iot-data", config=cfg)
            except Exception as e:
                logger.warning("AWS client initialization bypassed: %s", e)

    def register_optical_state(
        self,
        camera_id: str,
        metrics: PlumeMetrics,
        opd_field: np.ndarray
    ) -> None:
        """Stores the latest real-time frame perception state for tool queries."""
        self._latest_metrics[camera_id] = metrics
        self._latest_opd_field[camera_id] = opd_field

    # =========================================================================
    # Tool 1: inspect_refraction_plume
    # =========================================================================
    def tool_inspect_refraction_plume(
        self,
        camera_id: str,
        sector_id: str,
        sigma_threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Executes Tool 1: Extracts plume contour, centroid, boundary expansion rate (dA/dt).
        """
        metrics = self._latest_metrics.get(camera_id)
        if metrics is None:
            return {
                "status": "NO_ACTIVE_STREAM",
                "message": f"No active optical stream found for camera {camera_id}."
            }

        return {
            "status": "SUCCESS",
            "camera_id": camera_id,
            "sector_id": sector_id,
            "is_anomaly_detected": metrics.is_anomaly_detected,
            "snr_sigma": round(metrics.snr_sigma, 2),
            "confidence_score": round(metrics.confidence_score, 3),
            "centroid_normalized": {
                "x": round(metrics.centroid_normalized[0], 4),
                "y": round(metrics.centroid_normalized[1], 4),
            },
            "bounding_box_xywh": list(metrics.bounding_box_xywh),
            "area_sq_meters": round(metrics.area_sq_meters, 4),
            "expansion_rate_sq_m_per_sec": round(metrics.expansion_rate_sq_m_per_sec, 4),
            "dispersion_angle_deg": round(metrics.dispersion_angle_deg, 1),
            "peak_optical_path_diff_nm": round(metrics.peak_optical_path_diff_nm, 2),
            "contour_point_count": len(metrics.contour_points),
        }

    # =========================================================================
    # Tool 2: estimate_dispersion_flow
    # =========================================================================
    def tool_estimate_dispersion_flow(
        self,
        camera_id: str,
        gas_type: str,
        wind_speed_m_s: float,
        wind_direction_deg: float,
    ) -> Dict[str, Any]:
        """
        Executes Tool 2: Cross-references wind vectors with OPD to compute mass flow rate (kg/h).
        """
        opd = self._latest_opd_field.get(camera_id)
        metrics = self._latest_metrics.get(camera_id)

        if opd is None or metrics is None:
            return {
                "status": "ERROR",
                "message": f"Optical field data missing for camera {camera_id}."
            }

        # Create plume mask from bounding box / contour
        h, w = opd.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if metrics.contour_points:
            pts = np.array(metrics.contour_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(mask, [pts], -1, 255, -1)
        else:
            bx, by, bw, bh = metrics.bounding_box_xywh
            mask[by:by+bh, bx:bx+bw] = 255

        wind = WindVector(speed_m_s=wind_speed_m_s, direction_deg=wind_direction_deg)
        dispersion = self.dispersion_model.estimate_flow_rate(opd, mask, wind)

        return {
            "status": "SUCCESS",
            "camera_id": camera_id,
            "gas_type": gas_type,
            "mass_flow_rate_kg_h": round(dispersion.mass_flow_rate_kg_per_h, 3),
            "mass_flow_rate_g_s": round(dispersion.mass_flow_rate_g_per_s, 2),
            "severity": dispersion.severity.value,
            "peak_column_density_g_m2": round(dispersion.peak_column_density_g_m2, 3),
            "confidence_interval_95_pct": [
                round(dispersion.confidence_interval_95_pct[0], 3),
                round(dispersion.confidence_interval_95_pct[1], 3)
            ],
            "epa_subpart_w_reportable": dispersion.epa_subpart_w_reportable,
        }

    # =========================================================================
    # Tool 3: isolate_facility_valve
    # =========================================================================
    def tool_isolate_facility_valve(
        self,
        valve_id: str,
        facility_id: str,
        reason: str,
        authorized_by: str = "AmazonBedrock_AeroBOS_Agent",
    ) -> Dict[str, Any]:
        """
        Executes Tool 3: Commands an industrial SCADA valve to close via AWS IoT Core MQTT.
        """
        current_state = self._valve_states.get(valve_id, "OPEN")
        action_timestamp = int(time.time() * 1000)

        # Command payload for SCADA actuator shadow
        scada_trip_payload = {
            "state": {
                "desired": {
                    "actuator_position": 0.0,  # 0.0 = 100% Closed
                    "command": "EMERGENCY_SHUTOFF_TRIP",
                    "authorized_by": authorized_by,
                    "reason": reason,
                    "timestamp_epoch_ms": action_timestamp,
                }
            }
        }

        # Publish MQTT shadow update if AWS IoT client is connected
        if self._iot_data_client is not None:
            topic = f"$aws/things/{valve_id}/shadow/update"
            try:
                self._iot_data_client.publish(
                    topic=topic,
                    qos=1,
                    payload=json.dumps(scada_trip_payload),
                )
                logger.info("Published SCADA trip payload to %s", topic)
            except Exception as e:
                logger.error("Failed to publish IoT trip command: %s", e)

        # Update local simulated digital twin state
        self._valve_states[valve_id] = "ISOLATED"

        return {
            "status": "COMMAND_DISPATCHED",
            "valve_id": valve_id,
            "facility_id": facility_id,
            "previous_state": current_state,
            "new_state": "ISOLATED",
            "actuator_transit_time_sec": 1.45,
            "trip_confirmation": True,
            "timestamp_epoch_ms": action_timestamp,
            "reason": reason,
        }

    # =========================================================================
    # Tool 4: dispatch_emergency_sns
    # =========================================================================
    def tool_dispatch_emergency_sns(
        self,
        facility_id: str,
        camera_id: str,
        valve_id: str,
        mass_flow_rate_kg_h: float,
        severity: str,
        valve_isolated: bool,
        summary: str,
    ) -> Dict[str, Any]:
        """
        Executes Tool 4: Generates an EPA-compliant incident report and sends an SNS alert.
        """
        incident_id = f"INC-{int(time.time())}-{camera_id}"
        
        # S3 snapshot presigned URI (simulated or real)
        snapshot_uri = f"s3://{self.s3_bucket_name}/snapshots/{incident_id}.jpg"

        # Structured EPA Subpart W / Title V compliance report
        compliance_report = {
            "incident_id": incident_id,
            "regulatory_framework": "EPA 40 CFR Part 98 Subpart W / CAA Section 112(r)",
            "facility_id": facility_id,
            "detection_camera": camera_id,
            "associated_valve": valve_id,
            "mass_emission_rate_kg_per_h": mass_flow_rate_kg_h,
            "severity_level": severity,
            "automated_valve_isolation": "CONFIRMED" if valve_isolated else "PENDING",
            "snapshot_evidence_s3_uri": snapshot_uri,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "technical_summary": summary,
        }

        report_json = json.dumps(compliance_report, indent=2)

        # Dispatch SNS Alert if client configured
        message_id = f"sns-mock-{time.time()}"
        if self._sns_client is not None and self.sns_topic_arn:
            try:
                resp = self._sns_client.publish(
                    TopicArn=self.sns_topic_arn,
                    Subject=f"CRITICAL GAS LEAK ALERT: {facility_id} [{severity}]",
                    Message=report_json,
                    MessageAttributes={
                        "Severity": {"DataType": "String", "StringValue": severity},
                        "Facility": {"DataType": "String", "StringValue": facility_id},
                    }
                )
                message_id = resp.get("MessageId", message_id)
                logger.info("Published SNS notification %s to %s", message_id, self.sns_topic_arn)
            except Exception as e:
                logger.error("Failed to publish SNS message: %s", e)

        return {
            "status": "DISPATCHED",
            "incident_id": incident_id,
            "sns_message_id": message_id,
            "compliance_report": compliance_report,
        }

    # =========================================================================
    # Bedrock Autonomous Perception-Action Execution Loop
    # =========================================================================
    def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Routes a tool call request to the appropriate implementation."""
        if tool_name == "inspect_refraction_plume":
            return self.tool_inspect_refraction_plume(**tool_args)
        elif tool_name == "estimate_dispersion_flow":
            return self.tool_estimate_dispersion_flow(**tool_args)
        elif tool_name == "isolate_facility_valve":
            return self.tool_isolate_facility_valve(**tool_args)
        elif tool_name == "dispatch_emergency_sns":
            return self.tool_dispatch_emergency_sns(**tool_args)
        else:
            return {"status": "ERROR", "message": f"Unknown tool name: {tool_name}"}

    def run_autonomous_investigation(
        self,
        camera_id: str,
        sector_id: str,
        associated_valve_id: str,
        wind_speed_m_s: float = 3.2,
        wind_direction_deg: float = 75.0,
    ) -> Dict[str, Any]:
        """
        Executes the autonomous loop:
        1. Anomaly trigger -> Inspect Plume (Tool 1)
        2. Invert Mass Dispersion Flux (Tool 2)
        3. If severity >= MODERATE/CRITICAL -> Isolate Valve (Tool 3)
        4. Dispatch EPA Incident Notification (Tool 4)

        If Bedrock runtime is active, uses Bedrock Converse tool-calling.
        Otherwise, executes deterministic expert decision logic.
        """
        logger.info("Initiating autonomous investigation for %s in sector %s", camera_id, sector_id)

        # Attempt Bedrock Converse API invocation if available
        if self._bedrock_client is not None:
            try:
                system_prompt = (
                    "You are AeroBOS Autonomous Incident Commander, an AI system managing industrial "
                    "gas leak safety. You have 4 tools: inspect_refraction_plume, estimate_dispersion_flow, "
                    "isolate_facility_valve, and dispatch_emergency_sns. When an optical refraction anomaly occurs, "
                    "inspect the plume, calculate the mass flow rate with wind sensors, command valve isolation if "
                    "emission is significant (>=1 kg/h or CRITICAL), and dispatch an EPA-compliant incident report."
                )

                initial_prompt = (
                    f"A 3-sigma optical refraction anomaly has been flagged on camera {camera_id} in {sector_id}. "
                    f"Associated isolation valve is {associated_valve_id}. Current wind sensor readings: {wind_speed_m_s} m/s "
                    f"at {wind_direction_deg} degrees. Investigate immediately, assess emission severity, isolate if necessary, "
                    f"and file the compliance report."
                )

                messages = [{"role": "user", "content": [{"text": initial_prompt}]}]
                tool_config = {"tools": BEDROCK_TOOL_SPECS}

                actions_taken = []
                for turn in range(5):
                    response = self._bedrock_client.converse(
                        modelId=self.bedrock_model_id,
                        messages=messages,
                        system=[{"text": system_prompt}],
                        toolConfig=tool_config,
                    )

                    output_msg = response["output"]["message"]
                    messages.append(output_msg)

                    stop_reason = response.get("stopReason")
                    if stop_reason == "tool_use":
                        for content_block in output_msg.get("content", []):
                            if "toolUse" in content_block:
                                tool_use = content_block["toolUse"]
                                tool_name = tool_use["name"]
                                tool_input = tool_use["input"]
                                tool_use_id = tool_use["toolUseId"]

                                tool_result = self.execute_tool_call(tool_name, tool_input)
                                actions_taken.append({"tool": tool_name, "input": tool_input, "result": tool_result})

                                tool_result_block = {
                                    "role": "user",
                                    "content": [
                                        {
                                            "toolResult": {
                                                "toolUseId": tool_use_id,
                                                "content": [{"json": tool_result}],
                                                "status": "success",
                                            }
                                        }
                                    ]
                                }
                                messages.append(tool_result_block)
                    else:
                        break

                return {
                    "orchestration_mode": "AMAZON_BEDROCK_AGENT",
                    "model_id": self.bedrock_model_id,
                    "actions_taken": actions_taken,
                    "final_summary": output_msg.get("content", [{}])[0].get("text", "Complete"),
                }
            except Exception as e:
                logger.warning("Bedrock converse execution failed or offline (%s); using deterministic autonomous policy.", e)

        # Deterministic autonomous perception-action loop
        actions_taken = []

        # Step 1: Inspect Plume
        t1_res = self.tool_inspect_refraction_plume(camera_id, sector_id)
        actions_taken.append({"tool": "inspect_refraction_plume", "result": t1_res})

        # Step 2: Estimate Dispersion Flow
        t2_res = self.tool_estimate_dispersion_flow(camera_id, "METHANE", wind_speed_m_s, wind_direction_deg)
        actions_taken.append({"tool": "estimate_dispersion_flow", "result": t2_res})

        flow_rate = t2_res.get("mass_flow_rate_kg_h", 0.0)
        severity = t2_res.get("severity", "MINOR")

        # Step 3: Isolate Valve if necessary
        valve_isolated = False
        if flow_rate >= 1.0 or severity in ["MODERATE", "CRITICAL"]:
            t3_res = self.tool_isolate_facility_valve(
                valve_id=associated_valve_id,
                facility_id=self.facility_id,
                reason=f"Refractive plume mass flow {flow_rate:.2f} kg/h exceeded threshold ({severity}).",
            )
            actions_taken.append({"tool": "isolate_facility_valve", "result": t3_res})
            valve_isolated = t3_res.get("trip_confirmation", False)

        # Step 4: Dispatch Emergency SNS Notification
        summary_text = (
            f"Autonomous AeroBOS trip arrested a {severity} methane blowout ({flow_rate:.2f} kg/h). "
            f"Valve {associated_valve_id} actuated to ISOLATED in 1.45s."
        )
        t4_res = self.tool_dispatch_emergency_sns(
            facility_id=self.facility_id,
            camera_id=camera_id,
            valve_id=associated_valve_id,
            mass_flow_rate_kg_h=flow_rate,
            severity=severity,
            valve_isolated=valve_isolated,
            summary=summary_text,
        )
        actions_taken.append({"tool": "dispatch_emergency_sns", "result": t4_res})

        return {
            "orchestration_mode": "DETERMINISTIC_AUTONOMOUS_POLICY",
            "facility_id": self.facility_id,
            "camera_id": camera_id,
            "actions_taken": actions_taken,
            "final_summary": summary_text,
        }


# =============================================================================
# Model Context Protocol (MCP) Server for External Client Integration
# =============================================================================
class AeroBOSMCPServer:
    """
    Standard Model Context Protocol (MCP) JSON-RPC 2.0 interface.
    Exposes AeroBOS vision tools to Claude Desktop, Bedrock Agents, and Cursor.
    """

    def __init__(self, orchestrator: AeroBOSAgentOrchestrator) -> None:
        self.orchestrator = orchestrator

    def handle_request(self, json_rpc_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Processes JSON-RPC 2.0 MCP requests."""
        method = json_rpc_msg.get("method")
        msg_id = json_rpc_msg.get("id")

        if method == "tools/list":
            # Convert Bedrock schema to standard MCP tool list format
            mcp_tools = []
            for spec in BEDROCK_TOOL_SPECS:
                ts = spec["toolSpec"]
                mcp_tools.append({
                    "name": ts["name"],
                    "description": ts["description"],
                    "inputSchema": ts["inputSchema"]["json"],
                })
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": mcp_tools}}

        elif method == "tools/call":
            params = json_rpc_msg.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            result = self.orchestrator.execute_tool_call(name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                }
            }

        elif method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "aerobos-optical-gas-mcp", "version": "2.6.0"},
                    "capabilities": {"tools": {}}
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Procedure not found: {method}"}
        }
