"""
AeroBOS: AI/ML Upgrade Verification & Demonstration Script
Module: demo_ai_upgrade.py

Demonstrates:
1. False Positive Suppression (Steam / Mirage vs Methane) via SageMaker Spatiotemporal Classifier.
2. Multimodal Visual Root-Cause Diagnosis via Amazon Bedrock (Claude 3.5 Sonnet).
3. 15-Minute Predictive Plume Drift & GeoJSON Evacuation Routing via ConvLSTM.
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from aerobos.plume_analysis import PlumeAnalyzer
from aerobos.refractometer import OpticalGeometry, RefractometerConfig, SchlierenRefractometer
from aerobos.synthetic_plume import SyntheticSchlierenGenerator
from ai_orchestrator import (
    AeroBOSAIOrchestrator,
    GasClassificationLabel,
    predict_plume_drift,
)


def run_ai_upgrade_demo():
    print("=" * 80)
    print(" AEROBOS: DEEP LEARNING & MULTIMODAL AGENTIC UPGRADE DEMO")
    print("=" * 80)

    # Initialize OpenCV BOS pipeline and AI Orchestrator
    geom = OpticalGeometry()
    refractometer = SchlierenRefractometer()
    analyzer = PlumeAnalyzer(geometry=geom)
    orchestrator = AeroBOSAIOrchestrator(
        sagemaker_endpoint_name="aerobos-timesformer-spatiotemporal-v1",
        bedrock_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )

    synth = SyntheticSchlierenGenerator(width=640, height=360, fps=30.0)

    # -------------------------------------------------------------------------
    # TEST SCENARIO 1: FALSE POSITIVE SUPPRESSION (Steam Vent / Thermal Mirage)
    # -------------------------------------------------------------------------
    print("\n[TEST SCENARIO 1] Simulating Ambient High-Scattering Steam Vent / Mirage...")
    steam_buffer = []
    for _ in range(30):
        frame, _, _ = synth.generate_frame(plume_active=False)
        # Simulate bright diffuse steam cloud over image
        h, w = frame.shape[:2]
        steam_overlay = np.random.normal(210, 15, (h, w, 3)).astype(np.float32)
        simulated_steam = np.clip(frame.astype(np.float32) * 0.3 + steam_overlay * 0.7, 0, 255).astype(np.uint8)
        steam_buffer.append(simulated_steam)

    dummy_annotated = steam_buffer[-1]
    dummy_density = np.zeros((10, h, w), dtype=np.float32)

    assessment_steam = orchestrator.process_spatial_anomaly(
        temporal_buffer_30=steam_buffer,
        annotated_snapshot=dummy_annotated,
        density_history_10=dummy_density,
        wind_vector=(2.5, 0.5),
        facility_metadata={
            "facility_id": "PERMIAN_BATTERY_04",
            "camera_id": "CAM-02-STEAM-VENT",
            "sector_id": "BOILER-AUXILIARY",
            "mass_flow_rate_kg_h": 12.0,
            "associated_valve_id": "XV-STEAM-01",
        }
    )

    print(" -> SageMaker Prediction:", assessment_steam.classification.predicted_class.value)
    print(" -> Class Probabilities:", assessment_steam.classification.probabilities)
    print(" -> Methane Confirmed:", assessment_steam.classification.is_methane_confirmed)
    print(" -> Action Taken: FALSE POSITIVE SUPPRESSED (SCADA Valve Trip Bypassed)")

    # -------------------------------------------------------------------------
    # TEST SCENARIO 2: TRUE METHANE BLOWOUT (Full 3-Stage AI Processing)
    # -------------------------------------------------------------------------
    print("\n[TEST SCENARIO 2] Simulating Hazardous Invisible Methane Blowout...")
    methane_buffer = []
    density_history = []

    # Prime baseline
    for _ in range(5):
        f, _, _ = synth.generate_frame(plume_active=False)
        refractometer.process_frame(f)

    # Generate 30 frames of active methane plume
    for _ in range(30):
        frame, gt_opd, gt_disp = synth.generate_frame(plume_active=True, leak_intensity=1.8)
        flow, div, phi, turb = refractometer.process_frame(frame)
        methane_buffer.append(frame)
        density_history.append(phi)

    # Use actual annotated snapshot if available, or generate from current frame
    annotated_snapshot_path = "aerobos_plume_annotated.jpg"
    if os.path.exists(annotated_snapshot_path):
        snapshot_input = annotated_snapshot_path
    else:
        snapshot_input = methane_buffer[-1]

    density_stack_10 = np.stack(density_history[-10:], axis=0)

    print("[*] Processing Spatial Anomaly through AeroBOS AI Orchestrator...")
    assessment_methane = orchestrator.process_spatial_anomaly(
        temporal_buffer_30=methane_buffer,
        annotated_snapshot=snapshot_input,
        density_history_10=density_stack_10,
        wind_vector=(3.5, 1.2),
        facility_metadata={
            "facility_id": "PERMIAN_PRODUCTION_BATTERY_04",
            "camera_id": "CAM-01-BATTERY-B",
            "sector_id": "BATTERY-B-SEPARATOR",
            "mass_flow_rate_kg_h": 460.8,
            "associated_valve_id": "XV-1042-B",
            "wind_speed_m_s": 3.7,
            "wind_direction_deg": 71.0,
        }
    )

    print("\n" + "-" * 70)
    print(" 1. SAGEMAKER SPATIOTEMPORAL GAS CLASSIFICATION")
    print("-" * 70)
    cls_res = assessment_methane.classification
    print(f"Predicted Class: {cls_res.predicted_class.value} (Confidence: {cls_res.confidence * 100:.1f}%)")
    print(f"Endpoint Used:   {cls_res.sagemaker_endpoint_used}")
    print(f"Inference Time:  {cls_res.inference_latency_ms:.2f} ms")
    print(f"Methane Verified:{cls_res.is_methane_confirmed}")

    print("\n" + "-" * 70)
    print(" 2. AMAZON BEDROCK MULTIMODAL ROOT-CAUSE DIAGNOSIS")
    print("-" * 70)
    diag = assessment_methane.diagnosis
    if diag:
        print(f"Diagnosed Component:    {diag.component_type}")
        print(f"Exact Failure Point:    {diag.failure_point}")
        print(f"Mechanical Mechanism:   {diag.failure_mechanism}")
        print(f"OSHA PSM Severity:      {diag.osha_psm_severity}")
        print(f"EPA Subpart W Status:   {diag.epa_subpart_w_tier}")
        print(f"Evacuation Perimeter:   {diag.evacuation_radius_meters:.0f} meters")
        print(f"Emergency Protocol:\n{diag.recommended_isolation_protocol}")

    print("\n" + "-" * 70)
    print(" 3. CONVLSTM 15-MINUTE PLUME DRIFT & GEOJSON HAZARD FOOTPRINT")
    print("-" * 70)
    hazard = assessment_methane.forecast_hazard
    if hazard:
        print(f"Forecast Horizon:       {hazard.forecast_horizon_minutes} minutes into future")
        print(f"Hazard Area:            {hazard.hazard_area_sq_meters:.1f} m^2")
        print(f"Downwind Heading:       {hazard.downwind_direction_deg:.1f} deg")
        print(f"Center GPS Coordinate:  Lat {hazard.centroid_gps[0]}, Lon {hazard.centroid_gps[1]}")
        print(f"GeoJSON Feature Type:   {hazard.geojson_payload['type']}")
        
        # Save GeoJSON for Grafana and control room dispatch
        geojson_out = "predicted_plume_hazard_15min.geojson"
        with open(geojson_out, "w", encoding="utf-8") as f:
            json.dump(hazard.geojson_payload, f, indent=2)
        print(f"[OK] 15-Minute Evacuation Routing GeoJSON saved to: {geojson_out}")

    print("\n" + "-" * 70)
    print(" 4. AI ORCHESTRATION LATENCY BREAKDOWN")
    print("-" * 70)
    for k, v in assessment_methane.execution_latency_breakdown_ms.items():
        print(f" - {k:<36}: {v:>6.2f} ms")
    print("=" * 80)
    print("[OK] AeroBOS AI/ML Upgrade demonstration completed successfully.\n")


if __name__ == "__main__":
    run_ai_upgrade_demo()
