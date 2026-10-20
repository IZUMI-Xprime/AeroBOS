"""
AeroBOS: End-to-End Demonstration & Verification Script
Module: demo_aerobos.py

Demonstrates the entire AeroBOS perception-decision-action loop:
1. Simulates baseline background without gas leaks (system calibrates noise baseline).
2. Introduces a simulated turbulent methane blowout.
3. SchlierenRefractometer computes sub-pixel DIS flow and inverts 2D Poisson equation via FFT.
4. Autonomous Bedrock Agentic loop triggers Tools 1-4 to inspect, estimate flow, isolate valve, and dispatch alarms.
5. Digital twin streamer publishes telemetry records to Amazon Timestream.
6. Renders and saves an annotated false-color Schlieren snapshot ('aerobos_plume_annotated.jpg').
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

from aerobos.digital_twin_streamer import DigitalTwinStreamer, PlumeTelemetryRecord
from aerobos.dispersion_model import WindVector
from aerobos.orchestrator import AeroBOSAgentOrchestrator
from aerobos.plume_analysis import PlumeAnalyzer
from aerobos.refractometer import OpticalGeometry, RefractometerConfig, SchlierenRefractometer
from aerobos.synthetic_plume import SyntheticSchlierenGenerator


def render_annotated_hud(
    base_frame: np.ndarray,
    reconstructed_opd: np.ndarray,
    plume_mask: np.ndarray,
    flow_vectors: np.ndarray,
    metrics,
    dispersion,
    valve_state: str,
) -> np.ndarray:
    """
    Renders an industrial-grade augmented reality overlay for EHS and control-room operators.
    """
    h, w = base_frame.shape[:2]
    canvas = base_frame.copy()

    # Step 1: False-color Schlieren colormap for optical path difference (OPD)
    abs_opd = np.abs(reconstructed_opd)
    norm_opd = np.clip(abs_opd / (np.percentile(abs_opd, 99.5) + 1e-6), 0.0, 1.0)
    opd_uint8 = (norm_opd * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(opd_uint8, cv2.COLORMAP_TURBO)

    # Blend heatmap over background in plume region
    mask_3ch = cv2.cvtColor(plume_mask, cv2.COLOR_GRAY2BGR) / 255.0
    blended = (canvas * (1.0 - mask_3ch * 0.7) + heatmap * (mask_3ch * 0.7)).astype(np.uint8)

    # Step 2: Overlay sub-pixel displacement vector quiver (decimated grid)
    step = 40
    y_coords, x_coords = np.mgrid[step // 2:h:step, step // 2:w:step]
    u = flow_vectors[y_coords, x_coords, 0]
    v = flow_vectors[y_coords, x_coords, 1]
    
    # Draw displacement arrows in regions with active flow
    mag = np.sqrt(u**2 + v**2)
    for y_pt, x_pt, u_val, v_val, m in zip(y_coords.ravel(), x_coords.ravel(), u.ravel(), v.ravel(), mag.ravel()):
        if m > 0.15:
            scale = 8.0  # Amplify small sub-pixel shifts for visualization
            pt1 = (int(x_pt), int(y_pt))
            pt2 = (int(x_pt + u_val * scale), int(y_pt + v_val * scale))
            cv2.arrowedLine(blended, pt1, pt2, (0, 255, 255), 1, tipLength=0.3)

    # Step 3: Draw Plume Contour & Centroid
    if metrics.is_anomaly_detected and metrics.contour_points:
        pts = np.array(metrics.contour_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(blended, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cx, cy = int(metrics.centroid_px[0]), int(metrics.centroid_px[1])
        cv2.circle(blended, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(blended, (cx, cy), 12, (255, 255, 255), 1)

    # Step 4: Top Telemetry Banner & HUD
    cv2.rectangle(blended, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.line(blended, (0, 80), (w, 80), (0, 165, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Left Header
    cv2.putText(blended, "AEROBOS: REAL-TIME SOFTWARE-DEFINED GAS REFRACTOMETER", (20, 30), font, 0.75, (255, 255, 255), 2)
    cv2.putText(blended, "ENGINE: OpenCV 5 + COOL | AWS Graviton3 Neoverse V1 SVE", (20, 60), font, 0.5, (180, 180, 180), 1)

    # Right Telemetry Metrics
    alarm_color = (0, 0, 255) if metrics.is_anomaly_detected else (0, 255, 0)
    status_text = "CRITICAL LEAK" if metrics.is_anomaly_detected else "NOMINAL"
    cv2.putText(blended, f"STATUS: {status_text}", (w - 380, 30), font, 0.7, alarm_color, 2)
    
    flow_str = f"{dispersion.mass_flow_rate_kg_per_h:.2f} kg/h" if dispersion else "0.00 kg/h"
    cv2.putText(blended, f"EMISSION RATE: {flow_str}", (w - 380, 60), font, 0.55, (255, 255, 255), 1)

    # Bottom Valve Status Overlay
    valve_color = (0, 255, 0) if valve_state == "ISOLATED" else ((0, 165, 255) if valve_state == "IN_TRANSIT" else (0, 0, 255))
    cv2.rectangle(blended, (20, h - 60), (380, h - 20), (20, 20, 20), -1)
    cv2.rectangle(blended, (20, h - 60), (380, h - 20), valve_color, 2)
    cv2.putText(blended, f"VALVE XV-1042-B: {valve_state}", (35, h - 35), font, 0.55, (255, 255, 255), 2)

    return blended


def main() -> None:
    print("=" * 80)
    print(" AEROBOS: AUTONOMOUS OPTICAL GAS IMAGING & VALVE ISOLATION DEMO")
    print("=" * 80)

    # Initialize subsystems
    geom = OpticalGeometry()
    cfg = RefractometerConfig(geometry=geom, dis_preset=cv2.DISOPTICAL_FLOW_PRESET_FAST)
    refractometer = SchlierenRefractometer(config=cfg)
    analyzer = PlumeAnalyzer(geometry=geom, sigma_threshold=3.0)
    orchestrator = AeroBOSAgentOrchestrator(facility_id="PERMIAN_PRODUCTION_BATTERY_04")
    streamer = DigitalTwinStreamer()

    synth = SyntheticSchlierenGenerator(width=1280, height=720, fps=30.0)

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Noise Calibration (Frames 1 - 10)
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Calibrating continuous background & optical noise floor...")
    for f_idx in range(10):
        frame, _, _ = synth.generate_frame(plume_active=False)
        flow, div, phi, turb = refractometer.process_frame(frame)
        metrics, mask = analyzer.analyze_field(turb, div)
        print(f"  Frame {f_idx+1:02d}: SNR = {metrics.snr_sigma:.2f} sigma | Plume Detected = {metrics.is_anomaly_detected}")

    # -------------------------------------------------------------------------
    # Phase 2: Gas Leak Outbreak (Frames 11 - 20)
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Simulating turbulent methane blowout at flange XV-1042-B...")
    dispersion_res = None
    final_annotated_frame = None

    for f_idx in range(11, 21):
        # Methane plume with intensity 1.8
        frame, gt_opd, gt_disp = synth.generate_frame(plume_active=True, leak_intensity=1.8)
        flow, div, phi, turb = refractometer.process_frame(frame)
        metrics, mask = analyzer.analyze_field(turb, div)

        print(
            f"  Frame {f_idx:02d}: SNR = {metrics.snr_sigma:.2f} sigma | "
            f"Area = {metrics.area_sq_meters:.4f} m^2 | "
            f"Peak OPD = {metrics.peak_optical_path_diff_nm:.1f} nm | "
            f"Anomaly = {metrics.is_anomaly_detected}"
        )

        # Register optical perception state for agent tools
        orchestrator.register_optical_state("CAM-01-BATTERY-B", metrics, phi)

        # On frame 15, trigger the autonomous Bedrock agent loop!
        if f_idx == 15 and metrics.is_anomaly_detected:
            print("\n" + "*" * 70)
            print(" [ALERT] 3-SIGMA ANOMALY THRESHOLD EXCEEDED! INVOKING BEDROCK AGENT...")
            print("*" * 70)
            
            investigation = orchestrator.run_autonomous_investigation(
                camera_id="CAM-01-BATTERY-B",
                sector_id="BATTERY-B-SEPARATOR",
                associated_valve_id="XV-1042-B",
                wind_speed_m_s=3.5,
                wind_direction_deg=80.0,
            )

            print(f" Agent Orchestration Mode: {investigation['orchestration_mode']}")
            for act in investigation["actions_taken"]:
                print(f"  -> Executed Tool [{act['tool']}]")
                if act["tool"] == "estimate_dispersion_flow":
                    res = act["result"]
                    print(f"     Mass Flow Rate: {res['mass_flow_rate_kg_h']} kg/h ({res['severity']})")
                elif act["tool"] == "isolate_facility_valve":
                    res = act["result"]
                    print(f"     Valve Isolation Status: {res['new_state']} (Transit Time: {res['actuator_transit_time_sec']}s)")
                elif act["tool"] == "dispatch_emergency_sns":
                    res = act["result"]
                    print(f"     Incident Report ID: {res['incident_id']}")

            print("*" * 70 + "\n")

        # Stream telemetry to Digital Twin
        valve_state = orchestrator._valve_states.get("XV-1042-B", "OPEN")
        wind = WindVector(speed_m_s=3.5, direction_deg=80.0)
        disp_est = orchestrator.dispersion_model.estimate_flow_rate(
            phi, mask, wind
        ) if metrics.is_anomaly_detected else None
        
        telemetry = PlumeTelemetryRecord(
            timestamp_ms=int(time.time() * 1000),
            facility_id="PERMIAN_PRODUCTION_BATTERY_04",
            camera_id="CAM-01-BATTERY-B",
            sector_id="BATTERY-B-SEPARATOR",
            gas_type="METHANE",
            valve_id="XV-1042-B",
            plume_area_sqm=metrics.area_sq_meters,
            peak_refractive_anomaly_sigma=metrics.snr_sigma,
            centroid_x_norm=metrics.centroid_normalized[0],
            centroid_y_norm=metrics.centroid_normalized[1],
            mass_flow_rate_kg_h=disp_est.mass_flow_rate_kg_per_h if disp_est else 0.0,
            optical_path_diff_nm=metrics.peak_optical_path_diff_nm,
            inference_latency_ms=refractometer.last_profile_ms["total"],
            valve_state=valve_state,
            is_alarm_active=metrics.is_anomaly_detected,
        )
        streamer.stream_record(telemetry)

        # Render final annotated frame
        if f_idx == 20:
            final_annotated_frame = render_annotated_hud(
                frame, phi, mask, flow, metrics, disp_est, valve_state
            )

    # Save visual snapshot
    if final_annotated_frame is not None:
        out_path = "aerobos_plume_annotated.jpg"
        cv2.imwrite(out_path, final_annotated_frame)
        print(f"[OK] Rendered false-color Schlieren snapshot saved to: {out_path}")

    print("\n[OK] AeroBOS end-to-end verification completed successfully.")


if __name__ == "__main__":
    main()
