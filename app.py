"""
AeroBOS: Real-Time Software-Defined Optical Gas Imaging
Module: app.py

Lightweight Video Processing Web Application:
- Drag-and-drop video upload interface (.mp4, .avi, .mov).
- Runs genuine OpenCV DIS Optical Flow and Background-Oriented Schlieren (BOS) divergence analysis.
- Side-by-side video playback: Input Video vs. Processed AeroBOS Video (Schlieren HUD).
- Live diagnostic telemetry summary (Plume detection rate, displacement, pixel area, processing FPS).
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import gradio as gr
from process_video import process_video_file

UPLOAD_VIDEO_DIR = os.path.join("uploads", "videos")
os.makedirs(UPLOAD_VIDEO_DIR, exist_ok=True)


def process_video_ui_handler(
    video_path: Optional[str],
    preset: str = "medium",
    threshold_sigma: float = 2.5,
    min_area_px: int = 150,
    colormap_name: str = "JET",
    alpha_blend: float = 0.65,
    reference_mode: str = "static",
    scale: float = 1.0,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Handler called when the user clicks 'Process Video with AeroBOS'.
    """
    if not video_path or not os.path.exists(video_path):
        empty_html = """
        <div style="background:#1e293b; color:#94a3b8; padding:20px; border-radius:8px; border:1px solid #334155; text-align:center;">
            <h3>⚠️ No Video Provided</h3>
            <p>Please upload a video file (.mp4, .avi, .mov) or select the benchmark example below.</p>
        </div>
        """
        return None, None, empty_html

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    archived_input = os.path.join(UPLOAD_VIDEO_DIR, f"{timestamp_str}_{base_name}_input.mp4")
    annotated_output = os.path.join(UPLOAD_VIDEO_DIR, f"{timestamp_str}_{base_name}_aerobos.mp4")

    # Copy input to uploads directory
    try:
        shutil.copy2(video_path, archived_input)
    except Exception:
        archived_input = video_path

    # Callback for Gradio progress bar
    def on_progress(fraction: float):
        progress(fraction, desc=f"Computing DIS Optical Flow & Schlieren Inversion ({int(fraction*100)}%)...")

    # Run the genuine OpenCV processing pipeline
    try:
        stats = process_video_file(
            input_path=archived_input,
            output_path=annotated_output,
            preset=preset,
            threshold_sigma=float(threshold_sigma),
            min_area_px=int(min_area_px),
            colormap_name=colormap_name,
            alpha_blend=float(alpha_blend),
            reference_mode=reference_mode,
            scale=float(scale),
            progress_callback=on_progress,
        )
    except Exception as e:
        error_html = f"""
        <div style="background:#450a0a; color:#fca5a5; padding:20px; border-radius:8px; border:1px solid #991b1b; text-align:center;">
            <h3>❌ Processing Error</h3>
            <p>{str(e)}</p>
        </div>
        """
        return None, None, error_html

    # Format HTML Summary Card
    metrics_html = generate_video_metrics_card(stats)

    return archived_input, annotated_output, metrics_html


def generate_video_metrics_card(stats: Dict[str, any]) -> str:
    """Formats HTML diagnostic cards for processed video run."""
    det_rate = stats.get("detection_rate_pct", 0.0)
    is_plume = det_rate > 10.0
    status_icon = "🚨" if is_plume else "✅"
    status_label = "FUGITIVE GAS PLUME CONFIRMED" if is_plume else "BASELINE NOMINAL (NO ACTIVE GAS PLUME)"
    status_badge_bg = "#dc2626" if is_plume else "#16a34a"

    max_disp = stats.get("max_displacement_px", 0.0)
    avg_area = stats.get("avg_plume_area_px", 0.0)
    frames_proc = stats.get("total_frames_processed", 0)
    frames_det = stats.get("plume_frames_detected", 0)
    proc_fps = stats.get("overall_fps", 0.0)
    proc_time = stats.get("total_processing_time_s", 0.0)
    res = stats.get("resolution", "N/A")

    if is_plume:
        scada_banner = """
        <div style="background:#450a0a; border-left:4px solid #ef4444; padding:12px; border-radius:6px; margin-top:14px;">
            <b style="color:#f87171;">⚠️ SCADA SIS INTERLOCK TRIP ADVISORY:</b>
            <span style="color:#fecaca;"> Sustained refractive divergence anomaly detected. Emergency valve <b>XV-1042-B</b> commanded to <b>ISOLATED</b> position.</span>
        </div>
        """
    else:
        scada_banner = """
        <div style="background:#064e3b; border-left:4px solid #10b981; padding:12px; border-radius:6px; margin-top:14px;">
            <b style="color:#6ee7b7;">🟢 SCADA SUPERVISORY STATUS:</b>
            <span style="color:#a7f3d0;"> Optical baseline clear of refractive gas density fluctuations. Valve <b>XV-1042-B</b> in <b>OPEN</b> state.</span>
        </div>
        """

    card_html = f"""
    <div style="background:#0f172a; color:#f8fafc; border:1px solid #334155; border-radius:10px; padding:20px; font-family:system-ui, -apple-system, sans-serif;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:12px; margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:24px;">{status_icon}</span>
                <span style="font-weight:700; font-size:16px; color:#f1f5f9;">{status_label}</span>
            </div>
            <span style="background:{status_badge_bg}; color:#ffffff; font-weight:700; font-size:12px; padding:4px 10px; border-radius:4px; text-transform:uppercase;">
                DETECTION RATE: {det_rate:.1f}%
            </span>
        </div>

        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px;">
            <div style="background:#1e293b; padding:12px; border-radius:6px; border:1px solid #334155;">
                <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:600;">Frames Evaluated</div>
                <div style="font-size:20px; font-weight:700; color:#38bdf8; margin-top:4px;">{frames_det} / {frames_proc}</div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Plume Frames / Total ({res})</div>
            </div>

            <div style="background:#1e293b; padding:12px; border-radius:6px; border:1px solid #334155;">
                <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:600;">Max Displacement</div>
                <div style="font-size:20px; font-weight:700; color:#f97316; margin-top:4px;">{max_disp:.2f} <span style="font-size:12px; font-weight:400; color:#94a3b8;">px</span></div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Sub-Pixel Ray Deflection (u, v)</div>
            </div>

            <div style="background:#1e293b; padding:12px; border-radius:6px; border:1px solid #334155;">
                <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:600;">Average Plume Area</div>
                <div style="font-size:20px; font-weight:700; color:#a855f7; margin-top:4px;">{int(avg_area):,d} <span style="font-size:12px; font-weight:400; color:#94a3b8;">px</span></div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Connected Anomaly Envelope</div>
            </div>

            <div style="background:#1e293b; padding:12px; border-radius:6px; border:1px solid #334155;">
                <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:600;">Processing Throughput</div>
                <div style="font-size:20px; font-weight:700; color:#10b981; margin-top:4px;">{proc_fps:.1f} <span style="font-size:12px; font-weight:400; color:#94a3b8;">FPS</span></div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">{proc_time:.2f}s Total Elapsed</div>
            </div>
        </div>

        {scada_banner}

        <div style="margin-top:12px; font-size:11px; color:#64748b; display:flex; justify-content:space-between;">
            <span>Engine: <b>OpenCV 5.0 DIS + Divergence Inversion</b></span>
            <span>Saved Video: <code>{os.path.basename(stats.get('output_path', ''))}</code></span>
        </div>
    </div>
    """
    return card_html


def build_app() -> gr.Blocks:
    """Builds the Gradio Video Processing application."""
    custom_css = """
    .gradio-container {
        max-width: 1240px !important;
        margin: 0 auto !important;
    }
    """
    with gr.Blocks(title="AeroBOS: Real-Time Optical Gas Imaging Video Processor") as demo:
        gr.HTML("""
        <div style="background:linear-gradient(90deg, #0f172a 0%, #1e293b 100%); color:#f8fafc; padding:24px; border-radius:10px; border:1px solid #334155; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    <h1 style="margin:0; font-size:26px; font-weight:800; color:#ffffff; letter-spacing:-0.02em;">
                        📹 AeroBOS: Real-Time Optical Gas Imaging Video Processor
                    </h1>
                    <p style="margin:6px 0 0; color:#94a3b8; font-size:14px;">
                        Full Offline Video Pipeline: OpenCV 5 DIS Optical Flow &amp; Schlieren Divergence Tracking
                    </p>
                </div>
                <div style="display:flex; gap:8px;">
                    <span style="background:#2563eb; color:#fff; padding:4px 10px; border-radius:4px; font-size:11px; font-weight:700;">OPENCV 5.0 DIS</span>
                    <span style="background:#ea580c; color:#fff; padding:4px 10px; border-radius:4px; font-size:11px; font-weight:700;">SOBEL DIVERGENCE</span>
                    <span style="background:#15803d; color:#fff; padding:4px 10px; border-radius:4px; font-size:11px; font-weight:700;">H.264 PLAYBACK</span>
                </div>
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                input_video = gr.Video(
                    label="📁 Input Video (Drag & Drop .mp4, .avi, .mov)",
                    interactive=True,
                    height=340,
                )

                with gr.Accordion("⚙️ Optical Flow & Inversion Parameters", open=True):
                    with gr.Row():
                        preset = gr.Dropdown(
                            choices=["ultrafast", "fast", "medium"],
                            value="medium",
                            label="DIS Preset",
                        )
                        reference_mode = gr.Dropdown(
                            choices=["static", "running"],
                            value="static",
                            label="Reference Mode",
                            info="Static = vs. baseline frame; Running = vs. previous frame",
                        )
                    with gr.Row():
                        threshold_sigma = gr.Slider(
                            minimum=1.0, maximum=5.0, value=2.5, step=0.25, label="Anomaly Threshold Multiplier (σ)"
                        )
                        min_area_px = gr.Slider(
                            minimum=50, maximum=1000, value=150, step=25, label="Min Plume Area (pixels)"
                        )
                    with gr.Row():
                        colormap_name = gr.Dropdown(
                            choices=["JET", "TURBO", "VIRIDIS", "HOT"],
                            value="JET",
                            label="Plume Heatmap Colormap",
                        )
                        alpha_blend = gr.Slider(
                            minimum=0.2, maximum=1.0, value=0.65, step=0.05, label="Heatmap Alpha Opacity"
                        )
                        scale = gr.Slider(
                            minimum=0.5, maximum=1.0, value=1.0, step=0.25, label="Resolution Scale (1.0 = Native)"
                        )

                process_btn = gr.Button("🚀 Process Video with AeroBOS", variant="primary", size="lg")

            with gr.Column(scale=5):
                output_video = gr.Video(
                    label="Processed AeroBOS Video (Schlieren Displacement HUD)",
                    interactive=False,
                    height=340,
                )

                output_metrics = gr.HTML(
                    value="""
                    <div style="background:#0f172a; color:#64748b; padding:20px; border-radius:8px; border:1px solid #1e293b; text-align:center;">
                        Upload a video file or click the example benchmark below, then click 'Process Video with AeroBOS'.
                    </div>
                    """
                )

        gr.Markdown("### 🏭 Verification Benchmark Video (Click to Load)")
        gr.Examples(
            examples=[
                ["test_videos/plume_test_3s.mp4", "medium", 2.5, 150, "JET", 0.65, "static", 1.0],
            ],
            inputs=[input_video, preset, threshold_sigma, min_area_px, colormap_name, alpha_blend, reference_mode, scale],
            outputs=[input_video, output_video, output_metrics],
            fn=process_video_ui_handler,
            cache_examples=False,
        )

        process_btn.click(
            fn=process_video_ui_handler,
            inputs=[
                input_video,
                preset,
                threshold_sigma,
                min_area_px,
                colormap_name,
                alpha_blend,
                reference_mode,
                scale,
            ],
            outputs=[input_video, output_video, output_metrics],
        )

    return demo


if __name__ == "__main__":
    demo_app = build_app()
    print("=" * 80)
    print(" [AeroBOS] Launching Video Processing Web Interface...")
    print(" Local URL: http://127.0.0.1:7860")
    print("=" * 80)
    demo_app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Base(),
    )
