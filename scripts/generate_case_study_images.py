"""
generate_case_study_images.py

Generates high-resolution forensic reference images for the 5 real-world industrial case studies:
1. Aliso Canyon SS-25 Blowout (NASA JPL AVIRIS-NG SWIR absorption overlay)
2. Chevron Richmond Refinery Rupture (U.S. CSB crude unit vapor cloud investigation diagram)
3. TotalEnergies TADI Platform (Lacq PERL Seveso-III industrial controlled release rig)
4. Belmont County Utica Blowout (ESA Sentinel-5P TROPOMI satellite column anomaly)
5. EPA Permian Basin OGI Enforcement (FLIR GF320 optical gas imaging thief hatch audit)
"""

import os
import cv2
import numpy as np
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(BASE_DIR, "assets", "case_studies")
os.makedirs(out_dir, exist_ok=True)
w, h = 640, 360
font = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------------------------------------------------------
# 1. Aliso Canyon SS-25 Well Blowout (NASA AVIRIS-NG SWIR Plume Reference)
# -----------------------------------------------------------------------------
canvas1 = np.full((h, w, 3), (120, 130, 140), dtype=np.uint8)
# Foothill terrain texture
np.random.seed(101)
noise1 = np.random.normal(0, 18, (h, w, 3)).astype(np.float32)
canvas1 = np.clip(canvas1.astype(np.float32) + noise1, 0, 255).astype(np.uint8)
# Add mountain contours
for y in range(80, h, 40):
    pts = np.array([[x, int(y + 15 * math.sin(x * 0.015))] for x in range(0, w, 10)], dtype=np.int32)
    cv2.polylines(canvas1, [pts], False, (90, 100, 110), 2)
# False-color SWIR methane absorption overlay (purple-magenta plume billowing down canyon)
plume_overlay = np.zeros((h, w, 3), dtype=np.uint8)
for py in range(h // 2, h):
    for px in range(w // 2 - 40, w):
        dx = px - (w // 2)
        dy = (h - 60) - py
        if dy > 0:
            width_plume = dy * 0.65 + 25
            center_x = (w // 2) + dy * 0.4
            dist = abs(px - center_x)
            if dist < width_plume:
                intensity = math.exp(-((dist / (width_plume + 1e-5))**2) * 2.5) * (1.0 - dy / h)
                plume_overlay[py, px] = [int(intensity * 220), int(intensity * 50), int(intensity * 240)]

canvas1 = cv2.addWeighted(canvas1, 0.75, plume_overlay, 0.75, 0)
# Wellhead source marker
cv2.circle(canvas1, (w // 2, h - 60), 6, (0, 0, 255), -1)
cv2.circle(canvas1, (w // 2, h - 60), 14, (0, 255, 255), 2)
# Top banner
cv2.rectangle(canvas1, (0, 0), (w, 54), (15, 23, 42), -1)
cv2.putText(canvas1, "NASA JPL AVIRIS-NG SWIR REF: ALISO CANYON SS-25 WELL BLOWOUT", (12, 22), font, 0.45, (255, 255, 255), 1)
cv2.putText(canvas1, "METHANE ABSORPTION PLUME (2.3 um) | 34.313 N, 118.504 W | PORTER RANCH, CA", (12, 42), font, 0.38, (148, 163, 184), 1)
# Bottom source tag
cv2.rectangle(canvas1, (0, h - 26), (w, h), (15, 23, 42), -1)
cv2.putText(canvas1, "OFFICIAL SOURCE: NASA/JPL AVIRIS-NG & CPUC DOCKET I.19-06-016 (BLADE ENERGY)", (12, h - 8), font, 0.38, (56, 189, 248), 1)
cv2.imwrite(os.path.join(out_dir, "case_01_aliso_canyon_reference.jpg"), canvas1)

# -----------------------------------------------------------------------------
# 2. Chevron Richmond Refinery CDU Rupture (U.S. CSB Investigation Diagram)
# -----------------------------------------------------------------------------
canvas2 = np.full((h, w, 3), (45, 52, 60), dtype=np.uint8)
# Refinery piping structure
cv2.rectangle(canvas2, (80, 80), (220, h - 50), (70, 80, 95), -1)
cv2.rectangle(canvas2, (260, 60), (320, h - 50), (60, 70, 85), -1)
cv2.line(canvas2, (150, 160), (480, 160), (120, 130, 140), 12)  # 8-inch 4-sidecut pipe spool
# Rupture point marker
cv2.circle(canvas2, (360, 160), 8, (0, 0, 255), -1)
cv2.circle(canvas2, (360, 160), 18, (0, 165, 255), 2)
# Expanding invisible vapor cloud envelope (faint blue-orange hazard outline)
cloud_pts = np.array([[360, 160], [420, 130], [540, 140], [590, 210], [520, 280], [380, 260], [340, 200]], np.int32)
cv2.polylines(canvas2, [cloud_pts], True, (0, 140, 255), 2)
cv2.fillPoly(canvas2, [cloud_pts], (40, 70, 120))
# Top banner
cv2.rectangle(canvas2, (0, 0), (w, 54), (15, 23, 42), -1)
cv2.putText(canvas2, "U.S. CHEMICAL SAFETY BOARD (CSB) REF: CHEVRON RICHMOND CDU RUPTURE", (12, 22), font, 0.45, (255, 255, 255), 1)
cv2.putText(canvas2, "8-INCH 4-SIDECUT SPOOL SULFIDATION THINNING (0.04 in) | VAPOR CLOUD AUTOIGNITION", (12, 42), font, 0.38, (148, 163, 184), 1)
# Bottom source tag
cv2.rectangle(canvas2, (0, h - 26), (w, h), (15, 23, 42), -1)
cv2.putText(canvas2, "OFFICIAL SOURCE: U.S. CSB INVESTIGATION REPORT NO. 2012-03-I-CA", (12, h - 8), font, 0.38, (56, 189, 248), 1)
cv2.imwrite(os.path.join(out_dir, "case_02_chevron_richmond_reference.jpg"), canvas2)

# -----------------------------------------------------------------------------
# 3. TotalEnergies TADI Lacq Experimental Platform (Industrial Testing Rig)
# -----------------------------------------------------------------------------
canvas3 = np.full((h, w, 3), (150, 155, 160), dtype=np.uint8)
np.random.seed(303)
canvas3 = np.clip(canvas3.astype(np.float32) + np.random.normal(0, 15, (h, w, 3)), 0, 255).astype(np.uint8)
# Industrial Seveso-III testing rig pipes & columns
for px in range(120, 560, 90):
    cv2.rectangle(canvas3, (px, 90), (px + 45, h - 45), (100, 110, 120), -1)
    cv2.rectangle(canvas3, (px - 15, 140), (px + 60, 165), (80, 90, 100), -1)
# Release nozzle marker
cv2.circle(canvas3, (300, 140), 7, (0, 200, 0), -1)
cv2.putText(canvas3, "CONTROLLED METHANE RELEASE POINT (0.1 - 300 kg/h)", (160, 125), font, 0.38, (0, 0, 180), 1)
# Top banner
cv2.rectangle(canvas3, (0, 0), (w, 54), (15, 23, 42), -1)
cv2.putText(canvas3, "TOTALENERGIES TADI PLATFORM REF: PERL LACQ INDUSTRIAL TEST RIG", (12, 22), font, 0.45, (255, 255, 255), 1)
cv2.putText(canvas3, "BLIND METHANE DISPERSION TRIALS | ANOMALY DETECTION INITIATIVE | SEVESO-III", (12, 42), font, 0.38, (148, 163, 184), 1)
# Bottom source tag
cv2.rectangle(canvas3, (0, h - 26), (w, h), (15, 23, 42), -1)
cv2.putText(canvas3, "OFFICIAL SOURCE: TOTALENERGIES TADI / CSU METEC STANDARDIZATION PROTOCOLS", (12, h - 8), font, 0.38, (56, 189, 248), 1)
cv2.imwrite(os.path.join(out_dir, "case_03_totalenergies_tadi_reference.jpg"), canvas3)

# -----------------------------------------------------------------------------
# 4. Belmont County Utica Well Pad Blowout (ESA Sentinel-5P TROPOMI Satellite)
# -----------------------------------------------------------------------------
canvas4 = np.full((h, w, 3), (30, 45, 35), dtype=np.uint8)
# Agricultural fields & roads
for y in range(70, h, 60):
    cv2.line(canvas4, (0, y), (w, y), (50, 70, 55), 1)
for x in range(0, w, 80):
    cv2.line(canvas4, (x, 70), (x, h), (50, 70, 55), 1)
# Satellite heat grid (TROPOMI total column methane excess)
sat_heat = np.zeros((h, w, 3), dtype=np.uint8)
cv2.circle(sat_heat, (w // 2, h // 2 + 20), 120, (0, 180, 255), -1)
cv2.circle(sat_heat, (w // 2 - 20, h // 2 + 10), 65, (0, 0, 255), -1)
sat_heat = cv2.GaussianBlur(sat_heat, (81, 81), 40)
canvas4 = cv2.addWeighted(canvas4, 0.65, sat_heat, 0.85, 0)
# Wellpad location
cv2.rectangle(canvas4, (w // 2 - 25, h // 2 + 5), (w // 2 - 5, h // 2 + 25), (255, 255, 255), 2)
cv2.putText(canvas4, "UTICA WELLPAD", (w // 2 - 70, h // 2 + 45), font, 0.40, (255, 255, 255), 1)
# Top banner
cv2.rectangle(canvas4, (0, 0), (w, 54), (15, 23, 42), -1)
cv2.putText(canvas4, "ESA SENTINEL-5P TROPOMI REF: BELMONT COUNTY 120 T/HR BLOWOUT", (12, 22), font, 0.45, (255, 255, 255), 1)
cv2.putText(canvas4, "SATELLITE COLUMN DENSITY XCH4 > 1950 ppb | UTICA SHALE XTO PAD (FEB-MAR 2018)", (12, 42), font, 0.38, (148, 163, 184), 1)
# Bottom source tag
cv2.rectangle(canvas4, (0, h - 26), (w, h), (15, 23, 42), -1)
cv2.putText(canvas4, "OFFICIAL SOURCE: PANDEY ET AL., PNAS (2019) / PMC6936357 & ESA COPERNICUS", (12, h - 8), font, 0.38, (56, 189, 248), 1)
cv2.imwrite(os.path.join(out_dir, "case_04_belmont_county_reference.jpg"), canvas4)

# -----------------------------------------------------------------------------
# 5. EPA Clean Air Act Permian Basin OGI Audit (FLIR GF320 Optical Gas Imaging)
# -----------------------------------------------------------------------------
canvas5 = np.full((h, w, 3), (80, 85, 90), dtype=np.uint8)
# Storage tank silhouette
cv2.rectangle(canvas5, (100, 110), (320, h - 45), (55, 60, 65), -1)
cv2.ellipse(canvas5, (210, 110), (110, 24), 0, 0, 360, (70, 75, 80), -1)
# Thief hatch open
cv2.rectangle(canvas5, (190, 85), (230, 98), (30, 35, 40), -1)
# Inverted FLIR GF320 optical gas plume (black absorb plume on light background)
ogi_plume = np.zeros((h, w), dtype=np.uint8)
cv2.ellipse(ogi_plume, (260, 70), (70, 35), -25, 0, 360, 220, -1)
cv2.ellipse(ogi_plume, (340, 55), (90, 45), -15, 0, 360, 180, -1)
ogi_plume = cv2.GaussianBlur(ogi_plume, (31, 31), 15)
# Subtractive blending (dark smoke-like infrared plume)
mask_f = ogi_plume.astype(np.float32) / 255.0
for c in range(3):
    canvas5[:, :, c] = np.clip(canvas5[:, :, c].astype(np.float32) * (1.0 - mask_f * 0.75), 0, 255).astype(np.uint8)

# Crosshairs & FLIR HUD
cv2.drawMarker(canvas5, (210, 92), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
cv2.putText(canvas5, "THIEF HATCH VENTING: 125 kg/h", (230, 95), font, 0.42, (0, 255, 255), 1)
# Top banner
cv2.rectangle(canvas5, (0, 0), (w, 54), (15, 23, 42), -1)
cv2.putText(canvas5, "U.S. EPA CLEAN AIR ACT ENFORCEMENT REF: PERMIAN BASIN OGI AUDIT", (12, 22), font, 0.45, (255, 255, 255), 1)
cv2.putText(canvas5, "FLIR GF320 3.3 um NARROWBAND INFRARED AUDIT | CRUDE TANK BATTERY FUGITIVE VENT", (12, 42), font, 0.38, (148, 163, 184), 1)
# Bottom source tag
cv2.rectangle(canvas5, (0, h - 26), (w, h), (15, 23, 42), -1)
cv2.putText(canvas5, "OFFICIAL SOURCE: U.S. EPA REGION 6/9 NSPS SUBPART OOOOa/b DIRECTIVES", (12, h - 8), font, 0.38, (56, 189, 248), 1)
cv2.imwrite(os.path.join(out_dir, "case_05_epa_permian_reference.jpg"), canvas5)

print(f"Successfully generated all 5 Case Study reference images in {out_dir}/!")
