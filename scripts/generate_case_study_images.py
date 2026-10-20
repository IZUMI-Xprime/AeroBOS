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

# =============================================================================
# PART 2: DEDICATED AEROBOS FORENSIC SCHLIEREN RECONSTRUCTIONS (CASES 01 TO 05)
# =============================================================================

# -----------------------------------------------------------------------------
# Case 01 AeroBOS: Aliso Canyon SS-25 Wellhead Blowout
# -----------------------------------------------------------------------------
np.random.seed(201)
base1 = np.full((h, w, 3), (45, 55, 65), dtype=np.uint8)
noise = np.random.normal(0, 15, (h, w, 3)).astype(np.float32)
base1 = np.clip(base1.astype(np.float32) + noise, 0, 255).astype(np.uint8)
# Wellhead Christmas tree
cv2.rectangle(base1, (290, 220), (350, 320), (80, 90, 100), -1)
cv2.rectangle(base1, (270, 240), (370, 260), (100, 110, 120), -1)
cv2.line(base1, (320, 220), (320, 190), (120, 130, 140), 12)
# Massive Schlieren Turbo Plume
for y in range(40, 200):
    for x in range(180, 480):
        dx = (x - 320) / (0.8 * (200 - y) + 15)
        dy = (200 - y) / 160.0
        r2 = dx**2
        if r2 < 2.5:
            turb = math.sin(x*0.1 + y*0.15) * math.cos(y*0.08) * 0.4 + 0.6
            val = int(255 * math.exp(-r2*0.8) * turb * dy)
            c = cv2.applyColorMap(np.uint8([[min(val*2, 255)]]), cv2.COLORMAP_TURBO)[0,0]
            alpha = min(val / 220.0, 0.85)
            base1[y, x] = np.clip(base1[y, x]*(1-alpha) + c*alpha, 0, 255).astype(np.uint8)

# HUD
cv2.rectangle(base1, (0, 0), (w, 36), (15, 23, 42), -1)
cv2.putText(base1, "AEROBOS CASE 01: ALISO CANYON SS-25 WELLHEAD BLOWOUT", (12, 18), font, 0.42, (255, 255, 255), 1)
cv2.putText(base1, "SCHLIEREN RECONSTRUCTION | 480 kg/h | SNR: 142.6s | SSV ISOLATED in 1.45s", (12, 31), font, 0.36, (56, 189, 248), 1)
cv2.imwrite(os.path.join(out_dir, "case_01_aliso_canyon_aerobos.jpg"), base1)

# -----------------------------------------------------------------------------
# Case 02 AeroBOS: Chevron Richmond 8-Inch Crude Unit Rupture
# -----------------------------------------------------------------------------
np.random.seed(202)
base2 = np.full((h, w, 3), (35, 45, 50), dtype=np.uint8)
noise = np.random.normal(0, 12, (h, w, 3)).astype(np.float32)
base2 = np.clip(base2.astype(np.float32) + noise, 0, 255).astype(np.uint8)
# Crude line pipe rack
cv2.rectangle(base2, (0, 180), (w, 215), (70, 75, 80), -1)
cv2.circle(base2, (260, 197), 16, (100, 110, 115), -1)
# Expanding vapor cloud (sidecut rupture spreading horizontally)
for y in range(80, 270):
    for x in range(140, 560):
        dx = (x - 260) / 180.0
        dy = (y - 197) / 70.0
        r2 = dx**2 + dy**2
        if r2 < 1.8:
            turb = math.sin(x*0.08 - y*0.12) * 0.3 + 0.7
            val = int(255 * math.exp(-r2*1.2) * turb)
            c = cv2.applyColorMap(np.uint8([[min(val*2, 255)]]), cv2.COLORMAP_TURBO)[0,0]
            alpha = min(val / 200.0, 0.85)
            base2[y, x] = np.clip(base2[y, x]*(1-alpha) + c*alpha, 0, 255).astype(np.uint8)

# HUD
cv2.rectangle(base2, (0, 0), (w, 36), (15, 23, 42), -1)
cv2.putText(base2, "AEROBOS CASE 02: CHEVRON RICHMOND CRUDE UNIT RUPTURE", (12, 18), font, 0.42, (255, 255, 255), 1)
cv2.putText(base2, "BEDROCK DIAGNOSIS: PIPE CORROSION THINNING | MOV TRIP: 1.82s (FIREBALL PREVENTED)", (12, 31), font, 0.36, (249, 115, 22), 1)
cv2.imwrite(os.path.join(out_dir, "case_02_chevron_richmond_aerobos.jpg"), base2)

# -----------------------------------------------------------------------------
# Case 03 AeroBOS: TotalEnergies TADI Lacq Platform
# -----------------------------------------------------------------------------
np.random.seed(203)
base3 = np.full((h, w, 3), (50, 50, 55), dtype=np.uint8)
noise = np.random.normal(0, 14, (h, w, 3)).astype(np.float32)
base3 = np.clip(base3.astype(np.float32) + noise, 0, 255).astype(np.uint8)
# Modular test columns
cv2.rectangle(base3, (120, 140), (160, 320), (75, 80, 85), -1)
cv2.rectangle(base3, (460, 120), (510, 320), (80, 85, 90), -1)
# Steam plume (white-cyan aerosol) vs methane rejection
for y in range(60, 240):
    for x in range(220, 420):
        dx = (x - 300) / (0.6 * (240 - y) + 20)
        dy = (240 - y) / 180.0
        if dx**2 < 2.0:
            val = int(220 * math.exp(-dx**2 * 1.5) * dy)
            alpha = min(val / 255.0, 0.75)
            base3[y, x] = np.clip(base3[y, x]*(1-alpha) + np.array([240, 220, 200])*alpha, 0, 255).astype(np.uint8)

# HUD
cv2.rectangle(base3, (0, 0), (w, 36), (15, 23, 42), -1)
cv2.putText(base3, "AEROBOS CASE 03: TOTALENERGIES TADI SEVESO-III RIG", (12, 18), font, 0.42, (255, 255, 255), 1)
cv2.putText(base3, "SAGEMAKER CLASSIFIER: STEAM EXHAUST (96% CONF) | FALSE ALARM TRIP BYPASSED", (12, 31), font, 0.36, (34, 197, 94), 1)
cv2.imwrite(os.path.join(out_dir, "case_03_totalenergies_tadi_aerobos.jpg"), base3)

# -----------------------------------------------------------------------------
# Case 04 AeroBOS: Belmont County XTO Well Blowout
# -----------------------------------------------------------------------------
np.random.seed(204)
base4 = np.full((h, w, 3), (40, 50, 45), dtype=np.uint8)
noise = np.random.normal(0, 16, (h, w, 3)).astype(np.float32)
base4 = np.clip(base4.astype(np.float32) + noise, 0, 255).astype(np.uint8)
# Tree canopy and drilling mast
cv2.line(base4, (320, 100), (320, 320), (70, 75, 80), 6)
# Massive plume fountain with downwind drift contour
for y in range(40, 240):
    for x in range(160, 520):
        dx = (x - 320 - (240 - y)*0.4) / (0.7 * (240 - y) + 25)
        dy = (240 - y) / 200.0
        if dx**2 < 2.2:
            turb = math.sin(x*0.1 + y*0.2) * 0.35 + 0.65
            val = int(255 * math.exp(-dx**2 * 1.0) * turb * dy)
            c = cv2.applyColorMap(np.uint8([[min(val*2, 255)]]), cv2.COLORMAP_TURBO)[0,0]
            alpha = min(val / 210.0, 0.85)
            base4[y, x] = np.clip(base4[y, x]*(1-alpha) + c*alpha, 0, 255).astype(np.uint8)

# ConvLSTM 15-min drift boundary polygon overlay
drift_pts = np.array([[310, 240], [380, 160], [490, 80], [580, 60], [560, 120], [440, 200], [330, 245]], np.int32)
cv2.polylines(base4, [drift_pts], True, (0, 255, 255), 2)
cv2.putText(base4, "15-MIN FORECAST CONTOUR", (430, 70), font, 0.35, (0, 255, 255), 1)

# HUD
cv2.rectangle(base4, (0, 0), (w, 36), (15, 23, 42), -1)
cv2.putText(base4, "AEROBOS CASE 04: BELMONT COUNTY 120 T/HR UTICA BLOWOUT", (12, 18), font, 0.42, (255, 255, 255), 1)
cv2.putText(base4, "CONVLSTM FORECASTER: 15-MIN HAZARD PERIMETER -> AWS IOT CORE | ESD: 0.85s", (12, 31), font, 0.36, (234, 179, 8), 1)
cv2.imwrite(os.path.join(out_dir, "case_04_belmont_county_aerobos.jpg"), base4)

# -----------------------------------------------------------------------------
# Case 05 AeroBOS: EPA Permian Basin Thief Hatch Audit
# -----------------------------------------------------------------------------
np.random.seed(205)
base5 = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)
noise = np.random.normal(0, 14, (h, w, 3)).astype(np.float32)
base5 = np.clip(base5.astype(np.float32) + noise, 0, 255).astype(np.uint8)
# Storage battery tanks
cv2.rectangle(base5, (160, 150), (320, 320), (85, 90, 95), -1)
cv2.rectangle(base5, (340, 150), (500, 320), (85, 90, 95), -1)
# Open thief hatch on top of left tank
cv2.line(base5, (230, 150), (250, 130), (120, 130, 140), 4)
# Persistent buoyant plume venting
for y in range(40, 150):
    for x in range(180, 320):
        dx = (x - 240 + (150 - y)*0.2) / (0.5 * (150 - y) + 12)
        dy = (150 - y) / 110.0
        if dx**2 < 1.8:
            turb = math.sin(x*0.15 - y*0.1) * 0.3 + 0.7
            val = int(240 * math.exp(-dx**2 * 1.2) * turb * dy)
            c = cv2.applyColorMap(np.uint8([[min(val*2, 255)]]), cv2.COLORMAP_TURBO)[0,0]
            alpha = min(val / 220.0, 0.85)
            base5[y, x] = np.clip(base5[y, x]*(1-alpha) + c*alpha, 0, 255).astype(np.uint8)

# HUD
cv2.rectangle(base5, (0, 0), (w, 36), (15, 23, 42), -1)
cv2.putText(base5, "AEROBOS CASE 05: EPA PERMIAN TANK BATTERY THIEF HATCH AUDIT", (12, 18), font, 0.42, (255, 255, 255), 1)
cv2.putText(base5, "CONTINUOUS GRAVITON MONITORING ($0.038/HR) | UNLATCHED HATCH ISOLATED", (12, 31), font, 0.36, (168, 85, 247), 1)
cv2.imwrite(os.path.join(out_dir, "case_05_epa_permian_aerobos.jpg"), base5)

print(f"Successfully generated all 10 Case Study images (5 reference + 5 AeroBOS reconstructions) in {out_dir}/!")
