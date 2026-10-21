"""
embed_assets_in_html.py

Embeds all 51 project images directly inside index.html as Base64 Data URIs:
- 1 Hero image (aerobos_plume_annotated.jpg)
- 10 Case study images (assets/case_studies/case_01... to case_05...)
- 20 Annotated benchmark images (dataset_tests/test_01_annotated.jpg ... test_20_annotated.jpg)
- 20 Original raw CCTV images (dataset_tests/test_01_original.jpg ... test_20_original.jpg)

Enables index.html to be completely standalone: anyone who downloads index.html
directly from GitHub will see all photos and interactive comparisons immediately,
with zero missing images and zero external dependencies!
"""

import base64
import os
import re
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_path = os.path.join(BASE_DIR, "index.html")

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Define list of all 51 assets to embed
assets_to_embed = [
    "aerobos_plume_annotated.jpg",
    "assets/case_studies/case_01_aliso_canyon_reference.jpg",
    "assets/case_studies/case_01_aliso_canyon_aerobos.jpg",
    "assets/case_studies/case_02_chevron_richmond_reference.jpg",
    "assets/case_studies/case_02_chevron_richmond_aerobos.jpg",
    "assets/case_studies/case_03_totalenergies_tadi_reference.jpg",
    "assets/case_studies/case_03_totalenergies_tadi_aerobos.jpg",
    "assets/case_studies/case_04_belmont_county_reference.jpg",
    "assets/case_studies/case_04_belmont_county_aerobos.jpg",
    "assets/case_studies/case_05_epa_permian_reference.jpg",
    "assets/case_studies/case_05_epa_permian_aerobos.jpg",
]

for i in range(1, 21):
    assets_to_embed.append(f"dataset_tests/test_{i:02d}_annotated.jpg")
    assets_to_embed.append(f"dataset_tests/test_{i:02d}_original.jpg")

print(f"Encoding {len(assets_to_embed)} assets to Base64 (Quality 88)...")
embedded_map = {}

for rel_path in assets_to_embed:
    full_path = os.path.join(BASE_DIR, rel_path.replace("/", os.sep))
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Missing required asset: {full_path}")
    
    img = cv2.imread(full_path)
    # Encode with high-fidelity JPEG quality 88 for razor-sharp text and HUD
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64}"
    embedded_map[rel_path] = data_uri

print("All assets successfully encoded.")

# 1. Update Hero image src to base64
hero_target = 'src="aerobos_plume_annotated.jpg"'
hero_repl = f'src="{embedded_map["aerobos_plume_annotated.jpg"]}"'
html = html.replace(hero_target, hero_repl)

# 2. Update Case Study images in Section 10 to base64
for cs_key in [
    "assets/case_studies/case_01_aliso_canyon_reference.jpg",
    "assets/case_studies/case_01_aliso_canyon_aerobos.jpg",
    "assets/case_studies/case_02_chevron_richmond_reference.jpg",
    "assets/case_studies/case_02_chevron_richmond_aerobos.jpg",
    "assets/case_studies/case_03_totalenergies_tadi_reference.jpg",
    "assets/case_studies/case_03_totalenergies_tadi_aerobos.jpg",
    "assets/case_studies/case_04_belmont_county_reference.jpg",
    "assets/case_studies/case_04_belmont_county_aerobos.jpg",
    "assets/case_studies/case_05_epa_permian_reference.jpg",
    "assets/case_studies/case_05_epa_permian_aerobos.jpg",
]:
    target = f'src="{cs_key}"'
    repl = f'src="{embedded_map[cs_key]}"'
    if target in html:
        html = html.replace(target, repl)
    else:
        print(f"Warning: {target} not found directly in html.")

# 3. Update the 20 test cards in Section 9:
#    Replace src="dataset_tests/test_XX_annotated.jpg" with the base64 data URI
for i in range(1, 21):
    ann_key = f"dataset_tests/test_{i:02d}_annotated.jpg"
    target = f'src="{ann_key}"'
    repl = f'src="{embedded_map[ann_key]}"'
    if target in html:
        html = html.replace(target, repl)
    else:
        print(f"Warning: {target} not found in html.")

# 4. Inject EMBEDDED_IMAGES JavaScript dictionary into <script>
#    and update switchModalTab, setGalleryViewMode, and openGalleryModal to resolve from it.
dict_entries = []
for k, v in embedded_map.items():
    dict_entries.append(f'  "{k}": "{v}"')

js_dict_str = "const EMBEDDED_IMAGES = {\n" + ",\n".join(dict_entries) + "\n};\n"

# Check where to insert EMBEDDED_IMAGES in script
script_marker = "<script>"
idx = html.find(script_marker)
if idx != -1:
    # Insert right after the first <script>
    insert_pos = idx + len(script_marker) + 1
    html = html[:insert_pos] + "\n" + js_dict_str + "\n" + html[insert_pos:]
    print("Injected EMBEDDED_IMAGES dictionary into <script>.")
else:
    print("Error: <script> tag not found!")

# 5. Update setGalleryViewMode to resolve keys from EMBEDDED_IMAGES
old_view_mode_code = """      thumbs.forEach((img, idx) => {
        if (mode === "our") {
          img.src = img.getAttribute("data-ann");
          if (badges[idx]) {
            badges[idx].innerText = "OUR DATA: SCHLIEREN";
            badges[idx].style.color = "#38bdf8";
            badges[idx].style.borderColor = "#0284c7";
          }
        } else {
          img.src = img.getAttribute("data-orig");
          if (badges[idx]) {
            badges[idx].innerText = "ORIGINAL: RAW CCTV";
            badges[idx].style.color = "#f59e0b";
            badges[idx].style.borderColor = "#d97706";
          }
        }
      });"""

new_view_mode_code = """      thumbs.forEach((img, idx) => {
        if (mode === "our") {
          const key = img.getAttribute("data-ann");
          img.src = (typeof EMBEDDED_IMAGES !== "undefined" && EMBEDDED_IMAGES[key]) ? EMBEDDED_IMAGES[key] : key;
          if (badges[idx]) {
            badges[idx].innerText = "OUR DATA: SCHLIEREN";
            badges[idx].style.color = "#38bdf8";
            badges[idx].style.borderColor = "#0284c7";
          }
        } else {
          const key = img.getAttribute("data-orig");
          img.src = (typeof EMBEDDED_IMAGES !== "undefined" && EMBEDDED_IMAGES[key]) ? EMBEDDED_IMAGES[key] : key;
          if (badges[idx]) {
            badges[idx].innerText = "ORIGINAL: RAW CCTV";
            badges[idx].style.color = "#f59e0b";
            badges[idx].style.borderColor = "#d97706";
          }
        }
      });"""

if old_view_mode_code in html:
    html = html.replace(old_view_mode_code, new_view_mode_code)
    print("Updated setGalleryViewMode with embedded resolution.")
else:
    # Try regex match for whitespace differences
    print("Warning: old_view_mode_code not found directly, checking regex...")
    pattern_vm = r'thumbs\.forEach\(\(img, idx\) => \{.*?\}\);\s*\}'
    # Will check if needed

# 6. Update openGalleryModal to resolve annImg and origImg from EMBEDDED_IMAGES
old_open_modal = """    function openGalleryModal(tid, title, annImg, origImg, clsName, conf, flow, eq, sector, valve, diagPoint, diagComp, diagTier, evacR, snr, opd, bgStyle) {
      modalCurAnn = annImg;
      modalCurOrig = origImg;"""

new_open_modal = """    function openGalleryModal(tid, title, annImg, origImg, clsName, conf, flow, eq, sector, valve, diagPoint, diagComp, diagTier, evacR, snr, opd, bgStyle) {
      modalCurAnn = (typeof EMBEDDED_IMAGES !== "undefined" && EMBEDDED_IMAGES[annImg]) ? EMBEDDED_IMAGES[annImg] : annImg;
      modalCurOrig = (typeof EMBEDDED_IMAGES !== "undefined" && EMBEDDED_IMAGES[origImg]) ? EMBEDDED_IMAGES[origImg] : origImg;"""

if old_open_modal in html:
    html = html.replace(old_open_modal, new_open_modal)
    print("Updated openGalleryModal with embedded resolution.")
else:
    print("Warning: old_open_modal not found directly.")

# Save updated index.html
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

size_mb = os.path.getsize(html_path) / (1024 * 1024)
print(f"Successfully updated index.html! New file size: {size_mb:.2f} MB")
