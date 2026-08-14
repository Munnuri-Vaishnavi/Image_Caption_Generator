"""
One-time utility: renames files in samples/ from numeric names (e.g. 23456.jpg)
to readable ones (Image1.jpg, Image2.jpg, ...).

Run once, from the same folder as app.py:
    python rename_samples.py
"""

import os

SAMPLES_DIR = "samples"

files = sorted(
    f for f in os.listdir(SAMPLES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

if not files:
    print(f"No image files found in '{SAMPLES_DIR}/'.")
else:
    for i, old_name in enumerate(files, start=1):
        ext = os.path.splitext(old_name)[1].lower()
        new_name = f"Image{i}{ext}"
        old_path = os.path.join(SAMPLES_DIR, old_name)
        new_path = os.path.join(SAMPLES_DIR, new_name)

        if old_path == new_path:
            continue
        if os.path.exists(new_path):
            print(f"Skipping {old_name} -> {new_name} (target already exists)")
            continue

        os.rename(old_path, new_path)
        print(f"{old_name}  ->  {new_name}")

    print(f"\nDone. Renamed files in '{SAMPLES_DIR}/'.")
