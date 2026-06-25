import os

import nibabel as nib

from config import DATA_DIR


patients = sorted(
    name for name in os.listdir(DATA_DIR)
    if name.startswith("BraTS2021") and os.path.isdir(os.path.join(DATA_DIR, name))
)
print(f"Patients: {len(patients)}")

shapes = set()
for patient_id in patients[:50]:
    flair_path = os.path.join(DATA_DIR, patient_id, f"{patient_id}_flair.nii.gz")
    shapes.add(nib.load(flair_path).shape)

print("Detected shapes:")
print(shapes)
