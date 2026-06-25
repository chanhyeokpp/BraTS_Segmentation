import os
from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler

from config import DATA_DIR


class BraTS2DDataset(Dataset):
    def __init__(self, patient_ids, data_dir=DATA_DIR, tumor_only=True, max_cached_patients=1):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.tumor_only = tumor_only
        self.max_cached_patients = max_cached_patients
        self.patient_cache = OrderedDict()
        self.modalities = ["flair", "t1", "t1ce", "t2"]
        self.samples = []
        self.patient_to_indices = {}
        self._build_index()

    def _build_index(self):
        print("Building slice index...")
        for pid in self.patient_ids:
            seg_path = os.path.join(self.data_dir, pid, f"{pid}_seg.nii.gz")
            seg = nib.load(seg_path).get_fdata(dtype=np.float32)
            for z in range(seg.shape[2]):
                if self.tumor_only and np.sum(seg[:, :, z]) == 0:
                    continue
                sample_idx = len(self.samples)
                self.samples.append((pid, z))
                self.patient_to_indices.setdefault(pid, []).append(sample_idx)
        print(f"Total slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def normalize(self, img):
        mask = img > 0
        if np.any(mask):
            img[mask] = (img[mask] - img[mask].mean()) / (img[mask].std() + 1e-8)
        return img

    def load_patient(self, pid):
        if pid in self.patient_cache:
            patient_data = self.patient_cache.pop(pid)
            self.patient_cache[pid] = patient_data
            return patient_data

        path = os.path.join(self.data_dir, pid)
        patient_data = {}
        for modality in self.modalities:
            img_path = os.path.join(path, f"{pid}_{modality}.nii.gz")
            patient_data[modality] = nib.load(img_path).get_fdata(dtype=np.float32)
        seg_path = os.path.join(path, f"{pid}_seg.nii.gz")
        patient_data["seg"] = nib.load(seg_path).get_fdata(dtype=np.float32)

        if self.max_cached_patients > 0:
            self.patient_cache[pid] = patient_data
            while len(self.patient_cache) > self.max_cached_patients:
                self.patient_cache.popitem(last=False)
        return patient_data

    def __getitem__(self, idx):
        pid, z = self.samples[idx]
        patient_data = self.load_patient(pid)
        channels = []
        for modality in self.modalities:
            slice_2d = patient_data[modality][:, :, z].copy()
            channels.append(self.normalize(slice_2d))
        image = np.stack(channels, axis=0)
        mask = patient_data["seg"][:, :, z]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)


class PatientBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        patient_ids = list(self.dataset.patient_to_indices.keys())
        if self.shuffle:
            patient_ids = np.random.permutation(patient_ids).tolist()
        for pid in patient_ids:
            indices = list(self.dataset.patient_to_indices[pid])
            if self.shuffle:
                indices = np.random.permutation(indices).tolist()
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        total = 0
        for indices in self.dataset.patient_to_indices.values():
            full_batches, remainder = divmod(len(indices), self.batch_size)
            total += full_batches
            if remainder and not self.drop_last:
                total += 1
        return total


def get_split_lists(data_dir=DATA_DIR):
    patients = sorted(
        name for name in os.listdir(data_dir)
        if name.startswith("BraTS2021") and os.path.isdir(os.path.join(data_dir, name))
    )
    train_ids, temp_ids = train_test_split(patients, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    return train_ids, val_ids, test_ids
