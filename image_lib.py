import nibabel as nib
import numpy as np
from skimage.measure import label

def get_nii_array(path):
    img = nib.load(path)
    img = np.array(img.dataobj).astype(np.float32)
    return img

def save_nii(image, image_name):
    img = nib.Nifti1Image(image, np.eye(4))
    out_path = image_name + '.nii.gz'
    nib.save(img, out_path)

def get_large_labels(seg1, min_density):
    labels, _ = label(seg1, background=0, return_num=True)
    names, counts = np.unique(labels, return_counts=True)
    s_idxs = np.argsort(counts)
    num_breaks = len(counts) -np.argwhere((counts[s_idxs]>=min_density))[0][0]
    filtered_labels = np.where(np.isin(labels, np.flip(names[s_idxs])[:num_breaks]), labels, 0)
    return filtered_labels, num_breaks