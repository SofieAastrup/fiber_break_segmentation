from scipy.ndimage import sobel
import nibabel as nib
import numpy as np
import os
from skimage.filters import threshold_triangle
from skimage.morphology import binary_dilation
from image_lib import get_nii_array, get_large_labels, save_nii
import argparse


def segment(filename, save_path, sobel_axis = None, threshold=None, min_density=None):
    image = get_nii_array(filename)
    print("creating segmentations for your image. This may take a while...")
    if sobel_axis: 
        sobel_image = sobel(image, axis=sobel_axis)
    else:
        sobel_image = sobel(image, 0)
    if threshold:
        sobel_image = sobel_image>threshold
    else:
        t_threshold = threshold_triangle(sobel_image)
        print(f"Using threshold: {t_threshold}")
        sobel_image = sobel_image>t_threshold 
    enhanced_threshold_image = binary_dilation(sobel_image)
    if min_density:
        segmentation, _ = get_large_labels(enhanced_threshold_image, min_density)
    else:
        segmentation, _ = get_large_labels(enhanced_threshold_image, 1000)
    save_nii((segmentation>0).astype(np.uint8), save_path)
    print(f"Finished. Saved to {save_path}.nii.gz")


parser = argparse.ArgumentParser(description="Creates fiber break segmentations for an image using a Sobel filter")
parser.add_argument("image_path", type=str, help="Path to the image")
parser.add_argument("output_name", type=str, help="Name of the output segmented image")
parser.add_argument("--min_density", type=int, help="Minimum denisty of segmented fiber break. Default: 1000")
parser.add_argument("--s", type=int, help="Which axis to apply Sobel filter to. Choose from 0, 1 or 2. Default:0")
parser.add_argument("--t", type=int, help="Threshold value. Determines how clear the edges found with Sobel filter has to be." +
                   "Default: Uses trianle algorith to find threshold.")

args = parser.parse_args()
print(args)

segment(args.image_path, args.output_name, sobel_axis=args.s, threshold=args.t, min_density=args.min_density)