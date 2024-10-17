#!/usr/bin/env python

"""
eval.py

This module provides functions to compute various error metrics between 3D predicted and reference 
data arrays. The primary function, `all_metrics()`, returns a dictionary of all computed metrics.

Metrics included: RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI, GXE.

Example:
    >>> import numpy as np
    >>> import metrics
    >>> pred_data = np.random.rand(100, 100, 100)
    >>> ref_data = np.random.rand(100, 100, 100)
    >>> roi = np.random.randint(0, 2, size=(100, 100, 100), dtype=bool)
    >>> metrics = metrics.all_metrics(pred_data, ref_data, roi)

Authors: Boyi Du <boyi.du@uq.net.au>, Ashley Stewart <ashley.stewart@uq.edu.au>

"""

import json
import argparse
import os
import numpy as np
import csv
import nibabel as nib

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_mutual_information
from skimage.measure import pearson_corr_coeff
from skimage.measure import shannon_entropy
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_laplace

def calculate_gradient_magnitude(pred_data):
    gradients = np.gradient(pred_data)
    grad_magnitude = np.sqrt(sum([g**2 for g in gradients]))
    return grad_magnitude.mean(), grad_magnitude.std()

def calculate_total_variation(image, weight=0.1):
    denoised_image = denoise_tv_chambolle(image, weight=weight)
    tv_norm = np.sum(np.abs(image - denoised_image))  # Total variation norm
    return tv_norm

def calculate_entropy(pred_data):
    return shannon_entropy(pred_data)

def calculate_cnr(region1, region2):
    mean_diff = np.abs(np.mean(region1) - np.mean(region2))
    noise = np.std(region1) + np.std(region2)
    if noise == 0:
        return float('inf')  # Handle the case where noise is zero
    return mean_diff / noise

def calculate_snr(pred_data, roi_foreground, roi_background):
    signal = np.mean(pred_data[roi_foreground == 1])  # Mean intensity in the signal region
    noise = np.std(pred_data[roi_background == 1])  # Standard deviation in the background
    if noise == 0:
        return float('inf')  # Handle the case where noise is zero
    return signal / noise

def calculate_edge_strength(pred_data):
    edges = gaussian_laplace(pred_data, sigma=1.5)
    return np.var(edges)

def calculate_rmse(pred_data, ref_data):
    """
    Calculate the Root Mean Square Error (RMSE) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated RMSE value.

    """
    mse = mean_squared_error(pred_data, ref_data)
    rmse = np.sqrt(mse)
    return rmse

def calculate_nrmse(pred_data, ref_data):
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated NRMSE value.

    References
    ----------
    .. [1] https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/metrics/simple_metrics.py#L50-L108
    """
    rmse = calculate_rmse(pred_data, ref_data)
    nrmse = rmse * np.sqrt(len(ref_data)) / np.linalg.norm(ref_data) * 100 # Frobenius norm
    return nrmse

def calculate_hfen(pred_data, ref_data):
    """
    Calculate the High Frequency Error Norm (HFEN) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated HFEN value.
    References
    ----------
    .. [1] https://doi.org/10.1002/mrm.26830

    """
    LoG_pred = gaussian_laplace(pred_data, sigma = 1.5)
    LoG_ref = gaussian_laplace(ref_data, sigma = 1.5)
    hfen = np.linalg.norm(LoG_ref - LoG_pred)/np.linalg.norm(LoG_ref)
    return hfen

def calculate_xsim(pred_data, ref_data, data_range=None):
    """
    Calculate the structural similarity (XSIM) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.
    data_range : float
        Expected data range.

    Returns
    -------
    float
        The calculated structural similarity value.

    References
    ----------
    .. [1] Milovic, C., et al. (2024). XSIM: A structural similarity index measure optimized for MRI QSM. Magnetic Resonance in Medicine. doi:10.1002/mrm.30271
    """
    if not data_range: data_range = ref_data.max() - ref_data.min()
    xsim = structural_similarity(pred_data, ref_data, win_size=3, K1=0.01, K2=0.001, data_range=data_range)
    return xsim

def calculate_mad(pred_data, ref_data):
    """
    Calculate the Mean Absolute Difference (MAD) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated MAD value.

    """
    mad = np.mean(np.abs(pred_data - ref_data))
    return mad

def calculate_gxe(pred_data, ref_data):
    """
    Calculate the gradient difference error (GXE) between the predicted and reference data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray
        Reference data as a numpy array.

    Returns
    -------
    float
        The calculated GXE value.

    """
    gxe = np.sqrt(np.mean((np.array(np.gradient(pred_data)) - np.array(np.gradient(ref_data))) ** 2))
    return gxe


def get_bounding_box(roi):
    """
    Calculate the bounding box of a 3D region of interest (ROI).

    Parameters
    ----------
    roi : numpy.ndarray
        A 3D numpy array representing a binary mask of the ROI,
        where 1 indicates an object of interest and 0 elsewhere.

    Returns
    -------
    bbox : tuple
        A tuple of slice objects representing the bounding box of the ROI. This can be 
        directly used to slice numpy arrays.

    Example
    -------
    >>> mask = np.random.randint(0, 2, size=(100, 100, 100))
    >>> bbox = get_bounding_box(mask)
    >>> sliced_data = data[bbox]

    Notes
    -----
    The function works by identifying the min and max coordinates of the ROI along 
    each axis. These values are used to generate a tuple of slice objects.
    The function will work for ROIs of arbitrary dimension, not just 3D.
    """
    coords = np.array(roi.nonzero())
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1
    return tuple(slice(min_coords[d], max_coords[d]) for d in range(roi.ndim))


def all_metrics(pred_data, ref_data=None, roi=None, roi_foreground=None, roi_background=None):
    """
    Calculate various error and quality metrics between the predicted data and the reference data (optional).

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    ref_data : numpy.ndarray, optional
        Reference data as a numpy array. If not provided, only quality metrics without a reference are computed.
    roi : numpy.ndarray, optional
        A binary mask defining a region of interest within the data. If not provided, the full extent of pred_data is used.
    roi_foreground : numpy.ndarray, optional
        A binary mask defining the foreground (signal region) for SNR calculation. Required for SNR.
    roi_background : numpy.ndarray, optional
        A binary mask defining the background (noise region) for SNR calculation. Required for SNR.

    Returns
    -------
    dict
        A dictionary of calculated metrics, including RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI, GXE, and quality measures
        such as gradient magnitude, total variation, entropy, CNR, SNR, and edge strength.
    """
    d = dict()

    # Define the region of interest if not provided
    if roi is None:
        roi = np.array(pred_data != 0, dtype=bool)

    bbox = get_bounding_box(roi)
    pred_data = pred_data[bbox] * roi
    roi = roi[bbox]

    if ref_data is not None:
        ref_data = ref_data[bbox] * roi

        # Handle NaN or zero variance cases
        if np.isnan(pred_data).any() or np.isnan(ref_data).any():
            print("Input arrays contain NaN values.")
        if np.std(pred_data) == 0 or np.std(ref_data) == 0:
            print(np.std(pred_data))
            print(np.std(ref_data))
            print("One of the input arrays has no variance.")

        # Metrics requiring ground truth (reference data)
        d['RMSE'] = calculate_rmse(pred_data[roi], ref_data[roi])
        d['NRMSE'] = calculate_nrmse(pred_data[roi], ref_data[roi])
        d['HFEN'] = calculate_hfen(pred_data, ref_data)
        d['MAD'] = calculate_mad(pred_data[roi], ref_data[roi])
        d['XSIM'] = calculate_xsim(pred_data, ref_data)
        d['CC'] = pearson_corr_coeff(pred_data[roi], ref_data[roi])
        d['NMI'] = normalized_mutual_information(pred_data[roi], ref_data[roi])
        d['GXE'] = calculate_gxe(pred_data, ref_data)

    # Quality measures that do not require ground truth
    d['Gradient Mean'], d['Gradient Std'] = calculate_gradient_magnitude(pred_data)
    d['Total Variation'] = calculate_total_variation(pred_data)
    d['Entropy'] = calculate_entropy(pred_data)
    d['Edge Strength'] = calculate_edge_strength(pred_data)

    # CNR calculation if roi_foreground and roi_background are provided
    if not (roi_foreground == roi_background == None):
        d['CNR'] = calculate_cnr(pred_data[roi_foreground], pred_data[roi_background])

    # SNR calculation if roi_foreground and roi_background are provided
    if not (roi_foreground == roi_background == None):
        d['SNR'] = calculate_snr(pred_data, roi_foreground, roi_background)

    return d

def save_as_csv(metrics_dict, filepath):
    """
    Save the metrics as a CSV file.

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics.
    filepath : str
        The path to the file to save the results.
    """
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics_dict.items():
            writer.writerow([key, value])

def save_as_markdown(metrics_dict, filepath):
    """
    Save the metrics as a markdown table.

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics.
    filepath : str
        The path to the file to save the results.
    """
    with open(filepath, 'w') as file:
        file.write("| Metric | Value |\n")
        file.write("|--------|-------|\n")
        for key, value in metrics_dict.items():
            if isinstance(value, tuple) and len(value) == 2:  # Assuming it's the PearsonRResult
                file.write(f"| {key} correlation | {value[0]:.6f} |\n")
                file.write(f"| {key} p-value | {value[1]:.6f} |\n")
            else:
                file.write(f"| {key} | {value:.6f} |\n")

def save_as_json(metrics_dict, filepath):
    """
    Save the metrics as a JSON file.

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics.
    filepath : str
        The path to the file to save the results.
    """
    with open(filepath, 'w') as file:
        json.dump(metrics_dict, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Compute metrics for 3D images.')
    parser.add_argument('--ground_truth', type=str, help='Path to the ground truth NIFTI image (optional).')
    parser.add_argument('--estimate', type=str, required=True, help='Path to the reconstructed NIFTI image.')
    parser.add_argument('--roi', type=str, help='Path to the ROI NIFTI image (optional).')
    parser.add_argument('--roi_foreground', type=str, help='Path to the ROI foreground (signal region) NIFTI image (optional, required for SNR/CNR).')
    parser.add_argument('--roi_background', type=str, help='Path to the ROI background (noise region) NIFTI image (optional, required for SNR/CNR).')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save metrics.')
    args = parser.parse_args()

    # Load reconstructed image
    print("[INFO] Loading reconstructed image...")
    recon_img = nib.load(args.estimate).get_fdata()

    # Load ground truth image (if provided)
    if args.ground_truth:
        print("[INFO] Loading ground truth image...")
        gt_img = nib.load(args.ground_truth).get_fdata()
    else:
        gt_img = None

    # Load ROI (if provided)
    if args.roi:
        print("[INFO] Loading ROI image...")
        roi_img = np.array(nib.load(args.roi).get_fdata(), dtype=bool)
    else:
        roi_img = None

    # Load ROI foreground and background (if provided)
    if args.roi_foreground:
        print("[INFO] Loading ROI foreground (signal region) image...")
        roi_foreground_img = np.array(nib.load(args.roi_foreground).get_fdata(), dtype=bool)
    else:
        roi_foreground_img = None

    if args.roi_background:
        print("[INFO] Loading ROI background (noise region) image...")
        roi_background_img = np.array(nib.load(args.roi_background).get_fdata(), dtype=bool)
    else:
        roi_background_img = None

    # Compute metrics
    print("[INFO] Computing metrics...")
    metrics = all_metrics(recon_img, ref_data=gt_img, roi=roi_img, 
                          roi_foreground=roi_foreground_img, roi_background=roi_background_img)

    # Save metrics
    print(f"[INFO] Saving results to {args.output_dir}...")
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    md_path = os.path.join(args.output_dir, 'metrics.md')
    json_path = os.path.join(args.output_dir, 'metrics.json')

    save_as_csv(metrics, csv_path)
    save_as_markdown(metrics, md_path)
    save_as_json(metrics, json_path)

    print(f"[INFO] Metrics saved to {csv_path}, {md_path}, and {json_path}")


if __name__ == "__main__":
    main()

