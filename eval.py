#!/usr/bin/env python

"""
eval.py

Authors: Boyi Du <boyi.du@uq.net.au>, Ashley Stewart <ashley.stewart@uq.edu.au>

- This module provides functions to compute various metrics given 3D predictions and reference data.
- When reference data are unavailable, a subset of metrics can still be computed.
- The primary function, `all_metrics()`, returns a dictionary of all computed metrics.
    - Metrics included: RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI, GXE.

Example:
    >>> import numpy as np
    >>> import metrics
    >>> pred_data = np.random.rand(100, 100, 100)
    >>> ref_data = np.random.rand(100, 100, 100)
    >>> mask_data = np.random.randint(0, 2, size=(100, 100, 100), dtype=bool)
    >>> segmentation_data = np.random.randint(0, 10, size=(100, 100, 100))
    >>> labels = {0: 'Background', 1: 'Region 1', 2: 'Region 2'}
    >>> metrics = metrics.all_metrics(pred_data, segmentation_data, mask_data, labels, ref_data)
    >>> print(metrics)
    >>> metrics.save_as_csv(metrics, 'metrics.csv')
    >>> metrics.save_as_markdown(metrics, 'metrics.md')
    >>> metrics.save_as_json(metrics, 'metrics.json')

It can also be run as a standalone script to compute metrics for a given set of images:

    $ python metrics.py \
        --ground_truth gt.nii.gz \
        --estimate pred.nii.gz \
        --roi roi.nii.gz \
        --segmentation seg.nii.gz \
        --output_dir output/

"""

import json
import argparse
import os
import numpy as np
import csv
import nibabel as nib

from sklearn.metrics import root_mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_mutual_information
from skimage.measure import pearson_corr_coeff
from skimage.measure import shannon_entropy
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_laplace

from scipy.ndimage import generic_filter
from skimage.morphology import disk, binary_opening, binary_dilation
    
def pad_if_necessary(array, min_size=2):
    """
    Pad the input array with zeros if any of its dimensions are smaller than the minimum size.

    Parameters
    ----------
    array : numpy.ndarray
        Input array to pad.
    min_size : int, optional
        Minimum size for each dimension.

    Returns
    -------
    numpy.ndarray
        Padded array.
    """

    padding = [(0, max(min_size - dim_size, 0)) for dim_size in array.shape]
    return np.pad(array, padding, mode='constant', constant_values=0)

def calculate_gradient_magnitude(pred_data):
    """
    Calculate the gradient magnitude of the input data.
    
    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.

    Returns
    -------
    float
        The mean gradient magnitude.
    float
        The standard deviation of the gradient magnitude.
    """
    
    # Pad the predicted data if necessary
    pred_data_padded = pad_if_necessary(pred_data)

    # Amplify the data to make gradient computation more sensitive
    amplified_data = pred_data_padded * 10
    
    # Calculate the gradients
    gradients = np.gradient(amplified_data)
    
    # Calculate the gradient magnitude
    grad_magnitude = np.sqrt(sum([g**2 for g in gradients]))
    
    # Return the mean and standard deviation of the gradient magnitude, ignoring zero regions
    return grad_magnitude[pred_data_padded != 0].mean(), grad_magnitude[pred_data_padded != 0].std()

def calculate_total_variation(pred_data, weight=0.1):
    amplified_data = pred_data * 1000
    denoised_image = denoise_tv_chambolle(amplified_data, weight=weight)
    tv_norm = np.sum(np.abs(amplified_data[pred_data != 0] - denoised_image[pred_data != 0]))  # Total variation norm
    tv_normalized = tv_norm / np.size(pred_data[pred_data != 0])  # Normalize by total number of non-zero voxels
    return tv_normalized

def calculate_entropy(pred_data):
    entropy_value = shannon_entropy(pred_data)
    max_entropy = np.log2(np.prod(pred_data.shape))  # Maximum possible entropy
    return entropy_value / max_entropy  # Normalised entropy

def calculate_cnr(region1, region2):
    mean_diff = np.abs(np.mean(region1) - np.mean(region2))
    noise = np.std(region1) + np.std(region2)
    if noise == 0:
        return float('inf')  # Handle the case where noise is zero
    return mean_diff / noise

def local_stddev(image_slice, size):
    """
    Calculate local standard deviation within a defined neighborhood.
    
    Parameters:
    image_slice (np.ndarray): 2D array representing a single slice of the image.
    size (int): Size of the neighborhood.

    Returns:
    np.ndarray: 2D array of local standard deviation values.
    """
    return generic_filter(image_slice, np.std, size=size)


def create_air_mask(gre_magnitude, quantile=0.4, min_variance=0.05):
    """
    Create an air mask for the GRE magnitude image by combining quantile thresholding 
    and variance filtering to isolate stable, low-intensity regions.

    Parameters:
    gre_magnitude (np.ndarray): 3D or 4D array of the GRE magnitude image.
    quantile (float): Quantile threshold to approximate very dark regions (default is 0.05).
    min_variance (float): Threshold to filter out high-variance regions within the low-intensity mask.

    Returns:
    np.ndarray: Binary mask representing air (background) regions with the same dimensions as input.
    """
    # Step 1: Quantile thresholding to get very dark regions
    threshold_value = np.quantile(gre_magnitude, quantile)
    low_intensity_mask = gre_magnitude < threshold_value
    low_intensity_mask = binary_opening(low_intensity_mask)

    return low_intensity_mask


def calculate_snr(pred_data, roi_foreground, roi_background):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for a given region of interest (ROI)
    based on a GRE magnitude image. The SNR is calculated as the ratio of the mean 
    signal intensity within the foreground ROI to the standard deviation of intensity 
    within the background ROI, adjusted by the Rayleigh constant.

    Parameters:
    pred_data (np.ndarray): 3D array of the GRE magnitude image.
    roi_foreground (np.ndarray): 3D binary mask of the foreground ROI, where 1 represents 
                                 the region of interest for signal measurement.
    roi_background (np.ndarray): 3D binary mask of the background (air) region, where 1 
                                 represents the area to measure noise.

    Returns:
    float: The calculated SNR for the foreground ROI. Returns infinity if noise is zero.
    """

    # Rayleigh constant to adjust for magnitude data
    rayleigh_constant = 0.655

    # Calculate the mean intensity within the foreground (signal) ROI
    signal = np.mean(pred_data[roi_foreground == 1])

    # Calculate the standard deviation of intensity within the background (noise) ROI
    noise = np.std(pred_data[roi_background == 1])

    # Handle the case where noise is zero to avoid division by zero
    if noise == 0:
        return float('inf')

    # Calculate SNR with the Rayleigh constant adjustment
    return (signal / noise) * rayleigh_constant

def calculate_snr_for_rois(gre_magnitude: np.ndarray, segmentation: np.ndarray, air_mask: np.ndarray) -> dict:
    """
    Calculates SNR for each structural ROI in a GRE magnitude image using a segmentation map.
    It determines an air mask dynamically with Otsu thresholding, then calculates SNR
    for each ROI by invoking the calculate_snr function.

    Parameters:
    gre_magnitude (np.ndarray): 3D array of the GRE magnitude image.
    segmentation (np.ndarray): 3D array of segmentation IDs, where each integer value represents an ROI.

    Returns:
    dict: Dictionary of segmentation IDs and their corresponding SNR values.
    """

    # Step 2: Calculate noise level as standard deviation of air region intensities
    snr_dict = {}
    for roi_id in np.unique(segmentation):
        if roi_id == 0:
            continue  # Skip background label if it is zero in segmentation

        # Create foreground mask for the current ROI
        roi_mask = segmentation == roi_id

        # Step 3: Calculate SNR for the ROI using the `calculate_snr` function
        snr = calculate_snr(gre_magnitude, roi_mask, air_mask)
        snr_dict[roi_id] = snr

    return snr_dict

def calculate_edge_strength(pred_data):
    amplified_data = pred_data * 100.0
    edges = gaussian_laplace(amplified_data, sigma=1.5)
    non_zero_edges = edges[pred_data != 0]
    return np.var(non_zero_edges)

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
    return root_mean_squared_error(pred_data, ref_data)

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
    nrmse = rmse * np.sqrt(len(ref_data)) / np.linalg.norm(ref_data) # Frobenius norm
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
    Pads the arrays with zeros if necessary to avoid errors during the SSIM calculation.

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
    # Determine the minimum size for the SSIM window
    min_size = 7

    # Pad pred_data and ref_data if necessary
    pred_data_padded = pad_if_necessary(pred_data, min_size)
    ref_data_padded = pad_if_necessary(ref_data, min_size)

    # Determine the appropriate win_size
    win_size = min(min(pred_data_padded.shape), min_size)

    # Set data range if not provided
    if data_range is None:
        data_range = ref_data_padded.max() - ref_data_padded.min()

    # Calculate the structural similarity index (XSIM)
    xsim = structural_similarity(pred_data_padded, ref_data_padded, win_size=win_size, K1=0.01, K2=0.001, data_range=data_range)
    
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
    Pads the arrays with zeros if necessary to avoid errors during gradient calculation.

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
    # Pad pred_data and ref_data if necessary
    pred_data_padded = pad_if_necessary(pred_data)
    ref_data_padded = pad_if_necessary(ref_data)
    
    # Compute the gradient difference error
    gxe = np.sqrt(np.mean((np.array(np.gradient(pred_data_padded)) - np.array(np.gradient(ref_data_padded))) ** 2))
    
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
    bbox : tuple or None
        A tuple of slice objects representing the bounding box of the ROI.
        Returns None if the ROI is empty.

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
    if coords.size == 0:
        return None  # Return None if the ROI is empty
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1
    return tuple(slice(min_coords[d], max_coords[d]) for d in range(roi.ndim))

def calculate_metrics_for_roi(pred_data, roi, ref_data=None, quality_metrics=True):
    bbox = get_bounding_box(roi)
    if bbox is None:
        return {}  # Return empty metrics if bbox is None

    roi = pad_if_necessary(np.array(roi[bbox], dtype=bool))
    pred_data = pad_if_necessary(pred_data[bbox] * roi)
    if ref_data is not None:
        ref_data = pad_if_necessary(ref_data[bbox] * roi)

    metrics = {
        'Voxels': roi.sum(),
        'Minimum': pred_data[roi].min(),
        'Maximum': pred_data[roi].max(),
        'Mean': pred_data[roi].mean(),
        'Median': np.median(pred_data[roi]),
        'Standard deviation': pred_data[roi].std()
    }
    if quality_metrics:
        metrics.update({
            'Gradient Mean': calculate_gradient_magnitude(pred_data)[0],
            'Gradient Std': calculate_gradient_magnitude(pred_data)[1],
            'Total Variation': calculate_total_variation(pred_data),
            'Entropy': calculate_entropy(pred_data),
            'Edge Strength': calculate_edge_strength(pred_data)
        })
    if ref_data is not None:
        metrics.update({
            'RMSE': calculate_rmse(pred_data[roi], ref_data[roi]),
            'NRMSE': calculate_nrmse(pred_data[roi], ref_data[roi]),
            'HFEN': calculate_hfen(pred_data, ref_data),
            'MAD': calculate_mad(pred_data[roi], ref_data[roi]),
            'GXE': calculate_gxe(pred_data, ref_data),
            'XSIM': calculate_xsim(pred_data, ref_data),
            'CC': pearson_corr_coeff(pred_data[roi], ref_data[roi])[0],
            'NMI': normalized_mutual_information(pred_data[roi], ref_data[roi])
        })
    return metrics

def all_metrics(pred_data, segmentation, mask, ref_data=None, quality_metrics=False):
    """
    Compute all metrics for the predicted data.

    Parameters
    ----------
    pred_data : numpy.ndarray
        Predicted data as a numpy array.
    segmentation : numpy.ndarray
        Segmentation data as a numpy array.
    mask : numpy.ndarray
        Mask data as a numpy array.
    ref_data : numpy.ndarray, optional
        Reference data as a numpy array.

    Returns
    -------
    dict
        A dictionary containing the metrics for each ROI.
    """
    
    metrics = {}
    for roi_label in np.unique(segmentation):
        print(f"[INFO] Calculating metrics for label {roi_label}")
        roi = np.array(segmentation == roi_label, dtype=bool)
        metrics[roi_label] = calculate_metrics_for_roi(pred_data, roi, ref_data, quality_metrics)
    metrics[-1] = calculate_metrics_for_roi(pred_data, mask, ref_data, quality_metrics)
    return metrics

def save_as_csv(metrics_dict, filepath):
    """
    Save the metrics as a CSV file

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics for each ROI.
    filepath : str
        The path to the file to save the results.
    """

    # Extract all unique metrics across all ROIs
    all_metrics = set()
    for metrics in metrics_dict.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)  # Sort to keep a consistent order

    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        header = ["Region"] + list(all_metrics)
        writer.writerow(header)
        
        # Write each ROI row
        for roi_label, metrics in metrics_dict.items():
            row = [roi_label] + [metrics.get(metric, "") for metric in all_metrics]
            writer.writerow(row)

def save_as_markdown(metrics_dict, filepath):
    """
    Save the metrics as a markdown table

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics for each ROI.
    filepath : str
        The path to the file to save the results.
    """
    
    # Extract all unique metrics across all ROIs
    all_metrics = set()
    for metrics in metrics_dict.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)  # Sort to keep a consistent order

    # Determine column widths
    max_region_len = max(len(str(roi_label)) for roi_label in metrics_dict.keys())
    max_metric_len = {metric: max(len(metric), max(len(f"{metrics.get(metric, ''):.6f}")
                                                 if isinstance(metrics.get(metric, 0), (int, float)) 
                                                 else len(str(metrics.get(metric, "")))
                                                 for metrics in metrics_dict.values()))
                      for metric in all_metrics}
    
    with open(filepath, 'w') as file:
        # Write header
        header = f"| {'Region'.ljust(max_region_len)} | " + " | ".join(f"{metric.ljust(max_metric_len[metric])}" for metric in all_metrics) + " |"
        file.write(header + "\n")
        file.write("|" + "-" * (max_region_len + 2) + "|" + "|".join("-" * (max_metric_len[metric] + 2) for metric in all_metrics) + "|\n")
        
        # Write each ROI row
        for roi_label, metrics in metrics_dict.items():
            row = f"| {roi_label.ljust(max_region_len)} | " + " | ".join(f"{metrics.get(metric, ''):.6f}".ljust(max_metric_len[metric]) if isinstance(metrics.get(metric, 0), (int, float)) else str(metrics.get(metric, "")).ljust(max_metric_len[metric]) for metric in all_metrics) + " |"
            file.write(row + "\n")

def save_as_json(metrics_dict, filepath):
    """
    Save the metrics as a JSON file

    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing the metrics.
    filepath : str
        The path to the file to save the results.
    """
    json_data = []

    # Structure the data with each region as a row, metrics as columns
    for roi_label, metrics in metrics_dict.items():
        region_data = {"Region": roi_label}
        region_data.update(metrics)
        json_data.append(region_data)

    # Save as JSON
    with open(filepath, 'w') as file:
        json.dump(json_data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Compute metrics for 3D images.')
    parser.add_argument('--ground_truth', type=str, help='Path to the ground truth NIFTI image (optional).')
    parser.add_argument('--estimate', type=str, required=True, help='Path to the reconstructed NIFTI image.')
    parser.add_argument('--roi', type=str, help='Path to the ROI NIFTI image (optional).')
    parser.add_argument('--segmentation', type=str, help='Path to the segmentation NIFTI image (optional, if provided will compute metrics for each ROI).')
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

    # Load segmentation (if provided)
    if args.segmentation:
        print("[INFO] Loading segmentation image...")
        segmentation_img = nib.load(args.segmentation).get_fdata().astype(int)
    else:
        segmentation_img = None

    # Compute metrics
    print("[INFO] Computing metrics...")
    metrics = all_metrics(
        pred_data=recon_img,
        ref_data=gt_img,
        roi=roi_img,
        segmentation=segmentation_img
    )

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

