#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
from matplotlib import colormaps
import json
import base64
import nibabel as nib
import numpy as np
from eval import all_metrics, save_as_csv, save_as_json, save_as_markdown

import seaborn as sns
import pandas as pd

def plot_error_metrics(metrics_dict, output_dir, title="Error Metrics"):
    """
    Generate plots for error measures, statistics, and quality measures.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
    output_dir : str
        Directory to save the generated plots.
    title : str, optional
        Title of the plot, by default "Error Metrics".
    """
    cmap = colormaps['tab10']
    
    # Categorize metrics
    error_measures = ['RMSE', 'NRMSE', 'HFEN', 'MAD', 'GXE', 'XSIM', 'CC', 'NMI']
    statistics = ['Minimum', 'Maximum', 'Mean', 'Median', 'Standard deviation']
    quality_measures = ['Gradient Mean', 'Gradient Std', 'Total Variation', 'Entropy', 'Edge Strength']

    # 1. Generate bar plots for Error Measures (per ROI)
    for roi_idx, (roi, metrics) in enumerate(metrics_dict.items()):
        # Check if error measures exist for this ROI
        roi_error_measures = {k: v for k, v in metrics.items() if k in error_measures}
        if roi_error_measures:
            plt.figure(figsize=(12, 6))
            metric_names = list(roi_error_measures.keys())
            metric_values = []
            
            # Handle tuple-based metrics (like Pearson correlation)
            for metric_value in roi_error_measures.values():
                if isinstance(metric_value, tuple) and len(metric_value) == 2:
                    metric_values.append(metric_value[0])
                else:
                    metric_values.append(metric_value)

            # Plot Error Measures for the current ROI
            x = range(len(metric_names))
            plt.bar(x, metric_values, width=0.4, label=roi, color=cmap(roi_idx % 10))

            # Annotate each bar with its value
            for i, value in enumerate(metric_values):
                plt.text(i, value + 0.01, f"{value:.3f}", ha='center', va='bottom', fontsize=8)

            # Customize the plot
            plt.title(f"{title} - Error Measures for {roi}")
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right')
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, f"metrics_plot_error_{roi}.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"[INFO] Saved error measures plot for {roi} to {plot_path}")

    # 2. Generate combined boxplots per ROI (for statistics)
    plt.figure(figsize=(12, 6))

    # Prepare data for boxplot: statistics for each ROI
    data = []
    roi_labels = []
    for roi_label, metrics in metrics_dict.items():
        if roi_label in ["Air", "Bone", "All", "Muscle", "Calcification"]:
            continue
        stats_values = [metrics.get(stat) for stat in statistics]
        # Only add ROI if all stats are available
        if all(stats_values):
            data.append(stats_values)
            roi_labels.append(roi_label)

    # Plot boxplots for each ROI's statistics
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    plt.boxplot(data, labels=roi_labels, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.ylim([-0.1, +0.3])
    plt.title("Statistics Combined for Each ROI")
    plt.xlabel('ROI')
    plt.ylabel('Value')
    plt.tight_layout()

    # Save boxplot
    plot_path = os.path.join(output_dir, "metrics_plot_statistics_combined.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Saved boxplot for combined ROI statistics to {plot_path}")

    # 3. Generate bar plots for Quality Measures (per ROI)
    for roi_idx, (roi, metrics) in enumerate(metrics_dict.items()):
        roi_quality_measures = {k: v for k, v in metrics.items() if k in quality_measures}
        if roi_quality_measures:
            plt.figure(figsize=(12, 6))
            metric_names = list(roi_quality_measures.keys())
            metric_values = []

            for metric_value in roi_quality_measures.values():
                metric_values.append(metric_value)

            # Plot Quality Measures for the current ROI
            x = range(len(metric_names))
            plt.bar(x, metric_values, width=0.4, label=roi, color=cmap(roi_idx % 10))

            # Annotate each bar with its value
            for i, value in enumerate(metric_values):
                plt.text(i, value + 0.01, f"{value:.3f}", ha='center', va='bottom', fontsize=8)

            # Customize the plot
            plt.title(f"{title} - Quality Measures for {roi}")
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right')
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, f"metrics_plot_quality_{roi}.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"[INFO] Saved quality measures plot for {roi} to {plot_path}")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def create_json_for_brainlife(encoded_images):
    """
    Creates a Brainlife-compatible JSON structure with multiple base64-encoded images.

    Parameters
    ----------
    encoded_images : list
        A list of dictionaries with image titles and base64 encoded images.
    
    Returns
    -------
    str
        JSON string of the Brainlife-compatible structure.
    """
    data = {
        "brainlife": [
            {
                "type": "image/png",
                "name": image['name'],
                "base64": image['base64']
            }
            for image in encoded_images
        ]
    }
    return json.dumps(data, indent=4)

def generate_html_table(metrics_dict):
    """
    Generate an HTML table with columns for Metric, Region (ROI), and Value.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.

    Returns
    -------
    str
        HTML string representing the table.
    """
    # Start creating the table HTML
    html = "<table border='1' cellpadding='5'>"
    html += "<thead><tr><th>Metric</th><th>Region</th><th>Value</th></tr></thead>"

    # Add rows for each ROI
    html += "<tbody>"
    for roi_label, metrics in metrics_dict.items():
        for metric, value in metrics.items():
            html += f"<tr><td>{metric}</td><td>{roi_label}</td>"

            # Handle tuple-based metrics (e.g., correlation and p-value)
            if isinstance(value, tuple) and len(value) == 2:
                html += f"<td>{value[0]:.3f} (correlation), {value[1]:.3f} (p-value)</td>"
            else:
                html += f"<td>{value:.3f}</td>"

            html += "</tr>"
    html += "</tbody></table>"

    return html

def generate_index_html(output_dir, image_paths, metrics_table_html):
    """
    Generates an HTML file with all metrics and images.

    Parameters
    ----------
    output_dir : str
        Directory to save the HTML file.
    image_paths : list
        List of paths to PNG images.
    metrics_table_html : str
        HTML content for the metrics table.
    """
    # Create the image HTML tags
    image_html = ""
    for image_path in image_paths:
        image_filename = os.path.basename(image_path)
        image_html += f'<h2>{image_filename}</h2>\n'
        image_html += f'<img src="html/{image_filename}" alt="{image_filename}">\n'

    # Create the index.html content
    html_content = f"""
    <html>
    <head><title>QSM Evaluation Results</title></head>
    <body>
    <h1>QSM Evaluation Metrics</h1>
    <h2>Metrics Table</h2>
    {metrics_table_html}
    <h2>Metrics Plots</h2>
    {image_html}
    </body>
    </html>
    """

    # Write the HTML to index.html in the output directory
    index_html_path = os.path.join(output_dir, "index.html")
    with open(index_html_path, "w") as html_file:
        html_file.write(html_content)
    
    print(f"[INFO] index.html generated at {index_html_path}")

# Load inputs from config.json
print("[INFO] Loading configuration...")
with open('config.json') as config_json_file_handle:
    config_json = json.load(config_json_file_handle)

# Prepare output directory
output_dir = "html"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'html'), exist_ok=True)

# Load QSM estimate
qsm_estimate_file = config_json.get('qsm_estimate', None)
print("[INFO] Loading QSM estimate...")
if not qsm_estimate_file:
    raise RuntimeError("QSM estimate missing from config.json!")
qsm_estimate_nii = nib.load(qsm_estimate_file)
qsm_estimate_np = qsm_estimate_nii.get_fdata()

# Load QSM ground truth
qsm_groundtruth_file = config_json.get('qsm_groundtruth', None)
qsm_groundtruth_nii = None
qsm_groundtruth_np = None
if qsm_groundtruth_file:
    print("[INFO] Loading QSM ground truth...")
    qsm_groundtruth_nii = nib.load(qsm_groundtruth_file)
    qsm_groundtruth_np = qsm_groundtruth_nii.get_fdata()

# Load segmentation
segmentation_file = config_json.get('parc', None)
segmentation_nii = None
segmentation_np = None
if segmentation_file:
    print("[INFO] Loading QSM segmentation...")
    segmentation_nii = nib.load(segmentation_file)
    segmentation_np = segmentation_nii.get_fdata()

# Load segmentation labels
labels_file = config_json.get('label', None)
labels = None
if labels_file:
    print("[INFO] Loading segmentation labels...")
    with open(labels_file, 'r') as labels_fh:
        label_data = json.load(labels_fh)
    labels = {item['voxel_value']: item['name'] for item in label_data}

# Load mask
mask_file = config_json.get('qsm_mask', None)
mask_nii = None
mask_np = None
if mask_file:
    print("[INFO] Loading QSM mask...")
    mask_nii = nib.load(mask_file)
    mask_np = mask_nii.get_fdata()
elif segmentation_np:
    mask_np = np.array(segmentation_np != 0, dtype=int)
else:
    mask_np = np.array(qsm_estimate_np != 0, dtype=int)

print("[INFO] Computing evaluation metrics...")

# FOR EACH DISTINCT ROI IN SEGMENTATION_NP...
current_roi = mask_np

# Call the updated all_metrics function with optional ground truth and segmentation
metrics_dict = all_metrics(
    pred_data=qsm_estimate_np,
    ref_data=qsm_groundtruth_np,
    roi=current_roi,
    segmentation=segmentation_np,
    labels=labels
)

save_as_markdown(metrics_dict=metrics_dict, filepath="metrics.md")
save_as_csv(metrics_dict=metrics_dict, filepath="metrics.csv")
save_as_json(metrics_dict=metrics_dict, filepath="metrics.json")

# Generate and save figures
print("[INFO] Generating figures...")
plot_dir = os.path.join(output_dir, 'html')
plot_error_metrics(metrics_dict, plot_dir)

# Collect all PNG files in plot_dir
png_paths = [os.path.join(plot_dir, f) for f in os.listdir(plot_dir) if f.endswith('.png')]

# Convert each PNG file to base64
encoded_images = []
for png_path in png_paths:
    print(f"[INFO] Converting {png_path} to base64...")
    encoded_image = encode_image_to_base64(png_path)
    encoded_images.append({"name": os.path.basename(png_path), "base64": encoded_image})

# Generate product.json with all base64 images
print("[INFO] Generating product.json...")
json_data = create_json_for_brainlife(encoded_images)
with open('product.json', 'w') as json_file:
    json_file.write(json_data)

# Generate HTML table for metrics
metrics_table_html = generate_html_table(metrics_dict)

# Generate index.html with all images and table
generate_index_html(output_dir, png_paths, metrics_table_html)

print("[INFO] Done!")

