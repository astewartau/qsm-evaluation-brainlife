#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import base64
import nibabel as nib
import numpy as np
from eval import all_metrics

def plot_error_metrics(all_metrics_dicts, output_dir, title="Error Metrics"):
    metrics_keys = list(all_metrics_dicts[0][0].keys())  # The metric names (x-axis categories)
    num_metrics = len(metrics_keys)
    num_estimates = len(all_metrics_dicts)

    # Create a color map for different estimates
    cmap = cm.get_cmap('tab10')

    # Get an array of indices for the x-axis based on the number of metrics, adding extra space between each group
    x = np.arange(0, num_metrics * 2, 2)
    
    # Define the width of each bar
    bar_width = 0.2

    plt.figure(figsize=(12, 6))

    # Loop through each set of metrics and plot with unique colors
    for idx, (metrics, label) in enumerate(all_metrics_dicts):
        metric_values = [metrics[metric] for metric in metrics_keys]  # Extract the values for this estimate
        plt.bar(x + idx * bar_width, metric_values, bar_width, label=label, color=cmap(idx / num_estimates))

        # Annotate each bar with its value
        for i, value in enumerate(metric_values):
            plt.text(x[i] + idx * bar_width, value + 0.01, f"{value:.3f}", ha='center', va='bottom', fontsize=8)

    # Set the labels and title
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('QSM Evaluation Metrics')
    plt.xticks(x + bar_width * (num_estimates - 1) / 2, metrics_keys, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the plot as a PNG file in the output directory
    plot_path = os.path.join(output_dir, "metrics_plot.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def create_json_for_brainlife(encoded_image, image_title="QSM Error Metrics"):
    data = {
        "brainlife": [
            {
                "type": "image/png",
                "name": image_title,
                "base64": encoded_image
            }
        ]
    }
    return json.dumps(data, indent=4)

def generate_html_table(all_metrics_dicts):
    # Extract unique metric names from the first estimate's metrics
    metrics_keys = list(all_metrics_dicts[0][0].keys())

    # Start creating the table HTML
    html = "<table border='1' cellpadding='5'>"
    html += "<thead><tr><th>Estimate</th>"

    # Add columns for each metric name
    for metric in metrics_keys:
        html += f"<th>{metric}</th>"
    html += "</tr></thead>"

    # Add rows for each QSM estimate
    html += "<tbody>"
    for metrics, label in all_metrics_dicts:
        html += f"<tr><td>{label}</td>"
        for metric in metrics_keys:
            html += f"<td>{metrics[metric]:.3f}</td>"
        html += "</tr>"
    html += "</tbody></table>"

    return html

def generate_index_html(output_dir, plot_path, metrics_table_html):
    # Create the index.html content
    html_content = f"""
    <html>
    <head><title>QSM Evaluation Results</title></head>
    <body>
    <h1>QSM Evaluation Metrics</h1>
    <h2>Metrics Table</h2>
    {metrics_table_html}
    <h2>Metrics Plot</h2>
    <img src="html/metrics_plot.png" alt="QSM Evaluation Metrics">
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
segmentation_file = config_json.get('qsm_segmentation', None)
segmentation_nii = None
segmentation_np = None
if segmentation_file:
    print("[INFO] Loading QSM segmentation...")
    segmentation_nii = nib.load(segmentation_file)
    segmentation_np = segmentation_nii.get_fdata()

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

# List to hold all metrics dictionaries with labels
all_metrics_dicts = []

# Iterate over each QSM estimate
for estimate in config_json['qsm_estimate']:
    print(f"[INFO] Loading QSM estimate: {estimate}")
    qsm = nib.load(estimate).get_fdata()

    print("[INFO] Computing evaluation metrics...")

    # Call the updated all_metrics function with optional ground truth and segmentation
    metrics_dict = all_metrics(
        pred_data=qsm,
        ref_data=qsm_groundtruth_np,
        roi=mask_np,
        roi_foreground=mask_np,
        roi_background=None
    )
    
    # Adjust any metrics if necessary
    if 'RMSE' in metrics_dict:
        del metrics_dict['RMSE']
    if 'NRMSE' in metrics_dict:
        metrics_dict['NRMSE'] /= 100.0
    if 'CC' in metrics_dict and isinstance(metrics_dict['CC'], tuple):
        metrics_dict['CC'] = (metrics_dict['CC'][0] + 1) / 2  # Normalise Pearson correlation
    if 'NMI' in metrics_dict:
        metrics_dict['NMI'] -= 1  # Normalisation step

    # Create label based on the corresponding entry in the _inputs section
    input_info = next(input for input in config_json['_inputs'] if input['task_id'] in estimate)
    label = input_info['id']
    if input_info.get('tags'):
        label += f" ({', '.join(input_info['tags'])})"
    
    # Add to the list of all metrics
    all_metrics_dicts.append((metrics_dict, label))

# Generate and save figure
print("[INFO] Generating figure...")
plot_path = plot_error_metrics(all_metrics_dicts, os.path.join(output_dir, 'html'))

# Convert figure to base64 (for JSON output)
print("[INFO] Converting figure to base64...")
encoded_image = encode_image_to_base64(plot_path)

# Generate product.json
print("[INFO] Generating product.json...")
json_data = create_json_for_brainlife(encoded_image)
with open('product.json', 'w') as json_file:
    json_file.write(json_data)

# Generate HTML table for metrics
metrics_table_html = generate_html_table(all_metrics_dicts)

# Generate index.html with plot and table
generate_index_html(output_dir, plot_path, metrics_table_html)

print("[INFO] Done!")
