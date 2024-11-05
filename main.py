#!/usr/bin/env python

import os
import json
import base64
import uuid

import nibabel as nib
import numpy as np
import qsm_forward
import eval
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import n_colors
from plotly.io import to_html

from skimage.filters import threshold_otsu

def plot_metrics_by_region(metrics_dict, output_dir, title="Error Metrics by Region"):
    """
    Generate a static bar plot with metrics on the x-axis and regions (ROIs) as the legend.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
    output_dir : str
        Directory to save the generated plot.
    title : str, optional
        Title of the plot, by default "Error Metrics by Region".
    """
    error_measures = ['RMSE', 'NRMSE', 'HFEN', 'MAD', 'GXE', 'XSIM', 'CC', 'NMI']
    roi_names = list(metrics_dict.keys())
    num_colors = len(roi_names)

    # if none of the measures are in the metrics_dict, return
    if all([metric not in metrics_dict[roi_names[0]] for metric in error_measures]):
        return
    
    # Generate distinct colors for each ROI
    colors = n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', num_colors, colortype='rgb')
    
    fig = go.Figure()
    for idx, roi in enumerate(roi_names):
        metric_values = [metrics_dict[roi].get(metric, None) for metric in error_measures]
        fig.add_trace(go.Bar(
            x=error_measures,
            y=metric_values,
            name=roi,
            marker_color=colors[idx]
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '-')}.html")
    fig.write_html(plot_path)
    print(f"[INFO] Saved plot with metrics on x-axis to {plot_path}")

    return fig

def plot_regions_by_metrics(metrics_dict, output_dir, title="Region-Wise Metrics"):
    """
    Generate a static bar plot with regions (ROIs) on the x-axis and metrics as the legend.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
    output_dir : str
        Directory to save the generated plot.
    title : str, optional
        Title of the plot, by default "Region-Wise Metrics".
    """
    error_measures = ['RMSE', 'NRMSE', 'HFEN', 'MAD', 'GXE', 'XSIM', 'CC', 'NMI']
    roi_names = list(metrics_dict.keys())
    num_colors = len(error_measures)

    # if none of the measures are in the metrics_dict, return
    if all([metric not in metrics_dict[roi_names[0]] for metric in error_measures]):
        return

    # Generate distinct colors for each metric
    colors = n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', num_colors, colortype='rgb')
    
    fig = go.Figure()
    for idx, metric in enumerate(error_measures):
        roi_values = [metrics_dict[roi].get(metric, None) for roi in roi_names]
        fig.add_trace(go.Bar(
            x=roi_names,
            y=roi_values,
            name=metric,
            marker_color=colors[idx]
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Region",
        yaxis_title="Value",
        barmode="group",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '-')}.html")
    fig.write_html(plot_path)
    print(f"[INFO] Saved plot with regions on x-axis to {plot_path}")

    return fig

def plot_quality_measures_by_region(metrics_dict, output_dir, title="Quality Measures by Region"):
    """
    Generate a static bar plot with quality metrics on the x-axis and regions (ROIs) as the legend.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
    output_dir : str
        Directory to save the generated plot.
    title : str, optional
        Title of the plot, by default "Quality Measures by Region".
    """
    quality_measures = ['Gradient Mean', 'Gradient Std', 'Total Variation', 'Entropy', 'Edge Strength']
    roi_names = list(metrics_dict.keys())
    num_colors = len(roi_names)

    # if none of the measures are in the metrics_dict, return
    if all([metric not in metrics_dict[roi_names[0]] for metric in quality_measures]):
        return
    
    # Generate distinct colors for each ROI
    colors = n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', num_colors, colortype='rgb')
    
    fig = go.Figure()
    for idx, roi in enumerate(roi_names):
        metric_values = [metrics_dict[roi].get(measure, None) for measure in quality_measures]
        fig.add_trace(go.Bar(
            x=quality_measures,
            y=metric_values,
            name=roi,
            marker_color=colors[idx]
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Quality Metric",
        yaxis_title="Value",
        barmode="group",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '-')}.html")
    fig.write_html(plot_path)
    print(f"[INFO] Saved plot with quality metrics on x-axis to {plot_path}")

    return fig

def plot_regions_by_quality_measures(metrics_dict, output_dir, title="Region-Wise Quality Measures"):
    """
    Generate a static bar plot with regions (ROIs) on the x-axis and quality metrics as the legend.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
    output_dir : str
        Directory to save the generated plot.
    title : str, optional
        Title of the plot, by default "Region-Wise Quality Measures".
    """
    quality_measures = ['Gradient Mean', 'Gradient Std', 'Total Variation', 'Entropy', 'Edge Strength']
    roi_names = list(metrics_dict.keys())
    num_colors = len(quality_measures)

    # if none of the measures are in the metrics_dict, return
    if all([metric not in metrics_dict[roi_names[0]] for metric in quality_measures]):
        return

    # Generate distinct colors for each quality metric
    colors = n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', num_colors, colortype='rgb')
    
    fig = go.Figure()
    for idx, measure in enumerate(quality_measures):
        roi_values = [metrics_dict[roi].get(measure, None) for roi in roi_names]
        fig.add_trace(go.Bar(
            x=roi_names,
            y=roi_values,
            name=measure,
            marker_color=colors[idx]
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Region",
        yaxis_title="Value",
        barmode="group",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '-')}.html")
    fig.write_html(plot_path)
    print(f"[INFO] Saved plot with regions on x-axis to {plot_path}")

    return fig

def plot_roi_statistics_boxplot(estimate, segmentation, labels, output_dir, title="Value Distributions by ROI",
                                  sample_fraction=0.1, min_samples=100, max_samples=1000, small_roi_threshold=500,
                                  reference_values_json=None):
    """
    Generate interactive boxplots for each ROI using an adaptive stratified sample of voxel values from QSM estimate
    and segmentation arrays. Plots literature values for each ROI as horizontal lines.

    Parameters
    ----------
    estimate : numpy.ndarray
        QSM estimate data.
    segmentation : numpy.ndarray
        Segmentation mask where each distinct integer label corresponds to a different ROI.
    labels : dict
        A dictionary containing metrics for each ROI, used to extract labels.
    output_dir : str
        Directory to save the generated plots.
    sample_fraction : float, optional
        Fraction of the data to sample from each large ROI, default is 10%.
    min_samples : int, optional
        Minimum number of samples to retain per large ROI, default is 100.
    max_samples : int, optional
        Maximum number of samples per ROI to prevent overloading large ROIs, default is 1000.
    small_roi_threshold : int, optional
        Threshold below which ROIs are considered small and are minimally sampled, default is 500.
    reference_values_json : str, optional
        Path to JSON file containing literature values for each ROI.
    """
    # Load literature values from the JSON file
    literature_values = {}
    if reference_values_json is not None:
        with open(reference_values_json, 'r') as f:
            literature_values = json.load(f)

    data = []
    roi_labels = []
    unique_labels = np.unique(segmentation)

    for roi_label in unique_labels:
        if roi_label == 0:  # Skip background
            continue

        label_name = labels.get(roi_label, f"ROI {roi_label}")
        if label_name in ['Air', 'Bone', 'All', 'Calcification', 'Muscle'] and len(unique_labels) != 2:
            continue

        roi_data = estimate[segmentation == roi_label]
        roi_size = len(roi_data)

        # Adaptive sampling with maximum limit
        if roi_size <= small_roi_threshold:
            sampled_data = roi_data if roi_size < min_samples else np.random.choice(roi_data, size=min_samples, replace=False)
        else:
            sample_size = min(max(int(roi_size * sample_fraction), min_samples), max_samples)
            sampled_data = np.random.choice(roi_data, size=sample_size, replace=False)

        data.extend(sampled_data)
        roi_labels.extend([label_name] * len(sampled_data))

    # Create the boxplot for ROI data with swapped axes
    fig = px.box(
        y=roi_labels,
        x=data,
        labels={"y": "ROI", "x": "Voxel Values"},
        title=title
    )

    # Add literature values as a trace
    if literature_values:
        literature_x = []
        literature_y = []
        for roi, value in literature_values.items():
            if roi in roi_labels:
                literature_y.append(roi)
                literature_x.append(value)

        fig.add_trace(
            go.Scatter(
                y=literature_y,
                x=literature_x,
                mode="lines",
                line=dict(color="blue", dash="dot"),
                name="Literature values"
            )
        )

    # Update layout for readability
    fig.update_layout(
        yaxis_tickangle=0,
        template="plotly_white",
        legend_title_text="Legend",
        shapes=[dict(type="line", x0=0, x1=0, yref="paper", y0=0, y1=1, line=dict(color="red", dash="dash"))]
    )

    # Save the plot to HTML
    plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '-')}.html")
    fig.write_html(plot_path)
    print(f"[INFO] Saved interactive boxplot for voxel values by ROI to {plot_path}")

    return fig

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

def convert_nii_to_base64(nii_path):
    """Convert a NIfTI file to a base64 string."""
    with open(nii_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8").replace('\n', '')
    return encoded_string

def get_nifti_metadata(nii_path):
    """Extract metadata from a NIfTI file."""
    img = nib.load(nii_path)
    dims = img.header['dim'][1:4].tolist()  # Dimensions (x, y, z)
    pixDims = img.header['pixdim'][1:4].tolist()  # Voxel sizes
    affine = img.affine.flatten().tolist()  # Affine transformation matrix, flattened
    datatypeCode = img.header['datatype']  # Data type code
    return dims, pixDims, affine, datatypeCode

def sync_niivue(*instance_ids):
    if len(instance_ids) < 2:
        return ""  # No sync needed if fewer than two instances

    sync_code = ""
    for i, id1 in enumerate(instance_ids):
        for id2 in instance_ids[i + 1:]:
            sync_code += f'window.nvInstances["{id1}"].syncWith(window.nvInstances["{id2}"], {{ "3d": true, "2d": true }});\n'
    
    return sync_code

def generate_niivue_html(nii_path, cal_range=(-0.1, 0.1), slider_range=(-1, 1), sync_id=None):
    # Convert NIfTI to base64
    nifti_base64 = convert_nii_to_base64(nii_path)
    
    # Generate a unique ID
    unique_id = str(uuid.uuid4()).replace('-', '_')

    sync_script = "" if sync_id is None else f"nv_{unique_id}.syncWith(nv_{sync_id}, {{ '3d': true, '2d': true }});"

    niivue_html = f"""
    <div style="margin-top: 10px; text-align: center;">
        <div id="calRangeSlider_{unique_id}" style="width: 100%; margin: 0 auto;"></div>
    </div>
    <main id="container_{unique_id}" style="position: relative; width: 100%; padding-top: 40%; overflow: hidden;">
        <canvas id="gl_{unique_id}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></canvas>
    </main>
    <footer id="intensity_{unique_id}" style="text-align: center; margin-top: 10px;">&nbsp;</footer>

    <script type="module" async>
        import * as niivue from "https://niivue.github.io/niivue/dist/index.js";

        var nv_{unique_id} = new niivue.Niivue({{
            dragAndDropEnabled: true,
            onLocationChange: function(data) {{
                document.getElementById("intensity_{unique_id}").innerHTML = "&nbsp;&nbsp;" + data.string;
            }}
        }});
        nv_{unique_id}.attachTo("gl_{unique_id}");

        // Load base64-encoded NIfTI image into NVImage
        let image_{unique_id} = niivue.NVImage.loadFromBase64({{
            base64: "{nifti_base64}",
            name: "{os.path.split(nii_path)[1]}",
            colormap: "gray",
            opacity: 1.0,
            cal_min: {cal_range[0]},
            cal_max: {cal_range[1]},
        }});

        nv_{unique_id}.addVolume(image_{unique_id});
        nv_{unique_id}.setSliceType(nv_{unique_id}.sliceTypeMultiplanar);
        nv_{unique_id}.opts.multiplanarShowRender = niivue.SHOW_RENDER.NEVER;
        nv_{unique_id}.opts.multiplanarLayout = niivue.MULTIPLANAR_TYPE.ROW;
        nv_{unique_id}.setInterpolation(true);

        {sync_script}

        // Initialize noUiSlider for cal_min and cal_max
        noUiSlider.create(document.getElementById("calRangeSlider_{unique_id}"), {{
            start: [{cal_range[0]}, {cal_range[1]}],
            connect: true,
            range: {{
                'min': {slider_range[0]},
                'max': {slider_range[1]}
            }},
            step: 0.01,
            tooltips: [true, true],
            format: {{
                to: value => value.toFixed(2),
                from: value => parseFloat(value)
            }}
        }});

        // Update cal_min and cal_max dynamically with noUiSlider
        document.getElementById("calRangeSlider_{unique_id}").noUiSlider.on("update", function(values, handle) {{
            let calMin = parseFloat(values[0]);
            let calMax = parseFloat(values[1]);
            nv_{unique_id}.volumes[0].cal_min = calMin;
            nv_{unique_id}.volumes[0].cal_max = calMax;
            nv_{unique_id}.updateGLVolume();
        }});

        window.nvInstances = window.nvInstances || {{}};
        window.nvInstances["{unique_id}"] = nv_{unique_id};
    </script>
    """
    return niivue_html, unique_id

def generate_index_html(output_dir, combined_metrics, figures_dict, qsm_estimate_file_path):
    """
    Generates an HTML file with all metrics, tables, embedded Plotly figures, and a NiiVue widget
    to visualize the qsm_estimate_file using base64 encoding of voxel data.

    Parameters
    ----------
    output_dir : str
        Directory to save the generated HTML file.
    combined_metrics : dict
        Dictionary containing QSM and Fieldmap metrics.
    figures_dict : dict
        Dictionary with Plotly figures to embed in the HTML.
    qsm_estimate_file_path : str
        Path to the qsm_estimate file to visualize in NiiVue.
    """

    metrics_table_html = ""
    for key, metrics_dict in combined_metrics.items():
        metrics_table_html += f"<h2>{key} Metrics</h2>"
        metrics_table_html += generate_html_table(metrics_dict)

    figures_html = ""
    for title, figure in figures_dict.items():
        if figure is not None:
            fig_html = to_html(figure, full_html=False)
            figures_html += f"<h3>{title}</h3>{fig_html}"

    # Generate Niivue viewers and get their unique IDs
    qsm_view_html, qsm_id = generate_niivue_html(qsm_estimate_file_path)
    fieldmap_view_html, fieldmap_id = generate_niivue_html(
        '/home/ashley/repos/qsm-evaluation-brainlife/outputs/fieldmap_tissue_estimate.nii.gz',
        cal_range=(-10, +10),
        slider_range=(-20, 20)
    )

    # Generate synchronization code using `sync_niivue`
    sync_code = sync_niivue(qsm_id, fieldmap_id)

    html_content = f"""
    <html>
    <head>
        <title>QSM Evaluation Results</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css">
        <script type="text/javascript" src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/14.6.3/nouislider.min.css" rel="stylesheet">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/14.6.3/nouislider.min.js"></script>

        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ width: 100%; margin-top: 20px; border-collapse: collapse; }}
            table, th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background-color: #f4f4f4; text-align: center; }}
            .dataTables_wrapper .dataTables_filter input {{
                margin-left: 0.5em;
                width: 50%;
            }}
            .dataTables_wrapper .dataTables_paginate .paginate_button {{
                color: white !important;
                background-color: #333 !important;
                border: none !important;
            }}
        </style>
        <script>
            $(document).ready(function() {{
                $('table.display').DataTable({{
                    "paging": true,
                    "searching": true,
                    "ordering": true,
                    "info": false,
                    "autoWidth": true,
                    "pageLength": 10
                }});
            }});
        </script>
    </head>
    <body>
        <h1>QSM Evaluation Summary</h1>
        {metrics_table_html}
        <h2>Interactive Figures</h2>
        {figures_html}
        <h2>QSM Estimate Visualization</h2>
        <div style="max-width: 900px; margin: 0 auto;">
            {qsm_view_html}
        </div>
        <h2>Fieldmap Estimate Visualization</h2>
        <div style="max-width: 900px; margin: 0 auto;">
            {fieldmap_view_html}
        </div>

        <script type="module">
            document.addEventListener("DOMContentLoaded", function() {{
                // Ensure all instances are initialized before syncing
                if (window.nvInstances) {{
                    {sync_code}
                }}
            }});
        </script>
    </body>
    </html>
    """

    index_html_path = os.path.join(output_dir, "index.html")
    with open(index_html_path, "w") as html_file:
        html_file.write(html_content)

    print(f"[INFO] index.html with embedded Plotly figures, DataTables, and NiiVue generated at {index_html_path}")

def generate_html_table(metrics_dict):
    """
    Generate an HTML table with each row representing an ROI and each column representing a metric,
    ordered by priority, and DataTables ready.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.

    Returns
    -------
    str
        HTML string representing the table.
    """
    # Define the desired metric order. Any metrics not in this list will appear in alphabetical order after.
    metric_order = [
        "Median", "Mean", "Standard deviation", "Minimum", "Maximum", "RMSE", "NRMSE", 
        "XSIM", "MAD", "HFEN", "GXE", "CC", "NMI", "Gradient Mean", "Gradient Std", 
        "Total Variation", "Entropy", "Edge Strength"
    ]

    # Get all unique metrics across ROIs and arrange them by the defined order
    all_metrics = set()
    for metrics in metrics_dict.values():
        all_metrics.update(metrics.keys())
    
    # Separate ordered metrics and those that aren't specified
    ordered_metrics = [metric for metric in metric_order if metric in all_metrics]
    remaining_metrics = sorted(set(all_metrics) - set(ordered_metrics))
    all_metrics = ordered_metrics + remaining_metrics

    # Start creating the table HTML with DataTables class for enhanced features
    html = "<table class='display' border='1' cellpadding='5'>"
    html += "<thead><tr><th>Region</th>" + "".join(f"<th>{metric}</th>" for metric in all_metrics) + "</tr></thead>"

    # Add a row for each ROI
    html += "<tbody>"
    for roi_label, metrics in metrics_dict.items():
        html += f"<tr><td>{roi_label}</td>"
        for metric in all_metrics:
            value = metrics.get(metric, "N/A")  # Default to "N/A" if a metric is missing

            # Handle tuple-based metrics (e.g., correlation and p-value)
            if isinstance(value, tuple) and len(value) == 2:
                html += f"<td>{value[0]:.3f} (correlation), {value[1]:.3f} (p-value)</td>"
            elif isinstance(value, (int, float)):
                html += f"<td>{value:.3f}</td>"
            else:
                html += f"<td>{value}</td>"  # Handles non-numeric or "N/A" cases

        html += "</tr>"
    html += "</tbody></table>"

    return html

def calculate_snr_for_rois(gre_magnitude: np.ndarray, segmentation: np.ndarray) -> dict:
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
    # Step 1: Define air mask using Otsu thresholding to differentiate air from tissue
    otsu_thresh = threshold_otsu(gre_magnitude)
    air_mask = gre_magnitude < otsu_thresh

    # Step 2: Calculate noise level as standard deviation of air region intensities
    snr_dict = {}
    for roi_id in np.unique(segmentation):
        if roi_id == 0:
            continue  # Skip background label if it is zero in segmentation

        # Create foreground mask for the current ROI
        roi_mask = segmentation == roi_id

        # Step 3: Calculate SNR for the ROI using the `calculate_snr` function
        snr = eval.calculate_snr(gre_magnitude, roi_mask, air_mask)
        snr_dict[roi_id] = snr

    return snr_dict

if __name__ == "__main__":
    # Load inputs from config.json
    print("[INFO] Loading configuration...")
    with open('config.json') as config_json_file_handle:
        config_json = json.load(config_json_file_handle)
    
    print("[INFO] Assuming B0 direction is along the z-axis...")
    B0_dir = (0, 0, 1)

    # Get MagneticFieldStrength from qsm_estimate
    B0 = None
    if "_inputs" in config_json:
        for entry in config_json["_inputs"]:
            if entry["id"] == "qsm":
                B0 = entry["meta"]["MagneticFieldStrength"]
                break
    
    if B0 is None and "B0" in config_json:
        B0 = config_json["B0"]
    else:
        raise RuntimeError("'B0' is needed in config.json or 'MagneticFieldStrength' in qsm_estimate metadata!")

    # Prepare output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load QSM estimate
    qsm_estimate_file = config_json.get('qsm_estimate', None)
    print("[INFO] Loading QSM estimate...")
    if not qsm_estimate_file:
        raise RuntimeError("QSM estimate missing from config.json!")
    qsm_estimate_nii = nib.load(qsm_estimate_file)
    voxel_size = qsm_estimate_nii.header.get_zooms()[:3]
    qsm_estimate_np = qsm_estimate_nii.get_fdata()

    # Load segmentation
    segmentation_file = config_json.get('parc', None)
    segmentation_nii = None
    segmentation_np = None
    if segmentation_file:
        print("[INFO] Loading segmentation...")
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
        labels[-1] = "Mask"

    # Load mask
    mask_file = config_json.get('qsm_mask', None)
    mask_nii = None
    mask_np = None
    if mask_file:
        print("[INFO] Loading QSM mask...")
        mask_nii = nib.load(mask_file)
        mask_np = mask_nii.get_fdata()
    elif segmentation_np is not None:
        mask_np = np.array(segmentation_np != 0, dtype=int)
    else:
        mask_np = np.array(qsm_estimate_np != 0, dtype=int)
    mask_np[mask_np != 0] = -1

    # Load magnitude image
    magnitude_file = config_json.get('magnitude', None)
    if magnitude_file:
        print("[INFO] Loading magnitude image...")
        magnitude_nii = nib.load(magnitude_file)
        magnitude_np = magnitude_nii.get_fdata()

        air_mask = eval.create_air_mask(magnitude_np)

        nib.save(
            img=nib.Nifti1Image(
                dataobj=air_mask,
                affine=magnitude_nii.affine,
                header=magnitude_nii.header
            ),
            filename=os.path.join(output_dir, "air_mask.nii.gz")
        )

        if segmentation_file:
            metrics_snr = eval.calculate_snr_for_rois(magnitude_np, segmentation_np, air_mask)
        else:
            metrics_snr = eval.calculate_snr(magnitude_np, mask_np, air_mask)

        # Map SNR values to ROI labels
        if labels:
            metrics_snr = { labels.get(roi_id, f"ROI {roi_id}"): snr for roi_id, snr in metrics_snr.items() }

        with open(os.path.join(output_dir, 'snr.json'), 'w') as json_file:
            json_file.write(json.dumps(metrics_snr, indent=4))

    # Load QSM ground truth
    qsm_groundtruth_file = config_json.get('qsm_groundtruth', None)
    qsm_groundtruth_np = None
    qsm_metrics_error = {}
    if qsm_groundtruth_file:
        print("[INFO] Loading QSM ground truth...")
        qsm_groundtruth_nii = nib.load(qsm_groundtruth_file)
        qsm_groundtruth_np = qsm_groundtruth_nii.get_fdata()

        # calculate absolute error
        qsm_error_np = np.abs(qsm_groundtruth_np - qsm_estimate_np)
        qsm_metrics_error = eval.all_metrics(
            pred_data=qsm_error_np,
            segmentation=segmentation_np,
            mask=mask_np,
            quality_metrics=False
        )
        if labels: qsm_metrics_error = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in qsm_metrics_error.items() }

        nib.save(
            img=nib.Nifti1Image(
                dataobj=qsm_error_np,
                affine=qsm_estimate_nii.affine,
                header=qsm_estimate_nii.header
            ),
            filename=os.path.join(output_dir, "qsm_error.nii.gz")
        )

    # Calculate metrics
    print("[INFO] Computing evaluation metrics on QSM...")
    qsm_metrics = eval.all_metrics(
        pred_data=qsm_estimate_np,
        segmentation=segmentation_np,
        mask=mask_np,
        ref_data=qsm_groundtruth_np
    )
    if labels: qsm_metrics = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in qsm_metrics.items() }
    eval.save_as_csv(qsm_metrics, os.path.join(output_dir, "qsm_metrics.csv"))

    # Load tissue fieldmap
    fieldmap_tissue_file = config_json.get('fieldmap_tissue', None)
    fieldmap_metrics = {}
    fieldmap_metrics_error = {}
    if fieldmap_tissue_file:
        print("[INFO] Loading tissue fieldmap...")
        fieldmap_tissue_nii = nib.load(fieldmap_tissue_file)
        fieldmap_tissue_np = fieldmap_tissue_nii.get_fdata()

        fieldmap_tissue_estimate_np = B0 * 42.576 * qsm_forward.generate_field(
            chi=qsm_estimate_np,
            voxel_size=voxel_size,
            B0_dir=B0_dir
        )
        nib.save(
            img=nib.Nifti1Image(
                dataobj=fieldmap_tissue_estimate_np,
                affine=qsm_estimate_nii.affine,
                header=qsm_estimate_nii.header
            ),
            filename=os.path.join(output_dir, "fieldmap_tissue_estimate.nii.gz")
        )

        print("[INFO] Computing evaluation metrics on fieldmap...")
        fieldmap_metrics = eval.all_metrics(
            pred_data=fieldmap_tissue_estimate_np,
            segmentation=segmentation_np,
            mask=mask_np,
            ref_data=fieldmap_tissue_np
        )
        if labels: fieldmap_metrics = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in fieldmap_metrics.items() }
        eval.save_as_csv(fieldmap_metrics, os.path.join(output_dir, "fieldmap_metrics.csv"))

        # calculate absolute error
        fieldmap_tissue_error_np = np.abs(fieldmap_tissue_np - fieldmap_tissue_estimate_np)
        nib.save(
            img=nib.Nifti1Image(
                dataobj=fieldmap_tissue_error_np,
                affine=qsm_estimate_nii.affine,
                header=qsm_estimate_nii.header
            ),
            filename=os.path.join(output_dir, "fieldmap_tissue_error.nii.gz")
        )
        fieldmap_metrics_error = eval.all_metrics(
            pred_data=fieldmap_tissue_error_np,
            segmentation=segmentation_np,
            mask=mask_np,
            quality_metrics=False
        )
        if labels: fieldmap_metrics_error = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in fieldmap_metrics_error.items() }
        eval.save_as_csv(fieldmap_metrics_error, os.path.join(output_dir, "fieldmap_metrics_error.csv"))

    # Generate and save figures
    print("[INFO] Generating figures...")
    figures_dict = {}
    for metrics_dict, data, name, reference_values_json in [
        (qsm_metrics, qsm_estimate_np, "QSM values (ppm)", 'literature-qsm-values.json'),
        (qsm_metrics_error, qsm_error_np, "QSM errors (ppm)", None),
        (fieldmap_metrics, fieldmap_tissue_estimate_np, "Field variations (Hz)", None),
        (fieldmap_metrics_error, fieldmap_tissue_error_np, "Fieldmap errors (Hz)", None)
    ]:
        figures_dict[f"{name} - error measures by region"] = plot_metrics_by_region(metrics_dict, output_dir, title=f"{name} - error measures by region")
        figures_dict[f"{name} - error measures by metric"] = plot_regions_by_metrics(metrics_dict, output_dir, title=f"{name} - error measures by metric")
        if segmentation_np is not None:
            figures_dict[f"{name} by ROI"] = plot_roi_statistics_boxplot(data, segmentation_np, labels, output_dir, title=f"{name} by ROI", reference_values_json=reference_values_json)
        figures_dict[f"{name} - quality measures by region"] = plot_quality_measures_by_region(metrics_dict, output_dir, title=f"{name} - quality measures by region")
        figures_dict[f"{name} - quality measures by measure"] = plot_regions_by_quality_measures(metrics_dict, output_dir, title=f"{name} - quality measures by measure")

    # Generate index.html with all images, metrics, and embedded figures
    generate_index_html(output_dir, {'QSM': qsm_metrics, 'Fieldmap': fieldmap_metrics}, figures_dict, qsm_estimate_file)

    print("[INFO] Done!")

