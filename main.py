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

def plot_metrics_by_region(metrics_dict, title="Error Metrics by Region"):
    """
    Generate a static bar plot with metrics on the x-axis and regions (ROIs) as the legend.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
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

    return fig

def plot_regions_by_metrics(metrics_dict, title="Region-Wise Metrics"):
    """
    Generate a static bar plot with regions (ROIs) on the x-axis and metrics as the legend.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
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

    return fig

def plot_quality_measures_by_region(metrics_dict, title="Quality Measures by Region"):
    """
    Generate a static bar plot with quality metrics on the x-axis and regions (ROIs) as the legend.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
    title : str, optional
        Title of the plot, by default "Quality Measures by Region".
    """
    quality_measures = ['Gradient Mean', 'Gradient Std', 'Total Variation', 'Entropy', 'Edge Strength']
    roi_names = list(metrics_dict.keys())
    num_colors = len(roi_names)

    # if none of the measures are in the metrics_dict, return
    if not any(metric in metrics_dict[roi] for roi in roi_names for metric in quality_measures):
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

    return fig

def plot_regions_by_quality_measures(metrics_dict, title="Region-Wise Quality Measures"):
    """
    Generate a static bar plot with regions (ROIs) on the x-axis and quality metrics as the legend.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metrics for each ROI.
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

    return fig

def plot_roi_statistics_boxplot(estimate, segmentation, labels, title="Value Distributions by ROI",
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

def generate_sync_code(viewer_ids):
    """
    Generates JavaScript code to set up bidirectional synchronization between NiiVue viewers.

    Parameters
    ----------
    viewer_ids : list of str
        List of unique IDs for the NiiVue viewers to synchronize.

    Returns
    -------
    str
        JavaScript code for setting up synchronization.
    """
    sync_code = ""
    for i, id1 in enumerate(viewer_ids):
        for id2 in viewer_ids[i + 1:]:
            sync_code += f'console.log("Syncing {id1} with {id2}");\n'
            sync_code += f'window.nvInstances["{id1}"].syncWith(window.nvInstances["{id2}"], {{ "3d": true, "2d": true }});\n'

    broadcast_code = ""
    for source_id in viewer_ids:
        target_ids = [f'window.nvInstances["{vid}"]' for vid in viewer_ids if vid != source_id]
        broadcast_code += f'window.nvInstances["{source_id}"].broadcastTo([{", ".join(target_ids)}], {{ "3d": true, "2d": true }});\n'

    return broadcast_code

def generate_niivue_html(nii_path, colormap="gray", cal_range=(-0.1, 0.1), slider_range=None):
    # Set slider range if not provided
    if slider_range is None:
        img_data = nib.load(nii_path).get_fdata()
        slider_range = (float(np.min(img_data)), float(np.max(img_data)))
    
    # Convert NIfTI to base64
    nifti_base64 = convert_nii_to_base64(nii_path)

    # Generate a unique ID
    unique_id = str(uuid.uuid4()).replace('-', '_')

    niivue_html = f"""
    <div id="{unique_id}" style="position: relative; max-width: 800px; width: 100%; height: auto; margin: 0 auto;">
        <div style="position: relative; width: 100%; height: 0; padding-bottom: 50%;"> <!-- 16:9 aspect ratio padding -->
            <canvas id="{unique_id}_canvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></canvas>
        </div>
        <div id="calRangeSlider_{unique_id}" style="width: 80%; margin: 10px auto;"></div>
        <footer id="{unique_id}_intensity" style="text-align: center; margin-top: 10px;">&nbsp;</footer>
    </div>

    <script type="module">
        import * as niivue from "https://niivue.github.io/niivue/dist/index.js";

        var nv_{unique_id} = new niivue.Niivue({{
            dragAndDropEnabled: true,
            onLocationChange: function(data) {{
                document.getElementById("{unique_id}_intensity").innerHTML = "&nbsp;&nbsp;" + data.string;
            }}
        }});
        nv_{unique_id}.attachTo("{unique_id}_canvas");

        // Load base64-encoded NIfTI image into NVImage
        let image_{unique_id} = niivue.NVImage.loadFromBase64({{
            base64: "{nifti_base64}",
            name: "{os.path.basename(nii_path)}",
            colormap: "{colormap}",
            opacity: 1.0,
            cal_min: {cal_range[0]},
            cal_max: {cal_range[1]},
        }});

        nv_{unique_id}.addVolume(image_{unique_id});
        nv_{unique_id}.setSliceType(nv_{unique_id}.sliceTypeMultiplanar);
        nv_{unique_id}.opts.multiplanarShowRender = niivue.SHOW_RENDER.NEVER;
        nv_{unique_id}.opts.multiplanarLayout = niivue.MULTIPLANAR_TYPE.ROW;
        nv_{unique_id}.setInterpolation(true);

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

def generate_niivue_overlay_html(nifti_path1, nifti_path2, cal_range1=(-0.1, 0.1), cal_range2=(-0.1, 0.1), 
                                 slider_range1=None, slider_range2=None, colormap1="gray", colormap2="gray"):
    """
    Generates HTML for a NiiVue viewer with two overlaid images and sliders for contrast adjustment 
    and opacity for the overlay image.

    Parameters
    ----------
    nifti_path1 : str
        Path to the base NIfTI file.
    nifti_path2 : str
        Path to the overlay NIfTI file.
    cal_range1 : tuple of (float, float), optional
        Default contrast range (cal_min, cal_max) for the base image.
    cal_range2 : tuple of (float, float), optional
        Default contrast range (cal_min, cal_max) for the overlay image.
    slider_range1 : tuple of (float, float), optional
        Range for the base image's contrast slider. If None, it uses the image data range.
    slider_range2 : tuple of (float, float), optional
        Range for the overlay image's contrast slider. If None, it uses the image data range.
    colormap1 : str, optional
        Colormap for the base image, by default "gray".
    colormap2 : str, optional
        Colormap for the overlay image, by default "gray".

    Returns
    -------
    tuple
        HTML string for embedding the NiiVue viewer with overlay functionality and a unique ID.
    """
    # Set slider ranges based on image data if not provided
    if slider_range1 is None:
        img_data1 = nib.load(nifti_path1).get_fdata()
        slider_range1 = (float(np.min(img_data1)), float(np.max(img_data1)))
    if slider_range2 is None:
        img_data2 = nib.load(nifti_path2).get_fdata()
        slider_range2 = (float(np.min(img_data2)), float(np.max(img_data2)))

    # Convert NIfTI files to base64
    nifti_base64_1 = convert_nii_to_base64(nifti_path1)
    nifti_base64_2 = convert_nii_to_base64(nifti_path2)

    # Generate a unique ID
    unique_id = str(uuid.uuid4()).replace('-', '_')

    # HTML content for NiiVue viewer with overlay and sliders
    html_content = f"""
    <div id="{unique_id}" style="position: relative; max-width: 800px; width: 100%; height: auto; margin: 0 auto;">
        <div style="position: relative; width: 100%; height: 0; padding-bottom: 50%;"> <!-- 16:9 aspect ratio padding -->
            <canvas id="{unique_id}_canvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></canvas>
        </div>
        <div id="calRangeSlider1_{unique_id}" style="width: 80%; margin: 10px auto;"></div>
        <div id="calRangeSlider2_{unique_id}" style="width: 80%; margin: 10px auto;"></div>
        <div id="{unique_id}_opacity_slider" style="width: 80%; margin: 10px auto;"></div>
        <footer id="{unique_id}_intensity" style="text-align: center; margin-top: 10px;">&nbsp;</footer>
    </div>

    <link href="https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/14.6.3/nouislider.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/noUiSlider/14.6.3/nouislider.min.js"></script>

    <script type="module" async>
        import * as niivue from "https://niivue.github.io/niivue/dist/index.js";

        var nv_{unique_id} = new niivue.Niivue({{
            dragAndDropEnabled: true,
            onLocationChange: function(data) {{
                document.getElementById("{unique_id}_intensity").innerHTML = "&nbsp;&nbsp;" + data.string;
            }}
        }});
        nv_{unique_id}.attachTo("{unique_id}_canvas");

        // Load base64-encoded NIfTI images into NVImage objects for overlay
        let baseImage = niivue.NVImage.loadFromBase64({{
            base64: "{nifti_base64_1}",
            name: "{os.path.basename(nifti_path1)}",
            colormap: "{colormap1}",
            opacity: 1.0,
            cal_min: {cal_range1[0]},
            cal_max: {cal_range1[1]},
        }});
        
        let overlayImage = niivue.NVImage.loadFromBase64({{
            base64: "{nifti_base64_2}",
            name: "{os.path.basename(nifti_path2)}",
            colormap: "{colormap2}",
            opacity: 0.0,  // Start with overlay hidden
            cal_min: {cal_range2[0]},
            cal_max: {cal_range2[1]},
        }});

        nv_{unique_id}.addVolume(baseImage);
        nv_{unique_id}.addVolume(overlayImage);
        nv_{unique_id}.setSliceType(nv_{unique_id}.sliceTypeMultiplanar);
        nv_{unique_id}.opts.multiplanarShowRender = niivue.SHOW_RENDER.NEVER;
        nv_{unique_id}.opts.multiplanarLayout = niivue.MULTIPLANAR_TYPE.ROW;
        nv_{unique_id}.setInterpolation(true);

        // Initialize noUiSlider for base image cal_min and cal_max
        noUiSlider.create(document.getElementById("calRangeSlider1_{unique_id}"), {{
            start: [{cal_range1[0]}, {cal_range1[1]}],
            connect: true,
            range: {{
                'min': {slider_range1[0]},
                'max': {slider_range1[1]}
            }},
            step: 0.01,
            tooltips: [true, true],
            format: {{
                to: value => value.toFixed(2),
                from: value => parseFloat(value)
            }}
        }});

        // Update cal_min and cal_max for the base image with slider
        document.getElementById("calRangeSlider1_{unique_id}").noUiSlider.on("update", function(values, handle) {{
            let calMin = parseFloat(values[0]);
            let calMax = parseFloat(values[1]);
            nv_{unique_id}.volumes[0].cal_min = calMin;
            nv_{unique_id}.volumes[0].cal_max = calMax;
            nv_{unique_id}.updateGLVolume();
        }});

        // Initialize noUiSlider for overlay image cal_min and cal_max
        noUiSlider.create(document.getElementById("calRangeSlider2_{unique_id}"), {{
            start: [{cal_range2[0]}, {cal_range2[1]}],
            connect: true,
            range: {{
                'min': {slider_range2[0]},
                'max': {slider_range2[1]}
            }},
            step: 0.01,
            tooltips: [true, true],
            format: {{
                to: value => value.toFixed(2),
                from: value => parseFloat(value)
            }}
        }});

        // Update cal_min and cal_max for the overlay image with slider
        document.getElementById("calRangeSlider2_{unique_id}").noUiSlider.on("update", function(values, handle) {{
            let calMin = parseFloat(values[0]);
            let calMax = parseFloat(values[1]);
            nv_{unique_id}.volumes[1].cal_min = calMin;
            nv_{unique_id}.volumes[1].cal_max = calMax;
            nv_{unique_id}.updateGLVolume();
        }});

        // Initialize noUiSlider for overlay opacity
        noUiSlider.create(document.getElementById("{unique_id}_opacity_slider"), {{
            start: [0.0],  // Initial opacity
            connect: [true, false],
            range: {{
                'min': 0,
                'max': 1
            }},
            step: 0.01,
            tooltips: true,
            format: {{
                to: value => value.toFixed(2),
                from: value => parseFloat(value)
            }}
        }});

        // Update overlay opacity dynamically with noUiSlider
        document.getElementById("{unique_id}_opacity_slider").noUiSlider.on("update", function(values) {{
            let overlayOpacity = parseFloat(values[0]);
            nv_{unique_id}.volumes[1].opacity = overlayOpacity;  // Adjust the opacity of the overlay
            nv_{unique_id}.updateGLVolume();
        }});

        // Register instance globally for optional syncing
        window.nvInstances = window.nvInstances || {{}};
        window.nvInstances["{unique_id}"] = nv_{unique_id};
    </script>
    """
    return html_content, unique_id

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

def generate_html_content(body, sync_code=""):
    return f"""
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
        {body}

        <script type="module">
            document.addEventListener("DOMContentLoaded", function() {{
                setTimeout(() => {{
                    if (window.nvInstances) {{
                        {sync_code}
                    }}
                }}, 500);  // Delay to ensure all instances are initialized
            }});
        </script>
    </body>
    </html>
    """


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
            if entry["id"] == "qsm_estimate":
                B0 = entry["meta"]["MagneticFieldStrength"]
                break
    
    if B0 is None and "B0" in config_json:
        B0 = config_json["B0"]
        if B0 is None:
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
    if segmentation_file:
        print("[INFO] Loading segmentation...")
        segmentation_nii = nib.load(segmentation_file)
        segmentation_np = segmentation_nii.get_fdata()

    # Load segmentation labels
    labels_file = config_json.get('label', None)
    if labels_file:
        print("[INFO] Loading segmentation labels...")
        with open(labels_file, 'r') as labels_fh:
            label_data = json.load(labels_fh)
        labels = {item['voxel_value']: item['name'] for item in label_data}
        labels[-1] = "Mask"
    else:
        labels = {}
        for roi_id in np.unique(segmentation_np):
            labels[roi_id] = f"ROI {roi_id}"
    ids = {}
    for roi_id, roi_name in labels.items():
        ids[roi_name] = roi_id

    # Load mask
    mask_file = config_json.get('qsm_mask', None)
    if mask_file:
        print("[INFO] Loading QSM mask...")
        mask_nii = nib.load(mask_file)
        mask_np = mask_nii.get_fdata()
    elif segmentation_np is not None:
        mask_np = np.array(segmentation_np != 0, dtype=int)
    else:
        mask_np = np.array(qsm_estimate_np != 0, dtype=int)
    mask_np[mask_np != 0] = 1

    # Load magnitude image
    magnitude_file = config_json.get('magnitude', None)
    if magnitude_file:
        print("[INFO] Loading magnitude image...")
        magnitude_nii = nib.load(magnitude_file)
        magnitude_np = magnitude_nii.get_fdata()

    # Load QSM ground truth
    qsm_groundtruth_file = config_json.get('qsm_groundtruth', None)
    qsm_groundtruth_np = None
    if qsm_groundtruth_file:
        print("[INFO] Loading QSM ground truth...")
        qsm_groundtruth_nii = nib.load(qsm_groundtruth_file)
        qsm_groundtruth_np = qsm_groundtruth_nii.get_fdata()

    # Load fieldmap
    fieldmap_gt_file = config_json.get('fieldmap', None)
    if fieldmap_gt_file:
        print("[INFO] Loading fieldmap...")
        fieldmap_gt_nii = nib.load(fieldmap_gt_file)
        fieldmap_gt_np = fieldmap_gt_nii.get_fdata()

    # Load phase quality map
    phase_quality_file = os.path.join(config_json.get('phase_quality', None), 'phase-quality.nii.gz')
    if phase_quality_file:
        print("[INFO] Loading phase quality map...")
        phase_quality_nii = nib.load(phase_quality_file)
        phase_quality_np = phase_quality_nii.get_fdata()

    # Calculate all metrics
    if magnitude_file:
        print("[INFO] Calculating SNR metrics...")
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
        
        if labels_file: metrics_snr = { labels.get(roi_id, f"ROI {roi_id}"): snr for roi_id, snr in metrics_snr.items() }

    # Generate metrics for QSM
    if qsm_groundtruth_file:
        print("[INFO] Calculating metrics on QSM error map...")
        qsm_metrics_error = eval.all_metrics(
            pred_data=np.abs(qsm_groundtruth_np - qsm_estimate_np),
            segmentation=segmentation_np,
            mask=mask_np,
            quality_metrics=False
        )
        if labels_file: qsm_metrics_error = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in qsm_metrics_error.items() }
        qsm_groundtruth_masked = qsm_groundtruth_np * mask_np
        qsm_groundtruth_masked_file = os.path.join(output_dir, "qsm_groundtruth_masked.nii.gz")
        nib.save(
            img=nib.Nifti1Image(
                dataobj=qsm_groundtruth_masked,
                affine=qsm_groundtruth_nii.affine,
                header=qsm_groundtruth_nii.header
            ),
            filename=qsm_groundtruth_masked_file
        )
    print("[INFO] Calculating QSM metrics...")
    qsm_metrics = eval.all_metrics(
        pred_data=qsm_estimate_np,
        segmentation=segmentation_np,
        mask=mask_np,
        quality_metrics=False,
        ref_data=qsm_groundtruth_np
    )
    if labels_file: qsm_metrics = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in qsm_metrics.items() }

    if fieldmap_gt_file:
        print("[INFO] Generating fieldmap estimate...")
        fieldmap_estimate_np = B0 * 42.576 * qsm_forward.generate_field(
            chi=qsm_estimate_np,
            voxel_size=voxel_size,
            B0_dir=B0_dir
        )
        print("[INFO] Calculating fieldmap metrics...")
        fieldmap_metrics = eval.all_metrics(
            pred_data=fieldmap_estimate_np,
            segmentation=segmentation_np,
            mask=mask_np,
            ref_data=fieldmap_gt_np
        )
        if labels_file: fieldmap_metrics = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in fieldmap_metrics.items() }

        print("[INFO] Calculating metrics on fieldmap error map...")
        fieldmap_metrics_error = eval.all_metrics(
            pred_data=np.abs(fieldmap_gt_np - fieldmap_estimate_np),
            segmentation=segmentation_np,
            mask=mask_np,
            quality_metrics=False
        )
        if labels_file: fieldmap_metrics_error = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in fieldmap_metrics_error.items() }

    if phase_quality_file:
        print("[INFO] Calculating metrics on phase quality map...")
        phase_quality_metrics = eval.all_metrics(
            pred_data=phase_quality_np,
            segmentation=segmentation_np,
            mask=mask_np,
            quality_metrics=False
        )
        if labels_file: phase_quality_metrics = { labels.get(roi_id, f"ROI {roi_id}"): metrics for roi_id, metrics in phase_quality_metrics.items() }

    # Saving outputs
    print("[INFO] Saving outputs...")
    if fieldmap_gt_file:
        print("[INFO] Saving fieldmap estimate...")
        fieldmap_estimate_file = os.path.join(output_dir, "fieldmap_estimate.nii.gz")
        nib.save(
            img=nib.Nifti1Image(
                dataobj=fieldmap_estimate_np,
                affine=qsm_estimate_nii.affine,
                header=qsm_estimate_nii.header
            ),
            filename=fieldmap_estimate_file
        )
        fieldmap_error_file = os.path.join(output_dir, "fieldmap_error.nii.gz")
        nib.save(
            img=nib.Nifti1Image(
                dataobj=fieldmap_estimate_np - fieldmap_gt_np,
                affine=qsm_estimate_nii.affine,
                header=qsm_estimate_nii.header
            ),
            filename=fieldmap_error_file
        )
        eval.save_as_csv(fieldmap_metrics, os.path.join(output_dir, "fieldmap_metrics.csv"))
        eval.save_as_csv(fieldmap_metrics_error, os.path.join(output_dir, "fieldmap_metrics_error.csv"))

    if qsm_groundtruth_file:
        print("[INFO] Saving QSM error map...")
        qsm_error_file = os.path.join(output_dir, "qsm_error.nii.gz")
        nib.save(
            img=nib.Nifti1Image(
                dataobj=np.abs(qsm_groundtruth_np - qsm_estimate_np),
                affine=qsm_estimate_nii.affine,
                header=qsm_estimate_nii.header
            ),
            filename=qsm_error_file
        )
        eval.save_as_csv(qsm_metrics_error, os.path.join(output_dir, "qsm_metrics_error.csv"))
    eval.save_as_csv(qsm_metrics, os.path.join(output_dir, "qsm_metrics.csv"))

    if magnitude_file:
        print("[INFO] Saving SNR metrics...")
        with open(os.path.join(output_dir, 'snr.json'), 'w') as json_file:
            json_file.write(json.dumps(metrics_snr, indent=4))

        # generate SNR NIfTI file with the SNR value substituted in for each ROI
        if segmentation_file:
            snr_img = np.zeros_like(segmentation_np)
            for roi_name, snr in metrics_snr.items():
                snr_img[segmentation_np == ids[roi_name]] = snr
            snr_file = os.path.join(output_dir, "snr.nii.gz")
            nib.save(
                img=nib.Nifti1Image(
                    dataobj=snr_img,
                    affine=segmentation_nii.affine,
                    header=segmentation_nii.header
                ),
                filename=snr_file
            )

    # Generate index.html with all images, metrics, and embedded figures
    print("[INFO] Generating index.html...")
    html_body = ""

    #qsm_niivue_html, qsm_niivue_id = generate_niivue_html(qsm_estimate_file)
    #html_body += f"<h2>QSM Estimate Visualization</h2>{qsm_niivue_html}"
    if qsm_groundtruth_file is not None:
        qsm_niivue_html, qsm_niivue_id = generate_niivue_overlay_html(qsm_estimate_file, qsm_groundtruth_masked_file, cal_range1=(-0.1, 0.1), cal_range2=(-0.1, 0.1), slider_range1=(-1, +1), slider_range2=(-1, +1))
        html_body += f"<h2>QSM Overlay Visualization</h2>{qsm_niivue_html}"
        #qsm_groundtruth_niivue_html, qsm_groundtruth_niivue_id = generate_niivue_html(qsm_groundtruth_file)
        #html_body += f"<h2>QSM Ground Truth</h2>{qsm_groundtruth_niivue_html}"
        #qsm_error_niivue_html, qsm_error_niivue_id = generate_niivue_html(qsm_error_file, colormap="jet", cal_range=(0, 0.2), slider_range=(0, 3))
        #html_body += f"<h2>QSM Error Visualization</h2>{qsm_error_niivue_html}"
    if phase_quality_file:
        phase_quality_niivue_html, phase_quality_niivue_id = generate_niivue_html(phase_quality_file, cal_range=(0, 1), slider_range=(0, 1))
        html_body += f"<h2>Phase Quality Visualization</h2>{phase_quality_niivue_html}"
    if magnitude_file:
        snr_niivue_html, snr_niivue_id = generate_niivue_html(snr_file, cal_range=(0, 300), slider_range=(0, 500))
        html_body += f"<h2>SNR Visualization</h2>{snr_niivue_html}"
    html_body += f"<h2>QSM Metrics</h2>{generate_html_table(qsm_metrics)}"
    if phase_quality_file:
        html_body += f"<h2>Phase Quality Metrics</h2>{generate_html_table(phase_quality_metrics)}"
    if fieldmap_gt_file:
        html_body += f"<h2>Fieldmap Metrics</h2>{generate_html_table(fieldmap_metrics)}"
    if fieldmap_gt_file:
        fieldmap_niivue_html, fieldmap_niivue_id = generate_niivue_html(fieldmap_gt_file, cal_range=(-10, +10), slider_range=(-20, +20))
        html_body += f"<h2>Fieldmap (forward from QSM) Visualization</h2>{fieldmap_niivue_html}"
        fieldmap_estimate_html, fieldmap_estimate_niivue_id = generate_niivue_html(fieldmap_estimate_file, cal_range=(-10, +10), slider_range=(-20, +20))
        html_body += f"<h2>Fieldmap (ROMEO + V-SHARP) Visualization</h2>{fieldmap_estimate_html}"
        fieldmap_error_niivue_html, fieldmap_error_niivue_id = generate_niivue_html(fieldmap_error_file, colormap="jet", cal_range=(0, 10), slider_range=(0, 20))
        html_body += f"<h2>Fieldmap Error Visualization</h2>{fieldmap_error_niivue_html}"

    fig = plot_roi_statistics_boxplot(qsm_estimate_np, segmentation_np, labels, title='QSM Values across ROIs', reference_values_json='literature-qsm-values.json')
    if fig is not None:
        html_body += f"<h2>QSM Values across ROIs</h2>{fig.to_html()}"

    if fieldmap_gt_file:
        #html_body += f"<h2>Fieldmap Values across ROIs</h2>{plot_roi_statistics_boxplot(fieldmap_estimate_np, segmentation_np, labels, title='Fieldmap Values across ROIs').to_html()}"
        fig = plot_roi_statistics_boxplot(abs(fieldmap_estimate_np - fieldmap_gt_np), segmentation_np, labels, title='Fieldmap Errors across ROIs')
        if fig is not None:
            html_body += f"<h2>Fieldmap Errors across ROIs</h2>{fig.to_html()}"
    
    fig = plot_metrics_by_region(qsm_metrics, title='QSM Metrics by Region')
    if fig is not None:
        html_body += f"<h2>QSM Metrics by Region</h2>{fig.to_html()}"
    #html_body += f"<h2>QSM Metrics</h2>{plot_regions_by_metrics(qsm_metrics, title='QSM Metrics by Region').to_html()}"
    #html_body += f"<h2>QSM Quality Measures by Region</h2>{plot_quality_measures_by_region(qsm_metrics, title='QSM Quality Measures by Region').to_html()}"
    #html_body += f"<h2>QSM Quality Measures by Measure</h2>{plot_regions_by_quality_measures(qsm_metrics, title='QSM Quality Measures by Measure').to_html()}"

    if fieldmap_gt_file:
        fig = plot_metrics_by_region(fieldmap_metrics, title='Fieldmap Metrics by Region')
        if fig is not None:
            html_body += f"<h2>Fieldmap Metrics by Region</h2>{fig.to_html()}"
        #html_body += f"<h2>Fieldmap Metrics</h2>{plot_regions_by_metrics(fieldmap_metrics, title='Fieldmap Metrics by Region').to_html()}"
        #html_body += f"<h2>Fieldmap Quality Measures by Region</h2>{plot_quality_measures_by_region(fieldmap_metrics, title='Fieldmap Quality Measures by Region').to_html()}"
        #html_body += f"<h2>Fieldmap Quality Measures by Measure</h2>{plot_regions_by_quality_measures(fieldmap_metrics, title='Fieldmap Quality Measures by Measure').to_html()}"


    sync_code = ""

    qsm_sync_ids = []
    if qsm_groundtruth_file is not None:
        qsm_sync_ids = [qsm_niivue_id]
    if magnitude_file:
        qsm_sync_ids.append(snr_niivue_id)
    if phase_quality_file:
        qsm_sync_ids.append(phase_quality_niivue_id)
    if qsm_sync_ids:
        sync_code += generate_sync_code(qsm_sync_ids)    

    if fieldmap_gt_file:
        sync_code += generate_sync_code([fieldmap_niivue_id, fieldmap_estimate_niivue_id, fieldmap_error_niivue_id])

    full_html = generate_html_content(html_body, sync_code)
    with open(os.path.join(output_dir, "index.html"), "w") as html_file:
        html_file.write(full_html)

    print("[INFO] Done!")

