#!/usr/bin/env python
import matplotlib.pyplot as plt
import json
import base64
import nibabel as nib
from metrics import all_metrics
	
def plot_error_metrics(metrics, title="Error Metrics"):
    # Create a bar plot for the metrics
    #sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))
    
    # stem function
    plt.ylim(0, 1.0)
    
    # stem function: If x is not provided, a sequence of numbers is created by python:
    plt.stem(metrics.keys(), metrics.values())

    plt.title('QSM evaluation metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Annotate each point with its value
    for key, value in metrics.items():
        plt.text(key, value + 0.01, f"{value:.3f}", 
                ha='left', va='bottom', 
                fontsize=8, rotation=0)
    
    # Save the plot as a PNG file
    plt.savefig("metrics_plot.png")
    plt.close()

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def create_json_for_brainlife(encoded_image, image_title="My image title"):
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

# load inputs from config.json
print("[INFO] Loading configuration...")
with open('config.json') as config_json_file_handle:
	config_json = json.load(config_json_file_handle)

print("[INFO] Loading QSM results for metrics...")
qsm_file = config_json['qsm_estimate']
ground_truth_file = config_json['qsm_groundtruth']
qsm = nib.load(qsm_file).get_fdata()
ground_truth = nib.load(ground_truth_file).get_fdata()

print("[INFO] Computing evaluation metrics...")
metrics_dict = all_metrics(qsm, ground_truth)
del metrics_dict['RMSE']
metrics_dict['NRMSE'] /= 100.0
#metrics_dict['HFEN'] 
#metrics_dict['MAD']
#metrics_dict['XSIM']
metrics_dict['CC'] = (metrics_dict['CC'][0] + 1) / 2
metrics_dict['NMI'] -= 1 
#metrics_dict['GXE']


print("[INFO] Generating figure...")
plot_error_metrics(metrics_dict)

print("[INFO] Converting figure to base64...")
encoded_image = encode_image_to_base64("metrics_plot.png")

print("[INFO] Generating product.json...")
json_data = create_json_for_brainlife(encoded_image)
with open('product.json', 'w') as json_file:
    json_file.write(json_data)

