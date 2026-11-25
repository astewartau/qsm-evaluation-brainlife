<h1 align="center">QSM Evaluation</h1>

<p align="center">
  Benchmarks one or more QSM susceptibility estimates against a ground-truth map, computing RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI and GXE overall and per region, and reports them as an HTML figure and table.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.822"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.822-blue.svg" alt="Run on brainlife.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-see_LICENSE-lightgrey.svg" alt="License"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Benchmarks one or more QSM susceptibility estimates against a ground-truth map, computing RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI and GXE overall and per region, and reports them as an HTML figure and table.**

Computes error metrics between each QSM estimate and a ground-truth susceptibility map (RMSE, NRMSE, HFEN, XSIM, MAD, CC, NMI, GXE), overall and per ROI of a segmentation, and reports per-ROI value distributions and SNR from the GRE magnitude when available. The field strength comes from the estimate's MagneticFieldStrength metadata or B0. Use it to benchmark QSM pipelines, for example on simulated data with known ground truth. The README is an unfilled template; behaviour is taken from main.py and eval.py.

| | |
|---|---|
| **Inputs** | `neuro/qsm` · `neuro/qsm` (optional) · `neuro/parcellation/volume` (optional) · `neuro/mask` (optional) · `neuro/fmap` (optional) · `raw` (optional) |
| **Outputs** | `report/html` |
| **Container** | `python:3.9-bookworm` |
| **Resources** | 16 GB memory · 1 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.822](https://doi.org/10.25663/brainlife.app.822) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `qsm_estimate` | `neuro/qsm` | yes | Estimated susceptibility map produced by a QSM pipeline. |
| `qsm_groundtruth` | `neuro/qsm` (ground-truth) | no | Ground truth susceptibility map (e.g. produced by COSMOS or the QSM Data Generator App). |
| `qsm_segmentation` | `neuro/parcellation/volume` | no | Optional volumetric segmentation (parc + label.json) defining ROIs for per-region metrics. |
| `qsm_mask` | `neuro/mask` | no | Optional mask restricting the evaluation region. |
| `fieldmap_tissue` | `neuro/fmap` (single, tissue) | no | Optional tissue field map with GRE magnitude; the magnitude is used to compute per-ROI SNR. |
| `phase_quality` | `raw` (phase-quality) | no | Optional phase-quality map (phase-quality.nii.gz) used in the evaluation. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `html` | `report/html` | Report of QSM Evaluation |

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.822](https://doi.org/10.25663/brainlife.app.822), choose `neuro/qsm`, `neuro/qsm`, `neuro/parcellation/volume`, `neuro/mask`, `neuro/fmap`, `raw` from your project and press **Execute**. The results are saved to your project as `report/html`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `python:3.9-bookworm`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "qsm_estimate": "/path/to/qsm_estimate/qsm_estimate",
       "qsm_groundtruth": "/path/to/qsm_groundtruth/qsm_groundtruth",
       "parc": "/path/to/qsm_segmentation/parc",
       "label": "/path/to/qsm_segmentation/label",
       "qsm_mask": "/path/to/qsm_mask/qsm_mask",
       "magnitude": "/path/to/fieldmap_tissue/magnitude",
       "magnitude_json": "/path/to/fieldmap_tissue/magnitude_json",
       "fieldmap": "/path/to/fieldmap_tissue/fieldmap",
       "fieldmap_json": "/path/to/fieldmap_tissue/fieldmap_json",
       "phase_quality": "/path/to/phase_quality/phase_quality"
   }
   ```

3. Run `./main`.

## Troubleshooting

- **'B0' is needed in config.json or 'MagneticFieldStrength' in qsm_estimate metadata!**: Set MagneticFieldStrength in the QSM estimate metadata.

## app-example-documentation

This is an example of how to write documentation (readme.md and license.md for Apps on brainlife.io)

Write the following here...

1) What the App does, and how it does it at the basic level.
2) Briefly explain what 1) means for novice users in a language that 1st year psychology student can understand it.
3) Briefly description of input / output files.

## Sample Datasets

If you don't have your own input file, you can download sample datasets from Brainlife.io, or you can use [Brainlife CLI](https://github.com/brainlife/cli).

```
npm install -g brainlife
bl login
mkdir input
bl dataset download 5a0f0fad2c214c9ba8624376#5a050966eec2b300611abff2 && mv 5a0f0fad2c214c9ba8624376#5a050966eec2b300611abff2 .
```

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- QSM Evaluation. brainlife.io App. [doi:10.25663/brainlife.app.822](https://doi.org/10.25663/brainlife.app.822)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Franco Pestilli

**Contributors**

- Franco Pestilli

#### Copyright (c) 2022 brainlife.io The University of Texas at Austin

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io, which runs this App, is publicly funded by Wellcome, the U.S. National Institutes of Health and the U.S. National Science Foundation; acknowledging that support helps keep it running.</sub>

## License

See [LICENSE](LICENSE).
