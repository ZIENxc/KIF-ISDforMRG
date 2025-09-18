<h1 align="center">KIF-ISDforMRG</h1>

Code of paper: "Graph-Driven Medical Report Generation with Adaptive Knowledge Distillation".

### Medical Report Generation(MRG):
Medical Report Generation (MRG) is a critical task in clinical AI that involves automatically generating comprehensive diagnostic reports from medical images such as chest X-rays. This task bridges computer vision and natural language processing to assist radiologists by reducing workload, minimizing diagnostic variability, and improving reporting efficiency. Unlike general image captioning, MRG requires precise medical terminology, structured reporting format, and clinically accurate findings description, making it both challenging and essential for practical clinical deployment.

## Description:
This repository presents a hierarchical framework for automated chest X-ray report generation that systematically integrates medical images with structured medical knowledge and enhance semantic information with relevant reports. The pipeline begins by extracting visual features from input X-rays while concurrently encoding anatomical concepts into structured embeddings. These representations are fused through cross-attention mechanisms to produce knowledge-enhanced visual features.  The resulting unified representation drives a diagnostic prompt generator and text decoder to produce coherent medical reports, while an iterative self-distillation mechanism optimizes both linguistic quality and diagnostic accuracy through consistency constraints between teacher and student models.

## Installation
1. Clone this repository.
```Shell
git clone https://github.com/ZIENxc/KIF-ISDforMRG.git
```
2. Create a new conda environment.
```Shell
conda create -n mrg python=3.10
conda activate mrg
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```

## Datasets Preparation
* **MIMIC-CXR**: The images can be downloaded from either [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/) or [R2Gen](https://github.com/zhjohnchan/R2Gen). 
* **IU-Xray**: The images can be downloaded from [R2Gen](https://github.com/zhjohnchan/R2Gen)


## Contributing:
Contributions to this project are welcome. If you would like to contribute, please feel free to:

Submit issues and bug reports

Propose new features or enhancements

Create pull requests for improvements

Share your results using our model

## Contact
For any inquiries or suggestions, please contact 2024220603009@mails.zstu.edu.cn.

Note: Include any additional sections or information that may be relevant to your specific project.

## Acknowledgment
We thank the authors of [R2Gen](https://github.com/zhjohnchan/R2Gen), [BLIP](https://github.com/salesforce/BLIP) for making their models publicly available.
