<h1 align="center">KIF-ISDforMRG</h1>

Code of paper: "Graph-Driven Medical Report Generation with Adaptive Knowledge Distillation".

## Abstract
Automated Medical Report Generation (MRG) faces a critical hurdle in seamlessly integrating detailed visual evidence with accurate clinical diagnoses. Current approaches often rely on static knowledge transfer, overlooking the complex interdependencies among pathological findings and their nuanced alignment with visual evidence, often yielding reports that are linguistically sound but clinically misaligned. To address these limitations, we propose a novel graph-driven medical report generation framework with adaptive knowledge distillation. Our architecture leverages a dual-phase optimization process. 
First, visual-semantic enhancement proceeds through the explicit correlation of image features with a structured knowledge network and their concurrent enrichment via cross-modal semantic fusion, ensuring that generated descriptions are grounded in anatomical and pathological context.
Second, a knowledge distillation mechanism iteratively refines both global narrative flow and local descriptive precision, enhancing the consistency between images and text.
Comprehensive experiments on the MIMIC-CXR and IU X-Ray datasets demonstrate the effectiveness of our approach, which achieves state-of-the-art performance in clinical efficacy metrics across both datasets.

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
* **IU-Xray**: The images can be downloaded from [R2Gen](https://github.com/zhjohnchan/R2Gen) and the annotation file can be downloaded from the [Google Drive]


## Contributing:
Contributions to this project are welcome. If you would like to contribute, Please feel free to:

Submit issues and bug reports

Propose new features or enhancements

Create pull requests for improvements

Share your results using our model

## Contact
For any inquiries or suggestions, please contact 2024220603009@zstu.mails.edu.cn.

Note: Include any additional sections or information that may be relevant to your specific project.

## Acknowledgment
We thank the authors of [R2Gen](https://github.com/zhjohnchan/R2Gen), [BLIP](https://github.com/salesforce/BLIP) for making their models publicly available.
