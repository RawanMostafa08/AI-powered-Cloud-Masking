# Cloud Masking and Segmentation Project

Optical satellite sensors are essential for Earth observation but have a major drawback—they cannot 
capture clear images when clouds obstruct their view. This limits their effectiveness in applications 
such as urban planning, deforestation monitoring, and agriculture. <br> <br>
 
To address the cloud obstruction issue in optical satellite imagery, a reliable method is needed to 
detect then remove clouds, ensuring clearer and more usable images for analysis. 

## Model Trials
1- [deepLabV3Plus](https://www.kaggle.com/models/rawanmostafa/deeplabv3_ckpt_epoch_18_88/Other/default/1) <br>
2- [FPN](https://www.kaggle.com/models/rmman2002/fpn_no_white_ckpt_epoch_20/Other/default/1) <br>
3- [UnetPlusPlus](https://www.kaggle.com/models/rmman2002/unetpp_more_filtered_ckpt_epoch_10/Other/default/1) <br>

## File Structure

The project is organized as follows, with all executable code and resources located in the `Final Deliverables` folder:

```
project_root/
└── Final Deliverables/
    ├── requirements.txt                # Required Python packages
    ├── bayes_clouds.ipynb              # Bayes classifier model implementation
    ├── svm_clouds.ipynb                # SVM model implementation
    ├── Deep_learning_models.ipynb      # Deep learning models notebook
    ├── model_logs.txt                  # Logs specifying the model parameter summary
    ├── rle_encoder_decoder.py          # The provided rle encoder decoder utility
    ├── run_inference.py                # The required inference scipt
    ├── sample_submission.csv           # The provided sample submission
    ├── team_13.csv                     # The submitted csv on Kaggle competition
    ├── profile.py                      # Our custom profiler

```


## Installation

1. **Navigate to the Final Deliverables folder**:
   ```bash
   cd "Final Deliverables"
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   The required packages are listed in `requirements.txt`. Install them using:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes the following key libraries:
   - `torch==2.7.0`, `torchvision==0.22.0`: PyTorch for deep learning
   - `segmentation_models_pytorch==0.5.0`: Pre-trained segmentation models
   - `rasterio==1.4.3`: For handling geospatial raster data
   - `opencv-python-headless==4.11.0.86`: Image processing
   - `albumentations==2.0.6`: Image augmentation
   - `matplotlib==3.10.1`, `numpy==2.2.5`, `pandas==2.2.3`: Data visualization and manipulation
   - `timm==1.0.15`: For additional pre-trained models
   - Other dependencies for utility and compatibility (e.g., `requests`, `PyYAML`, `tqdm`)

## Running the Code

1. **Ensure you are in the `Final Deliverables` directory**:
   ```bash
   cd "Final Deliverables"
   ```

2. **Run the inference script**:
   ```bash
   python run_inference.py
   ```

## Contributors

<table align="center" >
  <tr>
      <td align="center"><a href="https://github.com/SH8664"><img src="https://avatars.githubusercontent.com/u/113303945?v=4" width="150px;" alt=""/><br /><sub><b>Sara Bisheer</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/rawanMostafa08"><img src="https://avatars.githubusercontent.com/u/97397431?v=4" width="150px;" alt=""/><br /><sub><b>Rawan Mostafa</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com//mennamohamed0207"><img src="https://avatars.githubusercontent.com/u/90017398?v=4" width="150px;" alt=""/><br /><sub><b>Menna Mohammed</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/fatmaebrahim"><img src="https://avatars.githubusercontent.com/u/113191710?v=4" width="150;" alt=""/><br /><sub><b>Fatma Ebrahim</b></sub></a><br /></td>
  </tr>
</table>