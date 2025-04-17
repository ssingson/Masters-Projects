# Neural Style Transfer – Artist Aggregation Technique

## Overview
This project explores a new technique for Neural Style Transfer (NST) by aggregating style loss across **multiple style images**, instead of using a single image. By leveraging different pretrained Convolutional Neural Networks (CNNs) and custom style aggregation functions, this approach aims to create more holistic stylizations that reflect an entire artist’s body of work. The project was conducted as a final project for a graduate Deep Learning course at Fordham University.

## Requirements
- **Jupyter Notebook**
- **Google Drive** (if using original image paths)
- Python packages:
  - `tensorflow`
  - `numpy`
  - `matplotlib`
  - `PIL`
  - `os`, `glob`, `csv`, `time`

## Getting Started

### Dataset & Setup
All image datasets are prepackaged in the repo. However, you **must update file paths** for local use:

#### Update Style Dataset Paths
Each artist's style dataset path must be updated like so:
```python
Andy_Warhol_Dataset = tf.cast(tf.convert_to_tensor(
    np.reshape(image_to_dataset(
        load_images_from_drive('/your/local/path/Andy_Warhol')
    ), [-1, 1, 224,224, 3]), dtype=tf.float64), dtype=tf.float32)[:30]
```

#### Update Content Test Images
Set your test content images path:
```python
Test_Content_Images, test_image_sizes = load_images_from_drive(
    '/your/local/path/Test_Images', save_sizes=True
)
```

#### Update Output CSV Path
Each model’s training loss is recorded to a CSV file. Update the following path:
```python
loss_average.to_csv('/your/local/path/Adam_loss_vgg.csv')
```

### Running the Notebook
1. Open the Jupyter notebook and run all cells in sequence.
2. Style and content images are automatically processed once correct file locations are updated.
3. Outputs will include:
   - Stylized images
   - Content/style loss graphs
   - Aggregated training loss CSVs

## Included Datasets (Appendix Notes)
To fully reproduce and evaluate the models, the following appendix files should be present in the repo:
- **Appendix A:** Optimizer evaluation table (determines best optimizer per model)
- **Appendix B:** Loss over epochs (tracks convergence and early stopping)
- **Appendix C:** Final generated images and their lowest-style-loss reference image
- **Appendix D:** Test models of the generated images

## Key Features
- **Style Aggregation Techniques:**
  - *Simple Mean*
  - *Squared Mean* (MSE emphasis)
  - *Log-Aggregated Mean* (suppress outliers)
- **Pretrained CNN Options:**
  - VGG16
  - ResNet101
  - InceptionV3
  - MobileNetV3
- **Automatic Hyperparameter Selection:**
  - Optimizer chosen based on lowest loss curve
  - Epochs determined via early stopping (<0.1% delta)

## Results
- **VGG models** produced the most artistically faithful results. In particular:
  - **VGG Simple** and **VGG Squared** generated clear, stylized outputs
  - **Logged aggregation** often resulted in grainy, unstructured images
- **Quantitative evaluation**:
  - *Loss convergence charts* used to compare performance and efficiency
  - *Optimizer performance table* (Appendix A) helped determine ideal optimizer
  - Epochs were selected based on convergence trends (Appendix B)
- **Qualitative evaluation**:
  - Appendix C includes comparisons of stylized output with the closest style image based on loss, showing how different aggregation methods affect influence

## Acknowledgments
This project was developed by **Seth Singson-Robbins** at **Fordham University**.  
For further details, contact [seth.singson@gmail.com](mailto:seth.singson@gmail.com).