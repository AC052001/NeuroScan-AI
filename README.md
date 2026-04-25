# NeuroScan AI - Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=yellow&style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-yellow?logo=pytorch&logoColor=red&style=plastic)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-73C2FB?logo=numpy&logoColor=blue&style=plastic)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-orange?logo=streamlit&logoColor=orange&style=plastic)
![U-Net](https://img.shields.io/badge/U--Net-architecture-green.svg)

## Overview

NeuroScan AI is a deep learning-powered brain tumor segmentation tool that leverages a U-Net architecture to automatically detect and segment brain tumors in MRI scans. This application provides a user-friendly interface for medical professionals and researchers to analyze brain MRI images with high accuracy.

## Features

- **U-Net Architecture**: State-of-the-art segmentation model for medical imaging
- **Test-Time Augmentation (TTA)**: 4-fold ensemble for more robust predictions
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for improved image quality
- **Adaptive Thresholding**: Intelligent threshold selection for accurate segmentation
- **Post-processing**: Morphological operations to refine segmentation masks
- **Multi-format Support**: Accepts JPG, PNG, TIF, and BMP image formats
- **Clinical Visualization**: Overlay, heatmap, and binary mask outputs
- **Performance Metrics**: Tumor area percentage, confidence scores, and severity classification

## Architecture

The application uses a custom U-Net implementation with the following components:

1. **Data Preprocessing**: 
   - CLAHE enhancement in LAB color space
   - Multi-channel image handling
   - Normalization to model requirements

2. **Model Architecture**:
   - Encoder-decoder structure with skip connections
   - Double convolution blocks with batch normalization
   - 4-level feature extraction (64, 128, 256, 512 features)

3. **Inference Pipeline**:
   - Test-time augmentation (4-fold)
   - Adaptive threshold selection using Otsu's method
   - Post-processing with morphological operations

4. **Visualization**:
   - Overlay visualization with customizable opacity
   - Heatmap generation for confidence visualization
   - Binary mask output for further analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.11.0
- CUDA-compatible GPU (recommended for faster inference)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd neuroscan-ai
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the pre-trained model:
   - Place `unet_brain_tumor.pth` in the project root directory

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Upload a brain MRI scan (JPG, PNG, TIF, or BMP format)

3. Configure analysis settings in the sidebar:
   - Enable/disable Test-Time Augmentation
   - Adjust overlay opacity

4. Click "Run Segmentation" to process the image

5. View results in the tabs:
   - Overlay: Original image with tumor overlay
   - Side-by-Side: Original and segmented images
   - Heatmap: Confidence probability map
   - Mask: Binary segmentation mask

6. Download results using the export buttons

## Model Details

- **Architecture**: U-Net with skip connections
- **Input Size**: 256×256 pixels
- **Training Data**: TCGA-LGG and similar brain MRI datasets
- **Output**: Binary segmentation mask of tumor regions
- **Device**: Automatically uses GPU if available, falls back to CPU

## Technical Implementation

### Key Components

1. **Preprocessing**:
   - CLAHE enhancement in LAB color space for improved contrast
   - Multi-channel handling for various input formats

2. **Inference**:
   - 4-fold Test-Time Augmentation (horizontal flip + brightness variations)
   - Adaptive thresholding using Otsu's method with fallback

3. **Post-processing**:
   - Morphological closing and opening operations
   - Minimum area filtering to remove noise

4. **Visualization**:
   - Color-coded overlays (blue for tumor regions)
   - Heatmap visualization with Inferno colormap
   - Export functionality for all results

### Performance Optimizations

- Efficient U-Net implementation with skip connections
- GPU acceleration when available
- Optimized image processing pipeline
- Memory-efficient inference with batch processing

## Contributing

We welcome contributions to improve NeuroScan AI! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **Research Use Only** - This tool is not a medical device. Always seek a qualified radiologist for clinical diagnosis.

## Contact

For questions or support, please open an issue on the GitHub repository.
