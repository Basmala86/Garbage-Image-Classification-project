# 🗑️ Garbage Classification Project

An AI-powered image classification system that automatically categorizes waste into different types using deep learning. This project helps in waste management by identifying and sorting garbage items through computer vision.

## 🎯 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify garbage images into 6 different categories:
- **Cardboard** 📦
- **Glass** 🍾
- **Metal** 🥫
- **Organic** 🍌
- **Paper** 📄
- **Plastic** 🥤

## 🚀 Features

- **Multiple Model Types**: Custom CNN, VGG16, and ResNet50 transfer learning
- **Memory Efficient**: Handles large datasets using data generators
- **Interactive Web Interface**: Gradio-based UI for easy testing
- **Real-time Predictions**: Upload images and get instant classifications
- **Confidence Scores**: Shows prediction confidence for each class
- **Training Visualizations**: Loss and accuracy plots
- **Model Evaluation**: Confusion matrix and classification reports

## 📁 Project Structure

```
garbage-classification/
├── data/
│   └── original data/
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── organic/
│       ├── paper/
│       └── plastic/
├── notebooks/
│   └── garbage_classification.ipynb
├── models/
│   └── garbage_classifier_model.h5
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/garbage-classification.git
cd garbage-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Required packages:**
```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
gradio>=3.0.0
Pillow>=8.3.0
```

## 📊 Dataset

The project uses a garbage classification dataset with images organized in folders by category. Each category should contain hundreds to thousands of images for optimal training.

**Dataset Structure:**
```
data/original data/
├── cardboard/     # ~2000+ images
├── glass/         # ~2000+ images
├── metal/         # ~2000+ images
├── organic/       # ~2000+ images
├── paper/         # ~2000+ images
└── plastic/       # ~2000+ images
```

## 🚄 Quick Start

### Option 1: Full Training (Recommended for final model)
```python
from garbage_classifier import GarbageClassifier

# Initialize classifier
classifier = GarbageClassifier(
    data_path="data/original data",
    img_size=(224, 224),
    batch_size=32
)

# Train with transfer learning
history, train_gen, val_gen = classifier.train_model_with_generators(
    model_type='vgg16',
    epochs=50,
    validation_split=0.2
)

# Evaluate model
accuracy, y_pred, y_true = classifier.evaluate_model_with_generator(val_gen)

# Save model
classifier.save_model("garbage_classifier_model.h5")

# Launch web interface
interface = classifier.create_gradio_interface()
interface.launch(share=True)
```

### Option 2: Fast Training (For quick testing)
```python
# Train on subset for faster results
history, X_val, y_val = classifier.train_model_subset(
    model_type='custom',
    epochs=15,
    max_samples_per_class=200
)
```

## 🔧 Model Architecture

### Custom CNN
- 4 Convolutional layers with MaxPooling
- 2 Dense layers with Dropout
- Softmax output for 6 classes

### Transfer Learning Models
- **VGG16**: Pre-trained on ImageNet, fine-tuned for garbage classification
- **ResNet50**: Deep residual network with skip connections

## 📈 Performance

| Model | Accuracy | Training Time | Memory Usage |
|-------|----------|---------------|--------------|
| Custom CNN | 70-80% | 30-60 mins | Low |
| VGG16 | 85-90% | 1-2 hours | High |
| ResNet50 | 85-92% | 1-2 hours | High |

## 🖥️ Web Interface

The project includes a Gradio-based web interface that allows users to:
- Upload garbage images
- Get real-time predictions
- View confidence scores for all classes
- Access the model through a shareable link

## 📱 Usage Examples

```python
# Load trained model
classifier = GarbageClassifier(data_path="data/original data")
classifier.load_model("garbage_classifier_model.h5")

# Predict single image
predicted_class, confidence, probabilities = classifier.predict_image("test_image.jpg")
print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")

# Launch web interface
interface = classifier.create_gradio_interface()
interface.launch()
```

## 🔍 Model Evaluation

The project provides comprehensive evaluation metrics:
- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Visual representation of classification performance
- **Training History**: Loss and accuracy curves

## ⚡ Performance Tips

### For Faster Training:
- Use smaller image sizes (128x128 or 96x96)
- Increase batch size if you have sufficient RAM
- Train on subset of data for quick prototyping
- Use custom CNN instead of transfer learning

### For Better Accuracy:
- Use transfer learning (VGG16 or ResNet50)
- Implement data augmentation
- Train for more epochs with early stopping
- Use larger image resolution (224x224)

## 🚨 Memory Management

The project includes memory-efficient training methods:
- **Data Generators**: Load images on-demand during training
- **Batch Processing**: Process images in small batches
- **Memory Cleanup**: Garbage collection after training
- **Subset Training**: Train on limited samples for testing

## 📋 Requirements

### Hardware:
- **RAM**: 8GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 5GB+ for dataset and models

### Software:
- **Python**: 3.7+
- **TensorFlow**: 2.8.0+
- **CUDA**: 11.0+ (for GPU acceleration)

## 🛡️ Troubleshooting

### Common Issues:

1. **Memory Error**: Reduce batch size or use subset training
2. **Slow Training**: Use GPU acceleration or reduce image size
3. **Low Accuracy**: Increase epochs, use transfer learning, or get more data
4. **Import Errors**: Check all dependencies are installed

### Performance Optimization:
```python
# For limited memory
classifier = GarbageClassifier(
    img_size=(128, 128),
    batch_size=16
)

# For faster training
classifier.train_model_subset(
    model_type='custom',
    epochs=10,
    max_samples_per_class=100
)
```

## 📝 Future Improvements

- [ ] Add more garbage categories
- [ ] Implement data augmentation techniques
- [ ] Add mobile app deployment
- [ ] Include real-time camera feed
- [ ] Add multi-language support
- [ ] Implement model quantization for edge devices



## 🙏 Acknowledgments

- Dataset providers for garbage classification images
- TensorFlow team for the deep learning framework
- Gradio team for the easy-to-use interface
- OpenAI for inspiration and guidance

#
