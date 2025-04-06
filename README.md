# Automated-Pneumonia-Classification-from-Chest-X-rays-Using-Deep-learning-CNN

# Pneumonia Detection Using Convolutional Neural Networks (CNN)

This project presents an end-to-end deep learning pipeline for detecting pneumonia from chest X-ray images using a Convolutional Neural Network built with PyTorch.

## Dataset Structure

The dataset is organized into `train/` and `test/` folders, each containing subfolders for the `NORMAL` and `PNEUMONIA` classes. The image distribution is visualized to ensure class balance.

## Workflow Overview

![image](https://github.com/user-attachments/assets/7b906113-e0a5-4f20-9965-39557b57a42f)


### 1. Data Exploration and Preprocessing

- Count and visualize image distribution across classes
- Display sample images for both categories
- Analyze image dimensions for consistency
- Normalize and resize images to a uniform shape (224x224)

### 2. Data Augmentation and Class Balancing

- Augment underrepresented classes using torchvision transforms
- Combine original and augmented datasets
- Visualize class distribution after augmentation
- Use a weighted sampler and weighted loss to address class imbalance

### 3. Model Architecture

A custom CNN is defined with the following layers:
- Three convolutional layers with ReLU activation and max pooling
- A fully connected classifier with dropout for regularization
- Output layer for binary classification

### 4. Training and Validation

- Train on 80% of the dataset with the remaining 20% used for validation
- Track accuracy and loss for both training and validation
- Save the best model based on validation accuracy

### 5. Inference

- Load the saved model and run predictions on new images
- Display the predicted class label alongside the input image.


![image](https://github.com/user-attachments/assets/427220a0-b4c9-42fe-a2bf-fbc294b2ee6a)

Summary and Result
The model has processed a chest X-ray image and classified it as PNEUMONIA. The image displays typical indicators that the convolutional neural network has been trained to associate with pneumonia cases, such as increased opacity and irregular lung patterns.

This prediction result suggests that the trained CNN model is able to identify pathological signs of pneumonia from chest radiographs. However, it’s important to note that this classification should not replace clinical judgment. The model’s output should be used as a supportive diagnostic tool and verified by a radiologist or physician.



## How to Run

1. Place your dataset in `train/` and `test/` folders under the project root
2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib pillow tqdm
   ```
3. Run the training script to train and validate the model
4. Use the inference script to test predictions on individual X-ray images

## Notes

- The model uses weighted cross-entropy loss and a balanced sampling strategy to handle class imbalance
- Data augmentation is manually applied to strengthen the model's generalization
- Results are visualized using matplotlib for easy interpretation

