# 🦴 Bone Fracture Detection Using X-Ray Images

This project focuses on detecting **bone fractures from X-ray images** using **Machine Learning (XGBoost)**.  
Image features are extracted from X-ray images and used to classify them as **Fractured** or **Healthy** bones.

---

## 📌 Project Overview

Bone fractures are commonly diagnosed using X-ray images, which requires expert medical analysis.  
This project aims to assist learning and research by building a **machine learning–based fracture detection system**.

The workflow includes:
- X-ray image preprocessing
- Feature extraction
- Classification using **XGBoost**
- Model evaluation and deployment using **Streamlit**

---

## 🧠 Machine Learning Model

### Model Used
- **Algorithm:** XGBoost (Extreme Gradient Boosting)
- **Task:** Binary Classification
- **Input:** Extracted features from X-ray images
- **Output:** Fractured / Healthy

### Why XGBoost?
- High performance on structured feature data
- Handles non-linear relationships efficiently
- Provides feature importance for better interpretability

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🧪 Dataset

The dataset used in this project is publicly available on Kaggle:

🔗 **X-Ray Images of Fractured and Healthy Bones**  
https://www.kaggle.com/datasets/foyez767/x-ray-images-of-fractured-and-healthy-bones

### Dataset Details
- X-ray images of various bones
- Two classes:
  - **Fractured**
  - **Healthy**

> ⚠️ **Note:**  
> The dataset is **not included** in this repository due to size constraints.  
> Please download it manually from Kaggle.

---

## ⚙️ Project Structure

