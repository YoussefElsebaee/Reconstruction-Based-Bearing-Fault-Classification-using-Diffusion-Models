# Reconstruction-Based Bearing Fault Classification using Diffusion Models

This repository contains the source code and supplementary files for my bachelor thesis, which investigates a **generative, reconstruction-based approach to bearing fault classification** using diffusion models and convolutional neural networks (CNNs).

Instead of formulating fault diagnosis purely as a discriminative classification problem, this work explores whether **diffusion-based signal reconstruction** can preserve fault-related characteristics that remain informative for downstream classification.

---

## 📌 Thesis Overview

**Title:** Reconstruction-Based Bearing Fault Classification using Diffusion Models  
**Author:** Youssef Elsebaee  
**Degree:** Bachelor of Science in Automation and Control Engineering  
**Institution:** German International University (GIU)  
**Supervisor:** Prof. Dr. Marco Wagner  

The thesis focuses on:
- Generative modeling of vibration signals using diffusion models
- Reconstruction of bearing fault signals under different noise schedules
- Feature extraction using log-mel spectrograms
- CNN-based classification of reconstructed signals

---

## 🧠 Methodology Summary

The implemented pipeline consists of the following main stages:

1. **Signal Segmentation & Preprocessing**
   - Raw vibration signals are segmented into fixed-length windows
   - Zero-padding is applied where necessary to ensure dimensional consistency

2. **Feature Extraction**
   - Each segment is transformed into a log-mel spectrogram using STFT
   - Parameters are fixed across experiments to ensure fair comparison

3. **Diffusion-Based Reconstruction**
   - Diffusion models are trained to reconstruct signals from noisy inputs
   - Multiple noise schedules and reconstruction step counts are evaluated

4. **CNN Classification**
   - Reconstructed signals are fed into a CNN classifier
   - Classification performance is compared against baseline approaches

---

## 📊 Key Results

- The best-performing configuration achieved a classification accuracy of **~81%**
- Reconstruction quality was found to be sensitive to noise scheduling
- While the approach demonstrated feasibility, robustness to noise remains a limitation

Detailed analysis and discussion can be found in the thesis manuscript.

---

## 📂 Repository Structure

```text
.
├── data preprocessing/
├── training files/        
├── testing files/     
├── classification/      
├── visualization/          
└── README.md
