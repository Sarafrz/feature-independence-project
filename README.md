# Feature Independence & Optimization Analysis

This project investigates how **linear independence** and **feature-space quality** affect the stability, convergence speed, and performance of machine learning models. The goal is to understand how reducing **collinearity** improves optimization for both derivative-based methods (e.g., SGD) and iterative/derivative-free methods (e.g., KMeans).

## Objectives
- Extract independent or low-collinearity features using **PCA, ICA, and SVD**.
- Apply basic feature selection (Variance Threshold, SelectKBest, RFE).
- Analyze covariance/correlation structure and identify collinearity.
- Study optimization behavior in:
  - **Regression:** LinearRegression (analytical OLS) vs SGDRegressor  
  - **Classification:** KNN vs RandomForest  
  - **Clustering:** KMeans (EM-style optimization)

## Key Ideas
- Independent or orthogonal features improve numerical stability.
- PCA reduces collinearity and often accelerates SGD convergence.
- Distance-based models (KNN, KMeans) benefit from reduced dimensions.
- Tree-based models (RandomForest) are generally robust to collinearity.

## Datasets Used
- **WDBC (Breast Cancer Diagnostic)** — Classification  
- **Boston Housing** — Regression  
- **Iris** — Clustering  


## Workflow Summary
1. Load and standardize data  
2. Analyze collinearity via covariance/correlation + heatmaps  
3. Apply feature extraction (PCA, ICA, SVD)  
4. Apply feature selection (SelectKBest, RFE, variance filtering)  
5. Train models on:  
   - original features  
   - PCA-transformed features  
   - feature-selected subsets  
6. Compare performance + optimization behavior across all feature spaces  

## Tools & Requirements
numpy
pandas
scikit-learn
matplotlib
seaborn


## Purpose
This project demonstrates how **feature design** and **optimization** are tightly connected, and why understanding independence/collinearity is essential for reliable machine learning models.

