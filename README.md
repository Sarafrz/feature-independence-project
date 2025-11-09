Feature Independence & Optimization Analysis

This project explores how linear independence and feature-space quality influence the performance and convergence of machine learning algorithms. The goal is to help students understand how reducing collinearity improves optimization stability—both in derivative-based (e.g., SGD) and derivative-free (e.g., KMeans) methods.

Objectives

Extract independent / low-collinearity features using PCA, ICA, SVD.

Apply basic feature selection methods (SelectKBest, RFE, variance filters).

Analyze covariance, correlation, and the effect of feature redundancy.

Study how feature quality affects:

Regression (OLS vs SGD)

Classification (KNN vs RandomForest)

Clustering (KMeans optimization behavior)

Datasets

WDBC – Classification

Boston Housing – Regression

Iris – Clustering

What the Project Shows

PCA and other extraction methods create orthogonal features that often stabilize optimization.

SGD typically converges faster in de-correlated, reduced spaces.

KMeans and KNN (distance-based models) are sensitive to dimensionality, improving after PCA.

RandomForest is generally robust to collinearity, showing less change across spaces.

Repository Structure
data/        # Raw datasets
notebooks/   # Three Jupyter notebooks (classification, regression, clustering)
outputs/     # Figures (heatmaps, EVR, loss curves) and tables of metrics

Workflow Summary

Load & standardize data

Analyze collinearity (covariance + heatmaps)

Apply PCA / ICA / SVD

Apply SelectKBest / RFE

Train models on:

original features

PCA features

selected features

Compare performance, convergence, stability

Requirements
numpy
pandas
scikit-learn
matplotlib
seaborn
