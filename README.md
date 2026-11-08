# Data Science and Analytics (IT-411) — Lab Practical Repository

**Candidate:** Dhruv Sarin  
**Course:** IT-411 Data Science and Analytics (7th Semester, Information Technology)  
**Institution:** Indian Institute of Information Technology, Bhopal  
**Instructor:** Dr. Rekha Kaushik, Assistant Professor, Department of Information Technology  
**Repository:** [https://github.com/DhruvSaringit/DSA_practical.git](https://github.com/DhruvSaringit/DSA_practical.git)

---

## Overview

This repository contains the end-to-end practical implementations, reproducible Jupyter notebooks, and datasets for all 15 experiments prescribed in the **IT-411 Data Science and Analytics** curriculum.

> **Candidate Uniqueness & Anti-Plagiarism Compliance:**  
> All 15 practical experiments use **100% unique datasets** strictly distinct from peer submissions. Every experiment includes an explicit Dataset Description, Aim, Software/Libraries, Theoretical Foundation, Procedure, Clean Code Implementation, Measured Observations (with empirical tables), and Critical Conclusions.

---

## Index of Practical Experiments

| Exp No. | Experiment Title | Unique Dataset Used | Dimensions / Scope | Notebook |
| :---: | :--- | :--- | :--- | :--- |
| **01** | Exploratory Data Analysis (EDA) on a Real-World Dataset Using Python | Medical Insurance Cost Dataset | 1,338 rows × 7 features | [`exp01_eda.ipynb`](notebooks/exp01_eda.ipynb) |
| **02** | Visualization of Data Using Charts, Histograms, and Scatter Plots in Python | UCI Seeds Kernel Morphology Dataset | 210 rows × 8 features (3 classes) | [`exp02_data_visualization.ipynb`](notebooks/exp02_data_visualization.ipynb) |
| **03** | Implement Linear, Ridge, and Lasso Regression in Python | Scikit-Learn Diabetes Progression Dataset | 442 rows × 10 features (continuous) | [`exp03_regression.ipynb`](notebooks/exp03_regression.ipynb) |
| **04** | Implement SVM and KNN Classifiers in Python | Pima Indians Diabetes Diagnostic Dataset | 768 rows × 8 features (binary) | [`exp04_svm_knn.ipynb`](notebooks/exp04_svm_knn.ipynb) |
| **05** | Implement Naive Bayes Model in Python for Text Classification | IMDB Movie Reviews Sentiment Dataset | 2,500 balanced texts (TF-IDF) | [`exp05_naive_bayes_text.ipynb`](notebooks/exp05_naive_bayes_text.ipynb) |
| **06** | Implement Decision Tree and Random Forest Models in Python | Cleveland Heart Disease Clinical Dataset | 297 complete rows × 13 clinical features | [`exp06_decision_tree_random_forest.ipynb`](notebooks/exp06_decision_tree_random_forest.ipynb) |
| **07** | Implement Dimensionality Reduction Using PCA in Python | Breast Cancer Wisconsin Diagnostic Dataset | 569 rows × 30 nuclear morphology features | [`exp07_pca.ipynb`](notebooks/exp07_pca.ipynb) |
| **08** | Implement Artificial Neural Network Using Python | Fashion-MNIST Benchmark Dataset | 70,000 images (28×28), 10 apparel classes | [`exp08_ann.ipynb`](notebooks/exp08_ann.ipynb) |
| **09** | Implement CNN Model Using Python to Classify Bird Species | Curated 5 Bird Species Image Corpus | 300 high-res images (5 distinct species) | [`exp09_bird_species_cnn.ipynb`](notebooks/exp09_bird_species_cnn.ipynb) |
| **10** | Implement K-Means Clustering in Python for Data Segmentation | UCI Wholesale Customers Spending Dataset | 440 commercial accounts × 6 spend categories | [`exp10_kmeans_clustering.ipynb`](notebooks/exp10_kmeans_clustering.ipynb) |
| **11** | Analysis and Implementation of Performance Metrics in Python | Cleveland Heart Disease Clinical Dataset | 297 rows × 13 features (diagnostic target) | [`exp11_performance_metrics.ipynb`](notebooks/exp11_performance_metrics.ipynb) |
| **12** | Implement Hierarchical Clustering and Dendrogram Analysis in Python | UCI Seeds Morphology Dataset | 210 rows × 7 features (3 ground-truth varieties) | [`exp12_hierarchical_clustering.ipynb`](notebooks/exp12_hierarchical_clustering.ipynb) |
| **13** | Install Hadoop and Explore HDFS Using Basic Commands | Telemetry Sensor Stream (`telemetry_logs_2026.csv`) | 1,000 event records (NameNode/DataNode) | [`exp13_hadoop_hdfs.ipynb`](notebooks/exp13_hadoop_hdfs.ipynb) |
| **14** | Install Apache Spark and Explore Its Core Features | E-Commerce Orders Dataset (`ecommerce_orders.csv`) | 100 transactions × 8 features (PySpark DAG) | [`exp14_apache_spark.ipynb`](notebooks/exp14_apache_spark.ipynb) |
| **15** | Implement NoSQL Database Operations Using MongoDB in Python | Smart Healthcare System (`hospital_db.patients`) | Semi-structured BSON documents with nested vitals | [`exp15_mongodb.ipynb`](notebooks/exp15_mongodb.ipynb) |

---

## Directory Structure

```text
.
├── datasets/
│   ├── cleveland_heart.csv
│   ├── ecommerce_orders.csv
│   ├── insurance.csv
│   ├── movie_data.csv.gz
│   ├── pima_diabetes.csv
│   ├── seeds.csv
│   └── wholesale_customers.csv
├── notebooks/
│   ├── exp01_eda.ipynb
│   ├── exp02_data_visualization.ipynb
│   ├── exp03_regression.ipynb
│   ├── exp04_svm_knn.ipynb
│   ├── exp05_naive_bayes_text.ipynb
│   ├── exp06_decision_tree_random_forest.ipynb
│   ├── exp07_pca.ipynb
│   ├── exp08_ann.ipynb
│   ├── exp09_bird_species_cnn.ipynb
│   ├── exp10_kmeans_clustering.ipynb
│   ├── exp11_performance_metrics.ipynb
│   ├── exp12_hierarchical_clustering.ipynb
│   ├── exp13_hadoop_hdfs.ipynb
│   ├── exp14_apache_spark.ipynb
│   └── exp15_mongodb.ipynb
└── README.md
```

## Environment & Requirements

- Python 3.11 / 3.12
- Libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`/`torch`, `pyspark`, `pymongo` / `mongomock`
