# Comparing Classifiers: Fisher's Linear Discriminant vs. Random Forest  

## Overview  
This project explores the performance of two classification algorithms, **Fisher's Linear Discriminant (FLD)** and **Random Forest (RF)**, applied to a dataset containing key indicators of heart disease. The dataset, sourced from a 2022 CDC survey of over 400,000 adults, highlights the importance of addressing risk factors such as high blood pressure, high cholesterol, and smoking to mitigate heart disease.  

The objective was to preprocess the data, train the classifiers, and evaluate their performance based on computational efficiency, accuracy, and other metrics, including confusion matrices and ROC curves.  

## Dataset  
- **Source:** [Kaggle - Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)  
- **Size:** 246,013 data points (after preprocessing)  
  - Training set: 184,509 samples  
  - Testing set: 61,504 samples  
- **Key features:** Age, blood pressure, cholesterol levels, smoking status, and more.  
- **Target:** Binary classification of heart disease presence.  

## Methods  
1. **Data Preprocessing**  
   - Dropped missing values and duplicates.  
   - Encoded categorical data into numeric values.  
   - Grouped age into four categories: Young Adult, Adult, Middle-Aged Adult, and Senior.  
   - Split data into training (75%) and testing (25%) sets using `train_test_split` from Scikit-learn.  

2. **Classification Models**  
   - **Fisher's Linear Discriminant (FLD):** Used `LinearDiscriminantAnalysis` from Scikit-learn.  
   - **Random Forest (RF):** Used `RandomForestClassifier` from Scikit-learn.  

3. **Performance Evaluation**  
   - Measured training and testing times.  
   - Evaluated accuracy, precision, recall, and F1-score.  
   - Generated confusion matrices and ROC curves for comparison.  

## Results  
| Model | Training Time (sec) | Testing Time (sec) | Training Accuracy (%) | Testing Accuracy (%) |  
|-------|----------------------|-------------------|-----------------------|----------------------|  
| FLD   | 1.17                 | 0.07              | 93.98                 | 93.93                |  
| RF    | 37.52                | 2.56              | 99.99                 | 94.80                |  

### Visualizations  
- **Confusion Matrices:** Highlighted the misclassification rates for each model.  
- **Density Plot:** Compared the probability distributions of predictions.  
- **ROC Curves:** Demonstrated the trade-off between true positive rate and false positive rate for both classifiers.  

## Technologies Used  
- **Programming Language:** Python  
- **Libraries:**  
  - `numpy`, `pandas`, `matplotlib`, `seaborn`: Data analysis and visualization  
  - `sklearn`: Machine learning models and evaluation metrics  

## Key Takeaways  
- **FLD** is faster and computationally efficient, making it suitable for simpler, real-time applications.  
- **RF** provides higher accuracy at the cost of greater computational resources, making it ideal for complex datasets and tasks.  


