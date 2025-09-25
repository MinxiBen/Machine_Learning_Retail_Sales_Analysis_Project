# Machine_Learning_Retail_Sales_Analysis_Project
This project analyzes a bookstore’s monthly sales data and uses Support Vector Machine (SVM) to classify whether a book is a bestseller (hot-selling) or not.
Project Overview

## Goal: 
Build a machine learning model that predicts if a book is likely to sell well(copies>=2 per months is called 'bestsellers') based on price, category, editions and reinstock time.

Method: Applied SVM (Support Vector Machine) for binary classification.

## Data: 
Monthly sales statistics, including sku, book title,price, category, sales volume, editions, and other indicators.

## Output: 
A trained model that labels books as either:

1 = Bestseller

0 = Non-bestseller

## Features

### 1. Data preprocessing 
handling format changing and forming new variables, standard scale numeric data and one-hot coding for categorical data.

### 2.Exploratory data analysis (EDA)
identify patterns in sales using jupyter notebook.

### 3.Model training 
using scikit-learn’s SVM implementation.

### 4.Model evaluation 

metrics: accuracy, precision, recall, and F1-score.

Prediction function for new book sales data.

### 5.Example Result

Achieved around 85% accuracy on test data, but recall is lower for "bestseller" books (38%) 

### 6.Future Work

Incorporate other variables (book titles).

Try other models (Random Forest, XGBoost) for comparison.
