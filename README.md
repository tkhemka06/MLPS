# Predicting Vaccine Uptake and COVID-19 Test Positivity Rates Using Behavioral and Belief Indicators

## Overview

This project was developed as part of the course *Machine Learning for Problem Solving (95-828)* at Carnegie Mellon University. The goal is to assist the Centers for Disease Control and Prevention (CDC) in supporting public health decisions by predicting vaccine uptake and COVID-19 test positivity rates based on behavioral and belief survey data.

The analysis is built on the **COVID-19 Trends and Impact Survey (CTIS)** dataset, provided by the Delphi Group at CMU and Facebook. Our models help address key public health questions:

* Can vaccine uptake be predicted from behavior and belief indicators?
* Can COVID-19 test positivity rates be predicted from the same?

## Dataset

The dataset includes survey responses at the U.S. county level collected from Jan 7 to Feb 12, 2021. Key features are categorized into:

* **COVID Activity Indicators**
* **Behavioral Indicators**
* **Belief Indicators**
* **Metadata (time, geo location)**

Target variables:

* `smoothed_wcovid_vaccinated`
* `smoothed_wtested_positive_14d`

Data cleaning, imputation, and normalization steps are covered in the EDA phase.

## Project Structure
```
MLPS_Proj_repo-main/
├── Dataset/                       # Contains original and preprocessed data
├── EDA/                           # Scripts for Exploratory Data Analysis
│   └── EDA_class.py
├── Models/                        # Implementation of ML models
│   ├── mlp.py
│   └── transformer.py
├── Model_weights/                 # Trained model weights (if applicable)
├── MLPS_Project_Final.ipynb       # Main Jupyter Notebook
├── train.py                       # Training pipeline for models
├── requirements.txt               # Project dependencies
└── README.md
```

## Approach

### Phase I: Understanding the Problem

* Defined the objectives in alignment with CDC needs.
* Identified the target variables and relevant predictors.

### Phase II: Data Exploration & Cleaning

* Handled missing values and outliers.
* Normalized features and reduced dimensionality.
* Used correlation analysis to study predictor relationships.

### Phase III: Modeling & Evaluation

#### Models Used:

* **Linear Regression** (Baseline)
* **Multi-Layer Perceptron (MLP)**
* **Transformer-based Regression**

#### Evaluation Metrics:

* RMSE, MAE, and R²
* Cross-validation with standard deviation reporting

#### Best Performing Setup:

* Transformer model with appropriate learning rate scheduling and dropout.
* Significant performance gains over baseline models in both tasks.

## Key Findings

* Strong correlations were found between belief-based indicators and vaccine uptake.
* Predictive models for test positivity rate performed slightly worse, suggesting noise in daily testing behaviors.
* Public health policies targeting **community trust** (e.g., belief in health officials) could improve vaccine uptake significantly.

## Policy Recommendations

* **Localized vaccine outreach** based on predictive modeling to target hesitant regions.
* **Use of trust-building channels**, such as WHO or healthcare professionals, to increase uptake.
* **Dynamic resource allocation** based on real-time predictions of positivity rates.

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Code

To train and evaluate models, run:

```bash
python train.py
```

Make sure the dataset is placed under the `Dataset/` directory and properly preprocessed before training.

## Requirements

See `requirements.txt` for full list of packages:

* torch
* torchvision
* torchaudio
* scikit-learn
* pandas
* numpy
* matplotlib
* tqdm

## Acknowledgements

This project was developed for educational purposes under the course 95-828 at Carnegie Mellon University. Data was provided by the [Delphi Group](https://delphi.cmu.edu/) and Facebook’s COVID-19 Trends and Impact Survey.
