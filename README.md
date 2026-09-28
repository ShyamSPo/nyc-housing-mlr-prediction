# NYC Housing Price Regression Analysis

An end-to-end data analysis and machine learning pipeline built in **R** and **RStudio** to clean, process, and model residential real estate prices across New York City boroughs.

---

## Project Overview
This project investigates the key drivers of residential property prices in New York City using public housing data. By leveraging robust data cleaning techniques, logarithmic price transformations, and multiple linear regression (MLR) with stepwise AIC variable selection, the model isolates structural attributes (square footage, bedrooms, bathrooms) and geographic locations that dictate market value.

---

## Key Features & Methodology

* **Data Preprocessing & Cleaning:** Filtered out missing entries, resolved data types, and isolated listings within realistic parameter bounds (removing bottom/top 1% price and square footage outliers to prevent luxury skew).
* **Feature Engineering:** Created derived features including `PRICE_PER_SQFT` and standardized granular localities into the 5 primary NYC boroughs (Manhattan, Brooklyn, Queens, Bronx, Staten Island).
* **Statistical Modeling:** Fitted multiple linear regression models using log-transformed prices ($\log(\text{PRICE})$) to stabilize variance across property classes and address right-skewness.
* **Variable Selection:** Implemented bidirectional stepwise selection via AIC to optimize predictor subsets and eliminate redundant features.
* **Model Evaluation:** Assessed performance using Residual Standard Error (RSE), Adjusted $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) across train/test splits.

---

## Tech Stack & Tools
* **Language:** R
* **Environment:** RStudio
* **Libraries:** `tidyverse`, `dplyr`, `ggplot2`, `caret`, `MASS`, `gridExtra`

---

## Repository Structure
```text
nyc-housing-mlr-prediction/
│
├── data/
│   ├── NY-House-Dataset.csv       # Raw input dataset
│   └── NY-House-Cleaned.csv       # Processed and filtered dataset
│
├── scripts/
│   ├── HousingPricesProject.R     # Data cleaning and feature engineering script
│   └── MLRCode.R                  # Multiple linear regression and evaluation script
│
└── outputs/
    ├── eda_price_distribution.png # Price vs. log-price distribution histograms
    ├── eda_price_vs_sqft.png      # Square footage vs. log-price scatter plots by borough
    ├── mlr_actual_vs_predicted.png# Model validation scatter plot
    ├── mlr_residuals_vs_fitted.png# Residual diagnostics plot
    ├── mlr_avg_price_by_borough.png# Borough-level average price comparison
    └── mlr_evaluation_results.csv # Final model metrics summary
