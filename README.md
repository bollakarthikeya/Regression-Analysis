# Istanbul Stock Exchange Returns Regression Analysis

## Overview

This is an analysis of **USD-based Istanbul Stock Exchange (ISE) returns** using global market-index returns as predictors. Two complementary regression workflows are implemented:

1. Variable selection using **Best Subset Selection** with cross-validation.
2. Dimensionality reduction using **Principal Component Regression (PCR)**.

The objective is to model `USD_BASED_ISE` while addressing predictor redundancy and correlation among the seven market indices.

## Dataset

**Response variable**

- `USD_BASED_ISE`

**Predictors**

- `SP`
- `DAX`
- `FTSE`
- `NIKKEI`
- `BOVESPA`
- `EU`
- `EM`

The data is split chronologically:

- **Training:** 2009–2010
- **Test:** 2011

## Workflow

### 1. Exploratory Data Analysis (EDA)

The analysis:

- Computes summary statistics
- Examines yearly return distributions
- Compares average index returns for 2009 and 2010
- Visualizes relationships between individual predictors and ISE returns
- Computes a predictor correlation matrix and heatmap

The correlation analysis identifies substantial dependence among several market indices, motivating both feature selection and PCA/PCR.

### 2. Best Subset and Stepwise Selection

The R implementation evaluates:

- **Best Subset Selection**
- **Forward Stepwise Selection**
- **Backward Stepwise Selection**

Models containing 1–7 predictors are compared using:

- Residual Sum of Squares (RSS)
- R²
- Adjusted R²
- Mallows' Cp

All three selection methods produce the same candidate models in the implementation, so **Best Subset Selection** is used for subsequent analysis.

### 3. Cross-Validation

Best Subset Selection is combined with **10-fold cross-validation**.

For every fold:

1. Fit best-subset models containing 1–7 predictors
2. Generate validation predictions manually from the selected regression coefficients
3. Calculate Mean Squared Error (MSE)
4. Store validation errors for each model size
5. Average MSE across folds
6. Select the model size with the lowest mean validation error

The lowest validation error occurs with **3 predictors**.

The resulting model selects:

```text
BOVESPA
EU
EM
```

The final three-variable regression model is then evaluated on the 2011 test set using **R²** and **Adjusted R²**.

## Principal Component Regression

A second approach addresses multi-collinearity through **Principal Component Analysis (PCA)** followed by linear regression.

### Correlation Findings

The notebook observes:

- `EU` is correlated with `DAX` and `FTSE`.
- `SP` is correlated with `BOVESPA`.
- `EM` is correlated with `EU`, `FTSE`, `DAX`, and `BOVESPA`.
- `NIKKEI` exhibits the least correlation with the other predictors.

### PCA

A covariance matrix is computed from the training predictors and eigendecomposition is performed.

Reported cumulative variance:

```text
PC1       ≈ 65.0%
PC1–PC2   ≈ 81.6%
PC1–PC3   ≈ 91.5%
```

PCA transforms the correlated predictors into linearly uncorrelated principal-component directions.

### Component Selection

The notebook evaluates regression models using **5-fold cross-validation** across different numbers of components.

Although using all seven components gives the lowest MSE, **2 principal components** are selected as a dimensionality-reduction compromise because of the sharp improvement from one to two components.

### PCR Model

The training data is projected onto the first two eigenvectors:

```text
X_PCR = X × [PC1 PC2]
```

An Ordinary Least Squares regression model is then fitted:

```text
USD_BASED_ISE = β0 + β1 PC1 + β2 PC2 + ε
```

Both principal-component coefficients are reported as statistically significant at `α = 0.05`.

The model is evaluated on the 2011 test set using:

- R²
- Adjusted R²

## Key Results

Two modeling strategies emerge from the analysis:

**Best Subset Regression**

```text
7 predictors
     ↓
10-fold cross-validation
     ↓
3 selected predictors
     ↓
BOVESPA + EU + EM
```

**Principal Component Regression**

```text
7 correlated predictors
     ↓
PCA
     ↓
2 principal components
     ↓
≈81.6% cumulative variance
     ↓
OLS regression
```

## Packages Used

**R**

- `leaps`
- `dplyr`
- `tidyr`

**Python**

- pandas
- NumPy
- SciPy
- scikit-learn
- statsmodels
- Matplotlib
- seaborn
- PrettyTable

## Conclusion
Best Subset Selection reduces the original seven predictors to **BOVESPA, EU, and EM** using 10-fold cross-validation. PCR instead transforms the correlated market indices into principal components and builds a reduced two-component regression model capturing approximately **81.6% of the predictor variance**.

Together, the implementations demonstrate two approaches to building more compact regression models when predictors exhibit substantial correlation.
