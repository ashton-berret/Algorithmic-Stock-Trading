# CSC-4444

* Goal: Predict the next candlestick for a given sequence of candlesticks

# Work Done:

* Features:
  * Volatility
  * Momentum 
  * Price-to-Earnings Ratio
  * Sentiment of News Data and Tweets
  
* Models:
  * Linear Regression
  * Ridge Regression
  * Lasso Regression
  * Ridge/Lasso Combination

# Running the project

Make sure you have Conda installed. You can download and install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (a minimal installer for Conda) or [Anaconda](https://www.anaconda.com/download) (which includes Conda and other useful tools).

1. Create a Conda environment

```bash
conda create --name env python=3.10.8
```

2. Activate the enviornment 

```bash
conda activate env
```

3. Install project dependencies

```bash
pip install -r requirements.txt
```

4. Run the project 

```bash
python run_project.py
```
---

# 👍

# Linear Regression Model

* Using historical price, four models are compiled to predict `Open`, `Close`, `High`, and `Low` prices.

* To train the models, the target price within the data was pushed back one row (one day of data) thus allowing our inputs to be alligned with the next day's target price. (ie: `Open`, `Close`, `High`, `Low`)

    * $X$ = All values except the target price:

        - Three of (`Open`, `Close`, `High`, `Low`)

        - Adj Close

        - Volume

    * $Y$ = target price: One of (`Open`, `Close`, `High`, `Low`)

* $X$ and $Y$ are then split into $80$% training data and $20$% testing data.

* The linear regression model finds target prices by:

$$y = (\sum_{i = 1}^{5} B_i x_i) + B_0 + \epsilon$$

where:

- $y$ is the target price

- $B_i$ is the coefficient for $x_i$, the $i^{th}$ feature

- $B_0$ is the intercept

- $\epsilon$ is the error term

* The models are then used to predict their respective target prices for the entire dataset (both training and testing data).
            

# Lasso Regression Model 

* Using historical price, four models are compiled to predict `Open`, `Close`, `High`, and `Low` prices.

* To train the models, the target price within the data was pushed back one row (one day of data) thus allowing our inputs to be alligned with the next day's target price. (ie: `Open`, `Close`, `High`, `Low`)

    * $X$ = All values except the target price:

        - Three of (`Open`, `Close`, `High`, `Low`)

        - Adj Close

        - Volume

    * $Y$ = target price: One of (`Open`, `Close`, `High`, `Low`)

* $X$ and $Y$ are then split into $80$% training data and $20$% testing data.

* The data is then standardized:

$$z = \frac{x - \mu}{\sigma}$$

where:

- $z$ is the standardized price

- $x$ is the original price

- $\mu$ is the mean of the original price

- $\sigma$ is the standard deviation of the original price

* Lasso Regression is then performed with a hyperparameter grid search for alpha values in the range $[10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100]$.

* In Lasso Regression, the model learns by minimizing the cost function:

$$COST = \sum_{i = 1}^{5} (y_i - \hat{y}_i)^2 + \alpha\sum_{i = 1}^{5}|\beta_i|$$

where:

- $\beta_i$ are the model coefficients

- $\alpha_i$ is the regularization parameter

* The model learns the best coefficients $\beta_i$ that minimize $COST$.

* The regularization term penalizes larger coefficients, thus preventing overfitting. It has the potential to reduce some coefficients to zero, thus performing feature selection.

* The hyperparameter grid search is performed using K-fold cross validation, with $k = 5$, to divide the dataset into 5 nearly equal-sized folds.
    - For each fold, the model is trained on 4 of the folds and tested on the remaining fold.
    - Once all 5 folds have been tested, the model performance is evaluated using Mean Squared Error:

$$MSE = \frac{1}{5} \sum_{i = 1}^{5} (y_i - \hat{y}_i)^2$$

where:

- $y_i$ are the true target prices

- $\hat{y}_i$ are the predicted target prices

* The model with the lowest MSE is then selected as the best model.

* The best models are then used to predict their respective target prices for the entire dataset (both training and testing data).
            

# Ridge Regression Model 

* Using historical price, four models are compiled to predict `Open`, `Close`, `High`, and `Low` prices.

* To train the models, the target price within the data was pushed back one row (one day of data) thus allowing our inputs to be alligned with the next day's target price. (ie: `Open`, `Close`, `High`, `Low`)

    * $X$ = All values except the target price:

        - Three of (`Open`, `Close`, `High`, `Low`)

        - Adj Close

        - Volume

    * $Y$ = target price: One of (`Open`, `Close`, `High`, `Low`)

* $X$ and $Y$ are then split into $80$% training data and $20$% testing data.

* The data is then standardized:

$$z = \frac{x - \mu}{\sigma}$$

where:

- $z$ is the standardized price

- $x$ is the original price

- $\mu$ is the mean of the original price

- $\sigma$ is the standard deviation of the original price

* Ridge Regression is then performed with a hyperparameter grid search for alpha values in the range $[10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100]$.

* In Ridge Regression, the model learns by minimizing the cost function:

$$COST = \sum_{i = 1}^{5} (y_i - \hat{y}_i)^2 + \alpha\sum_{i = 1}^{5}\beta_i^2$$

where:

- $\beta_i$ are the model coefficients

- $\alpha_i$ is the regularization parameter 

* The model learns the best coefficients $\beta_i$ that minimize $COST$.

* The regularization term penalizes larger coefficients, thus preventing overfitting.

* The hyperparameter grid search is performed using K-fold cross validation, with $k = 5$, to divide the dataset into 5 nearly equal-sized folds.
    - For each fold, the model is trained on 4 of the folds and tested on the remaining fold.
    - Once all 5 folds have been tested, the model performance is evaluated using Mean Squared Error:

$$MSE = \frac{1}{5} \sum_{i = 1}^{5} (y_i - \hat{y}_i)^2$$

where:

- $y_i$ are the true target prices

- $\hat{y}_i$ are the predicted target prices

* The model with the lowest MSE is then selected as the best model.

* The best models are then used to predict their respective target prices for the entire dataset (both training and testing data).
