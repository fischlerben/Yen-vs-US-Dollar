# Findings and Conclusions

## Time-Series Conclusions:
Based on the time series analysis, I would *not* in fact buy the yen now.  First off, both the ARMA and the ARIMA model have p-values that are greater than .05 (.42 and .65, respectively), and therefore, the coefficient for the autoregressive term is *not* statistically significant and those terms should not be kept in the models.  Additionally, as the upward-trending GARCH Model shows us, the exchange rate risk is expected to increase, and a more conservative investor may not be comfortable with this level of risk.  Although I would not use either of these models, the AIC of the ARMA model (15,798) is significantly lower than that of the ARIMA model (83,905), and therefore it is performing significantly better.

When it comes to investing real money, I would not base my decisions solely on the results of these models.  Before using them, I would want to improve them by training the models and making them statistically significant.   If I did that, I would use them as one of the factors I consider, but would want to look at other factors as well before making any investment decisions.

However, a more opportunistic investor may take a look at this GARCH plot, expect increased short-term volatility in the markets, and invest in derivatives.  Prices of derivative assets tend to increase as volatility increases, and a prudent investor may take advantage of this.

## Linear Regression Conclusions:    
The out-of-sample RMSE (.42) is lower than the in-sample RMSE (.60). RMSE is typically lower for training data, but is higher in this case. This means the model made better predictions on data it has never seen before (the test set) than the actual training set.  Therefore, I would *not* trust these predictions, and would instead develop a new model.