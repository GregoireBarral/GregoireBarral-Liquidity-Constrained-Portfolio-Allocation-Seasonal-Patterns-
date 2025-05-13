import polars as pl
import pandas as pd 
import numpy as np
from scipy.stats import norm, kurtosis



class VaR_module: 
    
    
    def VaR_99(portfolio_prices: pl.DataFrame, filter: list, alpha=0.01, portfolio_value=100, scenario = 50000,kurt=None):
        """
        Return a measure of the parametric Value at Risk (VaR) at a given confidence level.using a modified approach 

        Args:
            portfolio_prices (pl.DataFrame): Historical prices of the portfolio components.
            filter (list): List of column names in portfolio_prices to include in the VaR calculation.
            alpha (float): Significance level (default 0.01 for 99% confidence).
            portfolio_value (float): Total value of the portfolio (default $100).

        Returns:
            float: One-day Value at Risk.
        """
        def simulate_student_t(mean_returns, cov_matrix, df, size):
            
            n_assets = len(mean_returns)
            g = np.random.chisquare(df, size=(size, 1))  # Chi-squared variable
            z = np.random.multivariate_normal(np.zeros(n_assets), cov_matrix, size=size)
            t_samples = mean_returns + z / np.sqrt(g / df)
            return t_samples

        # Convert to pandas for easier return calculation
        portfolio_cp = portfolio_prices.to_pandas()

        portfolio_cp  = portfolio_cp.set_index("Date")

        portfolio_cp = portfolio_cp[portfolio_cp.index > "2018-12-31"]

        portfolio_cp = portfolio_cp[filter]

        # Calculate arithmetic returns and drop nan
        returns = np.log(portfolio_cp/portfolio_cp.shift(1)).dropna()

        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean().values
        cov_matrix = returns.cov()

        # Calculate weights
        n_assets = len(filter)
        weights = np.full(n_assets, 1 / n_assets)

        #Estimate degrees of freedom from kurtosis
        excess_kurt = kurtosis(returns, fisher=True).mean()
        df = max(4.1, (6 / excess_kurt) + 4) if excess_kurt > 0 else 1000
        simulated_returns = simulate_student_t(mean_returns, cov_matrix, df=df, size=scenario)

        simulated_returns = np.exp(simulated_returns) -1

        simulated_portfolio_returns = simulated_returns @ weights

        # One-day VaR
        VaR_1d = -np.percentile(simulated_portfolio_returns, alpha*100)

        return pd.Series(VaR_1d).map("{:.2%}".format)

    def CVaR_99(portfolio_prices: pl.DataFrame,  filter: list,alpha = 0.01, simulations = 50000, kurt=None):

        """
        objective : get the expected shortfall using a modified approach and student distribution to handle excess kurtosis 
        
        
        """
        def simulate_student_t(mean_returns, cov_matrix, df, size):
            n_assets = len(mean_returns)
            g = np.random.chisquare(df, size=(size, 1))  # Chi-squared variable
            z = np.random.multivariate_normal(np.zeros(n_assets), cov_matrix, size=size)
            t_samples = mean_returns + z / np.sqrt(g / df)

            return t_samples

        portfolio_cp = portfolio_prices.to_pandas()

        portfolio_cp  = portfolio_cp.set_index("Date")

        portfolio_cp = portfolio_cp[portfolio_cp.index > "2018-12-31"]

        portfolio_cp = portfolio_cp[filter]

        # Calculate arithmetic returns and drop nan
        returns = np.log(portfolio_cp/portfolio_cp.shift(1)).dropna()

        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean().values
        cov_matrix = returns.cov()

        # Calculate weights
        n_assets = len(filter)
        weights = np.full(n_assets, 1 / n_assets)

        excess_kurt = kurtosis(returns, fisher=True).mean() #Get the average excess kurtosis among each stock in the portfolio
        df = max(4.1, (6 / excess_kurt) + 4) if excess_kurt > 0 else 1000 # no excess kurtosis assume normality
        simulated_log_returns = simulate_student_t(mean_returns, cov_matrix, df=df, size=simulations)

        simulated_returns = np.exp(simulated_log_returns)-1

        portfolio_returns_simulated = simulated_returns @ weights

        # Sort the losses and the gains in order : 

        sorted_losses = np.sort(portfolio_returns_simulated)

        get_cut_off = int(simulations * alpha)

        expected_shortfall = -np.mean(sorted_losses[:get_cut_off])

        return pd.Series(expected_shortfall).map("{:.2%}".format)
