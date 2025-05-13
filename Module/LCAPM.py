import yfinance as yf 
import pandas as pd 
import numpy as np 
import plotly as ply
import plotly.express as px
import time
import xlwings 
import os
import datetime as dt 
import pyarrow
import plotly.graph_objects as go

#polars is MIT library that offers a new generation of dataframe faster than pandas and more complete 

import polars as pl


class lcapm_class:
    
    def get_market_cap(directory_BBG): 

        """
        using a bloomberg function BDH compute the market cap historically on a monthly basis

        args:

        =BDH("AAPL US Equity", "CUR_MKT_CAP", "01/01/2020", "12/31/2024", "Period", "M") this function in BBG Excel addins allows to visualize the 
        the market capitalization for the selected stocks over the selected period this can be achieved using the function builder. 

        output = a polars data frame containint the values of the market capitalization
        
        """

        market_cap = pl.read_excel(os.path.join(directory_BBG,"market_cap.xlsx"))

        return market_cap


        
    
    def relative_illiquidity_cost(monthly_illiq:pl.DataFrame , stockname, market_cap: pl.DataFrame):

        """
        obtain the relative illiquidity cost of each stock over the period as it specified by Amihud 

        args: 

        illiq = average monthly illiquidity
        values = stock prices
        market_cap of stock traded on the market at the given date
        
        """
        #Convert all values to a pandas dataframe: 

        market_cap =  market_cap.with_columns([pl.col(c).alias(c) for c in stockname]).to_pandas()

        #NAME the variable with an illiq notion
        monthly_illiq = monthly_illiq.with_columns([pl.col(c).alias(c + "_illiq")  for c in stockname]).to_pandas()
        
        #loop throught the columns of the selected stock to compute the illiqudity based on the Amihud measure:

        illiq_cost = pd.DataFrame()
        illiq_cost["Date"] = market_cap["Date"]


        for stock in stockname:

            cap = market_cap[stock]
            illiq = monthly_illiq[stock + "_illiq"]

            portfolio_cap = cap.shift(1) / cap.iloc[0]
        
        #Choose the minimum illiquidity for each stock
        illiq_cost = np.minimum(0.25 + 0.30*monthly_illiq* portfolio_cap ,30.00)
        
        return pl.DataFrame(illiq_cost)

    #Answer to this question what is the expected market return ( which model to use and same question for the expected current illiquidity based on the privious input)

    def get_portfolio_illiquidity_cost(illiq_cost, weights = None):

        """
        Getting the portfolio illiquidity cost over all the stock  will return an idea of 
        the possibilities when optimizing our ptf with AIILIQ measure of Amihud 

        args: 

        illiq_cost = dataframe of the illiquidity cost for each stock consider in out ptf

        stock_list = list of stock selected when choosing the best Sharpe ratio using ILLIQ

        weights = automatically computed depending on the lenght of the the stock list 
        
        """
        col_date = "Date"
        stock_col = [c for c in illiq_cost.columns if c != col_date]
        num_stocks = len(stock_col)

        
        # Default to equal weights if not provided
        if weights is None:

            weights = 1 / num_stocks
        
        #sum horizontaly accros the cols to get the illiquidity cost for the portfolio

        weighted_sum = sum(pl.col(stock) * w for stock, w  in zip(stock_col, weights))

        portfolio_illiq = illiq_cost.select([pl.col("Date"), weighted_sum.alias("Portfolio illiquidity cost")])

        return portfolio_illiq
    
    def expected_market_return(market_returns:pl.DataFrame): 

        expected_market_return = market_returns.select([
        pl.mean(pl.col(col)).alias(f"average_{col}") for col in market_returns.columns])

        return expected_market_return

   
    
    def BETAS(expected_market_return,portfolio_returns,
               stock_returns: pl.DataFrame,
               stock_illiq: pl.DataFrame,
               market_illiq:pl.DataFrame,
              stockname:list )-> pl.DataFrame:
        

        """using the Beta 1 function to determine the first Beta of the CAPM"""


        betas = []

        for stock in stockname: 

            # Convert to pandas for easier computation
            r_i = stock_returns[stock].to_pandas()
            r_m = portfolio_returns["Market"].to_pandas()
            c_i = stock_illiq[stock].to_pandas()
            c_m = market_illiq.to_pandas()

            # Estimate expectations using lagged values
            E_r_m = r_m.shift(1)
            E_c_i = c_i.shift(1)
            E_c_m = c_m.shift(1)

            r_m_demeaned = r_m - E_r_m
            c_i_demeaned = c_i - E_c_i
            c_m_demeaned = c_m - E_c_m
            
            beta1 = np.cov(r_i, r_m_demeaned)[0, 1] / np.var(r_m_demeaned)

            beta2 = np.cov(c_i_demeaned, c_m_demeaned)[0, 1] / np.var(r_m_demeaned)

            beta3 = np.cov(r_i, c_m_demeaned)[0, 1] / np.var(r_m_demeaned)

            beta4 = np.cov(c_i_demeaned, r_m_demeaned)[0, 1] / np.var(r_m_demeaned)

            betas.append([stock, beta1, beta2, beta3, beta4])

        beta_df = pd.DataFrame(betas, columns=["Stock", "Beta1", "Beta2", "Beta3", "Beta4"])

        return pl.from_pandas(beta_df)
    
    def get_lambda(betas_df,
                   portfolio_returns,
                   illiq_cost, 
                   stockname) -> pl.DataFrame :
        
        import statsmodels.api as sm

        portfolio_returns = portfolio_returns.to_pandas()
        illiq_cost = illiq_cost.to_pandas()

        return None

        



