import pandas as pd
import numpy as np
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
import polars as pl
import plotly.graph_objects as go


class LinarRagression: 

    def simple_linear_regression(x,y):

        x = x
        y=  y

        X = sm.add_constant(x)
        Y = y

        model = sm.OLS(Y,X).fit()

        summary = model.summary()

        return summary
    
    def adf_test(illiq_change, stockname):

        illiq_change = illiq_change.to_pandas()


        illiq_change = illiq_change.drop(columns = "Date")

        illiq_change = illiq_change[stockname]

        adf_results = []

        for col in illiq_change.columns: 

            series = illiq_change[col].dropna()

            if series.empty: 
                continue 

            result = adfuller(series)

            adf_results.append({
            "Stock": col,
            "Test Statistic": result[0],
            "p-value": result[1],
            "Stationary": "Yes" if result[1] < 0.05 else "No"
        })
            
        adf_results =  pd.DataFrame(adf_results)
        adf_results["p-value"] = adf_results["p-value"].map("{:.2%}".format)

        return adf_results
    

    def adf_evol_illiq(monthly_illiq:pl.DataFrame, stockname:list):

        """
        Create a line chart showing the evolution of monthly illiquidity over time.

        Args:
            monthly_illiq: Polars DataFrame of monthly illiquidity measures.

        Returns:
            Plotly figure object.
        """

        monthly_illiq = monthly_illiq.drop_nans()

        monthly_df = monthly_illiq.to_pandas()

        xaxis= monthly_df["Date"]
        yaxis= monthly_df[stockname]


        illiq_evol = go.Figure()
        
        for column in yaxis.columns: 
            illiq_evol.add_trace(go.Scatter(x=xaxis,y=yaxis[column], mode='lines', name = column, line=dict(width=0.8)))
     
        illiq_evol.update_layout(
            title = 'Monthly illiquidity evolution over time',
            xaxis_title ="Date",
            yaxis_title ="Average ILLIQ",
            yaxis=dict(tickformat='.3%'),
            xaxis_tickangle=45,
        )
    
        return illiq_evol
