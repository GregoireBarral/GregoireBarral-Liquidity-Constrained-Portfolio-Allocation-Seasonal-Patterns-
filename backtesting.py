
import numpy as np
import datetime as dt 
import plotly.graph_objects as go 
import polars as pl
import streamlit as st
import random
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



class scenario: 
    
    def apply_shock(values:pl.DataFrame,selection, start_date:str,shock:float): 

        """
        applies a shock percentage previously choose in the scenario analysis tool of streamlit, 
        the scenario deducted from this shock will be visually represented and then analysed

        values = excel values 
        selection = streamlit filter
        shock = input in the slider of streamlit 
        
        """

        shocked_columns=[]


        for col in selection:
            shocked_col = pl.when(pl.col("Date") >= start_date) \
                            .then(pl.col(col) * (1 + shock / 100)) \
                            .otherwise(pl.col(col)) \
                            .alias(f"{col} shock")
            shocked_columns.append(shocked_col)
  
        return values.with_columns(shocked_columns).select(
        ["Date"] + [col for pair in selection for col in (pair, f"{pair} shock")]
    )
    
#En construction: 
  

    
    def efficient_frontier(values: pl.DataFrame, num_portfolios, seed: int = 42): 
         
        #  Set a seed to get the same results every time: 
         rng = random.Random(seed)
        #  Convert the different values to pandas 
         values_df = values.to_pandas()

        #  Set the index of the dataframe on the Date col
         values_df = values_df.set_index("Date")

        # Calculate the returns 
         returns = values_df.pct_change().dropna()

        # Get a list of the different assets
         asset_list = returns.columns.tolist()
        
         num_asset = len(asset_list)

         risk_free_rate = 0.05

         if num_asset < 20:
            raise ValueError("Dataset must contain at least 20 assets.")

         results = {"Returns": [], 
                    "Risk":[],
                    "Sharpe ratio":[], 
                    "Asset Sample":[], 
                    "weight":[]}
         
         for i in range(num_portfolios): 
              
              selected_assets = rng.sample(asset_list,20)

              sub_returns=  returns[selected_assets]

              mean_return = sub_returns.mean()

              cov_matrix = sub_returns.cov()

              weights = np.ones(20)/20


              mean_daily_return = np.dot(weights, mean_return)
              port_return = mean_daily_return*252 
              port_stdev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
              sharpe_ratio = (port_return - risk_free_rate) / port_stdev


              results["Returns"].append(port_return)
              results["Risk"].append(port_stdev)
              results["Sharpe ratio"].append(sharpe_ratio)
              results["Asset Sample"].append(selected_assets)
              results["weight"].append(weights)

       
         return pd.DataFrame(results)
    
    def plot_efficient_frontier(portfolios_df):

        fig = go.Figure()

        # Scatter plot of all portfolios
        fig.add_trace(go.Scatter(
            x=portfolios_df["Risk"],
            y=portfolios_df["Returns"],
            mode='markers',
            marker=dict(
                size=6,
                color=portfolios_df["Sharpe ratio"],
                colorscale='Viridis',
                colorbar=dict(title="Sharpe ratio"),
                showscale=False
            ),
            text=[
                f"Sharpe: {sharpe:.2f}<br>Assets: {assets}" 
                for sharpe, assets in zip(portfolios_df["Sharpe ratio"], portfolios_df["Asset Sample"])
            ],
            hoverinfo="text",
            name="Portfolios"
        ))

        # === FIT SMOOTH EFFICIENT FRONTIER ===
        # Sort by risk
        sorted_df = portfolios_df.sort_values(by="Risk")

        # Keep only non-dominated portfolios
        efficient_risks = []
        efficient_returns = []
        max_return = -np.inf

        for _, row in sorted_df.iterrows():
            if row["Returns"] > max_return:
                efficient_risks.append(row["Risk"])
                efficient_returns.append(row["Returns"])
                max_return = row["Returns"]

        # Fit polynomial (degree 3) regression to smooth the frontier
        X = np.array(efficient_risks).reshape(-1, 1)
        y = np.array(efficient_returns)
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        # Generate smooth curve
        x_curve = np.linspace(min(efficient_risks), max(efficient_risks), 200)
        x_curve_poly = poly.transform(x_curve.reshape(-1, 1))
        y_curve = model.predict(x_curve_poly)

        # Plot the smooth efficient frontier
        fig.add_trace(go.Scatter(
            x=x_curve,
            y=y_curve,
            mode='lines',
            line=dict(color='red', width=3, dash='solid'),
            name="Efficient frontier"
        ))

        # Layout
        fig.update_layout(
            title="Frontière Efficiente Lissée (MPT — Equiponderated portfolios)",
            xaxis_title="Risk (Annual volatility)",
            yaxis_title="Annualized return",
            template="plotly_dark",
            height=600
        )

        return fig
    
    def illiquidity_efficient_frontier(values: pl.DataFrame,illiq_measure:pl.DataFrame,num_portfolios:int, seed : int = 43): 

        rng1 = random.Random(seed)

        values_df = values.to_pandas()

        illiquidity = illiq_measure.to_pandas()

        values_df = values_df.set_index("Date")

        returns = values_df.pct_change().dropna()

        asset_list = returns.columns.to_list()

        num_asset = len(asset_list)

        risk_free_rate = 0.05

        if num_asset < 20:
            raise ValueError("Dataset must contain at least 20 assets.")

        results = {"Returns": [], 
                    "Illiq":[],
                    "Sharpe illiq ratio":[], 
                    "Asset Sample":[], 
                    "weight":[]}
         
        for i in range(num_portfolios): 
              
              selected_assets = rng1.sample(asset_list,20)

              sub_returns=  returns[selected_assets]

              sub_illiq = illiquidity[selected_assets]

              illiq_mean = sub_illiq.mean()

              mean_return = sub_returns.mean()

              cov_matrix = sub_returns.cov()

              weights = np.ones(20)/20

              mean_daily_return = np.dot(weights, mean_return)
              mean_illiq = np.dot(weights,illiq_mean)

              port_return = mean_daily_return*252 
              port_illiq = mean_illiq * 252
              port_stdev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
              sharpe_ratio = (port_return - risk_free_rate) / port_illiq


              results["Returns"].append(port_return)
              results["Illiq"].append(port_stdev)
              results["Sharpe illiq ratio"].append(sharpe_ratio)
              results["Asset Sample"].append(selected_assets)
              results["weight"].append(weights)


        return pd.DataFrame(results)
    
    def illiq_plot_efficient_frontier(portfolios_df):

            fig = go.Figure()

            # Scatter plot of all portfolios
            fig.add_trace(go.Scatter(
                x=portfolios_df["Illiq"],
                y=portfolios_df["Returns"],
                mode='markers',
                marker=dict(
                    size=6,
                    color=portfolios_df["Sharpe illiq ratio"],
                    colorscale='Viridis',
                    colorbar=dict(title="Sharpe illiq ratio"),
                    showscale=False
                ),
                text=[
                    f"Sharpe illiq: {sharpe:.2f}<br>Assets: {assets}" 
                    for sharpe, assets in zip(portfolios_df["Sharpe illiq ratio"], portfolios_df["Asset Sample"])
                ],
                hoverinfo="text",
                name="Portfolios"
            ))

            # === FIT SMOOTH EFFICIENT FRONTIER ===
            # Sort by risk
            sorted_df = portfolios_df.sort_values(by="Illiq")

            # Keep only non-dominated portfolios
            efficient_risks = []
            efficient_returns = []
            max_return = -np.inf

            for _, row in sorted_df.iterrows():
                if row["Returns"] > max_return:
                    efficient_risks.append(row["Illiq"])
                    efficient_returns.append(row["Returns"])
                    max_return = row["Returns"]

            # Fit polynomial (degree 3) regression to smooth the frontier
            X = np.array(efficient_risks).reshape(-1, 1)
            y = np.array(efficient_returns)
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X)

            model = LinearRegression()
            model.fit(X_poly, y)

            # Generate smooth curve
            x_curve = np.linspace(min(efficient_risks), max(efficient_risks), 200)
            x_curve_poly = poly.transform(x_curve.reshape(-1, 1))
            y_curve = model.predict(x_curve_poly)

            # Plot the smooth efficient frontier
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=y_curve,
                mode='lines',
                line=dict(color='red', width=3, dash='solid'),
                name="Efficient frontier"
            ))

            # === FILTER PORTFOLIOS ABOVE OR EQUAL TO THE EFFICIENT FRONTIER ===

            # Interpolate predicted returns from the efficient frontier curve for each portfolio
            portfolios_df["Predicted Return"] = np.interp(portfolios_df["Illiq"], x_curve, y_curve)

            # Filter portfolios where actual return >= predicted return
            portfolios_above_frontier = portfolios_df[portfolios_df["Returns"] >= portfolios_df["Predicted Return"]]

            # Sort by Returns (Descending)
            portfolios_above_frontier_sorted = portfolios_above_frontier.sort_values(by="Sharpe illiq ratio", ascending=False)


            # Layout
            fig.update_layout(
                title="Efficient frontier using illiq amihud measure as risk (MPT — Equiponderated portfolios)",
                xaxis_title="illiq (Annualized illiq)",
                yaxis_title="Annualized return",
                template="plotly_dark",
                height=600
            )

            return fig, portfolios_above_frontier_sorted
    
    def illiquidity_correlation(illiq_measure: pl.DataFrame): 
        
        """

        the objective is to study the correlation of illiquidity among the different stocks over the years considered 
        
        """

        illiq_data = illiq_measure.to_pandas()

        average_illiq = illiq_data.mean()

        correlation_illiquidity  = illiq_measure.corr()

                    # Format correlation values as percentage strings for display
        text_matrix = correlation_illiquidity.applymap(lambda x: f"{x:.1%}").values

            # Create a mask for the upper triangle (k=1 means exclude diagonal)
        mask_upper = np.triu(np.ones_like(correlation_illiquidity, dtype=bool), k=1)  # Upper triangle mask

            # Mask the correlation values and text for the upper triangle
        correl_upper_masked = np.where(mask_upper, np.nan, correl.values)
        text_matrix_upper_masked = np.where(mask_upper, "", text_matrix)

            # Create the heatmap using Plotly's graph_objects (go.Heatmap)
        heatmap = go.Figure(
        data=go.Heatmap(
        z=correl_upper_masked,  # Apply the masked correlation values
        x=correlation_illiquidity.columns,
        y=correlation_illiquidity.columns,
        
        text=text_matrix_upper_masked,  # Text for the lower triangle only
        texttemplate="%{text}",
        colorscale="RdBu",  # Color scale: Blue = positive, Red = negative
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
        showscale=True  # Show color scale
                )
            )

            # Update layout for better visuals
        heatmap.update_layout(
                title="Correlation Matrix (Blue = Positive, Red = Negative)",
                xaxis=dict(tickangle=45),
                yaxis=dict(autorange="reversed"),
                template="plotly_white"
            )

        return heatmap
