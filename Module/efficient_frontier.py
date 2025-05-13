import numpy as np
import plotly.graph_objects as go 
import polars as pl
import random
import pandas as pd


class efficient_frontier: 

    def efficient_frontier(values: pl.DataFrame, num_portfolios, seed: int = 25): 
         
        #  Set a seed to get the same results every time: 
         rng = np.random.RandomState(seed)
        #  Convert the different values to pandas 
         values_df = values

        #  Set the index of the dataframe on the Date col
         values_df = values_df.set_index("Date")

         values_df = values_df[values_df.index > "2018-12-31"]

        # Calculate the returns 
         returns = values_df.pct_change().dropna()

        # Get a list of the different assets
         asset_list = returns.columns.tolist()
        
         num_asset = len(asset_list)

         risk_free_rate = 0.0114* np.sqrt(252)

         if num_asset < 20:
            raise ValueError("Dataset must contain at least 20 assets.")

         results = {"Returns": [], 
                    "Risk":[],
                    "Sharpe ratio":[], 
                    "Asset Sample":[], 
                    "weight":[]}
         
         for i in range(num_portfolios): 
              
              selected_assets = rng.choice(asset_list,size = 20, replace = False)

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
    

    def plot_efficient_frontier(df):
        """
        Plots the Efficient Frontier with a strictly increasing logarithmic fit.
        
        Args:
            df (pd.DataFrame): DataFrame containing portfolio 'Risk' and 'Returns'.
            
        Returns:
            plotly.graph_objects.Figure
        """

        results_df = pd.DataFrame(df)
    
        # Create a Plotly figure
        results_df = results_df.sort_values('Risk')

        # Remove dominated portfolios: only keep the highest return for each risk level 
        efficient_frontier = []
        last_return = -np.inf

        for _, row in results_df.iterrows():
            if row['Returns'] > last_return:  # Ensure we have only increasing returns
                efficient_frontier.append(row)
                last_return = row['Returns']

        efficient_frontier_df = pd.DataFrame(efficient_frontier)

        # Create a Plotly figure
        fig = go.Figure()

        # Add scatter plot of portfolios
        fig.add_trace(go.Scatter(
            x=results_df['Risk'], 
            y=results_df['Returns'], 
            mode='markers', 

            name='Portfolios'
        ))

        # Highlight the maximum Sharpe ratio portfolio
        max_sharpe_idx = results_df['Sharpe ratio'].idxmax()
        max_sharpe_return = results_df.loc[max_sharpe_idx, 'Returns']
        max_sharpe_risk = results_df.loc[max_sharpe_idx, 'Risk']
        
        fig.add_trace(go.Scatter(
            x=[max_sharpe_risk], 
            y=[max_sharpe_return], 
            mode='markers', 
            marker=dict(symbol='star', color='green', size=15),
            name='Max Sharpe Ratio'
        ))

        # Efficient frontier plot (raw data for the curve)
        fig.add_trace(go.Scatter(
            x=efficient_frontier_df['Risk'], 
            y=efficient_frontier_df['Returns'], 
            mode='lines', 
            name='Efficient Frontier',
            line=dict(color='red', width=0.5)
        ))

        # Update layout
        fig.update_layout(
            title='Efficient Frontier for Equally Weighted Portfolios',
            xaxis_title='Risk (Standard Deviation)',
            yaxis_title='Expected Return',
            showlegend=True,
            template='seaborn',
            yaxis=dict(tickformat='.3%'),
            xaxis=dict(tickformat='.3%'),
        )

        return fig



 
    def illiquidity_efficient_frontier(values: pl.DataFrame,illiq_measure:pl.DataFrame,num_portfolios:int, seed : int = 26): 

        rng1 = random.Random(seed)

        values_df = values

        illiquidity = illiq_measure
    
        

        values_df = values_df.set_index("Date")

        illiquidity = illiquidity.set_index("Date")

        values_df = values_df[values_df.index > "2018-12-31"]

        illiquidity = illiquidity[illiquidity.index > "2018-12-31"]

        returns = values_df.pct_change().dropna()

        asset_list = returns.columns.to_list()

        num_asset = len(asset_list)

        risk_free_rate = 0.0114 * np.sqrt(252)

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

       
        sorted_df = portfolios_df.sort_values(by="Illiq")
        efficient_risks = []
        efficient_returns = []
        last_return = -float("inf")

        for _, row in sorted_df.iterrows():
            if row["Returns"] > last_return:
                efficient_risks.append(row["Illiq"])
                efficient_returns.append(row["Returns"])
                last_return = row["Returns"]

       
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

        
        max_sharpe_idx = portfolios_df['Sharpe illiq ratio'].idxmax()
        max_sharpe_return = portfolios_df.loc[max_sharpe_idx, 'Returns']
        max_sharpe_Illiq = portfolios_df.loc[max_sharpe_idx, 'Illiq']

   

        
        fig.add_trace(go.Scatter(
            x=efficient_risks,
            y=efficient_returns,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='red', width=2)
        ))

       
        frontier_set = set(zip(efficient_risks, efficient_returns))
        portfolios_above_frontier = portfolios_df[
            portfolios_df.apply(lambda row: (row["Illiq"], row["Returns"]) in frontier_set, axis=1)
        ]

        portfolios_above_frontier_sorted = portfolios_above_frontier.sort_values(by="Sharpe illiq ratio", ascending=False)

       
        fig.update_layout(
            title="Efficient Frontier using Illiq (Amihud Measure) — Equiponderated Portfolios",
            xaxis_title="Illiq (Annualized)",
            yaxis_title="Return (Annualized)",
            template="plotly_dark",
            height=600,
            yaxis=dict(tickformat='.3%'),
            xaxis=dict(tickformat='.3%'),
        )
        

        return fig, portfolios_above_frontier_sorted
