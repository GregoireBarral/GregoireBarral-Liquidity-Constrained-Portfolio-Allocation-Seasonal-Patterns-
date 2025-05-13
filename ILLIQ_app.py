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


class illiquidity_function: 

    @staticmethod
    
    def Get_returns(values:pl.DataFrame):

        """
        Calculate the returns for each security in the dataset.

        Args:
            values: Polars DataFrame containing price data with a "Date" column.

        Returns:
            Polars DataFrame with calculated returns for each security.
        """

        # Calculate percentage change for each column except "Date"
        returns_securities = values.with_columns([
        (pl.col(c) - pl.col(c).shift(1)) / pl.col(c).shift(1).alias(c)
        for c in values.columns if c != "Date"
    ])
        
        returns_polars  = pl.DataFrame(returns_securities)

        return returns_polars 


#comput the daily illiquidity measure of Amihud
    def Get_daily_ILLIQ(returns_securities: pl.DataFrame, volume: pl.DataFrame, values: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the daily Amihud illiquidity measure for each security.

        Args:
            returns_securities: Polars DataFrame of daily returns.
            volume: Polars DataFrame of daily trading volumes.
            values: Polars DataFrame of daily prices.

        Returns:
            Polars DataFrame with daily illiquidity measures for each security.
        """
        # Filter data to include only common dates
        # Compute absolute returns and daily traded USD
        # Calculate illiquidity as |r| / (traded USD in millions)

        common_dates = set(returns_securities["Date"].to_list()) & \
                    set(volume["Date"].to_list()) & \
                    set(values["Date"].to_list())

        # Convert back to a sorted list by ascending order: 

        common_dates = sorted(common_dates)

        # Filter each DataFrame to include only the common dates

        returns_securities = returns_securities.filter(pl.col("Date").is_in(common_dates))
        volume = volume.filter(pl.col("Date").is_in(common_dates))
        values = values.filter(pl.col("Date").is_in(common_dates))
        

        # Get absolute returns (|r|) for each security
        return_abs = returns_securities.with_columns(
            [pl.col(c).abs().alias(c) for c in returns_securities.columns if c != "Date"]
        )

        #Join price and volume on Date to compute daily traded USD
        joined = values.join(volume, on="Date", how="inner", suffix="_vol")

        #Multiply price * volume to get traded USD per security
        asset_cols = [col for col in values.columns if col != "Date"]

        daily_traded_usd = joined.select(
            ["Date"] + [
                (pl.col(col) * pl.col(f"{col}_vol")).alias(col) for col in asset_cols
            ]
        )

        # Join abs returns with traded USD
        illiq_dataframe = return_abs.join(daily_traded_usd, on="Date", how="inner")

        # Avoid division by zero and nulls
        illiq_dataframe = illiq_dataframe.drop_nulls()

        #Compute Amihud Illiquidity: |r| / (traded USD in millions)

        daily_illiq = pl.DataFrame({"Date": illiq_dataframe["Date"]})

        for col in asset_cols:
            abs_return = illiq_dataframe[col]
            usd_volume = daily_traded_usd[col] / 1_000_000  
            illiq = abs_return / usd_volume

            daily_illiq = daily_illiq.with_columns(illiq.alias(col))

        return daily_illiq #Sort the daily illiq as the output


    #def Liquidity_capm(ILLIQ: pl.DataFrame, rf: pl.DataFrame, Beta1: float, Beta2: float,Beta3: float): 

    def Get_monthly_illiq(illiq_measure: pl.DataFrame): 

        """
        Aggregate daily illiquidity measures to a monthly level.

        Args:
            illiq_measure: Polars DataFrame of daily illiquidity measures.

        Returns:
            Polars DataFrame with monthly illiquidity measures.
        """

    #Order the dataframe on the date object

        daily_illiq = illiq_measure.sort("Date")

        daily_illiq = daily_illiq.drop_nulls()

        daily_illiq = daily_illiq.drop_nans()

    # Create a 'Year-Month' column for easier grouping

        daily_illiq = daily_illiq.with_columns([
            pl.col("Date").dt.year().alias("Year"),
            pl.col("Date").dt.month().alias("Month")
        ])

    # alias allows to rename the content, create a filter on the dataframe :

        columns_to_process = [c for c in daily_illiq.columns if c not in ["Vol.","Date","Year","Month"]]


    #Merge the year and month in the polar dataframe to        
        
        monthly_illiq = daily_illiq.group_by(pl.col("Date").dt.truncate("1mo")).agg(
            [pl.col(col).mean().alias(col) for col in columns_to_process]
        ).sort("Date")



        return pl.DataFrame(monthly_illiq)
    
    def Get_evol_illiq(monthly_illiq:pl.DataFrame): 

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
        yaxis= monthly_df.drop(columns=["Date"])


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
    
    def Average_illiq_versus_returns(return_illiq: pl.DataFrame):

        """
        Plot the relationship between average illiquidity and average returns.

        Args:
            return_illiq: Polars DataFrame with average returns and illiquidity.

        Returns:
            Plotly figure object.
        """

        # Exclude the first column and keep the name of all the other columns
        stock_names = return_illiq.columns[1:] 

        # Drop the first column if it's not needed (assuming it's an index column)*

        return_illiq = return_illiq.drop(return_illiq.columns[0])

        # Convert data to NumPy arrays
        y_axis = np.array(return_illiq.row(0), dtype=np.float64)  # Average returns
        x_axis = np.array(return_illiq.row(1), dtype=np.float64)  # Average illiquidity

        # Create the figure
        risk_return = go.Figure()

        #add the points to the previously created figure

        risk_return.add_trace(go.Scatter(
            x=x_axis,
            y=y_axis,

            mode="markers+text",  
            name="Illiq vs returns",

            marker=dict(color='navy', size=5),

            #Add the name of the stocks to the label: 

            text=stock_names,
            textposition="top center",  
        ))

        
        # Calculate and add trendline
        if len(x_axis) > 1 and len(y_axis) > 1:  

            slope, intercept = np.polyfit(x_axis, y_axis, 1)  #using numpy create a linear regression

            trendline_y = slope * x_axis + intercept

            # Compute R² of the trendline
            y_pred = slope * x_axis + intercept

            # Sum of squared residuals
            ss_res = np.sum((y_axis - y_pred) ** 2)  

            # Total sum of squares
            ss_tot = np.sum((y_axis - np.mean(y_axis)) ** 2) 

            #display the r-squared 
            r_squared = 1 - (ss_res / ss_tot)

            risk_return.add_trace(go.Scatter(
                x=x_axis,
                y=trendline_y,
                mode="lines",
                name="Trendline",
                line=dict(color='darkred', dash='dash')
            ))

            # Add trendline equation and R² as annotation
            equation = f"y = {slope:.4f}x + {intercept:.4f}<br>R² = {r_squared:.4f}"
            
            risk_return.add_annotation(
                x=min(x_axis),
                y=max(y_axis),
                text=equation,
                showarrow=False,
                font=dict(size=10, color="navy"),
                align="right",
                bordercolor="grey",
                borderwidth=0,
                bgcolor="white",
                opacity=0.8
            )

        risk_return.update_layout(
            title="Returns versus Amihud illiquidity measure relationship",
            xaxis_title='Average illiquidity',
            yaxis_title='Average return over the period',
            yaxis=dict(tickformat='.3%'),
            xaxis=dict(tickformat='.3%'),
        )

        return risk_return
        

#debug the function to get the last date of each month 

    def Get_monthly_returns(values:pl.DataFrame): 

        """
        Calculate monthly returns for each security.

        Args:
            values: Polars DataFrame of daily prices.

        Returns:
            Polars DataFrame of monthly returns.
        """


    #Implement two additionnal columns to compute the year and the month for each period: 

        stock_prices = values.to_pandas()
        # Resample the data to monthly frequency and calculate the returns

        stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
        stock_prices.set_index('Date', inplace=True)


        monthly_prices = stock_prices.resample('M').ffill()

        monthly_returns = monthly_prices.pct_change().dropna()

        # Reset the index to have the date as a column
        monthly_returns.reset_index(inplace=True)

        return  pl.DataFrame(monthly_returns)

    
    def illiq_period(filters, monthly_illiq:pl.DataFrame, stockname:list): 

        """
        Filter illiquidity data for a specific period and selected stocks.

        Args:
            filters: List of dates to filter.
            monthly_illiq: Polars DataFrame of monthly illiquidity measures.
            stockname: List of stock names to include.

        Returns:
            Polars DataFrame with filtered data.
        """

        
        # Filter based on selected months
        illiq_over_period = monthly_illiq.filter(pl.col("Date").is_in(filters))

        # Select only relevant columns
        selected_columns = ["Date"] + stockname
        illiq_over_period = illiq_over_period.select(selected_columns)

        return illiq_over_period
    


    def illiq_visual(illiq_change: pl.DataFrame):
    
        """
        Create a line chart showing illiquidity changes over time.

        Args:
            illiq_change: Polars DataFrame of illiquidity changes.

        Returns:
            Plotly figure object.
        """

        # Convert Polars DataFrame to Pandas
        illiq_visu = illiq_change.to_pandas()

        # Extract X and Y values
        xaxis = illiq_visu["Date"]
        yaxis = illiq_visu.drop(columns=["Date"])

        yaxis_columns = yaxis.columns.to_list()

        illiq_visu[yaxis_columns] =(illiq_visu[yaxis_columns].replace('%', '', regex=True).astype(float))/ 100

        yaxis = illiq_visu.drop(columns=["Date"])

        # Initialize figure
        illiq_month = go.Figure()

        # Add traces for each column
        for column in yaxis.columns:
            illiq_month.add_trace(go.Scatter(x=xaxis, y=yaxis[column], mode='markers+lines', name=column))

        # Update layout
        illiq_month.update_layout(
            title='Monthly Illiquidity Evolution Over Time',
            xaxis_title="Date",
            yaxis_title="Average ILLIQ",
            xaxis_tickangle=45,
             #format the yaxis to a percentage format 
            yaxis=dict(tickformat='.3%'),
             
        )

        return illiq_month

    def Get_correlation(returns:pl.DataFrame, select_correl):

        """
        Compute and visualize the correlation matrix for selected stocks.

        Args:
            monthly_returns: Polars DataFrame of monthly returns.
            select_correl: List of stock names to include in the correlation.

        Returns:
            Plotly heatmap figure object.
        """
        #Using the returns we want to get the correlation matrix for the different stocks that we need to analyse: 
        #since correlation vary over time we will select only the correlation using the last 3 years 

        if len(select_correl) >= 2:

            # Convert to pandas and select the desired columns
            returns_period = returns.to_pandas()

            # Filter to get the returns after the specified date only

            returns_period = returns_period[returns_period["Date"] > "2018-12-31"]

            returns_period = returns_period[select_correl]

            # Compute correlation matrix
            correl = returns_period.corr()

            # Format correlation values as percentage strings for display
            text_matrix = correl.applymap(lambda x: f"{x:.1%}").values

            # Create a mask for the upper triangle (k=1 means exclude diagonal)
            mask_upper = np.triu(np.ones_like(correl, dtype=bool), k=1)  # Upper triangle mask

            # Mask the correlation values and text for the upper triangle
            correl_upper_masked = np.where(mask_upper, np.nan, correl.values)
            text_matrix_upper_masked = np.where(mask_upper, "", text_matrix)

            # Create the heatmap using Plotly's graph_objects (go.Heatmap)
            heatmap = go.Figure(
                data=go.Heatmap(
                    z=correl_upper_masked,  # Apply the masked correlation values
                    x=correl.columns,
                    y=correl.columns,
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

        else:
            return None



    def get_market_return(values:pl.DataFrame): 

        """
        Obtain the returns of the reference market to use them in the LCAPM module and compare our performance to the market one

        args: 

        market_data = index prices evolution

        output: 

        returns of the market without dropping the null values on a monthly basis
        """

        #to get the market return we need to use the dividend reinvested index
        #compute the illiquidity cost DT

        market_data = values.to_pandas()

        market_data = market_data.set_index("Date")

        weight = 1/len(market_data.columns.to_list())   

        market_data["^GSPC"] = (market_data*weight).sum(axis=1)

        market_data = market_data["^GSPC"]

        market_data = market_data.reset_index()

        market_data = pl.DataFrame(market_data)

        market_data = market_data.drop_nans()
        market_data = market_data.drop_nulls()

        market_return = market_data.with_columns(pl.col('^GSPC').pct_change().alias('market_returns'))

        market_data = market_return.to_pandas()


        market_data['Date'] = pd.to_datetime(market_data['Date'])
        market_data.set_index('Date', inplace=True)


        monthly_data = market_data.resample('M').ffill()

        monthly_data = monthly_data.pct_change().dropna()

        # Reset the index to have the date as a column
        monthly_data.reset_index(inplace=True)

        return pl.DataFrame(market_return["Date", "market_returns"])
    


#build a portfolio of stocks: 

    def portfolio_returns(stock_list,monthly_returns, weights = None): 

        """
        Calculate portfolio returns based on selected stocks and weights.

        Args:
            stock_list: List of stock names in the portfolio.
            monthly_returns: Polars DataFrame of monthly returns.
            weights: List of weights for each stock (default: equal weights).

        Returns:
            Polars DataFrame of portfolio returns.
        """

        #count the number of stock selected
        num_stocks = len(stock_list)
        
        # Default to equal weights if not provided
        if weights is None:
            weights = 1 / num_stocks
        

        # Select relevant columns (Month, Year, and stocks)
        date_cols = monthly_returns.select("Date")
        stock_returns = monthly_returns.select(stock_list)

        # Multiply stock returns by weights (row-wise multiplication)
        weighted_returns = stock_returns * weights

        # Compute portfolio returns by summing across columns (correct Polars method)
        portfolio_returns = weighted_returns.with_columns(
            portfolio_return = pl.sum_horizontal(*stock_returns.columns)).select("portfolio_return")

   

        # Convert to Series before adding to DataFrame
        result = date_cols.with_columns(pl.Series("Portfolio_Returns", portfolio_returns))

        result= result.filter(pl.col("Date") > pl.lit("2018-12-31").cast(pl.Date))
        # Convert 'Month and Year' to datetime format

        return result
    
    def get_histo(returns): 
            
        """
            Create a histogram of portfolio returns.

            Args:
                returns: Polars DataFrame of portfolio returns.

            Returns:
                Plotly histogram figure object.
            """

        returns_df = pd.DataFrame(returns["Portfolio_Returns"])

        # Determine number of bins dynamically based on data size
        nbins = min(15, len(returns_df))

        # Create the histogram with count of occurrences (no normalization)
        fig = px.histogram(
            returns_df,
            nbins=nbins,
            labels={'value': 'Returns'}
        )

        fig.update_layout(
            xaxis_title="Returns",
            yaxis_title="Count of Occurrences",
            template="plotly_white", 
            showlegend=False
        )

        fig.update_xaxes(tickformat=".2%")

        return fig


    #Créer une fonction donnant un visuel sur les stocks offrant le meilleur rendement moyen tout en minimisant illiq top 20 

    #Obtenir les covariances 


    
    def get_sharpe_ratio_illiq(risk_free,return_illiq,values): 

        """
        The objective of the function is to compute a modified version of the Sharpe Ratio to plot the stock maximizing the returns 
        while minimizing the illiquidity  -


        args: 

        risk_free = average risk free rate over the period
        illiq_return = matrix of illiq versus returns previously computed containing both average return / illiq
    
        """

        risk_free = risk_free.to_pandas()

        rf = risk_free["risk_free_rate"].values

        average_rf = rf.mean() * np.sqrt(12)


        #promote the first row to headers 

        values = values.to_pandas()

        #Extract the first row of the values
        first_row = pd.DataFrame(values.iloc[0]).T

        # Extract the last row of the values
        last_row = pd.DataFrame(values.iloc[-1]).T

        # Create a dataframe containing the first and the last value of our values dataframe
        df_ptf_return = pd.concat([first_row, last_row])
        
        # Add the average illiquidity measure of Amihud to the existing dataframe
        illiq_values = return_illiq.to_pandas()

        # Location of the measure
        illiq_values = pd.DataFrame(illiq_values.iloc[-1]).T

        # use indexing to avoid concatenation errors
        illiq_values = illiq_values.set_index("name")

        df_ptf_return = df_ptf_return.set_index("Date")

        # Concat the dataframe on the tickers of the stocks
        df_ptf_return = pd.concat([df_ptf_return, illiq_values])
        
        # transpose the dataframe
        df_ptf_return = df_ptf_return.T

        # Procede to the return calculation
        date_1 = df_ptf_return.columns[0]
        date_2 = df_ptf_return.columns[1]

        # Calculate the return of each stock over the period 

        df_ptf_return["return"] = ((df_ptf_return[date_2]/df_ptf_return[date_1])**(1/14.9))-1

        # Filter the dataframe on the input we are interested in
        df_ptf_return = df_ptf_return[["return","Average illiquidity"]]

        # Add the column of the risk free rate:

        df_ptf_return["risk_free_rate"] = average_rf

        df_ptf_return["Sharpe illiq ratio"] = (df_ptf_return["return"]-df_ptf_return["risk_free_rate"])/df_ptf_return["Average illiquidity"]


        df_ptf_return = df_ptf_return.reset_index()

        return pl.DataFrame(df_ptf_return)
    
    def sharpe_classique(values:pl.DataFrame, risk_free): 




        returns = values.select(pl.col(c).pct_change() for c in values.columns)

        values = values.to_pandas()

        #Extract the first row of the values
        first_row = pd.DataFrame(values.iloc[0]).T

        # Extract the last row of the values
        last_row = pd.DataFrame(values.iloc[-1]).T

        df_ptf_return = pd.concat([first_row, last_row])

        sharpe = df_ptf_return.set_index("Date")

        # transpose the dataframe
        sharpe = sharpe.T

        # Procede to the return calculation
        date_1 = sharpe.columns[0]
        date_2 = sharpe.columns[1]

        # Calculate the return of each stock over the period 

        sharpe["return"] = ((sharpe[date_2]/sharpe[date_1])**(1/14.9))-1

        sharpe = sharpe["return"]

        sharpe = sharpe.reset_index(    )
        
        risk_free_rate =  risk_free.select(pl.col("risk_free_rate").mean()).to_numpy()[0][0] 

        standard_deviation = returns.std()

        standard_deviation= standard_deviation.to_pandas()

        standard_deviation = standard_deviation.transpose().iloc[0::1]


        standard_deviation = standard_deviation.reset_index()

        standard_deviation.columns = ["index","stdev"]

        final = pd.merge(sharpe,standard_deviation,on= "index")    

        #add the risk_free rate

        final["risk_free"] = risk_free_rate

        final["Sharpe"] = (final["return"]- final["risk_free"])/final["stdev"]

        #sort the values based on the highest sharpe ratio: 

        final = final.sort_values(by="Sharpe", axis = 0, ascending= False)

        return final

    #Obtenir les ratios de sharpe pour les différentes assets 

    #remplacer la vol par la mesure ILLIQ 

    #importer rf dans le dataset 

    def portfolio_returns_bis(stock_list_bis,monthly_returns, weights = None ): 

        """
        In this function we will use portfolios composed of 20 stocks using the countries that provide the best 
        return for the less illiquidity possible 

        args: 

        weights = weights of portfolios
        list_stock = stock selection
        monthly_retursn = stock returns
        """

        #count the number of stock selected
        num_stocks = len(stock_list_bis)
        
        # Default to equal weights if not provided
        if weights is None:
            weights = 1 / num_stocks
        

        # Select relevant columns (Month, Year, and stocks)
        date_cols = monthly_returns.select("Date")
        stock_returns = monthly_returns.select(stock_list_bis)

        # Multiply stock returns by weights (row-wise multiplication)
        weighted_returns = stock_returns * weights

        # Compute portfolio returns by summing across columns (correct Polars method)
        portfolio_returns = weighted_returns.with_columns(
            portfolio_return = pl.sum_horizontal(*stock_returns.columns)).select("portfolio_return")

   

        # Convert to Series before adding to DataFrame
        result_bis = date_cols.with_columns(pl.Series("Portfolio_Returns", portfolio_returns))
        # Convert 'Month and Year' to datetime format

        return result_bis
    

    def get_VL_BASE100_bis(results_bis:pl.DataFrame): 
        
        """
        Calculate and visualize the base 100 evolution of portfolio returns.

        Args:
            results_bis: Polars DataFrame of portfolio returns.

        Returns:
            Plotly figure object showing base 100 evolution.
        """

        results = results_bis

        results = results.with_columns(
            (100 * np.exp(results["Portfolio_Returns"].cum_sum())).alias("VL_base100")
        )

        #return the evolution of the VL in the report: 

        vl_fig = go.Figure()

        vl_fig.add_trace(go.Scatter( x= results["Date"].to_list(),
                                    y= results["VL_base100"].to_list(),
                                    mode = 'lines',
                                    name = 'VL base 100 du portefeuille' ,
                                    )
        )

        vl_fig.update_layout(
            title ='VL Base 100', 
            xaxis_title = 'Date',
            yaxis_title = "VL",
        )

        return vl_fig
    
        

    def VL_100(values,stock_list,market_returns ):

        """
        Compare portfolio and market performance using base 100 values.

        Args:
            values: Polars DataFrame of stock prices.
            stock_list: List of stock names in the portfolio.
            market_returns: Polars DataFrame of market returns.

        Returns:
            Pandas DataFrame with base 100 values for portfolio and market.
        """ 

        # get the import to the pandas format

        values = values.to_pandas()

        benchmark = values

        benchmark = benchmark.set_index("Date")

        market_returns = market_returns.to_pandas()

        selected_columns = ["Date"] + stock_list

        values = values[selected_columns]
        # create a pounderation for the value of each stock before doing the cumulative sum and get afterward the theoritical values of the ptf
        count= len(stock_list)

        ratio = 1/count

        ratio2 = 1/len(values.columns.to_list())

        benchmark["^GSPC"] =(benchmark*ratio2).sum(axis=1)

        benchmark = benchmark.reset_index()

        values[stock_list] = values[stock_list]*ratio #Apply the weights to the stocks in the portfolio

        

        

        values["ptf_value"] =  values[stock_list].sum(axis=1)

        df = pd.merge(values,benchmark[["Date","^GSPC"]]  ,on ="Date", how ='inner')[-1500:]

        df['index_100'] =(df['^GSPC']/df['^GSPC'].iloc[0])*100
        df['ptf_100'] =(df['ptf_value']/df["ptf_value"].iloc[0])*100 #plot the portfolio valueas and normalize it to 100 basis
        

        return df[["Date","index_100","ptf_100"]]


    def illiquidity_correlation(illiq_measure: pl.DataFrame, filter : list): 
        
        """
        Compute and visualize the correlation of illiquidity among stocks.

        Args:
            illiq_measure: Polars DataFrame of illiquidity measures.
            filter: List of stock names to include in the correlation.

        Returns:
            Plotly heatmap figure object.
        """

        illiq_data = illiq_measure.to_pandas()

        illiq_data = illiq_data.set_index("Date")

        illiq_data = illiq_data[illiq_data.index > "2018-12-31"]

        illiq_data = illiq_data[filter]

    

        correlation_illiquidity  = illiq_data.corr()

                    # Format correlation values as percentage strings for display
        text_matrix = correlation_illiquidity.applymap(lambda x: f"{x:.1%}").values

            # Create a mask for the upper triangle (k=1 means exclude diagonal)
        mask_upper = np.triu(np.ones_like(correlation_illiquidity, dtype=bool), k=1)  # Upper triangle mask

            # Mask the correlation values and text for the upper triangle
        correl_upper_masked = np.where(mask_upper, np.nan, correlation_illiquidity.values)
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
                title="Illiquidity Correlation Matrix (Blue = Positive, Red = Negative)",
                xaxis=dict(tickangle=45),
                yaxis=dict(autorange="reversed"),
                template="plotly_white"
            )

        return heatmap
    

    
   
