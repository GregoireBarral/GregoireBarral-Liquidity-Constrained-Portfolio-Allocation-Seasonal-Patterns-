
import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
import time

import os
import datetime as dt 
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.subplots as sp

#polars is MIT library that offers a new generation of dataframe faster than pandas and more complete 
import polars as pl

#Import our own library built for the need of the paper

from ILLIQ_app import illiquidity_function as illiq
from backtesting import scenario as sce
from LCAPM import lcapm_class as lcapm
from VaR import VaR_module as var
from efficient_frontier import efficient_frontier as ef 
from linear_regression import LinarRagression as linreg



st.set_page_config(
    page_title = "Illiquidity mémoire",
    layout = "wide", 
    page_icon = ""
)
#Search for the different inputs of the table : 

path_directory = r"C:\Users\grego\Documents\python"

#file that contain the information about the asset prices:
path_file = os.path.join(path_directory,"test.xlsx")



#file that contain the information about the USD Volume: using the indice on the USD Future for the moment 

@st.cache_data

def get_riskfree(link = r"C:\Users\grego\OneDrive\Bureau\Mémoire article de recherche\data quanti\risk_free_rate.xlsx"):

    """
     read and store the data in the streamlit app   
    """
    read = pl.read_excel(link, sheet_name="Daily")

    # treat the data and handle the nan / null values from the table

    read = read.with_columns(pl.col("risk_free_rate").drop_nans())
    read = read.with_columns(pl.col("risk_free_rate").drop_nulls())

    return read

@st.cache_data
def get_values(path_file):

    return  pl.read_excel(path_file)

@st.cache_data
def get_volume(path_file):

    return  pl.read_excel(path_file, sheet_name = 'volume')



def get_market_values(path_file):

    values = pl.read_excel(path_file)

    market_data = values.to_pandas()

    market_data = market_data.set_index("Date")

    weight = 1/len(market_data.columns.to_list())   

    market_data["^GSPC"] = (market_data*weight).sum(axis=1)

    market_data = market_data["^GSPC"]

    market_data = market_data.reset_index()

    

    return pl.DataFrame(market_data)

@st.cache_data
def compute_illiq_ef_frontier(values, illiq_measure, num_portfolios):
    return ef.illiquidity_efficient_frontier(values, illiq_measure, num_portfolios)

@st.cache_data
def compute_efficient_frontier(values,num_porfolios):
    return ef.efficient_frontier(values, num_portfolios)



path_file_USD = os.path.join(path_directory,"Futures indice du dollar US - Données Historiques.csv")

risk_free = get_riskfree()

#remove the columns that do not have any information  null values: 

risk_free = risk_free.drop_nulls()


# Read the Excel file using Pandas
values = get_values(path_file)

volume = get_volume(path_file)

market_returns1 = get_market_values(path_file)

daily_traded_usd = pl.read_csv(path_file_USD)["Date","Vol."]

# modify the order of the dataframe about the USD

daily_traded_usd = pl.DataFrame(daily_traded_usd)

#remove NAN values of the dataset:
values = values.drop_nans()
volume = volume.drop_nans()
market_returns = market_returns1.drop_nans()


#use the function Get_returns from the class illiquidity_function to compute the returns
returns = illiq.Get_returns(values)

#Delete the columns where we do not have any information about the daily return
returns = returns.drop_nans()


market_returns = illiq.get_market_return(market_returns)

  
#Now we will focus on getting the ILLIQ measure of Amihud using the daily trades USD volume and the absolute returns illiq module:
#Get the daily illiq measure

illiq_measure = illiq.Get_daily_ILLIQ(returns,volume,values)

#plot the average illiq over the selected period:

average_illiq = pl.DataFrame(illiq_measure)

#Handle errors of infinite values when calculating the daily illiquidity: 

average_illiq = average_illiq.with_columns(
    [pl.col(col).replace(float('inf'), None) for col in average_illiq.columns if col != "Date"])

average_illiq = average_illiq.drop_nans()

average_illiq = average_illiq.mean()

#get the average return over the selected period
average_return = returns.mean()

#Create a matrix of illiq versus return in order to compute the regression illiq versus returns: (returns in first row / illiq in second row) 

return_illiq =pl.concat([average_return.select(pl.exclude("Date")),average_illiq.select(pl.exclude(["Vol.","Date"]))])

# Create a list with the exact "Name" values for each row

names =pl.Series("name",["Average return", "Average illiquidity"])

return_illiq = return_illiq.with_columns(names)

#set as first row the column name : 

col_to_move = "name"

new_order = [col_to_move] + [col for col in return_illiq.columns if col != col_to_move]

return_illiq = return_illiq.select(new_order)     

#observe the evolution of illiquidity over time and plot the curve for the different data points

monthly_illiq = illiq.Get_monthly_illiq(illiq_measure)

#Call the function from the illiq class to visualize the evolution of illiquidity over time: 

list_month_year = monthly_illiq.select(pl.col("Date")).to_series().to_list()

illiq_evol = illiq.Get_evol_illiq(monthly_illiq)

#illiq_evol.show() #Attention sur certaines périodes pas de donnnées donc à considérer sur une analyse

monthly_returns = illiq.Get_monthly_returns(values)


#Build the selection of the portfolio: 

sharpe = illiq.get_sharpe_ratio_illiq(risk_free,return_illiq,values)

sharpe = sharpe.to_pandas()


sharpe = sharpe.sort_values("Sharpe illiq ratio", ascending = False)

#Sort the top 20 sharpe ratio of the portfolio


top_sharpe = sharpe.iloc[0:20:1]

# Sort the worst sharpe illiquidity ratio
worst_sharpe = sharpe.iloc[-20:]


#Select the top 20 illiq sharpe ratio by default
select_header = top_sharpe["index"].to_list()

worst_sharpe_list = worst_sharpe["index"].to_list()
#Plot a streamlit interface to visualize the results of our analysis: 

st.title("Master thesis:")

tab1, tab2, tab3, tab4, tab5= st.tabs(["illiquidity over time", "portfolio optimization", "scenarios","Illiquidity correlation","polars"])


with tab1: 

   
    st.subheader("**Disclaimer**")
    st.write("""
        Concatenating the different measures using a python API in a single space to return an idea of the different measures that we might 
        encounter when it comes to estimate and quantify an illiquidity variable. \n
        This application only serve as an indicator and do not at any time justify a decision to invest in some products, the only objective is to provide some 
        useful information about Illiquidity trends and how we can manage them. \n
        Most of the measure used in this paper have been proved to be efficient but they can be subject to some limitations and we do not 
        guarantee at any time the accuracy of all the results
         """)

    st.markdown("***")
#Create part that will return information about the illiq versus returns relationship: 

    #Present the evolution of illiquidity over time using a monthly basis: 
    st.subheader("Illiquidity Evolution over time (Monthly Basis):")

    st.write("""
            
            In the following graph, we will present the evolution of the monthly illiquidity over time\n 
            this evolution is based on the function get monthly illiq

    """)
    st.plotly_chart(illiq_evol)


    st.subheader("Illiquidity over months comparison")

    st.write("""In this part we will try to present the average illiquidity during the month of the year and make a special focus on the january month versus
             the other month of the year """)
    
    #create a filter to select the stock period you wish to study 

    filters = st.multiselect("Choose a month and year input", list_month_year, default = list_month_year[0:12:1])

    header_filter = monthly_illiq.drop(["Date"])

    #Create a dictionnary with the options available
    portfolio_options = {
        "Least Liquid": worst_sharpe_list,
        "Most Liquid": select_header
    }

    pd.DataFrame(portfolio_options  ).to_clipboard()

    #Choose among the available options
    selected_portfolio_label = st.selectbox(
        "Choose between the most liquid or less liquid stocks",
        list(portfolio_options.keys())
    )

    # Create a list composed of the desired stocks 
    selected_portfolio = portfolio_options[selected_portfolio_label]


    stockname = st.multiselect(
        "Choose the stocks that might interest you",
        options= monthly_illiq.columns,
        default=selected_portfolio
    )

    illiq_change = illiq.illiq_period(filters, monthly_illiq, stockname)

    st.markdown("""
        <h5> Table of illiquidity change: \n</h5>
        """, unsafe_allow_html=True)
    
    # Print the table composed of the most liquid stocks 

    illiq_change =  illiq_change.with_columns([
                        pl.col(c).map_elements(lambda x: "{:.3%}".format(x)).alias(c)
                        for c in illiq_change.columns if c != "Date"
                    ])
    
    illiq_change = illiq_change.with_columns(pl.col("Date").dt.strftime("%Y-%m-%d"))

    st.dataframe(illiq_change,
                 use_container_width= True,
                   hide_index = True)

    yesno = st.button("get visual", use_container_width= True)


    if yesno: 
    #Build a function to get the visual based on the previous data frame
        st.plotly_chart(illiq.illiq_visual(illiq_change))
    
    # Compute the descriptive statistics of the illiquidity measure over the selected period
        stats = illiq.illiq_period(filters, monthly_illiq, stockname)

        stats_bis = stats.clone()

        des_stats = stats.describe()

        des_stats = des_stats.to_pandas()

        des_stats =  des_stats.drop(columns = "Date")

        des_stats = des_stats[des_stats.iloc[:, 0].isin([ 'mean', 'min', 'max', ])]

        des_stats.set_index("statistic", inplace = True)

# Compute spread (max - min)
        spread_row = np.round((des_stats.loc['max'] - des_stats.loc['min'])*100, 3)
        spread_row.name = 'spread *100 '

        
        des_stats = des_stats.applymap(lambda x: "{:.3%}".format(x) if isinstance(x, (int, float)) else x)

        des_stats = pd.concat([des_stats, spread_row.to_frame().T])

        st.dataframe(des_stats,use_container_width=True, hide_index = False)

     #Test the stationarity of the illiquidity measure

    st.markdown("***")
    st.subheader("Test the stationarity of illiquidity using the ADF test")

    adf_test = linreg.adf_test(monthly_illiq, stockname)

    stationary_true = adf_test[adf_test["Stationary"] == "Yes"]

    stationary_false = adf_test[adf_test["Stationary"] == "No"]

    st.table(stationary_true.set_index("Stock"))
    list_stationary = stationary_true["Stock"].to_list()


    st.plotly_chart(linreg.adf_evol_illiq(monthly_illiq, list_stationary), use_container_width=True)
    st.table(stationary_false.set_index("Stock"))


    list_not_stationary = stationary_false["Stock"].to_list()
    st.plotly_chart(linreg.adf_evol_illiq(monthly_illiq, list_not_stationary), use_container_width=True)

    st.markdown("***")
    st.subheader("Asset pricing with liquidity risk")

    st.write("""We will now compute the measure of the returns using the LCAPM of Amihud, using the liquidity 
             adjusted returns we exptect that the model will present a compensation for the illiquidity carried by the most illiquid
             ptf


""")


    st.markdown("""<h6> Get the illiquidity cost Ct to define a normalized measure of illiquidity:</h6>
                """,
                unsafe_allow_html = True)
    
    image = r"C:\Users\grego\Documents\Captures d’écran\market portfolio cap.png"
    illiq_cost = r"C:\Users\grego\Documents\Captures d’écran\illiquidity_persistent_cost.png"
    betas = r"C:\Users\grego\Documents\Captures d’écran\get_illiqBetas.png"



    col1,col2 =st.columns(2)

    with col1: 
        st.image(image)

    with col2: 
        st.image(illiq_cost)

    st.image(betas)

with tab2: 
    
    #read the excel containing the data sample previously randomly drawed using pandas: 
    path_directory1 = r"C:\Users\grego\Documents\python"

    path_file1 = os.path.join(path_directory1, "SP_500 composition.xlsx") 


    df_sample = pd.read_excel(path_file1, sheet_name ="sample")

    #créer une liste des tickers que nous souhaitons analyser
    list_ticker = df_sample["Symbole boursier"].to_list()

    # proposer un dataframe editable ou les stocks sont classés par poids dans le SP500:
    st.subheader("List of stock consider in the analysis")

    st.dataframe(df_sample.sort_values("Poids dans l'indice (en %)", ascending= False), 
                   use_container_width=True, 
                   hide_index = True)

    st.subheader("Illiquidity measure of Amihud versus returns:")

    #à partir de la liste de ticker que nous avons sélectionnés montrer le rapport entre les returns à la mesure illiq de AMIHUD: 

    st.plotly_chart(illiq.Average_illiq_versus_returns(return_illiq))  

    st.write("""
            Now that we have plot the return versus illiquidity we will display there relation ship in the following dataframe, as a portfolio manager, 
            we wish to maximise the return will minimizing the illiquidity we could be subject to, considering this finding of the modern
            portfolio theory we will observe and select the stocks that seems to offer the best return versus illiquidity profile
    """
            )
    
    df = return_illiq.to_pandas()
    
    # Proceed to a linear regression to observe the relationship between the illiq and returns 
    df1 = df.copy()
    df1 = df1.set_index("name").T
    regression = linreg.simple_linear_regression(df1["Average illiquidity"], df1["Average return"])
    st.write(regression)

    df_header = df.columns.to_list()

    df_header.remove("name")

    df[df_header] = df[df_header].applymap(lambda x: "{:.2%}".format(x))

    st.dataframe(df, hide_index = True)

    #after ploting the returns observe the correlations between the stocks that we have selected over the last 3 years

    st.markdown("***")

    num_portfolios = 10000

    values1 = values.to_pandas()
    illiq_measure1 = illiq_measure.to_pandas()

    illiq_eff_frontier =  compute_illiq_ef_frontier(values1,illiq_measure1, num_portfolios)
    illiq_eff_frontier= illiq_eff_frontier.sort_values(by="Sharpe illiq ratio", ascending = False)
    
    chart, data_best = ef.illiq_plot_efficient_frontier(illiq_eff_frontier)

    st.plotly_chart(chart)

    st.subheader("Analysis of the portfolio returns best sharpe illiq ratio over the period:")

  


    st.dataframe(data_best.head(), use_container_width = True, hide_index = True)
    


    #plot the dataframe of the top sharpe ratio 

    headers_best_sharpe = data_best['Asset Sample'].to_list()[0]

   



    select_correl = st.multiselect("Please select the components of your portfolios to build the correlation matrix",
                                   headers_best_sharpe, 
                                   default = headers_best_sharpe)
    st.write("""
    **Correlation analysis are often used to build portfolio that are efficient, as we wish to dispose from both the performance and the diversification benefit inside our portfolio.
        A quick reminder of the rules about correlation could be done here:**\n 
        - If the correlation is equal to 0 it means that the two underlying are not related\n 
        - A correlation close to 1 means that the underlying are moving in the same way \n
        - A correlation that is negative mean that the underlyings are moving in opposite ways wich offer a diversification benefit
    """)


    st.plotly_chart(illiq.Get_correlation(returns, select_correl))



    st.subheader("Get the relative illiquidity and the return of the portfolio")

    #get the header of the monthly returns dataframe
    stock_list = st.multiselect("Choose the components of your portfolio", monthly_returns.columns,default=headers_best_sharpe)

    st.write(values.head())

    col1,col2 = st.columns(2)

    with col1:

        if len(stock_list)>=1: 

            st.subheader("Returns portfolio")
            portfolio_returns= illiq.portfolio_returns(stock_list,monthly_returns)

            portfolio_returns_df = portfolio_returns.to_pandas()

            portfolio_returns_df["Date"]= portfolio_returns_df["Date"].dt.strftime("%Y-%m-%d")
            portfolio_returns_df["Portfolio_Returns"] = portfolio_returns_df["Portfolio_Returns"].map(lambda x:"{:.2%}".format(x))

            st.dataframe(portfolio_returns_df, use_container_width = True, hide_index = True)

            

    with col2: 

        
        vl100= illiq.VL_100(values,stock_list,market_returns1)

        
        # Create the figure
        create_fig = go.Figure()

        # Add market index trace
        create_fig.add_trace(go.Scatter(
            x=vl100["Date"],
            y=vl100["index_100"],
            mode='lines',
            name='Market Index (Base 100)'
        ))

        # Add portfolio trace
        create_fig.add_trace(go.Scatter(
            x=vl100["Date"],
            y=vl100["ptf_100"],
            mode='lines',
            name='Portfolio (Base 100)-best illiq'
        ))

        st.plotly_chart(create_fig,use_container_width = True)

        
    if len(stock_list)>=1:

        st.subheader("descriptive statistics ptf vs market return")
            
        ptf_return = portfolio_returns.to_pandas()

        ptf_return = ptf_return.set_index("Date")

        des_stat_ptf = pd.DataFrame(ptf_return)["Portfolio_Returns"].describe()

        des_stat_ptf = pd.DataFrame(des_stat_ptf).transpose()

        market_returns_stats = market_returns1.to_pandas()

        market_returns_stats["Date"] = pd.to_datetime(market_returns_stats["Date"])


        market_returns_stats.set_index("Date", inplace=True)

        market_returns_stats = market_returns_stats.resample('M').ffill()

        market_returns_stats  = market_returns_stats.pct_change()

        market_returns_stats = market_returns_stats[market_returns_stats.index > "2018-12-31"]

        des_stat_market = market_returns_stats["^GSPC"].describe()


        des_stat_market = pd.DataFrame(des_stat_market).transpose()

        descriptive_stats = pd.concat([des_stat_market,des_stat_ptf])

        # Format the table for the descriptive statistics

        descriptive_stats_list = descriptive_stats.columns.to_list()

        descriptive_stats_list.remove("count")

        for c in descriptive_stats_list: 

            descriptive_stats[c] = descriptive_stats[c].map("{:.2%}".format)

        st.dataframe(descriptive_stats, use_container_width=True)       


    # Create a formatting between the parts of the report
    st.markdown("***")
    st.header("Construction of the portfolio with the worst illiquidity sharpe ratio")


    
    #Manage the format of the worst sharpe ratio: 


    st.dataframe(illiq_eff_frontier.tail())

    headers_worst_sharpe = illiq_eff_frontier['Asset Sample'].to_list()[-1]

    # Give a header to the table that will create a  separation in the table: 

    st.markdown("""
                <h6>Build a correlation matrix for the worst returns: </h6>
                """, unsafe_allow_html=True)
    
    selected_worst = st.multiselect("list of the worst sharpe illiq ratio", 
                   headers_worst_sharpe, 
                   default= headers_worst_sharpe)
    
    st.plotly_chart(illiq.Get_correlation(returns, selected_worst))
    
    # Plot an analysis of the worst portfolio based on the porfolio returns using
    #  the worst possiblities in term of sharpe illiq ratio
    colworst1, colworst2 = st.columns(2)

    with colworst1: 

        if len(selected_worst)>1: 

            worst_portfolio = illiq.portfolio_returns(selected_worst,monthly_returns)

            # Format the values in the dataframe in a % format and date format: 
            worst_portfolio_df = worst_portfolio.to_pandas()
            worst_portfolio_df["Date"]= worst_portfolio_df["Date"].dt.strftime("%Y-%m-%d")
            worst_portfolio_df["Portfolio_Returns"] = worst_portfolio_df["Portfolio_Returns"].map(lambda x:"{:.2%}".format(x))

            # show the dataframe to the user
            st.dataframe(worst_portfolio_df, use_container_width=True, hide_index=True)

    with colworst2: 

        vl100_worst= illiq.VL_100(values,selected_worst,market_returns1)

        

        # Create the figure
        create_fig_worst = go.Figure()

        # Add market index trace
        create_fig_worst.add_trace(go.Scatter(
            x=vl100_worst["Date"],
            y=vl100_worst["index_100"],
            mode='lines',
            name='Market Index (Base 100)'
        ))

        # Add portfolio trace
        create_fig_worst.add_trace(go.Scatter(
            x=vl100_worst["Date"],
            y=vl100_worst["ptf_100"],
            mode='lines',
            name='Portfolio (Base 100)-worst illiq'
        ))

        st.plotly_chart(create_fig_worst,use_container_width = True)
    
    if len(stock_list)>=1:

        st.subheader("descriptive statistics less liquid portfolio vs market return")
            
        
        # Plot the descriptive statistics for the portfolio with the worst returns

        des_stat_ptf_worst = pd.DataFrame(worst_portfolio.to_pandas())["Portfolio_Returns"].describe()

        # tranpose the result for formatting purposes
        des_stat_ptf_worst = pd.DataFrame(des_stat_ptf_worst).transpose()

        # Concat the two tables in a single one: 
        descriptive_stats_worst = pd.concat([des_stat_market,des_stat_ptf_worst])

        # Format the table for the descriptive statistics

        # create a list for the portfolio headers
        descriptive_stats_list = descriptive_stats.columns.to_list()

        # drop the columns count from the sample to convert in % 
        descriptive_stats_list.remove("count")

        for c in descriptive_stats_list: 

            descriptive_stats_worst[c] = descriptive_stats_worst[c].map("{:.2%}".format)

        # Show the result to the user
        st.dataframe(descriptive_stats_worst, use_container_width=True)
    
    st.markdown("***")

    st.subheader("Portfolio optimized with a classical sharpe ratio")

    # Plot the efficient frontier for the portfolio equiponderated and sort the most liquid portfolio among the values 

    

    portfolios_df = compute_efficient_frontier(values1, num_portfolios)

    portfolios_df = portfolios_df.sort_values(by=["Sharpe ratio"], ascending= False)
    
    # Best Sharpe ratio portfolios
    st.dataframe(portfolios_df.head(), hide_index = True)

    # Show the efficient frontier: 
    st.plotly_chart(ef.plot_efficient_frontier(portfolios_df))


    # Create a list of the values we have in the extract and sort the best sharpe ratio portfolio
    classic_sharpe = portfolios_df["Asset Sample"].to_list()[0]

    # select by default the
    stock_list_bis = st.multiselect("Choose values in the portfolio",header_filter.columns, default = classic_sharpe)

    result_bis = illiq.portfolio_returns_bis(stock_list_bis, monthly_returns)

    # Get the correlation matrix

    sharpe_correl = illiq.Get_correlation(returns,stock_list_bis)
    
    st.plotly_chart(sharpe_correl)

    st.markdown("""
                <h6> Portfolio analysis when we select stock with a classic sharpe ratio</h6>
    """, unsafe_allow_html=True)
    # Define the columns to cover in the analysis: 

    sharp1,sharp2 = st.columns(2)
    
    # Set the frist columns analysis

    with sharp1: 

        # apply the correct format to the date and to the values (get them in %)

        result_sharpe_classic = result_bis.to_pandas()

        sharpe_classic_header = result_sharpe_classic.columns.to_list()

        sharpe_classic_header.remove("Date")

        result_sharpe_classic[sharpe_classic_header] = result_sharpe_classic[sharpe_classic_header].map("{:.2%}".format)
        #  format the date : 
        

        result_sharpe_classic["Date"] = result_sharpe_classic["Date"].dt.strftime("%Y-%m-%d")

        result_sharpe_classic = result_sharpe_classic[result_sharpe_classic["Date"] > "2018-12-31"]


        st.dataframe(result_sharpe_classic, use_container_width = True ,hide_index = True)
    
    with sharp2: 

        
        vl100_sharpe= illiq.VL_100(values,stock_list_bis,market_returns1)

        

        # Create the figure
        create_fig_sharpe = go.Figure()

        # Add market index trace
        create_fig_sharpe.add_trace(go.Scatter(
            x=vl100_sharpe["Date"],
            y=vl100_sharpe["index_100"],
            mode='lines',
            name='Market Index (Base 100)'
        ))

        # Add portfolio trace
        create_fig_sharpe.add_trace(go.Scatter(
            x=vl100_sharpe["Date"],
            y=vl100_sharpe["ptf_100"],
            mode='lines',
            name='Portfolio (Base 100)- Sharpe ratio'
        ))

        st.plotly_chart(create_fig_sharpe)

    if len(stock_list)>=1:

        st.subheader("descriptive statistics classic sharpe portfolio vs market return")
            
        
        # Plot the descriptive statistics for the portfolio  optimized with a normal sharpe ratio

        des_stat_sharpe= pd.DataFrame(result_bis.to_pandas())

        des_stat_sharpe["Date"] = des_stat_sharpe["Date"].dt.strftime("%Y-%m-%d")

        des_stat_sharpe = des_stat_sharpe[des_stat_sharpe["Date"] > "2018-12-31"]
        
        des_stat_sharpe= des_stat_sharpe["Portfolio_Returns"].describe()

        # tranpose the result for formatting purposes
        des_stat_ptf_sharpe = pd.DataFrame(des_stat_sharpe).transpose()

        # Concat the two tables in a single one: 
        descriptive_stats_sharpe = pd.concat([des_stat_market,des_stat_ptf_sharpe])

        # Format the table for the descriptive statistics

        # create a list for the portfolio headers
        descriptive_stats_list = descriptive_stats_sharpe.columns.to_list()

        # drop the columns count from the sample to convert in % 
        descriptive_stats_list.remove("count")

        for c in descriptive_stats_list: 

            descriptive_stats_sharpe[c] = descriptive_stats_sharpe[c].map("{:.2%}".format)

        # Show the result to the user
        st.dataframe(descriptive_stats_sharpe, use_container_width=True)

    # Create a new part containing the comparison of the different portfolios: 
    st.markdown("***")
    st.subheader("Comparison of the different portfolio")

    st.write("""In this part we wish to compare our different portfolio and see wish strategy offer the best possibilities over
             the selected period, by doing so we expect to compare different indicators such as: """)

    
    compare_ptf = go.Figure()

    # Add the portfolio trace when we optimize the portfolio with the classic sharpe illiq ratio
    compare_ptf.add_trace(go.Scatter(
            x=vl100_sharpe["Date"],
            y=vl100_sharpe["ptf_100"],
            mode='lines',
            name='Portfolio (Base 100)- Sharpe ratio'
        ))
    
    # Add portfolio trace when we optimize with worst sharpe illiq ratio
    compare_ptf.add_trace(go.Scatter(
            x=vl100_worst["Date"],
            y=vl100_worst["ptf_100"],
            mode='lines',
            name='Portfolio (Base 100)-worst illiq'
        ))
    
    # Add portfolio trace when we optimise with best sharpe illiq ratio
    compare_ptf.add_trace(go.Scatter(
            x=vl100["Date"],
            y=vl100["ptf_100"],
            mode='lines',
            name='Portfolio (Base 100)-best illiq'
        ))
    
    compare_ptf.update_layout(
             title ='Comparison of the VL Base 100 of our different porfolios', 
            xaxis_title = 'Date',
             yaxis_title = "VL base 100"

    )

    
    st.plotly_chart(compare_ptf, use_container_width=True)
    
    # Compare the descriptive statistics of the portfolio: 

    st.markdown("""**Get the descriptive stats for all portfolios**""")

    # Format the most liquid portfolio in the sample: 
    best = pd.DataFrame(descriptive_stats.iloc[-1])

    best.rename(columns={"Portfolio_Returns": "Most liquid portfolio"}, inplace=True)

    best = best.transpose()

    # Format the less liquid portfolio
    worst = pd.DataFrame(descriptive_stats_worst.iloc[-1])

    worst.rename(columns={"Portfolio_Returns": "Less liquid portfolio"}, inplace=True)

    worst = worst.T

    # Format the classic optimised portfolio: 

    classic = pd.DataFrame(descriptive_stats_sharpe.iloc[-1])

    classic.rename(columns={"Portfolio_Returns": "Classic Sharpe portfolio"}, inplace=True)

    classic = classic.T

    # compute and show the descriptive stats 
    final_des_stat = pd.concat([classic,best,worst])

    st.dataframe(final_des_stat.reset_index(), use_container_width=True, hide_index = True)
    
    # Study the skewness and the kurtosis of the 3 portfolios: 


    # Resample the data of the sharpe optimized portoflio on a correct horiszon
    sharpe_data_ptf = result_bis.to_pandas()
    sharpe_data_ptf["Date"] = sharpe_data_ptf["Date"].dt.strftime("%Y-%m-%d")
    sharpe_data_ptf = sharpe_data_ptf[sharpe_data_ptf['Date']> '2018-12-31']

    valdf = values.to_pandas()

    valdf["Date"] = valdf["Date"].dt.strftime("%Y-%m-%d")
    valdf = valdf[valdf['Date']> '2018-12-31']

#   Get the daily returns of each portfolio

    best_daily_returns = illiq.portfolio_returns_bis(headers_best_sharpe,illiq.Get_returns(pl.DataFrame(valdf))).to_pandas().set_index("Date")    
    worst_daily_returns = illiq.portfolio_returns_bis(headers_worst_sharpe,illiq.Get_returns(pl.DataFrame(valdf))).to_pandas().set_index("Date")
    portfolio_returns = illiq.portfolio_returns_bis(classic_sharpe,illiq.Get_returns(pl.DataFrame(valdf))).to_pandas().set_index("Date")
    st.write(best_daily_returns.head())

    kurt_best= pd.DataFrame(best_daily_returns).kurt()
    skew_best = pd.DataFrame(best_daily_returns).skew()


    kurt_worst = pd.DataFrame(worst_daily_returns).kurt()
    skew_worst = pd.DataFrame(worst_daily_returns).skew()

    kurt_sharpe = pd.DataFrame(portfolio_returns).kurt()
    skew_sharpe = pd.DataFrame(portfolio_returns).skew()


    # Create a dictionnary with our inputs: 

    dico = {"Kurt most liquid ptf":kurt_best,
            "Skew most liquid ptf":skew_best,
            "Kurt less liquid ptf":kurt_worst,
            "Skew less liquid ptf":skew_worst,
            "Kurt classic ptf":kurt_sharpe,
            "Skew classic ptf":skew_sharpe,
            }
    
   
    skewness_kurt_study = pd.DataFrame(dico)

    st.markdown("**Skewness and Kurtosis analysis for the portfolios returns (dayly basis)**")

    st.dataframe(skewness_kurt_study, use_container_width = True, hide_index = True)

    # Plot the distribution of returns for each values : 

    st.markdown("\n")
    dis1,dis2,dis3 = st.columns(3)
    
    with dis1: 
        st.markdown("**Most liquid portfolio returns distribution**")
        st.plotly_chart(illiq.get_histo(best_daily_returns
                                        ))
    with dis2: 
        st.markdown("**Less liquid portfolio returns distribution**")
        st.plotly_chart(illiq.get_histo(worst_daily_returns))

    with dis3: 

        st.markdown("**Sharpe portfolio returns distribution**")
        st.plotly_chart(illiq.get_histo(portfolio_returns))

    
    #print the list of common stock between the two porfolio select_header / classic_sharpe

    common = list(set(classic_sharpe)& set(headers_best_sharpe))

    #Imprimer les valeurs commune entre les deux portefeuilles

    common_stock = pd.DataFrame({"Symbole boursier":common})

    common_variables = pd.merge(common_stock, df_sample[["Symbole boursier","Entreprise","Secteur d'activité"]], on="Symbole boursier", how='inner')

# Present the number of outputs in common and the sector two portolios are commonly exposed 
    st.write("**Common stock between the most liquid and sharpe portfolio**")
    st.dataframe(common_variables,use_container_width=True, hide_index=True)

# Count the number of stocks in common between the two portfolios: 
    st.write(f"There is between the two portfolios **{common_variables["Entreprise"].count()}** stocks in common")

    # Create a dataframe with the portfolio values: 

    portfolio_all = pd.merge(vl100[["Date",'ptf_100']],vl100_worst[["Date",'ptf_100']], on = "Date")

    

    portfolio_all = pd.merge(portfolio_all, vl100_sharpe[["Date",'ptf_100']], on= "Date")


    portfolio_all = portfolio_all.rename(columns ={"ptf_100_x":"Most liquid ptf",
                                                  "ptf_100_y":"Less liquid ptf",
                                                  "ptf_100":"MPT optimized ptf"})
    
    portfolio_all_returns = portfolio_all.copy()

    portfolios_name = portfolio_all.columns.to_list()

    portfolios_name.remove("Date")

    portfolio_all = pl.DataFrame(portfolio_all)

    portfolio_all_returns = portfolio_all_returns.set_index("Date").pct_change()

    portfolio_all_returns = portfolio_all_returns.reset_index()

    portfolio_all_returns = pl.DataFrame(portfolio_all_returns)

    
    st.plotly_chart(illiq.Get_correlation(portfolio_all_returns,portfolios_name))

    # Get a measure of the portfolio VaR :

    var_liq = var.VaR_99(values,headers_best_sharpe).rename("VaR 99% Liquid ptf")
    var_illiq = var.VaR_99(values, worst_sharpe_list).rename("VaR 99% Illiquid ptf")
    var_sharpe = var.VaR_99(values, classic_sharpe).rename("VaR 99% Sharpe ptf")

    var_measure = pd.concat([var_liq, var_illiq, var_sharpe], axis=1)

    st.markdown("""
                <h6> Get the measure of the VaR 99% :</h6>
                """, unsafe_allow_html=True)

    st.dataframe(var_measure, hide_index= True, use_container_width=True)

    es_liq = var.CVaR_99(values,headers_best_sharpe).rename("cVaR 99% Liquid ptf")
    cvar_illiq = var.CVaR_99(values, worst_sharpe_list).rename("cVaR 99% Illiquid ptf")
    cvar_sharpe = var.CVaR_99(values, classic_sharpe).rename("cVaR 99% sharpe ptf")

    cvar_measure = pd.concat([es_liq, cvar_illiq, cvar_sharpe], axis=1)
    
    st.markdown("""
                <h6> Get the measure of the cVaR 99% :</h6>
                """, unsafe_allow_html=True)
    
    st.dataframe(cvar_measure, hide_index= True, use_container_width=True)

with tab3:

    st.subheader("Create a shock on a specific portfolio: ")
    

    # Create the portfolio VL dataframe and use is to compare the VL base 100

    selection = st.multiselect("Choose a portfolio to shock",portfolios_name, default = 'Most liquid ptf')
    shock = st.slider("Choose the impact in percent you wanna do",-50,50,1 )
    start_date = st.selectbox("choose the date to apply the shock",portfolio_all["Date"].to_list())

    shocked_vl = sce.apply_shock(portfolio_all, selection,start_date, shock)

    #create the graph of the shocked vl using plotly: 
        
    shocked_vl = shocked_vl.to_pandas()

    shocked_vl["Date"] = shocked_vl["Date"].dt.strftime("%Y-%m-%d")

    st.dataframe(shocked_vl.tail(), use_container_width = True, hide_index = True)

    xaxis = shocked_vl["Date"]
    yaxis = shocked_vl.drop(columns=["Date"])  # Drop to get numerical data

    # Initialize figure
    fig = go.Figure()

    # Add traces for each column
    for column in yaxis.columns:
        fig.add_trace(go.Scatter(x=xaxis, y=yaxis[column], mode='lines', name=column))
        
    # Update layout
    fig.update_layout(
        title='VL change over time after a shock',
        xaxis_title="Date",
        yaxis_title="Adjusted Price",
        xaxis_tickangle=45
    )

    st.plotly_chart(fig)



with tab4: 

    st.write("""The objective of this tab is to study how the illiqudity measure of the stock 
    selected in the most liquid porfolio comoves versus the movement we can observe inside the sharpe optimized portofolio
    correlation is calculated on a daily basis since 2010 for the different stocks selected. We are therefore ensuring that the market is exposed to
             tension scenarios and normal scenarios """)

    st.markdown("***")

    st.subheader("Illiquidty correlation matrix for the optimized portofolio with illiq sharpe ratio")

    st.plotly_chart(illiq.illiquidity_correlation(illiq_measure,headers_best_sharpe))

    st.subheader("Illiquidty correlation matrix for the optimized portofolio with sharpe ratio")

    st.plotly_chart(illiq.illiquidity_correlation(illiq_measure,classic_sharpe))

    st.subheader("Illiquidity correlation matrix for the optimized portfolio with the worst illiq sharpe ratio")

    st.plotly_chart(illiq.illiquidity_correlation(illiq_measure, worst_sharpe_list))



with tab5:

    st.markdown("""

                <h2> polars documentation </h2>

                   <iframe title="Polars documentation"
                    width="1000"
                    height="600"
                    src="https://docs.pola.rs/"
                    frameborder="0"
                    allowFullScreen="true">
                    </iframe>      
                """, unsafe_allow_html=True)
    
    st.markdown("""

                <h2> Asset pricing with Liquidity risk </h2>

                   <iframe title="Polars documentation"
                    width="1000"
                    height="600"
                    src="http://docs.lhpedersen.com/liquidity_risk.pdf"
                    frameborder="0"
                    allowFullScreen="true">
                    </iframe>      
                """, unsafe_allow_html=True)
    
