import os 
import pandas as pd
import numpy as np
import yfinance as yf


path_directory = r"C:\Users\grego\Documents\python"

path_file = os.path.join(path_directory, "SP_500 composition.xlsx")

#Using sample and random draw before our analysis is done to avoid selectionship bias in the analysis

df  = pd.read_excel(path_file)

print(df.head())

sample_extract = df.sample(100)

sample_extract.to_clipboard()

#get the financial datas for the extracted values 


#get the ticker list of the sp 500

df_sample = pd.read_excel(path_file, sheet_name = "sample")

list_ticker = df_sample["Symbole boursier"].to_list()

print(list_ticker)

#import streamlit as st

#st.data_editor(df_sample.sort_values("Poids dans l'indice (en %)", ascending= False), use_container_width=True)


yahoo_data = yf.download(list_ticker, start ='2010-01-01', end ='2024-12-31')["Volume"]

yahoo_data.to_clipboard()


market_returns = yf.download('^GSPC',start ='2010-01-01', end ='2024-12-31')["Close"]

market_returns.to_clipboard()