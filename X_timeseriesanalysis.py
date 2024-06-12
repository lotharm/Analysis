# -*- coding: utf-8 -*-

## Imports & Inits 
from turtle import color
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


from pathlib import Path
import csv

import datetime
# from datetime import datetime
# from datetime import date
import calendar

import os
from os import listdir
from os.path import isfile, join

import sys, getopt

# Imports
#from pandas_datareader import data as pdr
#from yahoo_fin import stock_info as si

import numpy as np
from sklearn import linear_model
import scipy.stats


import pandas as pd
import numpy as np

import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
###################################################################

import os
import sys

# matplotlib qt

plt.rcParams['figure.figsize'] = [10,6]
plt.rcParams.update({'font.size': 18})
plt.style.use('seaborn')
plt.rcParams['axes.grid'] = True

currentdir = os.path.abspath('')
parentdir = os.path.realpath(os.path.join(currentdir, '..'))
sys.path.insert(0, parentdir) 
libdir = os.path.realpath(os.path.join(parentdir, 'TC'))
sys.path.insert(0, libdir)

import signalgenerator as sg
import underly  as ul

from lib import Indikatoren

#from sympy import maximum

import RSI_strat_SETUP


#init_notebook_mode()

from lib import Indikatoren

#pio.renderers.default = "vscode"
#pio.renderers.default = "browser"

def plot_system2(data,vol="FALSE"):
    df2 = data.copy()
    dates = np.arange(len(df2)) # We need this for mpl.plot()
    price = df2['Close']
    h_sma = df2['Close'].ewm(span=200,adjust=False).mean()
    l_sma = df2['Close'].ewm(span=100,adjust=False).mean()
    c_ema = df2['Close'].ewm(span=2,adjust=False).mean()
    
    if vol==True:
        with plt.style.context('fivethirtyeight'):
            fig, axes = plt.subplots(2,figsize=(10,10))
            mpf.plot(df2, ax=axes[0],  show_nontrading=False, type='candle')
            #ax.plot(dates, h_sma, linewidth=2, color='red', label='200EMA')
            #ax.plot(dates, l_sma, linewidth=2, color='green', label='100EMA')
            axes[0].plot(dates, h_sma, linewidth=1, color='blue', label='200EMA')
            axes[0].axhline(y=1935)
            axes[1].plot(df2["Volume"],color='purple', label='Vol.')
            axes[1].plot(df2["Volume"].ewm(span=21,adjust=False).mean(),linewidth=0.2,color='red', label='Vol.')
            plt.title("A System ")
            #ax.set_ylabel('Price($)')
            #plt.legend()
    else:
        with plt.style.context('fivethirtyeight'):
            mpf.plot(df2,tight_layout=True,show_nontrading=False, figscale=2, type='candle')
            plt.title("A System ")
    plt.show() # This is needed outside of Jupyter

def figupdate_nonHist(figure,xtit="",ytit=""):
    figure.update_layout(
        height=500,
        xaxis_title=xtit,
        yaxis_title=ytit,
        font=dict(
                family="Arial",
                size=8,
                color='#000000'
            ),
        )
    figure.show()

def figupdate(figure):
    figure.update_layout(
        height=500,
        font=dict(
                family="Arial",
                size=8,
                color='#000000'
            ),
        )
    figure.show()


def hist_cum_plot(df,title="Histogramn and Cum Dist.",bin=30,blocker=False): 
    fig, ax = plt.subplots(2,figsize=(8,6))
    fig.suptitle(title,fontsize = 6)
    sns.histplot(df,bins=bin,stat="density",ax=ax[0])
    
    print("------------------------------------------")
    print(title+"\n")
    print(df.describe())
    print("------------------------------------------")
    sns.histplot(df,bins=bin,cumulative=1,stat="density",ax=ax[1])
    plt.show(block=blocker)

def hist_cum_plot_simple(df,title="Histogramn and Cum Dist.",bin=30,blocker=False): 
    fig, ax = plt.subplots(1,figsize=(6,4))
    fig.suptitle(title,fontsize = 8)
    sns.histplot(df,bins=bin,stat="density",ax=ax)
    print(df.describe())
    plt.show(block=blocker)


#Relative Strength Index
def RSI(df, n):
    #df["EMA200"]=df["Factor"].ewm(span=200,adjust=False).mean()
    close = df["Close"]
    delta = close.diff()
# Get rid of the first row, which is NaN since it did not have a previous
# row to calculate the differences
    delta = delta[1:]

# Make the positive gains (up) and negative gains (down) Series
    up, down = delta.clip(lower=0), delta.clip(upper=0).abs()

# Calculate the RSI based on EWMA
# Reminder: Try to provide at least `window_length * 4` data points!
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / roll_down
    rsi_ema = 100.0 - (100.0 / (1.0 + rs))

# Calculate the RSI based on SMA
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / roll_down
    rsi_sma = 100.0 - (100.0 / (1.0 + rs))



    df["rsi_ewma"]=rsi_ema
    df["rsi_sma"]=rsi_sma
    return df

# Widgets
startdate=widgets.Text(
    value='2022-01-01',
    placeholder='yyyy-mm-dd',
    description='Start Date',
    disabled=False   
)

enddate=widgets.Text(
    value='2023-06-06',
    placeholder='yyyy-mm-dd',
    description='End Date',
    disabled=False   
)


# Widgets
startdate2=widgets.Text(
    value='2022-10-01',
    placeholder='yyyy-mm-dd',
    description='Plot von',
    disabled=False   
)

enddate2=widgets.Text(
    value='2023-06-06',
    placeholder='yyyy-mm-dd',
    description='PLot bis',
    disabled=False   
)


#display(startdate,enddate,startdate2,enddate2)

## Read Data
universe = "fx"
ticker = "cl-15m_bk"

t_bwd = 10
r_bwd = 10000
t_fwd=  10
greatersmaller="smaller"

startdatum = "2023-05-01"
enddatum = "2023-10-01"

pf=ul.underlying(universe,ticker)
pf.read_grabbed_data(True)
# startdatum = "2011-01-01"

#Alternativ:
#file_to_read="C:\\Temp\\Trading\\ETFS\\RES_fx\\Data\\gc-15m_bk.csv"
#custom_date_parser = lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M")
#test = pd.read_csv(file_to_read,sep=";",decimal='.',index_col=["Date"],usecols=["Date","High","Open","Low","Close","Volume"],parse_dates=['Date'],date_parser=custom_date_parser)
    

rational  = "Nach " + str(t_bwd) + " Balken und r " + greatersmaller + str(r_bwd)
rational = rational + "% Rendite"+": Wie sieht die Verteilung der Renditen nach weiteren "+str(t_fwd) + " Balken aus."
print(rational)

# Fuelle den Dataframe mit weiteren Indikatoren und relevanten Scalars
mypath = RSI_strat_SETUP.mypath

#onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and not f.startswith("_")) ]

b=[]
c=[]
figs=[]
counter  = 0

###########################
##########################
time_gap_fwd = t_fwd
time_gap_back = t_bwd
rsi_threshold =75
return_threshold = r_bwd
my_bins = 50
########################
########################



data=pf.grabbed_data.truncate(before=startdatum,after=enddatum)

# make hourly timeseries
#############################
f="H"
#############################
High_d=data["High"].groupby(pd.Grouper(freq=f)).max()
Low_d=data["Low"].groupby(pd.Grouper(freq=f)).min()
open_d=data["Open"].groupby(pd.Grouper(freq=f)).first()
volume_d=data["Volume"].groupby(pd.Grouper(freq=f)).sum()
close_d=data["Close"].groupby(pd.Grouper(freq=f)).last()



d = {"High":High_d,"Low":Low_d,"Open":open_d,"Close":close_d,"Volume":volume_d}
daten_hourly= pd.DataFrame(d)
daten_hourly=daten_hourly.dropna()
## daten[] sind die auf den gewuenschten Zeitrahemn gebrachten uersprungsdaten
##########################################

#daten["pctchg"]=daten["Close"].pct_change()
#daten['Factor'] =  (daten['pctchg'] + 1).cumprod()
#daten=Indikatoren.ATR(daten,1)
#daten=Indikatoren.ATR(daten,100)
#daten["Mean_Vol"]=daten["Volume"].ewm(span=21,adjust=False).mean()
#daten["ATR_Vol"]=daten["Volume"]*(daten["ATR1"]/daten["ATR100"])
#daten=Indikatoren.Zscore_rolling(daten,"ATR_Vol",100)


# make daily timeseries
#############################
f="D"
#############################
High_d=data["High"].groupby(pd.Grouper(freq=f)).max()
Low_d=data["Low"].groupby(pd.Grouper(freq=f)).min()
open_d=data["Open"].groupby(pd.Grouper(freq=f)).first()
volume_d=data["Volume"].groupby(pd.Grouper(freq=f)).sum()
close_d=data["Close"].groupby(pd.Grouper(freq=f)).last()



d = {"High":High_d,"Low":Low_d,"Open":open_d,"Close":close_d,"Volume":volume_d}
daten_daily= pd.DataFrame(d)
daten_daily=daten_daily.dropna()


# make weekly timeseries
#############################
f="W-Mon"
#############################
High_d=data["High"].groupby(pd.Grouper(freq=f)).max()
Low_d=data["Low"].groupby(pd.Grouper(freq=f)).min()
open_d=data["Open"].groupby(pd.Grouper(freq=f)).first()
volume_d=data["Volume"].groupby(pd.Grouper(freq=f)).sum()
close_d=data["Close"].groupby(pd.Grouper(freq=f)).last()



d = {"High":High_d,"Low":Low_d,"Open":open_d,"Close":close_d,"Volume":volume_d}
daten_weekly= pd.DataFrame(d)
daten_weekly=daten_weekly.dropna()

Weeks=daten_weekly



# data=Indikatoren.ATR(data,20)
# data["EMA50"]=data["Factor"].ewm(span=50,adjust=False).mean()
# data["EMA10"]=data["Factor"].ewm(span=10,adjust=False).mean()
# data["EMA21"]=data["Factor"].ewm(span=21,adjust=False).mean()
# data["EMA100"]=data["Factor"].ewm(span=100,adjust=False).mean()
# data["EMA80"]=data["Factor"].ewm(span=80,adjust=False).mean()
# data["EMA200"]=data["Factor"].ewm(span=200,adjust=False).mean()

startzeit="00:00:00"
endzeit="23:30:00"

# 0 == Monday, 1= Tuesd...


Day1 = 2
Day2 = 3
## Waehle Tage aus den Stunden Zeitreiehn aus 

for d in [0,1,2,3,4,6]:
    #TheDay_data_hourly=data[data.index.weekday==d]
    TheDay_data_hourly=daten_hourly[daten_hourly.index.weekday==d]
#Friday_data_hourly=daten_hourly[daten_hourly.index.weekday==Day2]
# Waehle aus den gewählten TAgen das relevante Zeitfenseter heraus
    TheDay_data_hourly_subset=TheDay_data_hourly.loc[(TheDay_data_hourly.index.time >= pd.Timestamp(startzeit).time()) & (TheDay_data_hourly.index.time <= pd.Timestamp(endzeit).time())]

    occ=0
    for i in range(len(Weeks)):
        # Get the data for the specific Wednesday
        Week = Weeks.iloc[i]
        w=Week.name.strftime("%W")
        y=Week.name.strftime("%Y")
        # Filter the half-hourly data between 09:00 and 11:30 AM on that Wednesday
        TheDay_prices = TheDay_data_hourly_subset[(TheDay_data_hourly_subset.index.strftime("%W") == w)
                                          & (TheDay_data_hourly_subset.index.strftime("%Y") == y)]


        # Check if the maximum price during that time period is equal to the Wednesday's high
        print("TheDay High: ",round(TheDay_prices['High'].max(),3),"  ", "TheWeekHigh: ",  round(Week["High"],3))
        if round(TheDay_prices['High'].max(),3) >= round(Week["High"],3):
            print("catch!",round(TheDay_prices['High'].max(),3)," ", round(Week["High"],3))
            occ += 1
            #TheDay_prices

    print(d, occ , len(Weeks))

    Week=Weeks.iloc[2]
w=Week.name.strftime("%W")
y=Week.name.strftime("%Y")
        # Filter the half-hourly data between 09:00 and 11:30 AM on that Wednesday
TheDay_data_hourly_subset=TheDay_data_hourly.loc[(TheDay_data_hourly.index.time >= pd.Timestamp(startzeit).time()) & (TheDay_data_hourly.index.time <= pd.Timestamp(endzeit).time())]
TheDay_prices = TheDay_data_hourly_subset[(TheDay_data_hourly_subset.index.strftime("%W") == w)
                                          & (TheDay_data_hourly_subset.index.strftime("%Y") == y)]




