# -*- coding: utf-8 -*-



"""
Created on Wed Jul  7 17:46:12 2021

@author: Schroeder
"""

from turtle import color
import matplotlib
from matplotlib import colors



import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



from pathlib import Path
import csv

from datetime import datetime
from datetime import date
import calendar

## test of GIT

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


import numpy as np
from sklearn import linear_model

import pandas as pd

import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
###################################################################

import os
import sys
currentdir = os.path.abspath('')
parentdir = os.path.realpath(os.path.join(currentdir, '..'))
sys.path.insert(0, parentdir) 
libdir = os.path.realpath(os.path.join(parentdir, 'TC'))
sys.path.insert(0, libdir)

import signalgenerator as sg
import underly  as ul



#from sympy import maximum

import RSI_strat_SETUP

plt.rcParams['axes.grid'] = True


#init_notebook_mode()

from lib import Indikatoren

#pio.renderers.default = "vscode"
#pio.renderers.default = "browser"


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


def hist_cum_plot(df,title="Histogramn and Cum Dist.",bins=30,blocker=False): 
    fig, ax = plt.subplots(2,figsize=(8,6))
    fig.suptitle(title,fontsize = 6)
    sns.histplot(df,bins =30,stat="density",ax=ax[0])
    
    print("------------------------------------------")
    print(title+"\n")
    print(df.describe())
    print("------------------------------------------")
    sns.histplot(df,bins =30,cumulative=1,stat="density",ax=ax[1])
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



def main(pf,t_bwd,t_fwd,r_bwd, greatersmaller,condi):

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



    data=pf.grabbed_data.truncate(before=startdatum)
    
    # data[['MACD_day'],['MACDsign_day'],['MACDdif_day']].plot()
    # plt.show(block=False)
    
    # data[['MACD_week']['MACDsign_week'],['MACDdif_week']].plot()
    # plt.show(block=False)



    data = RSI(data, 21)
    
    data["EMA50_dist"] = data["Factor"]-data["EMA100"]
    # Some statistics:  pecentile:
    #df.rolling(window=3, center=False).apply(lambda x: pd.Series(x).quantile(0.75))
    
    #z-Score
    ############################
    window=100
    a=pf.zscore_rolling_scaled(1000,'Factor',"EMA100")
    data=pd.merge(data,a,left_index=True, right_index=True)

    hist_cum_plot(a["z-score"].dropna(),ticker+": z-score Histogramm ",60) 
    print("a:\n")
    print(a)
    #target_column = 'EMA50_dist'
    #roll = df[target_column].rolling(window)
    #df['z-score'] = (df[target_column] - roll.mean()) / roll.std()
    
    fig, ax = plt.subplots(figsize=(16,12))
    fig.suptitle(ticker+": Price(fat) vs. zscore")
    ax.plot(data[["Factor","EMA200","EMA100","EMA21","EMA50"]])
    c2 = ax.twinx()

    ax.set_title("z-score EMA_100/250, 100days sample")
    c2.plot(data["z-score"], color='green',linewidth=3.0)
    plt.show(block=False)

    
    #Schreibe raus den z-score
    ############################
    df = pd.DataFrame()
    ### Formeln geprueft und korrekt:
    # ############################################################################## 
    # mit positivem time_gap in .diff(time_gap):  df[t]= (f(t)-f(t-time_gap))/f(t)
    # also Veränderung ggü. f(t)
    ### Also backward looking return !
    df["back_"+str(t_bwd)]= 100*(data["Close"].diff(t_bwd)/data["Close"])
    
    # mit negativem time_gap in .diff(time_gap):  df[t]= (f(t+time_gap)-f(t))/f(t)
    # also Veränderung ggü. f(t). Der klassische Differenzenequotient also !!!!
    ### Also forckward looking return      
    df["fwd_"+str(t_fwd)]= 100*(-data["Close"].diff(-t_fwd)/data["Close"])
    #################################################################################
    fwd_yield=df["fwd_"+str(t_fwd)].dropna()
    hist_cum_plot(fwd_yield,ticker+": r_(+"+str(t_fwd)+"d)"+" yield Histogramm ",30) 


    ####   select where backward threshold has been passed


    if greatersmaller=="greater":
        a = df["fwd_"+str(t_fwd)][df["back_"+str(t_bwd)] > r_bwd]
    elif greatersmaller=="smaller":
        a = df["fwd_"+str(t_fwd)][df["back_"+str(t_bwd)] < r_bwd]
    
    a_with_threshold_backward = a.dropna()
    ####   select where forward threshold has been passed AS WELL :
    #a_with_threshold_backward_forward =a_with_threshold_backward[a_with_threshold_backward>=r_fwd]
    
    fg, ax = plt.subplots(figsize=(16,12))
    ax.bar(a_with_threshold_backward.index.to_list(),a_with_threshold_backward)
    ax.set_title("Dates, fullfilling bwd condition and the yield after fwd days")
    plt.show(block=False)
    
    #figupdate_nonHist(fig00,"Datum","---> Rendite nach "+ str(t_fwd)+ " Bars" + " ab r>" + str(r_fwd
        


    ##### rsi Statistik:
    rsi = data["rsi_sma"].dropna()
    hist_cum_plot(rsi,ticker+": RSI Histogramm",30) 


    rsi_sma_hist = np.histogram(rsi, bins=my_bins)
    rsi_sma_hist_dist = scipy.stats.rv_histogram(rsi_sma_hist)


    ##### renditen nach time_gap_back Zeitschrittten Statistik:
    backward = df["back_"+str(t_bwd)].dropna()
    hist_cum_plot(backward,ticker+": r_(-"+str(t_bwd)+"d)"+" yield Histogramm ",30) 

    # fig, ax = plt.subplots(2,figsize=(16,12))
    # ax.set_title(str(t_bwd)+"day yield Histogramm ")
    # sns.histplot(backward,bins =30,stat="density",ax=ax[0])
    # print(backward.describe())
    # sns.histplot(backward,bins =30,cumulative=1,stat="density",ax=ax[1])

    #### bilde das Histogrammobjekt, um die Cummulierten Wahrscienlichkeiten zu berechen
    backward_hist = np.histogram(backward, bins=my_bins)
    backward_hist_dist = scipy.stats.rv_histogram(backward_hist)

    ### 3. Histogramm der Renditen nach "time_gap_back" Tagen


    ### 2. Histogramm
    laenge = len(a_with_threshold_backward)
    pos = a_with_threshold_backward[a_with_threshold_backward>0]
    neg = a_with_threshold_backward[a_with_threshold_backward<0]
    negativ = len(neg)
    positiv = len(pos)
    tite =  condi + "\n"  + " t_bwd= -"+ str(t_bwd)+" bars,  t_fwd = "+ str(t_fwd)+ " bars "
    tite = tite +"\n" + "Anzahl fwd Intervalle mit r(t_fwd)>0 : "+str(positiv) + "|" +  " r(t_fwd) < 0 : "  + str(negativ) + "|"+ "p(r<0): " + str(round(negativ/laenge,2)) + "|"  
    tite = tite + "p(r>0): " +str(round(positiv/laenge,2)) + ", <r_pos>: " +  str(round(pos.mean(),2)) + ", <r_neg>: " +  str(round(neg.mean(),2))
    #ax = b_df.plot.hist(bins=my_bins,title="a)"+tit)
    hist_cum_plot(a_with_threshold_backward,ticker+": "+tite,30,True) 
    z=2

    ##RSI_strat_SETUP.figures_to_html([fig, figs[0],figRSIHist,fig0,fig1,fig2,fig3,fig4],RSI_strat_SETUP.output_path + "dashboard.html")


# if len(sys.argv)>=3:
#     universe = sys.argv[1]
#     ticker = sys.argv[2]
# else:
#     print("No Input from you; Need 2 Inputs=> <UNiverse> <ticker>")
#     exit()

# tickers = [ticker,]

if __name__ == "__main__":


    # if len(sys.argv)==8:
    #     universe = sys.argv[1]
    #     ticker=sys.argv[2]
    #     startdatum = sys.argv[3]
    #     t_bwd=int(sys.argv[4])
    #     r_bwd=float(sys.argv[5])
    #     t_fwd=int(sys.argv[6])
    #     greatersmaller= sys.argv[7]
    # else:  
    #     print("No Input from you; Need 7 Inputs=> <Universe> <ticker> <yyyy-mm-dd> <t_bwd> <r_bwd> <t_fwd>  <greater/smaller>")
    #     exit()

    universe = "fx"
    ticker = "COMEX_GC1"
    
    t_bwd = 10
    r_bwd = 10000
    t_fwd=  10
    greatersmaller="smaller"



    pf=ul.underlying(universe,ticker)
    pf.read_grabbed_data()
    # startdatum = "2018-07-01"

    enddatum = "2029-04-06"


    rational  = "Nach " + str(t_bwd) + " Balken und r " + greatersmaller + str(r_bwd)
    rational = rational + "% Rendite"+": Wie sieht die Verteilung der Renditen nach weiteren "+str(t_fwd) + " Balken aus."
    print(rational)

    #main(stock,          back_gap, fwd_gap, threshold rendite backward, threshold renidte forward,comment )

    main(pf,  t_bwd   ,  t_fwd, r_bwd , greatersmaller, rational)


