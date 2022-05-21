# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:46:12 2021

@author: Schroeder
"""


from pathlib import Path
import csv

from datetime import datetime
from datetime import date
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


import numpy as np
from sklearn import linear_model

import pandas as pd

import plotly.io as pio
import plotly.express as px
from plotly.figure_factory import create_candlestick
import plotly.graph_objects as go

from pandas import ExcelWriter


import RSI_strat_SETUP
import plotly.io as pio
pio.renderers.default = "vscode"
#pio.renderers.default = "browser"



#import plotly.io as pio
#import plotly.express as px
#from plotly.figure_factory import create_candlestick


#MACD, MACD Signal and MACD difference
# def MACD(df, n_fast, n_slow):
#     mac = pd.dataframe()
#     #df["EMA200"]=df["Factor"].ewm(span=200,adjust=False).mean()
#     EMAfast = df['Close'].ewm(span = n_fast, min_periods = n_slow - 1).mean()
#     EMAslow = df['Close'].ewm(span = n_slow, min_periods = n_slow - 1).mean()
#     MACD['MACD_' + str(n_fast) + '_' + str(n_slow)] = EMAfast - EMAslow
#     MACD['MACDsign_' + str(n_fast) + '_' + str(n_slow)] = MACD['MACD_' + str(n_fast) + '_' + str(n_slow)].ewm(span = 9, min_periods = 8).mean()
#     MACD['MACDdiff_' + str(n_fast) + '_' + str(n_slow)] = MACD - MACDsign
#     df = df.join(MACD)
#     df = df.join(MACDsign)
#     df = df.join(MACDdiff)
#     return df

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


def main(stock,back_gap,fwd_gap,rendite):

    enddatum = RSI_strat_SETUP.enddatum
    startdatum = RSI_strat_SETUP.startdatum
    roll_window = RSI_strat_SETUP.roll_window


    mypath = RSI_strat_SETUP.mypath

    #onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and not f.startswith("_")) ]

    b=[]
    c=[]
    figs=[]
    counter  = 0

    ###########################
    ##########################
    time_gap_fwd = fwd_gap
    time_gap_back = back_gap
    rsi_threshold =75
    return_threshold = rendite
    my_bins = 50
    ########################
    ########################

 

    ###########################
    ##########################
    # time_gap_fwd = 10
    # time_gap_back = 12
    # rsi_threshold =75
    # return_threshold = -12
    # my_bins = 50
    ########################
    ########################

    for ticker in [stock,]:
    #for ticker in ["DIA.csv","XLE.csv","XLK.csv","XLP.csv","XLY.csv","XLV.csv"]:
    #for ticker in onlyfiles:

    #onlyfiles:
    # if ticker != RSI_strat_SETUP.benchmarkfile:
            df = pd.read_csv(mypath + ticker,sep=";",decimal=',',
                            parse_dates=True,
                            index_col=0)

            df=df.sort_index()

            df=df.truncate(before=startdatum)
            df=df.truncate(after=enddatum)

            df = RSI(df, 21)

            ### Formel geprueft und korrekt: rechnet df[t]= (f(t+delta t)-f(t))/f(t)
            ### als forwared looking return !
            #df["fwd_"+str(time_gap)]= 100*(-df["Close"].diff(-time_gap)/df["Close"])

            ### Formeln geprueft und korrekt:
            # ############################################################################## 
            # mit positivem time_gap in .diff(time_gap):  df[t]= (f(t)-f(t-time_gap))/f(t)
            # also Veränderung ggü. f(t)
            ### Also backward looking return !
            df["back_"+str(time_gap_back)]= 100*(df["Close"].diff(time_gap_back)/df["Close"])
            
            # mit negativem time_gap in .diff(time_gap):  df[t]= (f(t+time_gap)-f(t))/f(t)
            # also Veränderung ggü. f(t). Der klassische Differenzenequotient also !!!!
            ### Also forckward looking return
            df["fwd_"+str(time_gap_fwd)]= 100*(-df["Close"].diff(-time_gap_fwd)/df["Close"])
            #################################################################################
            #### RSi Test
            #rsi_given  = df["RSI_"][df["rsi_sma"] > rsi_threshold]

            

            #### Percent Test #####################################################################
            #####    Wenn in time_gap_back Tagen merh als eturn_threshold rendite, dann schreibe
            ####  die time_gap_fwd rendite nach "a"
            a = df["fwd_"+str(time_gap_fwd)][df["back_"+str(time_gap_back)] < return_threshold]

            a = a.dropna()
            bb = a[a>12]
            bb.plot(kind="bar",title="anz: "+str(len(bb))+" Gesamt:"+str(len(a)))



            b= a[a!=0].tolist()
            c = c + b
            figs.append(px.line(df["rsi_sma"],title= "figs: 1. fig show: " + ticker))
            counter+=1






    for i in range(counter):
        figs[i].show()


    ##### rsi Statistik:
    rsi = df["rsi_sma"].dropna()
    rsi_sma=rsi.to_list()
    rsi_sma_hist = np.histogram(rsi_sma, bins=my_bins)
    rsi_sma_hist_dist = scipy.stats.rv_histogram(rsi_sma_hist)

    ### 1. Histogramm
    rsi_sma_df = pd.DataFrame(rsi_sma)
    rsi_sma_df.plot.hist(bins=my_bins,title="RSI Histogramm")


    ##### renditen nach time_gap_back Zeitschrittten Statistik:
    renditen = df["back_"+str(time_gap_back)].dropna()
    renditen_1=renditen.to_list()
    renditen_1_df = pd.DataFrame(renditen_1)

    ### plotte das HIstogramm der Renditen auf zwei arten
    renditen_1_df.plot.hist(bins=my_bins,title="a) back_"+str(time_gap_back)+" Renditen")
    fig0=px.histogram(renditen_1_df, nbins=my_bins, title="b) back_"+str(time_gap_back)+" Renditen")
    fig0.show()


    #### bilde das Histogrammobjekt, um die Cummulierten Wahrscienlichkeiten zu berechen
    renditen_1_hist = np.histogram(renditen_1, bins=my_bins)
    renditen_1_hist_dist = scipy.stats.rv_histogram(renditen_1_hist)

    ### 3. Histogramm der Renditen nach "time_gap_back" Tagen



    rsi_von = int(min(rsi_sma)-1.0)
    rsi_bis = int(max(rsi_sma)+1.0)

    X = np.linspace(rsi_von, rsi_bis,my_bins)
 
    cu = pd.DataFrame(data=X)
    cu["rsi"]= [rsi_sma_hist_dist.cdf(X)[i] for i in range(my_bins)]
    cu = cu.set_index(0)
    cu.index.name = "rsi val."
    fig1 = px.line(cu,title= "fig1: RSI Distr.Com.: " + ticker)
    fig1.show()



    #### renditen Statistik bei gegebenen rsi  thresholds bzw. renditen absolut  !
    hist = np.histogram(c, bins=my_bins)
    hist_dist = scipy.stats.rv_histogram(hist)

    b_df = pd.DataFrame(c)


    ### 2. Histogramm
    positiv = len(b_df[b_df[0]>0])
    negativ = len(b_df[b_df[0]<=0])
    tit = "fig2: g/t:"+str(time_gap_back) +"/"+ str(rsi_threshold) + " p:"+str(positiv) + " n:" + str(negativ) + " negP:" + str(round(hist_dist.cdf(0),3)) + " posP:" + str(round(1-hist_dist.cdf(0),3)) + " mu:" +  str(round(hist_dist.mean(),2))
    ax = b_df.plot.hist(bins=my_bins,title="a)"+tit)
    fig2 = px.histogram(b_df, nbins=my_bins, title="b)" + tit)
    fig2.show()


    ###############################################################
    ## cummulative wahrschienlichkeiten anzeigen

    c_von = int(min(c)-1.0)
    c_bis = int(max(c)+1.0)

    X = np.linspace(c_von, c_bis,my_bins)

    cu = pd.DataFrame(data=X)
    cu["p"]= [hist_dist.cdf(X)[i] for i in range(my_bins)]
    cu = cu.set_index(0)
    cu.index.name = "return[%]"
    fig3 = px.line(cu, title="fig3: Commul.Wahrsch der Renditen NACH Ereignis")
    fig3.show()   


    cu["p"]= [renditen_1_hist_dist.cdf(X)[i] for i in range(my_bins)]
    fig4 = px.line(cu, title="fig4 Commul.Wahrsch der Renditen OHNE Ereignis")
    #fig4.write_html(RSI_strat_SETUP.output_path + "Cu.Prob.OHNE Incident.hml")
    fig4.show()


    RSI_strat_SETUP.figures_to_html([figs[0],fig0,fig1,fig2,fig3,fig4],RSI_strat_SETUP.output_path + "dashboard.html")




    """ fig2 = px.histogram(df["d_"+"int_val"+"d"], histnorm='probability density', nbins=bin_val)

    st.plotly_chart(fig2) """



if __name__ == "__main__":
        opts, args = getopt.getopt(sys.argv,"hs:b:f:r:",["back_gap=","fwd_gap=","rendite="])
        for opt, arg in opts:
            if opt == '-h':
                print('RSI_strat.py -b <back_gap> -f <fwd_gap> -r <rendite>')
                sys.exit()  
            elif opt =="-b":
                back_gap = arg
            elif opt =="-s":
                stock = arg    
            elif opt =="-f":
                fwd_gap = arg
            elif opt =="-r":
               rendite = arg        
       #main(stock,          back_gap, fwd_gap, rendite)
        main("holc_data.csv",  31   ,  31       , 1000)

