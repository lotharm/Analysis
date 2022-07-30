# -*- coding: utf-8 -*-

#%%

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







def main(stock,bwd_gap,fwd_gap,threshold_bwd, threshold_fwd, condi):

    enddatum = RSI_strat_SETUP.enddatum
    startdatum = RSI_strat_SETUP.startdatum
    roll_window = RSI_strat_SETUP.roll_window


    mypath = RSI_strat_SETUP.mypath +"//"

    #onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and not f.startswith("_")) ]

    b=[]
    c=[]
    figs=[]
    counter  = 0

    ###########################
    ##########################
    time_gap_fwd = fwd_gap
    time_gap_back = bwd_gap
    rsi_threshold =75
    return_threshold = threshold_bwd
    my_bins = 50
    ########################
    ########################


    for ticker in [stock,]:
   
            df = pd.read_csv(mypath + ticker,sep=";",decimal=',',
                            parse_dates=True,
                            index_col=0)

            df=df.sort_index()
           
            
            df=df.truncate(before=startdatum)
            df=df.truncate(after=enddatum)

            fig = px.line(df, x=df.index, y="Close")
            fig.show()

            df = RSI(df, 21)


            ### Formeln geprueft und korrekt:
            # ############################################################################## 
            # mit positivem time_gap in .diff(time_gap):  df[t]= (f(t)-f(t-time_gap))/f(t)
            # also Veränderung ggü. f(t)
            ### Also backward looking return !
            df["back_"+str(bwd_gap)]= 100*(df["Close"].diff(bwd_gap)/df["Close"])
            
            # mit negativem time_gap in .diff(time_gap):  df[t]= (f(t+time_gap)-f(t))/f(t)
            # also Veränderung ggü. f(t). Der klassische Differenzenequotient also !!!!
            ### Also forckward looking return      
            df["fwd_"+str(fwd_gap)]= 100*(-df["Close"].diff(-fwd_gap)/df["Close"])
            #################################################################################
            #### RSi Test
            #rsi_given  = df["RSI_"][df["rsi_sma"] > rsi_threshold]

            

            #### Percent Test #####################################################################
            #####    Wenn in time_gap_back Tagen merh als "return_threshold" rendite, dann schreibe
            ####  die  darauf folgende time_gap_fwd rendite nach "a"

            ####   select where backward threshold has been passed
            a = df["fwd_"+str(fwd_gap)][df["back_"+str(bwd_gap)] <= threshold_bwd]

            a_with_threshold_backward = a.dropna()
            
            ####   select where forward threshold has been passed AS WELL :
            a_with_threshold_backward_forward =a_with_threshold_backward[a_with_threshold_backward>=threshold_fwd]
            
            #a_with_threshold_backward_forward.plot(kind="bar",title="P(fwd_thres AND bwd_thresH): N_bwd_&_fwd="+str(len(a_with_threshold_backward_forward))+" N_bwd:"+str(len(a)))
            fig00 = px.bar(a_with_threshold_backward_forward, title = "Terminecluster: bwd AND fwd Thresholds erfuellt ")
            figupdate_nonHist(fig00,"Datum","Rendite nach "+ str(fwd_gap)+ " Bars")

            a_with_threshold_backward_forward_list= a_with_threshold_backward_forward[a_with_threshold_backward_forward!=0].tolist()

            figs.append(px.line(df["rsi_sma"],title= "figs: 1. fig show: " + ticker))
            counter+=1


    for i in range(counter):
        figupdate(figs[i])
        


    ##### rsi Statistik:
    rsi = df["rsi_sma"].dropna()
    rsi_sma=rsi.to_list()
    rsi_sma_hist = np.histogram(rsi_sma, bins=my_bins)
    rsi_sma_hist_dist = scipy.stats.rv_histogram(rsi_sma_hist)

    ### 1. Histogramm
    rsi_sma_df = pd.DataFrame(rsi_sma)
    #rsi_sma_df.plot.hist(bins=my_bins,title="RSI Histogramm")
    tit = "RSI Historgram"
    figRSIHist = px.histogram(rsi_sma_df, nbins=my_bins, title=tit)
    figupdate(figRSIHist)
    
    rsi_von = int(min(rsi_sma)-1.0)
    rsi_bis = int(max(rsi_sma)+1.0)

    X = np.linspace(rsi_von, rsi_bis,my_bins)
 
    cu = pd.DataFrame(data=X)
    cu["rsi"]= [rsi_sma_hist_dist.cdf(X)[i] for i in range(my_bins)]
    cu = cu.set_index(0)
    cu.index.name = "RSI"
    fig1 = px.line(cu,title= "Cum. Distr. RSI: " + ticker)
    figupdate(fig1)


    ##### renditen nach time_gap_back Zeitschrittten Statistik:
    backward = df["back_"+str(bwd_gap)].dropna()
    backward_list=backward.to_list()
    backward_df = pd.DataFrame(backward_list)

    ### plotte das HIstogramm der backward auf zwei arten
    #backward_df.plot.hist(bins=my_bins,title="a) back_"+str(bwd_gap)+" backward")
    fig0=px.histogram(backward_df, nbins=my_bins, title="Histogramm bkwd, dt = -"+str(bwd_gap)+" bars. Ohne Rendite Bedingung !")
    figupdate(fig0)


    #### bilde das Histogrammobjekt, um die Cummulierten Wahrscienlichkeiten zu berechen
    backward_hist = np.histogram(backward, bins=my_bins)
    backward_hist_dist = scipy.stats.rv_histogram(backward_hist)

    ### 3. Histogramm der Renditen nach "time_gap_back" Tagen





    #### renditen Statistik bei gegebenen rsi  thresholds bzw. renditen absolut  !
    a_with_threshold_backward_forward_hist = np.histogram(a_with_threshold_backward_forward_list, bins=my_bins)
    a_with_threshold_backward_forward_hist_dist = scipy.stats.rv_histogram(a_with_threshold_backward_forward_hist)

    


    ### 2. Histogramm
    b_df = pd.DataFrame(a_with_threshold_backward_forward_list)
    positiv = len(b_df[b_df[0]>0])
    negativ = len(b_df[b_df[0]<0])
    tite = "Cond. Cum.Prob. der Renditen bkwd= -"+ str(bwd_gap)+" bars,  fwd = "+ str(fwd_gap)+ " bars "
    tit = "Histogram: "+ tite +"<br>" + "#pos. renditen: "+str(positiv) + " #neg. renditen: " + str(negativ) + "<br>"+ "negP: " + str(round(a_with_threshold_backward_forward_hist_dist.cdf(0),3)) + " posP: " + str(round(1-a_with_threshold_backward_forward_hist_dist.cdf(0),3)) + " <r>: " +  str(round(a_with_threshold_backward_forward_hist_dist.mean(),2)) + "<br>" + condi
    #ax = b_df.plot.hist(bins=my_bins,title="a)"+tit)
    fig2 = px.histogram(b_df, nbins=my_bins, title=tit)
    figupdate(fig2)


    ###############################################################
    ## cummulative wahrschienlichkeiten anzeigen

    c_von = int(min( a_with_threshold_backward_forward_list)-1.0)
    c_bis = int(max( a_with_threshold_backward_forward_list)+1.0)

    X = np.linspace(c_von, c_bis,my_bins)

    cu = pd.DataFrame(data=X)
    cu["p"]= [a_with_threshold_backward_forward_hist_dist.cdf(X)[i] for i in range(my_bins)]
    cu = cu.set_index(0)
    cu.index.name = "return[%]"
    fig3 = px.line(cu, title="Cond. Cum.Prob. der Renditen bkwd= -"+ str(bwd_gap)+" bars,  fwd = "+ str(fwd_gap)+" bars" + "<br>" + condi)
    figupdate(fig3)

    

    a_von = int(min( backward_list)-1.0)
    a_bis = int(max( backward_list)+1.0)

    Y = np.linspace(a_von, a_bis,my_bins)
    
    cu = pd.DataFrame(data=Y)
    cu["p"]= [backward_hist_dist.cdf(Y)[i] for i in range(my_bins)]
    cu = cu.set_index(0)
    cu.index.name = "return[%]"
    fig4 = px.line(cu, title="Cum.Prob. der Renditen bkwd, mit dt= -"+ str(bwd_gap)+" bars" + "<br>" +" ohne Bedingung an Renditen")
    figupdate_nonHist(fig4)
    #fig4.write_html(RSI_strat_SETUP.output_path + "Cu.Prob.OHNE Incident.hml")
    


    RSI_strat_SETUP.figures_to_html([fig, figs[0],figRSIHist,fig0,fig1,fig2,fig3,fig4,fig00],RSI_strat_SETUP.output_path + "//"+dashboard.html")




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
       #main(stock,          back_gap, fwd_gap, threshold rendite backward, threshold renidte forward,comment )
        main("holc_data.csv",  1   ,  22       , 1000 , 10, "bel. % nach 1 Bars. Wie sieht es nach 22 Bars nach vorne aus ?")


# %%
