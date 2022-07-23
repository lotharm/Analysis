
#mypath = "C:/Users/r889990/OneDrive - innogy SE/PYTHON/DATA/SPYVIX/"
#mypath = "C:/Users/_schr/OneDrive/Trading/ETFS/RES_USSectors/Data/"
#mypath="C:/Michael/ETFS/RES_USSectors/Data/"

import os
import sys


#sys.path.insert(0, 'C:/Users/Schroeder/OneDrive/Trading/ETFS/python')
#sys.path.insert(0, 'C:/Users/_schr/OneDrive/Trading/ETFS/python')

###### Insert the following three lines to make any import lib in he project dir setup visible to an other
###### Directory in the project setup
currentdir = os.path.abspath('')
parentdir = os.path.realpath(os.path.join(currentdir, '..'))
sys.path.insert(0, parentdir) 
#############################################################

from MomentumScreening import my_setup
###
### The whole /ETFS/ Tree has to be located on ame level as repository !
pfad = os.path.realpath(os.path.join(parentdir, '..'))
pfad = os.path.realpath(os.path.join(pfad, 'ETFS'))
##########################################################################
mypath = os.path.realpath(os.path.join(pfad, my_setup.specific_datapath)) 
output_path = mypath

startdatum = "2011-10-01"
enddatum = "2022-05-30"

#https://www.tradingview.com/x/RHiQkrp0/

print("hier: ",mypath )

quantil = 0.1
riskquantil = 0.00005


SingleEMAperiod = 50
BenchmarkEMAperiod = 100
#das Fenster um die Regression zu rechnen:
roll_window = 60

Num_of_positions = 10
CutOff_positions = 20

#desktoppfad = 'C:/Users/_schr/Desktop/'
desktoppfad = output_path


benchmarkfile = "SPY.csv"

def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")