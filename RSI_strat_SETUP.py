
#mypath = "C:/Users/r889990/OneDrive - innogy SE/PYTHON/DATA/SPYVIX/"
#mypath = "C:/Users/_schr/OneDrive/Trading/ETFS/RES_USSectors/Data/"
#mypath="C:/Michael/ETFS/RES_USSectors/Data/"

import sys

#sys.path.insert(0, 'C:/Users/Schroeder/OneDrive/Trading/ETFS/python')
#sys.path.insert(0, 'C:/Users/_schr/OneDrive/Trading/ETFS/python')
import my_setup



mypath = my_setup.path + my_setup.specific_datapath 
output_path = my_setup.output_path

startdatum = "2011-10-01"
enddatum = "2022-05-30"

#https://www.tradingview.com/x/RHiQkrp0/




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

#desktoppfad = 'C:/Users/r889990/Desktop/'
# Surface:
#benchmarkfile = "SPY.csv"
#Desktop
benchmarkfile = "SPY.csv"

def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")