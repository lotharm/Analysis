# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:21:14 2021

@author: Schroeder
"""
#https://www.dividendenadel.de/indexmonitor-maerz-2021/
#zykliker: Chemie , rohstoofe (spaet im zyklus)Bauträger, Maschiennebau, REITS, Banken, Versicheurngen, Autobauer, ReisenHotels, Kreuztfahreten
#Antizyklishc/Defneisv: telekom, nestle, Metro,

###### Insert the following three lines to make any import lib in he project dir setup visible to an other
###### Directory in the project setup
import os
import sys
currentdir = os.path.abspath('')
parentdir = os.path.realpath(os.path.join(currentdir, '..'))
sys.path.insert(0, parentdir) 
#############################################################

from MomentumScreening import my_setup
###
### The whole /ETFS/ Tree has to be located on ame level as repository !
pfad = os.path.realpath(os.path.join(parentdir, 'ETFS'))



######################################################
stock = ["OSG",]
#####################################################
keyword = "macro"

mypath = os.path.realpath(os.path.join(pfad, my_setup.specific_datapath)) 

output_path = mypath

keyword = "macro"

Universe = {'quellpfad':[pfad],
            'quelldatei': [keyword], 
            'plotdir':[mypath],
            'resdir':[mypath],
            'vamsdir':[mypath],
            'tmpdatadir':[mypath],
            } 


Anzahl = 0 

periode = "15y"
ll=600
