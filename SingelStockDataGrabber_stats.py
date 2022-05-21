# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:21:14 2021

@author: Schroeder
"""
#https://www.dividendenadel.de/indexmonitor-maerz-2021/
#zykliker: Chemie , rohstoofe (spaet im zyklus)Bauträger, Maschiennebau, REITS, Banken, Versicheurngen, Autobauer, ReisenHotels, Kreuztfahreten
#Antizyklishc/Defneisv: telekom, nestle, Metro,

import my_setup

path = my_setup.path

######################################################
stock = ["LIN",]
#####################################################
keyword = "macro"

Universe = {'quellpfad':[path],
            'quelldatei': [keyword], 
            'plotdir':[path+my_setup.specific_datapath],
            'resdir':[path+my_setup.specific_datapath],
            'vamsdir':[path+my_setup.specific_datapath],
            'tmpdatadir':[path+my_setup.specific_datapath],
            } 

enddatum=my_setup.enddatum 
Anzahl = 0 

periode = "15y"
ll=600
