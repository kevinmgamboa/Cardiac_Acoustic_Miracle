# -*- coding: utf-8 -*-
"""
Loading Database
Created on Wed Jul 11 11:31:31 2018
@author: kevin machado
"""
#
import os
import pandas as pd
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Own Library
import ppfunctions_1 as ppf

# Looking for heart sounds data absolute path 
path = 'Data Base HS\\training\\training-a\\a0001.wav'
No_Files = 409                                    # Total number of files inside folder
matrix_path = []                                  # Initializing matrix to save all paths
# -----------------------------------------------------------------------------
# Reading the data reference with Pandas
ref = 'Data Base HS\\training\\training-a\\REFERENCE.csv'
ref = ref.replace('\\', '/')
ref = pd.read_csv(ref)
# -----------------------------------------------------------------------------
# Defining 3D plot environment
fig1 = plt.figure(1)
en2D = fig1.add_subplot(111)
en2D.set(xlabel='Energy', ylabel='PWR')
plt.style.use('dark_background')
plt.title('Feature Space in 2D')


fig2 = plt.figure(2)
en3D = fig2.add_subplot(111, projection='3d')
en3D.set(xlabel='Energy', ylabel='PWR', zlabel='K')
plt.title('Feature Space in 3D')
# Starting reading files
for i in range (1, No_Files+1):
    print (i)
    if i<=9:
        # Looking for heart sounds data absolute path
        l = path.replace("01", "0%i"%i)
        l = os.path.abspath(l)
        l = l.replace('\\','/')
        matrix_path.append(l)
        # ---------------------------------------------------------------------
        # Reading file
        Fs, data = wf.read(l)
        # ---------------------------------------------------------------------
        # Getting features ... Replace or add more feature extraccion algorithms
        PWR, SePCG =ppf.features_1(data,Fs)
        # ---------------------------------------------------------------------
        # Plotting features
        if ref.iloc[(i-1),1] == 1:
            en3D.scatter(SePCG,PWR, 1, c='r', marker='o')
            en2D.plot(SePCG,PWR,'ro')
        else:
            en3D.scatter(SePCG,PWR, -1, c='b', marker='s')
            en2D.plot(SePCG,PWR,'bs')
# -----------------------------------------------------------------------------            
    if i>=10 and i<=99:
        # Looking for heart sounds data absolute path
        l = path.replace("001", "0%i"%i)
        l = os.path.abspath(l)
        l = l.replace('\\','/')
        matrix_path.append(l)
        # ---------------------------------------------------------------------
        # Reading file
        Fs, data = wf.read(l)
        # ---------------------------------------------------------------------
        # Getting features ... Replace or add more feature extraccion algorithms
        PWR, SePCG =ppf.features_1(data,Fs)
        # ---------------------------------------------------------------------
        # Plotting features
        if ref.iloc[(i-1),1] == 1:
            en3D.scatter(SePCG,PWR, 1, c='r', marker='o')
            en2D.plot(SePCG,PWR,'ro')
        else:
            en3D.scatter(SePCG,PWR, -1, c='b', marker='s')
            en2D.plot(SePCG,PWR,'bs')
# -----------------------------------------------------------------------------        
    if i>=100 and i<=999:
        # Looking for heart sounds data absolute path
        l = path.replace("0001", "0%i"%i)
        l = os.path.abspath(l)
        l = l.replace('\\','/')
        matrix_path.append(l)
        # ---------------------------------------------------------------------
        # Reading file
        Fs, data = wf.read(l)
        # ---------------------------------------------------------------------
        # Getting features ... Replace or add more feature extraccion algorithms
        PWR, SePCG =ppf.features_1(data,Fs)
        # ---------------------------------------------------------------------
        # Plotting features
        if ref.iloc[(i-1),1] == 1:
            en3D.scatter(SePCG,PWR, 1, c='r', marker='o')
            en2D.plot(SePCG,PWR,'ro')
        else:
            en3D.scatter(SePCG,PWR, -1, c='b', marker='s')
            en2D.plot(SePCG,PWR,'bs')
# -----------------------------------------------------------------------------