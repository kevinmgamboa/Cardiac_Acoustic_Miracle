# -*- coding: utf-8 -*-
"""
Matrix de caracteristicas de 5 registros NO PATOLOGICOS (normales)
Created on Thu May  3 07:09:12 2018

@author: Kevin Machado G.
Ref:
    [1] https://www.datacamp.com/community/blog/python-pandas-cheat-sheet
"""
import matplotlib.pyplot as plt

import numpy as np
# Data manipulation
import pandas as pd
# Own Library
import ppfunctions_1 as ppf

import os 
import scipy.io.wavfile as wf

# -----------------------------------------------------------------------------
print('importing all training-a data base')
# Looking for heart sounds data absolute path
l1=os.path.abspath('Data Base HS\\non_pathologic\\1.1.wav')
l1=l1.replace('\\','/')
l2=os.path.abspath('Data Base HS\\non_pathologic\\1.2.wav')
l2=l2.replace('\\','/')
l3=os.path.abspath('Data Base HS\\non_pathologic\\1.3.wav')
l3=l3.replace('\\','/')
l4=os.path.abspath('Data Base HS\\non_pathologic\\1.4.wav')
l4=l4.replace('\\','/')
l5=os.path.abspath('Data Base HS\\non_pathologic\\1.5.wav')
l5=l5.replace('\\','/')

# reading file
Fs1, data1 = wf.read(l1)
Fs2, data2 = wf.read(l2)
Fs3, data3 = wf.read(l3)
Fs4, data4 = wf.read(l4)
Fs5, data5 = wf.read(l5)

# Clear paths
del l1, l2, l3, l4, l5

#

duration1 = 1/Fs1*np.size(data1)
duration2 = 1/Fs2*np.size(data2)
duration3 = 1/Fs3*np.size(data3)
duration4 = 1/Fs4*np.size(data4)
duration5 = 1/Fs5*np.size(data5)

vt1 = np.linspace(0,duration1,np.size(data1)) # Vector time 1
vt2 = np.linspace(0,duration2,np.size(data2)) # Vector time 3
vt3 = np.linspace(0,duration3,np.size(data3)) # Vector time 5
vt4 = np.linspace(0,duration4,np.size(data4)) # Vector time 7
vt5 = np.linspace(0,duration5,np.size(data5)) # Vector time 9

data1 = ppf.vec_nor(data1)
data2 = ppf.vec_nor(data2)
data3 = ppf.vec_nor(data3)
data4 = ppf.vec_nor(data4)
data5 = ppf.vec_nor(data5)

pcgFFT1, vTfft1 = ppf.fft_k_N(data1, Fs1, 2000)
pcgFFT2, vTfft2 = ppf.fft_k_N(data2, Fs2, 2000)
pcgFFT3, vTfft3 = ppf.fft_k_N(data3, Fs3, 2000)
pcgFFT4, vTfft4 = ppf.fft_k_N(data4, Fs4, 2000)
pcgFFT5, vTfft5 = ppf.fft_k_N(data5, Fs5, 2000)

idx1 = (np.abs(vt1-5)).argmin()                # Find the index of time vector in 10 seconds
idx2 = (np.abs(vt2-5)).argmin()                # Find the index of time vector in 10 seconds
idx3 = (np.abs(vt3-5)).argmin()                # Find the index of time vector in 10 seconds
idx4 = (np.abs(vt4-5)).argmin()                # Find the index of time vector in 10 seconds
idx5 = (np.abs(vt5-5)).argmin()                # Find the index of time vector in 10 seconds
#
# -----------------------------------------------------------------------------
# Energy of vibratory signal spectrum
# 1. 0-5Hz, 2. 5-25Hz; 3. 25-120Hz; 4. 120-240Hz; 5. 240-500Hz; 6. 500-1000Hz; 7. 1000-2000Hz
EVS1 = ppf.E_VS(pcgFFT1, vTfft1, 'percentage')
EVS2 = ppf.E_VS(pcgFFT2, vTfft2, 'percentage')
EVS3 = ppf.E_VS(pcgFFT3, vTfft3, 'percentage')
EVS4 = ppf.E_VS(pcgFFT4, vTfft4, 'percentage')
EVS5 = ppf.E_VS(pcgFFT5, vTfft5, 'percentage')

# Showing Results in Pandas
data = {'N1': np.round(EVS1),'N2': np.round(EVS2), 'N3': np.round(EVS3), 'N4': np.round(EVS4), 'N5': np.round(EVS5)}
print('Registros Patologicos')
df=pd.DataFrame(data,index=['Total (%)','0-5Hz','5-25Hz','25-120Hz','120-240Hz','240-500Hz','500-1kHz','1k-2kHz'],columns=['P1','P2', 'P3', 'P4', 'P5'])
print (df)

# df.to_excel('pato_5.xlsx') # writing to excel

plt.figure(1)

plt.subplot(5,1,1)
plt.title('Transformada de Fourier')
plt.plot(vTfft1, pcgFFT1,'r')

plt.subplot(5,1,2)
plt.plot(vTfft2, pcgFFT2,'r')

plt.subplot(5,1,3)
plt.plot(vTfft3, pcgFFT3,'r')

plt.subplot(5,1,4)
plt.plot(vTfft4, pcgFFT4,'r')

plt.subplot(5,1,5)
plt.plot(vTfft5, pcgFFT5,'r')