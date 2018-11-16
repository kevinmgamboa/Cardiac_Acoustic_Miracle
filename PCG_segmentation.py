"""
Segmenting & Timing
Created on Wed Nov 14 05:55:10 2018

@author: colossus
"""
import os
import numpy as np
#import wavio as wa
#import sounddevice as sd
import ppfunctions_1 as ppf
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#     Signal Loading Process
# -----------------------------------------------------------------------------
print('importing all training-a data base')
# Looking for heart sounds data absolute path 
l1=os.path.abspath('Data Base HS\\non_pathologic\\1.1.wav')
#l1=os.path.abspath('Data Base HS\\pathologic\\2.1.wav')
l1=l1.replace('\\','/')
l2=os.path.abspath('Data Base HS\\pathologic\\2.3.wav')
l2=l2.replace('\\','/')
l3=os.path.abspath('Data Base HS\\pathologic\\3.2.wav')
l3=l3.replace('\\','/')
l4=os.path.abspath('Data Base HS\\pathologic\\4.3.wav')
l4=l4.replace('\\','/')
l5=os.path.abspath('Data Base HS\\pathologic\\4.5.wav')
l5=l5.replace('\\','/')

# reading file
Fs1, data1 = wf.read(l1)
Fs2, data2 = wf.read(l2)
Fs3, data3 = wf.read(l3)
Fs4, data4 = wf.read(l4)
Fs5, data5 = wf.read(l5)

# Clear paths
del l1, l2, l3, l4, l5
# -----------------------------------------------------------------------------
#     Pre-Process: Signal basic information (SBI)
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
#     Pre-Process: Segmentationg and Timing
# -----------------------------------------------------------------------------
# Segmenting in an specific frequency band
pcgPeaks, peaks, allpeaks = ppf.PDP(data1, Fs1)

f_data1 = ppf.butter_bp_fil(data1, 50, 150, Fs1)
f_pcgPeaks, f_peaks, f_allpeaks = ppf.PDP(f_data1, Fs1)
# -----------------------------------------------------------------------------
# Time processing
dT = 0.4          # Diastole time in ms
timeV = []
timeV.append(0)
pointV = []
pointV.append(0)
segmV = np.zeros(len(data1))

for i in range(len(pcgPeaks)-1):
    if pcgPeaks[i]>0.5:
        timeV.append(i/Fs1)       # Gives the time when a peak get found
        pointV.append(i)       # Gives the time when a peak get found
        if (pointV[-1]/Fs1)-(pointV[-2]/Fs1)< dT:
            segmV[pointV[-2]:pointV[-1]] = 0.6  # Marc a diastolic segment
        else:
            segmV[pointV[-2]:pointV[-1]] = 0.4  # Marc a systolic segment

# -----------------------------------------------------------------------------
                         # Plotting Time Signal
# -----------------------------------------------------------------------------
plt.figure(1)
plt.title('Pathologic Heart sounds in time')

plt.subplot(3,1,1)
plt.plot(vt1, data1,'r', vt1, pcgPeaks, 'b')

plt.subplot(3,1,2)
plt.plot(vt1, data1,'r', vt1, peaks, 'b')

plt.subplot(3,1,3)
plt.plot(vt1, data1,'r', vt1, allpeaks, 'b')

plt.figure(2)
plt.title('Pathologic Heart sounds in time')

plt.subplot(3,1,1)
plt.plot(vt1, data1,'r', vt1, pcgPeaks, 'b')

plt.subplot(2,1,2)
plt.plot(vt1, data1,'r', vt1, segmV, 'b')

plt.show()
