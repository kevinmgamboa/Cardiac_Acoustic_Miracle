# -*- coding: utf-8 -*-
"""
Personal Processing Functions 1 (ppfunctions_1)
Module file containing Python definitions and statements
Created on Mon Apr  9 11:48:37 2018
Modify: 02/11/2018
@author: Kevin MAchado
"""

# Libraries
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, butter
from scipy.fftpack import fft
from scipy import signal
#import peakutils                                # Librery to help in peak detection

# Functions 
# Normal energy = np.sum(pcgFFT1**2)
# Normalized average Shannon Energy = sum((l**2)*np.log(l**2))/l.shape[0]
# Third Order Shannon Energy = sum((l**3)*np.log(l**3))/l.shape[0]

def vec_nor(x):
    """
    Normalize the amplitude of a vector from -1 to 1
    """
    lenght=np.size(x)				# Get the length of the vector	
    xMax=max(x);					   # Get the maximun value of the vector
    nVec=np.zeros(lenght);		   # Initializate derivate vector
    nVec = np.divide(x, xMax)
    nVec=nVec-np.mean(nVec);
    nVec=np.divide(nVec,np.max(nVec));
        
    return nVec

def derivate (x):
    """
    Derivate of an input signal as y[n]= x[n+1]- x[n-1] 
    """
    lenght=x.shape[0]				# Get the length of the vector
    y=np.zeros(lenght);				# Initializate derivate vector
    for i in range(lenght-1):
        y[i]=x[i-1]-x[i];		
    return y
# -----------------------------------------------------------------------------
# Energy
# -----------------------------------------------------------------------------
def Energy_value (x):
    """
    Energy of an input signal  
    """
    y = np.sum(x**2)
    return y

def shannonE_value (x):
    """
    Shannon energy of an input signal  
    """
    y = sum((x**2)*np.log(x**2))/x.shape[0]
    return y

def shannonE_vector (x):
    """
    Shannon energy of an input signal  
    """
    mu = -(x**2)*np.log(x**2)/x.shape[0]
    y = -(((x**2)*np.log(x**2)) - mu)/np.std(x)
    return y

def E_VS_LF (pcgFFT1, vTfft1, on):
    """
    Energy of PCG Vibratory Spectrum in low Frequencies (E_VS_LF)
    (frequency components, frequency value vector, on = on percentage or not)
    According with [1] The total vibratory spectrum can be divided into 7 bands.
    This is a modification of this 7 bands
    1. 0-5Hz, 2. 5-25Hz; 3. 25-60Hz; 4. 60-120Hz; 5. 120-400Hz

The PCG signal producess vibrations in the spectrum between 0-2k Hz. 
[1] Abbas, Abbas K. (Abbas Khudair), Bassam, Rasha and Morgan & Claypool Publishers Phonocardiography signal processing. Morgan & Claypool Publishers, San Rafael, Calif, 2009.
    """
    c1 = (np.abs(vTfft1-5)).argmin()
    c2 = (np.abs(vTfft1-25)).argmin()
    c3 = (np.abs(vTfft1-120)).argmin()
    c4 = (np.abs(vTfft1-240)).argmin()
    c5 = (np.abs(vTfft1-500)).argmin()
    
    # All vector energy
    xAll = Energy_value(pcgFFT1)

    # Procesando de 0.01-5 Hz
    pcgFFT_F1 = pcgFFT1[0:c1]
    x1 = Energy_value(pcgFFT_F1)
    
    # Procesando de 5-25 Hz
    pcgFFT_F2 = pcgFFT1[c1:c2]
    x2 = Energy_value(pcgFFT_F2)
    
    # Procesando de 25-120 Hz
    pcgFFT_F3 = pcgFFT1[c2:c3]
    x3 = Energy_value(pcgFFT_F3)
    
    # Procesando de 120-240 Hz
    pcgFFT_F4 = pcgFFT1[c3:c4]
    x4 = Energy_value(pcgFFT_F4)
    
    # Procesando de 240-500 Hz
    pcgFFT_F5 = pcgFFT1[c4:c5]
    x5 = Energy_value(pcgFFT_F5)
    
    x = np.array([xAll, x1, x2, x3, x4, x5])
    
    if (on == 'percentage'):
        x = 100*(x/x[0])

    return x

def E_VS (pcgFFT1, vTfft1, on):
    """
    Energy of PCG Vibratory Spectrum
    (frequency components, frequency value vector, on = on percentage or not)
    According with [1] The total vibratory spectrum can be divided into 7 bands:
    1. 0-5Hz, 2. 5-25Hz; 3. 25-120Hz; 4. 120-240Hz; 5. 240-500Hz; 6. 500-1000Hz; 7. 1000-2000Hz

The PCG signal producess vibrations in the spectrum between 0-2k Hz. 
[1] Abbas, Abbas K. (Abbas Khudair), Bassam, Rasha and Morgan & Claypool Publishers Phonocardiography signal processing. Morgan & Claypool Publishers, San Rafael, Calif, 2009.
    """
    c1 = (np.abs(vTfft1-5)).argmin()
    c2 = (np.abs(vTfft1-25)).argmin()
    c3 = (np.abs(vTfft1-120)).argmin()
    c4 = (np.abs(vTfft1-240)).argmin()
    c5 = (np.abs(vTfft1-500)).argmin()
    c6 = (np.abs(vTfft1-1000)).argmin()
    c7 = (np.abs(vTfft1-2000)).argmin()
    
    # All vector energy
    xAll = Energy_value(pcgFFT1)

    # Procesando de 0.01-5 Hz
    pcgFFT_F1 = pcgFFT1[0:c1]
    x1 = Energy_value(pcgFFT_F1)
    
    # Procesando de 5-25 Hz
    pcgFFT_F2 = pcgFFT1[c1:c2]
    x2 = Energy_value(pcgFFT_F2)
    
    # Procesando de 25-120 Hz
    pcgFFT_F3 = pcgFFT1[c2:c3]
    x3 = Energy_value(pcgFFT_F3)
    
    # Procesando de 120-240 Hz
    pcgFFT_F4 = pcgFFT1[c3:c4]
    x4 = Energy_value(pcgFFT_F4)
    
    # Procesando de 240-500 Hz
    pcgFFT_F5 = pcgFFT1[c4:c5]
    x5 = Energy_value(pcgFFT_F5)
    
    # Procesando de 500-1000 Hz
    pcgFFT_F6 = pcgFFT1[c5:c6]
    x6 = Energy_value(pcgFFT_F6)
    
    # Procesando de 1000-2000 Hz
    pcgFFT_F7 = pcgFFT1[c6:c7]
    x7 = Energy_value(pcgFFT_F7)
    
    x = np.array([xAll, x1, x2, x3, x4, x5, x6, x7])
    
    if (on == 'percentage'):
        x = 100*(x/x[0])

    return x
#-------------------------------------------
def features_1(data_P1, fs):
    
    # Defining the Vibratory Frequency Bands
    bVec = [0.01, 5, 25, 120, 240, 500, 999]
    # Initializing Vectors
    band_matrix = np.zeros((len(bVec),len(data_P1))) # Band Matrix
    power_matrix = np.zeros((len(bVec),int(1+(len(data_P1)/2)))) # Band Matrix
    freqs_matrix = np.zeros((len(bVec),int(1+(len(data_P1)/2)))) # Band Matrix
    SePCG = np.zeros(len(bVec)-1)                            # Shannon Energy in each Band
    SePWR = np.zeros(len(bVec)-1)                            # Shannon Energy in each Band
      
    
    for i in range(len(bVec)-1):
        band_matrix[i,:] = butter_bp_fil(data_P1, bVec[i], bVec[i+1], fs)       
        freqs_matrix[i,:], power_matrix[i,:] = signal.periodogram(band_matrix[i,:], fs, scaling = 'density')
        SePCG[i] = Energy_value(band_matrix[i,:])
        SePWR[i] = Energy_value(power_matrix[i,:])
     
    SePWR = 1*np.log10(SePWR)
    SePCG = 1*np.log10(SePCG)
    
    return SePWR, SePCG

# -----------------------------------------------------------------------------
# Filter Processes
# -----------------------------------------------------------------------------

def butter_bp_coe(lowcut, highcut, fs, order=1):
    """
    Butterworth passband filter coefficients b and a
    Ref: 
    [1] https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    [2] https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bp_fil(data, lowcut, highcut, fs, order=1):
    """
    Butterworth passband filter
    Ref: 
    [1] https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    [2] https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
    """
    b, a = butter_bp_coe(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def Fpass(X,lp):
    """
    Fpass is the function to pass the coefficients of a filter trough a signal'
    """
    llp=np.size(lp)	  	        # Get the length of the lowpass vector		

    x=np.convolve(X,lp);		        # Disrete convolution 
    x=x[int(llp/2):-int(llp/2)];
    x=x-(np.mean(x));
    x=x/np.max(x);
    
    y=vec_nor(x);				# Vector Normalizing
        
    return y

def FpassBand(X,hp,lp):
    """
    FpassBand is the function that develop a pass band filter of the signal 'x' through the
    discrete convolution of this 'x' first with the coeficients of a High Pass Filter 'hp' and then
    with the discrete convolution of this result with a Low Pass Filter 'lp'
    """
    llp=np.shape(lp)	  	        # Get the length of the lowpass vector		
    llp=llp[0];				       # Get the value of the length
    lhp=np.shape(hp)			    # Get the length of the highpass vector		
    lhp=lhp[0];				       # Get the value of the length	

    x=np.convolve(X,lp);		        # Disrete convolution 
    x=x[int(llp/2):-int(llp/2)];
    x=x-(np.mean(x));
    x=x/np.max(x);
	
    y=np.convolve(x,hp);			# Disrete onvolution
    y=y[int(lhp/2):-int(lhp/2)];
    y=y-np.mean(y);
    y=y/np.max(y);

    x=np.convolve(y,lp);		        # Disrete convolution 
    x=x[int(llp/2):-int(llp/2)];
    x=x-(np.mean(x));
    x=x/np.max(x);
	
    y=np.convolve(x,hp);			# Disrete onvolution
    y=y[int(lhp/2):-int(lhp/2)];
    y=y-np.mean(y);
    y=y/np.max(y);
        
    y=vec_nor(y);				# Vector Normalizing
        
    return y

def FpassBand_1(X,Fs, f1, f2):
    """
    Ref: http://scipy-cookbook.readthedocs.io/items/FIRFilter.html
    http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.signal.firwin.html
    FpassBand_1 is a function to develop a passband filter using 'firwin'
    and 'lfilter' functions from the "Scipy" library
    """

    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2.0
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # 
    taps = firwin(100, [f1, f2], pass_zero=False)
#    taps = firwin(N, L_cutoff_hz/nyq_rate, window=('kaiser', beta))
#    taps_2 = firwin(N, H_cutoff_hz/nyq_rate, pass_zero=True)
    # Use lfilter to filter x with the FIR filter.
    X_pb= lfilter(taps, 1.0, X)
   # X_pb= lfilter(taps_2, 1.0, X_l)
    
    return X_pb[N-1:]

def FpassBand_2(X, f1, f2, Fs):
    
    Y = FhighPass(X, Fs, f1)
    
    Y = FlowPass(X, Fs, f2)
    
    return Y

def FhighPass(X, Fs, H_cutoff_hz):
    """
    Ref: http://scipy-cookbook.readthedocs.io/items/FIRFilter.html
    http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.signal.firwin.html
    FhighPass is a function to develop a highpass filter using 'firwin'
    and 'lfilter' functions from the "Scipy" library
    """
    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2.0
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps_2 = firwin(N, H_cutoff_hz/nyq_rate, pass_zero=False)
    # Use lfilter to filter x with the FIR filter.
    X_h= lfilter(taps_2, 1.0, X)
    
    return X_h[N-1:]
    
def FlowPass(X, Fs, L_cutoff_hz):
    """
    Ref: http://scipy-cookbook.readthedocs.io/items/FIRFilter.html
    http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.signal.firwin.html
    FlowPass is a function to develop a lowpass filter using 'firwin'
    and 'lfilter' functions from the "Scipy" library
    """
    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2.0
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, L_cutoff_hz/nyq_rate, window=('kaiser', beta))
    # Use lfilter to filter x with the FIR filter.
    X_l= lfilter(taps, 1.0, X)
    
    return X_l[N-1:]
# -----------------------------------------------------------------------------
# Peak Detection
# -----------------------------------------------------------------------------
def PDP(Xf, samplerate):
    """
    Peak Detection Process
    """
    timeCut = samplerate*0.25                       # time to count another pulse
    vCorte = 0.6
    
    Xf = vec_nor(Xf)
    dX=derivate(Xf);				                      # Derivate of the signal
    dX=vec_nor(dX);			                         # Vector Normalizing
    
    size=np.shape(Xf)				                  # Rank or dimension of the array
    fil=size[0];					                     # Number of rows
 
    positive=np.zeros((1,fil+1));                   # Initializating Vector 
    positive=positive[0];                           # Getting the Vector

    points=np.zeros((1,fil));                       # Initializating the Peak Points Vector
    points=points[0];                               # Getting the point vector

    points1=np.zeros((1,fil));                      # Initializating the Peak Points Vector
    points1=points1[0];                             # Getting the point vector
    
    vCorte = 0.6
       
    '''
    FIRST! having the positives values of the slope as 1
    And the negative values of the slope as 0
    '''
    for i in range(0,fil):
        if dX[i]>0:
            positive[i]=1;
        else:
            positive[i]=0;
    '''
    SECOND! a peak will be found when the ith value is equal to 1 &&
    the ith+1 is equal to 0
    '''
    for i in range(0,fil):
        if (positive[i]==1 and positive[i+1]==0):
            points[i]=Xf[i];
        else:
            points[i]=0;
    '''
    THIRD! Define a minimun Peak Height
    '''
    p=0;
    for i in range(0,fil):
        if (points[i] > vCorte and p==0):
            p=i
            points1[i]=Xf[i]
            
        else:
            points1[i]=0;
            if (p+timeCut < i):
                p=0
                    
    return points1
# -----------------------------------------------------------------------------
# Fast Fourier Transform
# -----------------------------------------------------------------------------
def fft_k(data, samplerate, showFrequency):
    '''
    Fast Fourier Transform
    Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    '''
    pcgFFT = fft(data)                                                    # FFT Full Vector 'k' coefficients
    short_pcgFFT = 2.0/np.size(data) * np.abs(pcgFFT[0:np.size(data)//2]) # FFT positives values
    vTfft = np.linspace(0.0, 1.0/(2.0*(1/samplerate)), np.size(data)//2)  # Vector of frequencies (X-axes)
       
    idx = (np.abs(vTfft-showFrequency)).argmin()             # find the value closest to a value
    
    return short_pcgFFT[0:idx], vTfft[0:idx]

def fft_k_N(data, samplerate, showFrequency):
    '''
    Normalized Fast Fourier Transform
    Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    '''
    pcgFFT = fft(data)                                                    # FFT Full Vector
    short_pcgFFT = 2.0/np.size(data) * np.abs(pcgFFT[0:np.size(data)//2]) # FFT positives values
    vTfft = np.linspace(0.0, 1.0/(2.0*(1/samplerate)), np.size(data)//2)  # Vector of frequencies (X-axes)
       
    idx = (np.abs(vTfft-showFrequency)).argmin()             # find the value closest to a value
    
    return vec_nor(short_pcgFFT[0:idx]), vTfft[0:idx]
