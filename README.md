# Introduction

Hello every one. The set of codes here will show an introduction to anyone interesting in phonocardiogram processing.

Please copy and share without doubt !!

## Description of files
In these codes we uses Pandas library to manipulate the data
* "loading_data_base.py" this code help us get the .wav files from their local folder "semi-automatically".
* "comparacion_normales.py" displays the DFT of 5 normal heart sounds and from each extract the vibratory spectrum
* "comparacion_patologi.py" displays the DFT of 5 pathological heart sounds and from each extract the vibratory spectrum
* "ppfunctions_1.py" contains personal processing functions


### Loading_data_base
The example shows how to get audio files inside the folder "training-a". The files corresponds to the PhysioNet/Computing in Cardiology Challenges: Classification of Normal/Abnormal Heart Sound Recordings[1]. Here we read 409 .wav files containing PCG recordings and we apply a feature extraction algorithm. This is very uselfull in order to do experiments with your data and check how does it look like after some processing. The result could be as shown next :

![screenshot from 2018-10-27 18-06-42](https://user-images.githubusercontent.com/15948497/47607059-34cec500-da13-11e8-825b-af12b5689e8a.png)
Figure 1.

### Comparacion - normales y patologi
This codes shows a basic method using Fourier’s analysis over phonocardiography signals to extract features from the named "Energy of PCG Vibratory Spectrum". According with reference [2] heart sounds and murmurs frequency ranges and energy distributions could be around  8 Hz to 2048 Hz and 0.1e-3 Dyne/cm2 to 10 Dyne/cm2 respectively. Base on this fact the heart sound total vibratory spectrum can be divided into 7 bands:
    
    1. 0-5Hz, 2. 5-25Hz; 3. 25-120Hz; 4. 120-240Hz; 5. 240-500Hz; 6. 500-1000Hz; 7. 1000-2000Hz

Each of this bands contains different vibrations that occurs during the heart cycle and its energy can be use as features in order to classify the sounds. The energy tha we calculate for each band is the discrete time signal energy defined as the signal square in [3].

In this case, the DFT of normal sounds (Fig. 2, left-side) show us that the main frequency's are below 150Hz, meanwhile the DFT of pathological sounds (Fig. 2, right-side) are mainly above 150Hz.

![image](https://user-images.githubusercontent.com/15948497/47231887-10953780-d3c6-11e8-937e-a19d5e22f498.png)
Figure 2.

This evidence could be numerically presented as the energy of the vibratory spectrum (Fig. 3). Then, this vectors could feed a machine learning algorithm in order to classify the signals

![image](https://user-images.githubusercontent.com/15948497/47232639-21df4380-d3c8-11e8-9747-3075c1053d99.png)
Figure 3.


## References
[1] Liu, Chengyu & Springer, David & Li, Qiao & Moody, Benjamin & Abad Juan, Ricardo & J Chorro, Francisco & Castells, Francisco & Roig, Jos & Silva, Ikaro & E W Johnson, Alistair & Syed, Zeeshan & Schmidt, Samuel & Papadaniil, Chrysa & Hadjileontiadis, Leontios & Naseri, Hosein & Moukadem, Ali & Dieterlen, Alain & Brandt, Christian & Tang, Hong & D Clifford, Gari. (2016). An open access database for the evaluation of heart sound algorithms. Physiological Measurement. 37. 2181-2213. 10.1088/0967-3334/37/12/2181. 

[2]. Abbas, Abbas K. (Abbas Khudair), Bassam, Rasha and Morgan & Claypool Publishers. Phonocardiography signal processing. Morgan & Claypool Publishers, San Rafael, Calif, 2009.

[3] Hamed Beyramienanlou and Nasser Lotfivand, “Shannon’s Energy Based Algorithm in ECG Signal Processing,” Computational and Mathematical Methods in Medicine, vol. 2017, Article ID 8081361, 16 pages, 2017. https://doi.org/10.1155/2017/8081361.
