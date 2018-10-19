# Introduction
This codes shows a basic method using Fourier’s analysis over phonocardiography signals to extract features from the named "Energy of PCG Vibratory Spectrum". According with reference [1] heart sounds and murmurs frequency ranges and energy distributions could be around  8 Hz to 2048 Hz and 0.1e-3 Dyne/cm2 to 10 Dyne/cm2 respectively. Base on this fact the heart sound total vibratory spectrum can be divided into 7 bands:
    
    1. 0-5Hz, 2. 5-25Hz; 3. 25-120Hz; 4. 120-240Hz; 5. 240-500Hz; 6. 500-1000Hz; 7. 1000-2000Hz

Each of this bands contains different vibrations that occurs during the heart cycle and its energy can be use as features in order to classify the sounds. The energy tha we calculate for each band is the discrete time signal energy defined as the signal square in [2].

In this case, the DFT of normal sounds (Fig. 1, left-side) show us that the main frequency's are below 150Hz, meanwhile the DFT of pathological sounds (Fig. 1, right-side) are mainly above 150Hz.

![image](https://user-images.githubusercontent.com/15948497/47231887-10953780-d3c6-11e8-937e-a19d5e22f498.png)
Figure. 1.

This evidence could be numerically presented as the energy of the vibratory spectrum (Fig. 2). Then, this vectors could feed a machine learning algorithm in order to classify the signals

![image](https://user-images.githubusercontent.com/15948497/47232639-21df4380-d3c8-11e8-9747-3075c1053d99.png)
Figure. 2.


## Description of files
In these codes we uses Pandas library to manipulate the data
* "comparacion_normales.py" displays the DFT of 5 normal heart sounds and from each extract the vibratory spectrum
* "comparacion_patologi.py" displays the DFT of 5 pathological heart sounds and from each extract the vibratory spectrum
* "ppfunctions_1.py" contains personal processing functions


## References
[1] Abbas, Abbas K. (Abbas Khudair), Bassam, Rasha and Morgan & Claypool Publishers. Phonocardiography signal processing. Morgan & Claypool Publishers, San Rafael, Calif, 2009.
[2] Hamed Beyramienanlou and Nasser Lotfivand, “Shannon’s Energy Based Algorithm in ECG Signal Processing,” Computational and Mathematical Methods in Medicine, vol. 2017, Article ID 8081361, 16 pages, 2017. https://doi.org/10.1155/2017/8081361.
