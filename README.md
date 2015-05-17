# Jingju Arias MIR Analysis
python

## usage
Youtube video demo: http://youtu.be/YZMK5vJ_A8M

download this folder,  
go into this folder in your terminal,  
run demo.py

syllable marker file should be utf-8 encoding

download also font folder which contains the font used in plot

## modules required
numpy  
scipy  
matplotlib
cython
intonation (tonic)
pypeaks (tonic)

essentia:  
https://github.com/MTG/essentia

sms-tools:  
https://github.com/MTG/sms-tools  

intonation:
https://github.com/gopalkoduri/intonation

pypeaks:
https://github.com/gopalkoduri/pypeaks

1. if you meet "cython compile" problem, please delete build folder and utilFunctions_C.so file in sms-models directory, then re-compile utilFunctions according to method:  

go to the directory /sms-models/utilFunctions_C and type:

$ python compileModule.py build_ext --inplace 

