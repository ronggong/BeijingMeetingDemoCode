# Beijing Meeting Demo Code
python

## use
Youtube video demo: http://youtu.be/YZMK5vJ_A8M

download this folder,  
go into this folder in your terminal,  
run demo.py

## modules required
numpy  
scipy  
matplotlib
cython

essentia:  
https://github.com/MTG/essentia

sms-tools:  
https://github.com/MTG/sms-tools  

if you meet "cython compile" problem, please delete build folder and utilFunctions_C.so file in sms-models directory, then re-compile utilFunctions according to method:  

go to the directory software/models/utilFunctions_C and type:

$ python compileModule.py build_ext --inplace 
