import sys
import numpy as np

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def findMinDisFrameNum(t, vec_len, hop, fs):
    '''we find the minimum distance frame between t and a frame in vector,
    the principle is the distance decreases then increases'''
    oldDis = sys.float_info.max
    for p in range(1, vec_len):
        dis = abs(t - (p + 0.5) * hop / fs)
        if oldDis < dis:
            return p
        oldDis = dis
        
def meanValueRejectZero(vec, startFrameNum, endFrameNum):
    '''vector is a list'''
    vec = vec[startFrameNum:endFrameNum]
    out = []
    for e in vec:
        if e != 0:
            out.append(e)
    return np.mean(out)
    
def stdValueRejectZero(vec, startFrameNum, endFrameNum):
    '''vector is a list'''
    vec = vec[startFrameNum:endFrameNum]
    out = []
    for e in vec:
        if e != 0:
            out.append(e)
    return np.std(out)
    
def readSyllableMrk(syllableFilename):
    '''read syllable marker file'''
    inFile = open(syllableFilename)
    
    title = None
    startMrk = []
    endMrk = []
    syl = []
    
    for line in inFile:
        fields = line.split()
        if len(fields) == 0:
            continue
        if not isfloat(fields[0]):
            #this line is title
            title = line.strip() # remove \n
        else:
            startMrk.append(float(fields[0]))
            endMrk.append(float(fields[1]))
            syl.append(fields[2])
            
    return (title, startMrk, endMrk, syl)

def hz2cents(hz, tuning):
    '''convert Hz to cents
    input: float num in Hz
    output: float num in cents
    
    if tuning is 0, 0 cents is C5'''
    assert type(hz) == float
    cents = 1200 * log2(hz/(440 * pow(2,(0.25 + tuning))))
    if math.isinf(cents):
        cents = -1.0e+04
    return cents
    