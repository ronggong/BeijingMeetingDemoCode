import sys, csv, codecs, cStringIO
import numpy as np
import essentia.standard as ess
from scipy.signal import freqz
from scipy.fftpack import fft, ifft, fftshift

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
    out = vecRejectZero(vec)
    return np.mean(out)
    
def stdValueRejectZero(vec, startFrameNum, endFrameNum):
    '''vector is a list'''
    vec = vec[startFrameNum:endFrameNum]
    out = vecRejectZero(vec)
    return np.std(out)

def vecRejectZero(vec, vecRef = []):
    out = []
    if len(vecRef) == 0:
        vecRef = vec
    for e in range(len(vecRef)):
        if vecRef[e] != 0:
            out.append(vec[e])
    return out
    
def readSyllableMrk(syllableFilename):
    '''read syllable marker file'''
    inFile = codecs.open(syllableFilename, 'r', 'utf-8')
    
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
    
def autolabelBar(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'%float(height),
                ha='center', va='bottom')

def lpcEnvelope(audioSamples, npts, order):
    '''npts is even number'''
    lpc = ess.LPC(order = order)
    lpcCoeffs = lpc(audioSamples)
    frequencyResponse = fft(lpcCoeffs[0], npts) 
    return frequencyResponse[:npts/2]
    
class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") if isinstance(s, basestring) else s for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
    