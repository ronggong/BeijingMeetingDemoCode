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

def vecRejectValue(vec, vecRef = [], threshold = 0):
    out = []
    if len(vecRef) == 0:
        vecRef = vec
    for e in range(len(vecRef)):
        if vecRef[e] > threshold:
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

def readMelodiaPitch(inputFile):
    '''read syllable marker file'''
    inFile = open(inputFile, 'r')
    
    timeStamps = []
    pitch = []
    newPoints = [] # new point marker
    for line in inFile:
        fields = line.split()
        timeStamps.append(float(fields[0]))
        pitch.append(float(fields[1]))
        
        if len(fields) > 2:
            newPoints.append(fields[2])
        else:
            newPoints.append('')
            
    return (timeStamps, pitch, newPoints)

def readYileSegmentation(inputFile):
    '''read Yile's segmentation file'''
    inFile = open(inputFile, 'r')
    
    startTime = []
    dur = []
    segNum = []
    segMarker = []
    
    for line in inFile:
        fields = line.split()
        if len(fields) == 4:
            startTime.append(float(fields[0]))
            dur.append(float(fields[2]))
            segNum.append(int(fields[1]))
            segMarker.append(fields[3])
    return (startTime, dur, segNum, segMarker)

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

def hz2centsRafa(timeVec, pitchInHz, tonic=261.626, plotHisto = False):

    # with open(document, 'r') as f:
#         data = f.readlines()
# 
#     data2 = []
#     for i in range(len(data)):
#         x = []
#         time = float(data[i].split('\t')[0])
#         x.append(time)
#         value = float(data[i].split('\t')[1].rstrip('\r\n'))
#         x.append(value)
#         data2.append(x)

    cents = [-10000]*len(pitchInHz)
    for i in xrange(len(pitchInHz)):
        if pitchInHz[i] > 0:
            cents[i] = 1200*np.log2(1.0*pitchInHz[i]/tonic)
    data = zip(timeVec, cents)
    data_hist = np.array(data)

    pitch_obj = intonation.Pitch(data_hist[:, 0], data_hist[:, 1])
    #print data_hist[:,0], data_hist[:,1]
    rec_obj = intonation.Recording(pitch_obj)
    rec_obj.compute_hist()
    
    if plotHisto == True:
        rec_obj.histogram.plot()
        
    rec_obj.histogram.get_peaks()
    peaks = rec_obj.histogram.peaks
    return peaks['peaks']
    
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

def spectralSlope(spec,frameSize,fs,xlim):
    startHz = xlim[0]
    endHz = xlim[1]
    freqRes = fs/float(frameSize)/2
    startP = np.round(startHz/freqRes)
    endP = np.round(endHz/freqRes)
    
    freqBins = np.arange(spec.shape[0])*freqRes
    xvals = freqBins[startP:endP]
    yvals = spec[startP:endP]
    
    a, b = np.polyfit(xvals, yvals, 1)
    # a is slope
    return (a, b)
    
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
    