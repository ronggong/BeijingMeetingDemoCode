import sys, csv, os, collections
import essentia as es
import essentia.standard as ess
import matplotlib.pyplot as plt
import utilFunctionsRong as UFR
import numpy as np
from scipy.signal import get_window, lfilter

eps = np.finfo(np.float).eps

#set line width
plt.rcParams['lines.linewidth'] = 1
#set font size for titles 
plt.rcParams['axes.titlesize'] = 16
#set font size for labels on axes
plt.rcParams['axes.labelsize'] = 16
#set size of numbers on x-axis
plt.rcParams['xtick.major.size'] = 5
#set size of numbers on y-axis
plt.rcParams['ytick.major.size'] = 5

class FeaturesExtraction(object):
    '''abstract class'''
    
    def __init__(self, filename, fs = 44100, frameSize = 2048, hopSize = 256):
        self.features = ['speccentroid', 'specloudness', 'specflux']
        self.feature = None
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.fs = fs
        self.audio = ess.MonoLoader(filename = filename, sampleRate = fs)()
        self.mX = [] # the spectrogram
        self.frames = []
        self.featureVec = []
    
    def getAudio(self):
        return self.audio
    
    def getFrames(self):
        return self.frames
        
    def getFs(self):
        return self.fs
    
    def getFrameSize(self):
        return self.frameSize
    
    def getHopSize(self):
        return self.hopSize
        
    def getFeatures(self):
        return self.features
        
    def spectrogram(self):
        winAnalysis = 'hann'
        N = 2 * self.frameSize	# padding frameSize
        SPECTRUM = ess.Spectrum(size=N)
        WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N-self.frameSize) 
        
        print 'calculating spectrogram ... ...'
        mX = []
        for frame in ess.FrameGenerator(self.audio, frameSize=self.frameSize, hopSize=self.hopSize):
            self.frames.append(frame)
            frame = WINDOW(frame)
            mXFrame = SPECTRUM(frame)
            #store/append values per frame in an array
            mX.append(mXFrame)
        
        self.mX = mX
        print "spectrogram calculation done, return " + str(len(self.mX)) + ' frames.\n'
        return self.mX
        
    def extractFeature(self, feature):
        if len(self.mX) == 0:
            print 'Please run function spectrogram() firsly, then do ' + feature + 'calculation.'
            return
        
        if feature not in self.features:
            print 'the argument feature should be one of ', self.features
        
        self.feature = feature
        featureObject = None
        out = []
        if feature == self.features[0]: 
            featureObject = ess.Centroid(range=self.fs/2.0)
        elif feature == self.features[1]:
            featureObject = ess.Loudness()
        elif feature == self.features[2]:
            featureObject = ess.Flux()
        
        print 'extracting feature: ', feature
        
        for s in self.mX:               
            out.append(featureObject(s))
                
        self.featureVec = out
        print feature + ' calculation done, return ' + str(len(self.featureVec)) + ' values.\n'
        return self.featureVec
        
    def plotFeature(self):        
        if len(self.featureVec) == 0:
            print 'Please run extractFeature(feature) function firstly, then plot feature.'
            return
            
        xlabel = 'Time (s)'
        if self.feature == self.features[0]: 
            ylabel = 'Frequency (Hz)'
        elif self.feature == self.features[1]:
            ylabel = 'Norm Loudness'
        elif self.feature == self.features[2]:
            ylabel = 'Flux'
            
        featureVec = np.array(self.featureVec)
        timeStamps = np.arange(featureVec.size)*self.hopSize/float(self.fs)                             
        plt.plot(timeStamps,featureVec)
        meanValue = round(np.mean(featureVec),2)
        stdValue = round(np.std(featureVec),2)
        cvValue = meanValue/stdValue
        title = self.feature + ' mean: ' + '%.2f'%meanValue \
                + ' coeff of variation: ' + '%.2f'%cvValue
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.autoscale(tight=True)

class FeaturesExtractionSyllable(FeaturesExtraction):
    '''syllable level feature extraction'''
    
    def __init__(self, filename, syllableFilename, fs = 44100, frameSize = 2048, hopSize = 256):
        FeaturesExtraction.__init__(self, filename, fs = fs, frameSize = frameSize, hopSize = hopSize)
        self.syllableMrk = UFR.readSyllableMrk(syllableFilename)
        self.syllableMean = []
        self.syllableStd = []
    
    def getLegend(self):
        return self.syllableMrk[0]
    
    def getXticklabels(self):
        return self.syllableMrk[3]
        
    def getStartTime(self):
        return self.syllableMrk[1]
    def getEndTime(self):
        return self.syllableMrk[2]
    
    def meanStdSyllable(self):
        if len(self.featureVec) == 0:
            print 'Please run extractFeature(feature) function firstly, then do syllable level statistics.'
        
        startMrk = self.syllableMrk[1]
        endMrk = self.syllableMrk[2]
        
        frameLen = len(self.mX)
        syllableMean = []
        syllableStd = []
        for mrkNum in range(len(startMrk)):
            tStart = startMrk[mrkNum]
            tEnd = endMrk[mrkNum]
            fStart = UFR.findMinDisFrameNum(tStart, frameLen, self.hopSize, self.fs)
            fEnd = UFR.findMinDisFrameNum(tEnd, frameLen, self.hopSize, self.fs)
            
            mean = UFR.meanValueRejectZero(self.featureVec, fStart, fEnd)
            std = UFR.stdValueRejectZero(self.featureVec, fStart, fEnd)
            syllableMean.append(mean)
            syllableStd.append(std)
            
        self.syllableMean = syllableMean
        self.syllableStd = syllableStd
        print self.feature + ' syllable level calculation done, return ' \
        + str(len(self.syllableMean)) + ' values.\n'
        print 'syllable level mean features value: ', syllableMean
        print 'syllable level std features value: ', syllableStd, '\n'
        
        #return a dictionary
        return (syllableMean, syllableStd)
    
    def plotFeatureSyllable(self, ax = None):
        if len(self.syllableMean) == 0:
            print 'Please run meanStdSyllable function firstly, then do the plot.'
        
        yLabel = None
        if self.feature == self.features[0]: 
            ylabel = 'Frequency (Hz)'
        elif self.feature == self.features[1]:
            ylabel = 'Norm Loudness'
        elif self.feature == self.features[2]:
            ylabel = 'Flux'
        
        syllableNum = len(self.syllableMean)
        ind = np.arange(syllableNum)
        width = 0.35
        meanValue = round(np.mean(self.syllableMean),2)
        stdValue = round(np.std(self.syllableMean),2)
        cvValue = meanValue/stdValue
        barGraph = plt.bar(ind, self.syllableMean, width, color='r', yerr = self.syllableStd)
        title = self.feature + ' mean: ' + '%.2f'%meanValue + \
        ' coeff of variation: ' + '%.2f'%cvValue
        
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(ind+width/2.0, self.syllableMrk[3])
        if ax != None:
            UFR.autolabelBar(barGraph, ax)
        plt.legend((self.syllableMrk[0],))
        
def compareFeaturesSyllableMean(filenames, syllableFilenames, feature = 'speccentroid'):
    if type(filenames) == str:
        filenames = (filenames, )
        if syllableFilenames != None:
            syllableFilenames = (syllableFilenames, )
    
    if len(filenames) > 3:
        print 'we can''t compare more than 3 files right now.'
        return
        
    if feature == 'specloudness' and len(filenames) > 1:
        print("Warning: It doesn't make sense to compare loudness if two files are " + 
        "recorded differently. Because the recording environment, the use of recording " +
        "mixing technique (the use of compressor, expander or other dynamic control" + 
        "in music post production) are different.")
        
    obj = FeaturesExtractionSyllable(filenames[0], syllableFilenames[0])
    availableFeatures = obj.getFeatures()
    
    if feature not in availableFeatures:
        print 'the argument feature should be one of ', availableFeatures
        return

    legends = []
    xticklabelsObjs = []
    sylMeans = []
    sylStds = []
    for ii in range(len(filenames)):
        obj = FeaturesExtractionSyllable(filenames[ii], syllableFilenames[ii])
        obj.spectrogram()
        obj.extractFeature(feature)
        rdictObj = obj.meanStdSyllable()
        legendObj = obj.getLegend()
        xticklabelsObj = obj.getXticklabels()
    
        sylMean = rdictObj[0]
        sylStd = rdictObj[1]
        
        legends.append(legendObj)
        xticklabelsObjs.append(xticklabelsObj)
        sylMeans.append(sylMean)
        sylStds.append(sylStd)
    
    if len(filenames) > 1:
        for ii in range(1, len(filenames)):
            if len(sylMeans[ii]) != len(sylMeans[ii-1]):
                print('two syllable markers files contain different syllable number, ' +
                'please make sure the their syllable number be the same.')
                return
            elif len(sylMeans[ii]) == 0:
                print 'file doesn''t contain any syllable, please check audio file or syllable marker file.'
                return
            elif collections.Counter(xticklabelsObjs[ii]) != collections.Counter(xticklabelsObjs[ii-1]):
                print 'two syllable files doesn''t contain the same syllable list, please check syllable file.'
                return
                
    writeCSV(sylMeans, sylStds, legends, xticklabelsObj)
    plotFeaturesCompare(sylMeans, sylStds, legends, xticklabelsObj, feature, availableFeatures)
    
def plotFeaturesCompare(sylMeans, sylStds, legends, xticklabels, feature, availableFeatures):
    yLabel = None
    if feature == availableFeatures[0]: 
        ylabel = 'Frequency (Hz)'
    elif feature == availableFeatures[1]:
        ylabel = 'Norm Loudness'
    elif feature == availableFeatures[2]:
        ylabel = 'Flux'
    
    fig, ax = plt.subplots()
    syllableNum = len(sylMeans[0])
    width = 0.35
    ind = np.arange(syllableNum) * (width * (len(sylMeans)+1))
    title = feature
    colors = ('r', 'y','b')
    for ii in range(len(sylMeans)):
        bar = plt.bar(ind + ii*width, sylMeans[ii], width, color=colors[ii], yerr = sylStds[ii])
        meanValue = round(np.mean(sylMeans[ii]),2)
        stdValue = round(np.std(sylMeans[ii]),2)
        cvValue = meanValue/stdValue
        title = title + ' ' + legends[ii] + ' mean: ' + '%.2f'%meanValue + \
        ' coeff of variation:' + '%.2f'%cvValue + '\n'
        UFR.autolabelBar(bar, ax)
        
    title = title[:-2]
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(ind+(len(sylMeans)/2.0) * width, xticklabels)
    plt.legend(legends)
    plt.autoscale(tight=True)
    plt.show()
    
def writeCSV(sylMeans, sylStds, legendObjs, xticklabelsObj, outputFilename = None):
    if outputFilename == None:
        outputFilename = 'meanStdSyllableResults.csv'
    
    # make copy
    t = list(xticklabelsObj)
    
    t.insert(0, '')
    t.insert(0, '')
    
    things2write = [t,]
    for ii in range(len(sylMeans)):
        m = list(sylMeans[ii])
        s = list(sylStds[ii])
        m.insert(0, 'Mean')
        m.insert(0, legendObjs[ii])
        s.insert(0, 'Std')
        s.insert(0, '')
        things2write.append(m)
        things2write.append(s)
    
    length = len(m)
    f = open(outputFilename, 'w')
    csv_writer = UFR.UnicodeWriter(f)
    for y in range(length):
        csv_writer.writerow([x[y] for x in things2write])
    
    print 'result is wrote into: ', outputFilename, '\n'
    
def compareLPCSyllable(filenames, syllableFilenames, xaxis = 'linear'):
    if type(filenames) == str:
        filenames = (filenames, )
        if syllableFilenames != None:
            syllableFilenames = (syllableFilenames, )
    
    if len(filenames) > 3:
        print 'we can''t compare more than 3 files right now.'
        return
        
    audios = []
#     frames = []
    startTimes = []
    endTimes = []
    legendObjs = []
    xticklabelsObjs = []
        
    for ii in range(len(filenames)):
        obj = FeaturesExtractionSyllable(filenames[ii], syllableFilenames[ii])
#         obj.spectrogram() # to get frame
        frameSize = obj.getFrameSize()
        hopSize = obj.getHopSize()
        fs = obj.getFs()
        audio = obj.getAudio()
        frame = obj.getFrames()
        startTime = obj.getStartTime()
        endTime = obj.getEndTime()
        legendObj = obj.getLegend()
        xticklabelsObj = obj.getXticklabels()
            
        audios.append(audio)
#         frames.append(frame)
        startTimes.append(startTime)
        endTimes.append(endTime)
        legendObjs.append(legendObj)
        xticklabelsObjs.append(xticklabelsObj)
            
    if len(filenames) > 1:
        for ii in range(1, len(filenames)):
            if len(startTimes[ii]) != len(startTimes[ii-1]):
                print('two syllable markers files contain different syllable number, ' +
                'please make sure the their syllable number be the same.')
                return
            elif len(startTimes[ii]) == 0:
                print 'file doesn''t contain any syllable, please check audio file or syllable marker file.'
                return
            elif collections.Counter(xticklabelsObjs[ii]) != collections.Counter(xticklabelsObjs[ii-1]):
                print 'two syllable files doesn''t contain the same syllable list, please check syllable file.'
                return
                
    npts = 512
    styles = ['b-', 'r--', 'k:']
    for mrk in range(len(startTime)):
        fig, ax = plt.subplots()
        for ii in range(len(filenames)):
            startTime = startTimes[ii]
            endTime = endTimes[ii]
            audio = audios[ii]
#             frame = frames[ii]
            xticklabelsObj = xticklabelsObjs[ii]
            
            startSample = int(startTime[mrk]*fs)
            endSample = int(endTime[mrk]*fs)
            
#             frameLen = len(frame)
#             fStart = UFR.findMinDisFrameNum(startTime[mrk], frameLen, hopSize, fs)
#             fEnd = UFR.findMinDisFrameNum(endTime[mrk], frameLen, hopSize, fs)
#             sylFrame = frame[fStart: fEnd]
#             sylFrame = np.array(sylFrame)
#             sylSum = sylFrame.mean(1)
#             sylFrame = UFR.vecRejectZero(sylFrame, sylSum)
#             
#             frequencyResponses = []
#             for frame in sylFrame:
#                 fr = UFR.lpcEnvelope(frame, npts)
#                 frequencyResponses.append(abs(fr))
#             frequencyResponses = np.array(frequencyResponses)
#             frm = frequencyResponses.mean(0)
#             mY1 = 20*np.log10(frm)
            
            sylAudio = audio[startSample:endSample]
            sylAudio = UFR.vecRejectZero(sylAudio)
            sylAudio = np.array(sylAudio)
            
            # windowing signal
            window = get_window('hann', len(sylAudio))
            sylAudio = sylAudio * window
                        
            #pre-emphasis filter
            b = [1, -0.9375]
            sylAudio = lfilter(b, 1, sylAudio)
            
            frequencyResponse = UFR.lpcEnvelope(sylAudio.astype(np.float32), npts)
            mY2 = 20*np.log10(abs(frequencyResponse))
            syl = xticklabelsObj[mrk]
            style = styles[ii]
#             plotLPCCompare(mY1, styles[0], npts, fs, xaxis)
            plotLPCCompare(mY2, style, npts, fs, xaxis)
            
        plt.title('LPC envelope, syllable: ' + syl, family='Droid Sans Fallback')
        plt.legend(legendObjs, loc = 'best')
#         plt.legend(('lpc small frame average', 'lpc one frame for a syllable'), loc = 'best')
    plt.show()
            
def plotLPCCompare(mY, style, npts, fs, xaxis):
    if xaxis != 'linear' and xaxis != 'log':
        print 'xaxis should one of linear of log. use default xaxis = ''linear'''
        xaxis = 'linear'
    if xaxis == 'log':
        plt.semilogx()   
        
    plt.plot(np.arange(0, fs/2.0, fs/float(npts)), -mY, style)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.autoscale(tight=True)
    plt.grid(True)
    
def compareLTAS(filenames, syllableFilenames = None, singerName = None,xaxis = 'linear', plotSD = True):
    if type(filenames) == str:
        filenames = (filenames, )
        if syllableFilenames != None:
            syllableFilenames = (syllableFilenames, )
    
    if len(filenames) > 3:
        print 'we can''t compare more than 3 files right now.'
        return
    
    styles = ['b-', 'r--', 'k:']
    if syllableFilenames == None:
        fig, ax = plt.subplots()
        legend = []
        meanPlots = []
        singer = 1
        for f in filenames:
            obj = FeaturesExtraction(f)
            frameSize = obj.getFrameSize()
            fs = obj.getFs()
            spec = obj.spectrogram()
            spec = np.array(spec)
            sumSpec = spec.mean(1)
            spec = UFR.vecRejectZero(spec, sumSpec) # reject zero
            spec = np.array(spec)
            specDB = 20*np.log10(spec+eps)
            
            meanSpec = spec.mean(0)
            meanSpecDB = 20 * np.log10(meanSpec+eps)
            meanSpecDB = meanSpecDB - max(meanSpecDB)
            
            # this is physically incorrect to calculate std in db scale, 
            # however we just want to show the variation in each frequency
            stdSpecDB = specDB.std(0) 
            
            meanPlot = plotLTAS(meanSpecDB, stdSpecDB, styles[singer - 1], fs, \
            frameSize, xaxis, plotSD)
            meanPlots.append(meanPlot[0])
            
            if singerName == None:
                singerString = 'singer' + str(singer)
            elif len(singerName) >= len(filenames):
                singerString = singerName[singer-1]
            else:
                print 'singerName contains less singers than the file number.'
                return
                
            legend.append(singerString)
            singer = singer + 1
            
        plt.title('LTAS')
        plt.legend(meanPlots, legend, loc = 'best')
        plt.show()
    else:
        spectro = []
        frameLens = []
        startTimes = []
        endTimes = []
        legendObjs = []
        xticklabelsObjs = []
        
        for ii in range(len(filenames)):
            obj = FeaturesExtractionSyllable(filenames[ii], syllableFilenames[ii])
            frameSize = obj.getFrameSize()
            hopSize = obj.getHopSize()
            fs = obj.getFs()
            spec = obj.spectrogram()
            spec = np.array(spec)
            startTime = obj.getStartTime()
            endTime = obj.getEndTime()
            legendObj = obj.getLegend()
            xticklabelsObj = obj.getXticklabels()
            
            spectro.append(spec)
            frameLens.append(len(spec))
            startTimes.append(startTime)
            endTimes.append(endTime)
            legendObjs.append(legendObj)
            xticklabelsObjs.append(xticklabelsObj)
            
        if len(filenames) > 1:
            for ii in range(1, len(filenames)):
                if len(startTimes[ii]) != len(startTimes[ii-1]):
                    print('two syllable markers files contain different syllable number, ' +
                    'please make sure the their syllable number be the same.')
                    return
                elif len(startTimes[ii]) == 0:
                    print 'file doesn''t contain any syllable, please check audio file or syllable marker file.'
                    return
                elif collections.Counter(xticklabelsObjs[ii]) != collections.Counter(xticklabelsObjs[ii-1]):
                    print 'two syllable files doesn''t contain the same syllable list, please check syllable file.'
                    return

        for mrkNum in range(len(startTime)):
            fig = plt.subplots()
            meanPlots = []
            for ii in range(len(filenames)):
                startTime = startTimes[ii]
                endTime = endTimes[ii]
                #legendObj = legendObjs[ii]
                xticklabelsObj = xticklabelsObjs[ii]
                
                syllable = xticklabelsObj[mrkNum]
                tStart = startTime[mrkNum]
                tEnd = endTime[mrkNum]
                fStart = UFR.findMinDisFrameNum(tStart, frameLens[ii], hopSize, fs)
                fEnd = UFR.findMinDisFrameNum(tEnd, frameLens[ii], hopSize, fs)
            
                sylSpec = spectro[ii][fStart:fEnd]
                sumSpec = sylSpec.mean(1)
                sylSpec = UFR.vecRejectZero(sylSpec, sumSpec)
                sylSpec = np.array(sylSpec)
                sylSpecDB = 20*np.log10(sylSpec+eps)

                meanSpec = 20*np.log10(sylSpec.mean(0) + eps)
                meanSpec = meanSpec - np.max(meanSpec)
                stdSpec = sylSpecDB.std(0)
                
                style = styles[ii]
                meanPlot = plotLTAS(meanSpec, stdSpec, style, fs, frameSize, xaxis, plotSD)
                meanPlots.append(meanPlot[0])
                plt.title('LTAS, syllable: ' + xticklabelsObj[mrkNum], family='Droid Sans Fallback')
                plt.legend(meanPlots, legendObjs, loc = 'best')
        plt.show()
        
            
def plotLTAS(meanSpec, stdSpec, style, fs, frameSize, xaxis, plotSD):
    freqBins = np.arange(meanSpec.shape[0])*(fs/float(frameSize)/2)
    indexNum = 100
    if xaxis != 'linear' and xaxis != 'log':
        print 'xaxis should one of linear of log. use default xaxis = ''linear'''
        xaxis = 'linear'
    if xaxis == 'log':
        plt.semilogx()
        index = np.logspace(0, 1, num = indexNum, base = frameSize)
    else:
        index = np.linspace(0, len(freqBins)-1, indexNum)
        
    index = index.astype(int)
    if plotSD == True:
        step = round(200 / (fs/float(frameSize)/2))
        meanPlot = plt.errorbar(freqBins[index], meanSpec[index], yerr = stdSpec[index], fmt = style)
    else:
        meanPlot = plt.plot(freqBins, meanSpec, style)  
        
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.xlim(0, 20000)
    plt.grid(True)
    
    return meanPlot
