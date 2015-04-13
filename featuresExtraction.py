import sys, csv, os, collections
import essentia as es
import essentia.standard as ess
import matplotlib.pyplot as plt
import utilFunctionsRong as UFR
import numpy as np
from scipy.signal import get_window
from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft, fftshift

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
        self.mX = []	# the spectrogram
        self.featureVec = []
    
    def getAudio(self):
        return self.audio
        
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
        title = self.feature + ' mean: ' + '%.2f'%round(np.mean(featureVec),2) \
                + ' std: ' + '%.2f'%round(np.std(featureVec),2)
        
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
        barGraph = plt.bar(ind, self.syllableMean, width, color='r', yerr = self.syllableStd)
        title = self.feature + ' mean: ' + '%.2f'%round(np.mean(self.syllableMean),2) + \
        ' std: ' + '%.2f'%round(np.std(self.syllableMean),2)
        
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(ind+width/2.0, self.syllableMrk[3])
        if ax != None:
            UFR.autolabelBar(barGraph, ax)
        plt.legend((self.syllableMrk[0],))
        
def compareFeaturesSyllableMean(filename1, syllableFilename1, filename2, syllableFilename2, feature = 'speccentroid'):
    if feature == 'specloudness':
        print("Warning: It doesn't make sense to compare loudness if two files are " + 
        "recorded differently. Because the recording environment, the use of recording " +
        "mixing technique (the use of compressor, expander or other dynamic control" + 
        "in music post production) are different.")
        
    obj1 = FeaturesExtractionSyllable(filename1, syllableFilename1)
    availableFeatures = obj1.getFeatures()
    
    if feature not in availableFeatures:
        print 'the argument feature should be one of ', availableFeatures
        return

    obj1.spectrogram()
    obj1.extractFeature(feature)
    rdictObj1 = obj1.meanStdSyllable()
    legendObj1 = obj1.getLegend()
    xticklabelsObj1 = obj1.getXticklabels()
    
    obj2 = FeaturesExtractionSyllable(filename2, syllableFilename2)
    obj2.spectrogram()
    obj2.extractFeature(feature)
    rdictObj2 = obj2.meanStdSyllable()
    legendObj2 = obj2.getLegend()
    xticklabelsObj2 = obj2.getXticklabels()
    
    sylMean1 = rdictObj1[0]
    sylStd1 = rdictObj1[1]
    sylMean2 = rdictObj2[0]
    sylStd2 = rdictObj2[1]
    
    if len(sylMean1) != len(sylMean2):
        print 'two syllable markers files contain different syllable number, \
        please make sure the their syllable number be the same.'
        return
    elif len(sylMean1) == 0:
        print 'file doesn''t contain any syllable, please check audio file or syllable marker file.'
        return
    elif collections.Counter(xticklabelsObj1) != collections.Counter(xticklabelsObj2):
        print 'two syllable files doesn''t contain the same syllable list, please check syllable file.'
        return
    else:
        writeCSV(sylMean1, sylStd1, legendObj1, sylMean2, sylStd2, legendObj2, xticklabelsObj1)
        plotFeaturesCompare(sylMean1, sylStd1, legendObj1, sylMean2, sylStd2, \
        legendObj2, xticklabelsObj1, feature, availableFeatures)
    
def plotFeaturesCompare(sylMean1, sylStd1, legendObj1, sylMean2, sylStd2, legendObj2, xticklabels,feature, availableFeatures):
    yLabel = None
    if feature == availableFeatures[0]: 
        ylabel = 'Frequency (Hz)'
    elif feature == availableFeatures[1]:
        ylabel = 'Norm Loudness'
    elif feature == availableFeatures[2]:
        ylabel = 'Flux'
    
    fig, ax = plt.subplots()
    syllableNum = len(sylMean1)
    ind = np.arange(syllableNum)
    width = 0.35
    bar1 = plt.bar(ind, sylMean1, width, color='r', yerr = sylStd1)
    bar2 = plt.bar(ind+width, sylMean2, width, color='y', yerr = sylStd2)
    title = feature + ' red bar mean: ' + '%.2f'%round(np.mean(sylMean1),2) + \
    ' std: ' + '%.2f'%round(np.std(sylMean1),2) + '\n' + \
    'yellow bar mean: ' + '%.2f'%round(np.mean(sylMean2),2) + \
    ' std: ' + '%.2f'%round(np.std(sylMean2),2)
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(ind+width, xticklabels)
    plt.legend((legendObj1, legendObj2))
    
    UFR.autolabelBar(bar1, ax)
    UFR.autolabelBar(bar2, ax)
    plt.show()
    
def writeCSV(sylMean1, sylStd1, legendObj1, sylMean2, sylStd2, legendObj2, xticklabelsObj, outputFilename = None):
    if outputFilename == None:
        outputFilename = 'meanStdSyllableResults.csv'
    
    if len(sylMean1) == 0 or len(sylMean2) == 0 or len(sylMean1) != len(sylMean2):
        return
    
    # make copy
    m1 = list(sylMean1)
    s1 = list(sylStd1)
    m2 = list(sylMean2)
    s2 = list(sylStd2)
    t = list(xticklabelsObj)
    
    t.insert(0, '')
    t.insert(0, '')
    m1.insert(0, 'Mean')
    m1.insert(0, legendObj1)
    s1.insert(0, 'Std')
    s1.insert(0, '')
    
    m2.insert(0, 'Mean')
    m2.insert(0, legendObj2)
    s2.insert(0, 'Std')
    s2.insert(0, '')
    
    things2write = [t,m1, s1, m2, s2]
    
    length = len(sylMean1)
    with open(outputFilename, 'wb') as csv_handle:
        csv_writer = csv.writer(csv_handle, delimiter=' ')
        for y in range(length):
            csv_writer.writerow([x[y] for x in things2write])
    
    print 'result is wrote into: ', outputFilename, '\n'
    
def compareLPCSyllable(filenames, syllableFilenames, xaxis = 'linear'):
    if len(filenames) > 3:
        print 'we can''t compare more than 3 files right now.'
        return
        
    audios = []
    startTimes = []
    endTimes = []
    legendObjs = []
    xticklabelsObjs = []
        
    for ii in range(len(filenames)):
        obj = FeaturesExtractionSyllable(filenames[ii], syllableFilenames[ii])
        frameSize = obj.getFrameSize()
        hopSize = obj.getHopSize()
        fs = obj.getFs()
        audio = obj.getAudio()
        startTime = obj.getStartTime()
        endTime = obj.getEndTime()
        legendObj = obj.getLegend()
        xticklabelsObj = obj.getXticklabels()
            
        audios.append(audio)
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
            xticklabelsObj = xticklabelsObjs[ii]
            
            startSample = int(startTime[mrk]*fs)
            endSample = int(endTime[mrk]*fs)
            sylAudio = audio[startSample:endSample]
            syl = xticklabelsObj[mrk]
            
            frequencyResponse = UFR.lpcEnvelope(sylAudio, npts)
            mY = 20*np.log10(abs(frequencyResponse))
            
            style = styles[ii]
            plotLPCCompare(mY, style, npts, fs, xaxis)
            
        plt.title('LPC envelope, syllable: ' + syl)
        plt.legend(legendObjs)
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
    
def compareLTAS(filenames, syllableFilenames = None, singerName = None,xaxis = 'linear'):
    if len(filenames) > 3:
        print 'we can''t compare more than 3 files right now.'
        return
    
    styles = ['b-', 'r--', 'k:']
    if syllableFilenames == None:
        fig, ax = plt.subplots()
        legend = []
        singer = 1
        for f in filenames:
            obj = FeaturesExtraction(f)
            frameSize = obj.getFrameSize()
            fs = obj.getFs()
            spec = obj.spectrogram()
            spec = np.array(spec)
            mean = spec.mean(0)
            plotLTAS(meanSpec = mean, style = styles[singer - 1], fs = fs, \
            frameSize = frameSize, xaxis = xaxis)
            
            if singerName == None:
                singerString = 'singer' + str(singer)
            else:
                singerString = singerName[singer-1]
            legend.append(singerString)
            singer = singer + 1
            
        plt.title('LTAS')
        plt.legend(legend)
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
            
                meanSpec = spectro[ii][fStart:fEnd].mean(0)
                #std = UFR.stdValueRejectZero(self.featureVec, fStart, fEnd)
                
                style = styles[ii]
                plotLTAS(meanSpec, style, fs, frameSize, xaxis)
                plt.title('LTAS, syllable: ' + xticklabelsObj[mrkNum])
                plt.legend(legendObjs)
        plt.show()
        
            
def plotLTAS(meanSpec, style, fs, frameSize, xaxis):
    freqBins = np.arange(meanSpec.shape[0])*(fs/float(frameSize)/2)
    meanSpec = plt.plot(freqBins, 20 * np.log10(meanSpec), style)        
    if xaxis != 'linear' and xaxis != 'log':
        print 'xaxis should one of linear of log. use default xaxis = ''linear'''
        xaxis = 'linear'
    if xaxis == 'log':
        plt.semilogx()   
        
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.xlim(0, 10000)
    plt.grid(True)      


# filename1 = '../daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section.wav'
# syllableFilename1 = '../daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section-words-mrkRearrange.txt'
# filename2 = '../daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section.wav'
# syllableFilename2 = '../daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'
# 
# compareFeaturesSyllableMean(filename1, syllableFilename1, filename2, syllableFilename2, feature = 'speccentroid')

# rdict = UFR.readSyllableMrk(syllableFilename1)
# print rdict[0]
# print rdict[1]
# print rdict[2]
# print rdict[3]

# test1 = FeaturesExtraction(filename = filename);
# test1.spectrogram()
# 
# test1.extractFeature('speccentroid')

# centroid = 0
# plt.figure(centroid)
# test1.plotFeature()
# 
# test1.extractFeature('specloudness')
# 
# loudness = 1
# plt.figure(loudness)
# test1.plotFeature()
# 
# test1.extractFeature('specflux')
# flux = 2
# plt.figure(flux)
# test1.plotFeature()
# 
# plt.show()

# test2 = FeaturesExtractionSyllable(filename1, syllableFilename1)
# test2.spectrogram()
# test2.extractFeature('specloudness')
# test2.meanStdSyllable()
# plt.figure(0)
# test2.plotFeature()
# 
# plt.figure(1)
# test2.plotFeatureSyllable()
# plt.show()
