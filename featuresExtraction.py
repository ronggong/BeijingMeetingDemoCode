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
    
    def getFeatures(self):
        return self.features
        
    def spectrogram(self):
        winAnalysis = 'hann'
        N = 2 * self.frameSize	# padding frameSize
        SPECTRUM = ess.Spectrum(size=N)
        WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N-self.frameSize) 
        
        mX = []
        for frame in ess.FrameGenerator(self.audio, frameSize=self.frameSize, hopSize=self.hopSize):
            frame = WINDOW(frame)
            mXFrame = SPECTRUM(frame)
            #store/append values per frame in an array
            mX.append(mXFrame)
        
        self.mX = mX
        print "spectrogram calculation done, return " + str(len(self.mX)) + ' frames.'
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
            
        for s in self.mX:
            out.append(featureObject(s))
        self.featureVec = out
        print feature + ' calculation done, return ' + str(len(self.featureVec)) + ' values.'
        return self.featureVec
        
    def plotFeature(self):        
        if len(self.featureVec) == 0:
            print 'Please run extractFeature(feature) function firstly, then plot feature.'
            return
        
        yLabel = None
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
        plt.xlabel('Time (s)')
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
        + str(len(self.syllableMean)) + ' values.'
        
        #return a dictionary
        return (syllableMean, syllableStd)
    
    def plotFeatureSyllable(self):
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
    
    plt.figure()
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
    plt.show()
    
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


        