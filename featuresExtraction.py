import sys, csv, os, collections
import essentia as es
import essentia.standard as ess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import utilFunctionsRong as UFR
import numpy as np
from scipy.signal import get_window, lfilter

eps = np.finfo(np.float).eps
droidTitle = fm.FontProperties(fname='font/DroidSansFallback.ttf', size = 16)
droidLegend = fm.FontProperties(fname='font/DroidSansFallback.ttf', size = 12)
droidTick = fm.FontProperties(fname='font/DroidSansFallback.ttf', size = 12)

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
        self.features = ['speccentroid', 'specloudness', 'specflux', 'tristimulus']
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
        
    def extractFeature(self, feature, normTo = 0):
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
            loudnessObject = ess.Loudness()
            featureObject = ess.Flux()
        elif feature == self.features[3]:
            PEAKS = ess.SpectralPeaks(sampleRate = self.fs)
            HPEAKS = ess.HarmonicPeaks()
            TRIST = ess.Tristimulus()
            PITCH = ess.PitchYinFFT(minFrequency = 50, maxFrequency = 1000, sampleRate = self.fs)
        
        print 'extracting feature: ', feature
        
        if feature == self.features[3]:
        # to plot the tristimulus
            for s in self.mX:
                peaksFreq, peaksMag = PEAKS(s)
                if len(peaksFreq) == 0:
                    trist = np.array([0,0,0])
                else:
                    if peaksFreq[0] == 0:
                        peaksFreq = peaksFreq[1:]
                        peaksMag = peaksMag[1:]
                    pitch, confidence = PITCH(s)
                    #print 'pitch: ', pitch, confidence
                    if confidence > 0.7:
                        hpeaksFreq, hpeaksMag = HPEAKS(peaksFreq, peaksMag, pitch)
                        trist = TRIST(hpeaksFreq, hpeaksMag)
                    else:
                        trist = np.array([0,0,0])
                out.append(trist)
        else:
            if normTo > 0:
                # when Extract Flux feature, normalize loudness
                if feature == self.features[2]:
                    for s in self.mX:               
                        out.append(loudnessObject(s))
                else:
                    for s in self.mX:               
                        out.append(featureObject(s))
            
                # if feature is loudness normalize
                if feature == self.features[1] or feature == self.features[2]:
                    meanLoud = np.mean(UFR.vecRejectZero(np.array(out)))
                    normCoeff = pow(normTo/meanLoud, 1/0.67/2.0)
                    out = []
                    for s in self.mX:
                        out.append(featureObject(s * normCoeff))
            else:
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
        elif self.feature == self.features[3]:
            ylabel = 'Trist'
            trist0 = []
            trist1 = []
            trist2 = []
            for item in self.featureVec:
                trist0.append(item[0])
                trist1.append(item[1])
                trist2.append(item[2])
        
        if self.feature == self.features[3]:
            trist0 = np.array(trist0)
            trist1 = np.array(trist1)
            trist2 = np.array(trist2)
            timeStamps = np.arange(trist0.size)*self.hopSize/float(self.fs)     

            t0Plt = plt.plot(timeStamps, trist0)
            t1Plt = plt.plot(timeStamps, trist1)
            t2Plt = plt.plot(timeStamps, trist2)
            meant0 = np.mean(UFR.vecRejectValue(trist0, threshold = 0))
            meant1 = np.mean(UFR.vecRejectValue(trist1, threshold = 0))
            meant2 = np.mean(UFR.vecRejectValue(trist2, threshold = 0))
            stdt0 = np.std(UFR.vecRejectValue(trist0, threshold = 0))
            stdt1 = np.std(UFR.vecRejectValue(trist1, threshold = 0))
            stdt2 = np.std(UFR.vecRejectValue(trist2, threshold = 0))
            
            sumMean = meant0 + meant1 + meant2
            # normalised mean
            nmt0 = meant0/sumMean
            nmt1 = meant1/sumMean
            nmt2 = meant2/sumMean
            title = self.feature
            
            legend0 = 't0 mean:'+str(round(meant0,2))+ ' norm mean:' + str(round(nmt0,2)) + ' std:'+str(round(stdt0,2))
            legend1 = 't1 mean:'+str(round(meant1,2))+ ' norm mean:' + str(round(nmt1,2)) + ' std:'+str(round(stdt1,2))
            legend2 = 't2 mean:'+str(round(meant2,2))+ ' norm mean:' + str(round(nmt2,2)) + ' std:'+str(round(stdt2,2))

            plt.legend((t0Plt[0], t1Plt[0], t2Plt[0]), (legend0, legend1, legend2))
        else:
            featureVec = np.array(self.featureVec)
            timeStamps = np.arange(featureVec.size)*self.hopSize/float(self.fs)                             
            plt.plot(timeStamps,featureVec)
            meanValue = np.mean(UFR.vecRejectZero(featureVec))
            stdValue = np.std(UFR.vecRejectZero(featureVec))
            cvValue = stdValue/meanValue
            title = self.feature + ' mean: ' + '%.3f'%round(meanValue,3) \
                    + ' standard deviation: ' + '%.3f'%round(stdValue,3)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.autoscale(tight=True)

class FeaturesExtractionSyllable(FeaturesExtraction):
    '''syllable level feature extraction'''
    
    def __init__(self, filename, syllableFilename, fs = 44100, frameSize = 2048, hopSize = 256):
        FeaturesExtraction.__init__(self, filename, fs = fs, frameSize = frameSize, hopSize = hopSize)
        self.syllableMrk = UFR.readSyllableMrk(syllableFilename)
        self.syllableVecs = []
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
        
    def getSyllableVecs(self):
        return self.syllableVecs
    
    def meanStdSyllable(self):
        if len(self.featureVec) == 0:
            print 'Please run extractFeature(feature) function firstly, then do syllable level statistics.'
        
        startMrk = self.syllableMrk[1]
        endMrk = self.syllableMrk[2]
        
        frameLen = len(self.mX)
        syllableMean = []
        syllableStd = []
        syllableVecs = []
        for mrkNum in range(len(startMrk)):
            tStart = startMrk[mrkNum]
            tEnd = endMrk[mrkNum]
            fStart = UFR.findMinDisFrameNum(tStart, frameLen, self.hopSize, self.fs)
            fEnd = UFR.findMinDisFrameNum(tEnd, frameLen, self.hopSize, self.fs)
            
            self.syllableVecs.append(self.featureVec[fStart:fEnd])
            
            mean = UFR.meanValueRejectZero(self.featureVec, fStart, fEnd)
            std = UFR.stdValueRejectZero(self.featureVec, fStart, fEnd)
            syllableMean.append(mean)
            syllableStd.append(std)
            
        self.syllableMean = syllableMean
        self.syllableStd = syllableStd
        print self.feature + ' syllable level calculation done, return ' \
        + str(len(self.syllableMean)) + ' values.\n'
        # print 'syllable level mean features value: ', syllableMean
        # print 'syllable level std features value: ', syllableStd, '\n'
        
        #return a dictionary
        return (syllableMean, syllableStd)
    
    def plotFeatureSyllableMean(self, ax = None):
        if len(self.syllableMean) == 0:
            print 'Please run meanStdSyllable function firstly, then do the plot.'
        
        if self.feature == self.features[0]: 
            ylabel = 'Frequency (Hz)'
        elif self.feature == self.features[1]:
            ylabel = 'Norm Loudness'
        elif self.feature == self.features[2]:
            ylabel = 'Flux'
        elif self.feature == self.features[3]:
            print 'we don''t support the syllable level tristimulus. sorry.'
            return
        
        syllableNum = len(self.syllableMean)
        ind = np.arange(syllableNum)
        width = 0.35
        meanValue = np.mean(self.syllableMean)
        stdValue = np.std(self.syllableMean)
        cvValue = stdValue/meanValue
        barGraph = plt.bar(ind, self.syllableMean, width, color='r', yerr = self.syllableStd)
        title = self.feature + ' mean: ' + '%.3f'%round(meanValue,3) + \
        ' standard deviation: ' + '%.3f'%round(stdValue,3)
        
        plt.title(title, fontproperties=droidTitle)
        plt.ylabel(ylabel)
        plt.xticks(ind+width/2.0, self.syllableMrk[3], fontproperties=droidTick)
        if ax != None:
            UFR.autolabelBar(barGraph, ax)
        plt.legend((self.syllableMrk[0],), prop=droidLegend)
        
def plotFeatureSyllable(filename, syllableFilename = None, pitchtrackFilename = None, textOffsetX = 1.5,feature = 'speccentroid'):
    if pitchtrackFilename != None:
        print 'The hopSize is going to be defined as 128. Please make sure that is the hopsize that you used for the pitch track.'
        hopSize = 128
    else: 
        hopSize = 256
    
    if syllableFilename != None:
        obj = FeaturesExtractionSyllable(filename, syllableFilename, hopSize = hopSize)
    else:
        obj = FeaturesExtraction(filename, hopSize = hopSize)
        
    availableFeatures = obj.getFeatures()
    
    if feature not in availableFeatures:
        print 'the argument feature should be one of ', availableFeatures
        return
    if feature == 'tristimulus':
        print 'we don''t support the syllable level tristimulus right now, sorry.'
        return
    
    obj.spectrogram()
    featureVec = obj.extractFeature(feature)
    hopSize = obj.getHopSize()
    fs = obj.getFs()
    
    yLabel = None
    if feature == availableFeatures[0]: 
        ylabel = 'Frequency (Hz)'
    elif feature == availableFeatures[1]:
        ylabel = 'Norm Loudness'
    elif feature == availableFeatures[2]:
        ylabel = 'Flux'
    
    if syllableFilename != None:
        obj.meanStdSyllable()
        syllableVecs = obj.getSyllableVecs()
        legends = obj.getLegend()
        xticklabels = obj.getXticklabels()
    else:
        syllableVecs = (featureVec, )
    
    if pitchtrackFilename == None: 
        for ii in range(len(syllableVecs)):
            sylVec = syllableVecs[ii]
            fig, ax = plt.subplots()
            sylVec = np.array(sylVec)
            timeStamps = np.arange(sylVec.size)*hopSize/float(fs)   
                                  
            plt.plot(timeStamps,sylVec)
        
            meanValue = np.mean(UFR.vecRejectZero(sylVec))
            stdValue = np.std(UFR.vecRejectZero(sylVec))
            cvValue = stdValue/meanValue
            if syllableFilename != None:
                title = feature + ' ' + xticklabels[ii] + ' mean: ' + '%.3f'%round(meanValue,3) \
                        + ' standard deviation: ' + '%.3f'%round(stdValue,3)
            else:
                title = feature + ' mean: ' + '%.3f'%round(meanValue,3) \
                        + ' standard deviation: ' + '%.3f'%round(stdValue,3)
        
            plt.title(title, fontproperties=droidTitle)
            plt.xlabel('Time (s)')
            plt.ylabel(ylabel)
            plt.autoscale(tight=True)
            plt.show()
        
    else:
        rtuple = UFR.readMelodiaPitch(pitchtrackFilename)
        timeStampsPitch = rtuple[0]
        pitch = rtuple[1]
        
        if syllableFilename != None:
            tStart = obj.getStartTime()
            tEnd = obj.getEndTime()
        else:
            tStart = [0] # tStart is not important here
            fStart = 0
            fEnd = len(pitch)
                
        max_yticks = 4
        linewidth = 2
        
        for ii in range(len(tStart)):
            if syllableFilename != None:
                fStart = UFR.findMinDisFrameNum(tStart[ii], len(pitch), hopSize, fs)
                fEnd = UFR.findMinDisFrameNum(tEnd[ii], len(pitch), hopSize, fs)
                plotOffset = tStart[ii]
            else:
                plotOffset = 0
            
            sylVec = syllableVecs[ii]
            sylVec = np.array(sylVec)
            stdValue = np.std(sylVec)
            timeStamps = np.arange(sylVec.size)*hopSize/float(fs) + plotOffset
        
            if 'xticklabels' in locals():
                text = xticklabels[ii] + ' SD=' + str(np.round(stdValue,3))
            else:
                text = 'SD=' + str(np.round(stdValue,3))

            text_xpos = (timeStamps[-1] - textOffsetX)
            text_ypos = ((np.max(sylVec) - np.min(sylVec)) * 0.9)
        
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(timeStamps,sylVec, linewidth = linewidth)
            # ax[0].text(text_xpos, text_ypos, text, fontproperties = droidLegend)
            ax[0].set_ylabel(ylabel)
            ax[0].autoscale(tight=True)
            ax[0].set_title(text, fontproperties=droidTitle)

            ax[1].plot(timeStampsPitch[fStart:fEnd], pitch[fStart:fEnd], linewidth = linewidth)
            ax[1].autoscale(tight=True)
            ax[1].set_ylabel('Freq (Hz)')
            fig.tight_layout()
            plt.xlabel('Time (s)')
            plt.show()
        
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
        " in music post production) are different.")
        
    if feature == 'tristimulus':
        print 'we can''t compare tristimulus right now, sorry.'
        return
        
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
        meanValue = np.mean(sylMeans[ii])
        stdValue = np.std(sylMeans[ii])
        cvValue = stdValue/meanValue
        title = title + ' ' + legends[ii] + ' mean: ' + '%.3f'%round(meanValue,3) + \
        ' standard deviation:' + '%.3f'%round(stdValue,3) + '\n'
        UFR.autolabelBar(bar, ax)
        
    title = title[:-1]
        
    plt.title(title, fontproperties=droidTitle)
    plt.ylabel(ylabel)
    plt.xticks(ind+(len(sylMeans)/2.0) * width, xticklabels, fontproperties=droidTick)
    plt.legend(legends, prop=droidLegend)
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
    
def compareLPCSyllable(filenames, syllableFilenames, lpcorder = 10, xaxis = 'linear', xlim = [], ylim = []):
    if type(filenames) == str:
        filenames = (filenames, )
        if syllableFilenames != None:
            syllableFilenames = (syllableFilenames, )
    
    if len(filenames) > 3:
        print 'we can''t compare more than 3 files right now.'
        return
    
    if lpcorder < 1 or lpcorder > 50:
        print 'please choose a reasonable lpc order, like between [8, 14].'
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
            
            #pre-emphasis filter
            b = [1, -0.9375]
            sylAudio = lfilter(b, 1, sylAudio)
            
            # windowing signal
            window = get_window('hann', len(sylAudio))
            sylAudio = sylAudio * window
            
            frequencyResponse = UFR.lpcEnvelope(sylAudio.astype(np.float32), npts, lpcorder)
            mY2 = 20*np.log10(abs(frequencyResponse))
            syl = xticklabelsObj[mrk]
            style = styles[ii]
#             plotLPCCompare(mY1, styles[0], npts, fs, xaxis)
            plotLPCCompare(mY2, style, npts, fs, xaxis, xlim, ylim)
            
        plt.title('LPC envelope, syllable: ' + syl, fontproperties=droidTitle)
        plt.legend(legendObjs, loc = 'best', prop=droidLegend)
#         plt.legend(('lpc small frame average', 'lpc one frame for a syllable'), loc = 'best')
    plt.show()
            
def plotLPCCompare(mY, style, npts, fs, xaxis, xlim, ylim):
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
    
    if len(xlim) != 0:
        plt.xlim(xlim)
    
    if len(ylim) != 0:
        plt.ylim(ylim)
    
def compareLTAS(filenames, syllableFilenames = None, singerName = None,xaxis = 'linear', plotSD = True, xlim = [], ylim = []):
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
            centroid = (centroidLTAS(meanSpec, fs))
            
            meanSpecDB = 20 * np.log10(meanSpec+eps)
            meanSpecDB = meanSpecDB - max(meanSpecDB)
            
            # ---- calculate spectral slope 
            # fitCoeffs = UFR.spectralSlope(meanSpecDB,frameSize,fs,[200, 5000])
            # slope = fitCoeffs[0] * (4186-261)/4.0
            
            # this is physically incorrect to calculate std in db scale, 
            # however we just want to show the variation in each frequency
            stdSpecDB = specDB.std(0) 
            
            meanPlot = plotLTAS(meanSpecDB, stdSpecDB, styles[singer - 1], fs, \
            frameSize, xaxis, plotSD, xlim, ylim)
            meanPlots.append(meanPlot[0])
            
            if singerName == None:
                singerString = 'singer' + str(singer)
            elif len(singerName) >= len(filenames):
                singerString = singerName[singer-1]
            else:
                print 'singerName contains less singers than the file number.'
                return
                
            legend.append(singerString + ' Centroid: ' + str(centroid))
            singer = singer + 1
            
        plt.title('LTAS')
        plt.legend(meanPlots, legend, loc = 'best', prop=droidLegend)
        # use different line style
        # ax.xaxis.grid(True, which = 'Major', linestyle = '-', linewidth = 0.25)
#         ax.xaxis.grid(True, which = 'Minor', linestyle = '-', linewidth = 0.25)
#         xfmt = ScalarFormatter(useOffset = False)
#         ax.xaxis.set_major_formatter(xfmt)
#         ax.xaxis.set_major_locator(plt.FixedLocator([200,500,1000,3000,5000]))
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
                meanPlot = plotLTAS(meanSpec, stdSpec, style, fs, frameSize, xaxis, plotSD, xlim, ylim)
                meanPlots.append(meanPlot[0])
                plt.title('LTAS, syllable: ' + xticklabelsObj[mrkNum], fontproperties=droidTitle)
                plt.legend(meanPlots, legendObjs, loc = 'best', prop=droidLegend)
        plt.show()
        
def centroidLTAS(meanSpec, fs):
    CENTROID = ess.Centroid(range=fs/2.0)
    centroid = CENTROID(meanSpec)
    return round(centroid, 2)
            
def plotLTAS(meanSpec, stdSpec, style, fs, frameSize, xaxis, plotSD, xlim, ylim, fitCoeffs = None, centroid = None):
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

# ---- plot slope 
        # if fitCoeffs != None:
#         estYVals = freqBins*fitCoeffs[0] + fitCoeffs[1]
#         fitPlot = plt.plot(freqBins, estYVals, style)  

# ----- plot centroid vertical
    # if centroid != None:
#         plt.axvline(centroid, color = style[0], linestyle = style[1], linewidth = 4)
#         plt.text(centroid, 1.5, str(int(centroid)), fontsize = labelFontsize, color = style[0])
        
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)

    if len(xlim) != 0:
        plt.xlim(xlim)
    else:
        plt.xlim(0, 20000)
        
    if len(ylim) != 0:
        plt.ylim(ylim)
    
    return meanPlot
