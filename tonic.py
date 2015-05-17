import matplotlib
matplotlib.use('Qt4Agg') # this is for ploting histogram

import utilFunctionsRong as UFR
import numpy as np
import sys, csv, os
import essentia as es
import essentia.standard as ess
import matplotlib.pyplot as plt
import intonation
import pickle

def cents2hz(do, name = ''):
    x = 261.626 * (2 ** (do / 1200))
    # if name != '':
#         with open('Tonics.txt', 'a') as f:
#             f.write(name + ' = ' + str(x) + '\n')
    return x

def pitch2letter(pitch):
    letters = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    letter = letters[pitch]
    return letter
    
def cents2pitch(cents, regDefault = 4):
    intPart = int(cents/100.00)
    remPart = cents%100.00
    if cents >= 0:
        if remPart > 50:
            intPart += 1
            remPart = remPart - 100
    else:
        if remPart > 50:
            remPart = remPart - 100
        else:
            intPart -= 1
    
    regAug = int(intPart/12) # register augmentation
    pitch = (intPart%12)
    #print pitch
    
    pitchLetter = pitch2letter(pitch)
    reg = regDefault + regAug
    
    if remPart >= 0:
        returnStr = pitchLetter + str(reg) + ' + ' + str(round(remPart,2)) + ' cents'
    else:
        returnStr = pitchLetter + str(reg) + ' - ' + str(round(abs(remPart),2)) + ' cents'

    return returnStr

def sortPeaksByAmp(peaks):
    freqPeaks = peaks[0]
    ampPeaks = peaks[1]
    indexSort = np.argsort(ampPeaks) # return index of sorted peak amplitude
    indexSort = indexSort[::-1]
    freqPeaksSort = freqPeaks[indexSort]
    ampPeaksSort = ampPeaks[indexSort]
    
    return (freqPeaksSort, ampPeaksSort)
    
def sortPeaksByFreq(peaks):
    freqPeaks = peaks[0]
    ampPeaks = peaks[1]
    indexSort = np.argsort(freqPeaks) # return index of sorted peak amplitude
    indexSort = indexSort[::-1]
    freqPeaksSort = freqPeaks[indexSort]
    ampPeaksSort = ampPeaks[indexSort]
    
    return (freqPeaksSort, ampPeaksSort)

def neiWaiZhuFromPeaks(peaks, shengqiang = 'xipi'):

    rtuple = sortPeaksByAmp(peaks)
    freqPeaksSort = rtuple[0]
    
    print freqPeaksSort
    
    if shengqiang == 'xipi':
        nn = 4
    elif shengqiang == 'erhuang':
        nn = 3
    
    freqPeaksFirst4 = freqPeaksSort[:nn] # take the first 4 peaks
    freqPeaksFirst4Sort = np.sort(freqPeaksFirst4)
    
    print freqPeaksFirst4Sort
    
    neiCents = freqPeaksFirst4Sort[0]
    
    # neiCents = peakFreq[0] 
#     # neiCents = -146.12
    neiHz = cents2hz(neiCents)
# 
    waiIndex = np.argmin(abs(neiCents + 700 - freqPeaks))
    waiCents = freqPeaks[waiIndex]
#     waiCents = peakFreq[4]
#     # waiCents = -267.14
    waiHz = cents2hz(waiCents)
#     
    if shengqiang == 'xipi':
        # xipi
        zhuyinIndex = np.argmin(abs(neiCents + 300 - freqPeaks))
        zhuyinCents = freqPeaks[zhuyinIndex]
#         zhuyinCents = peakFreq[2]
        zhuyinHz = cents2hz(zhuyinCents)
    elif shengqiang == 'erhuang':
        # erhuang
        zhuyinIndex = np.argmin(abs(neiCents + 500 - freqPeaks))
        zhuyinCents = freqPeaks[zhuyinIndex]
#         zhuyinCents = peakFreq[3]
        zhuyinHz = cents2hz(zhuyinCents)
    
    return (neiCents, neiHz, waiCents, waiHz, zhuyinCents-1200, zhuyinHz/2.0)

def zhuyinConfidenceJinghuAlgo(peaks, shengqiang = 'xipi'):
    rtuple = sortPeaksByAmp(peaks)
    freqPeaks = rtuple[0]
    ampPeaks = rtuple[1]
    
    freqThreshold = 50 # freq threshold 50 Hz
    
    if shengqiang == 'xipi':
        neiDis = 300
        waiDis = 400
    elif shengqiang == 'erhuang':
        neiDis = 500
        waiDis = 200
    
    zhuCand = [] # zhuyin candidates
    zhuConf = [] # zhuyin confidences
    
    for ii in range(len(freqPeaks)):
        zhu = freqPeaks[ii]
        conf = 0
        
        neiFound = False
        waiFound = False
        for jj in range(len(freqPeaks)):
            if zhu - neiDis - freqThreshold < freqPeaks[jj] and zhu - neiDis + freqThreshold > freqPeaks[jj]:
                neiFound = True
                conf += ampPeaks[jj]
            if zhu + waiDis - freqThreshold < freqPeaks[jj] and zhu + waiDis + freqThreshold > freqPeaks[jj]:
                waiFound = True
                conf += ampPeaks[jj]
        
        if neiFound or waiFound:
            conf += ampPeaks[ii]
            zhuCand.append(zhu)
            zhuConf.append(conf)
            
    if len(zhuCand) != 0:    
        zhuArray = np.array([np.array(zhuCand), np.array(zhuConf)])
        rtuple = sortPeaksByAmp(zhuArray)
    else:
        rtuple = None
    
    return rtuple

def zhuyinConfidenceSingingAlgo(peaks, penalize = True):
    rtuple = sortPeaksByAmp(peaks)
    freqPeaks = rtuple[0]
    ampPeaks = rtuple[1]
    
    freqThreshold = 50 # freq threshold 50 Hz
    
    disUpScores = [200, 400, 700, 900, 1200]
    disUpPenalise = [500, 1100, 100, 300, 600, 800, 1000]
    
    disDownScores = [300, 500, 800, 1000, 1200]
    disDownPenalise = [100, 700, 200, 400, 600, 900, 1100]
    
    zhuCand = [] # zhuyin candidates
    zhuConf = [] # zhuyin confidences
    
    for ii in range(len(freqPeaks)):
        zhu = freqPeaks[ii]
        conf = 0
        
        for jj in range(len(freqPeaks)):
            for upScore in disUpScores:
                if (zhu + upScore - freqThreshold < freqPeaks[jj] and zhu + upScore + freqThreshold > freqPeaks[jj]):
                    conf += ampPeaks[jj]
                    
            for downScore in disDownScores:
                if (zhu - downScore - freqThreshold < freqPeaks[jj] and zhu - downScore + freqThreshold > freqPeaks[jj]):
                    conf += ampPeaks[jj]
            
            if penalize == True:
                # penalize the confidence
                for upP in disUpPenalise:
                    if (zhu + upP - freqThreshold < freqPeaks[jj] and zhu + upP + freqThreshold > freqPeaks[jj]):
                        conf -= ampPeaks[jj]
                    
                for downP in disDownPenalise:
                    if (zhu - downP - freqThreshold < freqPeaks[jj] and zhu - downP + freqThreshold > freqPeaks[jj]):
                        conf -= ampPeaks[jj]
        
        if conf > 0:
            conf += ampPeaks[ii]
            zhuCand.append(zhu)
            zhuConf.append(conf)
            
    if len(zhuCand) != 0:    
        zhuArray = np.array([np.array(zhuCand), np.array(zhuConf)])
        rtuple = sortPeaksByAmp(zhuArray)
    else:
        rtuple = None
    
    return rtuple
    
def partExtraction(audioFilename, segFilename, fs, part):
    '''extract the segmentation part'''
    
    rtuple = UFR.readYileSegmentation(segFilename)

    startT = rtuple[0]
    dur = rtuple[1]
    segNum = rtuple[2]
    segMarker = rtuple[3]
    
     #----extract the jinghu parts
    partST = [] # start time
    partDur = [] # part duration
    for ii in range(len(segNum)):
        if segMarker[ii] == part:
            partST.append(startT[ii])
            partDur.append(dur[ii])
        
        
    partST = np.array(partST, dtype = np.float32)
    partDur = np.array(partDur, dtype = np.float32)
    partET = np.array(partST + partDur)

    #----concatenate jinghu parts
    audio = ess.MonoLoader(filename = audioFilename, sampleRate=fs)()
    partAudio = np.array([])
    for ii in range(len(partST)):
        partSegSS = int(partST[ii] * fs)
        partSegES = int(partET[ii] * fs) if partET[ii] * fs <=  len(audio) else len(audio)
        partAudio = np.concatenate((partAudio, audio[partSegSS:partSegES]), axis=0)

    partAudio = partAudio.astype(np.float32)
    
    return partAudio

def timeVec(pitch, fs, hopSize, frameSize):
    n_frames = len(pitch)
    # hopSize time in (s)
    hopOccupyTime = float(hopSize) / float(fs)
    frameSizeOccupyTime = float(frameSize) / float(fs)

    # time vector in (s)
    xTimeVec = np.asarray(range(n_frames), dtype = np.float32) * \
               hopOccupyTime + frameSizeOccupyTime/2.0
               
    return xTimeVec

def remOctaveZhu(zhuConf):
    '''remove the octave peak which has smaller amplitude'''
    zhu = zhuConf[0]
    conf = zhuConf[1]
    
    if len(zhu) >= 2:
        ind2rem = []
        threshold = 25
        for ii in range(len(zhu)-1):
            for jj in range(len(zhu)):
                if jj > ii and abs(zhu[ii] - zhu[jj]) < 1200 + threshold and abs(zhu[ii] - zhu[jj]) > 1200 - threshold:
                    if conf[ii] >= conf[jj]:
                        ind2rem.append(jj)
                    else:
                        ind2rem.append(ii)
        if len(ind2rem) > 0:
            ind2rem.sort()
            ind2rem = ind2rem[::-1]
            # print ind2rem, len(zhu)
        
            for ii in ind2rem:
                zhu = np.delete(zhu, ii)
                conf = np.delete(conf, ii)
    return (zhu, conf)
    
fontSize = 16
fs = 44100

def deOctave(zhuConf, centsThreshold):
    zhu = zhuConf[0]
    conf = zhuConf[1]
    
    for ii in range(len(zhu)):
        if zhu[ii] > centsThreshold:
            zhu[ii] = zhu[ii] - 1200
            
    return (zhu, conf)

def remSingingNotInJinghu(jinghu, singing):
    '''remove singing Zhuyin not in Jinghu Zhuyin'''
    jinghuZhu = jinghu[0]
    singingZhu = singing[0]
    singingConf = singing[1]
    
    ind2rem =[]
    threshold = 25
    
    for ii in range(len(singingZhu)):
        found = False
        found
        for jj in range(len(jinghuZhu)):
            if abs(singingZhu[ii] - jinghuZhu[jj]) < threshold:
                found = True
        
        if found == False:
            ind2rem.append(ii)
    
    # for ii in range(len(jinghuZhu)):
#         found = False
#         for jj in range(len(singingZhu)):
#             if abs(singingZhu[jj] - jinghuZhu[ii]) < threshold:
#                 found = True
    
    ind2rem.sort()
    ind2rem = ind2rem[::-1]
    
    for ii in ind2rem:
        singingZhu = np.delete(singingZhu, ii)
        singingConf = np.delete(singingConf, ii)
    
    return (singingZhu, singingConf)

def mergeJinghuSinging(jinghuTuple, singingTuple):
    jinghuZhu = jinghuTuple[0]
    jinghuConf = jinghuTuple[1]
    
    singingZhu = singingTuple[0]
    singingConf = singingTuple[1]
    
    threshold = 25
    mergeZhu = []
    mergeConf = []
    for ii in range(len(jinghuZhu)):
        maxConf = 0
        zhu = jinghuZhu[ii]
        for jj in range(len(singingZhu)):
            if abs(jinghuZhu[ii] - singingZhu[jj]) < threshold:
                # find the same Zhuyin in Jinghu and Singing with biggest amplitude
                if singingConf[jj] > maxConf:
                    maxConf = singingConf[jj]
                    zhu = singingZhu[jj]
                    
        mergeZhu.append((jinghuZhu[ii] + zhu)/2.0)
        mergeConf.append(jinghuConf[ii] * maxConf)
    mergeZhu = np.array(mergeZhu)
    mergeConf = np.array(mergeConf)
    return [mergeZhu, mergeConf]
    
# audioFilename = './tonic/daxp-Yu tang chun-Su San qi jie (Li Shengsu).wav'
# segFilename = './tonic/segAnnotationVJP_Li.txt'

def neiWaiZhu(audioFilename, segFilename, shengqiang = 'xipi', penalize = True, plotHisto = False, writeAudio = False):
    if not any(shengqiang in sq for sq in ('xipi', 'erhuang')):
        print 'shengqiang is not xipi or erhuang, quit program.'
        return

    jinghuAudio = partExtraction(audioFilename, segFilename, fs, 'J')
    singingAudio = partExtraction(audioFilename, segFilename, fs, 'V')

    #----extract melody jinghu
    hopSize = 128
    frameSize = 2048
    guessUnvoiced = True # read the algorithm's reference for more details
    
    if shengqiang == 'xipi':
        minFrequency = 349 #F4
        maxFrequency = 987.7 #B5
    elif shengqiang == 'erhuang':
        minFrequency = 349 #C4
        maxFrequency = 987.7 #B5
    
    jinghu_run_predominant_melody = ess.PredominantMelody(guessUnvoiced=guessUnvoiced,
                                               frameSize=frameSize,
                                               hopSize=hopSize,
                                               voicingTolerance = 1.4,
                                               minFrequency = minFrequency,
                                               maxFrequency = maxFrequency)

    # Load audio file, apply equal loudness filter, and compute predominant melody
    print 'extracting melody jinghu... ...'
    jinghuAudioEL = ess.EqualLoudness()(jinghuAudio)
    jinghuPitch, jinghuConfidence = jinghu_run_predominant_melody(jinghuAudioEL)
    
    #-----extract melody singing
    if shengqiang == 'xipi':
        minFrequency = 98 #F4
        maxFrequency = 987.7 #B5
    elif shengqiang == 'erhuang':
        minFrequency = 98 #C4
        maxFrequency = 987.7 #B5
        
    singing_run_predominant_melody = ess.PredominantMelody(guessUnvoiced=guessUnvoiced,
                                               frameSize=frameSize,
                                               hopSize=hopSize,
                                               voicingTolerance = 1.4,
                                               minFrequency = minFrequency,
                                               maxFrequency = maxFrequency)
                                               
    print 'extracting melody singing... ...'
    singingAudioEL = ess.EqualLoudness()(singingAudio)
    singingPitch, singingConfidence = singing_run_predominant_melody(singingAudioEL)
    
    jinghuTimeVec = timeVec(jinghuPitch, fs, hopSize, frameSize)
    singingTimeVec = timeVec(singingPitch, fs, hopSize, frameSize)

    # for saving the pitch
    # np.savetxt(audioFilename[:-4]+'-pitchtrack.txt', np.vstack((xTimeVec, pitch)).T)

    # find histogram peaks, to plot
    # print plotHisto
    
    jinghuPeaks = UFR.hz2centsRafa(jinghuTimeVec, jinghuPitch, plotHisto = plotHisto)
    singingPeaks = UFR.hz2centsRafa(singingTimeVec, singingPitch, plotHisto = plotHisto)
    
    # print singingPeaks
    
#     rtuple = neiWaiZhuFromPeaks(peaks, shengqiang)
    
    # neiCents = rtuple[0]
#     neiHz = rtuple[1]
#     waiCents = rtuple[2]
#     waiHz = rtuple[3]
#     zhuyinCents = rtuple[4]
#     zhuyinHz = rtuple[5]

    jinghuRtuple = zhuyinConfidenceJinghuAlgo(jinghuPeaks, shengqiang)
    jinghuRtuple = remOctaveZhu(jinghuRtuple)
    jinghuRtuple = deOctave(jinghuRtuple, 1000) # this will be written
    
    #---- penalize
    singingRtuple = zhuyinConfidenceSingingAlgo(singingPeaks, True)
    singingRtuple = remOctaveZhu(singingRtuple)
    singingRtuplePenalized = deOctave(singingRtuple, 1000) # this will be written
    
    # only keep overlapped part
    singingRtuplePost = remSingingNotInJinghu(jinghuRtuple, singingRtuplePenalized)
    jinghuRtuplePost = remSingingNotInJinghu(singingRtuplePenalized, jinghuRtuple)
    
    singingRtuplePost = sortPeaksByFreq(singingRtuplePost)
    jinghuRtuplePost = sortPeaksByFreq(jinghuRtuplePost)
    
    rtuple = mergeJinghuSinging(jinghuRtuplePost, singingRtuplePost)
    zhuSortPenalized = sortPeaksByAmp(rtuple) # this will be written
    
    #---- not penalize in singing algo
    singingRtuple = zhuyinConfidenceSingingAlgo(singingPeaks, False)
    singingRtuple = remOctaveZhu(singingRtuple)
    singingRtupleNoPenalized = deOctave(singingRtuple, 1000) # this will be written
    
    # only keep overlapped part
    singingRtuplePost = remSingingNotInJinghu(jinghuRtuple, singingRtupleNoPenalized)
    jinghuRtuplePost = remSingingNotInJinghu(singingRtupleNoPenalized, jinghuRtuple)
    
    singingRtuplePost = sortPeaksByFreq(singingRtuplePost)
    jinghuRtuplePost = sortPeaksByFreq(jinghuRtuplePost)
    
    rtuple = mergeJinghuSinging(jinghuRtuplePost, singingRtuplePost)
    zhuSortNoPenalized = sortPeaksByAmp(rtuple) # this will be written
    
    
    # print results
    print audioFilename
    # print jinghuRtuple
#     print singingRtuple
    print zhuSortPenalized
    
    # write result to text
    with open(audioFilename[:-4] + '-tonic.txt', "w") as f:
        f.write(audioFilename+'\n')
        # f.write('nei: ' + str(round(neiHz,2)) + ' Hz (' + cents2pitch(neiCents) + ')\n')
#         f.write('wai: ' + str(round(waiHz,2)) + ' Hz (' + cents2pitch(waiCents) + ')\n')
        f.write('\nMerged Zhuyin with penalized singing algo:\n')
        for ii in range(len(zhuSortPenalized[0])):
            zhuyinCents = zhuSortPenalized[0][ii]
            zhuyinHz = cents2hz(zhuyinCents)
            f.write(str(round(zhuyinHz,2)) + '\tHz (' + cents2pitch(zhuyinCents) + ')\t' + str(zhuSortPenalized[1][ii]) + '\n')
        
        f.write('\nMerged Zhuyin without penalized singing algo:\n')
        for ii in range(len(zhuSortNoPenalized[0])):
            zhuyinCents = zhuSortNoPenalized[0][ii]
            zhuyinHz = cents2hz(zhuyinCents)
            f.write(str(round(zhuyinHz,2)) + '\tHz (' + cents2pitch(zhuyinCents) + ')\t' + str(zhuSortNoPenalized[1][ii]) + '\n')
        
        f.write('\njinghu Zhuyin algorithm results:\n')
        for ii in range(len(jinghuRtuple[0])):
            zhuyinHz = cents2hz(jinghuRtuple[0][ii])
            f.write(str(round(zhuyinHz,2)) + '\tHz (' + cents2pitch(jinghuRtuple[0][ii]) + ')\t' + str(jinghuRtuple[1][ii]) + '\n')
            
        f.write('\nsinging Zhuyin algorithm with penalized results:\n')
        for ii in range(len(singingRtuplePenalized[0])):
            zhuyinHz = cents2hz(singingRtuplePenalized[0][ii])
            f.write(str(round(zhuyinHz,2)) + '\tHz (' + cents2pitch(singingRtuplePenalized[0][ii]) + ')\t' + str(singingRtuplePenalized[1][ii]) + '\n')
        
        f.write('\nsinging Zhuyin algorithm without penalized results:\n')
        for ii in range(len(singingRtupleNoPenalized[0])):
            zhuyinCents = singingRtupleNoPenalized[0][ii]
            zhuyinHz = cents2hz(zhuyinCents)
            f.write(str(round(zhuyinHz,2)) + '\tHz (' + cents2pitch(zhuyinCents) + ')\t' + str(singingRtupleNoPenalized[1][ii]) + '\n')

    
#     print ('nei: ' + str(round(neiHz,2)) + ' Hz (' + cents2pitch(neiCents) + ')')
#     print ('wai: ' + str(round(waiHz,2)) + ' Hz (' + cents2pitch(waiCents) + ')')
#     print ('zhuyin: ' + str(round(zhuyinHz,2)) + ' Hz (' + cents2pitch(zhuyinCents) + ')')
    


    # if writeAudio == True:
#         AUDIOWRITER = ess.MonoWriter(filename = audioFilename[:-4] + '-jinghu.wav')
#         AUDIOWRITER(jinghuAudio)

    # plot part
#     fig = plt.figure()
	# ax = fig.add_subplot(111)
	# plt.plot(np.arange(jinghuAudio.size)/float(fs), jinghuAudio)
	# plt.autoscale(tight=True)
	# plt.ylabel('Amplitude', fontsize = fontSize)    
	# plt.xlabel('Time (s)', fontsize = fontSize)
	# xLim = ax.get_xlim()
	# yLim = ax.get_ylim()
	# ax.set_aspect((xLim[1]-xLim[0])/(4.0*(yLim[1]-yLim[0])))    
	# plt.show()
	# 
	# ax = fig.add_subplot(111)
	# plt.autoscale(tight=True)
	# plt.plot(xTimeVec, pitch)
	# plt.ylabel('Freq (Hz)', fontsize = fontSize)    
	# plt.xlabel('Time (s)', fontsize = fontSize)
	# xLim = ax.get_xlim()
	# yLim = ax.get_ylim()
	# ax.set_aspect((xLim[1]-xLim[0])/(4.0*(yLim[1]-yLim[0])))    

    # plot pitch track
    # ax = fig.add_subplot(111)
#     plt.plot(xTimeVec, pitch)
#     plt.show()
