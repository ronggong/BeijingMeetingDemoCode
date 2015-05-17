# -*- coding: utf-8 -*-

##########################################################
# 1. -----melody extraction, harmonics, residual synthesis
from harmonicsSynthesis import *

filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section.wav'

# 1) test1 is harmonicsSynthesis object
#harmonicsSynthesis(filename = aName, fs = fs, frameSize = frameSize, hopSize = hopSize)
test1 = harmonicsSynthesis(filename1)

# 2) calculate spectrogram
test1.spectrogram()

## 3 - 1) calculate melody by essentia's algorithm
test1.getMelody()

## save the melody to .txt 
# saveMelody(outputFilename = aName, unit = 'hz' or 'cents')
test1.saveMelody()

## 3 - 2) or load melody from .txt, melody is exported from Melodia, unit should be 'hz'
# you can correct pitch track manually, then use it here
# jump step 3 - 1)
pitchTrackMelodia = 'Li Shengsu-pitchtrackMelodia.txt'
test1.loadMelody(pitchTrackMelodia)

# plot original spectrogram
fig = plt.figure(0)
test1.plotSpectrogram(fig = fig)

# 4) synthesis of harmonics and residual, 
# synthesis(harmonicsOutputFilename = aName1, residualOutputFilename = aName2):
test1.synthesis()

# plot harmonics or residual
# plotSpectrogram(fig, spectrogram = 'original' or 'harmonics' or 'residual'):
fig = plt.figure(1)
test1.plotSpectrogram(fig, 'harmonics')

plt.show()
 
#################################################################
# 2. -----compare timbre of singing schools on syllable level, 1 to 3 files
from featuresExtraction import *

# wav file
filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section_harmonics_pitchtrackCorrected.wav'

# syllable markers
syllableFilename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section-words-mrkRearrange.txt'

# one file, feature could be speccentroid or specloudness or specflux
compareFeaturesSyllableMean(filename1, syllableFilename1, feature = 'speccentroid')

filename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section_harmonics_pitchtrackCorrected.wav'
syllableFilename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'

# write these files into a tuple, if we have 3 file to compare, we write filenames = (filename1, filename2, filename3)
filenames = (filename1, filename2)
syllableFilenames = (syllableFilename1, syllableFilename2)

# compare 2 files
compareFeaturesSyllableMean(filenames, syllableFilenames, feature = 'speccentroid')

################################################
# 3. -----compare LPC envelope on syllable level, 1 to 3 files
from featuresExtraction import *

# wav file
filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section_harmonics_pitchtrackCorrected.wav'

# syllable markers
syllableFilename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section-words-mrkRearrange.txt'

# one file, xaxis : 'linear' or 'log', LPC order default: 10, choose interval [1 50]
compareLPCSyllable(filename1, syllableFilename1, lpcorder = 10, xaxis = 'log', xlim = [0, 20000], ylim = [-40, 45])

filename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section_harmonics_pitchtrackCorrected.wav'
syllableFilename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'

# write these files into a tuple, if we have 3 file to compare, we write filenames = (filename1, filename2, filename3)
filenames = (filename1, filename2)
syllableFilenames = (syllableFilename1, syllableFilename2)

# compare two files
compareLPCSyllable(filenames, syllableFilenames, lpcorder = 10, xaxis = 'log')

################################################
# 4. -----compare LTAS, 1 to 3 files
from featuresExtraction import *

# wav file
filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section_harmonics_pitchtrackCorrected.wav'

# syllable markers
syllableFilename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section-words-mrkRearrange.txt'

# one file, xaxis : 'linear' or 'log', plotSD: plot Standard deviation for LTAS
# remember adding the uni-8 code declaration: # -*- coding: utf-8 -*- at the beginning of this file
singerName = (u"李胜素(Li Shengsu)", )
compareLTAS(filename1, singerName = singerName, xaxis = 'linear', plotSD = True, xlim = [0, 10000], ylim = [-80, 10])

filename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section_harmonics_pitchtrackCorrected.wav'
syllableFilename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'

# write these filenames into a tuple, if we have 3 file to compare, we write filenames = (filename1, filename2, filename3)
filenames = (filename1, filename2)
syllableFilenames = (syllableFilename1, syllableFilename2)

# compare LTAS on whole file length, ignore the syllableFilenames arguments
singerName = (u"李胜素(Li Shengsu)", u"迟小秋(Chi Xiaoqiu)")
compareLTAS(filenames, singerName = singerName, xaxis = 'linear', plotSD = True)

# compare LTAS on syllable level once syllable file is available
# we don't need to give the singerName argument if syllableFilenames is given
# because syllableFilenames contains singerName in its first line
compareLTAS(filenames, syllableFilenames, xaxis = 'linear', plotSD = True)

###################################
# 5. -----plot features of syllables
from featuresExtraction import *

# wav file
filename = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section_harmonics_pitchtrackCorrected.wav'

# syllable markers
syllableFilename = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section-words-mrkRearrange.txt'

# one file, feature could be speccentroid or specloudness or specflux
plotFeatureSyllable(filename, syllableFilename, feature = 'speccentroid')

###################################
# 6. -----plot features and pitchtrack
from featuresExtraction import *

# wav file
filename = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section_harmonics_pitchtrackCorrected.wav'

# syllable markers
syllableFilename = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'

# pitchtrack
pitchtrack = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-pitchTrack.txt'

# one file, feature could be speccentroid or specloudness or specflux
plotFeatureSyllable(filename, syllableFilename = syllableFilename, pitchtrackFilename = pitchtrack,feature = 'speccentroid')

###################################
# 7. -----plot features of an audio
from featuresExtraction import *

filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section_harmonics_pitchtrackCorrected.wav'

test1 = FeaturesExtraction(filename1);

# calculate spectrogram
mX = test1.spectrogram()

# calculate spectral centroid
test1.extractFeature('speccentroid')

# draw spectral centroid vs. time
plt.figure(0)
test1.plotFeature()

# spectral loudness, normTo: norm loudness mean to 0.5, default not normalize
test1.extractFeature('specloudness', normTo = 0.5)

plt.figure(1)
test1.plotFeature()

# spectral flux, normTo: norm loudness mean to 0.5, default not normalize
test1.extractFeature('specflux', normTo = 0.5)

plt.figure(2)
test1.plotFeature()

# tristimulus
text1.extractFeature('tristimulus')

plt.figure(3)
test1.plotFeature()

plt.show()

###################################
# 8. -----tonic (first degree) identification
import tonic
from os import listdir
from os.path import isfile, join

audioFilename = './tonic/01 九江口：看夕阳照枫林红似血染.mp3'
segFilename = './tonic/01 九江口：看夕阳照枫林红似血染.txt'

# shengqiang: 'erhuang' or 'xipi'
tonic.neiWaiZhu(audioFilename, segFilename, shengqiang = 'erhuang', plotHisto = True)


