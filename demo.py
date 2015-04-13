##########################################################
# 1. -----melody extraction, harmonics, residual synthesis
from harmonicsSynthesis import *

filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section.wav'

#test1 is harmonicsSynthesis object
#harmonicsSynthesis(filename = aName, fs = fs, frameSize = frameSize, hopSize = hopSize)
test1 = harmonicsSynthesis(filename1)

# calculate spectrogram
test1.spectrogram()

# calculate melody by essentia's algorithm
test1.getMelody()

# save the melody to .txt 
# saveMelody(outputFilename = aName, unit = 'hz' or 'cents')
test1.saveMelody()	

# plot original spectrogram
fig = plt.figure(0)
test1.plotSpectrogram(fig = fig)

# synthesis of harmonics and residual, 
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

# one file, xaxis : 'linear' or 'log'
compareLPCSyllable(filename1, syllableFilename1, xaxis = 'log')

filename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section_harmonics_pitchtrackCorrected.wav'
syllableFilename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'

# write these files into a tuple, if we have 3 file to compare, we write filenames = (filename1, filename2, filename3)
filenames = (filename1, filename2)
syllableFilenames = (syllableFilename1, syllableFilename2)

# compare two files
compareLPCSyllable(filenames, syllableFilenames, xaxis = 'log')

################################################
# 4. -----compare LTAS, 1 to 3 files
from featuresExtraction import *

# wav file
filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section_harmonics_pitchtrackCorrected.wav'

# syllable markers
syllableFilename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section-words-mrkRearrange.txt'

# one file, xaxis : 'linear' or 'log'
singerName = ('Li Shengsu', )
compareLTAS(filename1, singerName = singerName, xaxis = 'log')

filename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section_harmonics_pitchtrackCorrected.wav'
syllableFilename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'

# write these filenames into a tuple, if we have 3 file to compare, we write filenames = (filename1, filename2, filename3)
filenames = (filename1, filename2)
syllableFilenames = (syllableFilename1, syllableFilename2)

# compare LTAS on whole file length, ignore the syllableFilenames arguments
singerName = ('Li Shengsu', 'Chi Xiaoqiu')
compareLTAS(filenames, singerName = singerName, xaxis = 'log')

# compare LTAS on syllable level once syllable file is available
# we don't need to give the singerName argument if syllableFilenames is given
# because syllableFilenames contains singerName in its first line
compareLTAS(filenames, syllableFilenames, xaxis = 'log')

###################################
# 5. -----plot features of an audio
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

# spectral loudness
test1.extractFeature('specloudness')

plt.figure(1)
test1.plotFeature()

# spectral flux
test1.extractFeature('specflux')

plt.figure(2)
test1.plotFeature()

plt.show()


