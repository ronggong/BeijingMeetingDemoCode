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
# 2. -----compare timbre of two singing schools on syllable level
from featuresExtraction import *

# wav file
filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section_harmonics_pitchtrackCorrected.wav'

# syllable markers
syllableFilename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section-words-mrkRearrange.txt'

filename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section_harmonics_pitchtrackCorrected.wav'
syllableFilename2 = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'

# feature could be speccentroid or specloudness or specflux
compareFeaturesSyllableMean(filename1, syllableFilename1, filename2, syllableFilename2, feature = 'speccentroid')

###################################
# 3. -----plot features of an audio
from featuresExtraction import *

filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section.wav'

test1 = FeaturesExtraction(filename1);

# calculate spectrogram
test1.spectrogram()

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

###################################################
# 4. -----plot mean feature value on syllable level
from featuresExtraction import *

filename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section_harmonics_pitchtrackCorrected.wav'

syllableFilename1 = 'daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section-words-mrkRearrange.txt'

test2 = FeaturesExtractionSyllable(filename1, syllableFilename1)

test2.spectrogram()

# calculate feature, speccentroid or specloudness or specflux
test2.extractFeature('speccentroid')

# calculate mean, standard deviation of feature value on syllable level
test2.meanStdSyllable()

# plot
plt.figure(0)
test2.plotFeatureSyllable()

plt.show()

