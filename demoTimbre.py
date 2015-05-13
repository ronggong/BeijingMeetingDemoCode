from featuresExtraction import *

# wav file
filename = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section_harmonics_pitchtrackCorrected.wav'

# syllable markers
# syllableFilename = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange2.txt'
syllableFilename = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-mrkRearrange.txt'


pitchTrack = 'daxp-Yu tang chun-Su San qi jie (Chi Xiaoqiiu)-section-words-pitchTrack.txt'

# one file, feature could be speccentroid or specloudness or specflux
plotFeatureSyllable(filename, syllableFilename = syllableFilename, pitchtrackFilename = pitchTrack,feature = 'speccentroid')