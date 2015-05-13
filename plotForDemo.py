import matplotlib.pyplot as plt
import utilFunctionsRong as UFR
import numpy as np

titleFontSize = 25
labelFontsize = 14
legendFontsize = 12
markersize = 15

#----spectral centroid
fig, ax = plt.subplots()

color = 'r'
width = 0.35
ind1 = np.arange(4)*width
sylMeans = (2469.01, 2418.73, 2577.81, 2481.53)
sylStds = (364, 336, 360, 353)
#xticklabels1 = ('ssqj\nLSs', 'fhc\nLYf', 'fhc\nSYh', 'fhc\nLSs','Mei\nmean')
xticklabels1 = ('Li Yufu', 'Shi Yihong', 'Li Shengsu','Mei mean')
bar1 = plt.bar(ind1, sylMeans, width, color=color, yerr = sylStds, error_kw=dict(ecolor='k'))
bar3 = plt.bar(ind1[3], sylMeans[3], width, color=color, yerr = sylStds[3], 
error_kw=dict(ecolor='k'), hatch = '\\')

# UFR.autolabelBar(bar1, ax)

color = 'y'
ind2 = np.arange(4)*width + width*6
sylMeans = (2022.63, 2309.16, 2396.71, 2242.83)
sylStds = (460, 510, 477, 482)
xticklabels2=('Chi Xiaoqiu', 'Li Peihong','Liu Guijuan','Cheng mean')
bar2 = plt.bar(ind2, sylMeans, width, color=color, yerr = sylStds, error_kw=dict(ecolor='k'))
bar4 = plt.bar(ind2[3], sylMeans[3], width, color=color, yerr = sylStds[3], 
error_kw=dict(ecolor='k'), hatch = '\\')
# UFR.autolabelBar(bar2, ax) # no auto label

plt.xticks(np.concatenate((ind1,ind2),0), xticklabels1+xticklabels2, rotation=45, fontsize = labelFontsize)
plt.legend((bar1[0], bar2[0]), ('Mei', 'Cheng'), loc = 'best', prop={'size':legendFontsize})
plt.ylabel('Frequency (Hz)', fontsize = labelFontsize)
# plt.title('Spectral Centroid')
plt.show()

#----Loudness
fig, ax = plt.subplots()

color = 'r'
width = 0.35
ind1 = np.arange(4)*width
sylMeans = (0.35, 0.27, 0.25, 0.29)
# xticklabels1 = ('ssqj\nLSs', 'fhc\nLYf', 'fhc\nSYh', 'fhc\nLSs','Mei\nmean')
xticklabels1 = ('Li Yufu', 'Shi Yihong', 'Li Shengsu','Mei mean')
bar1 = plt.bar(ind1, sylMeans, width, color=color)
bar3 = plt.bar(ind1[3], sylMeans[3], width, color=color, hatch = '\\')
# UFR.autolabelBar(bar1, ax)

color = 'y'
ind2 = np.arange(4)*width + width*6
sylMeans = (0.41, 0.46, 0.34, 0.40)
# xticklabels2=('ssqj\nCXq', 'sln\nCXq', 'sln\nLPh','sln\nLGj','Cheng\nmean')
xticklabels2=('Chi Xiaoqiu', 'Li Peihong','Liu Guijuan','Cheng mean')
bar2 = plt.bar(ind2, sylMeans, width, color=color)
bar4 = plt.bar(ind2[3], sylMeans[3], width, color=color, hatch = '\\')
# UFR.autolabelBar(bar2, ax) # no auto label

plt.xticks(np.concatenate((ind1,ind2),0), xticklabels1+xticklabels2, rotation=45, fontsize = labelFontsize)
plt.legend((bar1[0], bar2[0]), ('Mei', 'Cheng'), loc = 'best', prop={'size':legendFontsize})
plt.ylabel('Loudness standard deviation', fontsize = labelFontsize)
# plt.title('Loudness')
plt.show()

#----spectral flux
color = 'r'
width = 0.35
ind1 = np.arange(4)*width
sylMeans = (0.121, 0.123, 0.121, 0.122)
sylStds = (0.072, 0.065, 0.057, 0.065)
# xticklabels1 = ('ssqj\nLSs', 'fhc\nLYf', 'fhc\nSYh', 'fhc\nLSs','Mei\nmean')
xticklabels1 = ('Li Yufu', 'Shi Yihong', 'Li Shengsu','Mei mean')
#bar1 = plt.bar(ind1, sylMeans, width, color=color, yerr = sylStds)
bar1 = plt.bar(ind1, sylMeans, width, color=color)
bar3 = plt.bar(ind1[3], sylMeans[3], width, color=color, hatch = '\\')

# UFR.autolabelBar(bar1, ax)

color = 'y'
ind2 = np.arange(4)*width + width*6
sylMeans = (0.085, 0.102, 0.091, 0.093)
sylStds = (0.062, 0.069, 0.053, 0.061)
# xticklabels2=('ssqj\nCXq', 'sln\nCXq', 'sln\nLPh','sln\nLGj','Cheng\nmean')
xticklabels2=('Chi Xiaoqiu', 'Li Peihong','Liu Guijuan','Cheng mean')
# bar2 = plt.bar(ind2, sylMeans, width, color=color, yerr = sylStds)
bar2 = plt.bar(ind2, sylMeans, width, color=color)
bar4 = plt.bar(ind2[3], sylMeans[3], width, color=color, hatch = '\\')

plt.xticks(np.concatenate((ind1,ind2),0), xticklabels1+xticklabels2, rotation = 45, fontsize = labelFontsize)
plt.legend((bar1[0], bar2[0]), ('Mei', 'Cheng'), loc = 'best', prop={'size':legendFontsize})
plt.ylabel('Spectral Flux', fontsize = labelFontsize)
# plt.title('Spectral Flux')
plt.show()