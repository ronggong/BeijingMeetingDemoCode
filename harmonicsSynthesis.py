import sys, csv, os
import essentia as es
import essentia.standard as ess
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import get_window
from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft, fftshift
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # resize colorbar

eps = np.finfo(np.float).eps

path2SmsTools = 'sms-models'
sys.path.append(path2SmsTools)

import utilFunctions as UF
import utilFunctionsRong as UFR
import harmonicModel as HM
import dftModel as DFT

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

class harmonicsSynthesis(object):
    
    def __init__(self, filename, fs = 44100, frameSize = 2048, hopSize = 128):
        self.inputFilename = filename
        self.spectrograms = ['original', 'harmonics', 'residual']
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.fs = fs
        self.audio = ess.MonoLoader(filename = filename, sampleRate = fs)()
        self.audioEL = ess.EqualLoudness()(self.audio)
        self.mX = []	# the spectrogram
        self.yho = []
        self.xro = []
        self.featureVec = []
        self.pitch = []
        
    def spectrogram(self):
        winAnalysis = 'hann'
        N = 2 * self.frameSize	# padding frameSize
        SPECTRUM = ess.Spectrum(size=N)
        WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N-self.frameSize) 
        
        print '\ncalculating spectrogram ... ...'
        mX = []
        for frame in ess.FrameGenerator(self.audioEL, frameSize=self.frameSize, hopSize=self.hopSize):
            frame = WINDOW(frame)
            mXFrame = SPECTRUM(frame)
            #store/append values per frame in an array
            mX.append(mXFrame)
        
        self.mX = mX
        print "spectrogram calculation done, return " + str(len(self.mX)) + ' frames. \n'
        return self.mX
    
    def getMelody(self):
        if len(self.mX) == 0:
            print 'please do spectrogram analysis at first, then do getMelody.'
            return
        minf0 = 200
        maxf0 = 1000
        binResolution = 10.0
        guessUnvoiced = False    #to not detect unvoiced (non predominant voice) segments
        winAnalysis = 'hann'
        t = -80 

        #init objects
        SPECPEAKS = ess.SpectralPeaks(minFrequency=50, 
                                maxFrequency=10000, 
                                maxPeaks=100, 
                                sampleRate=self.fs, 
                                magnitudeThreshold= t,
                                orderBy="magnitude")
        PITCHSALIENCE = ess.PitchSalienceFunction(magnitudeThreshold=60, binResolution = binResolution)
        SALIENCEPEAKS = ess.PitchSalienceFunctionPeaks(minFrequency=minf0, maxFrequency=maxf0)
        PITCHCONTOURS = ess.PitchContours(hopSize=self.hopSize, peakFrameThreshold=0.7)
        COMPUTEMELODY = ess.PitchContoursMelody(guessUnvoiced=guessUnvoiced, hopSize=self.hopSize)

        #array to store pitch values
        salBinsArr = []
        salMagsArr = []

        #loop starts for every audio frame
        print 'extracting melody ... ...'
        for mXFrame in self.mX:
    
            mXFrameDB = 20*np.log10(mXFrame+eps)
            pFreq, pMags = SPECPEAKS(mXFrameDB)
            pMags = np.power(10, pMags/20.0)
    
            salFrame = PITCHSALIENCE(pFreq, pMags)
            salBins, salMags = SALIENCEPEAKS(salFrame)
            salBinsArr.append(salBins.tolist())
            salMagsArr.append(salMags.tolist())
 
        cBins, cSals, cBounds, dur = PITCHCONTOURS(salBinsArr, salMagsArr)
        pitch, confidence = COMPUTEMELODY(cBins, cSals, cBounds, dur)
        self.pitch = pitch
        print "Melody extraction done, return " + str(np.size(pitch)) + ' values.\n'
        return self.pitch
    
    def saveMelody(self, outputFilename = None, unit = 'hz'):
        if len(self.pitch) == 0:
            print 'please do getMelody at first, then do saveMelody.'
            return
        elif unit != 'hz' and unit != 'cents':
            print 'the output melody unit should be either hz or cents.'
            return
        if outputFilename == None:
            outputFilename = self.inputFilename[:-4] + '-pitch.txt'
            
        n_frames = len(self.pitch)
        # hopSize time in (s)
        hopOccupyTime = float(self.hopSize) / float(self.fs)
        frameSizeOccupyTime = float(self.frameSize) / float(self.fs)

        # time vector in (s)
        xTimeVec = np.asarray(range(n_frames), dtype = np.float32) * \
                   hopOccupyTime + frameSizeOccupyTime/2.0

        if unit == 'cents':
            # convert Hz to cents
            pitchInCents = np.array([], dtype = np.float32) # create an empty array
            for p in pitch.tolist():
                cents = UFR.hz2cents(p, -5) # convert Hz to cents, 0 cents is at C0
                pitchInCents.append(cents)

            pitch = ptichInCents
        else:
            pitch = self.pitch
        np.savetxt(outputFilename, np.vstack((xTimeVec, pitch)).T)
        print("Melody pitch is saved at: " + outputFilename + "\n")
    
    def synthesis(self, harmonicsOutputFilename = None, residualOutputFilename = None):
        if len(self.pitch) == 0:
            print 'please do getMelody at first, then do saveMelody.'
            return
            
        if harmonicsOutputFilename == None:
            harmonicsOutputFilename = self.inputFilename[:-4] + '-harmonics.wav'
        if residualOutputFilename == None:
            residualOutputFilename = self.inputFilename[:-4] + '-residual.wav'
		#----- synthesis code-----
        H = self.hopSize
        M = self.frameSize
        N = 2*self.frameSize
        fs = self.fs
		
        t = -60													# threshold peak detection
        devRatio = 10
        nH = 15
        x = self.audio
        winAnalysis = 'hann'
        w = get_window(winAnalysis, M)
        hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
        hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
        Ns = 4*H                                                # FFT size for synthesis (even)
        hNs = Ns/2      
        startApp = max(hNs, hM1)                                # init sound pointer in middle of anal window          
        pin = startApp
        pend = x.size - startApp                                # last sample to start a frame
        x = np.append(np.zeros(startApp),x)                          # add zeros at beginning to center first window at sample 0
        x = np.append(x,np.zeros(startApp))                          # add zeros at the end to analyze last sample
        fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
        yh = np.zeros(Ns)                                       # initialize output sound frame
        yho = np.zeros(x.size)                                  # initialize output array harmonics
        xr = np.zeros(Ns)                                    	# initialize output sound frame
        xro = np.zeros(x.size)                                  # initialize output array residual

        w = w / sum(w)                                          # normalize analysis window
        sw = np.zeros(Ns)                                       # initialize synthesis window
        ow = triang(2*H)                                        # overlapping window
        sw[hNs-H:hNs+H] = ow      
        bh = blackmanharris(Ns)                                 # synthesis window
        bh = bh / sum(bh)                                       # normalize synthesis window
        sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # window for overlap-add
        hfreqp = []
        f0t = 0
        f0stable = 0
        cnt = 0 

        print 'synthesizing ... ...'
        while pin<pend:             
		#-----analysis-----             
			x1 = x[pin-hM1:pin+hM2]                               # select frame
			x2 = x[pin-hNs-1:pin+hNs-1]
			mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
			ploc = UF.peakDetection(mX, t)                        # detect peak locations     
			iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
			ipfreq = fs * iploc/N
			f0t = self.pitch[cnt]
			if ((f0stable==0)&(f0t>0)) \
					or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
				f0stable = f0t                                     # consider a stable f0 if it is close to the previous one
			else:
				f0stable = 0
			hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, devRatio) # find harmonics
			hfreqp = hfreq
	
		#-----synthesis-----
			#-----harmonics-----
			Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs)     # generate spec sines
			fftbuffer = np.real(ifft(Yh))                         # inverse FFT
			yh[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
			yh[hNs-1:] = fftbuffer[:hNs+1] 
			yho[pin-hNs:pin+hNs] += sw*yh                         # overlap-add
	
			#-----residual-----
			X2 = fft(fftshift(x2*bh))
			Xr = X2 - Yh
			fftbuffer = np.real(ifft(Xr))                         # inverse FFT
			xr[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
			xr[hNs-1:] = fftbuffer[:hNs+1] 
			xro[pin-hNs:pin+hNs] += sw*xr                         # overlap-add
	
			pin += H                                              # advance sound pointer
			cnt+=1
	
        yho = np.delete(yho, range(startApp))                            # delete half of first window which was added in stftAnal
        yho = np.delete(yho, range(yho.size-startApp, yho.size))             # add zeros at the end to analyze last sample
        xro = np.delete(xro, range(startApp))                            # delete half of first window which was added in stftAnal
        xro = np.delete(xro, range(xro.size-startApp, xro.size))             # add zeros at the end to analyze last sample

        UF.wavwrite(yho, fs, harmonicsOutputFilename)
        UF.wavwrite(xro, fs, residualOutputFilename)
        
    	print('synthesis done, harmonics file is saved at :' + harmonicsOutputFilename + '\n' +
    	'residual file is saved at :' + residualOutputFilename + '\n')
    	
    	self.yho = yho
    	self.xro = xro
    	return (yho, xro)
    
    def plotSpectrogram(self, fig, spectrogram = 'original'):
        if not spectrogram in self.spectrograms:
            print 'argument spectrogram should be one of ', self.spectrograms
        
        mX = []
        H = self.hopSize
        M = self.frameSize
        N = 2*self.frameSize
        fs = self.fs
        winAnalysis = 'hann'
        SPECTRUM = ess.Spectrum(size=N)
        WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N-M)
        
        print 'ploting ... ...'
        if spectrogram == 'original':
            mX = self.mX
        elif spectrogram == 'harmonics':
            if len(self.yho) == 0:
                print(spectrogram + ' doesn''t contain any frame, please calculate' + 
                spectrogram + 'at first.')
                return
            for frame in ess.FrameGenerator(self.yho.astype(np.float32), frameSize=M, hopSize=H):
                frame = WINDOW(frame)
                mXFrame = SPECTRUM(frame)
                mX.append(mXFrame)
        elif spectrogram == 'residual':
            if len(self.xro) == 0:
                print(spectrogram + ' doesn''t contain any frame, please calculate' + 
                spectrogram + 'at first.')
                return
            for frame in ess.FrameGenerator(self.xro.astype(np.float32), frameSize=M, hopSize=H):
                frame = WINDOW(frame)
                mXFrame = SPECTRUM(frame)
                mX.append(mXFrame)
        
        mX = np.array(mX)
        mX = np.transpose(mX)
        maxplotfreq = 3001.0
        z_max = mX.max()
        z_min, z_max = -120, 20*np.log10(z_max+eps)
        
        mX = mX[:int(N*(maxplotfreq/fs))+1,:]
        timeStamps = np.arange(mX.shape[1])*H/float(fs)                             
        binFreqs = np.arange(mX.shape[0])*fs/float(N)
        
        ax = fig.add_subplot(111)
        specPlt = plt.pcolormesh(timeStamps, binFreqs, 20*np.log10(mX+eps), vmin=z_min, vmax=z_max)
        if spectrogram == 'original' and len(self.pitch) != 0:
            melodyPlt = plt.plot(timeStamps, self.pitch,  color = 'k', linewidth=1.5)
            ax.legend((melodyPlt[0],), ('Melody',))

        plt.autoscale(tight=True)
        plt.title(spectrogram + ' spectrogram')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        
        # xLim = ax.get_xlim()
#         yLim = ax.get_ylim()
#         ax.set_aspect((xLim[1]-xLim[0])/(4.0*(yLim[1]-yLim[0]))) 
     
        ax_cbar = inset_axes(ax, 
                             width="3%", 
                             height="100%", 
                             loc=6,
                             bbox_to_anchor=(1.0, 0, 1, 1),
                             bbox_transform=ax.transAxes)
        plt.colorbar(specPlt, cax = ax_cbar)
        ax_cbar.set_title('dB')

# filename1 = '../daxp-Yu tang chun-Su San qi jie (Li Shengsu)-section.wav'
# test1 = harmonicsSynthesis(filename1)
# test1.spectrogram()
# test1.getMelody()
# 
# fig = plt.figure(0)
# test1.plotSpectrogram(fig = fig)
# 
# #test1.saveMelody()
# test1.synthesis()
# 
# fig = plt.figure(1)
# test1.plotSpectrogram(fig, 'harmonics')
# 
# plt.show()

