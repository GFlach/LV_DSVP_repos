import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from pylab import *

def adsr_profile(env, x):
    tA = env[0]
    tD = env[1]
    tS = env[2]
    ED = env[3]
    ES = env[4]
    tR = len(x)
    NA = floor(tA * tR) + 1
    A = 1/(NA -1)
    E = 0
    y = np.zeros(tR)
    for k in np.arange(2, NA):
        E = E + A
        y[int(k)] = E * x[int(k)]
    ND = floor((tD - tA) * tR)
    D = (1 - ED)/ND
    for k in np.arange(NA, NA + ND):
        E = E - D
        y[int(k)] = E * x[int(k)]
    NS = floor((tS - tD) * tR)
    S = (ED - ES)/NS
    for k in np.arange(NA + ND, NA + ND + NS):
        E = E - S
        y[int(k)] = E * x[int(k)]
    NR = tR - k
    R = E/NR
    for k in np.arange(NA + ND + NS, tR):
        E = E - R
        y[int(k)] = E * x[int(k)]
    return y

def exponent_profile(env, x):
    tA = env[0]
    EE = env[1]
    tR = len(x)
    y = np.zeros(tR)
    NA = floor(tA * tR) + 1
    A = 1/(NA)
    E = 0
    for k in np.arange(0, NA):
        E = E + A
        y[int(k)] = E * x[int(k)]
    NE = tR - k
    EE = EE + 0.001
    a = -(np.log(EE) * NE)/(tR - NA)
    for k in np.arange(NA, tR):
        E = np.exp(a * (-(k - NA)/NE))
        y[int(k)] = E * x[int(k)]
    return y
    
def triangle(rel_dur, frequency, amplitude, fs, tempo):
    section = fs // (4 * frequency)
    n = amplitude/section
    s_p = []
    s = []
    x = np.arange(0,section)
    s1 = n * x
    s_p = np.append(s_p, s1)
    s2 = -n * x + amplitude
    s_p = np.append(s_p, s2)
    s3 = -n * x
    s_p = np.append(s_p, s3)
    s4 = s1 - amplitude
    s_p = np.append(s_p, s4)
    for i in np.arange(0, rel_dur * frequency * tempo):
        s = np.append(s, s_p)
    return s

def create_music(pitch, duration, stimulus, env, tempo, fs, akk):
    music = []
    for k in np.arange(0,len(pitch)):
        L = tempo * fs * duration[k]
        n = np.arange(0,L-1)
        if stimulus == 'sin':
            w = 2 * np.pi/fs * pitch[k]
            s = np.sin(w * n)
            ak = np.zeros(len(s))
        if stimulus == 'sin_ow':
            w = 2 * np.pi/fs * pitch[k]
            s = np.sin(w * n) + 0.25 * np.sin(2 * w * n) + 0.25 * np.sin(3 * w * n) + 0.25 * np.sin(4 * w * n) + np.sin(4 * w * n)
            ak = np.zeros(len(s))
        if stimulus == 'triangle':
            s = np.array(triangle(duration[k], pitch[k], 1, fs, tempo))
            ak = np.zeros(len(s))
        if len(env) == 5:
            s = adsr_profile(env, s)
        if len(env) == 2:
            s = exponent_profile(env, s)
        if akk == True:
            ak = ak + s
        music = np.append(music, s)
    music = np.append(music, ak)
    return music

def plot_zf(music, fs):
    t = np.arange(0, len(music)/fs, 1/fs)
    plt.figure(figsize=(15,5))
    plt.plot(t, music)
    plt.axis([0, len(music)/fs, -1, 1])
    plt.title('Zeitverlauf des Signals')
    plt.xlabel('t in s')
    plt.show()

def plot_stspec(music, fs):
    plt.figure(figsize=(15,5))
    atw_zf = 256
    plt.specgram(music, NFFT=atw_zf, Fs=fs, noverlap=0, cmap='hsv')
    plt.title('Kurzzeitspektrum')
    plt.xlabel('t in s')
    plt.ylabel('f in Hz')
    plt.show()

def plot_adsr(env):
    tA = env[0]
    tD = env[1]
    tS = env[2]
    ED = env[3]
    ES = env[4]
    t = np.arange(0, 100, 0.1)
    x = np.ones(len(t))
    x = adsr_profile([tA, tD, tS, ED, ES], x)
    plt.plot(t,x)
    plt.title('ADSR-Hüllkurve')
    plt.xlabel('% Signaldauer')
    plt.axis([0,100, 0, 1])
    plt.grid()
    plt.plot([0,100],[ED, ED], ':k')
    plt.plot([0,100],[ES, ES], ':k')
    plt.plot([tA * 100, tA * 100],[0.0, 1], ':k')
    plt.plot([tD * 100, tD * 100],[0.0, 1], ':k')
    plt.plot([tS * 100, tS * 100],[0.0, 1], ':k')
    
    plt.annotate('tA', xy=(tA * 100, 0), xytext=(tA * 100 + 1, 0.05))
    plt.annotate('tD', xy=(tD * 100, 0), xytext=(tD * 100 + 1, 0.05))
    plt.annotate('tS', xy=(tS * 100, 0), xytext=(tS * 100 + 1, 0.05))
    plt.annotate('ED', xy=(1, ED), xytext=(1, ED + 0.02))
    plt.annotate('ES', xy=(1, ES), xytext=(1, ES + 0.02))
    plt.savefig('image/adsr.jpg')
    plt.show()

def plot_exponent(env):
    tA = env[0]
    EE = env[1]
    t = np.arange(0, 100, 0.1)
    x = np.ones(len(t))
    x = exponent_profile([tA, EE], x)
    plt.plot(t,x)
    plt.title('Exponent-Hüllkurve')
    plt.xlabel('% Signaldauer')
    plt.axis([0,100, 0, 1])
    plt.grid()
    plt.plot([0,100],[EE, EE], ':k')
    plt.plot([tA * 100, tA * 100],[0.0, 1], ':k')

    plt.annotate('tA', xy=(tA * 100, 0), xytext=(tA * 100 + 1, 0.05))
    plt.annotate('EE', xy=(1, EE), xytext=(1, EE + 0.02))
    plt.savefig('image/exponent.jpg')
    plt.show()
