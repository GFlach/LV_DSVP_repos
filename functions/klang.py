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
    if tA == 0:
        A = 1
        NA = 3
    else:
        NA = floor(tA * tR) + 1
        A = 1/(NA -1)
    E = 0
    y = np.zeros(tR)
    for k in np.arange(2, NA):
        E = E + A
        y[int(k)] = E * x[int(k)]
    if tD == tA:
        ND = 1
    else:
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
    
def triangle(f, fs, dur):
    t = np.arange(1/fs,dur +1/fs,1/fs)
    tri = np.sin(t * 0)
    for a in np.arange(1,40,2):
        tri = tri + 8/(a**2 * np.pi**2) * np.cos(a * 2*np.pi*f*t)
#    plt.plot(t,tri)
#    plt.xlabel('t in s')
#    plt.show()
    return tri
      
def create_music(pitch, duration, stimulus, env, tempo, fs, ow=[]):
    music = []
    for k in np.arange(0,len(pitch)):
        L = tempo * fs * duration[k]
        n = np.arange(0,L-1)
        if stimulus == 'ton':
            w = 2 * np.pi/fs * pitch[k]
            s = np.cos(w * n)
        if stimulus == 'klang':
            s = np.cos(n * 0)
            w = 2 * np.pi/fs * pitch[k]
            for a in np.arange(0, len(ow)):
                s = s + ow[a] * np.cos(a * w * n)
            s = s - np.mean(s)
        if stimulus == 'triangle':
            s = np.array(triangle(pitch[k], fs, duration[k]*tempo))
        if len(env) == 5:
            s = adsr_profile(env, s)
        if len(env) == 2:
            s = exponent_profile(env, s)
        music = np.append(music, s)
    return music

def plot_zf(music, fs):
    t = np.arange(0, len(music)/fs, 1/fs)
    plt.figure(figsize=(12,4))
    plt.plot(t, music)
    plt.axis([0, len(music)/fs, min(music), max(music)])
    plt.title('Zeitverlauf des Signals')
    plt.xlabel('t in s')
    plt.show()

def plot_stspec(music, fs):
    plt.figure(figsize=(12,4))
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
    plt.savefig('images/adsr.jpg')
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
    plt.savefig('images/exponent.jpg')
    plt.show()
def klang(env = [0.01, 0.1], ow = [1, 0.5, 0.5, 0.5, 0.45]):
    stimulus = 'klang'
    A = 440
    pitch = np.array([A])
    duration = np.array([1])
    tempo = 0.5
    fs = 8000
    music = create_music(pitch, duration, stimulus, env, tempo, fs, ow= ow)
    return music

def triang(env):
    stimulus = 'triangle'
    A = 440
    pitch = np.array([A])
    duration = np.array([1])
    tempo = 0.5
    fs = 8000
    ow = []
    music = create_music(pitch, duration, stimulus, env, tempo, fs, ow= ow)
    return music

def accord(pitch, env, ow):
    duration = np.array([1, 1, 1, 1])
    tempo = 1/2
    fs = 8000
    stimulus = 'klang'
    music = create_music(pitch, duration, stimulus, env, tempo, fs, ow=ow)
    return music

def ton_prelude(env):
    A = 220
    Dh = A * 2**(5/12)
    C = A * 2**(3/12)
    B = A * 2**(2/12)
    G = A * 2**(-2/12)
    Fis = A * 2**(-3/12)
    E = A * 2**(-5/12)
    D = A * 2**(-7/12)
    pitch = np.array([D, G, G, A, B, G, Dh, B, B, C, Dh, C, B, C, Dh, A, G, A, B, A])
    duration = np.array([2, 2, 1, 1, 2, 2, 4, 3, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2])
    tempo = 1/4
    fs = 8000
    music = create_music(pitch, duration, 'ton', env, tempo, fs, ow = [1])
    return (music)

def klang_prelude(env):
    A = 220
    Dh = A * 2**(5/12)
    C = A * 2**(3/12)
    B = A * 2**(2/12)
    G = A * 2**(-2/12)
    Fis = A * 2**(-3/12)
    E = A * 2**(-5/12)
    D = A * 2**(-7/12)
    pitch = np.array([D, G, G, A, B, G, Dh, B, B, C, Dh, C, B, C, Dh, A, G, A, B, A])
    duration = np.array([2, 2, 1, 1, 2, 2, 4, 3, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2])
    tempo = 1/4
    fs = 8000
    music = create_music(pitch, duration, 'klang', env, tempo, fs, ow = [1, 0.5, 0.5, 0.5, 0.45])
    return (music)

def triang_prelude(env):
    A = 220
    Dh = A * 2**(5/12)
    C = A * 2**(3/12)
    B = A * 2**(2/12)
    G = A * 2**(-2/12)
    Fis = A * 2**(-3/12)
    E = A * 2**(-5/12)
    D = A * 2**(-7/12)
    pitch = np.array([D, G, G, A, B, G, Dh, B, B, C, Dh, C, B, C, Dh, A, G, A, B, A])
    duration = np.array([2, 2, 1, 1, 2, 2, 4, 3, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2])
    tempo = 1/4
    fs = 8000
    music = create_music(pitch, duration, 'triangle', env, tempo, fs, ow = [])
    return (music)