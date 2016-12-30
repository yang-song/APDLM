# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:26:30 2015

@author: yangsong
"""

import re
import numpy as np
import matplotlib.pylab as pl
if __name__ == '__main__':
    pat = re.compile('Accuracy:\s([0-9]+\.*[0-9]*)', re.MULTILINE)
    fin = open('../APDLM-finetuning/bird.log','r')
    aps = pat.findall(fin.read())
    aps = [float(x) for x in aps]
    print("mAP: %f" % np.mean(aps))
    N = 10
    rap = np.convolve(aps, np.ones((N,))/N, mode='valid')
    pl.plot(np.r_[0:len(rap)],rap,'-')
    pat = re.compile('Accuracy:\s([0-9]+\.*[0-9]*)', re.MULTILINE)
    fin = open('../APDLM-verifying/verifybird.log','r')
    aps = pat.findall(fin.read())
    aps = [float(x) for x in aps]
    print("mAP: %f" % np.mean(aps))
    N = 10
    rap = np.convolve(aps, np.ones((N,))/N, mode='valid')
    pl.plot(np.r_[0:len(rap)],rap,'-')
    pl.legend(['finetuning','converted'],loc=4)
    pl.xscale('log')