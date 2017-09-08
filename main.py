
import numpy as np

def getSampleMean(Sample):

    xT = Sample
    sampleMean = 0

    n = len(xT)
    sampleMean = np.sum(xT)/n

    return sampleMean

Y = [3497900, 2485475, 1782875, 1725450, 1645575, 1469800]
print(getSampleMean(Y))
