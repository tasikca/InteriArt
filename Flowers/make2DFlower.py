import numpy as np
from cpUtils import *

n = 4
#numPathsPerLeaf = 6

A,b = get2DData(n)
#c = getRand2DCvectors(A,numPathsPerLeaf)
c = np.zeros((8,2))
c[0,0] = 1
c[0,1] = 0.01
c[1,0] = 0.1
c[1,1] = 1
c[2,0] = -1
c[2,1] = 0.01
c[3,0] = -0.1
c[3,1] = 1
c[4,0] = -1
c[4,1] = -0.01
c[5,0] = -0.1
c[5,1] = -1
c[6,0] = 1
c[6,1] = -0.01
c[7,0] = 0.1
c[7,1] = -1
plot2DFlower(A,b,c,0,[0,0],"b")

