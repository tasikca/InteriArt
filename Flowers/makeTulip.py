import numpy as np
from cpUtils import *

numFacets = 6 # number of facets of the Platonic Solid

c = np.zeros((1,3))
c[0,0] = 1
c[0,1] = 0
c[0,2] = 0.1
# number of petals per layer
numPetalsPerLayer = 3
# vertical scaling for layers: inner layers first
vertScaling = [1.3,1.1, 1]
# offset angles per layer
rotPetalLayer = [np.pi/12,np.pi/6, 0]
# set transparency/opaqueness: 0=invisible, 1=no transparency
opaquePetalLayer = [0.9, 0.7, 0.6]
# each layer can have its own color
colorPetalLayer = ['#e0115f','#fca3b7','#de3163']
# off to plot
plotTulip(numFacets,c,numPetalsPerLayer,10,vertScaling,rotPetalLayer,\
          opaquePetalLayer,colorPetalLayer)

