import numpy as np
from cpUtils import *

numFacets = 6 # number of facets of the Platonic Solid

c = np.zeros((1,3))
c[0,0] = 1
c[0,1] = 0
c[0,2] = 0.2
# number of petals per layer
numPetalsPerLayer = 4
# number of layers
numLayers = 3
# number of paths per petal
numPathsPerPetal = 20
# vertical scaling for layers: inner layers first
vertScaling = [2,1.5,1]
#vertScaling = [1.3,1]
#vertScaling = [1]
# horizontal scaling for layers
hortScaling = [1, 1.02, 1.04]
# stem scaling, length
stemScale = 3
# offset angles per layer
rotPetalLayer = [np.pi/7,np.pi/3, 0]
# set transparency/opaqueness: 0=invisible, 1=no transparency
opaquePetalLayer = [0.9, 0.7, 0.6]
#opaquePetalLayer = [0.9, 0.6]
#opaquePetalLayer = [0.9]
# each layer can have its own color
colorPetalLayer = ['#e0115f','#fca3b7','#de3163']
#colorPetalLayer = ['#e0115f','#fca3b7']
#colorPetalLayer = ['#e0115f']
# radius of thickening for 3D printing, 
#   the k-gon to extrude for the stem, and the maxDist for linear
#   interpolation (i.e. Netwon point spacing)
r = 0.02
stemThickness = 0.18
numPtsPerStemPt = 5
maxDist = 0.001 

# off to plot
allPaths = plotTulip(numFacets,c,maxDist,numPetalsPerLayer,numPathsPerPetal,\
              vertScaling,hortScaling,r,stemScale,rotPetalLayer,opaquePetalLayer,\
              colorPetalLayer)

# thicken each path to create surfaces
allSurfaces = []
# don't forget that the last path is the stem
for i in range(len(allPaths)-1):
   print('working on path '+str(i)+' out of '+str(len(allPaths)))
   allSurfaces.append(surfaceCurve(allPaths[i],r,2))
# add stem by extruding a polygon along the last path
allSurfaces.append(surfaceCurve(allPaths[len(allPaths)-1],\
                   stemThickness,numPtsPerStemPt))

print(' ')
generateSTLsurf(allSurfaces,numPetalsPerLayer,numLayers,\
                numPathsPerPetal,numPtsPerStemPt,'tulipSTL')
