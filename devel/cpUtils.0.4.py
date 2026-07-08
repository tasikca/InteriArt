#
# cpUtils
# ver 0.0.4
#
# Version 4 adds a printDaisy function that attaches a central
# path stem to a 2D flower in a k-gon. The function calcCurvature
# has an added safety check to ensure we don't divide by zero.
#
# Version 3 aligns with the presentation in the paper, i.e. it 
# assumes maximization in the dual form and it calculates dx, ds, 
# and dy as explained in the paper. This version alters mu spacing
# from version 2 by dividing the mu interval if the primal curvature
# is to high, which then ensures numerical fidelity to the central path.
#
# Earlier function have been redefined/separated into 2D and 3D
# components. 3D utils still need development. Path generation
# works for both 2D and 3D.
#
#
# Al Holder
#

from __future__ import division # safety with double division
import numpy as np
from numpy import linalg
import scipy as sp
import matplotlib.pyplot as plt
import itertools

#
# get3DVertexData
#
# Returns a matrix with each row being the vertex
# coordinates of one of the Platonic solids, along 
# with some other pertinant information.
#
def get3DVertexData(numFacets):
   if numFacets == 4:
      # tetrahedron
      V = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
      numVertices = 4
      numVerticesOnFacet = 3
      numAdjacentFacets = 3
   elif numFacets == 6:
      # cube
      V = [[1, 1,  1], [-1, 1,  1], [-1, -1,  1], [1, -1,  1], \
           [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]
      numVertices = 8
      numVerticesOnFacet = 4
      numAdjacentFacets = 3
   elif numFacets == 8:
      # octahedron
      V = [[1, 0, 0], [-1, 0,  0], [0, 1, 0], [0, -1, 0], \
           [0, 0, 1], [ 0, 0, -1]]
      numVertices = 6
      numVerticesOnFacet = 3
      numAdjacentFacets = 4
   elif numFacets == 12:
      # dodecahedron
      p = (1 + sqrt(5))/2
      V = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], \
           [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1], \
           [0, 1/p, p], [0, -1/p, p], [0, 1/p, -p], [0, -1/p, -p], \
           [1/p, p, 0], [-1/p, p, 0], [1/p, -p, 0], [-1/p, -p, 0], \
           [p, 0, 1/p], [-p, 0, 1/p], [p, 0, -1/p], [-p, 0, -1/p]]
      numVertices = 20
      numVerticesOnFacet = 5
      numAdjacentFacets = 3
   elif numFacets == 20:
      # icosahedron
      p = (1 + sqrt(5))/2
      V = [[0, 1, p], [0, -1, p], [0, 1, -p], [0, -1, -p], \
           [1, p, 0], [-1, p, 0], [1, -p, 0], [-1, -p, 0], \
           [p, 0, 1], [-p, 0, 1], [p, 0, -1], [-p, 0, -1]]
      numVertices = 12
      numVerticesOnFacet = 3
      numAdjacentFacets = 5
   else:
      V = 0
      numVertices = 0
      numVerticesOnFacet = 0
      numAdjacentFacets = 0
      print("number of facets does not agree with a Platonic solid.")
   return V,numVertices,numVerticesOnFacet,numAdjacentFacets

#
# get3DData
#
# Returns a matrix A and vector b to represent
# a Platonic solid as {x : Ax <= b}. Standard descriptions
# of Platonic solids are vertex descriptions, but this
# function calculates facet descriptions.
#
def get3DData(numFacets):
   # Grab the vertex data for solid
   V,numVertices,numVerticesOnFacet,numAdjacentFacets = \
      get3DVertexData(numFacets)
   #
   # generate facets 
   #
   A = np.zeros(shape=(numFacets, 3))
   Aindex = 0

   for vert in itertools.combinations(V, numVerticesOnFacet):
      a = 0
      # make the perpendicular vertex a
      for v in vert:
         a = np.add(a, np.array(v))
      # solve for k
      k = np.round(np.dot(a, np.array(vert[0])), 4)
      br = False
      for v in vert:
         if np.round(np.dot(a, np.array(v)), 4) != k:
            br = True
      iter_anti = np.zeros(shape=(numVertices - numVerticesOnFacet, 3))
      anti_index = 0
      # check that a is a perpendicular vertex
      for v in V:
         minibr = False
         for p in vert:
            if v == p:
               minibr = True
         if minibr:
            continue
         iter_anti[anti_index] = v
         anti_index += 1
      for v in iter_anti:
         if np.dot(a, np.array(v)) >= k:
            br = True
      if br:
         continue
      A[Aindex] = a/np.linalg.norm(a,2)
      Aindex += 1
   #print(A)
   return A, np.ones((numFacets,1))

#
# get2DData
#
# Simple function that returns matrix A and vector b
# so that the 2D n-gon centered at (0,0) is
# {x : Ax <= b}, with b being a vector of ones.
#
def get2DData(numFacets):
   theta = np.array(np.arange(numFacets)*2*np.pi/numFacets)
   return np.vstack((np.cos(theta), np.sin(theta))).T, np.ones((numFacets,1))

#
# getRand2DCvectors
#
# Returns a random collection of c-vectors
# to crete 2D flower-like shapes. Leafs/petals
# are created for each vertex.
#
# Note: numPathsPerLeaf assumes
#       two "edge paths" defining the contour of a leaf
#       all other paths are "interior" paths in a leaf
#       ** so numPathsPerLeaf must be >= 2 **
#
def getRand2DCvectors(A,numPathsPerLeaf):
   rng=np.random.default_rng()
   m,n = np.shape(A)
   cVec = np.zeros((m*numPathsPerLeaf,2))
   #edgeMean = 0.08
   edgeMean = 0.03 + ((0.08 - 0.03)/(7-3))*(m-3)
   tmpA = np.vstack((A,A[0,:])) # removes the need for modular calculations
   for i in range(m):
      c = np.zeros((numPathsPerLeaf,2))
      # Edge paths
      alpha = edgeMean + (edgeMean/3)*rng.standard_normal()
      c[0,:] = (1-alpha)*tmpA[i,:]+alpha*tmpA[i+1,:]
      alpha = edgeMean + (edgeMean/3)*rng.standard_normal()
      c[1,:] = (1-alpha)*tmpA[i+1,:]+alpha*tmpA[i,:]
      # Interior paths (a bit a redundant term, no?)
      alpha = 2*edgeMean + (1-4*edgeMean)*rng.random(size=numPathsPerLeaf-2)
      c[2::,:] = (np.vstack((1-alpha, alpha)).T)@tmpA[i:i+2,:]
      cVec[i*numPathsPerLeaf:(i+1)*numPathsPerLeaf, :] = c
   return cVec

#
# calcPathElement
#
# Returns an element on the central path for A,b,c, and mu
# from starting point x,s,y.
#
def calcPathElement(A,b,c,mu,x,s,y):
   tol = 10**(-8)    # control tolarence for specific mu values
   er = np.linalg.norm(np.vstack((np.array(y)*np.array(s)-mu, A.T@y - c)))
   while er >= tol:
      dx = np.linalg.solve(A.T@np.diag((np.array(y)/np.array(s)).flatten(),0)@A, \
              c - mu*A.T@(1/np.array(s)))
      ds = -A@dx
      dy = mu/np.array(s) - y - np.diag((np.array(y)/np.array(s)).flatten(),0)@ds 
      # find a reasonable step size 
      # the world breaks if s or y become negative
      negInd = np.nonzero(ds < 0)
      if len(negInd[0]) == 0:
         alpha = 1
      else:
         alpha = min(1, 0.9*np.min(-s[negInd]/ds[negInd]))
      negInd = np.nonzero(dy < 0)
      if len(negInd[0]) == 0:
         beta = 1
      else:
         beta = min(1, 0.9*np.min(-y[negInd]/dy[negInd]))
      x  = np.array(x + alpha*dx)
      s  = np.array(s + alpha*ds)
      y  = np.array(y + beta*dy)
      er = np.linalg.norm(np.vstack((y*s-mu, A.T@y - c)))
   return {'x':x, 's':s, 'y':y}

#
# calcCurvature
#
# returns a forward curvature estimate from x1 to x2
#    and a backward curvature estimate from x2 to x3
#
def calcCurvature(x1,x2,x3):
   if np.linalg.norm(x2-x1) < 10**(-16) or np.linalg.norm(x3-x2) < 10**(-16):
      kappa1 = 0
      kappa2 = 0
   else:
      T1 = (x2 - x1) / np.linalg.norm(x2-x1)
      T2 = (x3 - x2) / np.linalg.norm(x3-x2)
      kappa1 = np.linalg.norm(T2-T1) / np.linalg.norm(x2-x1)
      kappa2 = np.linalg.norm(T2-T1) / np.linalg.norm(x3-x2)
   return kappa1, kappa2

#
# divideMuInterval
#
# The function is recursively called to add mu midpoints
# until we have reached linear fidelity to the central path.
#
# returns () #points are added to the path dictionary
#
def divideMuInterval(A,b,c,mu1,mu2,cpDict,maxKappa):
   # assumes mu1 > mu2, probably should add a check
   muMid = 0.5*(mu1+mu2)
   xMid = 0.5*(cpDict[mu1]['x'] + cpDict[mu2]['x'])
   sMid = 0.5*(cpDict[mu1]['s'] + cpDict[mu2]['s'])
   yMid = 0.5*(cpDict[mu1]['y'] + cpDict[mu2]['y'])
   cpDict[muMid] = calcPathElement(A,b,c,muMid,xMid,sMid,yMid)
   # estimate curvature
   kappa1,kappa2 = calcCurvature(cpDict[mu1]['x'],cpDict[muMid]['x'],cpDict[mu2]['x'])
   if kappa1 > maxKappa:
      divideMuInterval(A,b,c,mu1,muMid,cpDict,maxKappa)
   if kappa2 > maxKappa: 
      divideMuInterval(A,b,c,muMid,mu2,cpDict,maxKappa)
   return 

#
# generatePath
#
# Returns a central path for A, b, and c.
# Calculation assumes maximization in the dual form.
#
def generatePath(A,b,c):
   maxKappa = 0.1           # maximum curvature estimate
   muSmallest = np.exp(-16) # surrogate for mu = zero
   muLargest  = 10          # surrogate for mu = infty
   m,n = np.shape(A)
   c = np.matrix(c).T # Make sure to have the correct vector form
   # initialize at the analytic center
   x = np.zeros((n,1))
   s = np.copy(b) # this works because x = 0 is feasible
   y = np.array(muLargest*(1/s)) # muLargest is a starting surrogate for infty
   # instantiate the central path dictionary with keys being mu values
   cpDict = {}
   cpDict[muLargest] = {'x':np.matrix(x), 's':np.matrix(s), 'y':np.matrix(y)}
   cpDict[muSmallest] = calcPathElement(A,b,c,muSmallest,x,s,y)
   divideMuInterval(A,b,c,muLargest,muSmallest,cpDict,maxKappa)
   #
   # This is a bit cumbersome, but the rest of the code expects paths
   # to be x variables only and in a numpy matrix form. So we convert
   # the dictionary to an ordered matrix.
   #
   cp = np.zeros((len(cpDict),n))
   k = 0
   for mu in dict(sorted(cpDict.items())):
      cp[k,:] = cpDict[mu]['x'].T
      k = k+1
   return cp

#
# plot2DFlower
#
# Plots a 2D flower from A, b, and c, with
# each row of c defining a path.
#
# Permits rotation theta and translation 
# T = [xShift, yShift]. Also plots in colr.
#
def plot2DFlower(A,b,c,theta,T,colr):
   pltBorder = 0.1
   pltBound = 0
   fig = plt.figure(figsize=(5,5))
   maxKappa = np.zeros((len(c),1))
   maxKappaMu = np.zeros((len(c),1))
   for t in range(len(c)):
      #cp,maxKappa[t],maxKappaMu[t] = generatePath(A,b,c[t,:])
      cp = generatePath(A,b,c[t,:])
      pltBound = max(abs(cp).max(), pltBound)
      plt.plot(np.cos(theta)*(cp[:,0]+T[0]) + np.sin(theta)*(cp[:,1]+T[1]), \
               -np.sin(theta)*(cp[:,0]+T[0]) + np.cos(theta)*(cp[:,1]+T[1]), \
               linewidth=2.5, color=colr)
   plt.xlim((-pltBound-pltBorder, pltBound+pltBorder))
   plt.ylim((-pltBound-pltBorder, pltBound+pltBorder))
   fig.savefig("cpFig.png")
   plt.show()
   return


#
# plotDaisy
#
# Plots a 2D flower from A, b, and c, with
# each row of c defining a path. We then
# add a 3D stem of a central path in a
# 3D polytope that rotates skewAngle. The
# 3D polytope's bottom facet is the 2D
# polytope for the daisy's petals
#
# Permits rotation theta and translation 
# T = [xShift, yShift]. Also plots in colr.
#
def plotDaisy(A,b,c,theta,T,colr,skewAngle):
   pltBorder = 0.1
   pltBound = 0
   #fig = plt.figure(figsize=(5,5))
   fig = plt.figure().add_subplot(projection='3d')
   maxKappa = np.zeros((len(c),1))
   maxKappaMu = np.zeros((len(c),1))
   for t in range(len(c)):
      #cp,maxKappa[t],maxKappaMu[t] = generatePath(A,b,c[t,:])
      cp = generatePath(A,b,c[t,:])
      pltBound = max(abs(cp).max(), pltBound)
      plt.plot(np.cos(theta)*(cp[:,0]+T[0]) + np.sin(theta)*(cp[:,1]+T[1]), \
               -np.sin(theta)*(cp[:,0]+T[0]) + np.cos(theta)*(cp[:,1]+T[1]), \
               0, \
               linewidth=2.5, color=colr)
   # Add a path for the stem, all stems are generated in
   # 3D rectangle with a stretched z-coordinate
   Astem,bstem = get3DData(6)
   m,n = np.shape(Astem)
   cstem = np.ones((1,3))
   # adjustments
   stemLength = 1.5
   Astem[0,2] = Astem[0,2]/stemLength
   Astem[5,2] = Astem[5,2]/stemLength
   cstem[0,1] = cstem[0,1]-0.2
   cstem[0,2] = cstem[0,1]+0.1
   cp = generatePath(Astem,bstem,cstem)
   p,q = np.shape(cp)
   rotVec = cp[p-10,:]
   K = np.zeros((3,3))
   K[0,2] = -rotVec[0]
   K[1,2] = -rotVec[1]
   K[2,0] = rotVec[0]
   K[2,1] = rotVec[1]
   rotAng = np.acos(rotVec[2] / np.linalg.norm(rotVec))
   rotMat = np.eye(3) + np.sin(rotAng)*K + (1-np.cos(rotAng))*(K@K)
   cpRot = cp@rotMat.T
   plt.plot(cpRot[:,0],cpRot[:,1],cpRot[:,2],linewidth=2.5,color="g")
   fig.set_box_aspect((np.ptp(cpRot[:,0]),np.ptp(cpRot[:,1]),np.ptp(cpRot[:,2])))
   fig.set_axis_off()
   plt.xlim((-pltBound-pltBorder, pltBound+pltBorder))
   plt.ylim((-pltBound-pltBorder, pltBound+pltBorder))
   #fig.savefig("cpFig.png")
   plt.show()
   return

#
# plot2DTiling
#
# Plots 2D tilings. The inputs are
#    shp - dictionary of shapes defined by A, b, and c
#    trn - dictionary of translations for each shape
#
# Could use some improvements with boarders, scaling,
# figure size, etc. 
#
# You can send an optional dictionary:
#    0:filename.png
#    1:[xmin,xmax]
#    2:[ymin,ymax]
# the last two items draw x and y axes,
# which may or may not be commented (too lazy to add flag)
#
def plot2DTiling(shp,trn,*args):
   pltBorder = 0.1
   pltBound = 0
   fig = plt.figure()
   for s in shp:
      for t in range(len(shp[s]["c"])):
         cp = generatePath(shp[s]["A"],shp[s]["b"],shp[s]["c"][t,:])
         pltBound = max(abs(cp).max(), pltBound)
         for k in trn[s]:
            plt.plot(np.cos(trn[s][k]["theta"])*(cp[:,0]+trn[s][k]["shift"][0]) + \
                     np.sin(trn[s][k]["theta"])*(cp[:,1]+trn[s][k]["shift"][1]), \
                    -np.sin(trn[s][k]["theta"])*(cp[:,0]+trn[s][k]["shift"][0]) + \
                     np.cos(trn[s][k]["theta"])*(cp[:,1]+trn[s][k]["shift"][1]), \
                     linewidth=2, color=trn[s][k]["clr"])
   #plt.xlim((-pltBound-pltBorder, pltBound+pltBorder))
   #plt.ylim((-pltBound-pltBorder, pltBound+pltBorder))
   ax = plt.gca()
   ax.set_aspect('equal')
   #ax.set_aspect('equal', adjustable='box')
   # save figure if name requested (only png is allowed)
   if len(args) == 1:
      if isinstance(args[0][0],str):
         if args[0][0].find('.png'):
            #plt.plot([args[0][1][0],args[0][1][1]],[0,0],linewidth=1.5,color='k')
            #plt.plot([0,0],[args[0][2][0],args[0][2][1]],linewidth=1.5,color='k')
            #plt.axis("off")
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.savefig(args[0][0], bbox_inches='tight', \
                        pad_inches=0.05, transparent=True)
   plt.show()
         
