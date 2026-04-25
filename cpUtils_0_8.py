#
# cpUtils
# ver 0.0.8
#
# Version 0.08 adds the ability to extrude k-gons along a
# line in surfaceCurve
#
# Verion 0.07 adds a return list to plotTulip for 3D printing.
#
# Version 0.06 adds the ability to extrude a k-gon along a
# central path, with the vertices of the k-gons being
# triangulated to create a mesh that can then be exported to
# an stl file for 3D printing. I also added a return list
# to plotDaisy for 3D printing.
#
# Version 0.05 adds a tulip function that rotates a curve
# to create tulip petals surface.
#
# Version 0.04 adds a plotDaisy function that attaches a central
# path stem to a 2D flower in a k-gon. The function calcCurvature
# has an added safety check to ensure we don't divide by zero.
#
# Version 0.03 aligns with the presentation in the paper, i.e. it 
# assumes maximization in the dual form and it calculates dx, ds, 
# and dy as explained in the paper. This version alters mu spacing
# from version 2 by dividing the mu interval if the primal curvature
# is to high, which then ensures numerical fidelity to the central path.
#
# Earlier function have been redefined/separated into 2D and 3D
# components. 3D utils still need development. Path generation
# works for both 2D and 3D.
#
# Al Holder
#

from __future__ import division # safety with double division
import numpy as np
from numpy import linalg
import scipy as sp
import matplotlib.pyplot as plt
import itertools
from stl import mesh
import time
import datetime
import os
import re

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
   theta = theta+(np.pi/2) #so a vertex is always point up
   A = np.vstack((np.cos(theta), np.sin(theta))).T

   vertexAngle = theta+(np.pi/numFacets)
   circuR = 1/np.cos(np.pi/numFacets)
   verX = circuR*np.cos(vertexAngle)
   verY = circuR*np.sin(vertexAngle)

   scale = np.max(np.linalg.norm(np.hstack((verX.reshape(-1,1),verY.reshape(-1,1))),ord=np.inf,axis=1))
   
   scaledA = A*scale

   return scaledA, np.ones((numFacets,1))

   #return np.vstack((np.cos(theta), np.sin(theta))).T, np.ones((numFacets,1))

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
   edgeMean = 0.03 + ((0.08 - 0.03)/(7-3))*(m-3)
   #edgeMean = 0.08
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
   tol = 10**(-6)    # control tolarence for specific mu values
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
   for t in range(len(c)):
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
# plotTulip
#
# This function creates tulip petals by rotating
# paths in a cube (or other Platonic solid depending
# on numFacets). Input arguments are
#   numFacets - defines the Platonic solid
#   c - defines the central paths
#   numPetalsPerLayer - how many petals to distribute
#      in each rotational 'layer' of the tulip
#   numPathsPerPetal - the number of paths per petal
#   vertScaling - scale parameter to adust height
#   rotPetalLayer - offset the starting angle per layer
#   opaquePetalLayer - sets visibility between/among layers
#   colorPetalLayer - sets the color of a layer
#
# As of version 0.06 we have a return item
# that is a list of central paths for 3D printing.
#
def plotTulip(numFacets,c,numPetalsPerLayer,numPathsPerPetal,vertScaling,\
              rotPetalLayer,opaquePetalLayer,colorPetalLayer):
   pltBorder = 0.1
   pltBound = 0
   # allCP is a list of paths, returned for 3D printing
   allCP = []
   fig = plt.figure().add_subplot(projection='3d')
   # generate path for outer petals
   A,b = get3DData(numFacets)
   cp = generatePath(A,b,c[0,:])
   pltBound = max(abs(cp).max(), pltBound)
   # space between petals in the same layer - could be a sent parameter
   petalSpace = np.pi/10
   # angles for each petal in a single layer
   petalStartStop = np.linspace(0,2*np.pi,numPetalsPerLayer+1)
   # used to flip vertically for printing
   F = np.eye(3)
   F[2,2] = -1
   for v in range(len(vertScaling)):
      for p in range(len(petalStartStop)-1):
         rAngle = np.linspace(rotPetalLayer[v]+petalStartStop[p]+petalSpace,\
                              rotPetalLayer[v]+petalStartStop[p+1]-petalSpace,numPathsPerPetal)
         R = vertScaling[v]*np.eye(3)
         for k in range(len(rAngle)):
            R[0,0] = np.cos(rAngle[k])
            R[1,1] = np.cos(rAngle[k])
            R[0,1] = np.sin(rAngle[k])
            R[1,0] = -np.sin(rAngle[k])
            cpRot = (cp@R.T)@F
            plt.plot(cpRot[:,0], cpRot[:,1], cpRot[:,2], 
                     linewidth=4, color=colorPetalLayer[v], alpha=opaquePetalLayer[v])
            allCP.append(cpRot)
   # Add a stem
   Astem,bstem = get3DData(6)
   cstem = -np.ones((1,3))
   # adjustments
   stemLength = 1.4
   Astem[0,2] = Astem[0,2]/stemLength
   Astem[5,2] = Astem[5,2]/stemLength
   cstem[0,1] = cstem[0,1]-0.2
   cstem[0,2] = cstem[0,1]+0.1
   #for i in range(5):
   #   bstem[i] = 0.5*bstem[i]
   #Astem[0,2] = Astem[0,2]/stemLength
   #Astem[5,2] = Astem[5,2]/stemLength
   #Astem[1,1] = 3*Astem[1,1]
   #Astem[2,0] = 3*Astem[2,0]
   #Astem[3,0] = 3*Astem[3,0]
   #Astem[4,1] = 3*Astem[4,1]
   #cstem[0,0] = cstem[0,1]+0.6
   #cstem[0,1] = cstem[0,1]+0.8
   cp = generatePath(Astem,bstem,cstem)
   p,q = np.shape(cp)
   #rotVec = cp[min(round(0.9*p),p-2),:]
   rotVec = cp[p-20,:]
   K = np.zeros((3,3))
   K[0,2] = -rotVec[0]
   K[1,2] = -rotVec[1]
   K[2,0] = rotVec[0]
   K[2,1] = rotVec[1]
   rotAng = np.acos(rotVec[2] / np.linalg.norm(rotVec))
   rotMat = np.eye(3) + np.sin(rotAng)*K + (1-np.cos(rotAng))*(K@K)
   sclMat = np.eye(3)
   sclMat[0,0] = 0.2
   sclMat[1,1] = 0.2
   sclMat[2,2] = -1
   cpRot = (cp@rotMat.T)@sclMat
   plt.plot(cpRot[:,0],cpRot[:,1],cpRot[:,2],linewidth=2.5,color="g")
   allCP.append(cpRot)
   fig.set_box_aspect((np.ptp(cpRot[:,0]),np.ptp(cpRot[:,1]),np.ptp(cpRot[:,2])))
   fig.set_axis_off()
   plt.show()
   return allCP

#
# plotDaisy
#
# Plots a 2D flower from A, b, and c, with
# each row of c defining a path. We then
# add a 3D stem of a central path in a
# 3D rectangle (4-gon). The stem is rotated
# with so that it approximately appoaches the 
# daisy orthogonoly.
#
# As of version 0.06 we have a return item
# that is a list of central paths for 3D printing.
#
# Permits rotation theta and translation 
# T = [xShift, yShift]. Also plots in colr.
#
# returns a list of path coordinates for 3D printing
#
def plotDaisy(A,b,c,theta,T,colr):
   pltBorder = 0.1
   pltBound = 0
   # allCP is a list of paths, returned for 3D printing
   allCP = []
   #fig = plt.figure(figsize=(5,5))
   fig = plt.figure().add_subplot(projection='3d')
   for t in range(len(c)):
      cp = generatePath(A,b,c[t,:])
      pltBound = max(abs(cp).max(), pltBound)
      K = np.zeros((2,2))
      K[0,0] = np.cos(theta)
      K[0,1] = np.sin(theta)
      K[1,0] = -np.sin(theta)
      K[1,1] = np.cos(theta)
      cpRot = np.zeros((len(cp),3))
      cpRot[:,0:2] = cp@K.T + T
      plt.plot(cpRot[:,0], cpRot[:,1], cpRot[:,2], linewidth=2.5, color=colr)
      allCP.append(cpRot)
      #plt.plot(np.cos(theta)*(cp[:,0]+T[0]) + np.sin(theta)*(cp[:,1]+T[1]), \
      #         -np.sin(theta)*(cp[:,0]+T[0]) + np.cos(theta)*(cp[:,1]+T[1]), \
      #         0, \
      #         linewidth=2.5, color=colr)
   # Add a path for the stem, all stems are generated in
   # 3D rectangle with a stretched z-coordinate
   Astem,bstem = get3DData(6)
   m,n = np.shape(Astem)
   cstem = np.ones((1,3))
   # adjustments
   #stemLength = 1.5
   stemLength = 2
   Astem[0,2] = Astem[0,2]/stemLength
   Astem[5,2] = Astem[5,2]/stemLength
   cstem[0,1] = cstem[0,1]-0.2
   cstem[0,2] = cstem[0,1]+0.1
   cp = generatePath(Astem,bstem,cstem)
   p,q = np.shape(cp)
   rotVec = cp[p-20,:]
   K = np.zeros((3,3))
   K[0,2] = -rotVec[0]
   K[1,2] = -rotVec[1]
   K[2,0] = rotVec[0]
   K[2,1] = rotVec[1]
   rotAng = np.acos(rotVec[2] / np.linalg.norm(rotVec))
   rotMat = np.eye(3) + np.sin(rotAng)*K + (1-np.cos(rotAng))*(K@K)
   cpRot = cp@rotMat.T + [T[0],T[1],0]
   plt.plot(cpRot[:,0],cpRot[:,1],cpRot[:,2],linewidth=2.5,color="g")
   allCP.append(cpRot)
   fig.set_box_aspect((np.ptp(cpRot[:,0]),np.ptp(cpRot[:,1]),np.ptp(cpRot[:,2])))
   fig.set_axis_off()
   plt.xlim((-pltBound-pltBorder, pltBound+pltBorder))
   plt.ylim((-pltBound-pltBorder, pltBound+pltBorder))
   #fig.savefig("cpFig.png")
   plt.show()
   return allCP

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

#
# surfaceCurve
#
#    Generates points on a circle orthogonal to the path and
#    centered at the points defining the path.
#
#    The input arguments are:
#       cp - list of points on a central path
#       r  - radius of the circle
#       k  - number of points per circle
#
#    The return is:
#       crclPtsAroundCP - an array of matrices, with each
#          entry corresponding with an element of the path.
#          Each matrix is k by 3, with each row containing 
#          the coordinates of a point on the circle around
#          the point on the path.
#
#   The return structure is essentially a 3D matrix, which works
#   with Connor's stl code. I think this data structure could be
#   streamlined.
#
#   As of version 0.08 we can also extrude along a line. This
#   proces is triggered by defining a curve by the starting and
#   ending point of the line segment alone.
#
def surfaceCurve(cpOrig, r, k):
   # The path may have points too close to numerically
   # approximate TNB, so we first insepct and possibly cull the
   # path.
   # NOTE: We might need to reverse the culling process because
   #       this check could have issues with the end of the path.
   #       I am leaving it as is until we have errors.
   numPts = len(cpOrig)
   tol = 10**(-8)
   # make sure to include the first point
   rowInd = [0]
   for j in range(1,numPts-1):
      if np.linalg.norm(cpOrig[j]-cpOrig[j-1]) > tol:
         rowInd.append(j)
   # add the last point, which could, as noted above, be too
   # close to the previous point.
   rowInd.append(numPts-1)
   # grab the elements that we want for 3D printing
   cp = cpOrig[rowInd,:]
   # Instantiate space for points on circles around the path
   crclPtsAroundCP = np.array([]).reshape(0,3)
   ax = plt.axes(projection='3d') # axes for debugging
   # Loop through the points of the path
   for i in range(np.size(cp,0)):
      # Create the TNB reference vectors - depends on if the path
      # is a line.
      if np.size(cp,0) == 2:
         T = cp[1] - cp[0]
         if np.abs(T[0]+T[1]) > tol:
            N = np.array([-T[1],T[0],0])
         else:
            N = np.array([1,0,0])
      elif np.size(cp,0) >= 3:
         # Estimage T and N at each point of the path
         if i == 0:
            T = (cp[i+1]-cp[i])/np.linalg.norm(cp[i+1]-cp[i])
            T2 = (cp[i+2]-cp[i+1])/np.linalg.norm(cp[i+2]-cp[i+1])
            N = (T-T2)/np.linalg.norm(T2-T)
            Nprev = N
         elif i == np.size(cp,0)-1:
            T1 = (cp[i-1]-cp[i-2])/np.linalg.norm(cp[i-1]-cp[i-2])
            T = (cp[i]-cp[i-1])/np.linalg.norm(cp[i]-cp[i-1])
            N = (T1-T)/np.linalg.norm(T-T1)
            Nprev = N
         else:
            T1 = (cp[i+1]-cp[i])/np.linalg.norm(cp[i+1]-cp[i])
            T2 = (cp[i]-cp[i-1])/np.linalg.norm(cp[i]-cp[i-1])
            T = (T1+T2)/np.linalg.norm(T1+T2)
            if np.linalg.norm(T2-T1) < tol:
               # in this case the path is pretty linear, just use the
               # previous N
               N = Nprev
            else:
               N = (T2-T1)/np.linalg.norm(T2-T1)
      B = np.cross(T,N)
      # add points on circle around cp point i to our list of poins
      theta = np.linspace(0,2*np.pi-(2*np.pi/k),num=k)
      pts = np.zeros((k,3))
      for j in range(k):
         pts[j,:] = cp[i] + r*(np.cos(theta[j])*N + np.sin(theta[j])*B)
      crclPtsAroundCP = np.vstack([crclPtsAroundCP, pts])
      #ax.plot3D(pts[:,0], pts[:,1], pts[:,2], ".")
   #ax.plot3D(cp[:,0], cp[:,1], cp[:,2])
   #plt.show()
   return crclPtsAroundCP

#
# generateSTL
#
# This is an adaptation of Connor's stl code. The triangular
# tesselation is built in 'layers' by progressing down a path.
# This also works with linux, and I some added code to remove 
# the temporary stl files.
# 
# The input arguments are:
#    allcp - an array with each element corresponding with
#            a central path. Each element is a list of matrices
#            describing the points encircling the path elements.
#    numPointInCircle - number of points per circle
#    stlFilename - the name (string) of the resulting stl file
#
# The function returns nothing, but it does save the file.
#
# The data structure could use some help, and I don't think we
# need to write the files to disk. This is working, but uninspired, 
# code.
#
def generateSTL(allcp,k,stlFilename):
   #change dir to temp folder to save each path stl
   curdir = os.getcwd()
   # directory to place temporary individual stl files
   newdir = curdir+'/tmpSTL'
   # create the directory if it doesn't exist
   if not os.path.exists(newdir):
      os.makedirs(newdir)
   os.chdir(newdir)
   # loop through the vertices for each path
   for q, verts in enumerate(allcp):
      # add triangles for each layer between two consecutive
      # cp points - note that verts is a list of surrounding
      # vertices around cp points, with each row corresponding
      # with a cp point.
      #
      # instantiate a face index
      faceIndex = []
      numLayers = int(len(verts)/k)-1
      for t in range(numLayers):
         for i in range(k):
            s = t*k+i
            # this conditional avoids modular arithmetic,
            # probably could be more elegant
            if i == 0:
               faceIndex.append([s,s+k-1,s+2*k-1])
               faceIndex.append([s,s+2*k-1,s+k])
            else:
               faceIndex.append([s-1,s,k+s-1])
               faceIndex.append([s,k+s-1,k+s])
      # add caps at the top and bottom
      for t in range(2):
         s = numLayers*k*t
         for i in range(1,k-1):
            faceIndex.append([s,s+i,s+i+1])
      # put the face indices in an np array
      faceIndex = np.array(faceIndex)
      #print(faceIndex)
      #
      # Not really happy with the file based method below, but it
      # it works for now.
      #
      tempSTL = mesh.Mesh(np.zeros(faceIndex.shape[0], dtype=mesh.Mesh.dtype))
      for i, f in enumerate(faceIndex):
         for j in range(3):
            tempSTL.vectors[i][j] = verts[f[j], :]
      tempSTL.save('tmpSTL'+str(q)+'.stl')
   for file in os.listdir(newdir):
      if file == os.listdir(newdir)[0]:
         finalSTL = mesh.Mesh.from_file(file)
         finalSTL.save('finalSTL.stl')
      else:
         finalSTL = mesh.Mesh.from_file('finalSTL.stl')
         curSTL = mesh.Mesh.from_file(file)
         combined = mesh.Mesh(np.concatenate([finalSTL.data, curSTL.data]))
         combined.save('finalSTL.stl')
   finalSTL = mesh.Mesh.from_file('finalSTL.stl')
   os.chdir(curdir)
   finalSTL.save(stlFilename+'.stl')
   # Clean-up
   for f in os.listdir(newdir):
      if re.search('.stl',f):
         os.remove(os.path.join(newdir,f))
   print('Created stl file '+stlFilename+'.stl')

