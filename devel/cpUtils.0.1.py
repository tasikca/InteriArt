#
# cpUtils
# ver 0.0.1
#
# This is the first version that aligns with the presentation 
# in the paper, i.e. it assumes maximization in the dual form and
# it calculates dx, ds, and dy as explained in the paper.
#
# Earlier function have been redefined/separated into 2D and 3D
# components. 3D utils still need development. Path generation
# works for both 2D and 3D.
#
# Still need to better decide mu values.
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
# generatePath
#
# Returns a central path for A, b, and c.
# Calculation assumes maximization in the dual form.
#
def generatePath(A,b,c):
   tol = 10**(-8) # alter this to control tolarence
   numPts = 50  # alter this to get better approximations
   mu = 30*np.exp(-np.linspace(0, 14, numPts))
   m,n = np.shape(A)
   x = np.zeros((n,1))
   s = np.copy(b) # this works because x = 0 is feasible
   y = np.array(mu[0]*(1/s))
   cp = np.zeros((numPts+1,n))
   c = np.matrix(c).T
   for k in range(numPts):
      er = np.linalg.norm(np.vstack((y*s-mu[k], A.T@y - c)))
      while er >= tol:
         dx = np.linalg.solve(A.T@np.diag((y/s).flatten(),0)@A, \
                 c - mu[k]*A.T@(1/s))
         ds = -A@dx
         dy = mu[k]/s - y - np.diag((y/s).flatten(),0)@ds 
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
         er = np.linalg.norm(np.vstack((y*s-mu[k], A.T@y - c)))
      cp[k+1,:] = x.T
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
   #fig.savefig("cpFig.png")
   plt.show()

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
def plot2DTiling(shp,trn):
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
   plt.show()
         
