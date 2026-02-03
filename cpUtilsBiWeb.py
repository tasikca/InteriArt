from __future__ import division # safety with double division
import numpy as np
from numpy import linalg
import scipy as sp
import matplotlib.pyplot as plt

def get2DData(numFacets):
   theta = np.array(np.arange(numFacets)*2*np.pi/numFacets)
   return np.vstack((np.cos(theta), np.sin(theta))).T, np.ones((numFacets,1))


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

def divideMuInterval(A,b,c,mu1,mu2,cpDict,maxKappa,xMidDict):
   # assumes mu1 > mu2, probably should add a check
   muMid = 0.5*(mu1+mu2)
   xMid = 0.5*(cpDict[mu1]['x'] + cpDict[mu2]['x'])
   sMid = 0.5*(cpDict[mu1]['s'] + cpDict[mu2]['s'])
   yMid = 0.5*(cpDict[mu1]['y'] + cpDict[mu2]['y'])
   #cpDict[muMid] = calcPathElement(A,b,c,muMid,xMid,sMid,yMid)
   xMidDict[muMid] = {'x': xMid}
  

   # bisection vibes
   tol = 10**(-8)
   a,b = mu1, mu2 
   tempBi = calcPathElement(A,b,c,muMid,xMid,sMid,yMid)
   xtemp = tempBi['x']
   x1 = cpDict[mu1]['x']
   x2 = cpDict[mu2]['x']

   while np.abs(np.dot((x1-x2).T,xtemp-xMid).item()) > tol:
      #print('err',np.abs(np.dot((x1-x2).T,xtemp-xMid).item()))
      muMid = float((a + b)/2)
      #print('muMid',muMid)

      tempBi = calcPathElement(A,b,c,muMid,xMid,sMid,yMid)
      #if np.dot((x1-x2).T,tempBi['x']-xMid).item() - np.dot((x1-x2).T,xtemp-xMid).item() < 0:
      if np.dot((x1-x2).T,tempBi['x']-xMid).item() < 0:
         b = muMid
      else:
         a = muMid
      xtemp = tempBi['x']
 
   cpDict[muMid] = tempBi

   #estimate curvature
   #kappa1,kappa2 = calcDistToMidPoint(cpDict[mu1]['x'],cpDict[mu2]['x'],cpDict[muMid]['x'])
   kappa1,kappa2 = np.linalg.norm(xMid-xtemp), np.linalg.norm(xMid-xtemp)
   if kappa1 > maxKappa:
      divideMuInterval(A,b,c,mu1,muMid,cpDict,maxKappa,xMidDict)
   if kappa2 > maxKappa:
      divideMuInterval(A,b,c,muMid,mu2,cpDict,maxKappa,xMidDict)
   return

#vector rejection
def calcDistToMidPoint(x1,x2,xMu):
   xMid = (x1 + x2)/2
   x = (xMu - xMid).T
   xM1 = xMu - x1
   xM2 = xMu - x2
   xProjx1R = np.linalg.norm(x-(np.dot(x,xM1) / np.linalg.norm(xM1)).item()*xM1)
   xProjx2R = np.linalg.norm(x-(np.dot(x,xM2) / np.linalg.norm(xM2)).item()*xM2)
   xProjx1 = np.linalg.norm(np.dot(x,xMu-x1) / np.linalg.norm(xMu-x1))
   xProjx2 = np.linalg.norm(np.dot(x,xMu-x2) / np.linalg.norm(xMu-x2))
   #print('proj',xMu-x1)
   #print('proj',(np.dot(x,xMu-x1) / np.linalg.norm(xMu-x1)).item()*(xMu-x1))
   return np.max([np.exp(-xProjx1),np.exp(-xProjx1R)]), np.max([np.exp(-xProjx2),np.exp(-xProjx2R)])


def generatePath(A,b,c):
   maxKappa = 0.005           # maximum curvature estimate
   muSmallest = np.exp(-16) # surrogate for mu = zero
   #muSmallest = 1  # surrogate for mu = zero
   muLargest  = 10          # surrogate for mu = infty
   m,n = np.shape(A)
   c = np.matrix(c).T # Make sure to have the correct vector form
   # initialize at the analytic center
   x = np.zeros((n,1))
   s = np.copy(b) # this works because x = 0 is feasible
   y = np.array(muLargest*(1/s)) # muLargest is a starting surrogate for infty
   # instantiate the central path dictionary with keys being mu values
   cpDict = {}
   xMidDict = {}
   cpDict[muLargest] = {'x':np.matrix(x), 's':np.matrix(s), 'y':np.matrix(y)}
   cpDict[muSmallest] = calcPathElement(A,b,c,muSmallest,x,s,y)
   divideMuInterval(A,b,c,muSmallest,muLargest,cpDict,maxKappa,xMidDict)

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
   #print('cp',cp)

   xM = np.zeros((len(xMidDict),2))
   j = 0
   for mu in dict(sorted(xMidDict.items())):
      xM[j,:] = xMidDict[mu]['x'].T
      j = j+1

   return cp, xM

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

