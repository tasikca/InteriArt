from __future__ import division # safety with double division
import numpy as np
from numpy import linalg
import scipy as sp
import matplotlib.pyplot as plt

def get2DData(numFacets):
   theta = np.array(np.arange(numFacets)*2*np.pi/numFacets)
   return np.vstack((np.cos(theta), np.sin(theta))).T, np.ones((numFacets,1))

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

def getNiceRand2DCvectors(A,numPathsPerLeaf):
   rng=np.random.default_rng()
   m,n = np.shape(A)
   cVec = np.zeros((m*numPathsPerLeaf,2))
   edgeMean = 0.03 + ((0.08 - 0.03)/(7-3))*(m-3)
   niceMean = 0.04/np.log(m)
   tmpA = np.vstack((A,A[0,:])) # removes the need for modular calculations
   for i in range(m):
      c = np.zeros((numPathsPerLeaf,2))
      # Edge paths 
      alpha = niceMean + (niceMean/2)*rng.standard_normal()
      c[0,:] = (1-alpha)*tmpA[i,:]+alpha*tmpA[i+1,:]
      alpha = niceMean + (niceMean/2)*rng.standard_normal()
      c[1,:] = (1-alpha)*tmpA[i+1,:]+alpha*tmpA[i,:]
      # Interior paths (a bit a redundant term, no?)
      alpha = 2*edgeMean + (1-4*edgeMean)*rng.random(size=numPathsPerLeaf-2)
      c[2::,:] = (np.vstack((1-alpha, alpha)).T)@tmpA[i:i+2,:]
      cVec[i*numPathsPerLeaf:(i+1)*numPathsPerLeaf, :] = c
   return cVec


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

def calcNewtonPathElement(A,b,c,mu,x,s,y,x1,x2,mu1,mu2,theta):
   m,n = np.shape(A)

   tol = 10**(-8)    # control tolarence for specific mu values
   er = np.linalg.norm((np.array(y)*np.array(s)-mu))
   
   itr = 0
   while er >= tol and itr <= 1000:
      itr += 1

      M11 = A.T @ np.diag((np.array(y)/np.array(s)).flatten()) @ A
      M12 = -A.T @ (np.array(1/s))
      M21 = (x2 - x1).T
      M22 = np.zeros((1, 1))

      M_top = np.hstack((M11, M12))
      M_bot = np.hstack((M21, M22))
      M = np.vstack((M_top, M_bot))

      U1 = -mu * (A.T @ np.array((1/s))) + c
      U2_scalar = theta * (np.linalg.norm(x2 - x1)**2) - ((x2 - x1).T @ (x - x1)).item()
      U2 = np.array([[U2_scalar]])

      U = np.vstack((U1, U2))

      #solve Mz=U
      try:
         dz = np.linalg.solve(M, U)
      except np.linalg.LinAlgError:
         print("M is a bachelor")
         break

      dx = dz[:-1]
      dmu = dz[-1,0]

      ds = -A @ dx
      dy = np.diag(np.array(1/s).flatten())*(np.ones((m,1))
            *(mu-dmu)-(np.diag(np.array(y).flatten())*(s+ds)))


      # find a reasonable step size
      # the world breaks if s or y become negative
      negIndS = np.nonzero(ds < 0)
      if len(negIndS[0]) == 0:
         alpha = 1
      else:
         alpha = min(1, 0.9*np.min(-s[negIndS]/ds[negIndS]))

      negIndY = np.nonzero(dy < 0)
      if len(negIndY[0]) == 0:
         beta = 1
      else:
         beta = min(1, 0.9*np.min(-y[negIndY]/dy[negIndY]))

      if (mu-np.abs(dmu) < min(mu1,mu2)):
         gamma = 0.1*min(np.abs(mu-mu1), np.abs(mu-mu2))/dmu
      elif (mu+np.abs(dmu) > max(mu1,mu2)):
         gamma = 0.1*min(np.abs(mu-mu1), np.abs(mu-mu2))/dmu
      else:
         gamma = 1

      x  = np.array(x + alpha*dx)
      s  = np.array(s + alpha*ds)
      y  = np.array(y + beta*dy)
      mu = mu-gamma*dmu

      er = np.linalg.norm((y*s-mu))
   return {'x':x, 's':s, 'y':y, 'mu':mu}

def divideNewtonMuInterval(A,b,c,mu1,mu2,theta,cpDict,maxDist):
   x1 = cpDict[mu1]['x']
   x2 = cpDict[mu2]['x']

   muMid = (1-theta)*mu1 + theta*mu2
   xMid = (1-theta)*x1 + theta*x2
   sMid = 0.5*(cpDict[mu1]['s'] + cpDict[mu2]['s'])
   yMid = 0.5*(cpDict[mu1]['y'] + cpDict[mu2]['y'])

   tempDict = calcNewtonPathElement(A,b,c,muMid,xMid,sMid,yMid,x1,x2,mu1,mu2,theta)
   muNew = tempDict['mu'].item()
   cpDict[muNew] = tempDict

   # determine whether to keep dividing
   if np.linalg.norm(xMid-cpDict[muNew]['x']) > maxDist:
      divideNewtonMuInterval(A,b,c,mu1,muNew,theta,cpDict,maxDist)
      divideNewtonMuInterval(A,b,c,muNew,mu2,theta,cpDict,maxDist)
   return



def generatePath(A,b,c):
   maxDist = 0.001         # maximum distance error to path
   theta = 0.5             # convex combo of x1 and x2
   muSmallest = np.exp(-16) # surrogate for mu = zero
   muLargest  = 25          # surrogate for mu = infty
   muInf = muLargest + 1    # this makes (0,0) the center
   m,n = np.shape(A)
   c = np.matrix(c).T # Make sure to have the correct vector form

   # initialize at the analytic center
   x = np.zeros((n,1))
   s = np.copy(b) # this works because x = 0 is feasible
   y = np.array(muLargest*(1/s)) # muLargest is a starting surrogate for infty

   # instantiate the central path dictionary with keys being mu values
   cpDict = {}
   cpDict[muInf] = {'x':np.matrix(x), 's':np.matrix(s), 'y':np.matrix(y)}

   cpDict[muLargest] = calcPathElement(A,b,c,muLargest,x,s,y)
   cpDict[muSmallest] = calcPathElement(A,b,c,muSmallest,x,s,y)

   divideNewtonMuInterval(A,b,c,muLargest,muSmallest,theta,cpDict,maxDist)
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

