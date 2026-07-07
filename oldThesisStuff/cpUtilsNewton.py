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


def calcNewtonPathElement(A,b,c,mu,x,s,y,x1,x2,mu1,mu2):
   #print('newton start', mu1, mu2)
   tol = 10**(-8)    # control tolarence for specific mu values
   er = np.linalg.norm(np.vstack((np.array(y)*np.array(s)-mu, A.T@y - c)))
   while er >= tol:
      D = np.diag((np.array(y) / np.array(s)).flatten())
      top_left = A.T @ D @ A                   
      top_right = -A.T @ np.diag((1/np.array(s)).flatten())  @  np.ones((A.shape[0], 1))
      bottom_left = (x2 - x1).T                 
      bottom_right = np.array([[0]])             

      top = np.hstack([top_left, top_right])     
      bottom = np.hstack([bottom_left, bottom_right]) 

      M  = np.vstack([top, bottom]) 
      #U_top = A.T @ np.diag(np.array(y).flatten(), 0) @ s - mu * A.T @ np.ones((A.shape[0], 1))  # shape (m, 1)
      U_top = -(np.linalg.matmul(A,x) + s - b)
      U_bottom = -(x1-x).T@(x1-x2) #is x1 fine here?
      U = np.vstack([U_top[0:2]+U_top[2:4], U_bottom]) #this NEEEDS help
     
      '''
      print('A',A,'b',b,'c',c)
      print('mu',mu,'x',x,'s',s,'y',y,'x1',x1,'x2',x2,'mu1',mu1,'mu2',mu2)
      print('Utop',U_top[0:2])
      print('Utop',U_top)
      print('(x1+x2)T',((x1+x2)/x-x).T)
      print('(x2-x1)',x1-x2)
      print('Ubot',U_bottom)
      print('M',M)
      print('U',U)
      '''

      dz = np.linalg.solve(M,U)
      
      #split dz into dx and dmu
      dx = np.array(dz[:-1])
      dmu = np.array(dz[-1])
      
      # back calculate s and y from dmu and dx
      ds = -A@dx
      dy = mu/np.array(s) - y - np.diag((np.array(y)/np.array(s)).flatten(),0)@ds

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
        
      #print('mu',mu1,mu2)
      if mu-dmu < min(mu1,mu2):
         gamma = (1/dmu)*-0.5*min(np.abs(mu2-mu),np.abs(mu-mu1))
         #print('mu outside small', min(mu1,mu2))
      elif mu > max(mu1,mu2):
         gamma = (1/dmu)*0.5*min(np.abs(mu1-mu),np.abs(mu-mu2))
         #print('mu outside big', gamma)
      elif mu+dmu < 0:
         #print('mu neg')
         gamma = 0
      else:
         #print('mu fine') 
         gamma = 1

      x  = np.array(x + alpha*dx)
      s  = np.array(s + alpha*ds)
      y  = np.array(y + beta*dy)
      mu = mu - gamma*dmu
      er = np.linalg.norm(np.vstack((y*s-mu, A.T@y - c)))
      er = np.linalg.norm(y*s-mu)
      #print('dx',dx,'dmu',gamma*dmu)
      #print('er',er,'mu',mu)
   return {'x':x, 's':s, 'y':y, 'mu':mu}

def divideMuInterval(A,b,c,mu1,mu2,cpDict,maxKappa,xMidDict):
   # assumes mu1 > mu2, probably should add a check
   muMid = 0.5*(mu1+mu2)
   xMid = 0.5*(cpDict[mu1]['x'] + cpDict[mu2]['x'])
   sMid = 0.5*(cpDict[mu1]['s'] + cpDict[mu2]['s'])
   yMid = 0.5*(cpDict[mu1]['y'] + cpDict[mu2]['y'])
   #cpDict[muMid] = calcPathElement(A,b,c,muMid,xMid,sMid,yMid)
   xMidDict[muMid] = {'x': xMid}
   xtemp = xMid 

   # get intinal midpoint guess via "normal" inteior point
   tol = 10**(-8)
   theta0,theta1 = mu1, mu2 
   tempBi = calcPathElement(A,b,c,muMid,xMid,sMid,yMid)
   xMid = (tempBi['x']+xMid)/2
   sMid = (tempBi['s']+sMid)/2
   yMid = (tempBi['y']+yMid)/2

   x1 = cpDict[mu2]['x']
   x2 = cpDict[mu1]['x']
   '''
   # Newton steps
   while np.abs(np.dot((x1-x2).T,xtemp-xMid).item()) > tol:
      #print('err',np.abs(np.dot((x1-x2).T,xtemp-xMid).item()))
      muMid = (theta0 + theta1)/2
      #print('muMid',muMid)
      
      tempBi = calcNewtonPathElement(A,b,c,muMid,xMid,sMid,yMid,x1,x2,theta0,theta1)
      #if np.dot((x1-x2).T,tempBi['x']-xMid).item() - np.dot((x1-x2).T,xtemp-xMid).item() < 0:
      if np.dot((x1-x2).T,tempBi['x']-xMid).item() < 0:
         theta0 = muMid
      else:
         theta1 = muMid
      muMid = tempBi['mu'].item()
      
      xMid = tempBi['x']
      print('xMid',xMid)
      print('muMid',muMid)
      sMid = tempBi['s']
      yMid = tempBi['y']
   '''
  
   tempNewt = calcNewtonPathElement(A,b,c,muMid,xMid,sMid,yMid,x1,x2,theta0,theta1)
   tempDict = {'x':tempBi['x'],'s':tempBi['s'],'y':tempBi['y']} 
   #print(tempBi)
   xtemp = tempNewt['x']
   muMid = float(tempNewt['mu'])

   cpDict[muMid] = tempBi
   #print('xtemp',xtemp,'xMid',xMid)
   #print(np.linalg.norm(xMid-xtemp))
   #estimate curvature
   kappa1,kappa2 = np.linalg.norm(xMid-xtemp), np.linalg.norm(xMid-xtemp)
   if kappa1 > maxKappa:
      divideMuInterval(A,b,c,mu1,muMid,cpDict,maxKappa,xMidDict)
   if kappa2 > maxKappa:
      divideMuInterval(A,b,c,muMid,mu2,cpDict,maxKappa,xMidDict)
   return


def generatePath(A,b,c):
   maxKappa = 0.1           # maximum curvature estimate
   muSmallest = np.exp(-16) # surrogate for mu = zero
   #muSmallest = 1  # surrogate for mu = zero
   muLargest  = 10         # surrogate for mu = infty
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
