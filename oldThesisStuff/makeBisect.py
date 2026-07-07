from cpUtilsBiFix import *

#A = np.array([[1,0],[0,1],[-1,1],[1,1]])
#b = np.array([[1],[1],[1],[1]])

A,b = get2DData(4)
#c = np.array([[1],[0.001]])
#c = np.array([1, 0.2])
c = getRand2DCvectors(A,4)
#c = np.array([[1, 1],[1,0]])

print(A)
print(b)
print(c)

for path in c:
   cp, xMid = generatePath(A,b,path)
   print('nCp', np.size(cp,axis=0))
  
   numC = len(c)
   cmap = plt.get_cmap('tab10')
   clr = cmap(np.random.randint(1,numC)/numC)
   #print('nXmid',np.size(xMid,axis=0))
   plt.plot(cp[:,0], cp[:,1],color=clr,marker="*",ms=3)
   #print(xMid)
   plt.plot(xMid[:,0],xMid[:,1],"o",ms=3,color=clr)
   #print(cp)

plt.show()

#calcPathElement(A,b,c,np.array([0.5,0.5]),b,

