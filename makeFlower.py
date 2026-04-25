from cpUtilsTheta import *
n = 4
A,b = get2DData(n)
print(A)
c = getRand2DCvectors(A,4)

c = np.array([[0.95,0.05],
              [0.05,0.95],
              [0.5,0.5],
              [0.55,0.75]])

mask = np.all(c > 0, axis=1)
c = c[mask]
print(c)

#plot2DFlowerAdjust(A, b, c, 0, [0, 0], cmap_name="tab10", ls='-', lw=1.5)

plot2DFlowerAdjust(
    A, b, c, 0, [0, 0], 
    cmap_name="tab20b", 
    marker='o',      # Specify the marker shape
    ms=4,            # Markersize
    markevery=1,    # Only plot a marker every 10 points to avoid clutter
    ls='-'           # Ensure the line is still visible
)

#plot2DFlowerAdjust(A,b,c,0,[0,0],"b")

