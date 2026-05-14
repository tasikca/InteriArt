from cpUtilsTheta import *
'''
n = 4
A,b = get2DData(n)
print(A)
c = getRand2DCvectors(A,4)

c = np.array([[0.999,0.001],
              [0.01,0.99],
              [0.95,0.05],
              [0.5,0.5],
              [0.8,0.2],
              [0.5,0.75]])

mask = np.all(c > 0, axis=1)
c = c[mask]
print(c)

#plot2DFlowerAdjust(A, b, c, 0, [0, 0], cmap_name="tab10", ls='-', lw=1.5)

tol = 0.0000001
plot2DFlowerAdjust(
    A, b, c, 0, [0, 0],
    tol = tol,
    title= f"default tol={tol}",
    cmap_name="tab20b",
    fig_name=f"cpFigThetatol_{tol}",
    marker='o',      # Specify the marker shape
    ms=8,            # Markersize
    markevery=1,    # Only plot a marker every 10 points to avoid clutter
    ls='-'           # Ensure the line is still visible
)

'''

from cpUtilsTheta import *

n = 4
A, b = get2DData(n)

c = np.array([[0.999, 0.001],
              [0.01,  0.99],
              [0.95,  0.05],
              [0.5,   0.5],
              [0.8,   0.2],
              [0.5,   0.75]])

mask = np.all(c > 0, axis=1)
c = c[mask]
print("\nFiltered c vectors:\n", c)

tol = 0.001

# --- NEW ANALYSIS BLOCK ---
print("\n--- Path Curvature Analysis ---")
for t in range(len(c)):
    # Generate the path just like plot2DFlowerAdjust does internally
    cp = generatePath(A, b, c[t, :], tol)
    
    # Calculate the summed curvature for this path
    tot_k1, tot_k2 = sumPathCurvature(cp)
    
    print(f"Path {t+1} (c = {c[t, :]}):")
    print(f"  Points generated: {len(cp)}")
    print(f"  Total Kappa 1 (Forward) : {tot_k1:.4f}")
    print(f"  Total Kappa 2 (Backward): {tot_k2:.4f}\n")
# --------------------------

# Finally, plot the flower
plot2DFlowerAdjust(
    A, b, c, 0, [0, 0],
    tol=tol,
    title=f"default tol={tol}",
    cmap_name="tab20b",
    fig_name=f"cpFigThetatol_{tol}",
    marker='o',      
    ms=8,            
    markevery=1,    
    ls='-'           
)


def sumPathCurvature(cp):
    """
    Iterates through a generated path (cp) and sums the curvature
    using calcCurvature from cpUtilsTheta.
    """
    total_kappa1 = 0.0
    total_kappa2 = 0.0

    # We need at least 3 points to calculate curvature
    num_points = len(cp)
    if num_points < 3:
        return 0.0, 0.0

    for i in range(num_points - 2):
        x1 = cp[i, :]
        x2 = cp[i+1, :]
        x3 = cp[i+2, :]

        # Calculate forward (k1) and backward (k2) curvature estimates
        k1, k2 = calcCurvature(x1, x2, x3)

        total_kappa1 += k1
        total_kappa2 += k2

    return total_kappa1, total_kappa2

'''
numPoints = np.array([[4, 4, 4, 4, 4, 4],
                      [6, 6, 4, 4, 4, 4],
                      [6, 6, 6, 4, 6, 4],
                      [6, 6, 6, 4, 6, 4],
                      [6, 6, 8, 4, 8, 6],
                      [12, 12, 12, 4, 10, 6],
                      [16, 18, 18, 4, 14, 10],
                      [16, 20, 20, 4, 22, 14],
                      [22, 24, 28, 4, 26, 18],
                      [30, 36, 44, 4, 40, 28],
                      ])
'''
