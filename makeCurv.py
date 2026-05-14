from cpUtilsTheta import *

n = 4
A, b = get2DData(n)
print("Matrix A:\n", A)

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
