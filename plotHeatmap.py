import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from cpUtilsTheta import * 

# 1. Define a wrapper function for a single task
def compute_path_length(params):
    """
    Worker function to compute path length for a single (cx, cy) pair.
    params: (cx, cy, A, b, tol)
    """
    cx, cy, A, b, tol = params
    try:
        path = generatePath(A, b, [cx, cy], tol=tol)
        return len(path)
    except Exception:
        return 0

def main():
    # Setup parameters
    n_facets = 4
    A, b = get2DData(n_facets)
    tol = 0.0001
    grid_res = 500  # We can handle higher resolution now!
    
    cx_range = np.linspace(0.01, 1.0, grid_res)
    cy_range = np.linspace(0.01, 1.0, grid_res)
    
    # 2. Prepare the task list (flatten the grid)
    tasks = []
    for cy in cy_range:
        for cx in cx_range:
            tasks.append((cx, cy, A, b, tol))

    # 3. Execute in parallel
    # Pool() automatically uses the number of available CPU cores
    print(f"Starting parallel computation with {grid_res**2} tasks...")
    with Pool() as pool:
        results = pool.map(compute_path_length, tasks)

    # 4. Reshape results back into a 2D grid
    heatmap_data = np.array(results).reshape((grid_res, grid_res))

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(cx_range, cy_range, heatmap_data, shading='auto', cmap='magma')
    plt.colorbar(label='Number of Points in Path')
    plt.xlabel('$c_x$ value')
    plt.ylabel('$c_y$ value')
    plt.title(f'Central Path Point Density (n={n_facets}, tol={tol})')
    plt.savefig(f'central_path_density_grid_{grid_res}_tol_{tol}.heatmap.svg')
    plt.show()

if __name__ == "__main__":
    main()

