from cpUtilsBiFix import get2DData, generatePath, getRand2DCvectors
import matplotlib.pyplot as plt
from pyscript import display, document

def update_plot(event=None):
    # 1. Get values from UI
    num_facets = int(document.getElementById("numFacets").value)
    num_paths = int(document.getElementById("numPaths").value)
    hex_color = document.getElementById("pathColor").value
    
    # Update UI labels
    document.getElementById("facetValue").innerText = str(num_facets)
    document.getElementById("pathValue").innerText = str(num_paths)

    # 2. Run your mathematical logic
    A, b = get2DData(num_facets)
    c_vectors = getRand2DCvectors(A, num_paths)

    # 3. Create the Visualization
    plt.close('all') # Clear previous plots
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for path in c_vectors:
        cp, xMid = generatePath(A, b, path)
        ax.plot(cp[:, 0], cp[:, 1], color=hex_color, marker="*", ms=3, alpha=0.6)
        ax.plot(xMid[:, 0], xMid[:, 1], "o", ms=3, color=hex_color)

    ax.set_title(f"Central Path for {num_facets}-sided Polygon")
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout() # Ensures the plot fits perfectly in the container

    # 4. Render to HTML
    plot_target = document.getElementById("plot-target")
    plot_target.innerHTML = "" 
    display(fig, target="plot-target")

# Initialize the first plot
update_plot()
