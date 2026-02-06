from cpUtilsBiWeb import get2DData, generatePath, getRand2DCvectors
import matplotlib.pyplot as plt
from pyscript import display, document

def update_plot(event=None):
    # get UI values
    numFacets = int(document.getElementById("numFacets").value)
    numPaths = int(document.getElementById("numPaths").value)
    hexColor = document.getElementById("pathColor").value
    
    # update UI
    document.getElementById("facetValue").innerText = str(numFacets)
    document.getElementById("pathValue").innerText = str(numPaths)

    # cpUtils 
    A,b = get2DData(numFacets)
    c = getRand2DCvectors(A, numPaths)

    # plotting
    plt.close('all') # clear old plots
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for path in c:
        cp, xMid = generatePath(A, b, path)
        cpPlot = ax.plot(cp[:, 0], cp[:, 1], color=hexColor, marker="o", ms=3, alpha=0.6)
        xMidPlot = ax.plot(xMid[:, 0], xMid[:, 1], "o", ms=3, color="k")

    ax.set_title(f"{numFacets}-gon central path flower")
    ax.set_aspect('equal', adjustable='box')
    ax.legend([cpPlot,xMidPlot], ['accepted points on cp', 'bisec comparison points'])
    #ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()

    # render to HTML
    plot_target = document.getElementById("plot-target")
    plot_target.innerHTML = "" 
    display(fig, target="plot-target")

# Initialize the first plot
update_plot()
