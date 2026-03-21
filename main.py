import io
import base64
from cpUtilsBiWeb import get2DData, generatePath, getRand2DCvectors
import matplotlib.pyplot as plt
from pyscript import display, document

# Global variable to hold our current figure so the download button can access it
current_fig = None 

def update_plot(event=None):
    global current_fig
    
    # 1. Loading State: Disable button and change text
    btn = document.getElementById("generate-btn")
    btn.innerText = "Generating..."
    btn.disabled = True
    btn.classList.add("opacity-50", "cursor-not-allowed") # Tailwind classes for disabled look

    try:
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
            ax.plot(cp[:, 0], cp[:, 1], color=hexColor, marker="o", ms=3, alpha=0.6)
            ax.plot(xMid[:, 0], xMid[:, 1], "o", ms=3, color="k")

        ax.set_title(f"{numFacets}-gon central path flower")
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()

        # Save the figure globally so we can download it later
        current_fig = fig

        # render to HTML
        plot_target = document.getElementById("plot-target")
        plot_target.innerHTML = "" 
        display(fig, target="plot-target")

        # Un-hide the download button now that a plot exists
        document.getElementById("download-btn").classList.remove("hidden")

    finally:
        # 2. Reset the button state regardless of success or failure
        btn.innerText = "Generate Graph"
        btn.disabled = False
        btn.classList.remove("opacity-50", "cursor-not-allowed")

def download_svg(event=None):
    global current_fig
    if current_fig is None:
        return

    # Save figure to an in-memory byte buffer
    buf = io.BytesIO()
    current_fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    
    # Read the SVG data and encode it to base64
    svg_data = buf.read().decode('utf-8')
    encoded_data = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    
    # Create a data URI
    uri = f"data:image/svg+xml;base64,{encoded_data}"

    # Use the DOM to create a temporary anchor tag and trigger a download
    a = document.createElement("a")
    a.href = uri
    a.download = "central_path.svg"
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)

# Initialize the first plot
update_plot()
