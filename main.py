import io
import base64
import asyncio
from cpUtilsThetaWeb import get2DData, generatePath, getRand2DCvectors, getNiceRand2DCvectors
import matplotlib.pyplot as plt
from pyscript import display, document

# Global variable to store the figure for downloading
current_fig = None 

async def update_plot(event=None):
    global current_fig
    
    # 1. UI Feedback: Disable button and show "Generating"
    btn = document.getElementById("generate-btn")
    btn.innerText = "Generating..."
    btn.disabled = True
    btn.classList.add("opacity-50", "cursor-not-allowed")
    
    #So the generating... can show
    await asyncio.sleep(0.01)

    try:
        # Get UI values
        numFacets = int(document.getElementById("numFacets").value)
        numPaths = int(document.getElementById("numPaths").value)
        plotPoints = document.getElementById("pointToggle").checked
        # hexColor = document.getElementById("pathColor").value

        # Run the math from your utility files
        A, b = get2DData(numFacets)
        c = getNiceRand2DCvectors(A, numPaths)

        # Matplotlib plotting
        plt.close('all') 
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for path in c:
            cp = generatePath(A, b, path)
            ax.plot(cp[:, 0], cp[:, 1], color='red', linewidth=5, linestyle='-')
            
            if plotPoints:
               ax.plot(cp[:, 0], cp[:, 1], color='black', marker="o", ms=5, linestyle='', alpha=0.4)

        # ax.set_title(f"{numFacets}-gon central path flower")
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        fig.tight_layout()

        # Update the global figure reference
        current_fig = fig

        # Render to the page
        plot_target = document.getElementById("plot-target")
        plot_target.innerHTML = "" 
        display(fig, target="plot-target")

        # Show the download button
        document.getElementById("download-btn").classList.remove("hidden")

    finally:
        # 2. Reset the button state
        btn.innerText = "Generate Graph"
        btn.disabled = False
        btn.classList.remove("opacity-50", "cursor-not-allowed")

def download_svg(event=None):
    global current_fig
    if current_fig is None:
        return

    user_name = document.getElementById("user-name").value.strip()
    filename = f"{user_name if user_name else 'user'}_central_path.svg"

    # 1. Identify and hide point markers on the active axes[cite: 10]
    ax = current_fig.gca()
    original_states = []
    
    for line in ax.get_lines():
        marker = line.get_marker()
        original_states.append((line, marker))
        if marker not in ['', 'None', None]:
            line.set_marker('None')

    # 2. Save the cleaned figure to buffer[cite: 10]
    buf = io.BytesIO()
    current_fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    
    # 3. Restore the markers for the UI[cite: 10]
    for line, marker in original_states:
        line.set_marker(marker)

    # 4. Base64 encode and trigger download[cite: 10]
    svg_data = buf.read().decode('utf-8')
    encoded_data = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    uri = f"data:image/svg+xml;base64,{encoded_data}"

    a = document.createElement("a")
    a.href = uri
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)

# Initial run (optional)
# asyncio.ensure_future(update_plot())
