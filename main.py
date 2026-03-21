import io
import base64
import asyncio
from cpUtilsBiWeb import get2DData, generatePath, getRand2DCvectors
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
    
    # This tiny sleep allows the browser to actually paint the "Generating..." text 
    # before the CPU-heavy math blocks the thread.
    await asyncio.sleep(0.01)

    try:
        # Get UI values
        numFacets = int(document.getElementById("numFacets").value)
        numPaths = int(document.getElementById("numPaths").value)
        # hexColor = document.getElementById("pathColor").value

        # Run the math from your utility files
        A, b = get2DData(numFacets)
        c = getRand2DCvectors(A, numPaths)

        # Matplotlib plotting
        plt.close('all') 
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for path in c:
            cp, xMid = generatePath(A, b, path)
            ax.plot(cp[:, 0], cp[:, 1], color='red', linewidth=10)
            # ax.plot(cp[:, 0], cp[:, 1], color=hexColor, marker="o", ms=3, alpha=0.6)
            # ax.plot(xMid[:, 0], xMid[:, 1], "o", ms=3, color="k")

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

    # Get the name from the input field
    user_name = document.getElementById("user-name").value.strip()
    # Default to 'user' if the field is empty
    filename_prefix = user_name if user_name else "user"
    filename = f"{filename_prefix}_central_path.svg"

    # Save figure to buffer
    buf = io.BytesIO()
    current_fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    
    # Encode to Base64
    svg_data = buf.read().decode('utf-8')
    encoded_data = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    uri = f"data:image/svg+xml;base64,{encoded_data}"

    # Trigger browser download
    a = document.createElement("a")
    a.href = uri
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)

# Initial run (optional)
# asyncio.ensure_future(update_plot())
