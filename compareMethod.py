import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor
from cpUtilsTheta import *

def get_old_path_dict(A, b, c, tol):
    maxKappa = tol
    muSmallest = np.exp(-16)
    muLargest = 10
    m, n = np.shape(A)
    c_mat = np.matrix(c).T
    x = np.zeros((n,1))
    s = np.copy(b)
    y = np.array(muLargest*(1/s))
    
    cpDict = {}
    cpDict[muLargest] = {'x':np.matrix(x), 's':np.matrix(s), 'y':np.matrix(y)}
    cpDict[muSmallest] = calcPathElement(A,b,c_mat,muSmallest,x,s,y)
    
    divideMuInterval(A,b,c_mat,muLargest,muSmallest,cpDict,maxKappa)
    return cpDict

def get_new_path_dict(A, b, c, tol, theta=0.5):
    maxDist = tol
    muSmallest = np.exp(-16)
    muLargest = 10
    muInf = muLargest + 1
    m, n = np.shape(A)
    c_mat = np.matrix(c).T
    
    x = np.zeros((n,1))
    s = np.copy(b)
    y = np.array(muLargest*(1/s))
    
    cpDict = {}
    cpDict[muInf] = {'x':np.matrix(x), 's':np.matrix(s), 'y':np.matrix(y)}
    cpDict[muLargest] = calcPathElement(A,b,c_mat,muLargest,x,s,y)
    cpDict[muSmallest] = calcPathElement(A,b,c_mat,muSmallest,x,s,y)
    
    divideNewtonMuInterval(A,b,c_mat,muLargest,muSmallest,theta,cpDict,maxDist)
    return cpDict

def evaluate_path_midpoint_errors(A, b, c, cpDict, theta=0.5):
    sorted_mus = sorted(list(cpDict.keys()), reverse=True)
    c_mat = np.matrix(c).T
    max_err = 0.0
    
    for i in range(len(sorted_mus) - 1):
        mu1 = sorted_mus[i]
        mu2 = sorted_mus[i+1]
        if mu1 > 10.0: continue
            
        x1, s1, y1 = cpDict[mu1]['x'], cpDict[mu1]['s'], cpDict[mu1]['y']
        x2, s2, y2 = cpDict[mu2]['x'], cpDict[mu2]['s'], cpDict[mu2]['y']
        
        muMid = (1 - theta) * mu1 + theta * mu2
        xMid = (1 - theta) * x1 + theta * x2
        sMid = 0.5 * (s1 + s2)
        yMid = 0.5 * (y1 + y2)
        
        actual = calcPathElement(A, b, c_mat, muMid, xMid, sMid, yMid)
        err = np.linalg.norm(xMid - actual['x'])
        if err > max_err: max_err = err
            
    return max_err

# Helper functions for multiprocessing that pass through the path index (c_idx)
def process_old_tol(args):
    A, b, c, tol, c_idx = args
    cpDict = get_old_path_dict(A, b, c, tol)
    n_points = len(cpDict)
    max_err = evaluate_path_midpoint_errors(A, b, c, cpDict)
    return c_idx, tol, n_points, max_err, cpDict

def process_new_tol(args):
    A, b, c, tol, theta, c_idx = args
    cpDict = get_new_path_dict(A, b, c, tol, theta)
    n_points = len(cpDict)
    max_err = evaluate_path_midpoint_errors(A, b, c, cpDict, theta)
    return c_idx, tol, n_points, max_err, cpDict

def run_experiments():
    A, b = get2DData(4)
    
    # Define an array of highly distinct c-vectors targeting different geometry
    c_vecs = [
        np.array([0.999, 0.001]),    # Original off-axis path
        np.array([0.55, 0.45]),   # Quadrant 2, steeper approach
        np.array([1, 0.2])    # Quadrant 4, near vertical approach
    ]
    num_paths = len(c_vecs)
    
    max_workers = 10 
    print(f"Running parallel sweeps for {num_paths} paths utilizing {max_workers} cores...")
    
    tols_old = np.linspace(0.05, 0.5, 20) 
    tols_new = np.linspace(0.001, 0.5, 20) 
    
    # Pack arguments. Notice we iterate tols in the inner loop so results return ordered.
    old_args = [(A, b, c, t, i) for i, c in enumerate(c_vecs) for t in tols_old]
    new_args = [(A, b, c, t, 0.5, i) for i, c in enumerate(c_vecs) for t in tols_new]

    # Execute Map
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        old_results = list(executor.map(process_old_tol, old_args))
        new_results = list(executor.map(process_new_tol, new_args))

    # Initialize data storage partitioned by path
    old_data = {i: {'n_points': [], 'errs': [], 'dicts': []} for i in range(num_paths)}
    new_data = {i: {'n_points': [], 'errs': [], 'dicts': []} for i in range(num_paths)}

    # Unpack ordered results into path dictionaries
    for res in old_results:
        c_idx, tol, n, err, cpDict = res
        old_data[c_idx]['n_points'].append(n)
        old_data[c_idx]['errs'].append(err)
        old_data[c_idx]['dicts'].append(cpDict)

    for res in new_results:
        c_idx, tol, n, err, cpDict = res
        new_data[c_idx]['n_points'].append(n)
        new_data[c_idx]['errs'].append(err)
        new_data[c_idx]['dicts'].append(cpDict)

    # ================= PLOTTING =================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Generate distinct colors for each path
    colors = plt.cm.tab10.colors

    for i in range(num_paths):
        c_color = colors[i % len(colors)]
        
        # Plot 1: Predictability of Point Generation
        axs[0].plot(tols_old, old_data[i]['n_points'], linestyle='--', marker='x', color=c_color, alpha=0.6, markersize=4)
        axs[0].plot(tols_new, new_data[i]['n_points'], linestyle='-', marker='o', color=c_color, markersize=4)
        
        # Plot 2: Efficiency (Apples-to-Apples metric)
        axs[1].plot(old_data[i]['errs'], old_data[i]['n_points'], linestyle='', marker='x', color=c_color, alpha=0.6)
        axs[1].plot(new_data[i]['errs'], new_data[i]['n_points'], linestyle='', marker='o', color=c_color)
        
        # Plot 3: Visual Spatial Comparison
        target_N = 50
        idx_old = (np.abs(np.array(old_data[i]['n_points']) - target_N)).argmin()
        idx_new = (np.abs(np.array(new_data[i]['n_points']) - target_N)).argmin()
        
        cpDict_old = old_data[i]['dicts'][idx_old]
        cpDict_new = new_data[i]['dicts'][idx_new]
        
        old_mat = np.array([np.asarray(val['x']).flatten() for val in dict(sorted(cpDict_old.items())).values()])
        new_mat = np.array([np.asarray(val['x']).flatten() for val in dict(sorted(cpDict_new.items())).values()])
        
        axs[2].plot(old_mat[:, 0], old_mat[:, 1], linestyle='--', marker='x', color=c_color, alpha=0.6)
        axs[2].plot(new_mat[:, 0], new_mat[:, 1], linestyle='-', marker='o', color=c_color)

    # Clean formatting and custom legend handling
    axs[0].set_title('Control: Parameter vs Number of Points')
    axs[0].set_xlabel('Input Tolerance Parameter')
    axs[0].set_ylabel('Number of Points on Path')
    axs[0].set_yscale('log')
    
    axs[1].set_title('Efficiency: Actual Max Error vs Total Points')
    axs[1].set_xlabel('Actual Maximum Spatial Error (Midpoint Metric)')
    axs[1].set_ylabel('Total Points Required')
    axs[1].set_yscale('log')
    
    axs[2].set_title('Spatial Distribution Comparison (N ≈ 50)')
    axs[2].set_aspect('equal', 'box')
    
    # Custom Legend
    custom_lines = [
        Line2D([0], [0], linestyle='--', marker='x', color='black', alpha=0.6, label='Old Method (Curvature)'),
        Line2D([0], [0], linestyle='-', marker='o', color='black', label='New Method (Distance)')
    ]
    for i in range(num_paths):
        custom_lines.append(Line2D([0], [0], color=colors[i % len(colors)], lw=4, label=f'Path {i+1}'))

    # Apply the custom legend to the first plot (acts as a master legend)
    axs[0].legend(handles=custom_lines, loc='upper right')

    plt.tight_layout()
    plt.savefig('method_comparison_multipath.png', dpi=300)
    print("Saved plot to 'method_comparison_multipath.png'.")
    plt.show()

if __name__ == "__main__":
    run_experiments()

