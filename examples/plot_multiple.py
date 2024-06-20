#!/usr/bin/python3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import json
import pandas as pd
import sys  
sys.path.insert(1, '/home/marguerite/Documents/lab/TrajoptMPCReference')
import overloading


type_cost='urdf'
id=3
def get_error_from_file():
    df = pd.read_pickle(f'../data/{id}/results.plk')
    df = df.transpose()
    column_names = ['comp_time', 'J', 'Jx', 'Ju', 'error', 'E', 'exit_sqp', 'exit_soft', 'outer_iter', 'sqp_iter']
    df.columns = column_names
    df['Error in x']= df["error"].apply(lambda x: np.abs(x[0]))
    df['Error in y']= df["error"].apply(lambda x: np.abs(x[0]))
    df['Error Norm']=np.sqrt(df['Error in x']**2 + df['Error in y']**2 )
    return df['Error Norm'].values
#----------------------------------------------Plot error with xg----------------------------------------------------------
def plot_error_tot_feasible_set():
    max_radius=2
    n=20
    xs = np.linspace(-max_radius, max_radius, n)
    ys = np.linspace(-max_radius, max_radius, n)
    xgs = [[x, y, 0, 0] for x in xs for y in ys if x**2 + y**2 <= max_radius**2] # Filter out points that are inside the square but not the cirle 
    xs= [x[0] for x in xgs]
    ys= [x[1] for x in xgs]
    # Load results
    # with open(f'results/{type_cost}/results_xg_err.json', 'r') as f:
    #     error_data = json.load(f)
    
    error_data=get_error_from_file()
    

    error_array = np.array(error_data)
    print(error_array)
    error_norms = np.linalg.norm(error_array, axis=1)
    grid_size = 20  # Adjust as needed for resolution
    x_grid = np.linspace(-2, 2, grid_size)
    y_grid = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    mask = np.sqrt(X**2 + Y**2) <= 2

    # Create a grid to store the error values
    error_grid = np.full_like(X, np.nan)
    valid_xs = []
    valid_ys = []
    valid_errors = []
    # Interpolate error values onto the grid
    for i in range(len(xs)):
        # Find the closest point on the grid to (xs[i], ys[i])
        idx = np.argmin((X - xs[i])**2 + (Y - ys[i])**2)
        # Check if the point is inside the circle
        if mask.flat[idx]:
            # Map the error to the corresponding grid point
            error_grid.flat[idx] = error_norms[i]
            valid_xs.append(xs[i])
            valid_ys.append(ys[i])
            valid_errors.append(error_norms[i])
    plt.figure(figsize=(8, 6))
    sns.heatmap(error_grid, cmap='viridis', cbar=True, square=True, xticklabels=np.round(x_grid, 2), yticklabels=np.round(y_grid, 2), vmin=0.01)
    plt.title('Norm of the error for different initial conditions, 2 link, urdf cost, N=10, approx hessian')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'results/{type_cost}/error_xg_hess1')



#----------------------------------------------Plot time with xg----------------------------------------------------------

def plot_time_tot_feasible_set():
    # with open(f'results/{type_cost}/results_xg_time.json', 'r') as f:
    #     time_data = json.load(f)
    max_radius=2
    n=20
    xs = np.linspace(-max_radius, max_radius, n)
    ys = np.linspace(-max_radius, max_radius, n)
    xgs = [[x, y, 0, 0] for x in xs for y in ys if x**2 + y**2 <= max_radius**2] # Filter out points that are inside the square but not the cirle 
    xs= [x[0] for x in xgs]
    ys= [x[1] for x in xgs]
    
    grid_size = 20  # Adjust as needed for resolution
    x_grid = np.linspace(-2, 2, grid_size)
    y_grid = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    mask = np.sqrt(X**2 + Y**2) <= 2

    time_array = np.array(time_data)
    time_grid = np.full_like(X, np.nan)
    valid_times = []

    # Interpolate time values onto the grid
    for i in range(len(xs)):
        # Find the closest point on the grid to (xs[i], ys[i])
        idx = np.argmin((X - xs[i])**2 + (Y - ys[i])**2)
        # Check if the point is inside the circle
        if mask.flat[idx]:
            # Map the time to the corresponding grid point
            time_grid.flat[idx] = time_array[i]
            valid_times.append(time_array[i])

    plt.figure(figsize=(8, 6))
    sns.heatmap(time_grid, cmap='viridis', cbar=True, square=True,xticklabels=np.round(x_grid, 2), yticklabels=np.round(y_grid, 2), vmin=min(time_array), vmax=max(time_array))
    plt.title('Computational time for different initial conditions, 2 link, urdf cost, N=10, approx hessian')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'results/{type_cost}/time_xg_hess1')


#------------------------------------------Plot error with xg => Local set----------------------------------------------------------
def plot_error_local_set():
    # Sweep feasible_set 
    max_radius = 2
    n =20
    # xs = np.linspace(-1.5, -0.5, n)
    # ys = np.linspace(1, 2, n)
    xs = np.linspace(-1.5, -0.5, n)
    ys = np.linspace(-2, -1, n)
    xgs = [[x, y, 0, 0] for x in xs for y in ys if x**2 + y**2 <= max_radius**2] # Filter out points that are inside the square but not the circle
    xs = [x[0] for x in xgs]
    ys = [x[1] for x in xgs]

    # Load error data
    with open(f'results/{type_cost}/results_xg_err.json', 'r') as f:
        error_data = json.load(f)

    error_array = np.array(error_data)
    error_norms = np.linalg.norm(error_array, axis=1)
    grid_size = 30 # Adjust as needed for resolution
    x_grid = np.linspace(-1.5, -0.5, grid_size)
    y_grid = np.linspace(-1, -2, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    mask = np.sqrt(X**2 + Y**2) <= 2

    # Create a grid to store the error values
    error_grid = np.full_like(X, np.nan)
    valid_xs = []
    valid_ys = []
    valid_errors = []

    # Interpolate error values onto the grid
    for i in range(len(xs)):
        # Find the closest point on the grid to (xs[i], ys[i])
        idx = np.argmin((X - xs[i])**2 + (Y - ys[i])**2)
        # Check if the point is inside the circle
        if mask.flat[idx]:
            # Map the error to the corresponding grid point
            error_grid.flat[idx] = error_norms[i]
            valid_xs.append(xs[i])
            valid_ys.append(ys[i])
            valid_errors.append(error_norms[i])

    plt.figure(figsize=(8, 6))
    sns.heatmap(error_grid, cmap='viridis', cbar=True, square=True, xticklabels=np.round(x_grid, 2), yticklabels=np.round(y_grid, 2), vmin=0.01)
    plt.title('Norm of the error for different initial conditions, 2 link, urdf cost, N=10, no hessian')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'results/urdf/error_xg_top_left')
    plt.show()


# plot_error_local_set()
plot_error_tot_feasible_set()
# plot_time_tot_feasible_set()