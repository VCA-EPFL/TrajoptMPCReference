#!/usr/bin/python3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import json


type_cost=sys.argv[1]



# Sweep feasible_set 
max_radius=2
n=20
xs = np.linspace(-max_radius, max_radius, n)
ys = np.linspace(-max_radius, max_radius, n)
xgs = [[x, y, 0, 0] for x in xs for y in ys if x**2 + y**2 <= max_radius**2] # Filter out points that are inside the square but not the cirle 
xs= [x[0] for x in xgs]
ys= [x[1] for x in xgs]


# Load results
with open(f'results/{type_cost}/results_xg_err.json', 'r') as f:
    error_data = json.load(f)

error_array = np.array(error_data)
error_norms = np.linalg.norm(error_array, axis=1)

# # Define the grid for the heatmap
# x_grid = np.linspace(0, 2, 76)  # Adjust the number of points as needed
# y_grid = np.linspace(0, 2, 76)  # Adjust the number of points as needed

# # Create a meshgrid from x and y grids
# X, Y = np.meshgrid(x_grid, y_grid)

# # Interpolate the errors onto the grid
# Z = np.zeros_like(X)
# for i in range(len(xs)):
#     idx_x = np.abs(x_grid - xs[i]).argmin()
#     idx_y = np.abs(y_grid - ys[i]).argmin()
#     Z[idx_y, idx_x] = errors[i]

# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# plt.pcolormesh(X, Y, Z, cmap='viridis')
# plt.colorbar(label='Error')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Error Heatmap')
# plt.show()

# Define the grid for the heatmap



grid_size = 20  # Adjust as needed for resolution
x_grid = np.linspace(-2, 2, grid_size)
y_grid = np.linspace(-2, 2, grid_size)

# Create meshgrid for the coordinates
# X, Y = np.meshgrid(x_grid, y_grid)

## Compute the distance of each point on the grid from the origin
# mask = np.sqrt(X**2 + Y**2) <= 2

# # Create a grid to store the error values
# error_grid = np.full_like(X, np.nan)

# # Interpolate error values onto the grid
# for i in range(len(xs)):
#     # Find the closest point on the grid to (xs[i], ys[i])
#     idx = np.argmin((X - xs[i])**2 + (Y - ys[i])**2)
#     # Check if the point is inside the circle
#     if mask.flat[idx]:
#         # Map the error to the corresponding grid point
#         error_grid.flat[idx] = error_norms[i]
# # Plot the heatmap using Seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(error_grid, cmap='viridis', cbar=True, square=True, xticklabels=False, yticklabels=False, vmin=0.01)
# plt.title('Error Heatmap for Random Points Inside Circle')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()





X, Y = np.meshgrid(x_grid, y_grid)

# Compute the distance of each point on the grid from the origin
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
# Plot the heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(error_grid, cmap='viridis', cbar=True, square=True, xticklabels=False, yticklabels=False, vmin=0.01)
plt.title('Norm of the error for different initial conditions, 2 link, urdf cost, N=10, no hessian')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')






# x_grid = np.linspace(0, 2, 76)  # Adjust the number of points as needed
# y_grid = np.linspace(0, 2, 76)  # Adjust the number of points as needed

# # Create a meshgrid from x and y grids
# X, Y = np.meshgrid(x_grid, y_grid)

# # Interpolate the errors onto the grid
# Z = np.zeros_like(X)
# for i in range(len(xs)):
#     idx_x = np.abs(x_grid - xs[i]).argmin()
#     idx_y = np.abs(y_grid - ys[i]).argmin()
#     Z[idx_y, idx_x] = error_norms[i]

# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# plt.pcolormesh(X, Y, Z, cmap='viridis')
# plt.colorbar(label='Error')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Error Heatmap')
plt.savefig(f'results/{type_cost}/error_xg2')


# # Create a grid for the heatmap
# x_grid, y_grid = np.meshgrid(np.linspace(0, 2, 100), np.linspace(0, 2, 100))

# # Interpolate errors onto the grid
# error_grid = np.zeros_like(x_grid)
# for x, y, error in zip(xs, ys, error_norms):
#     x_idx = int((x / 2) * 99)
#     y_idx = int((y / 2) * 99)
#     error_grid[y_idx, x_idx] = error

# # Plot heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(error_grid, cmap='viridis', xticklabels=False, yticklabels=False)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Error Heatmap')
# plt.savefig(f'results/{type_cost}/error_xg')



# n = 20
# xs = np.linspace(-max_radius, max_radius, n)
# ys = np.linspace(-max_radius, max_radius, n)
# x_grid, y_grid = np.meshgrid(xs, ys)
# radius_grid = np.sqrt(x_grid**2 + y_grid**2)
# circle_mask = radius_grid <= max_radius

# # masked_errors = np.ma.masked_array(error_norms, ~circle_mask)
# plt.figure(figsize=(8, 6))
# sns.heatmap(error_norms, cmap='viridis', square=True, 
#             xticklabels=False, yticklabels=False, cbar_kws={'label': 'Error'})
# plt.title('Error Heatmap')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.savefig(f'results/{type_cost}/error_xg')





# time_xg=np.array([result[0] for result in results_xg])
# err_xg=np.array([result[1] for result in results_xg])
# time_xg_list = time_xg.tolist()
# err_xg_list = err_xg.tolist()
# save_results(f'results/{type_cost}/results_xg_err.json', err_xg_list)
# save_results(f'results/{type_cost}/results_xg_time.json', time_xg_list)


#Sweep N
# Ns=[5,10,15,20,30,40,50] 


# print("Plotting")
# # Mask for xg plot
# x_grid, y_grid = np.meshgrid(xs, ys)
# radius_grid = np.sqrt(x_grid**2 + y_grid**2)
# theta_grid = np.arctan2(y_grid, x_grid)
# circle_mask = radius_grid <= max_radius
# time_N=np.array([result[0] for result in results_N])
# err_N=np.array([result[1] for result in results_N])

# err_N=np.array([result[1] for result in results_N])
# err_x_N=err_N[:,0]
# err_y_N=err_N[:,1]
# err_vx_N=err_N[:,2]
# err_vy_N=err_N[:,3]
# err_norm_N=np.array([np.linalg.norm(err) for err in err_N])


# time_xg=np.array([result[0] for result in results_xg]).reshape(n,n) * circle_mask
# err_xg=np.array([result[1] for result in results_xg])
# err_x_xg=err_xg[:,0].reshape(n,n) * circle_mask
# err_y_xg=err_xg[:,1].reshape(n,n) * circle_mask
# err_vx_xg=err_xg[:,2].reshape(n,n)* circle_mask
# err_vy_xg=err_xg[:,3].reshape(n,n)* circle_mask
# err_norm_xg=np.array([np.linalg.norm(err) for err in err_xg]).reshape(n,n) * circle_mask


# real_times_xg=results_xg[:,0].reshape(n,n) * circle_mask
# errors_xg = results_xg[:,2].reshape(n,n) * circle_mask






