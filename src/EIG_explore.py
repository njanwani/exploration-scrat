"""
This file runs the EIG explore algorithm. The general layout and explanation
of the algorithm can be found in our report.
"""

from exploration import *
SAVE_DIR = '/Users/neiljanwani/Documents/exploration-scrat/images/mapbasic_eig'

grid_array = pickle.load(open(f'{MAP_DIR}/map20.pkl', 'rb'))
explore_array = np.ones(grid_array.shape) * 0.5
grid = ExplorationGrid(grid_array, explore_array, P_TRUE)

fig, axs = plt.subplots(nrows=1, ncols=2)

start = Pose(1, 0.5, np.pi / 2)
pose = start

grid.explore_grid.pose_grid[start.x, start.y].freespace = 0
iters = 1000

filenames = []
i = 0
path = []

while True:
    positions = grid.explore_grid.get_freespace_nodes()
    positions = sorted(positions, key=lambda v: corrupted_EIG(grid, *v), reverse=True)

    while path == []:
        goal = Pose(*positions[0])
        path = waypoint_gen(grid.explore_grid, pose, goal)
        if path != None:
            break
        else:
            path = []
        positions = positions[1:]
    pose = path[0]
    path = path[1:]
    if i % 1 == 0:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        grid.plot(axs, start=pose, goal=goal, path=path)
        fig.set_size_inches((10,5))
        fig.tight_layout()
        filenames.append(rf'{SAVE_DIR}/image{i}.png')
        plt.savefig(filenames[-1])
        plt.close()
    if unknown_area(grid.explore_grid, pose.x, pose.y, pose.theta):
        corrupted_measurement(grid, pose.x, pose.y, pose.theta)
        path = []
    print(np.mean([int(grid.explore_grid.pose_grid[(x,y)].freespace < FREESPACE_ALPHA or grid.explore_grid.pose_grid[(x,y)].freespace > 1 - FREESPACE_ALPHA) for x,y in grid.pose_grid]), end='\r')
    i += 1