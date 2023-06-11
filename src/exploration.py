from mapping.map import *

def corrupted_measurement(grid, x, y, theta):
    """
    Computes one corrupted measurement and updates the exploration grid
    """
    dx, dy = grid.cell_width * round(np.cos(theta)), grid.cell_width * round(np.sin(theta))
    dx2, dy2 = grid.cell_width * round(np.cos(theta + np.pi / 2)), grid.cell_width * round(np.sin(theta + np.pi / 2))
    measured = [(x + dx2, y + dy2), (x - dx2, y - dy2)] + [(x + dx, y + dy)] + [(x + dx + dx2, y + dy + dy2), (x - dx2 + dx, y - dy2 + dy)]
    for x,y in measured:
        if x >= 0 and y >= 0 and x < grid.xlim + grid.xoffset and y < grid.ylim + grid.yoffset:
            o = grid.pose_grid[(x,y)].freespace
            o = 1 - o if np.random.random() > P_TRUE else o
            grid.measurement_update(o, x, y)


def corrupted_EIG(grid, x, y, theta):
    """
    Computes the EIG given the grid with corrupted measurements for a given
    x,y position
    """
    dx, dy = grid.cell_width * np.round(np.cos(theta)), grid.cell_width * np.round(np.sin(theta))
    dx2, dy2 = grid.cell_width * np.round(np.cos(theta + np.pi / 2)), grid.cell_width * np.round(np.sin(theta + np.pi / 2))
    measured = [(x + dx2, y + dy2), (x - dx2, y - dy2)] + [(x + dx, y + dy)] + [(x + dx + dx2, y + dy + dy2), (x - dx2 + dx, y - dy2 + dy)]
    totalEIG = 0
    i = 0
    for x,y in measured:
        if x >= 0 and y >= 0 and x < grid.xlim + grid.xoffset and y < grid.ylim + grid.yoffset:
            totalEIG += grid.EIG(x,y)
            i += 1
    return totalEIG / i


def unknown_area(grid, x,y,theta):
    """
    Determines whether the area being sensed by the robot is unknown or not
    """
    dx, dy = grid.cell_width * np.round(np.cos(theta)), grid.cell_width * np.round(np.sin(theta))
    dx2, dy2 = grid.cell_width * np.round(np.cos(theta + np.pi / 2)), grid.cell_width * np.round(np.sin(theta + np.pi / 2))
    measured = [(x + dx2, y + dy2), (x - dx2, y - dy2)] + [(x + dx, y + dy)] + [(x + dx + dx2, y + dy + dy2), (x - dx2 + dx, y - dy2 + dy)]
    for x,y in measured:
        if x >= 0 and y >= 0 and x < grid.xlim + grid.xoffset and y < grid.ylim + grid.yoffset:
            if not (grid.pose_grid[x,y].freespace < FREESPACE_ALPHA or grid.pose_grid[x,y].freespace > 1 - FREESPACE_ALPHA):
                return True
    return False


def waypoint_gen(grid, start, goal):
    """
    Function requested in prelab. Uses Dijkstra's to generate path
    """
    start.round()
    goal.round()
    point_to_pose = {} # assemble a dictionary relating (x,y,theta) to Pose object (need this to store cost/parent)
    for x,y in grid.pose_grid:
        if grid.pose_grid[x,y].freespace <= FREESPACE_ALPHA:
            for theta in Pose.THETA_LIST:
                point_to_pose[(grid.pose_grid[x,y].x, grid.pose_grid[x,y].y, theta)] = Pose(grid.pose_grid[x,y].x, grid.pose_grid[x,y].y, theta)
                
    point_to_pose[(start.x, start.y, start.theta)] = start
    point_to_pose[(goal.x, goal.y, goal.theta)] = goal
    start.cost = start.compute_cost(start)

    onDeck = queue.PriorityQueue()
    onDeck.put(start)
    iters = 0
    while True:
        if onDeck.empty():
            return None
        
        iters += 1
        node = onDeck.get()
        if node == goal:
            break
        for neighbor_point in node.get_neighbors(grid):
            neighbor = point_to_pose[neighbor_point]
            new_cost = neighbor.compute_cost(start) + node.cost
            if new_cost < neighbor.cost: # only search if neighbor is 'more optimal'
                neighbor.cost = new_cost
                neighbor.parent = node
                onDeck.put(neighbor)

    node = goal
    path = [goal]
    while node != start: # construct path
        path.append(node.parent)
        node = node.parent

    path.reverse()
    return path