from mapping.constants import *
from mapping.node import *

class OccupancyGrid:
    """
    Datastructure to hold all the important information needed for an occupancy
    grid (offsets, cell width, etc...)
    """
    
    def __init__(self, grid_array, xoffset=0, yoffset=0, cell_width=0.5):
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.cell_width = cell_width
        self.make_map(grid_array)
    
    def make_map(self, grid_array):
        """
        Builds an OccupancySpace map from a given numpy array of freespace
        values. Numpy array is discarded to avoid data duplication.
        """
        self.pose_grid = {}
        for x in range(grid_array.shape[0]):
            for y in range(grid_array.shape[1]):
                self.pose_grid[(x * self.cell_width + self.xoffset, y * self.cell_width + self.yoffset)] = OccupancySpace(x * self.cell_width + self.xoffset, y * self.cell_width + self.yoffset, freespace=grid_array[x,y])
        
        self.xlim = grid_array.shape[0] * self.cell_width
        self.ylim = grid_array.shape[1] * self.cell_width
    
    def get_freespace(self, x, y):
        return self.pose_grid[(x,y)].freespace        
    
    def to_idx(self, position):
        x, y = position
        return int(x / self.cell_width - self.xoffset), int(y / self.cell_width - self.yoffset)
    
    def to_array(self):
        """
        Converts the Occupancy grid back to an array when needed (only used for
        plotting)
        """
        new_grid = np.zeros((int(self.xlim / self.cell_width), int(self.ylim / self.cell_width)))
        for x,y in self.pose_grid:
            i,j = self.to_idx((x,y))
            new_grid[i,j] = self.pose_grid[(x,y)].freespace
            
        return new_grid
    
    def get_freespace_nodes(self):
        """
        Gets all the freespace nodes in the entire occupancy grid
        """
        positions = []
        for x,y in self.pose_grid:
            if self.pose_grid[(x,y)].freespace <= FREESPACE_ALPHA:
                for theta in Pose.THETA_LIST:
                    positions.append((x,y,theta))
        
        return positions
        

class ExplorationGrid(OccupancyGrid):
    """
    Datastructure to hold all the important information needed for an
    Exploration grid (offsets, cell width, etc...). This datastructure is
    inherited from the Occupancy grid as it *is* an Occupancy Grid. However, it
    also contains another OccupancyGrid as an explore grid (which stores the 
    probabilities pij for all cells). This is useful as we now have G_T and G as
    defined in the report (ground truth and probability grids).
    """
    
    def __init__(self, grid_array, explore_grid, ptrue, xoffset=0, yoffset=0, cell_width=0.5):
        super().__init__(grid_array, xoffset, yoffset, cell_width)
        self.ptrue = ptrue
        self.explore_grid = OccupancyGrid(explore_grid, xoffset, yoffset, cell_width)
    
    def occupied_update(self, pij):
        return self.ptrue * pij / (self.ptrue * pij + (1 - self.ptrue) * (1 - pij))
    
    def unoccupied_update(self, pij):
        return 1 - self.ptrue * (1 - pij) / (self.ptrue * (1 - pij) + (1 - self.ptrue) * pij)
    
    def measurement_update(self, o, x, y):
        """
        Completes a measurement update given the measurement and position
        """
        pij = self.explore_grid.pose_grid[(x,y)].freespace
        if o > 0.5:
            self.explore_grid.pose_grid[(x,y)].freespace = self.occupied_update(pij)
        else:
            self.explore_grid.pose_grid[(x,y)].freespace = self.unoccupied_update(pij)  
            
    def E(self, x, y):
        pij = self.explore_grid.pose_grid[(x,y)].freespace
        return -P_TRUE * pij * np.log(P_TRUE * pij / (P_TRUE * pij + (1-P_TRUE) * (1-pij)) + PAD) - P_TRUE * (1-pij) * np.log(P_TRUE * (1-pij) / (P_TRUE * (1-pij) + (1-P_TRUE) * pij) + PAD)
            
    def Hp(self, x, y):
        pij = self.explore_grid.pose_grid[(x,y)].freespace
        return -pij * np.log(pij + PAD) - (1-pij) * np.log(1-pij + PAD)
    

    def EIG(self, x, y):
        """Computes EIG (expected information gain)"""
        return self.Hp(x,y) - self.E(x,y)
    
        
    def plot(self, axs, start=None, goal=None, path=None):
        assert len(axs) == 2
        
        visualize(self, axs[0], start=start, goal=goal, path=path)
        visualize(self.explore_grid, axs[1], start=start, goal=goal, path=path)
        axs[0].set_title('Occupancy Grid')
        axs[1].set_title('Exploration Grid')


def visualize(grid, ax, start=None, goal=None, path=None):
    """
    Visualization function for plotting the maps. Plots grid, start, goal and
    path
    """
    grid_array = grid.to_array()
    ax.imshow(1 - grid_array.T, 
                aspect='equal', 
                cmap='gray',
                extent=[0, grid_array.shape[0] * grid.cell_width, 0, grid_array.shape[1] * grid.cell_width], 
                origin='lower',
                vmin=0,
                vmax=1)
            
    if start != None:
        ax.add_patch(Rectangle((start.x, start.y), grid.cell_width, grid.cell_width, linewidth=1, edgecolor='Red', facecolor='Red'))
        ax.add_patch(Arrow(start.x,
                            start.y,
                            0.2 * np.cos(start.theta),
                            0.2 * np.sin(start.theta),
                            width=0.6,
                            edgecolor='Blue',
                            facecolor='Blue'))
    
    if goal != None: ax.add_patch(Rectangle((goal.x, goal.y), grid.cell_width, grid.cell_width, linewidth=1, edgecolor='Green', facecolor='Green'))
    
    if path != None: # draw the path
        for node in path:
            ax.add_patch(Arrow(node.x,
                            node.y,
                            0.2 * np.cos(node.theta),
                            0.2 * np.sin(node.theta),
                            width=0.6,
                            edgecolor='Blue',
                            facecolor='Blue'))
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_xticks(np.arange(grid.xoffset, grid.xlim + grid.xoffset + grid.cell_width, grid.cell_width))
    # ax.set_yticks(np.arange(grid.yoffset, grid.ylim + grid.yoffset + grid.cell_width, grid.cell_width))
    ax.set_aspect('equal', adjustable='box')
        

if __name__ == '__main__':
    grid_array = pickle.load(open(f'{MAP_DIR}/map1000.pkl', 'rb'))
    explore_array = np.ones(grid_array.shape) * 0.5
    grid = ExplorationGrid(grid_array, explore_array, P_TRUE)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    grid.plot(axs)
    fig.set_size_inches((10,5))
    plt.show()