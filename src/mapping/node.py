from mapping.constants import *

class OccupancySpace:
    """
    General Occupancy Space that stores x,y,freespace information for some cell
    in the grid.
    """
    
    def __init__(self, x, y, freespace):
        self.x = x
        self.y = y
        self.freespace = freespace
        
    def __str__(self):
        return f'({self.x}, {self.y})'

class Pose(OccupancySpace):
    """
    Pose Space that stores x,y,theta,freespace information for some cell
    in the grid. Freespace is assumed to be 0 here as a Pose is a robot position
    """
    THETA_LIST = [i * np.pi / 2 for i in range(0, 4)]
    
    def __init__(self, x, y, theta):
        super().__init__(x, y, freespace=1e-6)
        self.theta = theta
        self.cost = np.inf
        self.parent = None
        
    def compute_cost(self, other):
        """
        Euclidean cost metric
        """
        dtheta = (self.theta - other.theta)
        if abs(dtheta) > np.pi:
            dtheta = abs(dtheta) - np.pi
        return np.linalg.norm([self.x - other.x, self.y - other.y, 1*dtheta])
    
    def get_neighbors(self, grid, cell_width=0.5):
        """
        Gets neighbors in freespace for the (self) current pose
        """
        ret = []
        
        dx, dy = cell_width * round(np.cos(self.theta)), cell_width * round(np.sin(self.theta)) # compute possible dx, dy based on theta (can only drive straight)
        # NEED TO FIX THIS
        if  self.x + dx >= 0 and self.y + dy >= 0 \
            and self.x + dx < grid.xoffset + grid.xlim \
            and self.y + dy < grid.yoffset + grid.ylim \
            and grid.pose_grid[(self.x + dx, self.y + dy)].freespace <= FREESPACE_ALPHA:
            # print((self.x + dx, self.y + dy), grid.pose_grid[(self.x + dx, self.y + dy)].freespace)
            ret.append((self.x + dx, self.y + dy, self.theta))
            
        for t in Pose.THETA_LIST: # add in turns
            if self.theta != t:
                ret.append((self.x, self.y, t))

        return ret
                       
    def __lt__(self, other):
        return self.cost < other.cost # for priority queue
    
    def __str__(self):
        return f'({self.x}, {self.y}, {self.theta})'
    
    def round(self, cell_width=0.5):
        """Truncates, not round..."""
        self.x = int(self.x / cell_width) * cell_width
        self.y = int(self.y / cell_width) * cell_width
        self.theta = round(self.theta * 4 / np.pi) / 4 * np.pi