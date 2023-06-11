import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import queue
import cv2
import time
import pickle


P_TRUE = 0.90
FREESPACE_ALPHA = 0.05
PAD = 1e-6
MAP_DIR = '/Users/neiljanwani/Documents/exploration-scrat/maps'