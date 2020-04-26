import colorsys
import time
import numpy as np
from ransac import *
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import pdal
import numpy as np
import pandas as pd
import numpy.linalg as la
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import random
from ransac import *

start = time.time()

json = """
[
     "/home/arnesh/3D Ransac/pc0-11.xyz",
    
    {
    
        "type":"filters.normal",
        "knn":8
        
    }
    
]
"""


pipeline = pdal.Pipeline(json)
pipeline.validate() # check if our JSON and options were good
pipeline.loglevel = 8 # really noisy
count = pipeline.execute()
arrays = pipeline.arrays
metadata = pipeline.metadata
log = pipeline.log


data = []
for i in range(len(arrays)):
    for j in range(len(arrays[i])):
        data.append(arrays[i][j])
        
data = np.asarray(data)
data = pd.DataFrame(data)

data_xyz = np.asarray(data[['X','Y','Z']])

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    def plot_plane(a, b, c, d):
        xx, yy = np.meshgrid(np.arange(min(data_xyz[:,0]), max(data_xyz[:,0]), 0.1), np.arange(min(data_xyz[:,1]), max(data_xyz[:,1]), 0.1))
        return xx, yy, (-d - a * xx - b * yy) / c

    n = len(data_xyz)
    max_iterations = 100
    goal_inliers = n * 0.7

    # test data
    xyzs = data_xyz
    

    ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])

    # RANSAC
    planes = []
    while len(data_xyz)> 10:
    	plane = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.1), 3, goal_inliers, max_iterations, stop_at_goal = True)
    	planes.append(plane)

    	points_index = []

    	# To avoid errors since length of list will become smaller on removing points
    	for i in range(len(data_xyz)):	
            if is_inlier(m, data_xyz[j]):
            	points_index.append(j)
        
        for i in range(len(points_index)):
        	data_xyz.pop(j)

    a, b, c, d = planes[0]['m']
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz)
   #x.plot_surface(xx, yy, zz+1, color=(0, 1, 0, 0.5))
   #ax.plot_surface(xx, yy, zz-1, color=(0, 1, 0, 0.5))

    plt.show()
        