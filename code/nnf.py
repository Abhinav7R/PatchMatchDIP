"""
This is the implementation of the NNF class in the PatchMatch algorithm using python.
Regerence is taken from the original C++ implementation of pm_minimal.cpp
This class is used to compute the nearest neighbor field (NNF)
Input: 2 images
Output: NNF (nearest neighbor field) 
1. a 2D array of size (h, w, 2) where h and w are the height and width of the image
- NNF[i, j] = (x, y) where (x, y) is the coordinate of the nearest neighbor of pixel (i, j) in the other image
2. a 2D array of size (h, w) where h and w are the height and width of the image
- NNF[i, j] = d where d is the distance (rgb squared distance) between pixel (i, j) and its nearest neighbor in the other image
"""

from time import time
import numpy as np
import random

INF = np.inf

class NNF:
    def __init__(self, a, b, patch_w=7, pm_iters=5, rs_max=INF):
        """
        Initialize the NNF class
        Parameters:
        - a: the first image (numpy array, shape: h x w x 3)
        - b: the second image (numpy array, shape: h x w x 3)
        - patch_w: width of the patch (default: 7)
        - pm_iters: number of PatchMatch iterations (default: 5)
        - rs_max: maximum search radius for random search (default: INF)
        """
        self.patch_w = patch_w
        self.pm_iters = pm_iters
        self.rs_max = rs_max
        
        # Convert to int32 for precise calculations
        self.a = a.astype(np.int32)
        self.b = b.astype(np.int32)

        # Get height and width of the images
        self.ah, self.aw = a.shape[0], a.shape[1]
        self.bh, self.bw = b.shape[0], b.shape[1]

        # effective width and height (possible upper left corners of patches)
        self.aeh = self.ah - self.patch_w + 1
        self.aew = self.aw - self.patch_w + 1
        self.beh = self.bh - self.patch_w + 1
        self.bew = self.bw - self.patch_w + 1  
        
        # Initialize NNF and distances with zeros
        self.nnf = np.zeros((a.shape[0], a.shape[1], 2), dtype=np.int32)
        self.nnf_dist = np.zeros((a.shape[0], a.shape[1]), dtype=np.int32)
        
        self.initialize_nnf()

    def initialize_nnf(self):
        """
        Initialize the NNF with random coordinates and calculate initial distances
        """
        for ay in range(self.aeh):
            for ax in range(self.aew):
                bx = random.randint(0, self.bew - 1)
                by = random.randint(0, self.beh - 1)
                self.nnf[ay, ax] = (bx, by)
                self.nnf_dist[ay, ax] = self.patch_distance(ax, ay, bx, by)

    def patch_distance(self, ax, ay, bx, by):
        """
        Measure distance between 2 patches with upper left corners (ax, ay) and (bx, by)
        """
        patch_a = self.a[ay:ay + self.patch_w, ax:ax + self.patch_w]
        patch_b = self.b[by:by + self.patch_w, bx:bx + self.patch_w]
        ans = np.sum((patch_a - patch_b) ** 2)

        return ans
    
    def improve_guess(self, ax, ay, d_best, bx_new, by_new):
        """
        Improve the current guess if a better match is found.
        """
        d = self.patch_distance(ax, ay, bx_new, by_new)
        if d < d_best:
            self.nnf[ay, ax] = (bx_new, by_new)
            self.nnf_dist[ay, ax] = d
    
    def propagate(self, iter_num, ax, ay, x_change, y_change):
        """
        Perform propagation step
        """
        # current best guess
        # x_best, y_best = self.nnf[ay, ax]
        d_best = self.nnf_dist[ay, ax]

        if 0 <= ax - x_change < self.aew:
            x_prop, y_prop = self.nnf[ay, ax - x_change]
            x_prop += x_change
            if 0 <= x_prop < self.bew:
                self.improve_guess(ax, ay, d_best, x_prop, y_prop)

        if 0 <= ay - y_change < self.aeh:
            x_prop, y_prop = self.nnf[ay - y_change, ax]
            y_prop += y_change
            if 0 <= y_prop < self.beh:
                self.improve_guess(ax, ay, d_best, x_prop, y_prop)            


    def random_search(self, ax, ay):
        """
        Perform random search step
        """
        rs_start = min(self.rs_max, max(self.bw, self.bh))

        mag = rs_start

        while mag >= 1:
            # strt = time()
            x_best, y_best = self.nnf[ay, ax]
            d_best = self.nnf_dist[ay, ax]
            
            x_min = max(x_best - mag, 0)
            x_max = min(x_best + mag + 1, self.bew)
            y_min = max(y_best - mag, 0)
            y_max = min(y_best + mag + 1, self.beh)

            x_rand = random.randint(x_min, x_max - 1)
            y_rand = random.randint(y_min, y_max - 1)

            self.improve_guess(ax, ay, d_best, x_rand, y_rand)

            mag = mag // 2
            # print(f"Random search at mag {mag} done in {time() - strt} seconds")

    def compute_nnf(self):
        """
        Compute the nearest neighbor field using PatchMatch
        """
        for iter_num in range(self.pm_iters):
            t_start = time()
            y_start = 0
            y_end = self.aeh
            y_change = 1
            x_start = 0
            x_end = self.aew
            x_change = 1

            if iter_num % 2 == 1:
                y_start = y_end - 1
                y_end = -1
                y_change = -1
                x_start = x_end - 1
                x_end = -1
                x_change = -1

            for ay in range(y_start, y_end, y_change):
                for ax in range(x_start, x_end, x_change):
                    self.propagate(iter_num, ax, ay, x_change, y_change)
                    self.random_search(ax, ay)

            t_end = time()
            print("Iteration %d done in %f seconds" % (iter_num, t_end - t_start))
        return self.nnf, self.nnf_dist


