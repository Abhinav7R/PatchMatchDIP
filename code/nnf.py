"""
This is the implementation of the NNF class in the PatchMatch algorithm using python.
Regerence is taken from the original C++ implementation of pm_minimal.cpp
This class is used to compute the nearest neighbor field (NNF)
Input: 2 images
Output: NNF (nearest neighbor field) 
1. a 2D array of size (h, w, 2) where h and w are the height and width of the image
- NNF[i, j] = (y, x) where (y, x) is the coordinate of the nearest neighbor of pixel (i, j) in the other image
2. a 2D array of size (h, w) where h and w are the height and width of the image
- NNF[i, j] = d where d is the distance (rgb squared distance) between pixel (i, j) and its nearest neighbor in the other image

NOTE: 
All coordinates are in (y, x) - height, width format
We assume the size of the target is at least the size of the source image
We also assume that the patch size is odd
"""

from time import time
import numpy as np
import cv2
import random

INF = np.inf

class NNF:
    def __init__(self, a, b, mask_a=None, mask_b=None, patch_w=7, pm_iters=5, rs_max=INF, nnf_init="Random"):
        """
        Initialize the NNF class
        Parameters:
        - a: the first image (numpy array, shape: h x w x 3)
        - b: the second image (numpy array, shape: h x w x 3)
        - patch_w: width of the patch (default: 7)
        - pm_iters: number of PatchMatch iterations (default: 5)
        - rs_max: maximum search radius for random search (default: INF)
        - nnf_init: initialization method for NNF (default: "Random")
                    for mask and other NNF initialization, call the respective intialization functions
        """
        self.patch_w = patch_w
        self.pm_iters = pm_iters
        self.rs_max = rs_max
        self.nnf_init = nnf_init

        self.MAX_RGB_DIFF = 255 * 255 * 3
        self.MAX_PATCH_DIFF = self.MAX_RGB_DIFF * patch_w * patch_w
        
        # Convert to int32 for precise calculations
        self.a = a.astype(np.int32)
        self.b = b.astype(np.int32)
        self.mask_a = mask_a
        self.mask_b = mask_b

        self.ah, self.aw = self.a.shape[0], self.a.shape[1]
        self.bh, self.bw = self.b.shape[0], self.b.shape[1]
        
        # Initialize NNF and distances with zeros
        self.nnf = np.zeros((self.a.shape[0], self.a.shape[1], 2), dtype=np.int32)
        self.nnf_dist = np.zeros((self.a.shape[0], self.a.shape[1]), dtype=np.int32)
        
        if self.nnf_init == "Random":
            self.initialize_nnf()

    def _in_border(self, ax, ay):
        """
        Check if the pixel is in the border
        """
        return ax < self.patch_w // 2 or ax >= self.aw - self.patch_w // 2 or ay < self.patch_w // 2 or ay >= self.ah - self.patch_w // 2
        
    def initialize_nnf(self):
        """
        Initialize the NNF with random coordinates and calculate initial distances
        """
        for ay in range(self.ah):
            for ax in range(self.aw):
                # patch_w/2 from all sides to avoid out of bounds
                if self._in_border(ax, ay):
                    self.nnf[ay, ax] = (ay, ax)
                    self.nnf_dist[ay, ax] = 0
                else:
                    bx = random.randint(0, self.bw - 1)
                    by = random.randint(0, self.bh - 1)
                    self.nnf[ay, ax] = (by, bx)
                    self.nnf_dist[ay, ax] = self.patch_distance(ax, ay, bx, by)
                
    def initialize_nnf_with_other_nnf(self, other_nnf):
        """
        Initialize the NNF with another NNF and calculate initial distances
        """
        # upsample the other NNF by 2 using cv2.resize
        other_nnf = other_nnf.astype(np.float32) * 2
        other_nnf_upsampled = cv2.resize(other_nnf, (self.aw, self.ah))
        other_nnf_upsampled = other_nnf_upsampled.astype(np.int32)
        self.nnf = other_nnf_upsampled.copy()
        for ay in range(self.ah):
            for ax in range(self.aw):
                if self._in_border(ax, ay):
                    self.nnf[ay, ax] = (ay, ax)
                    self.nnf_dist[ay, ax] = 0
                else:
                    by, bx = other_nnf_upsampled[ay, ax]
                    self.nnf_dist[ay, ax] = self.patch_distance(ax, ay, bx, by)
    
    def initialize_nnf_with_mask(self, mask):
        """
        Initialize the NNF with a mask. 0 is for the pixels that need to be inpainted (mask), 1 is for the pixels that are known
        """
        for ay in range(self.ah):
            for ax in range(self.aw):
                if mask[ay, ax] == 0 and not self._in_border(ax, ay):
                    bx = random.randint(0, self.bw - 1)
                    by = random.randint(0, self.bh - 1)
                    self.nnf[ay, ax] = (by, bx)
                    self.nnf_dist[ay, ax] = self.patch_distance(ax, ay, bx, by)
                else:
                    self.nnf[ay, ax] = (ay, ax)
                    self.nnf_dist[ay, ax] = 0
    

    def patch_distance(self, ax, ay, bx, by):
        """
        Measure distance between 2 patches with center at (ay, ax) and (by, bx)
        """
        if self._in_border(ax, ay) or self._in_border(bx, by):
            return 0
        patch_a = self.a[ay - self.patch_w // 2:ay + self.patch_w // 2 + 1, ax - self.patch_w // 2:ax + self.patch_w // 2 + 1]
        patch_b = self.b[by - self.patch_w // 2:by + self.patch_w // 2 + 1, bx - self.patch_w // 2:bx + self.patch_w // 2 + 1]
        
        ssd = np.sum((patch_a - patch_b) ** 2, axis=2)
        if self.mask_a is not None:
            mask_patch_a = self.mask_a[ay - self.patch_w // 2:ay + self.patch_w // 2 + 1, ax - self.patch_w // 2:ax + self.patch_w // 2 + 1]
            ssd = np.where(mask_patch_a == 0, self.MAX_RGB_DIFF, ssd)

        if self.mask_b is not None:
            mask_patch_b = self.mask_b[by - self.patch_w // 2:by + self.patch_w // 2 + 1, bx - self.patch_w // 2:bx + self.patch_w // 2 + 1]
            ssd = np.where(mask_patch_b == 0, self.MAX_RGB_DIFF, ssd)

        return np.sum(ssd)
    
    def improve_guess(self, ax, ay, d_best, bx_new, by_new):
        """
        Improve the current guess if a better match is found.
        """
        d = self.patch_distance(ax, ay, bx_new, by_new)
        if d < d_best:
            self.nnf[ay, ax] = (by_new, bx_new)
            self.nnf_dist[ay, ax] = d
    
    def propagate(self, iter_num, ax, ay, x_change, y_change):
        """
        Perform propagation step
        """
        # current best guess
        # x_best, y_best = self.nnf[ay, ax]
        d_best = self.nnf_dist[ay, ax]

        if 0 <= ax - x_change < self.aw:
            y_prop, x_prop = self.nnf[ay, ax - x_change]
            x_prop += x_change
            if 0 <= x_prop < self.bw:
                self.improve_guess(ax, ay, d_best, x_prop, y_prop)

        if 0 <= ay - y_change < self.ah:
            y_prop, x_prop = self.nnf[ay - y_change, ax]
            y_prop += y_change
            if 0 <= y_prop < self.bh:
                self.improve_guess(ax, ay, d_best, x_prop, y_prop)            


    def random_search(self, ax, ay):
        """
        Perform random search step
        """
        rs_start = min(self.rs_max, max(self.bw, self.bh))

        mag = rs_start

        while mag >= 1:
            # strt = time()
            y_best, x_best = self.nnf[ay, ax]
            d_best = self.nnf_dist[ay, ax]
            
            x_min = max(x_best - mag, 0)
            x_max = min(x_best + mag + 1, self.bw)
            y_min = max(y_best - mag, 0)
            y_max = min(y_best + mag + 1, self.bh)

            if x_min == x_max or y_min == y_max:
                mag = mag // 2
                continue
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
            y_end = self.ah
            y_change = 1
            x_start = 0
            x_end = self.aw
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
            # print("Iteration %d done in %f seconds" % (iter_num, t_end - t_start))
        return self.nnf, self.nnf_dist


