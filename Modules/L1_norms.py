from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

class DartSimulation():
    """A simulation where we generate many random points and group the points
    based on where they fall relative to the different norms.  With many trials
    we can calculate relative sizes of different norm balls in high dimensional
    space.

    Parameters
    ----------
    n : int
        Number of times to 'throw' the dart
    dim : int
        Number of dimensions [default: 2]
    magnitude : int
        Max magnitude from origin [default: 1]


    Attributes
    ----------
    pts : ndarray, float
        random n x dim matrix centered about origin +/- magnitude/2

    L1_norm : ndarray, float
        L1 norm of pts

    L2_norm : ndarray, float
        L2 norm of pts

    Linf_norm : ndarray, float
        L infinity norm of points

    pt_colors : array, string
        colors for pt
    """
    def __init__(self, n, dim=2, magnitude=1):
        self.pts = self.throw_darts(n, dim, magnitude)

    def throw_darts(self, n, dim, magnitude):
        return (2*magnitude) * np.random.random([n, dim]) - magnitude

    @property
    def L1_norm(self):
        """Sum of |xi| for i from 1 to n."""
        self._L1_norm = abs(self.pts).sum(axis=1)
        return self._L1_norm

    @property
    def L2_norm(self):
        """(x1^2 + x2^2 + ...xi^2)^(1/2) for i from 1 to n."""
        self._L2_norm = (self.pts * self.pts).sum(axis=1)**(1/2)
        return self._L2_norm

    @property
    def Linf_norm(self):
        """Max{|x1|, |x2|, ..., |xi|} for i from 1 to n."""
        self._Linf_norm = abs(self.pts).max(axis=1)
        return self._Linf_norm

    @property
    def pt_colors(self):
        # Check calculated norms against boundary of norm balls L1, L2
        lt_L1_mask = (self.L1_norm < 1)
        lt_L2_mask = (self.L2_norm < 1)
        norm_mask = [(l1, l2) for l1, l2 in zip(lt_L1_mask, lt_L2_mask)]

        # Map different colors for inside different norm balls
        color_map = {(False, False): 'b',
                     (False, True): 'r',
                     (True, True): 'g'}
        self._pt_colors = np.array([color_map[pt] for pt in norm_mask])
        return self._pt_colors

    def plot_pts_2d(self):
        """Plot the color coded points in 2D"""
        plt.figure(figsize=(10,10))
        plt.scatter(self.pts[:, 0], self.pts[:, 1], c=self.pt_colors)

    def plot_pts_3d(self):
        """Plot the color coded points in 3D"""
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pts[:, 0],
                   self.pts[:, 1],
                   self.pts[:, 2],
                   c=self.pt_colors)
