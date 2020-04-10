import numpy as np
import matplotlib.pyplot as plt
import math

class vectors(object):

    def __init__(self, basis_vecs):
        self.coordinates = tuple([i for i in basis_vecs])
    
    @property
    def norm(self):
        self._norm = tuple([np.linalg.norm(x) for x in self.coordinates])
        return self._norm
    
    @property
    def inner(self):
        a,b = self.coordinates
        self._inner = a@b
        return self._inner
    
    def graph(self):
        plt.figure()
        ax = plt.gca()
        vec_a, vec_b = [np.array(i) for i in self.coordinates]
        #Equation for vec_a
        ax.quiver(0,0, vec_a[0], vec_a[1], angles='xy',scale_units='xy',scale=1,color='green')
        ax.text(vec_a[0]*1.1, vec_a[1]*1.1,'Vector 1', color='green')
        
        #Equation for vec_b
        ax.quiver(0,0,vec_b[0],vec_b[1],angles='xy',scale_units='xy',scale=1, color='orchid')
        ax.text(vec_b[0]*1.1, vec_b[1]*1.1,'Vector 2', color='orchid')

        #A-B
        a,b = self.coordinates
        connect_x,connect_y = a-b
        ax.quiver(vec_b[0],vec_b[1], connect_x, connect_y, angles='xy',scale_units='xy',scale=1,color='yellow')
        
        x,y = zip(*self.coordinates)        
        plt.ylim(-1,max(y)+1)
        plt.xlim(-1,max(x)+1)
        plt.grid()
        
        anorm, bnorm = self.norm
        prod_ab = anorm*bnorm
        plt.title(r'$\|\langle a,b \rangle|:${}   $||a||_2||b||_2:${}'.format(self.inner, round(prod_ab,1)))
        plt.show()