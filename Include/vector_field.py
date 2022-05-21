# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:58:03 2022

@author: lstar
"""
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class vector_field:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w
    
    "Sampling function, returns vector of vector field at u(x_n, y_n, z_n)"
    def Sample(self, x_n, y_n, z_n):
        
        expr_u = self.u
        expr_v = self.v
        expr_w = self.w
        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        
        u_x = expr_u.subs({x:x_n, y:y_n, z:z_n})               
        u_y = expr_v.subs({x:x_n, y:y_n, z:z_n})
        u_z = expr_w.subs({x:x_n, y:y_n, z:z_n})
        
        v = np.array([u_x, u_y, u_z])
        return(v)
     
    "Plots the vector field in range gridSize on a resolution*resolution*resolution grid"
    def Plot(self,gridSize,resolution):
        #calculate the size of the setup
        #x_min, y_min, z_min, x_max, y_max, z_max
        x = np.linspace(gridSize.x_min,gridSize.x_max,resolution)
        y = np.linspace(gridSize.y_min,gridSize.y_max,resolution)
        z = np.linspace(gridSize.z_min,gridSize.z_max,resolution)
        
        X, Y, Z = np.meshgrid(x,y,z)
        
        u_x_vec = np.zeros([resolution,resolution,resolution])
        u_y_vec = np.zeros([resolution,resolution,resolution])
        u_z_vec = np.zeros([resolution,resolution,resolution])
        
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    v = self.Sample(x[i], y[j], z[k])
                    
                    u_x_vec[i][j][k] = v[0]
                    u_y_vec[i][j][k] = v[1]
                    u_z_vec[i][j][k] = v[2]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, u_x_vec, u_y_vec, u_z_vec)
        ax.set_xlim([gridSize.x_min,gridSize.x_max])
        ax.set_ylim([gridSize.y_min,gridSize.y_max])
        ax.set_zlim([gridSize.z_min,gridSize.z_max])
        plt.show()
        
        return()