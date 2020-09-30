#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:42:09 2020

@author: egor
"""
import sys
sys.path.append('../')
import numpy as np
import time
from datetime import datetime
import os
import tucker.tucker as tuck
import solver.solver_tucker as Boltzmann

# =============================================================================
# from read_starcd import Mesh
# from read_starcd_old import Mesh as Mesh_old
#
# path = './mesh-cyl/'
#
# mesh = Mesh()
# mesh_old = Mesh_old()
#
# t1 = time.clock()
# mesh.read_starcd(path)
# t2 = time.clock()
#
# print 'Стало', str(t2 - t1)
#
# t1 = time.clock()
# mesh_old.read_starcd(path)
# t2 = time.clock()
#
# print 'Было', str(t2 - t1)
# =============================================================================


#resfile = open('test.txt', 'w')
#resfile.close()

#path = './' + 'impl' + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '/'
#os.mkdir(path)
#
#resfile = open(path + '123.txt', 'w')
#resfile.close()

#a = np.ones((4, 3))

#a_tuck = tuck.tensor(a)

#print(a_tuck.u[0].shape)

#np.dot(np.ones(4), a)

nv = 44
vmax = 22 * 6000.

hv = 2. * vmax / nv
vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes

v = Boltzmann.VelocityGrid(vx_, vx_, vx_)

print(v.vx)


























