#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:42:09 2020

@author: egor
"""

import numpy as np
import time

from read_starcd import Mesh
from read_starcd_old import Mesh as Mesh_old

path = './mesh-cyl/'

mesh = Mesh()
mesh_old = Mesh_old()

t1 = time.clock()
mesh.read_starcd(path)
t2 = time.clock()

print 'Стало', str(t2 - t1)

t1 = time.clock()
mesh_old.read_starcd(path)
t2 = time.clock()

print 'Было', str(t2 - t1)

