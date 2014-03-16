# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:36:49 2014

@author: Paulo
"""

import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

def shift(output_coords):
    return (output_coords[0], output_coords[1] - 512, output_coords[2])

def sind(angle):
    return np.sin(np.deg2rad(angle))

def cosd(angle):
    return np.cos(np.deg2rad(angle))

def arcsind(value):
    return np.rad2deg(np.arcsin(value))

def arctand2(x1, x2):
    return np.rad2deg(np.arctan2(x1, x2))

def original_lat(lat, lon, center_lat, center_lon):
    return arcsind(sind(center_lat)*cosd(lat)*cosd(lon) + cosd(center_lat)*sind(lat))

def original_lon(lat, lon, center_lat, center_lon):
    return center_lon + arctand2(cosd(lat)*sind(lon), cosd(center_lat)*cosd(lat)*cosd(lon) - sind(center_lat)*sind(lat))

def rotated_lat(lat, lon, center_lat, center_lon):
    return arcsind(sind(-center_lat)*cosd(lat)*cosd(lon - center_lon) + cosd(-center_lat)*sind(lat))

def rotated_lon(lat, lon, center_lat, center_lon):
    return arctand2(cosd(lat)*sind(lon - center_lon), cosd(-center_lat)*cosd(lat)*cosd(lon - center_lon) - sind(-center_lat)*sind(lat))

def rotate(output_coords, image_size, center_lat, center_lon):
    lat = -180*(output_coords[0]/float(image_size[0]) - 0.5)
    lon = 360*(output_coords[1]/float(image_size[1]) - 0.5)
    lat2 = original_lat(lat, lon, center_lat, center_lon)
    lon2 = original_lon(lat, lon, center_lat, center_lon)
    return (-image_size[0]*(lat2/180 - 0.5), image_size[1]*(lon2/360 + 0.5), output_coords[2])

def unrotate(output_coords, image_size, center_lat, center_lon):
    lat = -180*(output_coords[0]/float(image_size[0]) - 0.5)
    lon = 360*(output_coords[1]/float(image_size[1]) - 0.5)
    lat2 = rotated_lat(lat, lon, center_lat, center_lon)
    lon2 = rotated_lon(lat, lon, center_lat, center_lon)
    return (-image_size[0]*(lat2/180 - 0.5), image_size[1]*(lon2/360 + 0.5), output_coords[2])

earth = misc.imread("land_shallow_topo_350.jpg")
shifted = ndimage.interpolation.geometric_transform(earth, rotate, mode = 'wrap', extra_arguments = (earth.shape, 45, 45))
original = ndimage.interpolation.geometric_transform(shifted, unrotate, mode = 'wrap', extra_arguments = (shifted.shape, 45, 45))

#c1 = np.arange(0,32,1, dtype = 'double')
#c2 = np.arange(0,64,1, dtype = 'double')
#
#lats = -180*(c1/c1.size - 0.5)
#lons = 360*(c2/c2.size - 0.5)
#
#print lats
#
#glon, glat = np.meshgrid(lons, lats)
#rlat = original_lat(glat,glon,45,0)
#
#
#
#plt.imshow(-c1.size*(rlat/180 - 0.5),extent=[c2[0],c2[-1],c1[0],c1[-1]])
#plt.colorbar()
plt.figure(1)
plt.imshow(shifted)
plt.figure(2)
plt.imshow(original)
plt.show()