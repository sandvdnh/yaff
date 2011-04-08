# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import numpy as np

from molmod import angstrom

from yaff import *


__all__ = [
    'get_system_water32', 'get_system_graphene8',
    'get_system_polyethylene4', 'get_system_quartz', 'get_system_glycine',
    'get_system_cyclopropene', 'get_system_caffeine', 'get_system_butanol',
]


def get_system_water32():
    return System(
        numbers=np.array([8, 1, 1]*32),
        pos=np.array([ # Coordinates ripped from the CP2K test suite.
            [-4.583, 5.333, 1.560], [-3.777, 5.331, 0.943], [-5.081, 4.589,
            1.176], [-0.083, 4.218, 0.070], [-0.431, 3.397, 0.609], [0.377,
            3.756, -0.688], [-1.488, 2.692, 4.125], [-2.465, 2.433, 3.916],
            [-1.268, 2.145, 4.952], [-2.461, -2.548, -6.136], [-1.892, -2.241,
            -6.921], [-1.970, -3.321, -5.773], [4.032, 0.161, 2.183], [4.272,
            -0.052, 1.232], [4.044, -0.760, 2.641], [2.950, -3.497, -1.006],
            [2.599, -3.901, -0.129], [3.193, -4.283, -1.533], [-2.890, -4.797,
            -2.735], [-2.810, -5.706, -2.297], [-2.437, -4.128, -2.039],
            [-0.553, 0.922, -3.731], [-0.163, 1.552, -3.085], [-1.376, 0.544,
            -3.469], [4.179, 4.017, 4.278], [3.275, 3.832, 3.876], [4.658,
            4.492, 3.572], [5.739, 1.425, 3.944], [5.125, 2.066, 4.346], [5.173,
            1.181, 3.097], [0.988, -0.760, -5.445], [1.640, -1.372, -4.989],
            [0.546, -0.220, -4.762], [-0.748, 1.985, 1.249], [-0.001, 1.490,
            1.540], [-1.160, 2.255, 2.109], [4.127, -0.234, -3.149], [5.022,
            -0.436, -3.428], [3.540, -0.918, -3.601], [-2.473, 2.768, -1.395],
            [-1.533, 2.719, -1.214], [-2.702, 1.808, -1.479], [-0.124, -2.116,
            2.404], [0.612, -2.593, 2.010], [0.265, -1.498, 3.089], [0.728,
            2.823, -2.190], [0.646, 3.694, -2.685], [1.688, 2.705, -1.947],
            [4.256, -5.427, -2.644], [5.222, -5.046, -2.479], [4.174, -5.628,
            -3.593], [-3.178, -0.508, -4.227], [-2.762, -1.221, -4.818],
            [-3.603, 0.073, -4.956], [-1.449, 5.300, -4.805], [-1.397, 4.470,
            -5.317], [-2.102, 5.091, -4.067], [3.354, 2.192, -1.755], [3.407,
            1.433, -2.405], [3.971, 2.958, -2.196], [1.773, -4.018, 1.769],
            [1.121, -4.532, 1.201], [1.975, -4.529, 2.618], [1.526, 1.384,
            2.712], [2.317, 1.070, 2.251], [1.353, 0.657, 3.364], [2.711,
            -2.398, -4.253], [2.202, -3.257, -4.120], [3.305, -2.610, -5.099],
            [6.933, 0.093, -1.393], [6.160, -0.137, -0.795], [6.748, -0.394,
            -2.229], [-5.605, -2.549, 3.151], [-4.756, -2.503, 3.616], [-5.473,
            -3.187, 2.378], [0.821, -4.406, 6.516], [0.847, -3.675, 7.225],
            [-0.014, -4.240, 5.988], [1.577, 3.933, 3.762], [1.221, 2.975,
            3.640], [1.367, 4.126, 4.659], [-2.111, -3.741, -0.219], [-1.378,
            -4.425, -0.036], [-1.825, -2.775, 0.003], [0.926, -1.961, -2.063],
            [0.149, -1.821, -1.402], [1.725, -2.303, -1.536], [4.531, -1.030,
            -0.547], [4.290, -1.980, -0.581], [4.292, -0.597, -1.390], [-0.740,
            -1.262, -0.029], [-1.272, -0.422, -0.099], [-0.403, -1.349, 0.873],
            [3.655, 3.021, 0.988], [2.706, 3.053, 1.282], [3.542, 2.615, 0.020]
        ])*angstrom,
        ffatypes=['O', 'H', 'H']*32,
        bonds=np.array([[(i/3)*3,i] for i in xrange(96) if i%3!=0]),
        rvecs=np.array([[9.865, 0.0, 0.0], [0.0, 9.865, 0.0], [0.0, 0.0, 9.865]])*angstrom,
    )


def get_system_graphene8():
   return System(
        numbers=np.array([6]*8),
        pos=np.array([
            [2.461, 0.000, 0.000], [4.922, 1.421, 0.000], [3.692, 2.131, 0.000],
            [6.153, 3.552, 0.000], [1.231, 2.131, 0.000], [3.692, 3.552, 0.000],
            [0.000, 0.000, 0.000], [2.461, 1.421, 0.000]
        ])*angstrom,
        ffatypes=['C']*8,
        bonds=np.array([
            [0, 1], [1, 2], [1, 3], [3, 6], [4, 0], [2, 6], [2, 4], [7, 0], [7,
            3], [5, 6], [5, 4], [5, 7]
        ]),
        rvecs=np.array([[4.922, 0.0, 0.0], [2.462, 4.262, 0.0]])*angstrom,
    )


def get_system_polyethylene4():
    return System(
        numbers=np.array([6]*4 + [1]*8),
        pos=np.array([
            [4.4665, -0.2419, 0.0939], [3.1498, 0.5401, 0.0859], [1.9114,
            -0.3549, 0.0216], [0.6320, 0.4677, 0.0288], [4.4927, -0.8901,
            0.9785], [4.4935, -0.9016, -0.7825], [3.1442, 1.2221, -0.7735],
            [3.1012, 1.1657, 0.9858], [1.9051, -1.0408, 0.8766], [1.9429,
            -0.9675, -0.8867], [0.5566, 1.0682, 0.9410], [0.5949, 1.1431,
            -0.8319],
        ])*angstrom,
        ffatypes=['C']*4 + ['H']*8,
        bonds=np.array([
            [3, 2], [10, 3], [11, 3], [1, 0], [2, 1], [4, 0], [5, 0], [6, 1],
            [7, 1], [8, 2], [9, 2], [3, 0],
        ]),
        rvecs=np.array([[5.075, 0.187, 0.055]])*angstrom,
    )


def get_system_quartz():
    return System(
        numbers=np.array([14]*3 + [8]*6),
        pos=np.array([
            [ 1.999357437, -1.154329699, -1.801733563],
            [ 0.000000000,  2.308659399,  1.801733563],
            [-1.999357437, -1.154329699,  0.000000000],
            [ 1.762048976,  0.299963042, -1.159593954],
            [-1.140800226,  1.375997798,  2.443872642],
            [-0.621248751, -1.675960841,  0.642139609],
            [ 0.621248751, -1.675960841, -2.443872642],
            [-1.762048976,  0.299963042, -0.642139609],
            [ 1.140800226,  1.375997798,  1.159593954],
        ])*angstrom,
        ffatypes=['Si']*3 + ['O']*6,
        bonds=np.array([
            [2, 8], [1, 8], [2, 7], [0, 7], [1, 6], [0, 6], [2, 5], [1, 5],
            [1, 4], [0, 4], [2, 3], [0, 3],
        ]),
        rvecs=np.array([[0.0, 0.0, 5.405222], [0.0, 4.913416, 0.0], [-4.255154, 2.456708, 0.0]])*angstrom,
    )


def get_system_glycine():
    return System(
        numbers=np.array([7, 6, 6, 8, 8, 1, 1, 1, 1, 1]),
        pos=np.array([
            [ 1.421031,  0.728490,  0.340852],
            [ 0.372356,  0.085434, -0.431299],
            [-0.863219, -0.325923,  0.373824],
            [-1.853953, -0.929623, -0.341462],
            [-0.982444, -0.142046,  1.563332],
            [ 0.995984,  1.527545,  0.820162],
            [ 1.657981,  0.097447,  1.111926],
            [ 0.044404,  0.759440, -1.244685],
            [ 0.776702, -0.817449, -0.926065],
            [-1.568843, -0.983316, -1.266586],
        ]),
        ffatypes=['N', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'],
        bonds=np.array([[3, 9], [1, 8], [1, 7], [0, 6], [0, 5], [2, 4], [2, 3], [1, 2], [0, 1]]),
        rvecs=np.array([]),
    )


def get_system_cyclopropene():
    # structure taken from pubchem
    return System(
        numbers=np.array([6, 6, 6, 1, 1, 1, 1]),
        pos=np.array([
           [-0.8487, -0.0002,  0.0000],
           [ 0.4242,  0.6507,  0.0000],
           [ 0.4245, -0.6505,  0.0000],
           [-1.4015, -0.0004,  0.9258],
           [-1.4015, -0.0004, -0.9258],
           [ 0.9653,  1.5624,  0.0000],
           [ 0.9661, -1.5620,  0.0000],
        ])*angstrom,
        ffatypes=['C', 'C', 'C', 'H', 'H', 'H', 'H'],
        bonds=[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 5], [2, 6]],
        rvecs=np.array([]),
    )


def get_system_caffeine():
    # structure taken from pubchem
    return System(
        numbers=np.array([8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1]),
        pos=np.array([
           [ 0.4700,  2.5688,  0.0006],
           [-3.1271, -0.4436, -0.0003],
           [-0.9686, -1.3125,  0.0000],
           [ 2.2182,  0.1412, -0.0003],
           [-1.3477,  1.0797, -0.0001],
           [ 1.4119, -1.9372,  0.0002],
           [ 0.8579,  0.2592, -0.0008],
           [ 0.3897, -1.0264, -0.0004],
           [ 0.0307,  1.4220, -0.0006],
           [-1.9061, -0.2495, -0.0004],
           [ 2.5032, -1.1998,  0.0003],
           [-1.4276, -2.6960,  0.0008],
           [ 3.1926,  1.2061,  0.0003],
           [-2.2969,  2.1881,  0.0007],
           [ 3.5163, -1.5787,  0.0008],
           [-1.0451, -3.1973, -0.8937],
           [-2.5186, -2.7596,  0.0011],
           [-1.0447, -3.1963,  0.8957],
           [ 4.1992,  0.7801,  0.0002],
           [ 3.0468,  1.8092, -0.8992],
           [ 3.0466,  1.8083,  0.9004],
           [-1.8087,  3.1651, -0.0003],
           [-2.9322,  2.1027,  0.8881],
           [-2.9346,  2.1021, -0.8849],
        ])*angstrom,
        ffatypes=['O', 'O', 'N', 'N', 'N', 'N', 'C', 'C', 'C', 'C', 'C', 'C',
                  'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
        bonds=[[0, 8], [1, 9], [2, 7], [9, 2], [2, 11], [3, 6], [10, 3],
               [3, 12], [8, 4], [9, 4], [4, 13], [5, 7], [10, 5], [6, 7],
               [8, 6], [10, 14], [11, 15], [16, 11], [17, 11], [18, 12],
               [19, 12], [20, 12], [13, 21], [13, 22], [13, 23]],
        rvecs=np.array([]),
    )


def get_system_butanol():
    # structure taken from pubchem
    return System(
        numbers=np.array([8, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]),
        pos=np.array([
           [-1.8622,  0.0000, -0.1725],
           [-0.5458,  0.0001,  0.4145],
           [ 0.4329,  1.0636, -0.1254],
           [ 0.4327, -1.0637, -0.1253],
           [ 1.5424,  0.0000,  0.0087],
           [-0.6136,  0.0033,  1.5101],
           [ 0.5601,  1.9522,  0.4983],
           [ 0.2340,  1.3756, -1.1571],
           [ 0.2337, -1.3760, -1.1569],
           [ 0.5595, -1.9521,  0.4986],
           [ 2.2999, -0.0001, -0.7787],
           [ 2.0392, -0.0001,  0.9857],
           [-2.2842, -0.8098,  0.1615],
        ])*angstrom,
        ffatypes=['O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
        bonds=[[0, 1], [0, 12], [1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [2, 7],
               [3, 4], [8, 3], [9, 3], [10, 4], [11, 4]],
        rvecs=np.array([]),
    )
