# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
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
"""Valence interactions between centers of mass."""


import numpy as np

from yaff.pes.ext import delta_dtype, dlist_forward, dlist_back, comlist_dtype, \
    comlist_forward, comlist_back
from yaff.system import AbstractSystem


__all__ = ['COMList']


class COMList(AbstractSystem):
    """Center of Mass List."""

    def __init__(self, system, groups, numbers, ffbtypes, ffbtype_ids):
        """Initialize a COMList

        Parameters
        ----------
        system : yaff.system.System
            An object describing the atomistic system.
        groups : list of (indices, weights) arrays.
            Each item is a tuple with first an array of atom indices for who the center
            of mass is to be computed and second an array with weights, used for the
            center of mass of these atoms. Weights do not have to add up to one but they
            should be positive (not checked).
        numbers: numpy array of atomic numbers, one for each bead. These only influence
            subsequent visualization of .xyz files.
        ffbtypes: list of names of the bead types (e.g. 'M' or 'L')
        ffbtype_ids: numpy array of indices that indicate which bead is which ffbtype
        """
        AbstractSystem.__init__(self, numbers, None, None, ffbtypes, ffbtype_ids)
        self.system = system
        self.cell = self.system.cell
        self.groups = groups
        self._get_bonds() # derive bonds between beads from self.system
        self._init_derived() #build neighs based on self.bonds

        self.pos = np.zeros((len(groups), 3), float)
        self.gpos = np.zeros((len(groups), 3), float)

        self.comsizes = np.array([len(iatoms) for iatoms, weights in groups], dtype=int)

        # Centers of mass are computed from relative vectors because some groups may
        # be split over unit cell boundaries. On each relative vector, the minimum
        # image convention is applied before computing weighted averages.
        ndelta = self.comsizes.sum() - len(self.comsizes)
        self.deltas = np.zeros(ndelta, delta_dtype)
        self.comtab = np.zeros(self.comsizes.sum(), comlist_dtype)
        idelta = 0
        icom = 0
        for iatoms, weights in groups:
            # The first record in a group contains the reference atom and the total weight
            # of the group.
            self.comtab[icom]['i'] = iatoms[0]
            self.comtab[icom]['w'] = weights.sum()
            icom += 1
            for iatom, weight in zip(iatoms[1:], weights[1:]):
                self.deltas[idelta]['i'] = iatoms[0]
                self.deltas[idelta]['j'] = iatom
                # Subsquent records in one group then contain indices of relative vectors
                # and weights of the corresponding atoms.
                self.comtab[icom]['i'] = idelta
                self.comtab[icom]['w'] = weight
                idelta += 1
                icom += 1

    def forward(self):
        """Evaluate the relative vectors for ``self.system.pos``

           The actual computation is carried out by a low-level C routine.
        """
        dlist_forward(self.system.pos, self.system.cell, self.deltas, len(self.deltas))
        comlist_forward(self.deltas, self.system.pos, self.pos, self.comsizes, self.comtab)

    def back(self, gpos, vtens):
        """Derive gpos and virial from the derivatives towards the relative vectors

           The actual computation is carried out by a low-level C routine.
        """
        comlist_back(self.deltas, gpos, self.gpos, self.comsizes, self.comtab)
        dlist_back(gpos, vtens, self.deltas, len(self.deltas))

    def _get_bonds(self):
        '''
        Derives the bonds between the beads based on the bonds of the underlying
        atomistic system in self.system
        '''
        bonds = []
        for bond in self.system.bonds:
            a0 = bond[0]
            a1 = bond[1]
            candidate = [self._get_group(a0), self._get_group(a1)]
            if (tuple(candidate) not in bonds and tuple(candidate[::-1]) not in bonds):
                if candidate[0] != candidate[1]:
                    bonds.append(tuple(candidate))
        self.bonds = bonds

    def _get_group(self, atom):
        '''
        Returns the index of the group where atom belongs to

        Arguments
        ---------
        atom: index of the atom
        '''
        for i, group in enumerate(self.groups):
            members = group[0]
            if atom in members:
                return i
        return -1

