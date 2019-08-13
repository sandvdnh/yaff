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
'''Collective variables

   This module implements the computation of collective variables and their
   derivatives, typically used in advanced sampling methods such as umbrella
   sampling or metadynamics. The ``CollectiveVariable`` class is the main item
   in this module, which is normally used in conjuction with an instance of the
   ``Bias`` class. Note that many collective variables such as bond lengths,
   bending angles, improper angles, ... are already implemented by the
   :mod:`yaff.pes.iclist` module, so no separate implementation needs to be
   provided here.
'''


from __future__ import division

import numpy as np

from yaff.log import log
from yaff.pes.dlist import DeltaList
from yaff.pes.iclist import InternalCoordinateList
from yaff.sampling.utils import cell_lower


__all__ = [
    'CollectiveVariable', 'CVVolume', 'CVCOMProjection','CVInternalCoordinate',
    'CVLinCombIC',
]


class CollectiveVariable(object):
    '''Base class for collective variables.'''
    def __init__(self, name, system):
        """
           **Arguments:**

           name
                A name for the collective variable.

           system
                The system for the collective variable.
        """
        self.name = name
        self.system = system
        self.value = np.nan
        self.gpos = np.zeros((system.natom, 3), float)
        self.vtens = np.zeros((3, 3), float)

    def get_conversion(self):
        '''Auxiliary routine that allows base classes the specify the unit
           conversion associated with the internal coordinate.
        '''
        raise NotImplementedError

    def get_log(self):
        '''Describe the internal coordinate in a format that is suitable for
           screen logging.
        '''
        return '%s' % (self.__class__.__name__)

    def compute(self, gpos=None, vtens=None):
        """Compute the collective variable and optionally some derivatives

           The only variable inputs for the compute routine are the atomic
           positions and the cell vectors.

           **Optional arguments:**

           gpos
                The derivatives of the collective variable towards the Cartesian
                coordinates of the atoms. ('g' stands for gradient and 'pos'
                for positions.)
                This must be a writeable numpy array with shape (N, 3) where N
                is the number of atoms.

           vtens
                The force contribution to the pressure tensor. This is also
                known as the virial tensor. It represents the derivative of the
                energy towards uniform deformations, including changes in the
                shape of the unit cell. (v stands for virial and 'tens' stands
                for tensor.) This must be a writeable numpy array with shape (3,
                3).

           The collective variable value is returned. The optional arguments
           are Fortran-style output arguments. When they are present, the
           corresponding results are computed and **stored** to the current
           contents of the array.
        """
        #Subclasses implement their compute code here.
        raise NotImplementedError

    def get_last_computed_value(self):
        """Return the last value that was computed. It is not assured that this
           value reflects the value for the current state of the system. This
           is merely a convenience method to obtain the value without
           performing an actual computation.
        """
        return self.value


class CVInternalCoordinate(CollectiveVariable):
    '''
       An InternalCoordinate disguised as a CollectiveVariable so that it can
       be used together with a BiasPotential.
       This is less efficient than using the InternalCoordinate with a
       ValenceTerm, so the latter is preferred if it is possible.
    '''
    def __init__(self, system, ic, comlist=None):
        raise NotImplementedError
        self.system = system
        self.ic = ic
        self.comlist = comlist
        self.dlist = DeltaList(system if comlist is None else comlist)
        self.iclist = InternalCoordinateList(self.dlist)
        self.iclist.add_ic(ic)

    def get_conversion(self):
        return self.ic.get_conversion()

    def compute(self, gpos=None, vtens=None):
        if self.comlist is not None:
            self.comlist.forward()
        self.dlist.forward()
        self.iclist.forward()
        self.value = self.iclist.ictab[0]['value']
        if gpos is not None: gpos[:] = 0.0
        if vtens is not None: vtens[:] = 0.0
        if not ((gpos is None) and (vtens is None)):
            self.iclist.ictab[0]['grad'] = 1.0
            self.iclist.back()
            if self.comlist is None:
                self.dlist.back(gpos, vtens)
            else:
                self.comlist.gpos[:] = 0.0
                self.dlist.back(self.comlist.gpos, vtens)
                self.comlist.back(gpos)
        return self.value


class CVVolume(CollectiveVariable):
    '''The volume of the simulation cell.'''
    def __init__(self, system):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.
        '''
        if system.cell.nvec == 0:
            raise TypeError('Can not compute volume of a non-periodic system.')
        CollectiveVariable.__init__(self, 'CVVolume', system)

    def get_conversion(self):
        return np.power(log.length.conversion, self.system.cell.nvec)

    def compute(self, gpos=None, vtens=None):
        self.value = self.system.cell.volume
        if gpos is not None:
            # No dependence on atomic positions
            gpos[:] = 0.0
        if vtens is not None:
            vtens[:] = np.identity(3)*self.value
        return self.value


class CVCOMProjection(CollectiveVariable):
    '''Compute the vector connecting two centers of masses and return the
       projection along a selected vector. cv=(r_{COM}^{B}-r_{COM}^{A})[index]
       and r_{COM} is a vector with centers of mass of groups A and B:

            * first component: projected onto ``a`` vector of cell
            * second component: projected onto vector perpendicular to ``a``
              and in the plane spanned by ``a`` and ``b``
            * third component: projected onto vector perpendicular to ``a`` and
              ``b``

       Note that periodic boundary conditions are NOT taken into account

            * the centers of mass are computed using absolute positions; this is
              most likely the desired behavior
            * the center of mass difference can in principle be periodic, but
              the periodicity is not the same as the periodicity of the system,
              because of the projection on a selected vector
    '''
    def __init__(self, system, groups, index):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class

           groups
                List of 2 arrays, each array containing atomic indexes
                used to compute one of the centers of mass

           index
                Selected projection vector:
                    * if index==0, projection onto ``a`` vector of cell
                    * if index==1, projection onto vector perpendicular to ``a``
                      and in the plane spanned by ``a`` and ``b``
                    * if index==2, projection onto vector perpendicular to ``a``
                      and ``b``
        '''
        raise NotImplementedError
        CollectiveVariable.__init__(self, 'CVCOMProjection', system)
        self.index = index
        # Safety checks
        assert len(groups)==2, "Exactly 2 groups need to be defined"
        assert system.cell.nvec==3, "Only 3D periodic systems are supported"
        assert self.index in [0,1,2], "Index should be one of 0,1,2"
        # Masses need to be defined in order to compute centers of mass
        if self.system.masses is None:
            self.system.set_standard_masses()
        # Define weights w_i such that difference of centers of mass can be
        # computed as sum_i w_i r_i
        self.weights = np.zeros((system.natom))
        self.weights[groups[0]] = -self.system.masses[groups[0]]/np.sum(self.system.masses[groups[0]])
        self.weights[groups[1]] = self.system.masses[groups[1]]/np.sum(self.system.masses[groups[1]])

    def get_conversion(self):
        return log.length.conversion

    def compute(self, gpos=None, vtens=None):
        '''
           Consider a rotation of the entire system such that the ``a`` vector
           is aligned with the X-axis, the ``b`` vector is in the XY-plane, and
           the ``c`` vector chosen such that a right-handed basis is formed.
           The rotated cell is lower-diagonal in the Yaff notation.

           In this rotated system, it is fairly simple to compute the required
           projections and derivatives, because the projections are simply the
           Cartesian components. Values obtained in the rotated system are then
           transformed back to the original system.
        '''
        # Compute rotation that makes cell lower diagonal
        _, R = cell_lower(self.system.cell.rvecs)
        # The projected vector of centers of mass difference (aka the
        # collective variable) in the rotated system
        cv_orig = np.sum(self.weights.reshape((-1,1))*self.system.pos, axis=0)
        # Transform back to the original system
        cv = np.dot(R, cv_orig)
        self.value = cv[self.index]
        if gpos is not None:
            gpos[:] = 0.0
            gpos[:,self.index] = self.weights
            # Forces (vector) need to be rotated back to original system
            gpos[:] = np.einsum('ij,kj', gpos, R.T)
        if vtens is not None:
            vtens[:] = 0.0
            vtens[self.index,self.index:] = cv[self.index:]
            vtens[self.index:,self.index] = cv[self.index:]
            # Virial (tensor) needs to be rotated back to original system
            vtens[:] = np.dot(R.T,np.dot(vtens[:],R))
        return self.value


class CVLinCombIC(CollectiveVariable):
    '''
       A linear combination of InternalCoordinates:
        cv = w0*ic0 + w1*ic1 + ...
    '''
    def __init__(self, system, ics, weights, comlist=None):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           ics
                A list of InternalCoordinate instances.

           weights
                A list defining the weight of each InternalCoordinate that is
                used when computing the linear combination.

           **Optional arguments:**

           comlist

                An instance COMList; if provided, this is used instead of the
                normal DeltaList to compute the InternalCoordinates

        '''
        raise NotImplementedError
        assert len(weights)==len(ics)
        self.system = system
        self.ics = ics
        self.comlist = comlist
        self.dlist = DeltaList(system if comlist is None else comlist)
        self.iclist = InternalCoordinateList(self.dlist)
        for ic in self.ics:
            self.iclist.add_ic(ic)
        self.weights = weights

    def get_conversion(self):
        # Units depend on the particular linear combination of internal
        # coordinates
        return 1.0

    def compute(self, gpos=None, vtens=None):
        if self.comlist is not None:
            self.comlist.forward()
        self.dlist.forward()
        self.iclist.forward()
        self.value = 0.0
        for iic in range(len(self.ics)):
            self.value += self.weights[iic]*self.iclist.ictab[iic]['value']
        if gpos is not None: gpos[:] = 0.0
        if vtens is not None: vtens[:] = 0.0
        if not ((gpos is None) and (vtens is None)):
            for iic in range(len(self.ics)):
                # Derivative of the linear combination to this particular
                # internal coordinate
                self.iclist.ictab[iic]['grad'] = self.weights[iic]
            self.iclist.back()
            if self.comlist is None:
                self.dlist.back(gpos, vtens)
            else:
                self.comlist.gpos[:] = 0.0
                self.dlist.back(self.comlist.gpos, vtens)
                self.comlist.back(gpos)
        return self.value
