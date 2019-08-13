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
'''Bias potentials'''


from __future__ import division

import numpy as np

from yaff.log import log, timer


__all__ = [
    'BiasPotential', 'HarmonicBias', 'PathDeviationBias', 'LowerWallBias',
    'UpperWallBias', 'LegendreBias',
]


class BiasPotential(object):
    '''Base class for energy terms used to bias the system.'''
    def __init__(self, pars, cvs):
        '''
           **Arguments:**

           pars
                A list of parameters to be stored for this energy term.

           cvs
                A list of ``CollectiveVariables``
        '''
        self.pars = pars
        self.cvs = cvs

    def get_log(self):
        '''Describe the covalent energy term in a format that is suitable for
           screen logging.
        '''
        raise NotImplementedError

    def compute(self, gpos=None, vtens=None):
        """Compute the bias potential and optionally some derivatives

           The only variable inputs for the compute routine are the atomic
           positions and the cell vectors.

           **Optional arguments:**

           gpos
                The derivatives of the collective towards the Cartesian
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


class LegendreBias(BiasPotential):
    '''
    Legendre polynomial bias
    '''
    def __init__(self, order, domain, cv):
        '''
        order: 1D numpy array where the ith element specifies the coefficient
        of the ith polynomial

        domain: domain of the polynomials (which is automatically mapped to
        [-1, 1] by default)
        '''
        BiasPotential.__init__(self, (order, domain), [cv])
        self.func = np.polynomial.legendre.Legendre(order, domain=domain)
        self.deriv = self.func.deriv(m=1)

    def get_log(self):
        c = self.cvs[0].get_conversion()
        return '{}: pars: {} domain: {}'.format(
                self.__class__.__name__,
                self.order,
                self.domain
                )

    def compute(self, gpos=None, vtens=None):
        q = self.cvs[0].compute(gpos=gpos, vtens=vtens)
        energy = self.func(q)
        deriv = self.deriv(q)
        if gpos is not None:
            gpos *= deriv
        if vtens is not None:
            vtens *= deriv
        return energy


class HarmonicBias(BiasPotential):
    '''The harmonic energy term: 0.5*K*(q-q0)^2'''
    def __init__(self, fc, rv, cv):
        '''
           **Arguments:**

           fc
                The force constant (in atomic units).

           rv
                The rest value (in atomic units).

           cv
                A ``CollectiveVariable`` object.
        '''
        BiasPotential.__init__(self, [fc, rv], [cv])

    def get_log(self):
        c = self.cvs[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )

    def compute(self, gpos=None, vtens=None):
        q = self.cvs[0].compute(gpos=gpos,vtens=vtens)
        x = q-self.pars[1]
        e = 0.5*self.pars[0]*x**2
        if gpos is not None:
            gpos[:] *= self.pars[0]*x
        if vtens is not None:
            vtens[:] *= self.pars[0]*x
        return e


class LowerWallBias(BiasPotential):
    '''Harmonic energy term, but zero if q>q0: 0.5*K*(q-q0)^2*H(q0-q)'''
    def __init__(self, fc, rv, cv):
        '''
           **Arguments:**

           fc
                The force constant (in atomic units).

           rv
                The rest value (in atomic units).

           cv
                A ``CollectiveVariable`` object.
        '''
        BiasPotential.__init__(self, [fc, rv], [cv])

    def get_log(self):
        c = self.cvs[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )

    def compute(self, gpos=None, vtens=None):
        q = self.cvs[0].compute(gpos=gpos,vtens=vtens)
        x = q-self.pars[1]
        x*= q<self.pars[1]
        e = 0.5*self.pars[0]*x**2
        if gpos is not None:
            gpos[:] *= self.pars[0]*x
        if vtens is not None:
            vtens[:] *= self.pars[0]*x
        return e


class UpperWallBias(BiasPotential):
    '''Harmonic energy term, but zero if q<q0: 0.5*K*(q-q0)^2*H(q-q0)'''
    def __init__(self, fc, rv, cv):
        '''
           **Arguments:**

           fc
                The force constant (in atomic units).

           rv
                The rest value (in atomic units).

           cv
                A ``CollectiveVariable`` object.
        '''
        BiasPotential.__init__(self, [fc, rv], [cv])

    def get_log(self):
        c = self.cvs[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )

    def compute(self, gpos=None, vtens=None):
        q = self.cvs[0].compute(gpos=gpos,vtens=vtens)
        x = q-self.pars[1]
        x*= q>self.pars[1]
        e = 0.5*self.pars[0]*x**2
        if gpos is not None:
            gpos[:] *= self.pars[0]*x
        if vtens is not None:
            vtens[:] *= self.pars[0]*x
        return e


class PathDeviationBias(BiasPotential):
    '''Given a discrete path in the multidimensional collective variable space,
       this potential consists of the energy of the nearest point on the path plus
       a harmonic potential for the distance to the nearest point.

       Suppose the vectors of collective variables on the path are given by
       sigma_i for i=1..n. Suppose the current state of the system is sigma. First
       the index i is sought for which \|sigma-sigma_i\| is minimal. This expression
       for this bias potential is then U=U_path + 0.5*K*|sigma-sigma_i|**2

       U_path is computed as a three point interpolation:

       U_path = c_{i-1}*U_{i-1}+c_{i}*U_{i}+c_{i+1}*U_{i+1} with the coefficients:
            * c_{i} =   0.5*( d_{i-1}/(d_{i-1}+d_{i}) + d_{i+1}/(d_{i}+d_{i+1} )
            * c_{i-1} = 0.5*( d_{i}/(d_{i-1}+d_{i}) )
            * c_{i+1} = 0.5*( d_{i}/(d_{i}+d_{i+1}) )

       where d_{i} = \|sigma-sigma_i\|**2

       Note that this potential is **not** continuous when the nearest point on
       the path changes from one to the other.

       Defining a distance in collective variable space is problematic: different
       collective variables can even have different units. We use the following
       expression: \|sigma\| = \sqrt{ Sum_a (w_a sigma_a)**2 } where a runs over all
       collective variables and w_a are weights provided by the user.
    '''
    def __init__(self, cvs, path, fc, weights=None, periodicities=None):
        '''
           **Arguments:**

           cvs
                A list of CollectiveVariables

           path
                Specification of the path, either a filename to text file or a
                NumPy array. In either case, the first N columns provide the
                collective variable coordinates of the path, with N the number
                of collective variables. The next column provides the biasing
                energy along the path.

           fc
                Force constant K appearing in the harmonic restraint

           **Optional arguments:**

           weights
                Contribution of each collective variable when computing
                distances in collective variable space, default is all equal to
                1

           periodicities
                List containing the periodicity of each collective variable.
                Specify ``None`` if the collective variable is not periodic.
                Default is ``None`` for all collective variables.

           **TODO:**

                * Testing periodic boundary conditions
                * Testing the search for closest point on the path
        '''
        self.cvs = cvs
        self.fc = fc
        self.ncv = len(cvs)
        if weights is None:
            weights = np.ones((self.ncv))
        if periodicities is None:
            periodicities = [None]*self.ncv
        assert weights.shape[0]==self.ncv
        assert len(periodicities)==self.ncv
        self.weights = weights
        self.periodicities = periodicities
        # Load the path from a txt file if a filename was passed
        if isinstance(path, str):
            path = np.loadtxt(path)
        # Check that we now have a NumPy array of the correct shape
        assert isinstance(path, np.ndarray)
        assert path.shape[1]==self.ncv+1
        self.path_coordinates = path[:,:self.ncv]
        self.path_energies = path[:,self.ncv]
        self.extended_energies = np.insert(self.path_energies, 0, 0.0)
        self.extended_energies = np.append(self.extended_energies, 0.0)
        self.npoints = self.path_energies.shape[0]
        # Arrays that can hold derivatives for each collective variable
        self.values_cvs = np.zeros((self.ncv,))
        self.gpos_cvs = np.zeros((self.ncv, self.cvs[0].gpos.shape[0], self.cvs[0].gpos.shape[1]))
        self.vtens_cvs = np.zeros((self.ncv, self.cvs[0].vtens.shape[0], self.cvs[0].vtens.shape[1]))

    def get_log(self):
        return '%s'%self.__class__.__name__

    def find_nearest_point(self, x):
        '''
           Search the point on the path that is closest to x

           **Arguments:**

           x
                A NumPy array specifying the values of the collective variables

           **Returns:**

           index
                The index of the nearest point on the path

           sqdists
                the squares of the distances to the left, central, and right
                points on the path. Central point is the nearest point.

           deltas
                the vectors connecting x with the left, central, and right
                points on the path. Central point is the nearest point.
        '''
        assert x.ndim==1
        assert x.shape[0]==self.ncv
        deltas = x-self.path_coordinates
        # Apply minimum image convention for periodic collective variables
        for icv, period in enumerate(self.periodicities):
            if period is None: continue
            deltas[:,icv] -= period*np.floor(deltas[:,icv]/period)
        # Compute square of distances
        sqdists = np.einsum('ia,a',deltas**2,self.weights**2)
        index = np.argmin(sqdists)
        if index==0:
            sqdists = np.insert(sqdists[:index+2], 0, 0.0)
            deltas = np.insert(deltas[:index+2], 0, [0.0]*self.ncv, axis=0)
        elif index==self.npoints-1:
            sqdists = np.append(sqdists[index-2:], 0.0)
            deltas = np.append(deltas[index-2:], [0.0]*self.ncv, axis=0)
        else:
            sqdists = sqdists[index-1:index+2]
            deltas = deltas[index-1:index+2]
        assert sqdists.shape[0]==3
        assert deltas.shape[0]==3
        return index, sqdists, deltas

    def compute(self, gpos=None, vtens=None):
        # Compute collective variable values and optionally derivatives by
        # looping over all collective variables
        self.values_cvs[:] = 0.0
        self.gpos_cvs[:] = 0.0
        self.vtens_cvs[:] = 0.0
        for icv in range(self.ncv):
            if gpos is not None:
                my_gpos = self.gpos_cvs[icv]
            else: my_gpos = None
            if vtens is not None:
                my_vtens = self.vtens_cvs[icv]
            else: my_vtens = None
            self.values_cvs[icv] = self.cvs[icv].compute(gpos=my_gpos,vtens=my_vtens)
        # Get the index of the nearest point on the path, the square of the
        # distances, and the vectors connecting the path with the current point
        index, sqdists, deltas = self.find_nearest_point(self.values_cvs)
        energies = self.extended_energies[index:index+3]
        # First contribution: interpolate the path energy using following
        # coefficients
        coeff = np.zeros((3))
        coeff[0] = 0.5*sqdists[1]/(sqdists[0]+sqdists[1])
        coeff[1] = 0.5*sqdists[0]/(sqdists[0]+sqdists[1])
        coeff[1]+= 0.5*sqdists[2]/(sqdists[2]+sqdists[1])
        coeff[2] = 0.5*sqdists[1]/(sqdists[2]+sqdists[1])
        energy = np.sum(energies*coeff)
        # Special cases: endpoints of the path, so interpolation is only
        # between two points
        if index in [0, self.npoints]: energy *= 2.0
        if not ((gpos is None) and (vtens is None)):
            path_derivatives = np.zeros((self.ncv,))
            path_derivatives += (deltas[0]*sqdists[1]-deltas[1]*sqdists[0])*\
                (energies[1]-energies[0])/(sqdists[0]+sqdists[1])**2
            path_derivatives += (deltas[2]*sqdists[1]-deltas[1]*sqdists[2])*\
                (energies[1]-energies[2])/(sqdists[2]+sqdists[1])**2
            path_derivatives *= self.weights**2
            if index in [0, self.npoints]: path_derivatives *= 2.0
        # Derivatives towards atomic positions and cell tensor elements are
        # computed using the chain rule
        if gpos is not None:
            gpos[:] = np.einsum('aij,a->ij', self.gpos_cvs, path_derivatives)
        if vtens is not None:
            vtens[:] = np.einsum('aij,a->ij', self.vtens_cvs, path_derivatives)
        # Second contribution: harmonic restraint on deviation from the path
        energy += 0.5*self.fc*sqdists[1]
        if gpos is not None:
            gpos[:] += np.einsum('aij,a->ij', self.gpos_cvs, self.fc*self.weights**2*deltas[1])
        if vtens is not None:
            vtens[:] += np.einsum('aij,a->ij', self.vtens_cvs, self.fc*self.weights**2*deltas[1])
        return energy
