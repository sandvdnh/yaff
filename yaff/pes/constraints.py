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
"""
Implements all types of constraints used during the generation of a force field.
"""
import numpy as np
from yaff.pes.ff import ForcePartValence

__all__ = [
        'Constraint', 'ICConstraint',
        ]

class Constraint(object):
    '''
    Base class for anything that evaluates whether a ``ValenceTerm`` instance should be
    added to the current ``ForcePartValence``.

    Constraints are always defined separately for each interaction.
    '''

    def __init__(self, prefix, pars, **kwargs):
        self.prefix = prefix
        self.pars = pars

    def satisfy(self, vterm, pars, part, system):
        raise NotImplementedError

class ICConstraint(Constraint):
    '''
    Implements a constraint based on the current value of the internal coordinate of the ``ValenceTerm``.
    '''

    def __init__(self, prefix, pars, **kwargs):
        Constraint.__init__(self, prefix, pars)
        self.rv = kwargs['rv']
        self.eps = kwargs['eps']

    def satisfy(self, vterm, pars, part, system):
        if self.pars == pars:
            part_valence = ForcePartValence(system)
            part_valence.add_term(vterm)
            part_valence.compute()
            ic_index = part_valence.vlist.vtab[0]['ic0']
            ic = part_valence.vlist.iclist.ictab[ic_index]['value']
            return np.abs(ic - self.rv ) < self.eps
        else:
            return True
