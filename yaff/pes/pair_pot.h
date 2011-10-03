// YAFF is yet another force-field code
// Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
// for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
// reserved unless otherwise stated.
//
// This file is part of YAFF.
//
// YAFF is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// YAFF is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>
//
// --


#ifndef YAFF_PAIR_POT_H
#define YAFF_PAIR_POT_H

#include "nlists.h"


typedef double (*pair_fn_type)(void*, long, long, double, double*);

typedef struct {
  void *pair_data;
  pair_fn_type pair_fn;
  double rcut;
  int smooth;
} pair_pot_type;

typedef struct {
  long i;
  double scale;
} scaling_row_type;


pair_pot_type* pair_pot_new(void);
void pair_pot_free(pair_pot_type *pair_pot);
int pair_pot_ready(pair_pot_type *pair_pot);
double pair_pot_get_rcut(pair_pot_type *pair_pot);
void pair_pot_set_rcut(pair_pot_type *pair_pot, double rcut);
int pair_pot_get_smooth(pair_pot_type *pair_pot);
void pair_pot_set_smooth(pair_pot_type *pair_pot, int smooth);
void pair_data_free(pair_pot_type *pair_pot);

double pair_pot_compute(long center_index, nlist_row_type *nlist,
                        long nlist_size, scaling_row_type *scaling,
                        long scaling_size, pair_pot_type *pair_pot,
                        double *gpos, double* vtens);


typedef struct {
  double *sigma;
  double *epsilon;
} pair_data_lj_type;

void pair_data_lj_init(pair_pot_type *pair_pot, double *sigma, double *epsilon);
double pair_fn_lj(void *pair_data, long center_index, long other_index, double d, double *g);


typedef struct {
  double *sigma;
  double *epsilon;
} pair_data_mm3_type;

void pair_data_mm3_init(pair_pot_type *pair_pot, double *sigma, double *epsilon);
double pair_fn_mm3(void *pair_data, long center_index, long other_index, double d, double *g);


typedef struct {
  double *r0;
  double *c6;
} pair_data_grimme_type;

void pair_data_grimme_init(pair_pot_type *pair_pot, double *r0, double *c6);
double pair_fn_grimme(void *pair_data, long center_index, long other_index, double d, double *g);


typedef struct {
  double *amp;
  int amp_mix;
  double amp_mix_coeff;
  double *b;
  int b_mix;
  double b_mix_coeff;
} pair_data_exprep_type;

void pair_data_exprep_init(pair_pot_type *pair_pot, double *amp, int amp_mix, double amp_mix_coeff, double *b, int b_mix, double b_mix_coeff);
double pair_fn_exprep(void *pair_data, long center_index, long other_index, double d, double *g);
int pair_data_exprep_get_amp_mix(pair_pot_type *pair_pot);
double pair_data_exprep_get_amp_mix_coeff(pair_pot_type *pair_pot);
int pair_data_exprep_get_b_mix(pair_pot_type *pair_pot);
double pair_data_exprep_get_b_mix_coeff(pair_pot_type *pair_pot);


typedef struct {
  double *charges;
  double alpha;
} pair_data_ei_type;

void pair_data_ei_init(pair_pot_type *pair_pot, double *charges, double alpha);
double pair_fn_ei(void *pair_data, long center_index, long other_index, double d, double *g);


#endif