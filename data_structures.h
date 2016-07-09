/* Author: Ram Samudrala (me@ram.org)
 *
 * October 25, 1995.
 */

#ifndef __DATA_STRUCTURES__
#define __DATA_STRUCTURES__

#include "defines.h"
/******************************************************************/
/* Angular distribution structures */
struct _angular_distribution
{
 double *phi_mean;
 double *psi_mean;
 double *phi_dev;
 double *psi_dev;
 int *sector_probability_matrix;
 int total_sector_probability;
 int number_of_sectors;
};
typedef struct _angular_distribution angular_distribution;

/******************************************************************/
/* General structures */

struct _atom
{
  char aname[ATOM_NAME_LENGTH];
  int atype;
  int ano;  
  double loc[3]; /* 0..3 */
  struct _atom *next;
  struct _atom *prev;
};

typedef struct _atom atom;

struct _residue
{
  char rchar;
  int rid; 
  int rno;
  char rss;
  int rss_confidence;
  int num_alt_atoms;
  double r_phi;
  double r_psi;
  double chis[MAX_NUM_CHIS];
  atom *atoms;                      /* Main coordinates. */
  atom *alt_atoms[MAX_ALTERNATE_ATOM_CONFORMATIONS];  /* Alternate conformations. */
  int move_probability;
  int ss_triplet_probability_matrix[SS_PROBABILITY_SCALE];
  angular_distribution *angle_distro;
  int number_of_ss_triplets;
  int total_ss_triplet_probability;
  int moved;
  int ss;
  int allowed_sectors[36][36];
  double phi_std_dev;
  double psi_std_dev;
};

typedef struct _residue residue;

/******************************************************************/
/* Loop constraint structures. */

struct _noe_constraint
{
 int res_num1;
 int res_num2;
 int res_type1;
 int res_type2;
 int atom1;
 int atom2;
 double scores[NUM_NOE_BINS];
};
typedef struct _noe_constraint noe_constraint;
#define NOE_TABLE "/hosts/time/maxa/home/lhhung/allpro_pdb/clean/1khmA.rnoe"

struct _atom_constraint
{
 int r1;
 int r2;
 int a1;
 int a2;
 double upper_limit;
 double lower_limit;
 double upper_penalty;
 double lower_penalty;
 double bonus;
};
typedef struct _atom_constraint atom_constraint;

struct _constraint
{
  int residue1;
  int residue2;
  double distance;
  double tolerance;
};

typedef struct _constraint constraint;

#define constraint_res1(c) ((c)->residue1)
#define constraint_res2(c) ((c)->residue2)
#define constraint_distance(c) ((c)->distance)
#define constraint_tolerance(c) ((c)->tolerance)

/******************************************************************/
/* Torsion related structures. */

struct _torsion_def
{
  int num_atoms; /* Number of atoms in the structure (per residue). */
  char atoms[MAX_ATOMS_PER_RESIDUE][ATOM_NAME_LENGTH]; /* The atom order for the residue. */
  int num_phi_values;
  int num_psi_values;
  double accepted_phi_values[MAX_ACCEPTED_PHI_VALUES];
  double accepted_psi_values[MAX_ACCEPTED_PSI_VALUES];
  int num_chis; /* Number of chi angles. */
  int num_chi_values[MAX_NUM_CHIS];  /* Number of values per chi angle. */
  double accepted_chi_values[MAX_NUM_CHIS][MAX_ACCEPTED_CHI_VALUES];
};

typedef struct _torsion_def torsion_def;

/******************************************************************/
/* Torsion related macros */

#define torsion_def_num_atoms(t) ((t)->num_atoms)
#define torsion_def_atoms(t, i) ((t)->atoms[i])
#define torsion_def_num_phi_values(t) ((t)->num_phi_values)
#define torsion_def_num_psi_values(t) ((t)->num_psi_values)
#define torsion_def_phi_values(t, i) ((t)->accepted_phi_values[i])
#define torsion_def_psi_values(t, i) ((t)->accepted_psi_values[i])
#define torsion_def_num_chis(t) ((t)->num_chis)
#define torsion_def_num_chi_values(t, i) ((t)->num_chi_values[i])
#define torsion_def_chi_values(t, i, j) ((t)->accepted_chi_values[i][j])
#define torsion_def_hydrophobic(t, i) (((((t)->atoms[i])[0]) == 'C') && ((((t)->atoms[i])[1]) != '\0'))

/******************************************************************/
/* General macros */

#define atom_name(a) ((a)->aname)
#define atom_type(a) ((a)->atype)
#define atom_no(a) ((a)->ano)
#define atom_loc(a) ((a)->loc)
#define atom_x(a) ((a)->loc[0])
#define atom_y(a) ((a)->loc[1])
#define atom_z(a) ((a)->loc[2])
#define atom_next(a) ((a)->next)
#define atom_prev(a) ((a)->prev)

#define res_char(r) ((r)->rchar)
#define res_id(r) ((r)->rid)
#define res_no(r) ((r)->rno)
#define res_ss(r) ((r)->rss)
#define res_ss_confidence(r) ((r)->rss_confidence)
#define res_atoms(r) ((r)->atoms)
#define res_chis(r, i) ((r)->chis[i])
#define res_psi(r) ((r)->r_psi)
#define res_phi(r) ((r)->r_phi)

#define res_alt_atoms(r, i) ((r)->alt_atoms[i])
#define res_num_alt_atoms(r) ((r)->num_alt_atoms)

#define molecule_size(r) ((r[0])->rno)
#define molecule_number_of_residues(r) ((r[0])->num_alt_atoms)

/******************************************************************/
/* Graph related structures. */

struct _node
{
  int residue_nos[NUM_RESIDUES_PER_NODE];
  int sc_nos[NUM_RESIDUES_PER_NODE];
  int mc_nos[NUM_RESIDUES_PER_NODE];
  double node_score;
};

typedef struct _node node;

struct _clique 
{
  double c_score;
  int vertices[MAX_NODES];
};

typedef struct _clique clique;

/******************************************************************/
/* Graph-theory related macros */

/* For one residue per node */
#define node_res_no1(n) ((n)->residue_nos[0])
#define node_sc_no1(n) ((n)->sc_nos[0])
#define node_mc_no1(n) ((n)->mc_nos[0])
#define total_nodes(n) ((n[0])->residue_nos[0])

/* For two residues per node */
#ifdef DOUBLE_NODES
#define node_res_no2(n) ((n)->residue_nos[1])
#define node_sc_no2(n) ((n)->sc_nos[1])
#define node_mc_no2(n) ((n)->mc_nos[1])
#endif

#define node_score(n) ((n)->node_score)
#define edge_score(e) ((double) (((int) e) - MAX_SINGLE_EDGE_SCORE - 1))

/* Clique related structures */

#define clique_score(c) ((c)->c_score)
#define clique_size(c) ((c)->vertices[0])
#define clique_vertex(c, i) ((c)->vertices[i])
#define clique_vertices(c) ((c)->vertices)
  
#define clique_worst_score(c) ((c[0])->c_score)
#define clique_worst_score_index(c) ((c[0])->vertices[0])

/******************************************************************/
/* Miscellaneous macros. */

/* #define mcgen_conf_no(t) (t[0]) */

/******************************************************************/

#endif /* __DATA_STRUCTURES__ */
