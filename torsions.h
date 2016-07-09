/* Author: Ram Samudrala (me@ram.org)
 *
 * January 1, 1998.
 */

#ifndef __TORSIONS__
#define __TORSIONS__

/******************************************************************/

extern int convert_torsion_angles(torsion_def torsion_data[], residue chain[]);
extern int copy_torsions(double old_torsions[MAX_NUM_TORSIONS][MAX_RESIDUES], double new_torsions[MAX_NUM_TORSIONS][MAX_RESIDUES]);
extern int read_torsions(char torsion_list_filename[], residue chain[]);

/******************************************************************/

#endif /* __TORSIONS__ */
