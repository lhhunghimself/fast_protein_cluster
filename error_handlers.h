/* Author: Ram Samudrala (me@ram.org)
 *
 * November 10, 1995.
 * Hong Hung 2012
 */

#ifndef __ERROR_HANDLERS__
#define __ERROR_HANDLERS__

/******************************************************************/

#include "data_structures.h"

extern int check_atomp(atom *atomp, const char routine_name[]);
extern int check_eof(int status, const char routine_name[]);
extern int check_malloc(void *pointer, const char pointer_string[], const char routine_name[]);
extern int check_maximum_value(int value, int maximum_value, const char routine_name[]);
extern int check_null(void *pointer, const char routine_name[]);
extern int open_file(FILE **fp, const char filename[], const char status[],  const char routine_name[]);
extern int close_file(FILE **fp, const char filename[], const char routine_name[]);
extern int myisnan_(int *nanflag, double *value);
	
/******************************************************************/

#endif /* __ERROR_HANDLERS__ */
