/* Author: Ram Samudrala (me@ram.org)
 * March 1, 1996. 
 * modified Hong Hung Sept 2006 2012
 */

#include "lite.h"

/******************************************************************/

int check_atomp(atom *atomp, const char routine_name[])
{
  if (atomp == ((atom *) NULL))
    {
      fprintf(stderr, "%s(): atomp is NULL when it shouldn't be!\n", routine_name);
      exit(FALSE);
    }
  return TRUE;
}

/******************************************************************/

int check_eof(int status, const char routine_name[])
{
  if (status == EOF)
    fprintf(stderr, "check_eof(): EOF returned in %s()!\n", routine_name);
  return TRUE;
}

/******************************************************************/

int check_malloc(void *pointer, const char pointer_string[], const char routine_name[])
{
  if (pointer == ((void *) NULL))
    {
      fprintf(stderr, "%s(): unable to allocate memory for %s!\n", routine_name, pointer_string);
      exit(FALSE);
    }
  return TRUE;
}

/******************************************************************/

int check_maximum_value(int value, int maximum_value, const char routine_name[])
{
  if (value >= maximum_value)
    {
      fprintf(stderr, "%s(): value, %d, exceeded maximum_value (%d)!\n", 
	      routine_name, value, maximum_value);
      exit(FALSE);
    }
  return TRUE;
}

/******************************************************************/

int check_null(void *pointer, const char routine_name[])
{
  if (pointer == ((void *) NULL))
    {
      fprintf(stderr, "%s(): NULL encountered unexpectedly!\n", routine_name);
      exit(FALSE);
    }
  return TRUE;
}

/******************************************************************/

int open_file(FILE **fp, const char filename[], const char status[], const char routine_name[])
{
  char buf[20];

  if (strcmp(filename, STDIN_FILENAME) == 0)
    {
      *fp = stdin;
      return TRUE;
    }
  if (strcmp(filename, STDOUT_FILENAME) == 0)
    {
      *fp = stdout;
      return TRUE;
    }
  if (strcmp(filename, STDERR_FILENAME) == 0)
    {
      *fp = stderr;
      return TRUE;
    }

  switch (status[0])
    {
    case 'r':
      strcpy(buf, "reading");
      break;
    case 'a':
      strcpy(buf, "appending");
      break;
    case 'w':
      strcpy(buf, "writing");
      break;
    default:
      fprintf(stderr, "open_file(): unknown status (%s) encountered!\n", status);
      break;
    }

  if ((*fp = fopen(filename, status)) == NULL)
    {
      fprintf(stderr, "%s(): couldn't open %s for %s!\n", routine_name, filename, buf);
      exit(FALSE);
    }
   else if(routine_name)
    fprintf(stderr, "%s(): opening %s for %s...\n", routine_name, filename, buf);

  return TRUE;
}

/******************************************************************/

int close_file(FILE **fp, const char filename[], const char routine_name[])
{
  if ((strcmp(filename, STDOUT_FILENAME) != 0) && (strcmp(filename, STDIN_FILENAME) != 0) &&
      (strcmp(filename, STDERR_FILENAME) != 0))
    {
     if(routine_name) fprintf(stderr, "%s(): closing %s.\n", routine_name, filename);
      fclose(*fp);
    }
  return TRUE;
}

/******************************************************************/

int myisnan_(int *nanflag, double *value)
{
  if (isnan(*value) == 0)
    *nanflag = 0;
  else
    *nanflag = 1;

  return TRUE;
}

/******************************************************************/
