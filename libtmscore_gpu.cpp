//Ling-Hong Hung Jan 2013
//contains GPU specific routines
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include "lite.h"
#include "error_handlers.h"
#include <omp.h>
#define MAX_ELEMENTS 8192*4096
#include <iostream>
#define PI 3.1415926535897932
#define SQRT3 1.732050807568877

using namespace std;

char *print_cl_errstring(cl_int err);

//routines for OpenCL version of TM-score -
int convert_coords_to_float4 (int nstructs,int pdb_size, float *coords, float4 *coords4);
int calculate_number_of_frames(int nat);
int define_sizes_string (char **defines_string, int nthreads, int pdb4_size);
int read_source_file(char **array,const char *filename,char *defines_string);
int define_decoy_sizes_string (char **defines_string, int nthreads, int pdb4_size);

//input output routines - provided by libtmscore-cpu library
int read_list_of_decoys(int *nat, char *filename,float **coords,char **names, int **name_offsets, int mode);
int read_CAs(char *filename, float *coords, int count_only,int center_coords,int mode);
int read_list_of_decoy_names(char *filename,char **names, int **name_offsets);

int define_sizes_string (char **defines_string, int nthreads, int pdb4_size){
 char buffer[1000];
 int n;
 n=sprintf(buffer,"#define NTHREADS %d\n#define PDB4SIZE %d\n",nthreads,pdb4_size);
 if(*defines_string)free(*defines_string);
 if(!(*defines_string=(char*)malloc(sizeof(char)*(n+1))))exit(FALSE); 
 strcpy(*defines_string,buffer);
 return(n);
}
int read_source_file(char **array,const char *filename,char *defines_string){
 FILE *fp;
 int read,size,n=0;
 char *my_array=0;
 if(defines_string)n=strlen(defines_string)+1;
 open_file(&fp, filename, "r", "read_source_file");
 fseek (fp , 0 , SEEK_END);
 size = ftell (fp);
 rewind (fp);
 if(!(my_array=(char*)malloc(sizeof(char)*(size+n))))exit(FALSE);
 strcpy(my_array,defines_string); 
 if(size)
 {
  read=fread(&(my_array[n-1]),sizeof(char),size,fp);
  close_file(&fp, filename, "read_source_file");
  my_array[size+n-1]='\0';
  *array=my_array;
  return(read);
 }
 return(0);
}

int calculate_number_of_frames(int nat){ 
 int nframes=0,len=nat;
 int L_ini_min=4;
 int divisor=1;
 if(nat <3)
 {
  fprintf(stderr,"need at least 3 atoms for alignment\n"); 
  exit(FALSE);
 }
 while(len > L_ini_min && divisor <=16 )
 {
  nframes+=nat-len+1;
  divisor*=2;
  len=nat/divisor;
 }
 nframes+=nat-L_ini_min+1;
 fprintf(stderr,"nat %d seeds %d\n",nat,nframes);
 return(nframes);
}
int convert_coords_to_float4 (int nstructs,int pdb_size, float *coords, float4 *coords4){
 //just rearrange and pad with zeros to 4;
 float my_coords[12];
 int p,j,k=0,natoms,m=0,mod4=0;
 natoms=pdb_size/3;
 for (p=0;p<nstructs;p++)
 {
  for(j=0;j<natoms/4;j++)
  {
   coords4[m].x=coords[k++];
   coords4[m+1].x=coords[k++];
   coords4[m+2].x=coords[k++];
   coords4[m].y=coords[k++];
   coords4[m+1].y=coords[k++];
   coords4[m+2].y=coords[k++];
   coords4[m].z=coords[k++];
   coords4[m+1].z=coords[k++];
   coords4[m+2].z=coords[k++];
   coords4[m].w=coords[k++];
   coords4[m+1].w=coords[k++];
   coords4[m+2].w=coords[k++];
   m+=3;
  }
  if((mod4=(natoms%4))) //now pad with zeros
  {
   int n,q=0;
   for (n=0;n<12;n++)
    my_coords[n]=0;
   for (n=0;n<mod4*3;n++)
    my_coords[n]=coords[k++]; 
   q=0;
   coords4[m].x=my_coords[q++];
   coords4[m+1].x=my_coords[q++];
   coords4[m+2].x=my_coords[q++];
   coords4[m].y=my_coords[q++];
   coords4[m+1].y=my_coords[q++];
   coords4[m+2].y=my_coords[q++];
   coords4[m].z=my_coords[q++];
   coords4[m+1].z=my_coords[q++];
   coords4[m+2].z=my_coords[q++];
   coords4[m].w=my_coords[q++];
   coords4[m+1].w=my_coords[q++];
   coords4[m+2].w=my_coords[q++];
   m+=3;
  }
 }
 return(m);
}

int define_decoy_sizes_string (char **defines_string, int nthreads, int pdb4_size){
 char buffer[1000];
 int n;
 n=sprintf(buffer,"#define NTHREADS %d\n#define PDB4SIZE %d\n",nthreads,pdb4_size);
 if(*defines_string)free(*defines_string);
 if(!(*defines_string=(char*)malloc(sizeof(char)*(n+1))))exit(FALSE); 
 strcpy(*defines_string,buffer);
 return(n);
}
char *print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
}
