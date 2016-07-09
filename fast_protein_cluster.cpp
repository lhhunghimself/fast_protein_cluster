#ifdef GPU
 #define __CL_ENABLE_EXCEPTIONS
 #include <CL/cl.hpp>
#endif
#ifdef OPENMP
#include <omp.h>

#endif
#include <time.h>
#include "lite.h"
#include "error_handlers.h"
#include <iostream>
#include "libcluster.hpp"
#ifdef GPU
 char gtmscorecl_location[LINE_LENGTH], grmsdcl_location[LINE_LENGTH]; //path to tmscore.cl and rmsd.cl kernels 
#endif
double gtimer1=0;
double gtimer2=0;
cpu_properties gcpu_info;
//main function parses options
//prune and cluster functions that do the work - templated to allow for compact representation of data in either or both steps

template <class T,class T2>
 cluster_models_set<T2>* prune_it(int nthreads, prune_options *prune_options,cluster_options *cluster_options,unsigned int seed);
template <class T>
void cluster_it(int nthreads,cluster_options *cluster_options,cluster_models_set<T> *models,unsigned int seed);

double get_time();

//global cpu_properties
//needed to prevent OPENMP feature which causes slowdown with hyperthreading 
//really should be part of classes calling hierarchical clustering for portablility


int main(int argc, char *argv[]){
 const char *options_text = "Protinfo_cluster -i <input_file with decoy files or names> -o <output_file basename> --<options>\n\noptions (can be all upper case or all lower case)\n\nCompute_type\n--cpu             CPU is used for computations - (default)\n--prune_cpu       CPU is used for pruning computations\n--cluster_cpu     CPU is used for cluster computations\n--gpu             GPU is used for computations - this is the default\n--prune_gpu       GPU is used for pruning computations\n--cluster_gpu     GPU is used for cluster computations\n\nScore_type\n--rmsd            RMSD is used as the similarity metric (default)\n--prune_rmsd      RMSD is used as the similarity metric for pruning\n--cluster_rmsd    RMSD is used as the similarity metric for clustering\n--tmscore         TMSCORE is used as the similarity metric\n--prune_tmscore   TMSCORE is used as the similarity metric for pruning\n--cluster_tmscore TMSCORE is used as the similarity metric for clustering\n\nsimd_type\n--scalar	  normal scalar operations will be used (default)\n--sse2     SSE2 vectorization will be used when possible (Pentium 4 or newer, AMDK8 or newer)\n--sse3     SSE3 vectorization will be used when possible (Pentium 4 Prescott or newer, AMD Athlon 64 or newer)\n--avx		    AVX vectorization will be used when possible  (Sandy Bridge, Ivy Bridge, AMD Bulldozer/Piledriver)\n\ndistance_matrix_storage\n--float				       Use float (single precision 4 bytes) to represent distances (default)\n--prune_float		   Use float (single precision 4 bytes) to represent distances in pruning step\n--cluster_float	 	Use float (single precision 4 bytes) to represent distances in cluster step\n--compact			      Use unsigned char (1 byte) to represent distances\n--prune_compact   Use unsigned char (1 byte) to represent distances in prune step\n--cluster_compact Use unsigned char (1 byte) to represent distances in cluster step\n\nInput/output \n\nwrite matrix in binary form for clustering\n--write_matrix <matrix_file> \n--write_binary_matrix  <matrix_file> \n--cluster_write_matrix  <matrix_file> \n--cluster_write_binary_matrix  <matrix_file>   \n\nwrite distance matrix in binary form for pruning\n--PRUNE_WRITE_MATRIX  <matrix_file> --prune_write_matrix <matrix_file>  \n--PRUNE_WRITE_BINARY_MATRIX  <matrix_file>  --prune_write_binary_matrix <matrix_file> \n\nwrite distance matrix in text form\n--cluster_write_text_matrix  <matrix_file>    \n--prune_write_text_matrix <matrix_file> \n\nwrite distance matrix in compact form \n--cluster_write_compact_matrix  <matrix_file>    \n--prune_write_compact_matrix <matrix_file> \n\nwrite cluster distance matrix in binary form\n--write_matrix <matrix_file> \n--write_binary_matrix  <matrix_file> \n--cluster_write_matrix  <matrix_file> \n--cluster_write_binary_matrix  <matrix_file> \n\nwrite prune distance matrix in binary form \n--prune_write_matrix <matrix_file>  \n--prune_write_binary_matrix <matrix_file>\n\ninput binary coordinates instead pdb_list (application will look for <input_file>.coords and <input_file>.names\n--binary_coords --BINARY_COORDS\n--cluster_binary_coords --CLUSTER_BINARY_COORDS\n--prune_binary_coords --PRUNE_BINARY_COORDS\n\n\nCluster methods\n--nclusters		         number of clusters\n--prune_nclusters     number of clusters to be used in pruning\n\nHierarchical\n--hcomplete           complete linkage hierarchical clustering (max distance between clusters used)\n--haverage            average linkage hierarchical clustering (average distance between clusters used)\n--hsingle             single linkage hierarchical clustering (min distance between clusters used)\n\n--cluster_hcomplete   clustering step: complete linkage hierarchical clustering (max distance between clusters used)\n--cluster_haverage    clustering step: average linkage hierarchical clustering (average distance between clusters used)\n--cluster_hsingle     clustering step: single linkage hierarchical clustering (min distance between clusters used)\n\n--prune_hcomplete     prune using average distance to centers of clusters from complete linkage hierarchical clustering\n--prune_haverage     prune using average distance to centers of clusters from average linkage hierarchical clustering\n--prune_hsingle      prune using average distance to centers of clusters from single linkage hierarchical clustering\n\nkmeans \n--kmeans --cluster_kmeans       use kmeans for clustering step (default) -cannot use for pruning\n--min_cluster_size <n>          minimum cluster size (kmeans only)\n--max_iterations <n>            max iterations for each random starting partition to converge to a final partition\n--total_seeds <n>               number of different starting partitions to try\n--converge_seeds <n>            number of starting partitions without improvement kmeans is said to have converged \n--percentile <P>                used to calculate when the partition score is in the P percentile with confidence pvalue p\n--pvalue <p>                    P is a float value between 0-1 (higher is better value) and p is a positive float value\n--fixed centers <n>             the final n clusters of starting partition are not random but the most distant from the other clusters\n--fine_parallel                 uses finer level of parallelism for kmeans rather than the default seed level parallelism - useful for large sets and small numbers of seeds\n\ndensity\n--density --cluster_density     treat ensemble as single cluster and use min average distance to other structures to find center\n--prune_density                 use single cluster density as criteron for pruning (default) \n--density_only                  calculate density only - does not create similarity matrix so it is very memory efficient O(n)\n--sort_density                  sort the densities in the density_only mode\n\nkcenters\n--kcenters --cluster_kcenters   use kcenters algorithm for clustering\n--prune_kcenters                use distance from centers of kcenters to prune\n\npruning options\n--prune_until_size<size>        only option for all but density based pruning - prunes until the size of the ensemble is <size>\n--prune_stop_at_size            differs from prune_until_size by defining a minimum ensemble size regardless of other criteria\n--prune_outlier_ratio <r>       instead of removing the worst structure at each step use a ratio i.e. a ratio of .9 means that 99% best structures are kept after each step\n--prune_to_density <density>    prune until a threshhold average density <density> is reached\n--prune_stop_density <density>  stop pruning when average density reaches <density>\n--prune_zmode                   used in conjunction with --prune_to_density/--prune_stop_density for stopping criterion\n                                instead of <density> absolute values of density - <density> is number of standard deviations from mean\ngpu_options\n--gpu_id <id>                   specifies gpu whem there is more than one available\n--prune_gpu_id <id>             specifies gpu whem there is more than one available\n--cluster_gpu_id <id>           specifies gpu whem there is more than one available\n\nmisc\n-i <filename>         file with list of PDBs in normal (PDB_LIST) mode\n                      file with list of names when similarity matrix is being read in\n                      file with basename for binary_coords mode - program expects files <file>.coords <file>.names\n-o <output_basename>  default is 'output'\n-S <seed>  define integer seed to be used by random number generators\n-p <path>  define the path to the tmscore.cl and rmsd.cl kernels\n--help     output this text\n";

 cluster_options cluster_options;
 prune_options prune_options;
 int 
  //acceleration variables
  nthreads=1,                    //number of preferred threads -does not override system maxium thread settings
  prune_flag=0,  
  density_only_flag=0,           //simple density calculation does not require similarity matrix if nothing else is being done
  all_atoms_flag=0,              //all-atoms for RMSD calculation - when TMScore is to be calculated in prune or cluster step 
                                 //this is set back to 0 - change int the future to switch back and forth between all-atom and CA - already in reference code  
  srand (time(NULL));            //random number is used to seed the kmeans - might use better algorithm for parallel

 bool sort_density=false;        //in density_only_mode - determines whether the densities should be sorted 
 unsigned int seed=time(NULL);
 double start_time=get_time();  

 
 //tell us what compile options are enabled
 #ifdef SSE2
 fprintf(stderr,"SSE2 enabled\n");
 #endif
 #ifdef SSE3
 fprintf(stderr,"SSE3 enabled\n");
 #endif
 #ifdef AVX 
 fprintf(stderr,"AVX enabled\n");
 #endif
 #ifdef GPU
 fprintf(stderr,"GPU enabled\n");
 #ifdef NVIDIA
  sprintf(gtmscorecl_location,"/usr/local/cuda/include/tmscore.cl");
  sprintf(grmsdcl_location,"/usr/local/cuda/include/rmsd.cl");
 #endif
 #ifdef AMD
  sprintf(gtmscorecl_location,"/opt/AMDAPP/include/tmscore.cl");
  sprintf(grmsdcl_location,"/opt/AMDAPP/include/rmsd.cl");
 #endif 
 #endif
 #ifdef FAST_DIVISION
 fprintf(stderr,"FAST_DIVISION enabled\n");
 #endif
 #ifdef OPEN_MP
  omp_set_dynamic(1);
 #endif 
 //options block
 {
  int i=1;
  while(i<argc)
  {
   if(argv[i][0] == '-' && argv[i][1] != '\0')
   {
    char option=argv[i][1];
    char switch_word[LINE_LENGTH];

    //these have a --
    switch(option)
    {
     case '-':
     {
      strcpy(switch_word,&(argv[i][2]));      
      if(!strcmp(switch_word,"nthreads") || !strcmp(switch_word,"NTHREADS"))
      {
       i++;
       nthreads=atoi(argv[i]); 
      }      
             
      //compute type
      else if(!strcmp(switch_word,"CPU") || !strcmp(switch_word,"cpu"))
      {
       cluster_options.compute=cCPU;
       prune_options.compute=cCPU;
      }      
      else if(!strcmp(switch_word,"PRUNE_CPU") || !strcmp(switch_word,"prune_cpu"))
      {
       prune_flag=1;
       prune_options.compute=cCPU;
      } 
      else if(!strcmp(switch_word,"CLUSTER_CPU") || !strcmp(switch_word,"cluster_cpu") )
      {
       cluster_options.compute=cCPU;
      }      
      else if(!strcmp(switch_word,"GPU") ||!strcmp(switch_word,"gpu") )
      {
       cluster_options.compute=cGPU;
       prune_options.compute=cGPU;
      }
      else if(!strcmp(switch_word,"PRUNE_GPU") || !strcmp(switch_word,"prune_gpu"))
      {
       prune_flag=1;
       prune_options.compute=cGPU;
      }
       
      else if(!strcmp(switch_word,"cluster_gpu") || !strcmp(switch_word,"CLUSTER_GPU"))
      {
       cluster_options.compute=cGPU;
      }
      //score_type
      else if(!strcmp(switch_word,"RMSD") || !strcmp(switch_word,"rmsd"))
      {
       cluster_options.score_type=RMSD;
       prune_options.score_type=RMSD;
      }
      else if(!strcmp(switch_word,"PRUNE_RMSD") || !strcmp(switch_word,"prune_rmsd"))
      {
       prune_options.score_type=RMSD;
      }       
      else if(!strcmp(switch_word,"CLUSTER_RMSD") || !strcmp(switch_word,"cluster_rmsd"))
      {
       cluster_options.score_type=RMSD;
      }           
      else if(!strcmp(switch_word,"TMSCORE") || !strcmp(switch_word,"tmscore"))
      {
       cluster_options.score_type=TMSCORE;
       prune_options.score_type=TMSCORE;
      }
      else if(!strcmp(switch_word,"PRUNE_TMSCORE") || !strcmp(switch_word,"prune_tmscore"))
      {
       prune_options.score_type=TMSCORE;
      }       
      else if(!strcmp(switch_word,"CLUSTER_TMSCORE") || !strcmp(switch_word,"cluster_tmscore"))
      {
       cluster_options.score_type=TMSCORE;
      }      
      //SIMD TYPE
#ifdef SSE2      
      else if(!strcmp(switch_word,"SSE2") || !strcmp(switch_word,"sse2"))
      {
       cluster_options.simd_type=SSE2_;       
       prune_options.simd_type=SSE2_;
      }      
      else if(!strcmp(switch_word,"PRUNE_SSE2") || !strcmp(switch_word,"prune_sse2"))
      {
       prune_flag=1;       
       prune_options.simd_type=SSE2_;
      }      
      else if(!strcmp(switch_word,"CLUSTER_SSE2") || !strcmp(switch_word,"cluster_sse2"))
      {
       cluster_options.simd_type=SSE2_;       
      }
#endif      
#ifdef SSE3      
      else if(!strcmp(switch_word,"SSE3")|| !strcmp(switch_word,"sse3"))
      {
       cluster_options.simd_type=SSE3_;       
       prune_options.simd_type=SSE3_;
      }
      else if(!strcmp(switch_word,"PRUNE_SSE3") || !strcmp(switch_word,"prune_sse3"))
      {     
       prune_options.simd_type=SSE3_;
      }
      else if(!strcmp(switch_word,"CLUSTER_SSE3")|| !strcmp(switch_word,"prune_sse3"))
      {
       cluster_options.simd_type=SSE3_;       
      }
#endif
#ifdef AVX
      else if(!strcmp(switch_word,"AVX") || !strcmp(switch_word,"avx"))
      {
       cluster_options.simd_type=AVX_;       
       prune_options.simd_type=AVX_;
      }  
      else if(!strcmp(switch_word,"PRUNE_AVX") || !strcmp(switch_word,"prune_avx"))
      {
       prune_flag=1;
       prune_options.simd_type=AVX_;
      }
      else if(!strcmp(switch_word,"CLUSTER_AVX") || !strcmp(switch_word,"cluster_avx"))
      {
       cluster_options.simd_type=AVX_;       
      }      
#endif
      //distance matrix type - compact or float - default is float - fairly easy to add double later
      else if(!strcmp(switch_word,"COMPACT") || !strcmp(switch_word,"compact"))
      {
       cluster_options.distance_matrix_type=COMPACT;
       prune_options.distance_matrix_type=COMPACT;
      }
      else if(!strcmp(switch_word,"PRUNE_COMPACT") || !strcmp(switch_word,"prune_compact"))
      {
       prune_flag++;
       prune_options.distance_matrix_type=COMPACT;
      }
      else if(!strcmp(switch_word,"CLUSTER_COMPACT") || !strcmp(switch_word,"cluster_compact"))
      {
       cluster_options.distance_matrix_type=COMPACT;
      }
      else if(!strcmp(switch_word,"FLOAT") || !strcmp(switch_word,"float"))      
      {
       cluster_options.distance_matrix_type=FLOAT;
       prune_options.distance_matrix_type=FLOAT;
      }
      else if(!strcmp(switch_word,"PRUNE_FLOAT") || !strcmp(switch_word,"prune_float"))
      {
       prune_flag++;
       prune_options.distance_matrix_type=FLOAT;
      }      
      else if(!strcmp(switch_word,"PRUNE_LOG") || !strcmp(switch_word,"prune_log"))
      {
       prune_options.keep_log=1;
      }     
      else if(!strcmp(switch_word,"CLUSTER_FLOAT")|| !strcmp(switch_word,"cluster_float") )
      {
       cluster_options.distance_matrix_type=FLOAT;
      }
      //input/output_matrix_type
      else if (!strcmp(switch_word,"read_binary_matrix") || 
               !strcmp(switch_word,"READ_BINARY_MATRIX") || 
               !strcmp(switch_word,"READ_MATRIX") ||               
               !strcmp(switch_word,"read_matrix")
              )
      {
       i++;
       cluster_options.read_matrix_type=BINARY;
       strcpy(cluster_options.read_matrix_filename,argv[i]);      
      }      
      else if (!strcmp(switch_word,"read_text_matrix") ||!strcmp(switch_word,"cluster_read_text_matrix") || 
               !strcmp(switch_word,"READ_TEXT_MATRIX") ||!strcmp(switch_word,"CLUSTER_READ_TEXT_MATRIX")  )
      {
       i++;
       cluster_options.read_matrix_type=TEXT;
       strcpy(cluster_options.read_matrix_filename,argv[i]);
      }        
      else if (!strcmp(switch_word,"prune_read_binary_matrix") || !strcmp(switch_word,"PRUNE_READ_BINARY_MATRIX")  )
      {
       i++;
       prune_options.read_matrix_type=BINARY;
       strcpy(prune_options.read_matrix_filename,argv[i]);
      }
      else if (!strcmp(switch_word,"prune_read_text_matrix") || !strcmp(switch_word,"PRUNE_READ_TEXT_MATRIX")  )
      {
       i++;
       prune_options.read_matrix_type=TEXT;
       strcpy(prune_options.read_matrix_filename,argv[i]);
      }
      else if (!strcmp(switch_word,"cluster_read_binary_matrix") ||
               !strcmp(switch_word,"CLUSTER_READ_BINARY_MATRIX")  )
      {
       i++;
       cluster_options.read_matrix_type=BINARY;
       strcpy(cluster_options.read_matrix_filename,argv[i]);
      }      
       else if (!strcmp(switch_word,"read_compact_matrix") ||!strcmp(switch_word,"cluster_read_compact_matrix") || 
               !strcmp(switch_word,"READ_COMPACT_MATRIX") ||!strcmp(switch_word,"CLUSTER_READ_COMPACT_MATRIX")  )
      {
       i++;
       cluster_options.read_matrix_type=CHAR;
       strcpy(cluster_options.read_matrix_filename,argv[i]);
      }       
      else if (!strcmp(switch_word,"prune_read_compact_matrix") || !strcmp(switch_word,"PRUNE_READ_COMPACT_MATRIX")  )
      {
       i++;
       prune_options.read_matrix_type=CHAR;
       strcpy(prune_options.read_matrix_filename,argv[i]);
      }
      //write flags
      else if (!strcmp(switch_word,"write_binary_matrix") ||!strcmp(switch_word,"cluster_write_binary_matrix") ||
               !strcmp(switch_word,"WRITE_BINARY_MATRIX") ||!strcmp(switch_word,"CLUSTER_WRITE_BINARY_MATRIX") ||
               !strcmp(switch_word,"WRITE_MATRIX") ||!strcmp(switch_word,"CLUSTER_WRITE_MATRIX")
                )
      {
       i++;
       cluster_options.write_matrix_type=BINARY;
       strcpy(cluster_options.write_matrix_filename,argv[i]);
      }      
      else if (!strcmp(switch_word,"write_text_matrix") ||!strcmp(switch_word,"cluster_write_text_matrix") ||
               !strcmp(switch_word,"WRITE_TEXT_MATRIX") ||!strcmp(switch_word,"CLUSTER_WRITE_TEXT_MATRIX")  )
      {
       i++;
       cluster_options.write_matrix_type=TEXT;
       strcpy(cluster_options.write_matrix_filename,argv[i]);
      }        
      else if (!strcmp(switch_word,"prune_write_binary_matrix") || !strcmp(switch_word,"PRUNE_WRITE_BINARY_MATRIX") ||
               !strcmp(switch_word,"prune_write_matrix") || !strcmp(switch_word,"PRUNE_WRITE_MATRIX")  )
      {
       i++;
       prune_options.write_matrix_type=BINARY;
       strcpy(prune_options.write_matrix_filename,argv[i]);
      }
      else if (!strcmp(switch_word,"prune_write_text_matrix") || !strcmp(switch_word,"PRUNE_WRITE_TEXT_MATRIX"))
      {
       i++;
       prune_options.write_matrix_type=TEXT;
       strcpy(prune_options.write_matrix_filename,argv[i]);
      }
 
      else if (!strcmp(switch_word,"write_compact_matrix") ||!strcmp(switch_word,"cluster_write_compact_matrix") || 
               !strcmp(switch_word,"WRITE_COMPACT_MATRIX") ||!strcmp(switch_word,"CLUSTER_WRITE_COMPACT_MATRIX")  )
      {
       i++;
       cluster_options.write_matrix_type=CHAR;
       strcpy(cluster_options.write_matrix_filename,argv[i]);
      }       
      else if (!strcmp(switch_word,"prune_write_compact_matrix") || !strcmp(switch_word,"PRUNE_WRITE_COMPACT_MATRIX")  )
      {
       i++;
       prune_options.write_matrix_type=CHAR;
       strcpy(prune_options.write_matrix_filename,argv[i]);
      }
      //cluster flags
      else if (!strcmp(switch_word,"hcomplete") ||!strcmp(switch_word,"HCOMPLETE") ||
               !strcmp(switch_word,"cluster_hcomplete") ||!strcmp(switch_word,"CLUSTER_HCOMPLETE") )
      {
       cluster_options.method=HCOMPLETE;
      }
      else if (!strcmp(switch_word,"hsingle") ||!strcmp(switch_word,"HSINGLE") ||
               !strcmp(switch_word,"cluster_hsingle") ||!strcmp(switch_word,"cluster_HSINGLE") )
      {
       cluster_options.method=HSINGLE;
      }      
      
      else if (!strcmp(switch_word,"haverage") ||!strcmp(switch_word,"HAVERAGE") ||
               !strcmp(switch_word,"cluster_haverage") ||!strcmp(switch_word,"cluster_HAVERAGE") )
      {
       cluster_options.method=HAVERAGE;
      }      
      
      else if (!strcmp(switch_word,"prune_hcomplete") ||!strcmp(switch_word,"PRUNE_HCOMPLETE"))
      {
       prune_options.method=HCOMPLETE;
      }
      else if (!strcmp(switch_word,"prune_hsingle") ||!strcmp(switch_word,"PRUNE_HSINGLE"))
      {
       prune_options.method=HSINGLE;
      }      
      else if (!strcmp(switch_word,"prune_haverage") ||!strcmp(switch_word,"PRUNE_HAVERAGE")) 
      {
       prune_options.method=HAVERAGE;
      }
      else if (!strcmp(switch_word,"kcenters") ||!strcmp(switch_word,"KCENTERS") ||
               !strcmp(switch_word,"cluster_kcenters") ||!strcmp(switch_word,"CLUSTER_KCENTERS") )
      {
       cluster_options.method=KCENTERS;
      }
      else if (!strcmp(switch_word,"prune_kcenters") ||!strcmp(switch_word,"PRUNE_KCENTERS")) 
      {
       prune_options.method=KCENTERS;
      }
      else if (!strcmp(switch_word,"prune_density") ||!strcmp(switch_word,"PRUNE_DENSITY")) 
      {
       prune_options.method=DENSITY;
      }               
      else if (!strcmp(switch_word,"kmeans") ||!strcmp(switch_word,"KMEANS") ||
               !strcmp(switch_word,"cluster_kmeans") ||!strcmp(switch_word,"CLUSTER_KMEANS") ) 
      {
       cluster_options.method=KMEANS;
      } 
      else if (!strcmp(switch_word,"DENSITY") ||!strcmp(switch_word,"density") ||
               !strcmp(switch_word,"cluster_density") ||!strcmp(switch_word,"CLUSTER_DENSITY") ) 
      {
       cluster_options.method=DENSITY;
      }
      //cluster option flags
      else if (!strcmp(switch_word,"nclusters")||!strcmp(switch_word,"NCLUSTERS"))
      {
       i++; 
       cluster_options.nclusters=atoi(argv[i]); 
       prune_options.nclusters=atoi(argv[i]);
      }      
      else if (!strcmp(switch_word,"cluster_nclusters") ||!strcmp(switch_word,"CLUSTER_NCLUSTERS") ) 
      {
       i++; 
       cluster_options.nclusters=atoi(argv[i]);
      }      
      else if (!strcmp(switch_word,"prune_nclusters") ||!strcmp(switch_word,"PRUNE_NCLUSTERS") ) 
      {
       i++; 
       prune_options.nclusters=atoi(argv[i]);
      }      
      //kmeans specific  
      else if (!strcmp(switch_word,"min_cluster_size") || !strcmp(switch_word,"MIN_CLUSTER_SIZE"))
      {
       i++;
       cluster_options.min_cluster_size=atoi(argv[i]);
      }

      else if (!strcmp(switch_word,"max_iterations") || !strcmp(switch_word,"MAX_ITERATIONS"))
      {
       i++;
       cluster_options.maximum_iterations=atoi(argv[i]);
      }
      else if (!strcmp(switch_word,"total_seeds") || !strcmp(switch_word,"TOTAL_SEEDS"))
      {
       i++;
       cluster_options.total_iterations=atoi(argv[i]);
      }
      
      else if (!strcmp(switch_word,"converge_seeds") || !strcmp(switch_word,"CONVERGE_SEEDS") )
      {//number of repetitions without improving score that score converges
       i++;
       cluster_options.nsolutions_after_best_score=atoi(argv[i]);
      }
      else if (!strcmp(switch_word,"percentile") || !strcmp(switch_word,"PERCENTILE"))
      {//number of repetitions without improving score that score converges
       i++;
       cluster_options.percentile=atof(argv[i]);
      }
      else if (!strcmp(switch_word,"pvalue") || !strcmp(switch_word,"PVALUE"))
      {
       i++;
       cluster_options.pvalue=atof(argv[i]);
      }
      else if (!strcmp(switch_word,"fixed_centers")|| !strcmp(switch_word,"FIXED_CENTERS"))
      {
       i++;
       cluster_options.nfixed_centers=atoi(argv[i]);
      }      
      
      else if (!strcmp(switch_word,"fine_parallel")|| !strcmp(switch_word,"FINE_PARALLEL"))
      {
       cluster_options.fine_parallel=1;
      }
      //prune density specific options
      else if(!strcmp(switch_word,"prune_zmode") || !strcmp(switch_word,"PRUNE_ZMODE"))
      {
       prune_flag=1; 
       prune_options.prune_zmode=1;
      }
      else if(!strcmp(switch_word,"prune_to_density") || !strcmp(switch_word,"PRUNE_TO_DENSITY"))
      {
       i++;
       prune_options.prune_to=atof(argv[i]);
       prune_flag=1;
      }
      else if (!strcmp(switch_word,"PRUNE_STOP_DENSITY") || !strcmp(switch_word,"prune_stop_density"))
      {
       i++;
       prune_options.prune_stop=atof(argv[i]);
       prune_flag=1; 
      }
      else if (!strcmp(switch_word,"prune_outlier_ratio") || !strcmp(switch_word,"RRUNE_OUTLIER_RATIO"))
      {
       i++;
       prune_options.prune_outlier_ratio=atof(argv[i]);
       prune_flag=1;
      }
      else if (!strcmp(switch_word,"prune_stop_at_size") || !strcmp(switch_word,"PRUNE_STOP_AT_SIZE"))
      {
       i++;
       prune_options.prune_min_size=atoi(argv[i]);
       prune_flag=1;
      }
      else if (!strcmp(switch_word,"prune_until_size") || !strcmp(switch_word,"PRUNE_UNTIL_SIZE") )
      {
       i++;
       prune_options.prune_max_size=atoi(argv[i]);
       prune_flag=1;
      }
      //density only
      else if (!strcmp(switch_word,"density_only") || !strcmp(switch_word,"DENSITY_ONLY") )
      {
       density_only_flag=1; //calculate and output the density only
      }       
      else if (!strcmp(switch_word,"sort_density") || !strcmp(switch_word,"SORT_DENSITY") )
      {
       sort_density=true; //this only applies to density_only mode
      }
      //coords mode
      else if(!strcmp(switch_word,"binary_coords") || !strcmp(switch_word,"BINARY_COORDS") )
      {
       prune_options.input_type=BINARY_COORDS;
       cluster_options.input_type=BINARY_COORDS;
      }
      else if(!strcmp(switch_word,"cluster_binary_coords") || !strcmp(switch_word,"CLUSTER_BINARY_COORDS") )
      {
       cluster_options.input_type=BINARY_COORDS;
      }
      else if(!strcmp(switch_word,"prune_binary_coords") || !strcmp(switch_word,"PRUNE_BINARY_COORDS") )
      {
       prune_options.input_type=BINARY_COORDS;
      }
      else if(!strcmp(switch_word,"all_atoms") || !strcmp(switch_word,"ALL_ATOMS") )
      {
        all_atoms_flag=1;      
       prune_options.all_atoms=1;
       cluster_options.all_atoms=1;
      }               
      else if (!strcmp(switch_word,"help") || !strcmp(switch_word,"HELP") )
      {
       fprintf(stderr,"****OPTIONS*****\n%s",options_text);
       exit(FALSE);
      }
#ifdef GPU
      //gpu_flags
      else if (!strcmp(switch_word,"gpu_id") || !strcmp(switch_word,"GPU_ID") )
      {
       i++;
       prune_options.gpu_id=atoi(argv[i]);
       cluster_options.gpu_id=atoi(argv[i]);
      }           
      else if (!strcmp(switch_word,"prune_gpu_id") || !strcmp(switch_word,"PRUNE_GPU_ID") )
      {
       i++;
       prune_options.gpu_id=atoi(argv[i]);
      }
      else if (!strcmp(switch_word,"cluster_gpu_id") || !strcmp(switch_word,"CLUSTER_GPU_ID") )
      {
       i++;
       cluster_options.gpu_id=atoi(argv[i]);
      }      
#endif      
            
      else
      {
      fprintf(stderr,"unrecognized option %s\n",argv[i]);
      fprintf(stderr,"%s",options_text);
       exit(FALSE);
      }
     }
     break;
     case 'i' :
     {
      i++;
      strcpy(prune_options.input_filename, argv[i]);
      strcpy(cluster_options.input_filename, argv[i]);
     }
     break;
     case 'o' :
     {
      i++;
      strcpy(prune_options.output_filename, argv[i]);
      strcpy(cluster_options.output_filename, argv[i]);
     }
     break;
     case 'S' :
     {
      i++;
      seed = (unsigned int) atoi(argv[i]);
     }
     break;
#ifdef GPU     
     case 'p' :
     {
      //path to tmscore.cl rmsd.cl
      i++;
      sprintf(gtmscorecl_location,"%s/tmscore.cl",argv[i]);
      sprintf(grmsdcl_location,"%s/tmscore.cl",argv[i]);
     }
#endif         
     default :
     {
      fprintf(stderr,"unrecognized option %s\n",argv[i]);
      fprintf(stderr,"%s",options_text);
      exit(FALSE);
     }
    }
    i++;
   }
   else
   {
    fprintf(stderr,"unrecognized option %s\n",argv[i]);
    fprintf(stderr,"%s",options_text);
    exit(FALSE);
   }
  }
 }
 #ifdef OPENMP
 {
  int max_threads=omp_get_max_threads();
  nthreads=(max_threads<nthreads)? max_threads : nthreads;
  fprintf(stderr,"Using %d threads\n",nthreads);
 } 
 #endif

 
 if(all_atoms_flag){
  if(cluster_options.score_type==TMSCORE || prune_options.score_type==TMSCORE){
   fprintf(stderr,"all_atoms option presently only supported for RMSD - will continue with using only CA coords\n");
   all_atoms_flag=0;
   prune_options.all_atoms=0;
   cluster_options.all_atoms=0; 
  }   
 }  
 if (density_only_flag){

  //do not create a distance matrix - calculate the density vector directly and output
  //can generate the density de novo from pdb files or from a pre-exisiting matrix file
  //if a matrix is read in - the density vector is still calculated directly without memory being allocated for a temp distance matrix
  //uses the other options passed to cluster options wrt to file names, metric etc... 
  FILE *fp;
  cluster_models_set<float> *models;
  if(cluster_options.read_matrix_type != NO_MATRIX){
    models= new cluster_models_set<float>("READ_MATRIX",nthreads,cluster_options.score_type,cluster_options.all_atoms,cluster_options.input_filename,cluster_options.read_matrix_filename,
    cluster_options.read_matrix_type,cluster_options.distance_matrix_type,(int*)0,cluster_options.subset_filename,1);
  }  
  else if(cluster_options.input_type == BINARY_COORDS){
   models=new cluster_models_set<float>("READ_BINARY_COORDS",nthreads,cluster_options.score_type,cluster_options.all_atoms,cluster_options.compute,cluster_options.distance_matrix_type,
                 cluster_options.input_filename,(int*) 0,cluster_options.subset_filename,cluster_options.simd_type,1,cluster_options.gpu_id);
  } 
  else{ models=new cluster_models_set<float>("READ_PDB_LIST",nthreads,cluster_options.score_type,cluster_options.all_atoms,cluster_options.compute,cluster_options.distance_matrix_type,
                 cluster_options.input_filename,(int*) 0,cluster_options.subset_filename,cluster_options.simd_type,1,cluster_options.gpu_id);
  }    
  char temp_filename[FILENAME_LENGTH];;
  sprintf(temp_filename,"%s.density",cluster_options.output_filename);
  open_file(&fp, temp_filename, "w", 0);
  models->print_density(fp,sort_density);
  close_file(&fp, temp_filename,0);
  if(models) delete models;
  return(TRUE);    
 }
 
 void *vprune_models=0; 
 if(prune_flag){
  if(prune_options.distance_matrix_type == COMPACT){
   if(cluster_options.distance_matrix_type == COMPACT){
    vprune_models=prune_it<unsigned char,unsigned char>(nthreads,&prune_options,&cluster_options,seed);
   }
   else{
    vprune_models=prune_it<unsigned char,float>(nthreads,&prune_options,&cluster_options,seed);
   }
  } 
  else if(prune_options.distance_matrix_type == FLOAT){
   if(cluster_options.distance_matrix_type == COMPACT){
    vprune_models=prune_it<float,unsigned char>(nthreads,&prune_options,&cluster_options,seed);
   }
   else{
    vprune_models=prune_it<float,float>(nthreads,&prune_options,&cluster_options,seed);
   }
  }
  if (prune_options.write_matrix_type != NO_MATRIX){
   FILE *matrix_fp;
   char matrix_filename[FILENAME_LENGTH];
   sprintf(matrix_filename,"%s.post_prune",prune_options.write_matrix_filename);
   open_file(&matrix_fp, matrix_filename, "w", "main");
   char temp_filename[FILENAME_LENGTH];;
   sprintf(temp_filename,"%s.post_prune.names",prune_options.output_filename);
   
   if(prune_options.distance_matrix_type == COMPACT){
    cluster_models_set<unsigned char>* models=(cluster_models_set<unsigned char>*)vprune_models;
     models->write_names_to_file(temp_filename,0,0);
    if (prune_options.write_matrix_type == BINARY){
     models->dmatrix->write_matrix_to_binary_file(matrix_fp);
    } 
    else if(prune_options.write_matrix_type == TEXT){   
     models->dmatrix->write_matrix_to_text_file(matrix_fp);
    }
    else if(prune_options.write_matrix_type == CHAR ){ 
     models->dmatrix->write_matrix_to_compact_file(matrix_fp);
    }
    close_file(&matrix_fp, prune_options.write_matrix_filename,"main"); 
   }
   else{
    cluster_models_set<float>* models=(cluster_models_set<float>*)vprune_models;
    models->write_names_to_file(temp_filename,0,0);
    if (prune_options.write_matrix_type == BINARY){
     models->dmatrix->write_matrix_to_binary_file(matrix_fp);
    } 
    else if(prune_options.write_matrix_type == TEXT){   
     models->dmatrix->write_matrix_to_text_file(matrix_fp);
    }
    else if(prune_options.write_matrix_type == CHAR ){
     models->dmatrix->write_matrix_to_compact_file(matrix_fp);
    }    
   }
  }
 }
 if(cluster_options.distance_matrix_type == COMPACT){
  cluster_models_set<unsigned char> *models=(cluster_models_set<unsigned char>*)vprune_models;
  cluster_it<unsigned char>(nthreads,&cluster_options,models,seed);   
 }
 else if(cluster_options.distance_matrix_type == FLOAT){
  cluster_models_set<float> *models=(cluster_models_set<float>*)vprune_models;     
  cluster_it<float>(nthreads,&cluster_options,models,seed);
 }
}

template <class T,class T2> //types are those of the prune and cluster classes
 cluster_models_set<T2>* prune_it(int nthreads, prune_options *prune_options,cluster_options *cluster_options,unsigned int seed){
 fprintf(stderr,"pruning the set\n");
 void *vmodels=0;
 mapped_cluster_set<T>* prune_cluster=0;
 //if the distance_matrix type is the same in pruning and clustering models is returned otherwise cmodels is returned 
 //there is an obscure way of doing this

 cluster_models_set<T>* models=0;
 cluster_models_set<T2>* out_models=0;
 
 if(prune_options->read_matrix_type != NO_MATRIX){
  int *inverse_map=0;
  models=new cluster_models_set<T>("READ_MATRIX",nthreads,prune_options->score_type,prune_options->all_atoms,prune_options->input_filename,prune_options->read_matrix_filename,
  prune_options->read_matrix_type,prune_options->distance_matrix_type,(int*) 0,prune_options->subset_filename,0);
 }
 else if(prune_options->input_type==BINARY_COORDS){ models=new cluster_models_set<T>("READ_BINARY_COORDS",nthreads,prune_options->score_type,prune_options->all_atoms,prune_options->compute,prune_options->distance_matrix_type,
                  prune_options->input_filename,(int*)0,prune_options->subset_filename,prune_options->simd_type,0,prune_options->gpu_id);
 }
 else{ models=new cluster_models_set<T>("READ_PDB_LIST",nthreads,prune_options->score_type,prune_options->all_atoms,prune_options->compute,prune_options->distance_matrix_type,
                  prune_options->input_filename,(int*)0,prune_options->subset_filename,prune_options->simd_type,0,prune_options->gpu_id);
 }
 //write out the pre_prune matrices
  if (prune_options->write_matrix_type != NO_MATRIX){
   FILE *matrix_fp;
   char matrix_filename[FILENAME_LENGTH];
   sprintf(matrix_filename,"%s.pre_prune",prune_options->write_matrix_filename);
   open_file(&matrix_fp,matrix_filename, "w", "main");
   char temp_filename[FILENAME_LENGTH];;
   sprintf(temp_filename,"%s.pre_prune.names",prune_options->output_filename);
   
   if(prune_options->distance_matrix_type == COMPACT){
     models->write_names_to_file(temp_filename,0,0);
    if (prune_options->write_matrix_type == BINARY){
     models->dmatrix->write_matrix_to_binary_file(matrix_fp);
    } 
    else if(prune_options->write_matrix_type == TEXT){   
     models->dmatrix->write_matrix_to_text_file(matrix_fp);
    }
    else if(prune_options->write_matrix_type == CHAR ){ 
     models->dmatrix->write_matrix_to_compact_file(matrix_fp);
    }
    close_file(&matrix_fp, prune_options->write_matrix_filename,"main"); 
   }
   else{
    models->write_names_to_file(temp_filename,0,0);
    if (prune_options->write_matrix_type == BINARY){
     models->dmatrix->write_matrix_to_binary_file(matrix_fp);
    } 
    else if(prune_options->write_matrix_type == TEXT){   
     models->dmatrix->write_matrix_to_text_file(matrix_fp);
    }
    else if(prune_options->write_matrix_type == CHAR ){ 
     models->dmatrix->write_matrix_to_compact_file(matrix_fp);
    }    
   }
  }
 if(prune_options->method ==DENSITY){//no partition is needed if density is used
  FILE *log_fp=0;   
  char log_filename[FILENAME_LENGTH];
  if(prune_options->keep_log){
   sprintf(log_filename,"%s.pruned",prune_options->output_filename);
   open_file(&log_fp, log_filename, "w", "main");
  } 
  prune_cluster = new mapped_cluster_set<T>(models,1,1,seed);
  prune_cluster->models->idensity_filter(prune_options->prune_zmode,prune_options->prune_to,prune_options->prune_stop,prune_options->prune_min_size,prune_options->prune_max_size,prune_options->prune_outlier_ratio,log_fp);
  if(prune_options->keep_log){ 
   close_file(&log_fp, log_filename,"main");
   sprintf(log_filename,"%s.density",prune_options->output_filename);
   open_file(&log_fp, log_filename, "w", "main");
   prune_cluster->models->print_density(log_fp);
   close_file(&log_fp, log_filename,"main");
  } 
 }
 else{
  if(!prune_options->prune_max_size){
   fprintf(stderr,"For pruning by any method other than density, the prune_until_size option is the only recognized and must be set\n");
   exit(FALSE);
  }
  fprintf(stderr,"pruning from %d structures to %d structures\n",models->nmodels,prune_options->prune_max_size);
  double start_cluster=get_time();
  //hierarchical clustering
  if(prune_options->method == HSINGLE || prune_options->method == HCOMPLETE ||prune_options->method == HAVERAGE ){
   int *history=0;
   int initial_nmodels=models->nmodels;
   prune_cluster= new mapped_cluster_set<T>(models,models->nmodels,1,seed,0);
   history=new  int [initial_nmodels*2];
   if(prune_options->method == HCOMPLETE){
    fprintf(stderr,"using complete linkage/max-distance between elements as cluster distance\n");
    prune_cluster->best_partition->reduce_by_agglomeration_complete_linkage(prune_options->nclusters,prune_cluster->models,history,0,1,nthreads);
   }
   else if (prune_options->method == HSINGLE){
    fprintf(stderr,"using single linkage/min-distance between elements as cluster distance\n");
    prune_cluster->best_partition->reduce_by_agglomeration_single_linkage(prune_options->nclusters,prune_cluster->models,history,0,1,nthreads);
   }
   else if (prune_options->method == HAVERAGE){
    fprintf(stderr,"using average distance between elements as cluster distance\n");
    prune_cluster->best_partition->reduce_by_agglomeration_average(prune_options->nclusters,prune_cluster->models,history,0,1,nthreads);
   }
   if(history)
   {//print out history of groups joined
    FILE *fp;   
    char temp_filename[FILENAME_LENGTH];;
    sprintf(temp_filename,"%s.prune.agglomeration.history",prune_options->output_filename);
    fprintf(stderr,"writing out history file\n");
    open_file(&fp,temp_filename, "w", 0);
    for(int n=0;n<initial_nmodels;n++)
     fprintf(fp,"%d %d %d\n",n,history[2*n],history[2*n+1]);
    close_file(&fp,temp_filename, 0);
    if(history) delete [] history;
   }
  }
  else if(prune_options->method == KCENTERS){
   prune_cluster= new mapped_cluster_set<T>(models,prune_options->nclusters,1,seed);
   prune_cluster->best_partition->kcenters(100,100,prune_cluster->models, &cluster_partition<T>::assign_to_lowest_average_distance);
  }
  prune_cluster->print_centers(stderr,prune_cluster->best_partition);
  prune_cluster->filter_by_average_centroid_distance(prune_options->prune_max_size); 
 }
 //write the pruned state out to the real models in the class
 //save them as out_models

 //is the pruned matrix a simple subset of the original?
 if((prune_options->score_type == cluster_options->score_type && 
     prune_options->distance_matrix_type == cluster_options->distance_matrix_type)){
      
  fprintf(stderr,"simple subset\n");     
  prune_cluster->models->base_set->write_state_to_base(prune_cluster->models->map,prune_cluster->models->nmodels);
  if(prune_cluster)delete prune_cluster->best_partition;
  vmodels=prune_cluster->models->base_set;
  return((cluster_models_set<T2>*) vmodels);
 }
 else{
  //prepare new distance matrix 
  //are the distance matrix class type the same
  if(prune_options->distance_matrix_type == cluster_options->distance_matrix_type){
   fprintf(stderr,"same  type class %d\n",prune_options->distance_matrix_type);
   if(cluster_options->read_matrix_type != NO_MATRIX) //read new matrix
    prune_cluster->models->base_set->write_state_to_base(cluster_options->read_matrix_filename,cluster_options->score_type,cluster_options->read_matrix_type,prune_cluster->models->map,prune_cluster->models->nmodels);
   else 
    prune_cluster->models->base_set->write_state_to_base(cluster_options->compute,cluster_options->score_type,cluster_options->simd_type,prune_cluster->models->map,prune_cluster->models->nmodels,nthreads,prune_options->gpu_id);
   if(prune_cluster->best_partition)delete prune_cluster->best_partition;  
   vmodels=prune_cluster->models->base_set;
   return((cluster_models_set<T2>*) vmodels); 
  }
  else{//both are class type and metric type are different
   fprintf(stderr,"different type class\n");
   cluster_models_set<T2>* out_models=new cluster_models_set<T2>();
   if(cluster_options->read_matrix_type != NO_MATRIX) //read new matrix
    prune_cluster->models->base_set->write_state_to_different_type_base(out_models,cluster_options->read_matrix_filename,cluster_options->score_type,cluster_options->read_matrix_type,prune_cluster->models->map,prune_cluster->models->nmodels);
   else prune_cluster->models->base_set->write_state_to_different_type_base(out_models,cluster_options->compute,cluster_options->score_type,cluster_options->simd_type,prune_cluster->models->map,prune_cluster->models->nmodels,nthreads,cluster_options->gpu_id);   
   if(prune_cluster)delete prune_cluster;  
   return(out_models);
  } 
 }
  
} 
template <class T>
void cluster_it(int nthreads,cluster_options *cluster_options,cluster_models_set<T> *models,unsigned int seed){ 
 //check whether a distance matrix and models needs to be generated
 if(!models){
  if(cluster_options->read_matrix_type != NO_MATRIX){
    models=new cluster_models_set<T>("READ_MATRIX",nthreads,cluster_options->score_type,cluster_options->all_atoms,cluster_options->input_filename,cluster_options->read_matrix_filename,
   cluster_options->read_matrix_type,cluster_options->distance_matrix_type,(int*)0,cluster_options->subset_filename,0);
  }    
  else if (cluster_options->input_type == BINARY_COORDS){models=new cluster_models_set<T>("READ_BINARY_COORDS",nthreads,cluster_options->score_type,cluster_options->all_atoms,cluster_options->compute,cluster_options->distance_matrix_type,
                 cluster_options->input_filename,(int*) 0,cluster_options->subset_filename,cluster_options->simd_type,0,cluster_options->gpu_id);
  }
  else{ models=new cluster_models_set<T>("READ_PDB_LIST",nthreads,cluster_options->score_type,cluster_options->all_atoms,cluster_options->compute,cluster_options->distance_matrix_type,
                 cluster_options->input_filename,(int*) 0,cluster_options->subset_filename,cluster_options->simd_type,0,cluster_options->gpu_id);
  }   
 }

 //do we write the matrix
 if (cluster_options->write_matrix_type != NO_MATRIX){
  FILE *matrix_fp;
  open_file(&matrix_fp, cluster_options->write_matrix_filename, "w", "cluster_it");
  char temp_filename[FILENAME_LENGTH];;
   sprintf(temp_filename,"%s.names",cluster_options->output_filename);
   models->write_names_to_file(temp_filename,0,0);
  if (cluster_options->write_matrix_type == BINARY){
   models->dmatrix->write_matrix_to_binary_file(matrix_fp);
  } 
  else if(cluster_options->write_matrix_type == TEXT){   
   models->dmatrix->write_matrix_to_text_file(matrix_fp);
  }
  else if(cluster_options->write_matrix_type == CHAR ){ 
   models->dmatrix->write_matrix_to_compact_file(matrix_fp);
  }
  else{
   fprintf(stderr,"unrecognized write matrix type - %d binary format written\n",cluster_options->write_matrix_type);
   models->dmatrix->write_matrix_to_binary_file(matrix_fp); 
  }
  close_file(&matrix_fp,cluster_options->write_matrix_filename,"cluster_it");
 } 
 //which cluster method is to be used
 if(cluster_options->method == DENSITY || cluster_options->nclusters ==1){
  mapped_cluster_models_set<T> cluster = mapped_cluster_models_set<T>(models);
  FILE *log_fp;
  char log_filename[FILENAME_LENGTH];
  sprintf(log_filename,"%s.density",cluster_options->output_filename);
  open_file(&log_fp, log_filename, "w", "main");
  fprintf(stderr,"saving density to %s\n",log_filename); 
  cluster.print_density(log_fp);
  close_file(&log_fp,log_filename,"main"); 
  if(models)delete(models);
  return;
 }
 else if(cluster_options->method == KCENTERS || (cluster_options->method == KMEANS && cluster_options->nfixed_centers >= cluster_options->nclusters)){
  double start_cluster=get_time();
  mapped_cluster_set<T> kc_cluster=mapped_cluster_set<T>(models,cluster_options->nclusters,1,seed);
  kc_cluster.best_partition->kcenters(100,100,kc_cluster.models, &cluster_partition<T>::assign_to_lowest_average_distance);
  fprintf(stderr, "%8.3f seconds elapsed to cluster\n",get_time()-start_cluster);
  kc_cluster.print_centers(stderr,kc_cluster.best_partition);
  FILE *output_fp;
  char temp_filename[FILENAME_LENGTH];
  sprintf(temp_filename,"%s.cluster.stats",cluster_options->output_filename);
  open_file(&output_fp, temp_filename, "w", "main"); 
  kc_cluster.print_centers(output_fp,kc_cluster.best_partition);
  kc_cluster.print_cluster_stats(output_fp,kc_cluster.best_partition);
  close_file(&output_fp,temp_filename,"main");
  sprintf(temp_filename,"%s.clusters",cluster_options->output_filename);
  open_file(&output_fp, temp_filename, "w", "main"); 
  kc_cluster.print_density(output_fp,kc_cluster.best_partition);
  close_file(&output_fp,temp_filename,"main");
  
  if(models)delete(models);
  return;      
 } 
 //hierarchical clustering
 else if(cluster_options->method == HCOMPLETE || cluster_options->method == HAVERAGE || cluster_options->method == HSINGLE ){
  fprintf(stderr,"hierarchical clustering\n");
  double start_cluster=get_time();
  int *history=new  int[models->nmodels*2];
  for(int i=0;i<models->nmodels*2;i++)
   history[i]=-1;
  mapped_cluster_set<T> hcluster=mapped_cluster_set<T>((cluster_models_set<T>*) models,models->nmodels,1,seed,0);
  if(cluster_options->method == HCOMPLETE){
   fprintf(stderr,"using complete linkage/max-distance between elements as cluster distance\n");
   hcluster.best_partition->reduce_by_agglomeration_complete_linkage(cluster_options->nclusters,hcluster.models,history,0,1,nthreads);
  }
  else if (cluster_options->method == HSINGLE){
   fprintf(stderr,"using single linkage/min-distance between elements as cluster distance\n");
   hcluster.best_partition->reduce_by_agglomeration_single_linkage(cluster_options->nclusters,hcluster.models,history,0,1,nthreads);
  }
  else{
   fprintf(stderr,"using average distance between elements as cluster distance\n");
   hcluster.best_partition->reduce_by_agglomeration_average(cluster_options->nclusters,hcluster.models,history,0,1,nthreads);
  }
  
  if(history){//print out history of groups joined
   FILE *fp;   
   char temp_filename[FILENAME_LENGTH];;
   sprintf(temp_filename,"%s.agglomeration.history",cluster_options->output_filename);
   fprintf(stderr,"writing out history file\n");
   open_file(&fp,temp_filename, "w", 0);
   for(int n=0;n<models->nmodels;n++)
    fprintf(fp,"%d %d %d\n",n,history[2*n],history[2*n+1]);
   close_file(&fp,temp_filename, 0);
   if(history) delete [] history;
  }
  hcluster.print_centers(stderr,hcluster.best_partition);

  fprintf(stderr, "%8.3f seconds elapsed to cluster\n",get_time()-start_cluster);
  FILE *output_fp;
  char temp_filename[FILENAME_LENGTH];
  sprintf(temp_filename,"%s.cluster.stats",cluster_options->output_filename);
  open_file(&output_fp, temp_filename, "w", "main"); 
  hcluster.print_centers(output_fp,hcluster.best_partition);
  hcluster.print_cluster_stats(output_fp,hcluster.best_partition);
  close_file(&output_fp,temp_filename,"main");
  sprintf(temp_filename,"%s.clusters",cluster_options->output_filename);
  open_file(&output_fp, temp_filename, "w", "main"); 
  hcluster.print_density(output_fp,hcluster.best_partition);
  close_file(&output_fp,temp_filename,"main");
  if(models)delete models;
  return;
 }
 //kmeans clustering
 else if(cluster_options->method == KMEANS){
  double start_cluster=get_time();
  //fine parallelism - one kmeans seed at a time
  if(cluster_options->fine_parallel){
   #ifdef OPENMP
    nthreads=(nthreads<omp_get_max_threads())?nthreads:omp_get_max_threads();
   #endif
   if(!(cluster_options->total_iterations) && cluster_options->nsolutions_after_best_score <= 0 ){ 
   //convergence criterion set using pvalue and percential limits
    cluster_options->nsolutions_after_best_score=(int)(log(cluster_options->pvalue)/log(cluster_options->percentile))+1;
    fprintf(stderr,"%d scores before convergence to ensure final score is in %8.5f percentile with pvalue less than %8.5f\n",cluster_options->nsolutions_after_best_score,cluster_options->percentile,cluster_options->pvalue);
   } 
   mapped_cluster_set<T> kcluster=mapped_cluster_set<T>(models,cluster_options->nclusters,1,seed);
   //always update 
   kcluster.best_partition->recalc_density_ratio=1.0;
   float best_score=(kcluster.greater_is_better)? -FLT_MAX : FLT_MAX;     //absolute best_score as a function of densities
   bool change_flag=false,done =false;                                      //has there been a change in best score 
   int iterations=0,old_iterations=0,iterations_between_best_scores=0; //these variables control termination
   int limit;
   if (cluster_options->total_iterations) limit=cluster_options->total_iterations;
   else{
    limit =(cluster_options->nsolutions_after_best_score *2 >10000)?cluster_options->nsolutions_after_best_score *2 : 10000;
   }
   cluster_partition<T> best_partition=*(kcluster.best_partition);
   while(iterations<limit && !done){ 
    //loop exits after max_iterations or when the done flag is set if sufficient number of iterations pass with improvement in scores or the iterations done by all threads exceeds max_iterations
    int rvalue=kcluster.best_partition->parallel_kcluster(nthreads,10,100,kcluster.models,&cluster_partition<T>::parallel_assign_to_lowest_average_distance,cluster_options->nfixed_centers,true);
     if(rvalue != -1){
      if(rvalue && kcluster.best_partition->better(kcluster.best_partition->score,best_score)){
       best_score=kcluster.best_partition->score;
       kcluster.best_partition->fast_copy(&best_partition);
       change_flag=true;
      }
     }
     iterations++;
     if(cluster_options->nsolutions_after_best_score>=0){
      if(change_flag){
       iterations_between_best_scores=0;
       change_flag=false;
      }
      else{
       iterations_between_best_scores++;
      } 
      if(iterations_between_best_scores > cluster_options->nsolutions_after_best_score){ 
       done=true;
      }
      change_flag=false;
     } 
    }
   if(best_score != FLT_MAX &&  best_score !=  -FLT_MAX){ //this checks that at least one good answer has been obtained 
    if(cluster_options->nsolutions_after_best_score >=0 ){
     if(iterations_between_best_scores > cluster_options->nsolutions_after_best_score)
      fprintf(stderr,"score converged %9.5f after %d iterations\n",best_partition.score,iterations);
     else
      fprintf(stderr,"final score %9.5f after %d iterations\n",best_partition.score,iterations);
    }
    else fprintf(stderr,"final score %9.5f after %d iterations\n",best_partition.score,iterations);   
    best_partition.sort_clusters_insort();
    best_partition.find_center_structures_using_density_of_cluster();
    double end_cluster = get_time(); 
    fprintf(stderr, "%8.3f seconds elapsed to cluster\n",end_cluster-start_cluster);  
    kcluster.print_centers(stderr,&(best_partition));
    FILE *output_fp;
    char temp_filename[FILENAME_LENGTH];
    sprintf(temp_filename,"%s.cluster.stats",cluster_options->output_filename);
    open_file(&output_fp, temp_filename, "w", "main"); 
    kcluster.print_centers(output_fp,&(best_partition));
    kcluster.print_cluster_stats(output_fp,&(best_partition));
    close_file(&output_fp,temp_filename,"main");
    sprintf(temp_filename,"%s.clusters",cluster_options->output_filename);
    open_file(&output_fp, temp_filename, "w", "main"); 
    kcluster.print_density(output_fp,&(best_partition));
    close_file(&output_fp,temp_filename,"main");
    if(models)delete(models);   
   }
   else{
    fprintf(stderr,"unable to find valid kmeans partition after %d attempts\n",iterations);
   }
  }
  
  
  else{
   //coarse kmeans parallelism many kmeans seeds at a time 
   if(!(cluster_options->total_iterations) && cluster_options->nsolutions_after_best_score <= 0 ){ 
   //convergence criterion set using pvalue and percential limits
    cluster_options->nsolutions_after_best_score=(int)(log(cluster_options->pvalue)/log(cluster_options->percentile))+1;
    fprintf(stderr,"%d scores before convergence to ensure final score is in %8.5f percentile with pvalue less than %8.5f\n",cluster_options->nsolutions_after_best_score,cluster_options->percentile,cluster_options->pvalue);
   }
    parallel_cluster_set<T> parallel_set=parallel_cluster_set<T>(nthreads,models,seed,cluster_options->nclusters,cluster_options->min_cluster_size);
   if(cluster_options->total_iterations)
    parallel_set.parallel_cluster(&cluster_partition<T>::assign_to_lowest_average_distance,cluster_options->total_iterations,10,100,cluster_options->nfixed_centers);
   else{
    int limit =(cluster_options->nsolutions_after_best_score *2 >10000)?cluster_options->nsolutions_after_best_score *2 : 10000;
    parallel_set.parallel_cluster(&cluster_partition<T>::assign_to_lowest_average_distance,limit,10,100,cluster_options->nsolutions_after_best_score,cluster_options->nfixed_centers);
   }
   double end_cluster = get_time(); 
   fprintf(stderr, "%8.3f seconds elapsed to cluster\n",end_cluster-start_cluster);
   fprintf(stderr,"writing clusters to files\n");
   FILE *output_fp;
   char temp_filename[FILENAME_LENGTH];
   sprintf(temp_filename,"%s.cluster.stats",cluster_options->output_filename);
   open_file(&output_fp, temp_filename, "w", "main"); 
   parallel_set.print_cluster_summary(output_fp);
   close_file(&output_fp,temp_filename,"main");
   sprintf(temp_filename,"%s.clusters",cluster_options->output_filename);
   open_file(&output_fp, temp_filename, "w", "main"); 
   parallel_set.print_clusters(output_fp);
   close_file(&output_fp,temp_filename,"main");
   parallel_set.best_cluster_set->print_centers(stderr,parallel_set.best_cluster_set->best_partition);

  }
 }
}
