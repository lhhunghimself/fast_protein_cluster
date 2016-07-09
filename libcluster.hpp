#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include "my_sort.h"
#include <sys/stat.h>
#ifdef GPU
#define MAX_ELEMENTS 8192*4096
#ifdef AMD
 #define TM_MAX_SQ_BLOCK_SIZE 1280
 #define TM_MAX_TR_BLOCK_SIZE 2560
 #define RMSD_MAX_SQ_BLOCK_SIZE 1280
 #define RMSD_MAX_TR_BLOCK_SIZE 2560
#endif 
#ifdef NVIDIA
 #define TM_MAX_SQ_BLOCK_SIZE 128
 #define TM_MAX_TR_BLOCK_SIZE 128
 #define RMSD_MAX_SQ_BLOCK_SIZE 512
 #define RMSD_MAX_TR_BLOCK_SIZE 1024
#endif 
extern char gtmscorecl_location[LINE_LENGTH], grmsdcl_location[LINE_LENGTH]; //path to tmscore.cl and rmsd.cl kernels 
#endif
#define RMSD_MIN_VALUE 0.0
#define RMSD_STEP_SIZE_VALUE 0.06	
#define TMSCORE_MIN_VALUE 0.17056
#define TMSCORE_STEP_SIZE_VALUE 0.00324	
#define SQRT3 1.732050807568877
using namespace std;

enum _dmatrix_type {NO_DMATRIX,FLOAT,COMPACT};
enum _simd_type {SCALAR_,SSE2_,SSE3_,AVX_};
enum _metric_type {RMSD,TMSCORE};
enum _cluster_method{NO_CLUSTER,KMEANS,KCENTERS,HCOMPLETE,HSINGLE,HAVERAGE,DENSITY};
enum _compute_type{cCPU,cGPU};
enum _matrix_type{NO_MATRIX,BINARY,TEXT,CHAR};
enum _input_type{PDB_LIST,BINARY_COORDS};

//forward declaration of template classes - sometimes needed for friend class declarations
template <class T> class priority_heap_list;
template <class T> class triangular_matrix;
template <class T> class mapped_cluster_set;
template <class T> class parallel_cluster_set;
template <class T> class cluster_partition;
template <class T> class mapped_cluster_models_set;
template <class T> class cluster_models_set;
class cpu_properties;

//routines from tmscore/rmsd library

//scalar
void center_all_coords(int nstructs,int nat,float *coords);
float rmsd_cpu(int nat,float *coords1,float *coords2,float *rmatrix); //needed for final tmscore step
template <class T> float rmsd_cpu_par(int nthreads,int nat,int nmodels,float *coords,triangular_matrix<T> *matrix);
float rmsd_cpu_par(int nthreads,int nat,int nmodels,float *coords, float *density);
float tmscore_rmsd_cpu(int nat,float *coords1,float *coords2,float bR[3][3], float bt[3],float *rmsd);

//sse2
#ifdef SSE2
void shuffle_tmscore_coords_soa_sse(int nstructs, int nat, float *coords,float **x,float **y,float **z); 
int shuffle_coords4_sse (int nstructs,int pdb_size, float *coords, float *shuffled_coords,float *ssqs);
template <class T> float rmsd_sse2_par(int nthreads,int nat,int nmodels,float *coords, triangular_matrix<T> *matrix);
float rmsd_sse2_par(int nthreads,int nat,int nmodels,float *coords, float *density);
float tmscore_cpu_soa_sse2(int nat, float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3], float *rmsd);
#endif
#ifdef SSE3
//sse3
template <class T> float rmsd_sse3_par(int nthreads,int nat,int nmodels,float *coords, triangular_matrix<T> *matrix);
float rmsd_sse3_par(int nthreads,int nat,int nmodels,float *coords,float *density);
float tmscore_cpu_soa_sse3(int nat, float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3], float *rmsd);
#endif           
#ifdef AVX
//avx
template <class T> float rmsd_avx_par(int nthreads,int nat,int nmodels,float *coords,triangular_matrix<T> *matrix);
float rmsd_avx_par(int nthreads,int nat,int nmodels,float *coords, float *density);
void shuffle_tmscore_coords_soa_avx(int nstructs, int nat, float *coords,float **x,float **y,float **z);
float tmscore_cpu_soa_avx(int nat, float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3], float *rmsd);
int shuffle_coords8_avx (int nstructs,int pdb_size, float *coords, float *shuffled_coords,float *ssqs);
float shuffle_center_coords8_avx(int nat, float *coords, float *shuffled_coords,float centroid[3]);
float shuffle_center_coords8_unaligned_avx(int nat,float *coords, float *shuffled_coords,float centroid[3]);
#endif
int convert_coords_to_float4 (int nstructs,int pdb_size, float *coords, float4 *coords4);
int calculate_number_of_frames(int nat);
#ifdef GPU
int define_sizes_string (char **defines_string, int nthreads, int pdb4_size);
int read_source_file(char **array,const char *filename,char *defines_string);
int define_decoy_sizes_string (char **defines_string, int nthreads, int pdb4_size);
char *print_cl_errstring(cl_int err);
#endif
void shuffle_tmscore_coords_soa(int nstructs, int nat, float *coords,float **x,float **y,float **z);
float shuffle_center_coords4_sse(int nat, float *coords, float *shuffled_coords,float centroid[3]);

//misc common routines
double get_time();
void mean_stdev(int narray,float *array, float *mean, float *stdev);
void kshuffle(int k, int *arr, int narr, unsigned int *seed);
//sort routines
int sort_by_scores (int nstructs,float *scores, int *sorted,int min_first_flag);
void  sedgesort (KEY_T  *array, int len);
void  partial_quickersort (KEY_T *array, int lower, int upper);
void  insort (KEY_T  *array, int len);

extern double gtimer1,gtimer2;
extern cpu_properties gcpu_info;

class cpu_properties{
  public:
  char vendor[12];
  unsigned logical;
  unsigned cores;
  bool hyperthreads; 
  cpu_properties(){
   //thanks to jcoffland for the code  
   unsigned regs[4];

   // Get vendor
   
   cpuID(0, regs);
   ((unsigned *)vendor)[0] = regs[1]; // EBX
   ((unsigned *)vendor)[1] = regs[3]; // EDX
   ((unsigned *)vendor)[2] = regs[2]; // ECX
   string cpuVendor = string(vendor, 12);

    // Get CPU features
    cpuID(1, regs);
    unsigned cpuFeatures = regs[3]; // EDX

    // Logical core count per CPU
    cpuID(1, regs);
    logical = (regs[1] >> 16) & 0xff; // EBX[23:16]
//    cout << " logical cpus: " << logical << endl;
    cores = logical;

    if (cpuVendor == "GenuineIntel") {
     // Get DCP cache info
     cpuID(4, regs);
     cores = ((regs[0] >> 26) & 0x3f) + 1; // EAX[31:26] + 1

    } else if (cpuVendor == "AuthenticAMD") {  
     // Get NC: Number of CPU cores - 1
     cpuID(0x80000008, regs);
     cores = ((unsigned)(regs[2] & 0xff)) + 1; // ECX[7:0] + 1
     }

 //     cout << "    cpu cores: " << cores << endl;

    // Detect hyper-threads  
    hyperthreads = cpuFeatures & (1 << 28) && cores < logical;

//   cout << "hyper-threads: " << (hyperthreads ? "true" : "false") << endl;
  }
 void cpuID(unsigned i, unsigned regs[4]) {
  #ifdef _WIN32
   __cpuid((int *)regs, (int)i);

  #else
  asm volatile
    ("cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3])
     : "a" (i), "c" (0));
  // ECX is set to zero for CPUID function 4
  #endif
  }
};  

class cluster_options{
 public:
 _metric_type score_type;
 _matrix_type read_matrix_type,write_matrix_type;
 _dmatrix_type distance_matrix_type;
 _simd_type simd_type;
 _compute_type compute;
 _cluster_method method;
 _input_type input_type;
 int
  all_atoms,                   //whether to calculate RMSD on all_atoms or the default CA's only -not yet implemented
  //Cluster Variables
  nclusters,                   //number of final clusters -used for kmeans and kcenters
  //variables for kmeans    
  min_cluster_size,            //minimum number of elements in cluster
  nfixed_centers,              //used for hybrid kcenters kmeans 
                               //the first centers will be randomly determined and the last nfixed centers will be determined as the most distant from the random centers
  total_iterations,            //number of valid seeded solutions obtained for kmeans before stopping - if not set - will set to 10,000 or 2*nsolutions_after_best_score
  maximum_iterations,          //maximum number of iterations before convergence of a single solution
  nsolutions_after_best_score, //number of kmeans partitions with no improvement in score before terminating
  fine_parallel,               //use fine_level parallelism for kmeans instead of task level for multiple kmeans
  gpu_id;                      //optional gpu_id to specify different gpus in multigpu setups  - default is -1 when NO_MATRIX is specified
 float pvalue,                    //desired pvalue - used to calculate nsolutions_after_best_score
       percentile;                //desired estimated percentile where the best possible score is 1.00
 char read_matrix_filename[FILENAME_LENGTH],
      write_matrix_filename[FILENAME_LENGTH], 
      input_filename[FILENAME_LENGTH],
      output_filename[FILENAME_LENGTH];
 char *subset_filename;
 public:
  cluster_options(){
   input_type=PDB_LIST;
   score_type= RMSD;
   read_matrix_type=NO_MATRIX;write_matrix_type=NO_MATRIX;
   distance_matrix_type=FLOAT,simd_type =SCALAR_;
   compute=cCPU;
   method=KMEANS;
   all_atoms=0;
   nclusters=5;
   fine_parallel=0;
   gpu_id=-1;
   //variables for kmeans    
   min_cluster_size=2;
   nfixed_centers=0;
   total_iterations=0;
   maximum_iterations=10000;
   nsolutions_after_best_score=-1; 
   pvalue=.01;               
   percentile=.99;
   subset_filename=0;
   strcpy(input_filename,"cluster_input");
   strcpy(output_filename,"cluster_output");
  }
  ~cluster_options(){
   if(subset_filename) delete[] subset_filename;
  }
};
class prune_options{
 public:
  _metric_type score_type;
  _matrix_type read_matrix_type,write_matrix_type;
  _dmatrix_type distance_matrix_type;
  _simd_type simd_type;
  _compute_type compute;
  _cluster_method method;
  _input_type input_type;
  int
   all_atoms,                   //whether to calculate RMSD on all_atoms or the default CA's only 
   nclusters,
   prune_max_size,              //indicates what the maximum size of the set should be after pruning
   prune_min_size,              //indicates what the minimum size of the set should be after pruning
   prune_zmode,                 //determines whether outlier removal variables are calculated with zscores or using absolute values
   keep_log,                    //keep track of which models are discarded - can be expensive for large sets
   gpu_id;                      //optional gpu_id to specify different gpus in multigpu setups  default is -1
  float prune_min_zvalue,            //the number of standard deviations from mean before being a outlier
        prune_outlier_ratio,         //controls the ratio outliers_removed/ensemble to limit the maximum number of outliers removed before the density is recalculated - if set to zero means that only the worst one is removed
        prune_to,                    //prune until this density value is the worst remaining in ensemble
        prune_stop;                  //stop pruning when the worst density is this value

  char read_matrix_filename[FILENAME_LENGTH],
      write_matrix_filename[FILENAME_LENGTH], 
      input_filename[FILENAME_LENGTH],
      output_filename[FILENAME_LENGTH];
  char *subset_filename;
  //defaults
  prune_options(){
   input_type=PDB_LIST;
   score_type= RMSD;
   read_matrix_type=NO_MATRIX;write_matrix_type=NO_MATRIX;
   distance_matrix_type=FLOAT,simd_type =SCALAR_;
   compute=cCPU;
   method=DENSITY;
   all_atoms=0;
   keep_log=0;
   nclusters=3;
   gpu_id=-1;
   //variables for kmeans 
   subset_filename=0; 
   prune_max_size=0;
   prune_min_size=0;
   prune_zmode=0;
   prune_min_zvalue=0;
   prune_outlier_ratio=0;
   prune_to=0;
   prune_stop=0;
   strcpy(input_filename,"prune_input");
   strcpy(output_filename,"prune_output");
 }
 ~prune_options(){;
   if(subset_filename) delete[] subset_filename;
  }
};

template <class T>
class priority_heap_list{ //used to find worst n scores
 //This is used to keep track of the worst/best k scoring elements
 //heap structure is used - first node is best scoring of bad elements
 //if there are less than k in heap - element is added to heap
 //if there are k in the heap - replace element at top of heap with nee element if the score of the element is worst than the best score and restore heap
 //otherwise do not add to heap
 //should be O(nlogk) - non destructive - read selection from heap map
 //allow use of existing level of indirection through input *score_map allows for prefiltering - for example by a cutoff of the scores before the selection without much overhead
 public:
  int nscores,heap_size;
  T *scores;
  int *heap_map;  //maps the linear array into the mini-heap structure used to find the k worst/best - max scores as defined by is_greater go on top - contains the k selected set
  priority_heap_list (T *arr, int nworst,int mapped_arr_size,bool input_greater_is_better,int *score_map){
   heap_size=nworst;nscores=mapped_arr_size;scores=arr;
   heap_map=new int[nscores];
   greater_is_better=input_greater_is_better;
   if(score_map)
    memmove(heap_map,score_map,nscores*sizeof(int));
   else{
    for(int i=0;i<nscores;i++)
     heap_map[i]=i;
   }
   //heapify
   for (int start =heap_size/2-1; start >=0; start--){ 
    siftDown(start);
   }
   prioritize_rest_of_list();
  }
  ~priority_heap_list(){
   if(heap_map)delete [] heap_map;
  }
  int pop_heap(){
   int retvalue=heap_map[0];
   heap_map[0]=heap_map[heap_size-1];
   heap_size--;
   siftDown(0);
   return(retvalue);
  } 
  bool better (T a, T b){   
   return ((greater_is_better && a>b) || (!greater_is_better && a<b));
  }   
 private:
  bool greater_is_better;
  T insert(int index){
   heap_map[0]=heap_map[index];
   siftDown(0);
   return(scores[heap_map[0]]);
  }
  void siftDown(int start){
   int root = start;
   while (root*2+1 < heap_size ) {
    int ir=heap_map[root];
    T vr=scores[ir];
    int child = 2*root+1;

    int ic=heap_map[child];
    T vc=scores[ic];
    if (child +1 < heap_size  &&  better(scores[heap_map[child+1]],vc)){
     child++;
     vc=scores[heap_map[child]];
    }
    if (better(vc,vr)) {
     heap_map[root]=heap_map[child];
     heap_map[child]=ir;
     root=child;
    }
    else return;
   }
  }
  void prioritize_rest_of_list(){ //feed the rest of the scores to the mini_heap
   T best_score_of_bad_set=scores[heap_map[0]];
   for(int i=heap_size;i<nscores;i++){
    if(better(best_score_of_bad_set,scores[heap_map[i]])){
     best_score_of_bad_set=insert(i);
    } 
   }
  }
};

class pruned_model{ //keeps track of pruned models
 public:
  char *name;   
  float score;
  pruned_model(char *input_name,float input_score){
   int len=strlen(input_name)+1;
   name=new char[len];
   strcpy(name,input_name);
   score=input_score;
  }
  ~pruned_model(){
   if(name) delete [] name;
  }
  pruned_model(const pruned_model &A): score(A.score){
   name=new char[strlen(A.name)+1];
   strcpy(name,A.name);
  }
  pruned_model & operator = (const pruned_model &rhs){
   if(this != &rhs){
    score=rhs.score;
    if(name) delete [] name;
    name=new char[strlen(rhs.name)+1];
    strcpy(name,rhs.name);       
   }  
  }
}; 
template <class T>
class triangular_matrix{
 public:
  int element_size;
  int length;
  float min_value,step_size,max_value,inv_step_size;
  triangular_matrix(int input_length,float min_value,float step_size,bool greater_is_better){ //greater is better is only used in unsigned char case
   length=input_length;
   element_size=sizeof(T);
   tmatrix=new T*[length];
   for(int i=1;i<length;i++)
    tmatrix[i]= new T[i];
   fprintf(stderr,"allocate triangular matrix of %d length\n",length); 
   this->min_value= min_value;
   this->step_size= step_size;
   this ->max_value=min_value+255.5f*step_size;
   inv_step_size=1.0f/step_size;
  }
  triangular_matrix(const triangular_matrix &A) : length(A.length), element_size(A.element_size), tmatrix (new T*[A.length]){
   tmatrix=new T*[length];
   for(int i=1;i<length;i++){
    tmatrix[i]= new T[i];
    memmove(tmatrix[i],A.tmatrix[i],i*sizeof(T));
   }
   this ->max_value=min_value+255.5f*step_size;
   inv_step_size=1.0f/step_size;
  }
  triangular_matrix &operator=(const triangular_matrix &rhs ){
   if(this != &rhs){ 
    length=rhs.length;element_size=rhs.element_size;
    T **new_matrix=new T*[length];
    for(int i=1;i<length;i++){
     new_matrix[i]= new T[i];
     memmove(new_matrix[i],rhs.tmatrix[i],i*sizeof(T));
    }
    for(int i=1;i<length;i++)
     if(tmatrix[i])delete [] tmatrix[i];
    if(tmatrix)delete[] tmatrix;   
    tmatrix=new_matrix;
   }
   this ->max_value=min_value+255.5f*step_size;
   inv_step_size=1.0f/step_size;
   return *this;   
  }
  void adjust_max_min(){
   float max,min;
   max=get_matrix(1,0);
   min=get_matrix(1,0);
   for(int i=1;i<length;i++){
    for(int j=0;j<i;j++){
     float value=get_matrix(i,j);
     max=(value>max)? value :max;
     min=(min>value)? value :min;
    }
   }
   min_value=min;
   step_size=(max-min)/256.0f;
   inv_step_size=1.0/step_size;
   max_value=min_value+255.5f*step_size;
  }    
  void adjust_max_min(int *map, int nmap){
   float max,min;
   max=get_matrix(map[1],map[0]);
   min=get_matrix(map[1],map[0]);
   for(int i=1;i<nmap;i++){
    int a=map[i];
    for(int j=0;j<i;j++){
     int b=map[j];
     float value=get_matrix(a,b);
     max=(value>max)? value :max;
     min=(min>value)? value :min;
    }
   }
   min_value=min;
   step_size=(max-min)/256.0f;
   inv_step_size=1.0/step_size;
   max_value=min_value+255.5f*step_size;
  }

  int zero_matrix(){
   for(int i=1;i<length;i++){
    memset(tmatrix[i],0,i*sizeof(T));
   } 
  }  
 ~triangular_matrix(){//non-compact version
   for(int i=1;i<length;i++)
    if(tmatrix[i])delete [] tmatrix[i];
   if(tmatrix)delete [] tmatrix;
  } 

  T* get_address(int i, int j){
    return((i>j)?&(tmatrix[i][j]) :&(tmatrix[j][i]));
  } 
  T get_matrix(int i, int j){
   return((i<j)?tmatrix[j][i] :tmatrix[i][j]);
  }
  T get_matrix_fast(int i, int j){
   return (tmatrix[i][j]);
  }
  T get_native_matrix(int i, int j){
   return((i<j)?tmatrix[j][i] :tmatrix[i][j]);
  }  
  T get_native_matrix_fast(int i, int j){
   return (tmatrix[i][j]);
  }
  void set_matrix_fast (int i, int j, T value){
   tmatrix[i][j]=value;
  }
  T get_matrix_safe(int i, int j){
   if(i<0 || i>= length || j<0 || j>=length){
    fprintf(stderr,"subscripts %d %d out of range for size %d matrix\n",i,j,length);
    exit(FALSE); 
   }
   return((i<j)?tmatrix[j][i] :tmatrix[i][j]);
  }
  void set_matrix_safe (int i, int j, T value){
   if(i<0 || i>= length || j<0 || j>=length){
    fprintf(stderr,"subscripts %d %d out of range for size %d matrix\n",i,j,length);
    exit(FALSE); 
   } 
   if(i<j)tmatrix[j][i]=value;
   else tmatrix[i][j]=value;
  }
  void set_matrix(int i, int j, T value){
   if(i<j)tmatrix[j][i]=value;
   else tmatrix[i][j]=value;
  }   
  int read_matrix_from_binary_file(FILE *fp,int *inverse_map,int ninverse_map){
   int nread=0;
   int nseek=0;
   for(int i=1;i<ninverse_map;i++){
    int a=inverse_map[i];
    if(a >=0){
     for(int j=0;j<i;j++){
      int b=inverse_map[j];
      if(b>=0){
       T value;
       fseek(fp,nseek*element_size,SEEK_CUR);
       fread(&value,element_size,1,fp);
       set_matrix(a,b,value);
       nread++;
       nseek=0;
      }
      else nseek++;
     }
    }
    else{
     nseek+=i;
    }  
   }
   return(nread);
  }
  
  int read_matrix_from_binary_file(FILE *fp){
   int read,n=0,rvalue=1;
   char dum;
   for(int i=1;i<length;i++){
    read=fread(tmatrix[i],element_size,i,fp);
    if(!read){
     fprintf(stderr,"file too short - read aborted at %d rows\n",i-1);
     return(0);
    }
   }
   if(fread(&dum,sizeof(char),1,fp)){
    fprintf(stderr,"warning not at EOF after %d rows\n",length);
    rvalue=-1;
   }
   return(rvalue);
  }
  int read_matrix_from_compact_file(FILE *fp,int *inverse_map,int ninverse_map){	
   //reads and converts binary matrix to normal distance matrix
   int nread=0;
   int nseek=0;
   int my_length;
   {
    fread(&my_length,sizeof(int),1,fp);
    fread(&min_value,sizeof(float),1,fp);//compact write is always float min
    fread(&step_size,sizeof(float),1,fp);
   }
   for(int i=1;i<ninverse_map;i++){
    int a=inverse_map[i];
    if(a >=0){
     for(int j=0;j<i;j++){
      int b=inverse_map[j];
      if(b>=0){
       unsigned char ctemp;
       fseek(fp,nseek,SEEK_CUR);
       fread(&ctemp,1,1,fp);
       set_matrix(a,b,min_value+step_size*(float)ctemp);
       nread++;
       nseek=0;
      }
      else nseek++;
     }
    }
    else{
     nseek+=i;
    }  
   }
   return(nread);
		}
  int read_matrix_from_compact_file(FILE *fp){
   int nread=0;	
   //reads and converts binary matrix to normal distance matrix
   int my_length;
   {
    fread(&my_length,sizeof(int),1,fp);
    fread(&min_value,sizeof(float),1,fp);//compact write is always float min
    fread(&step_size,sizeof(float),1,fp);
   }
   
   for(int i=1;i<length;i++){
    for(int j=0;j<i;j++){
     unsigned char ctemp;
     fread(&ctemp,1,1,fp);
     tmatrix[i][j]=min_value+step_size*(float)ctemp;
     nread++;
    }
   }
   return(nread);
		}
  int read_matrix_from_text_file(FILE *fp,int *inverse_map){
   char line[LINE_LENGTH];
   int nread=0;
   while (fgets(line, LINE_LENGTH, fp)){
    float value;
    int m,n,i,j;;
    sscanf (line, "%d %d %f",&m,&n,&value);
    i=inverse_map[m];
    j=inverse_map[n];
    if(i>=0 && i<length && j>=0 && j <length){
     tmatrix[i][j]=value;
     nread++;
    }
   }
   return(nread);
  }
  int read_matrix_from_text_file(FILE *fp){
   char line[LINE_LENGTH];
   int nread=0;
   while (fgets(line, LINE_LENGTH, fp)){
    float value;
    int i,j;
    sscanf (line, "%d %d %f",&i,&j,&value);
    if(i<length && j <length){
     tmatrix[i][j]=value;
     nread++;
    }
   }
   return(nread);
  }
  unsigned char convert_to_char(T value){
   float findex=(value-min_value)/step_size;
   int index=(int) (findex+.5);
   if(index <0)index=0;
   if(index >255)index=255;
   return(unsigned char)index;
  } 
  
  void write_matrix_to_compact_file(FILE *fp){
   adjust_max_min();//find current max min
   fwrite(&length,sizeof(int),1,fp);
   fwrite(&min_value,sizeof(float),1,fp);
   fwrite(&step_size,sizeof(float),1,fp);;
   for(int i=1;i<length;i++)
    for(int j=0;j<i;j++){
     unsigned char cvalue=convert_to_char(tmatrix[i][j]);
     fwrite(&cvalue,1,1,fp);
    }
  }
  void write_matrix_to_compact_file(FILE *fp,int *map,int nmap){
   adjust_max_min(map,nmap);//find current max min
   fwrite(&length,sizeof(int),1,fp);
   fwrite(&min_value,sizeof(float),1,fp);
   fwrite(&step_size,sizeof(float),1,fp);
   for(int i=1;i<nmap;i++){
    int a=map[i];
    for(int j=0;j<i;j++){
     int b=map[j];
     unsigned char cvalue=convert_to_char(get_matrix(a,b));
     fwrite(&cvalue,1,1,fp);
    }
   }
  }
  
  void write_matrix_to_binary_file(FILE *fp){
   for(int i=1;i<length;i++)
    fwrite(tmatrix[i],element_size,i,fp);
  }  
  void write_matrix_to_binary_file(FILE *fp,int *map,int nmap){
   for(int i=1;i<nmap;i++)
    for(int j=0;j<i;j++){
     int a=map[i];
     int b=map[j];  
     if(a>b)
      fwrite(&(tmatrix[a][b]),element_size,1,fp);
     else
      fwrite(&(tmatrix[b][a]),element_size,1,fp);
    }
  }
  void write_matrix_to_text_file (FILE *fp){
   for (int i=1;i<length;i++)
    for(int j=0;j<i;j++)
     fprintf(fp,"%d %d %f\n",i,j,get_matrix(i,j));
  }
  void write_matrix_to_text_file (FILE *fp,int *map, int nmap){
   for (int m=1;m<nmap;m++){
    for(int n=0;n<m;n++){
     int i=map[m];
     int j=map[n];
     fprintf(fp,"%d %d %f\n",m,n,get_matrix(m,n));
    }
   } 
  }
 void generate_matrix(int nats,int pdb_size,float *coords,_compute_type compute,_metric_type score_type,_simd_type simd_type,int nthreads,int gpu_id){
  if(compute==cCPU){
   if(score_type==RMSD){
    if(simd_type==SCALAR_){
     //no SSE routine
#ifdef OPENMP    
     int max_threads=omp_get_max_threads();
     nthreads=(max_threads<nthreads)? max_threads : nthreads;
#else
     nthreads=1;
#endif          
     rmsd_cpu_par(nthreads,nats,length,coords,this); 
    }

#ifdef SSE2
    else if(simd_type ==SSE2_){//use SSE2 routine 
#ifdef OPENMP     
     int max_threads=omp_get_max_threads();
     nthreads=(max_threads<nthreads)? max_threads : nthreads;
#endif
     rmsd_sse2_par(nthreads,nats,length,coords,this);  
    }
#endif
#ifdef SSE3
    else if(simd_type ==SSE3_){//use SSE2 routine 
#ifdef OPENMP     
     int max_threads=omp_get_max_threads();
     nthreads=(max_threads<nthreads)? max_threads : nthreads;
#endif
     rmsd_sse3_par(nthreads,nats,length,coords,this);  
    }
#endif
#ifdef AVX
    else if(simd_type == AVX_){//use avx routine 
#ifdef OPENMP     
     int max_threads=omp_get_max_threads();
     nthreads=(max_threads<nthreads)? max_threads : nthreads;
#endif   
     rmsd_avx_par(nthreads,nats,length,coords,this);     
    }
#endif
   }//end rmsd
   else if (score_type == TMSCORE){ //TMscore
    float R[3][3],t[3];
    if(simd_type == SCALAR_){
#ifdef OPENMP     
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;     
    #pragma omp parallel for num_threads (nthreads) schedule (dynamic)
#endif
     for (int i=1;i<length;i++){
      for (int j=0;j<i;j++){      
       this->set_matrix(i,j,tmscore_rmsd_cpu(nats,&(coords[i*pdb_size]),&(coords[j*pdb_size]),R,t,0));
      }
     }
    }
#ifdef SSE2
    else if (simd_type == SSE2_){
     float *x=0,*y=0,*z=0;
     shuffle_tmscore_coords_soa_sse(length,nats,coords,&x,&y,&z);
     int anats=(nats%4)? (nats/4)*4+4 : nats;     
#ifdef OPENMP     
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;     
    #pragma omp parallel for num_threads (nthreads) schedule (dynamic)
#endif
     for (int i=1;i<length;i++){
      for (int j=0;j<i;j++){     
       this->set_matrix(i,j,tmscore_cpu_soa_sse2(nats,&(x[i*anats]),&(y[i*anats]),&(z[i*anats]),&(x[j*anats]),&(y[j*anats]),&(z[j*anats]),0,0,0));
      }
     }
     if(x) free (x);     
     if(y) free (y);  
     if(z) free (z);      
    }
#endif
#ifdef SSE3
    else if (simd_type == SSE3_){
     float *x=0,*y=0,*z=0;
     shuffle_tmscore_coords_soa_sse(length,nats,coords,&x,&y,&z);
     int anats=(nats%4)? (nats/4)*4+4 : nats;     
#ifdef OPENMP     
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;     
    #pragma omp parallel for num_threads (nthreads) schedule (dynamic)
#endif
     for (int i=1;i<length;i++){
      for (int j=0;j<i;j++){     
       this->set_matrix(i,j,tmscore_cpu_soa_sse3(nats,&(x[i*anats]),&(y[i*anats]),&(z[i*anats]),&(x[j*anats]),&(y[j*anats]),&(z[j*anats]),0,0,0));
      }
     }
     if(x) free (x);     
     if(y) free (y);  
     if(z) free (z);      
    } 
#endif
#ifdef AVX   
    else if (simd_type == AVX_){
     float *x=0,*y=0,*z=0;
     int unat8=(nats%8)? (nats/8+1)*8 : nats;
     shuffle_tmscore_coords_soa_avx(length,nats,coords,&x,&y,&z);
#ifdef OPENMP     
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;     
    #pragma omp parallel for num_threads (nthreads) schedule (dynamic)
#endif
     for (int i=1;i<length;i++){
      for (int j=0;j<i;j++){    
       this->set_matrix(i,j,tmscore_cpu_soa_avx(nats,&(x[i*unat8]),&(y[i*unat8]),&(z[i*unat8]),&(x[j*unat8]),&(y[j*unat8]),&(z[j*unat8]),0,0,0));
      }
     }         
     if(x) free (x);     
     if(y) free (y);  
     if(z) free (z);  
    }
#endif
  }//end TMSCORE
  else{
   fprintf(stderr,"unrecognized score type %d\n",score_type);
   exit(FALSE);
  }   
 }//end CPU

#ifdef GPU 
  else if(compute == cGPU){
   if (score_type== RMSD)this->find_rmsd_matrix(0,64,0,grmsdcl_location,nats,coords,gpu_id,nthreads);
   else if(score_type==TMSCORE)this->find_tmscore_matrix(0,64,0,gtmscorecl_location,nats,coords,gpu_id,nthreads);
   else{
    fprintf(stderr,"unrecognized score type %d\n",score_type);
    exit(FALSE);
   } 
  }
#endif
  else{     
   fprintf(stderr,"unrecognized compute type %d\n",compute);
   exit(FALSE);  
  }
 }
#ifdef GPU 
 int find_tmscore_matrix(int cpu_flag,int nt, int nwg_per_cu,const char *source,int nats,float *coords,int gpu_id,int nthreads){
  //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
  double start_program=0,start_rmsd=0,end=0;
  float4 *coords4;

  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_int4 sizes,start_points;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel tmscore_matrix,tmscore_matrix_rect;
  cl_mem tmscores_buffer,coords41_buffer,coords42_buffer;
  cl_float2 *tmscores;
  cl_int err;
  cl_uint ncu,num_of_devices,numPlatforms;
  clGetPlatformIDs( 1, &platform, NULL ); 
  //get all devices
  cl_device_id cpus[10],gpus[10];
  int ngpus=0;
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d cpus found\n",num_of_devices);
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_of_devices,cpus, 0);
  }
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=num_of_devices;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  // try to get a supported GPU device
  //test with CPU
  if(cpu_flag)
  { 
   device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"using cpu %d cores found\n",ncu);
  }
  else{
   if (!ngpus)
   {
    fprintf(stderr, "no gpu found - running with cpu");
    device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
   }
   else 
   {
    if(gpu_id > ngpus-1){
     fprintf(stderr,"gpu_id error %d gpus found - highest allowed gpu_id is %d - gpu_id of %d given\n",ngpus,ngpus-1,gpu_id);
     exit(FALSE);
    }  
    if(gpu_id ==-1){ 
     if(ngpus >1){
      find_tmscore_matrix_multiple_gpus (nt,nwg_per_cu,source,nats,coords,nthreads);
      return(1);
     }
     device=gpus[0];
    }
    else device=gpus[gpu_id];
    
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"%d compute units found\n",ncu);
   }
  }
  context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
  queue = clCreateCommandQueue(context, device, 0, &err);  
  //calculate maximum number of workgroups per compute unit
  //for gpus this depends on the memory available
  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  if(cpu_flag)lds*=2; 
   //memory used is memory to cache coords plus memory for the alignment and reduction - this is different from the simple tmscore where the coords memory is used once
   //there is a bug in OpenCL with local memory declared in a block that may or may not be freed... 
  int mem_per_wg=6*nats4*sizeof(float4)+nt*sizeof(float2);
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif
  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"creating coord buffers\n");
  //create hosts arrays and buffers
  if (!(coords4 = (float4*)  malloc(pdb4_size * length *sizeof(float4)))) exit(FALSE);
  convert_coords_to_float4 (length,pdb_size,coords,coords4);
  fprintf(stderr,"created  buffers\n");

  start_rmsd = get_time();  
  program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
  if ((err=clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS))
  {
   fprintf(stderr,"error %d %s\n",err,print_cl_errstring(err));
   char buf[0x10000];
   clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
   fprintf(stderr,"\n%s\n", buf);
   return 1;
  }
  end = get_time();
  fprintf(stderr,"creating kernel\n");
  tmscore_matrix = clCreateKernel(program, "tmscore_matrix", &err); 
  fprintf(stderr, "%8.3f seconds elapsed for program generation\n",end-start_rmsd);
  start_rmsd = get_time();  

  start_rmsd = get_time();  
  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  TM_MAX_TR_BLOCK_SIZE) max_structs=TM_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(length<max_structs)max_structs=length;

   //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix

  int ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  if (ngrid > 1)
  {
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(cl_float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > TM_MAX_SQ_BLOCK_SIZE) max_structs=TM_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(length<max_structs)max_structs=length;
   ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs;
  }
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  fprintf(stderr,"creating openCL coords buff %d\n",max_structs*pdb4_size * sizeof(float4)); 
  coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
  if(ngrid >1)
  {
   fprintf(stderr,"creading square buffers\n");
 //  tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
 //  coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
   if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_sq))))exit(FALSE);  
  }
  else
  {
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(cl_float2),NULL, NULL);
   if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_tr))))exit(FALSE);  
  }
  fprintf(stderr,"calculating frames\n");
  int nseeds=calculate_number_of_frames(nats);

  sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
  for (int ni=0;ni<ngrid;ni++)
  {
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
   int block_structs=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
   int nwu= block_structs*(block_structs-1)/2;
   int offset=ni*max_structs;
   sizes.z=block_structs;
   sizes.w=nwu;
   start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
   //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d %d %d \n",ni,ni,nwu,block_structs,ngrid,block_structs,block_structs);

   clSetKernelArg(tmscore_matrix, 0,sizeof(cl_int4),&sizes); 
   clSetKernelArg(tmscore_matrix, 1,sizeof(int),&nseeds);
   clSetKernelArg(tmscore_matrix, 2,sizeof(cl_int4),&start_points);
   clSetKernelArg(tmscore_matrix, 3,sizeof(coords41_buffer),&coords41_buffer);
   clSetKernelArg(tmscore_matrix, 4,sizeof(tmscores_buffer),&tmscores_buffer);
   clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
   clFinish( queue);
   size_t global,local=nt;
   global=max_nwg*local;
   clEnqueueNDRangeKernel(queue, tmscore_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
   clFinish(queue);
   clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
   clFinish( queue);

   //output to matrix

  int m=0;
  for(int i=0;i<block_structs-1;i++)
   for(int j=i+1;j<block_structs;j++){
    this->set_matrix(j+offset,i+offset,tmscores[m].x);
    m++;
   }
  }

  fprintf(stderr,"releasing resources from first kernel\n");
  clReleaseKernel(tmscore_matrix);
  clReleaseMemObject(coords41_buffer);
  clReleaseMemObject(tmscores_buffer);
  clReleaseCommandQueue(queue);
  
  fprintf(stderr,"allocating resources for second kernel\n");
  if(ngrid >1)
  {
   fprintf(stderr,"creating new queue\n");
   queue = clCreateCommandQueue(context, device, 0, &err); 
   fprintf(stderr,"creating new kernel\n");
   tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
   fprintf(stderr,"creating buffers\n");
   coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
   sizes.x=nats;sizes.y=nats4;
  
   for (int ni=0;ni<ngrid-1;ni++)
    for (int nj=ni+1;nj<ngrid;nj++)
     if(ni!=nj)
     {
      //rectangular tile
      int block_structs1=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
      int block_structs2=(max_structs<length-nj*max_structs) ? max_structs : length-nj*max_structs;
      int nwu= block_structs1*block_structs2;
      int offset1=ni*max_structs;
      int offset2=nj*max_structs;
      sizes.z=block_structs1;
      sizes.w=nwu;
      //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d %d %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid,block_structs1,block_structs2);

      start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
      clSetKernelArg(tmscore_matrix_rect, 0,sizeof(cl_int4),&sizes); 
      clSetKernelArg(tmscore_matrix_rect, 1,sizeof(int),&nseeds);
      clSetKernelArg(tmscore_matrix_rect, 2,sizeof(cl_int4),&start_points);
      clSetKernelArg(tmscore_matrix_rect, 3,sizeof(coords41_buffer),&coords41_buffer);
      clSetKernelArg(tmscore_matrix_rect, 4,sizeof(coords42_buffer),&coords42_buffer);
      clSetKernelArg(tmscore_matrix_rect, 5,sizeof(tmscores_buffer),&tmscores_buffer);
      clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
      clEnqueueWriteBuffer(queue, coords42_buffer , CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
      clFinish( queue );
      size_t global,local=nt;
      global=max_nwg*local;
      clEnqueueNDRangeKernel(queue, tmscore_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
      clFinish( queue );
      clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
      clFinish( queue);
      int m=0;
      for(int i=0;i<block_structs1;i++)
       for(int j=0;j<block_structs2;j++)
       {
        this->set_matrix(j+offset2,i+offset1,tmscores[m].x);
        m++;
       }
     }
   clReleaseMemObject(coords41_buffer);
   clReleaseMemObject(tmscores_buffer);
   clReleaseProgram(program);
   clReleaseMemObject(coords42_buffer);
   clReleaseKernel(tmscore_matrix_rect);
   clReleaseCommandQueue(queue);
  }
  fprintf(stderr,"finished\n");  
  end = get_time();  
  fprintf(stderr, "%8.3f seconds elapsed for %d TM-scores at %8.3f ms per TM-score\n",end-start_rmsd,length*(length-1)/2,(float)((end-start_rmsd)*1000)/(float)(length*(length-1)/2));
  clReleaseContext(context);
  if(coords4)free(coords4);
  if(tmscores)free(tmscores);
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
 }
 int find_tmscore_matrix_multiple_gpus (int nt, int nwg_per_cu,const char *source,int nats,float *coords,int nthreads){
  //this is called after the multiple gpu is detected
  //divides the work based upon number of compute units detected
  const int max_gpus=10;
  int total_cu=0;
  int max_threads=omp_get_max_threads();
  int omp_nt=(max_threads<nthreads)? max_threads : nthreads;
    //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  //need centered coords for this 
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
 
  //center coords - may not have been done previously
  center_all_coords(length,pdb_size/3,coords);
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_platform_id platform;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
  
 //get all devices
  cl_device_id gpus[max_gpus],ncus[max_gpus];
  int ngpus=0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices);
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=(omp_nt >num_of_devices)? num_of_devices : omp_nt;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  //find total number of compute units
  for (int i=0;i<ngpus;i++){
   cl_device_id device;
   device=gpus[i];
   clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL);
   total_cu+=ncu;
  }
  fprintf(stderr,"%d compute units found in %d gpus\n",total_cu,ngpus);
  
  //calculate maximum number of workgroups per compute unit - use smaller gpu size
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  //use same block sizes for now - can optimize for different gpus by adjusting blocksizes - would require storing sizes and offsets in arrays
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers

  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  TM_MAX_TR_BLOCK_SIZE) max_structs=TM_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(length<max_structs)max_structs=length;

  int ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  //adjust ngrid to ngpus or length
  if (ngrid > 1 || ngpus >1 ){
   //recalculate with MAX_ELEMENTS/2
   int min_split=ngpus*2;
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(length<max_structs)max_structs=length;
   ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs;
   if(ngrid <min_split){
    ngrid=min_split;
    max_structs=(length%min_split)? length/min_split+1 : length/min_split;
   }
  }
  
  //set up lock variables
  int *diagonal_lock=new int [ngrid];
  memset(diagonal_lock,0,sizeof(int)*ngrid);
  int *rectangle_lock=new int [ngrid*ngrid];
  memset(rectangle_lock,0,sizeof(int)*ngrid*ngrid);
  //set diagonals of rectangle lock file to zero
  for (int i=0;i<ngrid;i++)
   rectangle_lock[i*ngrid+i]=1;
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  double start_rmsd;  
  fprintf(stderr,"calculating frames\n");
  int num_seeds=calculate_number_of_frames(nats);
  #pragma omp parallel num_threads(ngpus)
  {
   int th=omp_get_thread_num();
   int nseeds=num_seeds;
   float *rmsd_scores=0;
   cl_int4 sizes,start_points;
   cl_device_id device=gpus[th];
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel tmscore_matrix,tmscore_matrix_rect;
   cl_mem tmscores_buffer,coords41_buffer,coords42_buffer;
   cl_float2 *tmscores;
   cl_int err;
   context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
   queue = clCreateCommandQueue(context, device, 0, &err); 
   program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
   
   if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS){
    fprintf(stderr,"Error building program\n");
    char buf[0x10000];
    clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
    fprintf(stderr,"\n%s\n", buf);
    exit(FALSE);
   }
   //create hosts arrays and buffers - local copies to avoid collisions
   float4 *coords4;
   if (!(coords4 = (float4*)  malloc(pdb4_size * length *sizeof(float4)))) exit(FALSE);
   convert_coords_to_float4 (length,pdb_size,coords,coords4);

   tmscore_matrix = clCreateKernel(program, "tmscore_matrix", &err); 

   
   if(th==0){
    start_rmsd=get_time();
   }
   coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

   if(ngrid >1){
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
    if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_sq))))exit(FALSE); 
   }
   else{
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(cl_float2),NULL, NULL);
    if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_tr))))exit(FALSE);  
   }
   sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
   for (int ni=0;ni<ngrid;ni++){    
    //get lock
    #pragma omp critical
    {
     while(diagonal_lock[ni] && ni <ngrid)ni++;
     if(ni < ngrid) diagonal_lock[ni]=1;
    }
    if(ni < ngrid){
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
     int block_structs=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
     int nwu= block_structs*(block_structs-1)/2;
     int offset=ni*max_structs;
     sizes.z=block_structs;
     sizes.w=nwu;
     start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
     //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d %d %d \n",ni,ni,nwu,block_structs,ngrid,block_structs,block_structs);
     clSetKernelArg(tmscore_matrix, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(tmscore_matrix, 1,sizeof(int),&nseeds);
     clSetKernelArg(tmscore_matrix, 2,sizeof(cl_int4),&start_points);
     clSetKernelArg(tmscore_matrix, 3,sizeof(coords41_buffer),&coords41_buffer);
     clSetKernelArg(tmscore_matrix, 4,sizeof(tmscores_buffer),&tmscores_buffer);
     clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue);
     size_t global,local=nt;
     global=max_nwg*local;
     clEnqueueNDRangeKernel(queue, tmscore_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish( queue);
     clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
     clFinish( queue);

     //output to matrix

     int m=0;
     for(int i=0;i<block_structs-1;i++){
      for(int j=i+1;j<block_structs;j++){
       this->set_matrix(j+offset,i+offset,tmscores[m].x);
       m++;
      }
     }
    }
   }  
   clReleaseKernel(tmscore_matrix);
   clReleaseMemObject(coords41_buffer);
   clReleaseMemObject(tmscores_buffer);
   clReleaseCommandQueue(queue);
   if(ngrid >1){
    queue = clCreateCommandQueue(context, device, 0, &err); 
    tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
    coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
    sizes.x=nats;sizes.y=nats4;   
    for (int ni=0;ni<ngrid-1;ni++){
     for (int nj=ni+1;nj<ngrid;nj++){
      if(ni!=nj){
       //obtain lock
       #pragma omp critical
       {
        while(rectangle_lock[ni*ngrid+nj] && nj <ngrid){
         nj++;
        }
        if(nj<ngrid){
         rectangle_lock[ni*ngrid+nj]=1;
        }
       }
       if(nj < ngrid && ni != nj){
        //rectangular tile
        int block_structs1=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
        int block_structs2=(max_structs<length-nj*max_structs) ? max_structs : length-nj*max_structs;
        int nwu= block_structs1*block_structs2;
        int offset1=ni*max_structs;
        int offset2=nj*max_structs;
        sizes.z=block_structs1;
        sizes.w=nwu;
        //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d %d %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid,block_structs1,block_structs2);

        start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
        clSetKernelArg(tmscore_matrix_rect, 0,sizeof(cl_int4),&sizes); 
        clSetKernelArg(tmscore_matrix_rect, 1,sizeof(int),&nseeds);
        clSetKernelArg(tmscore_matrix_rect, 2,sizeof(cl_int4),&start_points);
        clSetKernelArg(tmscore_matrix_rect, 3,sizeof(coords41_buffer),&coords41_buffer);
        clSetKernelArg(tmscore_matrix_rect, 4,sizeof(coords42_buffer),&coords42_buffer);
        clSetKernelArg(tmscore_matrix_rect, 5,sizeof(tmscores_buffer),&tmscores_buffer);
        clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
        clEnqueueWriteBuffer(queue, coords42_buffer , CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
        clFinish( queue );
        size_t global,local=nt;
        global=max_nwg*local;
        clEnqueueNDRangeKernel(queue, tmscore_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
        clFinish( queue );
        clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
        clFinish( queue);
        int m=0;
        for(int i=0;i<block_structs1;i++){
         for(int j=0;j<block_structs2;j++){
          this->set_matrix(j+offset2,i+offset1,tmscores[m].x);
          m++;
         }
        } 
       }
      }
     }
    }
   }
   #pragma omp barrier  
   if(th==0){
    double end = get_time();
    fprintf(stderr, "%8.3f seconds elapsed for %15.0f.0 RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)length*(double)(length-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)length*(float)(length-1)*0.5f));
    fprintf(stderr,"finished\n");
   } 
   clReleaseMemObject(coords41_buffer);
   if(coords4)free(coords4);
   if(ngrid >1){
    clReleaseMemObject(coords42_buffer);
   }
   clReleaseCommandQueue(queue);
   clReleaseContext(context);
   if(tmscores)free(tmscores);
  }//end parallel
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
  if(diagonal_lock) delete [] diagonal_lock;
  if(rectangle_lock) delete [] rectangle_lock;
 }

 int find_rmsd_matrix_multiple_gpus (int nt, int nwg_per_cu,const char *source,int nats,float *coords,int nthreads){
  //this is called after the multiple gpu is detected
  //divides the work based upon number of compute units detected
  const int max_gpus=10;
  int total_cu=0;
  int max_threads=omp_get_max_threads();
  int omp_nt=(max_threads<nthreads)? max_threads : nthreads;
    //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  //need centered coords for this 
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
 
  //center coords - may not have been done previously
  
  center_all_coords(length,pdb_size/3,coords);
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_platform_id platform;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
  
 //get all devices
  cl_device_id gpus[max_gpus],ncus[max_gpus];
  int ngpus=0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices);
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=(omp_nt >num_of_devices)? num_of_devices : omp_nt;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  //find total number of compute units
  for (int i=0;i<ngpus;i++){
   cl_device_id device;
   device=gpus[i];
   clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL);
   total_cu+=ncu;
  }
  fprintf(stderr,"%d compute units found in %d gpus\n",total_cu,ngpus);
  
  //calculate maximum number of workgroups per compute unit - use smaller gpu size
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  //use same block sizes for now - can optimize for different gpus by adjusting blocksizes - would require storing sizes and offsets in arrays
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers

  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  RMSD_MAX_TR_BLOCK_SIZE) max_structs=RMSD_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(length<max_structs)max_structs=length;

  int ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  //adjust ngrid to ngpus or length
  if (ngrid > 1 || ngpus >1 ){
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(length<max_structs)max_structs=length;
   ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs;
   if(ngrid <ngpus){
    ngrid=ngpus;
    max_structs=(length%ngpus)? length/ngpus+1 : length/ngpus;
   }
  }
  
  //set up lock variables
  int *diagonal_lock=new int [ngrid];
  memset(diagonal_lock,0,sizeof(int)*ngrid);
  int *rectangle_lock=new int [ngrid*ngrid];
  memset(rectangle_lock,0,sizeof(int)*ngrid*ngrid);
  //set diagonals of rectangle lock file to zero
  for (int i=0;i<ngrid;i++)
   rectangle_lock[i*ngrid+i]=1;
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  double start_rmsd;
  #pragma omp parallel num_threads(ngpus)
  {
   int th=omp_get_thread_num();
   float *rmsd_scores=0;
   cl_int4 sizes,start_points;
   cl_device_id device=gpus[th];
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel rmsd_matrix,rmsd_matrix_rect;
   cl_mem coords4_buffer,coords42_buffer,rmsd_buffer;
   cl_int err;
   context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
   queue = clCreateCommandQueue(context, device, 0, &err); 
   program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
   if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS){
    printf("Error building program\n");
    char buf[0x10000];
    clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
    fprintf(stderr,"\n%s\n", buf);
    exit(FALSE);
   }
   //create hosts arrays and buffers - local copies to avoid collisions
   float4 *coords4;
   if (!(coords4 = (float4*)  malloc(pdb4_size * length *sizeof(float4)))) exit(FALSE);
   convert_coords_to_float4 (length,pdb_size,coords,coords4);
   rmsd_matrix = clCreateKernel(program, "rmsd_matrix", &err); 
   rmsd_matrix_rect = clCreateKernel(program, "rmsd_matrix_rect", &err);
   #pragma omp barrier
   if(th==0){
    start_rmsd=get_time();
   }
   coords4_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

   if(ngrid >1){
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(float),NULL, NULL);
    if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_sq))))exit(FALSE);  
   }
   else{
    rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(float),NULL, NULL);
    if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_tr))))exit(FALSE);  
   }
   sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
   for (int ni=0;ni<ngrid;ni++){    
    //get lock
    #pragma omp critical
    {
     while(diagonal_lock[ni] && ni <ngrid)ni++;
     if(ni < ngrid) diagonal_lock[ni]=1;
    }
    if(ni < ngrid){
    //triangular tiles first - calculate the on diagonal submatrices
    //calculate block_size
     int block_structs=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
     int nwu= block_structs*(block_structs-1)/2;
     int offset=ni*max_structs;
     //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d\n",ni,ni,nwu,block_structs,ngrid);

     sizes.z=block_structs;
     sizes.w=nwu;
     start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
     clSetKernelArg(rmsd_matrix, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(rmsd_matrix, 1,sizeof(cl_int4),&start_points);
     clSetKernelArg(rmsd_matrix, 2,sizeof(coords4_buffer),&coords4_buffer);
     clSetKernelArg(rmsd_matrix, 3,sizeof(rmsd_buffer),&rmsd_buffer);
     clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue);
     size_t global,local=nt;
     global=max_nwg*local;
     clEnqueueNDRangeKernel(queue, rmsd_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish(queue);
     clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
     clFinish( queue);
     //output to matrix
     int m=0;
     for(int i=0;i<block_structs-1;i++){
      for(int j=i+1;j<block_structs;j++){
       this->set_matrix(j+offset,i+offset,rmsd_scores[m]);
       m++;
      }
     }
    }
   }
   for (int ni=0;ni<ngrid-1;ni++){
    for (int nj=ni+1;nj<ngrid;nj++){
     if(ni!=nj)
     {
      //obtain lock
      #pragma omp critical
      {
       while(rectangle_lock[ni*ngrid+nj] && nj <ngrid){
        nj++;
       }
       if(nj<ngrid){
        rectangle_lock[ni*ngrid+nj]=1;
       }
      }
      if(nj < ngrid && ni != nj){
       //rectangular tile
       int block_structs1=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
       int block_structs2=(max_structs<length-nj*max_structs) ? max_structs : length-nj*max_structs;
       int nwu= block_structs1*block_structs2;
       int offset1=ni*max_structs;
       int offset2=nj*max_structs;
       sizes.z=block_structs1;
       sizes.w=nwu;
       //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid);

       start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
       clSetKernelArg(rmsd_matrix_rect, 0,sizeof(cl_int4),&sizes); 
       clSetKernelArg(rmsd_matrix_rect, 1,sizeof(cl_int4),&start_points);
       clSetKernelArg(rmsd_matrix_rect, 2,sizeof(coords4_buffer),&coords4_buffer);
       clSetKernelArg(rmsd_matrix_rect, 3,sizeof(coords42_buffer),&coords42_buffer);
       clSetKernelArg(rmsd_matrix_rect, 4,sizeof(rmsd_buffer),&rmsd_buffer);
       clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
       clEnqueueWriteBuffer(queue, coords42_buffer, CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
       clFinish( queue );
       size_t global,local=nt;
       global=max_nwg*local;

       clEnqueueNDRangeKernel(queue, rmsd_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
       clFinish( queue );
       clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
       clFinish( queue);

       int m=0;
       for(int i=0;i<block_structs1;i++){
        for(int j=0;j<block_structs2;j++)
        {
         this->set_matrix(j+offset2,i+offset1,rmsd_scores[m]);
         m++;
        }
       }
      }
     }
    }
   }
   #pragma omp barrier  
   if(th==0)
   {
    double end = get_time();
    fprintf(stderr, "%8.3f seconds elapsed for %15.0f.0 RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)length*(double)(length-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)length*(float)(length-1)*0.5f));
    fprintf(stderr,"finished\n");
   } 
   clReleaseMemObject(coords4_buffer);
   if(coords4)free(coords4);
   if(ngrid >1)
   {
    clReleaseMemObject(coords42_buffer);
   }
   clReleaseCommandQueue(queue);
   clReleaseContext(context);
   if(rmsd_scores)free(rmsd_scores);
  }//end parallel
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
  if(diagonal_lock) delete [] diagonal_lock;
  if(rectangle_lock) delete [] rectangle_lock;
 }

 int find_rmsd_matrix(int cpu_flag,int nt, int nwg_per_cu,const char *source,int nats,float *coords,int gpu_id,int nthreads){
  //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  //need centered coords for this 
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
  float4 *coords4;
  float *rmsd_scores=0;
  double start_rmsd,end;
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_int4 sizes,start_points;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel rmsd_matrix,rmsd_matrix_rect;
  cl_mem coords4_buffer,coords42_buffer,rmsd_buffer;
  cl_int err;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
  
 //get all devices
  cl_device_id cpus[128],gpus[10];
  int ngpus=0;

  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d cpus found\n",num_of_devices);
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_of_devices,cpus, 0);
  }
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=num_of_devices;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  // try to get a supported GPU device
  //test with CPU
  if(cpu_flag)
  { 
   device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"using cpu %d cores found\n",ncu);
  }
  else
  {
   if (!ngpus)
   {
    fprintf(stderr, "no gpu found - running with cpu");
    device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
   }
   else 
   {
    if(gpu_id > ngpus-1){
     fprintf(stderr,"gpu_id error %d gpus found - highest allowed gpu_id is %d - gpu_id of %d given\n",ngpus,ngpus-1,gpu_id);
     exit(FALSE);
    }  
    if(gpu_id ==-1){ 
     if(ngpus >1){
      find_rmsd_matrix_multiple_gpus (nt,nwg_per_cu,source,nats,coords,nthreads);
      return(1);
     }
     device=gpus[0];
    }
    else device=gpus[gpu_id];
    
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"%d compute units found\n",ncu);
   }
  }
  context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
  queue = clCreateCommandQueue(context, device, 0, &err); 
  
  //calculate maximum number of workgroups per compute unit
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  if(cpu_flag)lds*=2;   //memory used is memory to cache coords
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
  {
   printf("Error building program\n");
   char buf[0x10000];
   clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
   fprintf(stderr,"\n%s\n", buf);
   return 1;
  }
  
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers

  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  RMSD_MAX_TR_BLOCK_SIZE) max_structs=RMSD_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(length<max_structs)max_structs=length;

  
  int ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  if (ngrid > 1)
  {
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(length<max_structs)max_structs=length;
   ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs;
  }

  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  //create hosts arrays and buffers
  if (!(coords4 = (float4*)  malloc(pdb4_size * length *sizeof(float4)))) exit(FALSE);
  
#ifdef OPENMP  
  if(nthreads >1){
   int max_threads=omp_get_max_threads();
   int nt=(max_threads<nthreads)? max_threads : nthreads;
   if(nt >= length) nt=1;
   #pragma omp parallel num_threads(nt)  
   {
    int th=omp_get_thread_num();
    if(th<nt){
     int offset=th*(length/nt);
     int tlength=(th<nthreads-1)? length/nt : length-offset;
     float *tcoords=&(coords[offset*pdb_size]);
     float4 *tcoords4=&(coords4[offset*pdb4_size]);
     center_all_coords(tlength,pdb_size/3,tcoords);
     convert_coords_to_float4 (tlength,pdb_size,tcoords,tcoords4);    
    }
   }
  }
  else{ 
   center_all_coords(length,pdb_size/3,coords);
   convert_coords_to_float4 (length,pdb_size,coords,coords4);
  }
#else
  center_all_coords(length,pdb_size/3,coords);
  convert_coords_to_float4 (length,pdb_size,coords,coords4);
#endif

  start_rmsd = get_time();  

  rmsd_matrix = clCreateKernel(program, "rmsd_matrix", &err);
  rmsd_matrix_rect = clCreateKernel(program, "rmsd_matrix_rect", &err);
  end = get_time();  
  fprintf(stderr, "%8.3f seconds elapsed for program generation\n",end-start_rmsd);
  start_rmsd = get_time();  
  coords4_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

  if(ngrid >1)
  {
   coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(float),NULL, NULL);
   if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_sq))))exit(FALSE);  
  }
  else
  {
   rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(float),NULL, NULL);
   if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_tr))))exit(FALSE);  
  }
  sizes.x=nats;sizes.y=nats4;
  //indices need to be worked out
  for (int ni=0;ni<ngrid;ni++)
  {
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
   int block_structs=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
   int nwu= block_structs*(block_structs-1)/2;
   int offset=ni*max_structs;
   //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d\n",ni,ni,nwu,block_structs,ngrid);

   sizes.z=block_structs;
   sizes.w=nwu;
   start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
   clSetKernelArg(rmsd_matrix, 0,sizeof(cl_int4),&sizes); 
   clSetKernelArg(rmsd_matrix, 1,sizeof(cl_int4),&start_points);
   clSetKernelArg(rmsd_matrix, 2,sizeof(coords4_buffer),&coords4_buffer);
   clSetKernelArg(rmsd_matrix, 3,sizeof(rmsd_buffer),&rmsd_buffer);
   clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
   clFinish( queue);
   size_t global,local=nt;
   global=max_nwg*local;
   clEnqueueNDRangeKernel(queue, rmsd_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
   clFinish(queue);
   clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
   clFinish( queue);
   //output to matrix


  int m=0;
  for(int i=0;i<block_structs-1;i++)
   for(int j=i+1;j<block_structs;j++){
    this->set_matrix(j+offset,i+offset,rmsd_scores[m]);
    m++;
   }

  }
  for (int ni=0;ni<ngrid-1;ni++)
   for (int nj=ni+1;nj<ngrid;nj++)
    if(ni!=nj)
    {
     //rectangular tile
     int block_structs1=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
     int block_structs2=(max_structs<length-nj*max_structs) ? max_structs : length-nj*max_structs;
     int nwu= block_structs1*block_structs2;
     int offset1=ni*max_structs;
     int offset2=nj*max_structs;
     sizes.z=block_structs1;
     sizes.w=nwu;
     //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid);

     start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
     clSetKernelArg(rmsd_matrix_rect, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(rmsd_matrix_rect, 1,sizeof(cl_int4),&start_points);
     clSetKernelArg(rmsd_matrix_rect, 2,sizeof(coords4_buffer),&coords4_buffer);
     clSetKernelArg(rmsd_matrix_rect, 3,sizeof(coords42_buffer),&coords42_buffer);
     clSetKernelArg(rmsd_matrix_rect, 4,sizeof(rmsd_buffer),&rmsd_buffer);
     clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clEnqueueWriteBuffer(queue, coords42_buffer, CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue );
     size_t global,local=nt;
     global=max_nwg*local;

     clEnqueueNDRangeKernel(queue, rmsd_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish( queue );
     clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
     clFinish( queue);

     int m=0;
     for(int i=0;i<block_structs1;i++)
      for(int j=0;j<block_structs2;j++)
      {
       this->set_matrix(j+offset2,i+offset1,rmsd_scores[m]);
       m++;
      }
    }
  end = get_time();  
  fprintf(stderr, "%8.3f seconds elapsed for %15.0f.0 RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)length*(double)(length-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)length*(float)(length-1)*0.5f));
  fprintf(stderr,"finished\n");  
  clReleaseMemObject(coords4_buffer);
  clReleaseProgram(program);
  if(ngrid >1)
  {
   clReleaseMemObject(coords42_buffer);
  }
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  if(coords4)free(coords4);
  if(rmsd_scores)free(rmsd_scores);
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
 }
#endif      
 T **tmatrix; //leave public for fast direct access if necessary
};

template <>
class triangular_matrix <unsigned char>{
 //unsigned char uses same routines
 public:
  int length;
  int element_size;
	 float min_value,step_size,max_value;
  float inv_step_size;
  bool greater_is_better;
  unsigned char **tmatrix; //leave public for fast direct access if necessary
  triangular_matrix<unsigned char>(int input_nstructs,float min_value,float step_size,bool greater_is_better){
   length=input_nstructs;
   element_size=1;
   tmatrix=new unsigned char *[length];
   for(int i=1;i<length;i++)
    tmatrix[i]= new unsigned char [i];
   this->min_value= min_value;
   this->step_size=step_size;
   this ->max_value=min_value+255.0f*this->step_size;
   this ->greater_is_better=greater_is_better;
   inv_step_size=1.0f/this->step_size;
  }
  triangular_matrix <unsigned char>(const triangular_matrix <unsigned char> &A) : min_value(A.min_value),max_value(A.max_value),step_size(A.step_size),greater_is_better(A.greater_is_better){
   tmatrix=new unsigned char* [length];
   for(int i=1;i<length;i++){
    tmatrix[i]= new unsigned char [i];
    memmove(tmatrix[i],A.tmatrix[i],i);
   }
   inv_step_size=1.0f/step_size;
  }
  triangular_matrix <unsigned char> &operator = (const triangular_matrix  <unsigned char> &rhs){
   if(this != &rhs){
    triangular_matrix <unsigned char>::operator=(rhs);
    for(int i=1;i<length;i++){
     tmatrix[i]= new unsigned char [i];
     memmove(tmatrix[i],rhs.tmatrix[i],i);
    }
   }
   min_value= rhs.min_value;
   max_value= rhs.max_value;
   step_size= rhs.step_size;
   inv_step_size=1.0f/rhs.step_size;
   greater_is_better=rhs.greater_is_better;
   return(*this);
  }
 ~triangular_matrix(){
   for(int i=1;i<length;i++)
    if(tmatrix[i])delete [] tmatrix[i];
   if(tmatrix)delete [] tmatrix;
  } 

  void adjust_max_min(){
   float max,min;
   max=get_matrix(1,0);
   min=get_matrix(1,0);
   for(int i=1;i<length;i++){
    for(int j=0;j<i;j++){
     float value=get_matrix(i,j);
     max=(value>max)? value :max;
     min=(min>value)? value :min;
    }
   }
   min_value=min;
   step_size=(max-min)/256.0f;
   inv_step_size=1.0f/step_size;
   max_value=min_value+255.5f*step_size;
  }    
  
  void adjust_max_min(int *map, int nmap){
   float max,min;
   max=get_matrix(map[1],map[0]);
   min=get_matrix(map[1],map[0]);
   for(int i=1;i<nmap;i++){
    int a=map[i];
    for(int j=0;j<i;j++){
     int b=map[j];
     float value=get_matrix(a,b);
     max=(value>max)? value :max;
     min=(min>value)? value :min;
    }
   }
   min_value=min;
   step_size=(max-min)/256.0f;
   inv_step_size=1.0f/step_size;
   max_value=min_value+255.5f*step_size;
  }
 
  void set_matrix(int i, int j,float value) {
   //careful with nans - let them fall through
   if(value >max_value){
    if(j>i)tmatrix[j][i]=(unsigned char)255;
    else tmatrix[i][j]=(unsigned char) 255;
   }
   else if(value < max_value && value >min_value){
    int index= (int)((value-min_value)* inv_step_size +0.5f);
    if(index > 255) index=255;
    if(j>i)tmatrix[j][i]=(unsigned char)index;
    else tmatrix[i][j]=(unsigned char) index;
   }
   else if (value <=min_value){ 
    if(j>i)tmatrix[j][i]=(unsigned char)0;
    else tmatrix[i][j]=(unsigned char) 0;
   }
   else{//nan case -return worst_value
    if (greater_is_better){
     if(j>i)tmatrix[j][i]=(unsigned char)0 ;
     else tmatrix[i][j]=(unsigned char) 0;
    }
    else{
     if(j>i)tmatrix[j][i]=(unsigned char)255;
     else tmatrix[i][j]=(unsigned char) 255;
    }
   } 
  }
  

  void set_matrix (int i, int j,unsigned char value){
   if(j>i)tmatrix[j][i]=value;
   else tmatrix[i][j]=value;
  }
  unsigned char get_matrix_fast (int i, int j) {
   return(tmatrix[i][j]);
  }
	 float get_matrix (int i, int j) {
   unsigned char uchar=(j>i)? tmatrix[j][i] : tmatrix[i][j];
   return((float)((unsigned int)uchar)*step_size+min_value);
  }
  unsigned char get_native_matrix(int i, int j){
   if(j>i)return(tmatrix[j][i]);
   return(tmatrix[i][j]);
  }
  unsigned char get_native_matrix_fast (int i, int j){
   return(tmatrix[i][j]);
  }  
  //these write to float - write compact writes to compact
  void write_matrix_to_binary_file(FILE *fp){
   for(int i=1;i<length;i++){
    for(int j=0;j<i;j++){
     float fvalue=(float) get_matrix(i,j);
     fwrite(&fvalue,sizeof(float),1,fp);
    }
   }
  }  
  void write_matrix_to_binary_file(FILE *fp,int *map,int nmap){
   for(int i=1;i<nmap;i++){
    int a=map[i];
    for(int j=0;j<i;j++){
     int b=map[j];
     float fvalue=(float) get_matrix(a,b);
     fwrite(&fvalue,sizeof(float),1,fp);
    }
   }
  }
  void write_matrix_to_compact_file(FILE *fp){
   adjust_max_min();
   fwrite(&length,sizeof(int),1,fp);
   fwrite(&min_value,sizeof(float),1,fp);
   fwrite(&step_size,sizeof(float),1,fp);
   for(int i=1;i<length;i++){
    fwrite(tmatrix[i],1,i,fp);
   }
  } 
  void write_matrix_to_compact_file(FILE *fp,int *map,int nmap){
    adjust_max_min();
   fwrite(&length,sizeof(int),1,fp);
   fwrite(&min_value,sizeof(float),1,fp);
   fwrite(&step_size,sizeof(float),1,fp);
   for(int i=1;i<nmap;i++){
    for(int j=0;j<i;j++){
     int a=map[i];
     int b=map[j];  
     if(a>b)
      fwrite(&(tmatrix[a][b]),1,1,fp);

     else
      fwrite(&(tmatrix[b][a]),1,1,fp);    ;
    }
   }
  }  

  int read_matrix_from_binary_file(FILE *fp,int *inverse_map,int ninverse_map){//assumes float
   int nread=0;
   int nseek=0;
   int size=sizeof(float);
   int my_length;
   float min,max;
   //two pass - first pass finds min,max
   for(int i=1;i<ninverse_map;i++){
    int a=inverse_map[i];
    if(a >=0){
     for(int j=0;j<i;j++){
      int b=inverse_map[j];
      if(b>=0){
       float value;
       fseek(fp,nseek*size,SEEK_CUR);
       fread(&value,size,1,fp);
       if(!nread || value >max)max=value;
       if(!nread || value <min)min=value; 
       nseek=0;
       nread++;
      }
      else nseek++;
     }
    }
    else{
     nseek+=i;
    }  
   }
   min_value=min;
   step_size=(max-min)/256.0;
   inv_step_size=1.0f/step_size;
   max_value=min_value+255.5f*step_size;   
   rewind(fp);
   nseek=0;
   for(int i=1;i<ninverse_map;i++){
    int a=inverse_map[i];
    if(a >=0){
     for(int j=0;j<i;j++){
      int b=inverse_map[j];
      if(b>=0){
       float value;
       fseek(fp,nseek*size,SEEK_CUR);
       fread(&value,size,1,fp);
       set_matrix(a,b,value);;
       nseek=0;
       nread++;
      }
      else nseek++;
     }
    }
    else{
     nseek+=i;
    }  
   }
   return(nread);
  }
  
  int read_matrix_from_binary_file(FILE *fp){//assumes float
   int nread=0;
   int my_length;
   //two pass - first pass finds min,max
   float min,max;
   fprintf(stderr,"finding min value\n");
   for(int i=1;i<length;i++){
    if(i%1000 == 0) fprintf(stderr,"%d rows read\n",i);
    for(int j=0;j<i;j++){
     float value;
     fread(&value,sizeof(float),1,fp);
     if(!nread || value >max)max=value;
     if(!nread || value <min)min=value;
     nread++;
    }
   }
   min_value=min;
   step_size=(max-min)/256.0;
   inv_step_size=1.0f/step_size;
   max_value=min_value+255.5f*step_size;   
   rewind(fp);
   fprintf(stderr,"rewound file\n");
   for(int i=1;i<length;i++){    
    if(i%1000 == 0) fprintf(stderr,"%d rows read\n",i);
    for(int j=0;j<i;j++){
     float value;
     fread(&value,sizeof(float),1,fp);
     set_matrix(i,j,value);
    }
   }
//   return(nread);
   return(1);
  }
   
  int read_matrix_from_compact_file(FILE *fp,int *inverse_map,int ninverse_map){
   int nread=0;
   int nseek=0;
   int my_length;
   fread(&my_length,sizeof(int),1,fp);
   fread(&min_value,sizeof(float),1,fp);
   fread(&step_size,sizeof(float),1,fp);
   for(int i=1;i<ninverse_map;i++){
    int a=inverse_map[i];
    if(a >=0){
     for(int j=0;j<i;j++){
      int b=inverse_map[j];
      if(b>=0){
       unsigned char value;
       fseek(fp,nseek,SEEK_CUR);
       fread(&value,1,1,fp);
       set_matrix(a,b,value);
       nread++;
       nseek=0;
      }
      else nseek++;
     }
    }
    else{
     nseek+=i;
    }  
   }   
   return(nread);
  } 
  int read_matrix_from_compact_file(FILE *fp){
   int nread=0;
   int my_length;
   fread(&my_length,sizeof(int),1,fp);
   fread(&min_value,sizeof(float),1,fp);
   fread(&step_size,sizeof(float),1,fp);
   for(int i=1;i<length;i++){ 
    fread(tmatrix[i],element_size,i,fp);;
    nread+=i;
   }
   return(nread);
  }
  //these assume float - really should make them so that they accept any type
  int read_matrix_from_text_file(FILE *fp,int *inverse_map){
   char line[LINE_LENGTH];
   int nread=0;
   int my_length;
   float min,max; 
   while (fgets(line, LINE_LENGTH, fp)){
    float value;
    int m,n,i,j;;
    sscanf (line, "%d %d %f",&m,&n,&value);
    i=inverse_map[m];
    j=inverse_map[n];
    if(i>=0 && i<length && j>=0 && j <length){
     if(!nread || value >max)max=value;
     if(!nread || value <min)min=value;
     nread++;   
    }
   }
   min_value=min;
   step_size=(max-min)/256.0;
   inv_step_size=1.0f/step_size;
   max_value=min_value+255.5f*step_size;   
   rewind(fp); 
   while (fgets(line, LINE_LENGTH, fp)){
    float value;
    int m,n,i,j;;
    sscanf (line, "%d %d %f",&m,&n,&value);
    adjust_max_min();
    i=inverse_map[m];
    j=inverse_map[n];
    if(i<length && j <length){
     set_matrix(i,j,value);
    }
   }
   return(nread);
  }
  int read_matrix_from_text_file(FILE *fp){
   char line[LINE_LENGTH];
   int nread=0;
    //two pass - first pass finds min,max
   float min,max;
   while (fgets(line, LINE_LENGTH, fp)){
    float value;
    int i,j;
    sscanf (line, "%d %d %f",&i,&j,&value);
    if(i>=0 && i<length && j>=0 && j <length){
     if(!nread || value >max)max=value;
     if(!nread || value <min)min=value; 
     nread++;       
    }
   }
   min_value=min;
   step_size=(max-min)/256.0;
   inv_step_size=1.0f/step_size;
   max_value=min_value+255.5f*step_size;   
   rewind(fp);
   while (fgets(line, LINE_LENGTH, fp)){
    float value;
    int i,j;
    sscanf (line, "%d %d %f",&i,&j,&value);
    if(i>=0 && i<length && j>=0 && j <length){
     set_matrix(i,j,value);
    }
   }
   return(nread);
  }

  void write_matrix_to_text_file (FILE *fp){//writes to float
   for (int i=1;i<length;i++)
    for(int j=0;j<i;j++)
     fprintf(fp,"%d %d %f\n",i,j,get_matrix(i,j));
  }   
  void write_matrix_to_text_file (FILE *fp,int *map, int nmap){
   for (int m=1;m<nmap;m++){
    for(int n=0;n<m;n++){
     int i=map[m];
     int j=map[n];
     fprintf(fp,"%d %d %f\n",i,j,get_matrix(i,j));
    }
   }   
  } 
 void generate_matrix(int nats,int pdb_size,float *coords,_compute_type compute,_metric_type score_type,_simd_type simd_type,int nthreads,int gpu_id){
 if(compute==cCPU){
   if(score_type==RMSD){
    if(simd_type==SCALAR_){
     //no SSE routine
#ifdef OPENMP    
     int max_threads=omp_get_max_threads();
     nthreads=(max_threads<nthreads)? max_threads : nthreads;
#else
     nthreads=1;
#endif          
     rmsd_cpu_par(nthreads,nats,length,coords,this); 
    }

#ifdef SSE2
    else if(simd_type ==SSE2_){//use SSE2 routine 
#ifdef OPENMP     
     int max_threads=omp_get_max_threads();
     nthreads=(max_threads<nthreads)? max_threads : nthreads;
#endif
     rmsd_sse2_par(nthreads,nats,length,coords,this);  
    }
#endif
#ifdef SSE3
    else if(simd_type ==SSE3_){//use SSE2 routine 
#ifdef OPENMP     
     int max_threads=omp_get_max_threads();
     nthreads=(max_threads<nthreads)? max_threads : nthreads;
#endif
     rmsd_sse3_par(nthreads,nats,length,coords,this);  
    }
#endif
#ifdef AVX
    else if(simd_type == AVX_){//use avx routine 
#ifdef OPENMP     
     int max_threads=omp_get_max_threads();
     nthreads=(max_threads<nthreads)? max_threads : nthreads;
#endif   
     rmsd_avx_par(nthreads,nats,length,coords,this);     
    }
#endif
   }//end rmsd
   else if (score_type == TMSCORE){ //TMscore
    float R[3][3],t[3];
    if(simd_type == SCALAR_){
#ifdef OPENMP     
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;     
    #pragma omp parallel for num_threads (nthreads) schedule (dynamic)
#endif
     for (int i=1;i<length;i++){
      for (int j=0;j<i;j++){      
       this->set_matrix(i,j,tmscore_rmsd_cpu(nats,&(coords[i*pdb_size]),&(coords[j*pdb_size]),R,t,0));
      }
     }
    }
#ifdef SSE2
    else if (simd_type == SSE2_){
     float *x=0,*y=0,*z=0;
     shuffle_tmscore_coords_soa_sse(length,nats,coords,&x,&y,&z);
     int anats=(nats%4)? (nats/4)*4+4 : nats;     
#ifdef OPENMP     
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;     
    #pragma omp parallel for num_threads (nthreads) schedule (dynamic)
#endif
     for (int i=1;i<length;i++){
      for (int j=0;j<i;j++){     
       this->set_matrix(i,j,tmscore_cpu_soa_sse2(nats,&(x[i*anats]),&(y[i*anats]),&(z[i*anats]),&(x[j*anats]),&(y[j*anats]),&(z[j*anats]),0,0,0));
      }
     }
     if(x) free (x);     
     if(y) free (y);  
     if(z) free (z);      
    }
#endif
#ifdef SSE3
    else if (simd_type == SSE3_){
     float *x=0,*y=0,*z=0;
     shuffle_tmscore_coords_soa_sse(length,nats,coords,&x,&y,&z);
     int anats=(nats%4)? (nats/4)*4+4 : nats;     
#ifdef OPENMP     
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;     
    #pragma omp parallel for num_threads (nthreads) schedule (dynamic)
#endif
     for (int i=1;i<length;i++){
      for (int j=0;j<i;j++){     
       this->set_matrix(i,j,tmscore_cpu_soa_sse3(nats,&(x[i*anats]),&(y[i*anats]),&(z[i*anats]),&(x[j*anats]),&(y[j*anats]),&(z[j*anats]),0,0,0));
      }
     }
     if(x) free (x);     
     if(y) free (y);  
     if(z) free (z);      
    } 
#endif
#ifdef AVX   
    else if (simd_type == AVX_){
     float *x=0,*y=0,*z=0;
     int unat8=(nats%8)? (nats/8+1)*8 : nats;
     shuffle_tmscore_coords_soa_avx(length,nats,coords,&x,&y,&z);
#ifdef OPENMP     
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;     
    #pragma omp parallel for num_threads (nthreads) schedule (dynamic)
#endif
     for (int i=1;i<length;i++){
      for (int j=0;j<i;j++){    
       this->set_matrix(i,j,tmscore_cpu_soa_avx(nats,&(x[i*unat8]),&(y[i*unat8]),&(z[i*unat8]),&(x[j*unat8]),&(y[j*unat8]),&(z[j*unat8]),0,0,0));
      }
     }         
     if(x) free (x);     
     if(y) free (y);  
     if(z) free (z);  
    }
#endif
  }//end TMSCORE
  else{
   fprintf(stderr,"unrecognized score type %d\n",score_type);
   exit(FALSE);
  }   
 }//end CPU

#ifdef GPU 
  else if(compute == cGPU){
   if (score_type== RMSD)this->find_rmsd_matrix(0,64,0,grmsdcl_location,nats,coords,gpu_id,nthreads);
   else if(score_type==TMSCORE)this->find_tmscore_matrix(0,64,0,gtmscorecl_location,nats,coords,gpu_id,nthreads);
   else{
    fprintf(stderr,"unrecognized score type %d\n",score_type);
    exit(FALSE);
   } 
  }
#endif
  else{     
   fprintf(stderr,"unrecognized compute type %d\n",compute);
   exit(FALSE);  
  }
 } 
#ifdef GPU 
  int find_tmscore_matrix(int cpu_flag,int nt, int nwg_per_cu,const char *source,int nats,float *coords,int gpu_id,int nthreads){
  //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
  double start_program=0,start_rmsd=0,end=0;
  float4 *coords4;

  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_int4 sizes,start_points;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel tmscore_matrix,tmscore_matrix_rect;
  cl_mem tmscores_buffer,coords41_buffer,coords42_buffer;
  cl_float2 *tmscores;
  cl_int err;
  cl_uint ncu,num_of_devices,numPlatforms;
  clGetPlatformIDs( 1, &platform, NULL ); 
  //get all devices
  cl_device_id cpus[10],gpus[10];
  int ngpus=0;
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d cpus found\n",num_of_devices);
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_of_devices,cpus, 0);
  }
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=num_of_devices;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  // try to get a supported GPU device
  //test with CPU
  if(cpu_flag)
  { 
   device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"using cpu %d cores found\n",ncu);
  }
  else{
   if (!ngpus)
   {
    fprintf(stderr, "no gpu found - running with cpu");
    device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
   }
   else 
   {
    if(gpu_id > ngpus-1){
     fprintf(stderr,"gpu_id error %d gpus found - highest allowed gpu_id is %d - gpu_id of %d given\n",ngpus,ngpus-1,gpu_id);
     exit(FALSE);
    }  
    if(gpu_id ==-1){ 
     if(ngpus >1){
      find_tmscore_matrix_multiple_gpus (nt,nwg_per_cu,source,nats,coords,nthreads);
      return(1);
     }
     device=gpus[0];
    }
    else device=gpus[gpu_id];
    
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"%d compute units found\n",ncu);
   }
  }
  context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
  queue = clCreateCommandQueue(context, device, 0, &err);  
  //calculate maximum number of workgroups per compute unit
  //for gpus this depends on the memory available
  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  if(cpu_flag)lds*=2; 
   //memory used is memory to cache coords plus memory for the alignment and reduction - this is different from the simple tmscore where the coords memory is used once
   //there is a bug in OpenCL with local memory declared in a block that may or may not be freed... 
  int mem_per_wg=6*nats4*sizeof(float4)+nt*sizeof(float2);
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif
  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"creating coord buffers\n");
  //create hosts arrays and buffers
  if (!(coords4 = (float4*)  malloc(pdb4_size * length *sizeof(float4)))) exit(FALSE);
  convert_coords_to_float4 (length,pdb_size,coords,coords4);
  fprintf(stderr,"created  buffers\n");

  start_rmsd = get_time();  
  program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
  if ((err=clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS))
  {
   fprintf(stderr,"error %d %s\n",err,print_cl_errstring(err));
   char buf[0x10000];
   clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
   fprintf(stderr,"\n%s\n", buf);
   return 1;
  }
  end = get_time();
  fprintf(stderr,"creating kernel\n");
  tmscore_matrix = clCreateKernel(program, "tmscore_matrix", &err); 
  fprintf(stderr, "%8.3f seconds elapsed for program generation\n",end-start_rmsd);
  start_rmsd = get_time();  

  start_rmsd = get_time();  
  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  TM_MAX_TR_BLOCK_SIZE) max_structs=TM_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(length<max_structs)max_structs=length;

   //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix

  int ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  if (ngrid > 1)
  {
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(cl_float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > TM_MAX_SQ_BLOCK_SIZE) max_structs=TM_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(length<max_structs)max_structs=length;
   ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs;
  }
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  fprintf(stderr,"creating openCL coords buff %d\n",max_structs*pdb4_size * sizeof(float4)); 
  coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
  if(ngrid >1)
  {
   fprintf(stderr,"creading square buffers\n");
 //  tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
 //  coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
   if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_sq))))exit(FALSE);  
  }
  else
  {
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(cl_float2),NULL, NULL);
   if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_tr))))exit(FALSE);  
  }
  fprintf(stderr,"calculating frames\n");
  int nseeds=calculate_number_of_frames(nats);

  sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
  for (int ni=0;ni<ngrid;ni++)
  {
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
   int block_structs=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
   int nwu= block_structs*(block_structs-1)/2;
   int offset=ni*max_structs;
   sizes.z=block_structs;
   sizes.w=nwu;
   start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
   //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d %d %d \n",ni,ni,nwu,block_structs,ngrid,block_structs,block_structs);

   clSetKernelArg(tmscore_matrix, 0,sizeof(cl_int4),&sizes); 
   clSetKernelArg(tmscore_matrix, 1,sizeof(int),&nseeds);
   clSetKernelArg(tmscore_matrix, 2,sizeof(cl_int4),&start_points);
   clSetKernelArg(tmscore_matrix, 3,sizeof(coords41_buffer),&coords41_buffer);
   clSetKernelArg(tmscore_matrix, 4,sizeof(tmscores_buffer),&tmscores_buffer);
   clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
   clFinish( queue);
   size_t global,local=nt;
   global=max_nwg*local;
   clEnqueueNDRangeKernel(queue, tmscore_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
   clFinish(queue);
   clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
   clFinish( queue);

   //output to matrix

  int m=0;
  for(int i=0;i<block_structs-1;i++)
   for(int j=i+1;j<block_structs;j++){
    this->set_matrix(j+offset,i+offset,tmscores[m].x);
    m++;
   }
  }

  fprintf(stderr,"releasing resources from first kernel\n");
  clReleaseKernel(tmscore_matrix);
  clReleaseMemObject(coords41_buffer);
  clReleaseMemObject(tmscores_buffer);
  clReleaseCommandQueue(queue);
  
  fprintf(stderr,"allocating resources for second kernel\n");
  if(ngrid >1)
  {
   fprintf(stderr,"creating new queue\n");
   queue = clCreateCommandQueue(context, device, 0, &err); 
   fprintf(stderr,"creating new kernel\n");
   tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
   fprintf(stderr,"creating buffers\n");
   coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
   sizes.x=nats;sizes.y=nats4;
  
   for (int ni=0;ni<ngrid-1;ni++)
    for (int nj=ni+1;nj<ngrid;nj++)
     if(ni!=nj)
     {
      //rectangular tile
      int block_structs1=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
      int block_structs2=(max_structs<length-nj*max_structs) ? max_structs : length-nj*max_structs;
      int nwu= block_structs1*block_structs2;
      int offset1=ni*max_structs;
      int offset2=nj*max_structs;
      sizes.z=block_structs1;
      sizes.w=nwu;
      //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d %d %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid,block_structs1,block_structs2);

      start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
      clSetKernelArg(tmscore_matrix_rect, 0,sizeof(cl_int4),&sizes); 
      clSetKernelArg(tmscore_matrix_rect, 1,sizeof(int),&nseeds);
      clSetKernelArg(tmscore_matrix_rect, 2,sizeof(cl_int4),&start_points);
      clSetKernelArg(tmscore_matrix_rect, 3,sizeof(coords41_buffer),&coords41_buffer);
      clSetKernelArg(tmscore_matrix_rect, 4,sizeof(coords42_buffer),&coords42_buffer);
      clSetKernelArg(tmscore_matrix_rect, 5,sizeof(tmscores_buffer),&tmscores_buffer);
      clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
      clEnqueueWriteBuffer(queue, coords42_buffer , CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
      clFinish( queue );
      size_t global,local=nt;
      global=max_nwg*local;
      clEnqueueNDRangeKernel(queue, tmscore_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
      clFinish( queue );
      clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
      clFinish( queue);
      int m=0;
      for(int i=0;i<block_structs1;i++)
       for(int j=0;j<block_structs2;j++)
       {
        this->set_matrix(j+offset2,i+offset1,tmscores[m].x);
        m++;
       }
     }
   clReleaseMemObject(coords41_buffer);
   clReleaseMemObject(tmscores_buffer);
   clReleaseProgram(program);
   clReleaseMemObject(coords42_buffer);
   clReleaseKernel(tmscore_matrix_rect);
   clReleaseCommandQueue(queue);
  }
  fprintf(stderr,"finished\n");  
  end = get_time();  
  //fprintf(stderr, "%8.3f seconds elapsed for %d TM-scores at %8.3f ms per TM-score\n",end-start_rmsd,length*(length-1)/2,(float)((end-start_rmsd)*1000)/(float)(length*(length-1)/2));
  clReleaseContext(context);
  if(coords4)free(coords4);
  if(tmscores)free(tmscores);
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
 }
 int find_tmscore_matrix_multiple_gpus (int nt, int nwg_per_cu,const char *source,int nats,float *coords,int nthreads){
  //this is called after the multiple gpu is detected
  //divides the work based upon number of compute units detected
  const int max_gpus=10;
  int total_cu=0;
  int max_threads=omp_get_max_threads();
  int omp_nt=(max_threads<nthreads)? max_threads : nthreads;
    //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  //need centered coords for this 
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
 
  //center coords - may not have been done previously
  center_all_coords(length,pdb_size/3,coords);
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_platform_id platform;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
  
 //get all devices
  cl_device_id gpus[max_gpus],ncus[max_gpus];
  int ngpus=0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices);
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=(omp_nt >num_of_devices)? num_of_devices : omp_nt;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  //find total number of compute units
  for (int i=0;i<ngpus;i++){
   cl_device_id device;
   device=gpus[i];
   clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL);
   total_cu+=ncu;
  }
  fprintf(stderr,"%d compute units found in %d gpus\n",total_cu,ngpus);
  
  //calculate maximum number of workgroups per compute unit - use smaller gpu size
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  //use same block sizes for now - can optimize for different gpus by adjusting blocksizes - would require storing sizes and offsets in arrays
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers

  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  TM_MAX_TR_BLOCK_SIZE) max_structs=TM_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(length<max_structs)max_structs=length;

  int ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  //adjust ngrid to ngpus or length
  if (ngrid > 1 || ngpus >1 ){
   //recalculate with MAX_ELEMENTS/2
   int min_split=ngpus*2;
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(length<max_structs)max_structs=length;
   ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs;
   if(ngrid <min_split){
    ngrid=min_split;
    max_structs=(length%min_split)? length/min_split+1 : length/min_split;
   }
  }
  
  //set up lock variables
  int *diagonal_lock=new int [ngrid];
  memset(diagonal_lock,0,sizeof(int)*ngrid);
  int *rectangle_lock=new int [ngrid*ngrid];
  memset(rectangle_lock,0,sizeof(int)*ngrid*ngrid);
  //set diagonals of rectangle lock file to zero
  for (int i=0;i<ngrid;i++)
   rectangle_lock[i*ngrid+i]=1;
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  double start_rmsd;  
  fprintf(stderr,"calculating frames\n");
  int num_seeds=calculate_number_of_frames(nats);
  #pragma omp parallel num_threads(ngpus)
  {
   int th=omp_get_thread_num();
   int nseeds=num_seeds;
   float *rmsd_scores=0;
   cl_int4 sizes,start_points;
   cl_device_id device=gpus[th];
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel tmscore_matrix,tmscore_matrix_rect;
   cl_mem tmscores_buffer,coords41_buffer,coords42_buffer;
   cl_float2 *tmscores;
   cl_int err;
   context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
   queue = clCreateCommandQueue(context, device, 0, &err); 
   program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
   
   if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS){
    fprintf(stderr,"Error building program\n");
    char buf[0x10000];
    clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
    fprintf(stderr,"\n%s\n", buf);
    exit(FALSE);
   }
   //create hosts arrays and buffers - local copies to avoid collisions
   float4 *coords4;
   if (!(coords4 = (float4*)  malloc(pdb4_size * length *sizeof(float4)))) exit(FALSE);
   convert_coords_to_float4 (length,pdb_size,coords,coords4);

   tmscore_matrix = clCreateKernel(program, "tmscore_matrix", &err); 

   
   if(th==0){
    start_rmsd=get_time();
   }
   coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

   if(ngrid >1){
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
    if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_sq))))exit(FALSE); 
   }
   else{
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(cl_float2),NULL, NULL);
    if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_tr))))exit(FALSE);  
   }
   sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
   for (int ni=0;ni<ngrid;ni++){    
    //get lock
    #pragma omp critical
    {
     while(diagonal_lock[ni] && ni <ngrid)ni++;
     if(ni < ngrid) diagonal_lock[ni]=1;
    }
    if(ni < ngrid){
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
     int block_structs=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
     int nwu= block_structs*(block_structs-1)/2;
     int offset=ni*max_structs;
     sizes.z=block_structs;
     sizes.w=nwu;
     start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
     //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d %d %d \n",ni,ni,nwu,block_structs,ngrid,block_structs,block_structs);
     clSetKernelArg(tmscore_matrix, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(tmscore_matrix, 1,sizeof(int),&nseeds);
     clSetKernelArg(tmscore_matrix, 2,sizeof(cl_int4),&start_points);
     clSetKernelArg(tmscore_matrix, 3,sizeof(coords41_buffer),&coords41_buffer);
     clSetKernelArg(tmscore_matrix, 4,sizeof(tmscores_buffer),&tmscores_buffer);
     clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue);
     size_t global,local=nt;
     global=max_nwg*local;
     clEnqueueNDRangeKernel(queue, tmscore_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish( queue);
     clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
     clFinish( queue);

     //output to matrix

     int m=0;
     for(int i=0;i<block_structs-1;i++){
      for(int j=i+1;j<block_structs;j++){
       this->set_matrix(j+offset,i+offset,tmscores[m].x);
       m++;
      }
     }
    }
   }  
   clReleaseKernel(tmscore_matrix);
   clReleaseMemObject(coords41_buffer);
   clReleaseMemObject(tmscores_buffer);
   clReleaseCommandQueue(queue);
   if(ngrid >1){
    queue = clCreateCommandQueue(context, device, 0, &err); 
    tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
    coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
    sizes.x=nats;sizes.y=nats4;   
    for (int ni=0;ni<ngrid-1;ni++){
     for (int nj=ni+1;nj<ngrid;nj++){
      if(ni!=nj){
       //obtain lock
       #pragma omp critical
       {
        while(rectangle_lock[ni*ngrid+nj] && nj <ngrid){
         nj++;
        }
        if(nj<ngrid){
         rectangle_lock[ni*ngrid+nj]=1;
        }
       }
       if(nj < ngrid && ni != nj){
        //rectangular tile
        int block_structs1=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
        int block_structs2=(max_structs<length-nj*max_structs) ? max_structs : length-nj*max_structs;
        int nwu= block_structs1*block_structs2;
        int offset1=ni*max_structs;
        int offset2=nj*max_structs;
        sizes.z=block_structs1;
        sizes.w=nwu;
        //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d %d %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid,block_structs1,block_structs2);

        start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
        clSetKernelArg(tmscore_matrix_rect, 0,sizeof(cl_int4),&sizes); 
        clSetKernelArg(tmscore_matrix_rect, 1,sizeof(int),&nseeds);
        clSetKernelArg(tmscore_matrix_rect, 2,sizeof(cl_int4),&start_points);
        clSetKernelArg(tmscore_matrix_rect, 3,sizeof(coords41_buffer),&coords41_buffer);
        clSetKernelArg(tmscore_matrix_rect, 4,sizeof(coords42_buffer),&coords42_buffer);
        clSetKernelArg(tmscore_matrix_rect, 5,sizeof(tmscores_buffer),&tmscores_buffer);
        clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
        clEnqueueWriteBuffer(queue, coords42_buffer , CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
        clFinish( queue );
        size_t global,local=nt;
        global=max_nwg*local;
        clEnqueueNDRangeKernel(queue, tmscore_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
        clFinish( queue );
        clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
        clFinish( queue);
        int m=0;
        for(int i=0;i<block_structs1;i++){
         for(int j=0;j<block_structs2;j++){
          this->set_matrix(j+offset2,i+offset1,tmscores[m].x);
          m++;
         }
        } 
       }
      }
     }
    }
   }
   #pragma omp barrier  
   if(th==0){
    double end = get_time();
    fprintf(stderr, "%8.3f seconds elapsed for %15.0f.0 RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)length*(double)(length-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)length*(float)(length-1)*0.5f));
    fprintf(stderr,"finished\n");
   } 
   clReleaseMemObject(coords41_buffer);
   if(coords4)free(coords4);
   if(ngrid >1){
    clReleaseMemObject(coords42_buffer);
   }
   clReleaseCommandQueue(queue);
   clReleaseContext(context);
   if(tmscores)free(tmscores);
  }//end parallel
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
  if(diagonal_lock) delete [] diagonal_lock;
  if(rectangle_lock) delete [] rectangle_lock;
 }
 int find_rmsd_matrix_multiple_gpus (int nt, int nwg_per_cu,const char *source,int nats,float *coords,int nthreads){
  //this is called after the multiple gpu is detected
  //divides the work based upon number of compute units detected
  const int max_gpus=10;
  int total_cu=0;
  int max_threads=omp_get_max_threads();
  int omp_nt=(max_threads<nthreads)? max_threads : nthreads;
    //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  //need centered coords for this 
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
 
  //center coords - may not have been done previously
  
  center_all_coords(length,pdb_size/3,coords);
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_platform_id platform;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
  
 //get all devices
  cl_device_id gpus[max_gpus],ncus[max_gpus];
  int ngpus=0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices);
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=(omp_nt >num_of_devices)? num_of_devices : omp_nt;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  //find total number of compute units
  for (int i=0;i<ngpus;i++){
   cl_device_id device;
   device=gpus[i];
   clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL);
   total_cu+=ncu;
  }
  fprintf(stderr,"%d compute units found in %d gpus\n",total_cu,ngpus);
  
  //calculate maximum number of workgroups per compute unit - use smaller gpu size
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  //use same block sizes for now - can optimize for different gpus by adjusting blocksizes - would require storing sizes and offsets in arrays
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers

  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  RMSD_MAX_TR_BLOCK_SIZE) max_structs=RMSD_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(length<max_structs)max_structs=length;

  int ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  //adjust ngrid to ngpus or length
  if (ngrid > 1 || ngpus >1 ){
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(length<max_structs)max_structs=length;
   ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs;
   if(ngrid <ngpus){
    ngrid=ngpus;
    max_structs=(length%ngpus)? length/ngpus+1 : length/ngpus;
   }
  }
  
  //set up lock variables
  int *diagonal_lock=new int [ngrid];
  memset(diagonal_lock,0,sizeof(int)*ngrid);
  int *rectangle_lock=new int [ngrid*ngrid];
  memset(rectangle_lock,0,sizeof(int)*ngrid*ngrid);
  //set diagonals of rectangle lock file to zero
  for (int i=0;i<ngrid;i++)
   rectangle_lock[i*ngrid+i]=1;
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  double start_rmsd;
  #pragma omp parallel num_threads(ngpus)
  {
   int th=omp_get_thread_num();
   float *rmsd_scores=0;
   cl_int4 sizes,start_points;
   cl_device_id device=gpus[th];
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel rmsd_matrix,rmsd_matrix_rect;
   cl_mem coords4_buffer,coords42_buffer,rmsd_buffer;
   cl_int err;
   context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
   queue = clCreateCommandQueue(context, device, 0, &err); 
   program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
   if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS){
    printf("Error building program\n");
    char buf[0x10000];
    clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
    fprintf(stderr,"\n%s\n", buf);
    exit(FALSE);
   }
   //create hosts arrays and buffers - local copies to avoid collisions
   float4 *coords4;
   if (!(coords4 = (float4*)  malloc(pdb4_size * length *sizeof(float4)))) exit(FALSE);
   convert_coords_to_float4 (length,pdb_size,coords,coords4);
   rmsd_matrix = clCreateKernel(program, "rmsd_matrix", &err); 
   rmsd_matrix_rect = clCreateKernel(program, "rmsd_matrix_rect", &err);
   #pragma omp barrier   
   if(th==0){
    start_rmsd=get_time();
   }
   coords4_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

   if(ngrid >1){
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(float),NULL, NULL);
    if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_sq))))exit(FALSE);  
   }
   else{
    rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(float),NULL, NULL);
    if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_tr))))exit(FALSE);  
   }
   sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
   for (int ni=0;ni<ngrid;ni++){    
    //get lock
    #pragma omp critical
    {
     while(diagonal_lock[ni] && ni <ngrid)ni++;
     if(ni < ngrid) diagonal_lock[ni]=1;
    }
    if(ni < ngrid){
    //triangular tiles first - calculate the on diagonal submatrices
    //calculate block_size
     int block_structs=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
     int nwu= block_structs*(block_structs-1)/2;
     int offset=ni*max_structs;
     //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d\n",ni,ni,nwu,block_structs,ngrid);

     sizes.z=block_structs;
     sizes.w=nwu;
     start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
     clSetKernelArg(rmsd_matrix, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(rmsd_matrix, 1,sizeof(cl_int4),&start_points);
     clSetKernelArg(rmsd_matrix, 2,sizeof(coords4_buffer),&coords4_buffer);
     clSetKernelArg(rmsd_matrix, 3,sizeof(rmsd_buffer),&rmsd_buffer);
     clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue);
     size_t global,local=nt;
     global=max_nwg*local;
     clEnqueueNDRangeKernel(queue, rmsd_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish(queue);
     clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
     clFinish( queue);
     //output to matrix
     int m=0;
     for(int i=0;i<block_structs-1;i++){
      for(int j=i+1;j<block_structs;j++){
       this->set_matrix(j+offset,i+offset,rmsd_scores[m]);
       m++;
      }
     }
    }
   }
   for (int ni=0;ni<ngrid-1;ni++){
    for (int nj=ni+1;nj<ngrid;nj++){
     if(ni!=nj)
     {
      //obtain lock
      #pragma omp critical
      {
       while(rectangle_lock[ni*ngrid+nj] && nj <ngrid){
        nj++;
       }
       if(nj<ngrid){
        rectangle_lock[ni*ngrid+nj]=1;
       }
      }
      if(nj < ngrid && ni != nj){
       //rectangular tile
       int block_structs1=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
       int block_structs2=(max_structs<length-nj*max_structs) ? max_structs : length-nj*max_structs;
       int nwu= block_structs1*block_structs2;
       int offset1=ni*max_structs;
       int offset2=nj*max_structs;
       sizes.z=block_structs1;
       sizes.w=nwu;
       //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid);

       start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
       clSetKernelArg(rmsd_matrix_rect, 0,sizeof(cl_int4),&sizes); 
       clSetKernelArg(rmsd_matrix_rect, 1,sizeof(cl_int4),&start_points);
       clSetKernelArg(rmsd_matrix_rect, 2,sizeof(coords4_buffer),&coords4_buffer);
       clSetKernelArg(rmsd_matrix_rect, 3,sizeof(coords42_buffer),&coords42_buffer);
       clSetKernelArg(rmsd_matrix_rect, 4,sizeof(rmsd_buffer),&rmsd_buffer);
       clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
       clEnqueueWriteBuffer(queue, coords42_buffer, CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
       clFinish( queue );
       size_t global,local=nt;
       global=max_nwg*local;

       clEnqueueNDRangeKernel(queue, rmsd_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
       clFinish( queue );
       clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
       clFinish( queue);

       int m=0;
       for(int i=0;i<block_structs1;i++){
        for(int j=0;j<block_structs2;j++)
        {
         this->set_matrix(j+offset2,i+offset1,rmsd_scores[m]);
         m++;
        }
       }
      }
     }
    }
   }
   #pragma omp barrier  
   if(th==0)
   {
    double end = get_time();
    fprintf(stderr, "%8.3f seconds elapsed for %15.0f.0 RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)length*(double)(length-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)length*(float)(length-1)*0.5f));
    fprintf(stderr,"finished\n");
   } 
   clReleaseMemObject(coords4_buffer);
   if(coords4)free(coords4);
   if(ngrid >1)
   {
    clReleaseMemObject(coords42_buffer);
   }
   clReleaseCommandQueue(queue);
   clReleaseContext(context);
   if(rmsd_scores)free(rmsd_scores);
  }//end parallel
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
  if(diagonal_lock) delete [] diagonal_lock;
  if(rectangle_lock) delete [] rectangle_lock;
 }
 int find_rmsd_matrix(int cpu_flag,int nt, int nwg_per_cu,const char *source,int nats,float *coords,int gpu_id,int nthreads)
 {
  //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
  float4 *coords4;
  float *rmsd_scores=0;
  double start_rmsd,end;
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_int4 sizes,start_points;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel rmsd_matrix,rmsd_matrix_rect;
  cl_mem coords4_buffer,coords42_buffer,rmsd_buffer;
  cl_int err;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
 
  cl_device_id cpus[128],gpus[10];
  int ngpus=0;

  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d cpus found\n",num_of_devices);
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_of_devices,cpus, 0);
  }
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=num_of_devices;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
 
  // try to get a supported GPU device
  //test with CPU
  if(cpu_flag)
  { 
   device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"using cpu %d cores found\n",ncu);
  }
  else
  {
   if (!ngpus)
   {
    fprintf(stderr, "no gpu found - running with cpu");
    device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
   }
   else 
   {
    if(gpu_id > ngpus-1){
     fprintf(stderr,"gpu_id error %d gpus found - highest allowed gpu_id is %d - gpu_id of %d given\n",ngpus,ngpus-1,gpu_id);
     exit(FALSE);
    }  
    if(gpu_id ==-1){ 
     if(ngpus >1){
      find_rmsd_matrix_multiple_gpus (nt,nwg_per_cu,source,nats,coords,nthreads);
      return(1);
     }
     device=gpus[0];
    }
    else device=gpus[gpu_id];
    
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"%d compute units found\n",ncu);
   }
  }  context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
  queue = clCreateCommandQueue(context, device, 0, &err); 
  
  //calculate maximum number of workgroups per compute unit
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  if(cpu_flag)lds*=2;   //memory used is memory to cache coords
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
  {
   printf("Error building program\n");
   char buf[0x10000];
   clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
   fprintf(stderr,"\n%s\n", buf);
   return 1;
  }
  
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers


  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  RMSD_MAX_TR_BLOCK_SIZE) max_structs=RMSD_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(length<max_structs)max_structs=length;

  
  int ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  if (ngrid > 1)
  {
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(length<max_structs)max_structs=length;
   ngrid=(length%max_structs)?length/max_structs+1 : length/max_structs;
  }

  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  //create hosts arrays and buffers
  if (!(coords4 = (float4*)  malloc(pdb4_size * length *sizeof(float4)))) exit(FALSE);
  #ifdef OPENMP  
  if(nthreads >1){
   int max_threads=omp_get_max_threads();
   int nt=(max_threads<nthreads)? max_threads : nthreads;
   if(nt >= length) nt=1;	  
   #pragma omp parallel num_threads(nt)  
   {

    int th=omp_get_thread_num();
    if(th<nt){
     int offset=th*(length/nt);
     int tlength=(th<nthreads-1)? length/nt : length-offset;
     float *tcoords=&(coords[offset*pdb_size]);
     float4 *tcoords4=&(coords4[offset*pdb4_size]);
     center_all_coords(tlength,pdb_size/3,tcoords);
     convert_coords_to_float4 (tlength,pdb_size,tcoords,tcoords4);    
    }
   }
  }
  else{ 
   center_all_coords(length,pdb_size/3,coords);
   convert_coords_to_float4 (length,pdb_size,coords,coords4);
  }
#else
  center_all_coords(length,pdb_size/3,coords);
  convert_coords_to_float4 (length,pdb_size,coords,coords4);
#endif


  start_rmsd = get_time();  

  rmsd_matrix = clCreateKernel(program, "rmsd_matrix", &err);
  rmsd_matrix_rect = clCreateKernel(program, "rmsd_matrix_rect", &err);
  end = get_time();  
  fprintf(stderr, "%8.3f seconds elapsed for program generation\n",end-start_rmsd);
  start_rmsd = get_time();  
  coords4_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

  if(ngrid >1)
  {
   coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(float),NULL, NULL);
   if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_sq))))exit(FALSE);  
  }
  else
  {
   rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(float),NULL, NULL);
   if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_tr))))exit(FALSE);  
  }
  sizes.x=nats;sizes.y=nats4;
  //indices need to be worked out
  for (int ni=0;ni<ngrid;ni++)
  {
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
   int block_structs=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
   int nwu= block_structs*(block_structs-1)/2;
   int offset=ni*max_structs;
   //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d\n",ni,ni,nwu,block_structs,ngrid);

   sizes.z=block_structs;
   sizes.w=nwu;
   start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
   clSetKernelArg(rmsd_matrix, 0,sizeof(cl_int4),&sizes); 
   clSetKernelArg(rmsd_matrix, 1,sizeof(cl_int4),&start_points);
   clSetKernelArg(rmsd_matrix, 2,sizeof(coords4_buffer),&coords4_buffer);
   clSetKernelArg(rmsd_matrix, 3,sizeof(rmsd_buffer),&rmsd_buffer);
   clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
   clFinish( queue);
   size_t global,local=nt;
   global=max_nwg*local;
   clEnqueueNDRangeKernel(queue, rmsd_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
   clFinish(queue);
   clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
   clFinish( queue);
   //output to matrix

  int m=0;
  for(int i=0;i<block_structs-1;i++)
   for(int j=i+1;j<block_structs;j++)
   {
    this->set_matrix(j+offset,i+offset,rmsd_scores[m]);
    m++;
   }

  } 
  for (int ni=0;ni<ngrid-1;ni++)
   for (int nj=ni+1;nj<ngrid;nj++)
    if(ni!=nj)
    {
     //rectangular tile
     int block_structs1=(max_structs<length-ni*max_structs) ? max_structs : length-ni*max_structs;
     int block_structs2=(max_structs<length-nj*max_structs) ? max_structs : length-nj*max_structs;
     int nwu= block_structs1*block_structs2;
     int offset1=ni*max_structs;
     int offset2=nj*max_structs;
     sizes.z=block_structs1;
     sizes.w=nwu;
     //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid);

     start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
     clSetKernelArg(rmsd_matrix_rect, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(rmsd_matrix_rect, 1,sizeof(cl_int4),&start_points);
     clSetKernelArg(rmsd_matrix_rect, 2,sizeof(coords4_buffer),&coords4_buffer);
     clSetKernelArg(rmsd_matrix_rect, 3,sizeof(coords42_buffer),&coords42_buffer);
     clSetKernelArg(rmsd_matrix_rect, 4,sizeof(rmsd_buffer),&rmsd_buffer);
     clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clEnqueueWriteBuffer(queue, coords42_buffer, CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue );
     size_t global,local=nt;
     global=max_nwg*local;

     clEnqueueNDRangeKernel(queue, rmsd_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish( queue );
     clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
     clFinish( queue);

     int m=0;
     for(int i=0;i<block_structs1;i++)
      for(int j=0;j<block_structs2;j++)
      {
       this->set_matrix(j+offset2,i+offset1,rmsd_scores[m]);
       m++;
      }
    }
  end = get_time();  
  fprintf(stderr, "%8.3f seconds elapsed for %15.0lf RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)length*(double)(length-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)length*(float)(length-1)*0.5f));
  fprintf(stderr,"finished\n");  
  clReleaseMemObject(coords4_buffer);
  clReleaseProgram(program);
  if(ngrid >1)
  {
   clReleaseMemObject(coords42_buffer);
  }
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  if(coords4)free(coords4);
  if(rmsd_scores)free(rmsd_scores);
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
 }
 #endif      
};

template <class T>
class cluster_models_set{ //contains names and coords of structures along with distance matrix - the good copy which is accessed through maps by the differen wrapper classes
 //ensemble of protein model with identical sequences - keeps track of subset of attibutes required for clustering
 friend class mapped_cluster_set<T>;
 friend class parallel_cluster_set<T>;
 friend class cluster_partition<T>;
 friend class mapped_cluster_models_set<T>;
 public:
  int nmodels; 
  int all_atoms_flag;
  int nthreads;
  _metric_type score_type;
  bool greater_is_better;                          
  char *names;
  int *names_offsets;
  float *coords;
  float *density;           //used for density only option as alternative to clustering using distance matrices
  int nat,pdb_size;         //attributes to partition coords into separate models                          
  triangular_matrix <T> *dmatrix;
  cluster_models_set () : nthreads(1),nat(0),names(0),names_offsets(0),coords(0),nmodels(0),score_type(RMSD),pdb_size(0),greater_is_better(0){//basic constructor to filled by 
   dmatrix=0;
   density=0;
  } 
  cluster_models_set  (const char *option,...) : nat(0),names(0),names_offsets(0),coords(0),nmodels(0),score_type(RMSD),pdb_size(0){
   //instantiated by reading coordinates or reading matrix
   dmatrix=0;coords=0;names=0;nat=0;pdb_size=0;density=0;
   va_list args;
   va_start(args,option);
   //first in list is always an identifier

   if(strcmp(option,"READ_PDB_LIST") == 0)
   {
    //means that dmatrix is to be generated
    //argument list is (nthreads score_type, all_atoms_flag, cpu_flag,names_filename,inverse_map,subset_filename,sse_flag,density_flag,gpu_id) 
    int *inverse_map=0;
    char *subset_filename=0;
    int density_flag=0,gpu_id=-1;
    int nthreads=va_arg(args, int);
    score_type=(_metric_type) va_arg(args, int);
    all_atoms_flag=va_arg(args,int);
    _compute_type compute =(_compute_type) va_arg(args,int);
    _dmatrix_type dm_type=  (_dmatrix_type) va_arg(args,int);
    char *names_filename=va_arg(args,char *);
    inverse_map =va_arg(args,int*);
    subset_filename =va_arg(args,char*);
    _simd_type simd_type = (_simd_type) va_arg(args,int);
    density_flag=va_arg(args,int);
    gpu_id=va_arg(args,int);
    va_end(args);
    double start_time = get_time();  
    read_list_of_models(names_filename,nthreads);//initializes nat and pdb_size and nmodels
    double end_time = get_time();
    fprintf(stderr, "%8.3f seconds elapsed for coords I/O\n",end_time-start_time);
    start_time = get_time();
    if(score_type == RMSD) greater_is_better=false;
    if(score_type == TMSCORE) greater_is_better=true;
    if(density_flag){
     //return the density and do nothing else
     fprintf(stderr,"calculating density only\n");
     generate_new_density(inverse_map,subset_filename,score_type,compute,simd_type,nthreads,gpu_id);
     end_time = get_time();    
     fprintf(stderr, "%8.4f seconds elapsed to calculate density %8.4f us per element\n",end_time-start_time,1.0e6*(end_time-start_time)/((double)nmodels*(double)(nmodels-1)/2.0));
    }
    else{
     if(dm_type==COMPACT){
      if(score_type==TMSCORE){
       generate_new_matrix(inverse_map,subset_filename,score_type,compute,simd_type,TMSCORE_MIN_VALUE,TMSCORE_STEP_SIZE_VALUE,nthreads,gpu_id); 
      }
      else if (score_type==RMSD){
       generate_new_matrix(inverse_map,subset_filename,score_type,compute,simd_type,RMSD_MIN_VALUE,RMSD_STEP_SIZE_VALUE,nthreads,gpu_id); 
      }
     }
     else
      generate_new_matrix(inverse_map,subset_filename,score_type,compute,simd_type,nthreads,gpu_id);
     end_time = get_time();    
     fprintf(stderr, "%8.4f seconds elapsed to calculate density %8.4f us per element\n",end_time-start_time,1.0e6*(end_time-start_time)/((double)nmodels*(double)(nmodels-1)/2.0));
    }
   }
   else if(strcmp(option,"READ_BINARY_COORDS") == 0)
   {
    //means that dmatrix is to be generated from a real binary coords file 
    //requires a names file in text
    //i.e. base_filename is foo then foo.coords and foo.names are two required files
    //then calculates the number of atoms from the text - if the number is not integer then returns an error
    //argument list is (nthreads score_type, all_atoms_flag, cpu_flag,base_filename,inverse_map,subset_filename,sse_flag,density_flag,gpu_id) 
    int *inverse_map=0;
    char *subset_filename=0,names_filename[FILENAME_LENGTH],coords_filename[FILENAME_LENGTH];
    int density_flag=0,gpu_id=-1;
    int nthreads=va_arg(args, int);
    score_type=(_metric_type) va_arg(args, int);
    all_atoms_flag=va_arg(args,int);
    _compute_type compute =(_compute_type) va_arg(args,int);
    _dmatrix_type dm_type=  (_dmatrix_type) va_arg(args,int);
    char *base_filename=va_arg(args,char *);
    inverse_map =va_arg(args,int*);
    subset_filename =va_arg(args,char*);
    _simd_type simd_type = (_simd_type) va_arg(args,int);
    density_flag=va_arg(args,int);
    gpu_id=va_arg(args,int);
    va_end(args);
    double start_time = get_time();  
    sprintf(names_filename,"%s.names",base_filename);
    sprintf(coords_filename,"%s.coords",base_filename);
    nmodels=read_names_list(names_filename);
    allocate_and_slurp_coords(coords_filename);
    double end_time = get_time();
    fprintf(stderr, "%8.3f seconds elapsed for coords I/O\n",end_time-start_time);
    start_time = get_time();
    if(score_type == RMSD) greater_is_better=false;
    if(score_type == TMSCORE) greater_is_better=true;
    if(density_flag){
     //return the density and do nothing else
     fprintf(stderr,"calculating density only\n");
     generate_new_density(inverse_map,subset_filename,score_type,compute,simd_type,nthreads,gpu_id);
     end_time = get_time();    
     fprintf(stderr, "%8.4f seconds elapsed to calculate density %8.4f us per element\n",end_time-start_time,1.0e6*(end_time-start_time)/((double)nmodels*(double)(nmodels-1)/2.0));
    }
    else{
     if(dm_type==COMPACT){
      if(score_type==TMSCORE){
       generate_new_matrix(inverse_map,subset_filename,score_type,compute,simd_type,TMSCORE_MIN_VALUE,TMSCORE_STEP_SIZE_VALUE,nthreads,gpu_id); 
      }
      else if (score_type==RMSD){
       generate_new_matrix(inverse_map,subset_filename,score_type,compute,simd_type,RMSD_MIN_VALUE,RMSD_STEP_SIZE_VALUE,nthreads,gpu_id); 
      }
     }
     else
      generate_new_matrix(inverse_map,subset_filename,score_type,compute,simd_type,nthreads,gpu_id);
     end_time = get_time();    
     fprintf(stderr, "%8.4f seconds elapsed to calculate density %8.4f us per element\n",end_time-start_time,1.0e6*(end_time-start_time)/((double)nmodels*(double)(nmodels-1)/2.0));
    }
   }   
   else if (strcmp(option,"READ_MATRIX") == 0){
    //argument list is (nthreads,score_type, all_atoms_flag,names_filename,matrix_filename,binary_flag,compact_flag,inverse_map,subset_filename,density)
     dmatrix=0;
    int *inverse_map=0;   
    int density_flag=0;
    char *subset_filename=0;
    int nthreads=va_arg(args,int);
    score_type=(_metric_type) va_arg(args,int);
    all_atoms_flag=va_arg(args,int);
    char *names_filename=va_arg(args,char *);
    char *matrix_filename=va_arg(args,char *);
    _matrix_type read_matrix_type= (_matrix_type)va_arg(args,int);
    _dmatrix_type distance_matrix_type=(_dmatrix_type)va_arg(args,int);
    inverse_map=(int*) va_arg(args,char*);
    subset_filename =va_arg(args,char*);
    density_flag=va_arg(args,int);
    va_end(args);
    if(score_type == RMSD) greater_is_better=false;
    if(score_type == TMSCORE) greater_is_better=true;
    coords=0;
    nmodels=read_list_of_model_names(names_filename);
    if(density_flag)
     density_from_matrix(matrix_filename,inverse_map,subset_filename,score_type,read_matrix_type);
    else
     read_new_matrix(matrix_filename,inverse_map,nmodels,subset_filename,score_type,read_matrix_type);
   }
   else{
    fprintf(stderr,"unknown option %s\nValid options are:\nREAD_PDB_LIST\nREAD_MATRIX\n",option);
   }
  }
  cluster_models_set (const cluster_models_set &A) : nthreads(A.nthreads), nat(A.nat),nmodels(A.nmodels),score_type(A.score_type),pdb_size(A.pdb_size),greater_is_better(A.greater_is_better){
   if(A.coords){
    coords= (float*) memalign(32,nmodels*pdb_size*sizeof(float));
    memmove(coords,A.coords,nmodels*pdb_size*sizeof(float)); 
   }
   else coords=0;
   if(A.names_offsets){
    names_offsets=new int[nmodels+1];
    memmove(names_offsets,A.names_offsets,(nmodels+1)*sizeof(int));
    names=new char[names_offsets[nmodels]];
    memmove(names,A.names,names_offsets[nmodels]*sizeof(char));
   }
   else{names=0;names_offsets=0;}
   dmatrix=new triangular_matrix<T>((const triangular_matrix<T> &)A.dmatrix);
  }
  cluster_models_set &operator= (const cluster_models_set &rhs){
   if(this != &rhs){
    nthreads=rhs.nthreads;nat=rhs.nat;nmodels=rhs.nmodels;score_type=rhs.score_type;pdb_size=rhs.pdb_size,greater_is_better=rhs.greater_is_better;
    if(coords)free(coords);
    if(names) delete [] names;
    if(names_offsets) delete [] names_offsets;    
    if(rhs.coords){
     coords= (float*) memalign(32,nmodels*pdb_size*sizeof(float));
     memmove(coords,rhs.coords,nmodels*pdb_size*sizeof(float)); 
    }
    else coords=0;
    if(rhs.names_offsets){
     names_offsets=new int[nmodels+1];
     memmove(names_offsets,rhs.names_offsets,(nmodels+1)*sizeof(int));
     names=new char[names_offsets[nmodels]];
     memmove(names,rhs.names,names_offsets[nmodels]*sizeof(char));
    }
    else{names=0;names_offsets=0;}
   }
   return *this;
  } 
  ~cluster_models_set(){
   if (coords) free(coords);
   if (dmatrix) delete dmatrix;
   if (names) delete [] names;
   if (names_offsets) delete [] names_offsets;
   if (density) delete [] density;
  }
  template <class T2>
  bool better (T2 a, T2 b){   
   return ((greater_is_better && a>b) || (!greater_is_better && a<b));
  }  
  void check_coords(){
   for(int n=0;n<nmodels;n++){
    int nzeros=0;
    float *coords1=&(coords[n*pdb_size]);
    for(int i=0;i<nat;i++)
     if(fabs(coords1[3*i]) <.001 && fabs(coords1[3*i+1] )< 0.001 && fabs(coords1[3*i+2]) <0.001) nzeros++;
    if(nzeros>10){
     fprintf(stderr,"warning %s %d zeros",get_names(n),nzeros);
    } 
   } 
  }
  char* get_names(int i){
   if(i < 0 || i >= nmodels){
    fprintf(stderr,"names index %d out of range nmodels is %d\n",i,nmodels);
    exit(FALSE);
   } 
   return(&(names[names_offsets[i]]));
  }
  float get_matrix(int i, int j){
   return(dmatrix->get_matrix(i,j));
  }     
  T get_matrix_fast(int i, int j){
   return(dmatrix->get_matrix_fast(i,j));
  } 
  void set_matrix(int i, int j,float value){
   dmatrix->set_matrix(i,j,value);
  }
  void set_matrix_fast(int i, int j,float value){
   dmatrix->set_matrix_fast(i,j,value);
  }
  int read_new_matrix(char *matrix_filename,int *inverse_map,int inverse_map_size,char *subset_filename,_metric_type score_type,_matrix_type read_matrix_type){
   FILE *fp;
   open_file(&fp, matrix_filename, "r", "read matrix");
   if(inverse_map || subset_filename){
    if(subset_filename) inverse_map=read_subset(subset_filename);
    int nmap=prune_to_subset(inverse_map);
    if (dmatrix) delete dmatrix;
    if(score_type == RMSD){
     dmatrix=new triangular_matrix<T>(nmap,RMSD_MIN_VALUE,RMSD_STEP_SIZE_VALUE,0); 
     greater_is_better=0;
    }
    else if(score_type == TMSCORE){
     dmatrix=new triangular_matrix<T>(nmap,TMSCORE_MIN_VALUE,TMSCORE_STEP_SIZE_VALUE,1);
     greater_is_better=1;
    }   
    else{
     fprintf(stderr,"undefined score type %d\n",score_type);
     exit(FALSE);
    }
    if(read_matrix_type == CHAR)dmatrix->read_matrix_from_compact_file(fp,inverse_map,inverse_map_size);
    else if (read_matrix_type == TEXT) dmatrix->read_matrix_from_text_file(fp,inverse_map);
    else if (read_matrix_type == BINARY)dmatrix->read_matrix_from_binary_file(fp,inverse_map,inverse_map_size);
    else{
     fprintf(stderr,"unknown read matrix type %d\n",read_matrix_type);
     exit(FALSE);
    }
    nmodels=nmap;
   } 
   else{     
    if(score_type == RMSD){
     dmatrix=new triangular_matrix<T>(nmodels,RMSD_MIN_VALUE,RMSD_STEP_SIZE_VALUE,0); 
     greater_is_better=0;
    }
    else if(score_type == TMSCORE){
     dmatrix=new triangular_matrix<T>(nmodels,TMSCORE_MIN_VALUE,TMSCORE_STEP_SIZE_VALUE,1);
     greater_is_better=1;
    }   
    else{
     fprintf(stderr,"undefined score type %d\n",score_type);
     exit(FALSE);
    }   
    if(read_matrix_type == CHAR){
     dmatrix->read_matrix_from_compact_file(fp);
    }
    else{
     if(read_matrix_type == TEXT) dmatrix->read_matrix_from_text_file(fp);
     else dmatrix->read_matrix_from_binary_file(fp);
    }
   }
   close_file(&fp, matrix_filename, "read matrix"); 
   if(subset_filename) delete [] inverse_map;
  }
  int density_from_matrix(char *matrix_filename,int *inverse_map,char *subset_filename,_metric_type score_type,_matrix_type read_matrix_type){
   FILE *fp;
   open_file(&fp, matrix_filename, "r", "read matrix"); 
   if(inverse_map || subset_filename){
    if(subset_filename) inverse_map=read_subset(subset_filename);
    int nmap=prune_to_subset(inverse_map);
    density=new float[nmap];
    if(read_matrix_type == CHAR)density_from_compact_file(fp,inverse_map,nmap);
    else if (read_matrix_type == TEXT) density_from_text_file(fp,inverse_map);
    else if (read_matrix_type == BINARY)density_from_binary_file(fp,inverse_map,nmap);
    else{
     fprintf(stderr,"unknown read matrix type %d\n",read_matrix_type);
     exit(FALSE);
    }
    nmodels=nmap;;
   } 
   else{       
    density=new float[nmodels];    
    if(read_matrix_type == CHAR)density_from_compact_file(fp);
    else{
     if(read_matrix_type == TEXT) density_from_text_file(fp);
     else density_from_binary_file(fp);
    }
   }
   close_file(&fp, matrix_filename, "read matrix"); 
   if(subset_filename) delete [] inverse_map;
  }
  int generate_new_density(int *inverse_map,char *subset_filename, _metric_type score_type, _compute_type compute,_simd_type simd_type, int nthreads,int gpu_id){
   if(inverse_map || subset_filename){
    if(subset_filename){
     inverse_map=read_subset(subset_filename);
     nmodels=prune_to_subset(inverse_map);
     if(inverse_map)delete [] inverse_map;
    }
    else
     nmodels=prune_to_subset(inverse_map);
   }
   density=new float[nmodels];
   generate_density(nmodels,nat,pdb_size,coords,compute,score_type,simd_type,nthreads,gpu_id);
   if(subset_filename) delete [] inverse_map;
  }
  int generate_new_matrix(int *inverse_map,char *subset_filename, _metric_type score_type, _compute_type compute,_simd_type simd_type, int nthreads,int gpu_id){
   if(inverse_map || subset_filename){
    if(subset_filename){
     inverse_map=read_subset(subset_filename);
     nmodels=prune_to_subset(inverse_map);
     if(inverse_map)delete [] inverse_map;
    }
    else
     nmodels=prune_to_subset(inverse_map);
   }
   if(dmatrix) delete dmatrix;
   if(score_type == RMSD){
    dmatrix=new triangular_matrix<T>(nmodels,RMSD_MIN_VALUE,RMSD_STEP_SIZE_VALUE,0); 
    greater_is_better=0;
   }
   else if(score_type == TMSCORE){
    dmatrix=new triangular_matrix<T>(nmodels,TMSCORE_MIN_VALUE,TMSCORE_STEP_SIZE_VALUE,1);
    greater_is_better=1;
   }   
   else{
    fprintf(stderr,"undefined score type %d\n",score_type);
    exit(FALSE);
   } 
   dmatrix->generate_matrix(nat,pdb_size,coords,compute,score_type,simd_type,nthreads,gpu_id);
   if(subset_filename) delete [] inverse_map;
  }
  int generate_new_matrix(int *inverse_map,char *subset_filename, _metric_type score_type, _compute_type compute,_simd_type simd_type,float min_size, float step_size, int nthreads,int gpu_id){
   if(inverse_map || subset_filename){
    if(subset_filename){
     inverse_map=read_subset(subset_filename);
     nmodels=prune_to_subset(inverse_map);
     if(inverse_map)delete [] inverse_map;
    }
    else
     nmodels=prune_to_subset(inverse_map);
   }
   if(dmatrix) delete dmatrix;      
   
   if(score_type == RMSD)greater_is_better=0;
   else if (score_type == TMSCORE) greater_is_better=1;
   else{
    fprintf(stderr,"undefined score type %d\n",score_type);
    exit(FALSE);
   }
   
   dmatrix=new triangular_matrix<T>(nmodels,min_size,step_size,greater_is_better); 
   dmatrix->generate_matrix(nat,pdb_size,coords,compute,score_type,simd_type,nthreads,gpu_id);
 
   if(subset_filename) delete [] inverse_map;
  }
  int read_list_of_model_names(char *filename) {
   //allocates names and names offsets - used when names are not actual pdb files for coordinates
   FILE *list_fp;
   char line[LINE_LENGTH],pdb_filename[FILENAME_LENGTH];
   int m=0,nstructs=0,name_length=0,current_offset=0;
   open_file(&list_fp, filename, "r", "read_list_of_names"); 
   while (fgets(line, LINE_LENGTH, list_fp))
    if(line[0] != '\n')
    {
     check_eof(sscanf (line, "%s", pdb_filename), "read_list_of_names");
     name_length+=strlen(pdb_filename)+1;
     if(name_length)nstructs++;
    }
   fprintf(stderr,"%d structs in list and %d chars in names\n",nstructs,name_length);
   if(nstructs){
    //allocate memory
    names=new char[name_length];
    names_offsets=new int[nstructs+1];
   } 
   else return(0);
   rewind(list_fp);
   current_offset=0;
   while (fgets(line, LINE_LENGTH, list_fp))
   {
    check_eof(sscanf (line, "%s", pdb_filename), "read_list_of_names");
    name_length=strlen(pdb_filename)+1;
    if(name_length)
    {
     names_offsets[m++]=current_offset;
     strncpy(&(names[current_offset]),pdb_filename,name_length);
     current_offset+=name_length;
    }
   }
   names_offsets[m]=current_offset;
   close_file(&list_fp,filename,"read_list_of_names");
   return(nstructs);
  }
  int read_names_list(char *filename){
   FILE *list_fp;
   char line[LINE_LENGTH],pdb_filename[FILENAME_LENGTH];
   int m=0,nstructs=0,name_length=0,current_offset;
   nat=0;
   open_file(&list_fp, filename, "r", "read_name_list"); 
   while (fgets(line, LINE_LENGTH, list_fp)){
    if(line[0] != '\n'){
     check_eof(sscanf (line, "%s", pdb_filename), "read_conformation_file");
     name_length+=strlen(pdb_filename)+1;
     if(name_length)nstructs++;
    }
   }
   fprintf(stderr,"%d structs in list and %d chars in names\n",nstructs,name_length,nat);
   if(nstructs){
    //allocate memory
    names=new char[name_length];
    names_offsets=new int[nstructs+1];
    coords= (float*) memalign(32,nstructs*pdb_size*sizeof(float));
   } 
   else return(0);
   rewind(list_fp);
   current_offset=0;
   while (fgets(line, LINE_LENGTH, list_fp))
   {
    check_eof(sscanf (line, "%s", pdb_filename), "read_conformation_file");
    name_length=strlen(pdb_filename)+1;
    if(name_length)
    {
     names_offsets[m]=current_offset;
     strncpy(&(names[current_offset]),pdb_filename,name_length);
     m++;
     current_offset+=name_length;
    }
   }
   names_offsets[m]=current_offset;
   close_file(&list_fp,filename,"read_list_of_structures");
   return(nstructs);
  }  
  int allocate_and_slurp_coords(char *filename){
   FILE *fp;
   int read,size;
   float *my_array=0;
   open_file(&fp, filename, "r", 0);
   fseek (fp , 0 , SEEK_END);
   size = ftell (fp);
   rewind (fp);
   //check size
   nat=size/(3*nmodels*sizeof(float));
   pdb_size=3*nat;
   if(size != nat*nmodels*3*sizeof(float)){
    fprintf(stderr,"non_integer number of models : %d models %d nats %d size\n",nmodels,nat,size);
    exit(FALSE);
   } 
   if(size){
    coords=new float[pdb_size*nmodels];
    read=fread(coords,1,size,fp);
    close_file(&fp, filename, "allocate_and_read_coords");
   }
   return(size);
  }
  int read_list_of_models(char *filename,int nthreads){
   //structures are assumed to be identical - only CA unless all_atom_flag is set
   FILE *list_fp;
   char line[LINE_LENGTH],pdb_filename[FILENAME_LENGTH];
   int m=0,nstructs=0,name_length=0,current_offset;
   nat=0;
   open_file(&list_fp, filename, "r", "read_list_of_structures"); 
   while (fgets(line, LINE_LENGTH, list_fp))
    if(line[0] != '\n'){
     check_eof(sscanf (line, "%s", pdb_filename), "read_conformation_file");
     name_length+=strlen(pdb_filename)+1;
     if(name_length)nstructs++;
     if(!nat){
      nat=read_coords(pdb_filename,coords,1,0,all_atoms_flag);
      pdb_size=3*nat;
     }
    }
    if(all_atoms_flag)
     fprintf(stderr,"%d structs in list and %d chars in names %d atoms in decoys\n",nstructs,name_length,nat);
    else
     fprintf(stderr,"%d structs in list and %d chars in names %d CA atoms in decoys\n",nstructs,name_length,nat);
   if(nstructs){
    //allocate memory
    names=new char[name_length];
    names_offsets=new int[nstructs+1];
    coords= (float*) memalign(32,nstructs*pdb_size*sizeof(float));
   } 
   else return(0);
   rewind(list_fp);
   current_offset=0;
   m=0;
   while (fgets(line, LINE_LENGTH, list_fp))
   {
    check_eof(sscanf (line, "%s", pdb_filename), "read_conformation_file");
    name_length=strlen(pdb_filename)+1;
    if(name_length)
    {
     names_offsets[m]=current_offset;
     strncpy(&(names[current_offset]),pdb_filename,name_length);
     names_offsets[m++]=current_offset;
     current_offset+=name_length;
    }
   }   
   nmodels=nstructs;
   names_offsets[nmodels]=current_offset;
   #ifdef OPENMP
   if(nthreads >1){
    #pragma omp parallel for num_threads(nthreads)
    for (int i=0;i<nstructs;i++){
     read_coords(get_names(i),&(coords[i*pdb_size]),0,0,all_atoms_flag);
    }
   }
   else
   #endif
   {
    for (int i=0;i<nstructs;i++){
     read_coords(get_names(i),&(coords[i*pdb_size]),0,0,all_atoms_flag);
    }
   }     
   close_file(&list_fp,filename,0);
   fprintf(stderr,"%d structures read\n",nstructs);
   //check_coords();
   return(nstructs);
  }  
  void get_coords_from_names(){
   //structures are assumed to be identical - only CA unless all_atom_flag is set
   if(!nat){
    nat=read_coords(names,coords,1,0,all_atoms_flag);
    pdb_size=3*nat;
   }
   if(!coords)coords= (float*) memalign(32,pdb_size*nmodels*sizeof(float));
#ifdef OPENMP 
   #pragma omp parallel for num_threads(nthreads)
#endif 
   for(int i=0;i<nmodels;i++){
    nat=read_coords(&(names[names_offsets[i]]),&(coords[pdb_size*i]),0,1,all_atoms_flag);
   }
  }
 
   int read_coords(char *filename,float *coords,int count_only,int center_coords,int mode){
   FILE *conformation_fp;
   char line[LINE_LENGTH],a_name[5];
   float a_x, a_y, a_z;
   int atom_id,res_id,m=0;
   float sum_x=0,sum_y=0,sum_z=0;
   open_file(&conformation_fp, filename, "r", 0);
   while (fgets(line, LINE_LENGTH, conformation_fp)){
    if (strncmp(line, "ATOM", 4) == 0){
     int i,j;
     char temp_str[LINE_LENGTH];
     j=0;
     for (i=12;i<=15;i++)
     temp_str[j++]=line[i];
     temp_str[j]='\0';
     check_eof(sscanf (temp_str, "%s", a_name), "read_conformation_file");
     if (mode || strcmp(a_name,"CA")==0)
     {
      if(!count_only)
      {
       j=0;
       for (i=30;i<=37;i++)
        temp_str[j++]=line[i];
       temp_str[j]='\0';
       check_eof(sscanf (temp_str, "%f", &a_x), "read_conformation_file");
       j=0;
       for (i=38;i<=45;i++)
        temp_str[j++]=line[i];
       temp_str[j]='\0';
       check_eof(sscanf (temp_str, "%f", &a_y), "read_conformation_file");
       j=0;
       for (i=46;i<=53;i++)
       temp_str[j++]=line[i];
       temp_str[j]='\0';
       check_eof(sscanf (temp_str, "%f", &a_z), "read_conformation_file");
       coords[m++]=a_x;
       coords[m++]=a_y;
       coords[m++]=a_z;
       if(center_coords)
       {
        sum_x+=a_x;
        sum_y+=a_y;
        sum_z+=a_z;
       }
      }
      else m+=3; 
     }
    }
   }
   if(center_coords)
   {
    float nat=(float)(m/3);
    int n=0;
    sum_x/=nat;
    sum_y/=nat;
    sum_z/=nat;
    for(int i=0;i<nat;i++)
    {
     coords[n++]-=sum_x;
     coords[n++]-=sum_y;
     coords[n++]-=sum_z;
    }
   }
   close_file(&conformation_fp, filename, 0);
   return(m/3);
  }
   int write_names_to_file(char *filename,int *map,int nmap){
   FILE *fp;
   open_file(&fp,filename, "w", "main");
   if(map)
   {
    for (int i=0;i<nmap;i++)
    {
     int a=map[i];
     fprintf(fp,"%s\n",&(names[names_offsets[a]]));
    }
   }
   else
   {
    for (int i=0;i<nmodels;i++)
     fprintf(fp,"%s\n",&(names[names_offsets[i]]));
   }
   close_file(&fp,filename, "write_names_to_file"); 
  }
  int write_matrix_to_file(FILE *fp,int binary_flag){
   if(binary_flag)
    dmatrix->write_matrix_to_binary_file(fp);
   else
    dmatrix->write_matrix_to_text_file(fp);
  }
  int write_state_to_base(_compute_type compute, _metric_type score_type,_simd_type simd_type,int *map, int nmap, int nthreads,int gpu_id){//generate new matrix
   fprintf(stderr,"template type size is %d\n",sizeof(T));
   if(score_type == RMSD)greater_is_better=0;
   else if (score_type == TMSCORE) greater_is_better=1;
   else{
    fprintf(stderr,"undefined score type %d\n",score_type);
    exit(FALSE);
   }
   //sync names;
   int new_name_length=0;
   int *new_names_offsets=new int[nmap+1];
   new_names_offsets[0]=0;
   for(int i=1;i<=nmap;i++){
    int a=map[i-1];
    new_name_length+=names_offsets[a+1]-names_offsets[a];
    new_names_offsets[i]=new_name_length;
   }
   char *new_names=new char[new_name_length];
  #ifdef OPENMP
   #pragma omp parallel for num_threads(nthreads) schedule (dynamic) 
  #endif 
   for(int i=0;i<nmap;i++)
    memmove(&(new_names[new_names_offsets[i]]),&(names[names_offsets[map[i]]]),new_names_offsets[i+1]-new_names_offsets[i]);
   if(names) delete [] names;
   if(names_offsets) delete [] names_offsets;
   names=new_names;
   names_offsets=new_names_offsets;
   
   //sync sizes (nmap)
   nmodels=nmap;
   
   //sync coords or read new coords
   if(coords){
    fprintf(stderr,"synching coords %d\n",pdb_size);
    float* new_coords= (float*) memalign(32,pdb_size*nmap*sizeof(float));
#ifdef OPENMP
   #pragma omp parallel for num_threads(nthreads) schedule (dynamic) 
#endif     
    for(int i=0;i<nmap;i++)
     memmove(&(new_coords[i*pdb_size]),&(coords[map[i]*pdb_size]),pdb_size*sizeof(float));
    if(coords)free(coords);
    coords=new_coords;
   }
   else{
    fprintf(stderr,"getting coords from names\n");
    get_coords_from_names();
   }
   if(dmatrix) delete dmatrix;
   if(score_type == RMSD){
    dmatrix=new triangular_matrix<T>(nmodels,RMSD_MIN_VALUE,RMSD_STEP_SIZE_VALUE,0); 
    greater_is_better=0;
   }
   else if(score_type == TMSCORE){
    dmatrix=new triangular_matrix<T>(nmodels,TMSCORE_MIN_VALUE,TMSCORE_STEP_SIZE_VALUE,1);
    greater_is_better=1;
   }   
   else{
    fprintf(stderr,"undefined score type %d\n",score_type);
    exit(FALSE);
   }
   fprintf(stderr,"generating new matrix\n"); 
   dmatrix->generate_matrix(nat,pdb_size,coords,compute,score_type,simd_type,nthreads,gpu_id);
   fprintf(stderr,"synched base_set to subset of %d models\n",nmodels);
   //sync map
   for(int i=0;i<nmap;i++)
    map[i]=i;
  }
   //this version assumes the same base
  int write_state_to_base(char *matrix_filename,_metric_type score_type,_matrix_type read_matrix_type,int *map, int nmap){//reads a new matrix
   if(score_type == RMSD)greater_is_better=0;
   else if (score_type == TMSCORE) greater_is_better=1;
   else{
    fprintf(stderr,"undefined score type %d\n",score_type);
    exit(FALSE);
   }
   int *inverse_map=new int [nmodels];
   for(int i=0;i<nmodels;i++)
    inverse_map[i]=-1;
   for (int i=0;i<nmap;i++)
    inverse_map[map[i]]=i;  
   read_new_matrix(matrix_filename,inverse_map,nmodels,0,score_type,read_matrix_type);//also synchs nmodels and names and coords with prune_to_subset
   if(inverse_map) delete [] inverse_map;
   for(int i=0;i<nmap;i++)
    map[i]=i;
  }
  int write_state_to_base(int *map, int nmap){
   //sync dmatrix
   triangular_matrix <T> *new_dmatrix=new triangular_matrix <T>(nmap,dmatrix->min_value,dmatrix->step_size,greater_is_better);
   #ifdef OPENMP
   #pragma omp parallel for num_threads(nthreads) schedule (dynamic) 
   #endif 
   for (int i=1;i<nmap;i++){
    for (int j=0;j<i;j++){
     float value=get_matrix(i,j);
     new_dmatrix->set_matrix(i,j,value);
    }
   }
   if(dmatrix) delete dmatrix;
   dmatrix=new_dmatrix;
   //sync names;
   int new_name_length=0;
   int *new_names_offsets=new int[nmap+1];
   new_names_offsets[0]=0;
  #ifdef OPENMP
   #pragma omp parallel for num_threads(nthreads) schedule (dynamic) 
  #endif    
   for(int i=1;i<=nmap;i++){
    int a=map[i-1];
    new_name_length+=names_offsets[a+1]-names_offsets[a];
    new_names_offsets[i]=new_name_length;
   }
   char *new_names=new char[new_name_length];
   #ifdef OPENMP
   #pragma omp parallel for num_threads(nthreads) schedule (dynamic) 
  #endif   
   for(int i=0;i<nmap;i++)
    memmove(&(new_names[new_names_offsets[i]]),&(names[names_offsets[map[i]]]),new_names_offsets[i+1]-new_names_offsets[i]);
   if(names) delete [] names;
   if(names_offsets)delete [] names_offsets;
   names=new_names;
   names_offsets=new_names_offsets;
   
   //sync coords
   if(coords){
    float* new_coords= (float*) memalign(32,pdb_size*nmap*sizeof(float));
  #ifdef OPENMP
   #pragma omp parallel for num_threads(nthreads) schedule (dynamic) 
  #endif     
    for(int i=0;i<nmap;i++)
     memmove(&(new_coords[i*pdb_size]),&(coords[map[i]*pdb_size]),pdb_size*sizeof(float));
    if(coords)free(coords);
    coords=new_coords;
   }
   //sync sizes (nmap)
   nmodels=nmap;
   fprintf(stderr,"synched base_set to subset of %d models\n",nmodels);
   
   //sync map
   for(int i=0;i<nmap;i++)
    map[i]=i;
  } 
  template <class T2>
  int write_state_to_different_type_base(cluster_models_set<T2> *cmodels,_compute_type compute, _metric_type score_type,_simd_type simd_type,int *map, int nmap,int nthreads,int gpu_id){//generate new matrix
   //sync variables
   cmodels->nat=nat;
   cmodels->nmodels=nmap;
   cmodels->all_atoms_flag=all_atoms_flag;
   cmodels->pdb_size=pdb_size;
   //sync names;
   int new_name_length=0;
   int *new_names_offsets=new int[nmap+1];
   new_names_offsets[0]=0;
   for(int i=1;i<=nmap;i++){
    int a=map[i-1];
    new_name_length+=names_offsets[a+1]-names_offsets[a];
    new_names_offsets[i]=new_name_length;
   }
   char *new_names=new char[new_name_length];
  #ifdef OPENMP
   #pragma omp parallel for num_threads(nthreads) schedule (dynamic) 
  #endif 
   for(int i=0;i<nmap;i++)
    memmove(&(new_names[new_names_offsets[i]]),&(names[names_offsets[map[i]]]),new_names_offsets[i+1]-new_names_offsets[i]);
   cmodels->names=new_names;
   cmodels->names_offsets=new_names_offsets;

   
   //sync coords or read new coords
   if(coords){
    fprintf(stderr,"synching %d structures of %d coords\n",nmap,pdb_size);
    float* new_coords=(float*) memalign(32,pdb_size*nmap*sizeof(float));
  #ifdef OPENMP
   #pragma omp parallel for num_threads(nthreads) schedule (dynamic) 
  #endif     
    for(int i=0;i<nmap;i++)
     memmove(&(new_coords[i*pdb_size]),&(coords[map[i]*pdb_size]),pdb_size*sizeof(float));
    cmodels->coords=new_coords;
   }
   else{
    fprintf(stderr,"getting coords from names\n");
    cmodels->get_coords_from_names();
   }
   if(score_type == RMSD){
    cmodels->dmatrix=new triangular_matrix<T2>(cmodels->nmodels,RMSD_MIN_VALUE,RMSD_STEP_SIZE_VALUE,0); 
    cmodels->greater_is_better=0;
   }
   else if(score_type == TMSCORE){
    cmodels->dmatrix=new triangular_matrix<T2>(cmodels->nmodels,TMSCORE_MIN_VALUE,TMSCORE_STEP_SIZE_VALUE,1);
    cmodels->greater_is_better=1;
   }   
   else{
    fprintf(stderr,"undefined score type %d\n",score_type);
    exit(FALSE);
   } 
   cmodels->dmatrix->generate_matrix(nat,pdb_size,cmodels->coords,compute,score_type,simd_type,nthreads,gpu_id);
   fprintf(stderr,"synched base_set to subset of %d models\n",cmodels->nmodels);
  }

  template <class T2>
  int write_state_to_different_type_base(cluster_models_set<T2> *cmodels,char *matrix_filename,_metric_type score_type,_matrix_type read_matrix_type,int *map, int nmap){//reads a new matrix
   //copy almost everything to cmodels - could use the copy constructor but the specialization might cause problems so copy things directly
   //change the score_type and greater_is_better
   //score_type is local copy not this->score_type - i.e. is new score_type
   //delete the original later
   cmodels->nat=nat;
   cmodels->pdb_size=pdb_size;
   cmodels->nmodels=nmodels;
   cmodels->score_type=score_type;    
   if(coords){
    cmodels->coords=(float*) memalign(32,pdb_size*nmodels*sizeof(float));
    memmove(cmodels->coords,coords,nmodels*pdb_size*sizeof(float)); 
   }
   else coords=0;
   if(names_offsets){
    cmodels->names_offsets=new int[nmodels+1];
    memmove(cmodels->names_offsets,names_offsets,(nmodels+1)*sizeof(int));
    cmodels->names=new char[names_offsets[nmodels]];
    memmove(cmodels->names,names,names_offsets[nmodels]*sizeof(char));
   }
   else{names=0;names_offsets=0;}
   if(score_type == RMSD)cmodels->greater_is_better=0;
   else if (score_type == TMSCORE) cmodels->greater_is_better=1;
   else{
    fprintf(stderr,"undefined score type %d\n",score_type);
    exit(FALSE);
   }
   int *inverse_map=new int [nmodels];   
   for (int i=0;i<nmodels;i++)
    inverse_map[i]=-1;
   for (int i=0;i<nmap;i++)
    inverse_map[map[i]]=i;  
   cmodels->read_new_matrix(matrix_filename,inverse_map,nmodels,0,score_type,read_matrix_type);//synchs everything else too - copy to cmodels  
   if(inverse_map)delete [] inverse_map;
  } 
 //density from read matrix file without generating new distance matrix
  int density_from_binary_file(FILE *fp,int *inverse_map,int ninverse_map){//assume float
   int nread=0;
   int element_size=sizeof(T);
   for(int i=1;i<ninverse_map;i++){
    int a=inverse_map[i];
    for(int j=0;j<i;j++){
     T value;
     fread(&value,element_size,1,fp);
     int b=inverse_map[j];
     if(a>=0 && b>=0){
      density[a]+=value;
      density[b]+=value;
      nread++;
     }
    }
   }
   return(nread);
  }  
  int density_from_binary_file(FILE *fp){//assume float
   int nread=0;
   int element_size=sizeof(T);
   for(int i=1;i<nmodels;i++){
    for(int j=0;j<i;j++){
     T value;
     fread(&value,element_size,1,fp);      
     density[i]+=value;
     density[j]+=value;
     nread++;
    }
   }
   return(nread);
  }
  
  int density_from_compact_file(FILE *fp,int *inverse_map,int ninverse_map){	
   //reads and converts binary matrix to normal distance matrix
   int nread=0,my_models;
   float min_value,step_size;
   int my_nmodels;
   {
    fread(&my_nmodels,sizeof(int),1,fp);
    fread(&min_value,sizeof(float),1,fp);//compact write is always float min
    fread(&step_size,sizeof(float),1,fp);
   }
   for(int i=1;i<ninverse_map;i++){
    int a=inverse_map[i];
    for(int j=0;j<i;j++){
     unsigned char ctemp;
     int b=inverse_map[j];
     fread(&ctemp,1,1,fp);     
     if(a>=0 && b>=0){
      float value =(float)(min_value+step_size*(float)ctemp);
      density[a]+=value;
      density[b]+=value;     
      nread++;
     }
    }
   }
   return(nread);
		}
  int density_from_compact_file(FILE *fp){
   int nread=0,my_models;
   float min_value,step_size;	
   //reads and converts binary matrix to normal distance matrix
   int my_nmodels;
   {
    fread(&my_nmodels,sizeof(int),1,fp);
    fread(&min_value,sizeof(float),1,fp);//compact write is always float min
    fread(&step_size,sizeof(float),1,fp);
   }
   for(int i=1;i<nmodels;i++){
    for(int j=0;j<i;j++){
     unsigned char ctemp;
     fread(&ctemp,1,1,fp);
      float value =(float)(min_value+step_size*(float)ctemp);
      density[i]+=value;
      density[i]+=value;     
     nread++;
    }
   }
   return(nread);
		}
  int density_from_text_file(FILE *fp,int *inverse_map){
   char line[LINE_LENGTH];
   int nread=0;
   while (fgets(line, LINE_LENGTH, fp)){
    float value;
    int m,n,i,j;;
    sscanf (line, "%d %d %f",&m,&n,&value);
    i=inverse_map[m];
    j=inverse_map[n];
    if(i>=0 &&i<nmodels && j>=0 && j <nmodels){
     density[i]+=value;
     density[j]+=value;
     nread++;
    }
   }
   return(nread);
  }
  int density_from_text_file(FILE *fp){
   char line[LINE_LENGTH];
   int nread=0;
   while (fgets(line, LINE_LENGTH, fp)){
    float value;
    int i,j;
    sscanf (line, "%d %d %f",&i,&j,&value);
    if(i>=0 && i<nmodels &&  j>=0 && j <nmodels){
     density[i]+=value;
     density[j]+=value;    
     nread++;
    }
   }
   return(nread);
  }
 void generate_density(int nmodels,int nats,int pdb_size,float *coords,_compute_type compute,_metric_type score_type,_simd_type simd_type,int nthreads, int gpu_id){
  if(compute==cCPU){ 
#ifdef OPENMP       
    int max_threads=omp_get_max_threads();
    nthreads=(max_threads<nthreads)? max_threads : nthreads;
#endif 
   if(score_type==RMSD){
    if(simd_type==SCALAR_){//no SSE routine
     rmsd_cpu_par(nthreads,nats,nmodels,coords,density);  
    }
#ifdef SSE2
    else if(simd_type ==SSE2_){//use SSE2 routine      
     rmsd_sse2_par(nthreads,nats,nmodels,coords,density);  
    }
#endif
#ifdef SSE3
    else if(simd_type ==SSE3_){//use SSE3 routine      
     rmsd_sse3_par(nthreads,nats,nmodels,coords,density);  
    }
#endif
#ifdef AVX
    else if(simd_type == AVX_){//use AVX routine 
     rmsd_avx_par(nthreads,nats,nmodels,coords,density);         
    }
#endif //end AVX
    else{
     fprintf(stderr,"unsupported SIMD type %d\n",simd_type);
    }
   }//end rmsd
   else if (score_type == TMSCORE){ //TMscore
    float R[3][3],t[3];
    float *my_density[nthreads];
    my_density[0]=density;
    for(int i=1;i<nthreads;i++){
     my_density[i]=new float[nmodels];
    } 
    for(int i=0;i<nthreads;i++)
     for(int j=0;j<nmodels;j++)
      my_density[i][j]=0.0f;
    
    if(simd_type == SCALAR_){  

    #ifdef OPENMP   
    #pragma omp parallel for num_threads(nthreads) schedule (dynamic)
    #endif  
     for (int i=0;i<nmodels;i++){
      for(int j=0;j<i;j++){  
       float value=tmscore_rmsd_cpu(nats,&(coords[i*pdb_size]),&(coords[j*pdb_size]),R,t,0);
        #ifdef OPENMP
        int th=omp_get_thread_num();
        my_density[th][i]+=value;
        my_density[th][j]+=value;
        #else
        density[i]+=value;
        density[j]+=value;
        #endif
       }
      }  
     }
   
#ifdef SSE2
   else if (simd_type == SSE2_){
    float *x,*y,*z;
    shuffle_tmscore_coords_soa_sse(nmodels,nats,coords,&x,&y,&z);
    const int anats=(nats%4)? (nats/4)*4+4 : nats;
    #ifdef OPENMP   
    #pragma omp parallel for num_threads(nthreads) schedule (dynamic)
    #endif  
    for (int i=0;i<nmodels;i++){
     for(int j=0;j<i;j++){  
      float value=tmscore_cpu_soa_sse2(nats,&(x[i*anats]),&(y[i*anats]),&(z[i*anats]),&(x[j*anats]),&(y[j*anats]),&(z[j*anats]),0,0,0);
      #ifdef OPENMP
      int th=omp_get_thread_num();
      my_density[th][i]+=value;
      my_density[th][j]+=value;
      #else
      density[i]+=value;
      density[j]+=value;
      #endif
     } 
    }
   
    if(x) free(x);
    if(y) free(y);
    if(z) free(z);
   }
#endif    
#ifdef SSE3
  else if (simd_type == SSE3_){
   float *x,*y,*z;
   shuffle_tmscore_coords_soa_sse(nmodels,nats,coords,&x,&y,&z);
   const int anats=(nats%4)? (nats/4)*4+4 : nats;
    #ifdef OPENMP   
    #pragma omp parallel for num_threads(nthreads) schedule (dynamic)
    #endif  
    for (int i=0;i<nmodels;i++){
     for(int j=0;j<i;j++){  
      float value=tmscore_cpu_soa_sse3(nats,&(x[i*anats]),&(y[i*anats]),&(z[i*anats]),&(x[j*anats]),&(y[j*anats]),&(z[j*anats]),0,0,0);
      #ifdef OPENMP
      int th=omp_get_thread_num();
      my_density[th][i]+=value;
      my_density[th][j]+=value;
      #else
      density[i]+=value;
      density[j]+=value;
      #endif
     } 
    }
   if(x) free(x);
   if(y) free(y);
   if(z) free(z);
  }    
#endif
#ifdef AVX   
    else if (simd_type == AVX_){
     float *x,*y,*z;
     shuffle_tmscore_coords_soa_avx(nmodels,nats,coords,&x,&y,&z);
     const int unat8=(nats%8)? (nats/8+1)*8 : nats;     
  #ifdef OPENMP   
    #pragma omp parallel for num_threads(nthreads) schedule (dynamic)
  #endif  
    for (int i=0;i<nmodels;i++){
     for(int j=0;j<i;j++){  
      float value= tmscore_cpu_soa_avx(nats,&(x[i*unat8]),&(y[i*unat8]),&(z[i*unat8]),&(x[j*unat8]),&(y[j*unat8]),&(z[j*unat8]),0,0,0);      
      #ifdef OPENMP
      int th=omp_get_thread_num();
      my_density[th][i]+=value;
      my_density[th][j]+=value;
      #else
      density[i]+=value;
      density[j]+=value;
      #endif
     } 
    }
    if(x) free(x);
    if(y) free(y);
    if(z) free(z);      
   }
#endif
   else{
    fprintf(stderr,"unsupported SIMD type %d\n",simd_type);
   }
#ifdef OPENMP 
   for(int i=1;i<nthreads;i++)
    for(int j=0;j<nmodels;j++)
     density[i]+=my_density[i][j];
   for(int i=1;i<nthreads;i++)
    if(my_density[i])delete [] my_density[i];
#endif    
  }//end TMSCORE
  else{
   fprintf(stderr,"unrecognized score type %d\n",score_type);
   exit(FALSE);
  }

 }//end CPU
#ifdef GPU 
  else if(compute == cGPU){
   if (score_type== RMSD)this->find_rmsd_density(0,64,0,grmsdcl_location,nats,coords,gpu_id,nthreads);    
   else if(score_type==TMSCORE)this->find_tmscore_density(0,64,0,gtmscorecl_location,nats,coords,gpu_id,nthreads);
   else{
    fprintf(stderr,"unrecognized score type %d\n",score_type);
    exit(FALSE);
   }
   return; 
  }
#endif
  else{     
   fprintf(stderr,"unrecognized compute type %d\n",compute);
   exit(FALSE);  
  }
  return;
 }
#ifdef GPU
 int find_tmscore_density(int cpu_flag,int nt, int nwg_per_cu,const char *source,int nats,float *coords,int gpu_id,int nthreads){
  //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
  double start_program=0,start_rmsd=0,end=0;
  float4 *coords4;

  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_int4 sizes,start_points;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel tmscore_matrix,tmscore_matrix_rect;
  cl_mem tmscores_buffer,coords41_buffer,coords42_buffer;
  cl_float2 *tmscores;
  cl_int err;
  cl_uint ncu,num_of_devices,numPlatforms;
  clGetPlatformIDs( 1, &platform, NULL ); 
  //get all devices
  cl_device_id cpus[10],gpus[10];
  int ngpus=0;
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d cpus found\n",num_of_devices);
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_of_devices,cpus, 0);
  }
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=num_of_devices;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  // try to get a supported GPU device
  //test with CPU
  if(cpu_flag)
  { 
   device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"using cpu %d cores found\n",ncu);
  }
  else{
   if (!ngpus)
   {
    fprintf(stderr, "no gpu found - running with cpu");
    device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
   }
   else 
   {
    if(gpu_id > ngpus-1){
     fprintf(stderr,"gpu_id error %d gpus found - highest allowed gpu_id is %d - gpu_id of %d given\n",ngpus,ngpus-1,gpu_id);
     exit(FALSE);
    }  
    if(gpu_id ==-1){ 
     if(ngpus >=1){
      find_tmscore_density_multiple_gpus (nt,nwg_per_cu,source,nats,coords,nthreads);
      return(1);
     }
     device=gpus[0];
    }
    else device=gpus[gpu_id];
    
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"%d compute units found\n",ncu);
   }
  }
  context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
  queue = clCreateCommandQueue(context, device, 0, &err);  
  //calculate maximum number of workgroups per compute unit
  //for gpus this depends on the memory available
  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  if(cpu_flag)lds*=2; 
   //memory used is memory to cache coords plus memory for the alignment and reduction - this is different from the simple tmscore where the coords memory is used once
   //there is a bug in OpenCL with local memory declared in a block that may or may not be freed... 
  int mem_per_wg=6*nats4*sizeof(float4)+nt*sizeof(float2);
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"creating coord buffers\n");
  //create hosts arrays and buffers
  if (!(coords4 = (float4*)  malloc(pdb4_size * nmodels *sizeof(float4)))) exit(FALSE);
  convert_coords_to_float4 (nmodels,pdb_size,coords,coords4);
  fprintf(stderr,"created  buffers\n");

  start_rmsd = get_time();  
  program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
  if ((err=clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS))
  {
   fprintf(stderr,"error %d %s\n",err,print_cl_errstring(err));
   char buf[0x10000];
   clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
   fprintf(stderr,"\n%s\n", buf);
   return 1;
  }
  end = get_time();
  fprintf(stderr,"creating kernel\n");
  tmscore_matrix = clCreateKernel(program, "tmscore_matrix", &err); 
  fprintf(stderr, "%8.3f seconds elapsed for program generation\n",end-start_rmsd);
  start_rmsd = get_time();  

  start_rmsd = get_time();  
  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  TM_MAX_TR_BLOCK_SIZE) max_structs=TM_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(nmodels<max_structs)max_structs=nmodels;

   //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix

  int ngrid=(nmodels%max_structs)?nmodels/max_structs+1 : nmodels/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  if (ngrid > 1)
  {
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(cl_float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > TM_MAX_SQ_BLOCK_SIZE) max_structs=TM_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(nmodels<max_structs)max_structs=nmodels;
   ngrid=(nmodels%max_structs)?nmodels/max_structs+1 : nmodels/max_structs;
  }
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  fprintf(stderr,"creating openCL coords buff %d\n",max_structs*pdb4_size * sizeof(float4)); 
  coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
  if(ngrid >1)
  {
   fprintf(stderr,"creading square buffers\n");
 //  tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
 //  coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
   if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_sq))))exit(FALSE);  
  }
  else
  {
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(cl_float2),NULL, NULL);
   if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_tr))))exit(FALSE);  
  }
  fprintf(stderr,"calculating frames\n");
  int nseeds=calculate_number_of_frames(nats);

  sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
  for (int ni=0;ni<ngrid;ni++)
  {
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
   int block_structs=(max_structs<nmodels-ni*max_structs) ? max_structs : nmodels-ni*max_structs;
   int nwu= block_structs*(block_structs-1)/2;
   int offset=ni*max_structs;
   sizes.z=block_structs;
   sizes.w=nwu;
   start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
   //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d %d %d \n",ni,ni,nwu,block_structs,ngrid,block_structs,block_structs);

   clSetKernelArg(tmscore_matrix, 0,sizeof(cl_int4),&sizes); 
   clSetKernelArg(tmscore_matrix, 1,sizeof(int),&nseeds);
   clSetKernelArg(tmscore_matrix, 2,sizeof(cl_int4),&start_points);
   clSetKernelArg(tmscore_matrix, 3,sizeof(coords41_buffer),&coords41_buffer);
   clSetKernelArg(tmscore_matrix, 4,sizeof(tmscores_buffer),&tmscores_buffer);
   clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
   clFinish( queue);
   size_t global,local=nt;
   global=max_nwg*local;
   clEnqueueNDRangeKernel(queue, tmscore_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
   clFinish(queue);
   clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
   clFinish( queue);

   //output to matrix

   int m=0;
   for(int i=0;i<block_structs-1;i++)
    for(int j=i+1;j<block_structs;j++){
     float value=tmscores[m].x;
     density[j+offset]+=value;
     density[i+offset]+=value;
     m++;
    }
  }

  fprintf(stderr,"releasing resources from first kernel\n");
  clReleaseKernel(tmscore_matrix);
  clReleaseMemObject(coords41_buffer);
  clReleaseMemObject(tmscores_buffer);
  clReleaseCommandQueue(queue);
  
  fprintf(stderr,"allocating resources for second kernel\n");
  if(ngrid >1)
  {
   fprintf(stderr,"creating new queue\n");
   queue = clCreateCommandQueue(context, device, 0, &err); 
   fprintf(stderr,"creating new kernel\n");
   tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
   fprintf(stderr,"creating buffers\n");
   coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
   sizes.x=nats;sizes.y=nats4;
  
   for (int ni=0;ni<ngrid-1;ni++)
    for (int nj=ni+1;nj<ngrid;nj++)
     if(ni!=nj)
     {
      //rectangular tile
      int block_structs1=(max_structs<nmodels-ni*max_structs) ? max_structs : nmodels-ni*max_structs;
      int block_structs2=(max_structs<nmodels-nj*max_structs) ? max_structs : nmodels-nj*max_structs;
      int nwu= block_structs1*block_structs2;
      int offset1=ni*max_structs;
      int offset2=nj*max_structs;
      sizes.z=block_structs1;
      sizes.w=nwu;
      //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d %d %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid,block_structs1,block_structs2);

      start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
      clSetKernelArg(tmscore_matrix_rect, 0,sizeof(cl_int4),&sizes); 
      clSetKernelArg(tmscore_matrix_rect, 1,sizeof(int),&nseeds);
      clSetKernelArg(tmscore_matrix_rect, 2,sizeof(cl_int4),&start_points);
      clSetKernelArg(tmscore_matrix_rect, 3,sizeof(coords41_buffer),&coords41_buffer);
      clSetKernelArg(tmscore_matrix_rect, 4,sizeof(coords42_buffer),&coords42_buffer);
      clSetKernelArg(tmscore_matrix_rect, 5,sizeof(tmscores_buffer),&tmscores_buffer);
      clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
      clEnqueueWriteBuffer(queue, coords42_buffer , CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
      clFinish( queue );
      size_t global,local=nt;
      global=max_nwg*local;
      clEnqueueNDRangeKernel(queue, tmscore_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
      clFinish( queue );
      clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
      clFinish( queue);
      int m=0;
      for(int i=0;i<block_structs1;i++)
       for(int j=0;j<block_structs2;j++)
       {
        float value=tmscores[m].x;
        density[j+offset2]+=value;
        density[i+offset1]+=value;
        m++;       
       }
     }
   clReleaseMemObject(coords41_buffer);
   clReleaseMemObject(tmscores_buffer);
   clReleaseProgram(program);
   clReleaseMemObject(coords42_buffer);
   clReleaseKernel(tmscore_matrix_rect);
   clReleaseCommandQueue(queue);
  }
  fprintf(stderr,"finished\n");  
  end = get_time();  
  fprintf(stderr, "%8.3f seconds elapsed for %d TM-scores at %8.3f ms per TM-score\n",end-start_rmsd,nmodels*(nmodels-1)/2,(float)((end-start_rmsd)*1000)/(float)(nmodels*(nmodels-1)/2));
  clReleaseContext(context);
  if(coords4)free(coords4);
  if(tmscores)free(tmscores);
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
 }
 int find_tmscore_density_multiple_gpus (int nt, int nwg_per_cu,const char *source,int nats,float *coords,int nthreads){
  //this is called after the multiple gpu is detected
  //divides the work based upon number of compute units detected
  const int max_gpus=10;
  int total_cu=0;
  int max_threads=omp_get_max_threads();
  int omp_nt=(max_threads<nthreads)? max_threads : nthreads;
    //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  //need centered coords for this 
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
 
  //center coords - may not have been done previously
  center_all_coords(nmodels,pdb_size/3,coords);
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_platform_id platform;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
  
 //get all devices
  cl_device_id gpus[max_gpus],ncus[max_gpus];
  int ngpus=0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices);
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=(omp_nt >num_of_devices)? num_of_devices : omp_nt;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  //find total number of compute units
  for (int i=0;i<ngpus;i++){
   cl_device_id device;
   device=gpus[i];
   clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL);
   total_cu+=ncu;
  }
  fprintf(stderr,"%d compute units found in %d gpus\n",total_cu,ngpus);
  
  //calculate maximum number of workgroups per compute unit - use smaller gpu size
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  //use same block sizes for now - can optimize for different gpus by adjusting blocksizes - would require storing sizes and offsets in arrays
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers

  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  TM_MAX_TR_BLOCK_SIZE) max_structs=TM_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(nmodels<max_structs)max_structs=nmodels;

  int ngrid=(nmodels%max_structs)?nmodels/max_structs+1 : nmodels/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  //adjust ngrid to ngpus or nmodels
  if (ngrid > 1 || ngpus >1 ){
   //recalculate with MAX_ELEMENTS/2
   int min_split=ngpus*2;
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(nmodels<max_structs)max_structs=nmodels;
   ngrid=(nmodels%max_structs)?nmodels/max_structs+1 : nmodels/max_structs;
   if(ngrid <min_split){
    ngrid=min_split;
    max_structs=(nmodels%min_split)? nmodels/min_split+1 : nmodels/min_split;
   }
  }
  
  //set up lock variables
  int *diagonal_lock=new int [ngrid];
  memset(diagonal_lock,0,sizeof(int)*ngrid);
  int *rectangle_lock=new int [ngrid*ngrid];
  memset(rectangle_lock,0,sizeof(int)*ngrid*ngrid);
  //set diagonals of rectangle lock file to zero
  for (int i=0;i<ngrid;i++)
   rectangle_lock[i*ngrid+i]=1;
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  fprintf(stderr,"calculating frames\n");
  int num_seeds=calculate_number_of_frames(nats);
  double start_rmsd;
  #pragma omp parallel num_threads(ngpus)
  {
   int th=omp_get_thread_num();
   int nseeds=num_seeds;
   float *tdensity=0;
   if(th){
    tdensity=new float [nmodels];
    for (int i=0;i<nmodels;i++)
     tdensity[i]=0.0f;
   }
   else{
    tdensity=density;
   } 
   cl_int4 sizes,start_points;
   cl_device_id device=gpus[th];
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel tmscore_matrix,tmscore_matrix_rect;
   cl_mem tmscores_buffer,coords41_buffer,coords42_buffer;
   cl_float2 *tmscores;
   cl_int err;
   context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
   queue = clCreateCommandQueue(context, device, 0, &err); 
   program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
   
   if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS){
    fprintf(stderr,"Error building program\n");
    char buf[0x10000];
    clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
    fprintf(stderr,"\n%s\n", buf);
    exit(FALSE);
   }
   //create hosts arrays and buffers - local copies to avoid collisions
   float4 *coords4=0;
   if (!(coords4 = (float4*)  malloc(pdb4_size * nmodels *sizeof(float4)))) exit(FALSE);
   convert_coords_to_float4 (nmodels,pdb_size,coords,coords4);

   tmscore_matrix = clCreateKernel(program, "tmscore_matrix", &err); 
   if(th==0){
    start_rmsd=get_time();
   }
   coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

   if(ngrid >1){
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
    if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_sq))))exit(FALSE); 
   }
   else{
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(cl_float2),NULL, NULL);
    if((!(tmscores=(cl_float2*)malloc(sizeof(cl_float2)*block_matrix_size_tr))))exit(FALSE);  
   }
   sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
   for (int ni=0;ni<ngrid;ni++){    
    //get lock
    #pragma omp critical
    {
     while(diagonal_lock[ni] && ni <ngrid)ni++;
     if(ni < ngrid) diagonal_lock[ni]=1;
    }
    if(ni < ngrid){
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
     int block_structs=(max_structs<nmodels-ni*max_structs) ? max_structs : nmodels-ni*max_structs;
     int nwu= block_structs*(block_structs-1)/2;
     int offset=ni*max_structs;
     sizes.z=block_structs;
     sizes.w=nwu;
     start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
     //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d %d %d \n",ni,ni,nwu,block_structs,ngrid,block_structs,block_structs);
     clSetKernelArg(tmscore_matrix, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(tmscore_matrix, 1,sizeof(int),&nseeds);
     clSetKernelArg(tmscore_matrix, 2,sizeof(cl_int4),&start_points);
     clSetKernelArg(tmscore_matrix, 3,sizeof(coords41_buffer),&coords41_buffer);
     clSetKernelArg(tmscore_matrix, 4,sizeof(tmscores_buffer),&tmscores_buffer);
     clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue);
     size_t global,local=nt;
     global=max_nwg*local;
     clEnqueueNDRangeKernel(queue, tmscore_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish( queue);
     clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
     clFinish( queue);

     //output to matrix

     int m=0;
     for(int i=0;i<block_structs-1;i++){
      for(int j=i+1;j<block_structs;j++){
       float value=tmscores[m].x;
       tdensity[j+offset]+=value;
       tdensity[i+offset]+=value;
       m++;
      }
     }
    }
   }  
   clReleaseKernel(tmscore_matrix);
   clReleaseMemObject(coords41_buffer);
   clReleaseMemObject(tmscores_buffer);
   clReleaseCommandQueue(queue);
   if(ngrid >1){
    queue = clCreateCommandQueue(context, device, 0, &err); 
    tmscore_matrix_rect = clCreateKernel(program, "tmscore_matrix_rect", &err);
    coords41_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    tmscores_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(cl_float2),NULL, NULL);
    sizes.x=nats;sizes.y=nats4;   
    for (int ni=0;ni<ngrid-1;ni++){
     for (int nj=ni+1;nj<ngrid;nj++){
      if(ni!=nj){
       //obtain lock
       #pragma omp critical
       {
        while(rectangle_lock[ni*ngrid+nj] && nj <ngrid){
         nj++;
        }
        if(nj<ngrid){
         rectangle_lock[ni*ngrid+nj]=1;
        }
       }
       if(nj < ngrid && ni != nj){
        //rectangular tile
        int block_structs1=(max_structs<nmodels-ni*max_structs) ? max_structs : nmodels-ni*max_structs;
        int block_structs2=(max_structs<nmodels-nj*max_structs) ? max_structs : nmodels-nj*max_structs;
        int nwu= block_structs1*block_structs2;
        int offset1=ni*max_structs;
        int offset2=nj*max_structs;
        sizes.z=block_structs1;
        sizes.w=nwu;
        //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d %d %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid,block_structs1,block_structs2);

        start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
        clSetKernelArg(tmscore_matrix_rect, 0,sizeof(cl_int4),&sizes); 
        clSetKernelArg(tmscore_matrix_rect, 1,sizeof(int),&nseeds);
        clSetKernelArg(tmscore_matrix_rect, 2,sizeof(cl_int4),&start_points);
        clSetKernelArg(tmscore_matrix_rect, 3,sizeof(coords41_buffer),&coords41_buffer);
        clSetKernelArg(tmscore_matrix_rect, 4,sizeof(coords42_buffer),&coords42_buffer);
        clSetKernelArg(tmscore_matrix_rect, 5,sizeof(tmscores_buffer),&tmscores_buffer);
        clEnqueueWriteBuffer(queue, coords41_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
        clEnqueueWriteBuffer(queue, coords42_buffer , CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
        clFinish( queue );
        size_t global,local=nt;
        global=max_nwg*local;
        clEnqueueNDRangeKernel(queue, tmscore_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
        clFinish( queue );
        clEnqueueReadBuffer(queue, tmscores_buffer, CL_TRUE, 0,nwu*sizeof(cl_float2),tmscores,0,NULL,NULL);
        clFinish( queue);
        int m=0;
        for(int i=0;i<block_structs1;i++){
         for(int j=0;j<block_structs2;j++){
          float value=(float)tmscores[m].x;
          tdensity[j+offset2]+=value;
          tdensity[i+offset1]+=value;
          m++;
         }
        } 
       }
      }
     }
    }
    clReleaseMemObject(coords41_buffer);
    clReleaseMemObject(coords42_buffer);
    clReleaseMemObject(tmscores_buffer);
    clReleaseCommandQueue(queue);   
   }
   if(th){
    #pragma omp critcal
    {
     for (int i=0;i<nmodels;i++){
      density[i]+=tdensity[i]; 
     }
    }
    if(tdensity)delete [] tdensity;
   }     
   #pragma omp barrier  
   if(th==0){
    double end = get_time();
    fprintf(stderr, "%8.3f seconds elapsed for %15.0f.0 RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)nmodels*(double)(nmodels-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)nmodels*(float)(nmodels-1)*0.5f));
    fprintf(stderr,"finished\n");
   } 
 
   if(coords4)free(coords4);
   clReleaseContext(context);

  }//end parallel
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
  if(diagonal_lock) delete [] diagonal_lock;
  if(rectangle_lock) delete [] rectangle_lock;
 }

 int find_rmsd_matrix_density_multiple_gpus (int nt, int nwg_per_cu,const char *source,int nats,float *coords,int nthreads){
  //this is called after the multiple gpu is detected
  //divides the work based upon number of compute units detected
  const int max_gpus=10;
  int total_cu=0;
  int max_threads=omp_get_max_threads();
  int omp_nt=(max_threads<nthreads)? max_threads : nthreads;
    //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  //need centered coords for this 
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
 
  //center coords - may not have been done previously
  
  center_all_coords(nmodels,pdb_size/3,coords);
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);
  read_source_file(&kernel_source,source,defines_string);

  //openCL
  cl_platform_id platform;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
  
 //get all devices
  cl_device_id gpus[max_gpus],ncus[max_gpus];
  int ngpus=0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices);
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=(omp_nt >num_of_devices)? num_of_devices : omp_nt;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
  //find total number of compute units
  for (int i=0;i<ngpus;i++){
   cl_device_id device;
   device=gpus[i];
   clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL);
   total_cu+=ncu;
  }
  fprintf(stderr,"%d compute units found in %d gpus\n",total_cu,ngpus);
  
  //calculate maximum number of workgroups per compute unit - use smaller gpu size
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif

  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  //use same block sizes for now - can optimize for different gpus by adjusting blocksizes - would require storing sizes and offsets in arrays
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers

  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  RMSD_MAX_TR_BLOCK_SIZE) max_structs=RMSD_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(nmodels<max_structs)max_structs=nmodels;

  int ngrid=(nmodels%max_structs)?nmodels/max_structs+1 : nmodels/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  //adjust ngrid to ngpus or length
  if (ngrid > 1 || ngpus >1 ){
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(nmodels<max_structs)max_structs=nmodels;
   ngrid=(nmodels%max_structs)?nmodels/max_structs+1 : nmodels/max_structs;
   if(ngrid <ngpus){
    ngrid=ngpus;
    max_structs=(nmodels%ngpus)? nmodels/ngpus+1 : nmodels/ngpus;
   }
  }
  
  //set up lock variables
  int *diagonal_lock=new int [ngrid];
  memset(diagonal_lock,0,sizeof(int)*ngrid);
  int *rectangle_lock=new int [ngrid*ngrid];
  memset(rectangle_lock,0,sizeof(int)*ngrid*ngrid);
  //set diagonals of rectangle lock file to zero
  for (int i=0;i<ngrid;i++)
   rectangle_lock[i*ngrid+i]=1;
  
  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  double start_rmsd;
  #pragma omp parallel num_threads(ngpus)
  {
   int th=omp_get_thread_num();
   float *rmsd_scores=0;
   float *tdensity=0;
   if(th){
    tdensity=new float [nmodels];
    for (int i=0;i<nmodels;i++)
     tdensity[i]=0.0f;
   }
   else{
    tdensity=density;
   } 
   cl_int4 sizes,start_points;
   cl_device_id device=gpus[th];
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel rmsd_matrix,rmsd_matrix_rect;
   cl_mem coords4_buffer,coords42_buffer,rmsd_buffer;
   cl_int err;
   context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
   queue = clCreateCommandQueue(context, device, 0, &err); 
   program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
   if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS){
    printf("Error building program\n");
    char buf[0x10000];
    clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
    fprintf(stderr,"\n%s\n", buf);
    exit(FALSE);
   }
   //create hosts arrays and buffers - local copies to avoid collisions
   float4 *coords4;
   if (!(coords4 = (float4*)  malloc(pdb4_size * nmodels *sizeof(float4)))) exit(FALSE);
   convert_coords_to_float4 (nmodels,pdb_size,coords,coords4);
   rmsd_matrix = clCreateKernel(program, "rmsd_matrix", &err); 
   rmsd_matrix_rect = clCreateKernel(program, "rmsd_matrix_rect", &err);
   #pragma omp barrier   
   if(th==0){
    start_rmsd=get_time();
   }
   coords4_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

   if(ngrid >1){
    coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
    rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(float),NULL, NULL);
    if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_sq))))exit(FALSE);  
   }
   else{
    rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(float),NULL, NULL);
    if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_tr))))exit(FALSE);  
   }
   sizes.x=nats;sizes.y=nats4;

  //indices need to be worked out
   for (int ni=0;ni<ngrid;ni++){    
    //get lock
    #pragma omp critical
    {
     while(diagonal_lock[ni] && ni <ngrid)ni++;
     if(ni < ngrid) diagonal_lock[ni]=1;
    }
    if(ni < ngrid){
    //triangular tiles first - calculate the on diagonal submatrices
    //calculate block_size
     int block_structs=(max_structs<nmodels-ni*max_structs) ? max_structs : nmodels-ni*max_structs;
     int nwu= block_structs*(block_structs-1)/2;
     int offset=ni*max_structs;
     //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d\n",ni,ni,nwu,block_structs,ngrid);

     sizes.z=block_structs;
     sizes.w=nwu;
     start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
     clSetKernelArg(rmsd_matrix, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(rmsd_matrix, 1,sizeof(cl_int4),&start_points);
     clSetKernelArg(rmsd_matrix, 2,sizeof(coords4_buffer),&coords4_buffer);
     clSetKernelArg(rmsd_matrix, 3,sizeof(rmsd_buffer),&rmsd_buffer);
     clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue);
     size_t global,local=nt;
     global=max_nwg*local;
     clEnqueueNDRangeKernel(queue, rmsd_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish(queue);
     clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
     clFinish( queue);
     //output to matrix
     int m=0;
     for(int i=0;i<block_structs-1;i++){
      for(int j=i+1;j<block_structs;j++){
       float value=rmsd_scores[m];
       tdensity[j+offset]+=value;
       tdensity[i+offset]+=value;
       m++;
      }
     }
    }
   }
   for (int ni=0;ni<ngrid-1;ni++){
    for (int nj=ni+1;nj<ngrid;nj++){
     if(ni!=nj)
     {
      //obtain lock
      #pragma omp critical
      {
       while(rectangle_lock[ni*ngrid+nj] && nj <ngrid){
        nj++;
       }
       if(nj<ngrid){
        rectangle_lock[ni*ngrid+nj]=1;
       }
      }
      if(nj < ngrid && ni != nj){
       //rectangular tile
       int block_structs1=(max_structs<nmodels-ni*max_structs) ? max_structs : nmodels-ni*max_structs;
       int block_structs2=(max_structs<nmodels-nj*max_structs) ? max_structs : nmodels-nj*max_structs;
       int nwu= block_structs1*block_structs2;
       int offset1=ni*max_structs;
       int offset2=nj*max_structs;
       sizes.z=block_structs1;
       sizes.w=nwu;
       //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid);

       start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
       clSetKernelArg(rmsd_matrix_rect, 0,sizeof(cl_int4),&sizes); 
       clSetKernelArg(rmsd_matrix_rect, 1,sizeof(cl_int4),&start_points);
       clSetKernelArg(rmsd_matrix_rect, 2,sizeof(coords4_buffer),&coords4_buffer);
       clSetKernelArg(rmsd_matrix_rect, 3,sizeof(coords42_buffer),&coords42_buffer);
       clSetKernelArg(rmsd_matrix_rect, 4,sizeof(rmsd_buffer),&rmsd_buffer);
       clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
       clEnqueueWriteBuffer(queue, coords42_buffer, CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
       clFinish( queue );
       size_t global,local=nt;
       global=max_nwg*local;

       clEnqueueNDRangeKernel(queue, rmsd_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
       clFinish( queue );
       clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
       clFinish( queue);

       int m=0;
       for(int i=0;i<block_structs1;i++){
        for(int j=0;j<block_structs2;j++){
         float value=(float)rmsd_scores[m];
         tdensity[j+offset2]+=value;
         tdensity[i+offset1]+=value;
         m++;
        }
       }
      }
     }
    }
   }
   if(th){
    #pragma omp critcal
    {
     for (int i=0;i<nmodels;i++){
      density[i]+=tdensity[i]; 
     }
    }
    if(tdensity)delete [] tdensity;
   }  
   #pragma omp barrier  
   if(th==0)
   {
    double end = get_time();
    fprintf(stderr, "%8.3f seconds elapsed for %15.0f.0 RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)nmodels*(double)(nmodels-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)nmodels*(float)(nmodels-1)*0.5f));
    fprintf(stderr,"finished\n");
   } 
   clReleaseMemObject(coords4_buffer);
   if(coords4)free(coords4);
   if(ngrid >1)
   {
    clReleaseMemObject(coords42_buffer);
   }
   clReleaseCommandQueue(queue);
   clReleaseContext(context);
   if(rmsd_scores)free(rmsd_scores);
  }//end parallel
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
  if(diagonal_lock) delete [] diagonal_lock;
  if(rectangle_lock) delete [] rectangle_lock;
 }

 int find_rmsd_density(int cpu_flag,int nt, int nwg_per_cu,const char *source,int nats,float *coords,int gpu_id,int nthreads){
  //this version outputs a lower triangle matrix
  //the original code was written to calculate upper triangular matrix - change order of indices to get lower matrix
  int nats4=(nats%4)?nats/4+1:nats/4;
  int pdb_size=3*nats,pdb4_size=3*nats4;
  char *defines_string=0,*kernel_source=0;
  float4 *coords4;
  float *rmsd_scores=0;
  double start_rmsd,end;
  
  //add define string to source to specify maximum size
  define_sizes_string (&defines_string,nt,pdb4_size);

  //openCL
  cl_int4 sizes,start_points;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel rmsd_matrix,rmsd_matrix_rect;
  cl_mem coords4_buffer,coords42_buffer,rmsd_buffer;
  cl_int err;
  cl_uint ncu,num_of_devices;
  clGetPlatformIDs( 1, &platform, NULL ); 
  
  cl_device_id cpus[128],gpus[10];
  int ngpus=0;

  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d cpus found\n",num_of_devices);
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_of_devices,cpus, 0);
  }
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0,0, &num_of_devices) == CL_SUCCESS)
  {
   fprintf(stderr, "%d gpus found\n",num_of_devices);
   ngpus=num_of_devices;
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_of_devices,gpus, 0);
  }
 
  // try to get a supported GPU device
  //test with CPU
  if(cpu_flag)
  { 
   device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"using cpu %d cores found\n",ncu);
  }
  else
  {
   if (!ngpus)
   {
    fprintf(stderr, "no gpu found - running with cpu");
    device=cpus[0];
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
   }
   else 
   {
    if(gpu_id > ngpus-1){
     fprintf(stderr,"gpu_id error %d gpus found - highest allowed gpu_id is %d - gpu_id of %d given\n",ngpus,ngpus-1,gpu_id);
     exit(FALSE);
    }  
    if(gpu_id ==-1){ 
     if(ngpus >1){
	  fprintf(stderr,"using %d gpus to calculate density\n",ngpus);
      find_rmsd_matrix_density_multiple_gpus (nt,nwg_per_cu,source,nats,coords,nthreads);
      return(1);
     }
     device=gpus[0];
    }
    else device=gpus[gpu_id];
    
    clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&ncu,NULL); 
    fprintf(stderr,"%d compute units found\n",ncu);
   }
  }
  read_source_file(&kernel_source,source,defines_string);  
  context = clCreateContext(NULL,1,&device,NULL,NULL,&err);
  queue = clCreateCommandQueue(context, device, 0, &err); 
  
  //calculate maximum number of workgroups per compute unit
  //for gpus this depends on the memory available

  int lds=1024*32; //local cache size per cu - but at least 2 workgroups/cu are active at any one time so half of this is really available
  if(cpu_flag)lds*=2;   //memory used is memory to cache coords
  int mem_per_wg=(3*nats4*sizeof(float4) + nt*sizeof(float));
  #ifdef AMD
  int max_wg_per_cu=lds/mem_per_wg/2;
  #endif
  #ifdef NVIDIA
  int max_wg_per_cu=2*lds/mem_per_wg;
  #endif
    
  if( max_wg_per_cu <1) max_wg_per_cu =1;
  if(nwg_per_cu)max_wg_per_cu=nwg_per_cu;
  unsigned int max_nwg=(max_wg_per_cu)*ncu;
  fprintf(stderr,"memper %d max wg %d max_wg_per_cu %d\n",mem_per_wg, max_nwg, max_wg_per_cu);

  program = clCreateProgramWithSource(context,1,(const char**)&kernel_source, NULL,&err);
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
  {
   printf("Error building program\n");
   char buf[0x10000];
   clGetProgramBuildInfo( program,device,CL_PROGRAM_BUILD_LOG,0x10000,buf,NULL);
   fprintf(stderr,"\n%s\n", buf);
   return 1;
  }
  
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  //matrices for larger numbers of structures need to be calculated as smaller triangular and square submatrices
  //for large arrays handle in blocks - there is a limit on both size of arrays that can be passed
  //and the size of the output array
  
  //the maximum buffer size is 8192 *8192 but the AMD compiler seems to cheat and use some of this memory sometimes
  //reduce the memory in half and then half again if there are two coord buffers

  int max_structs_for_coords=(int)(MAX_ELEMENTS/pdb4_size);
  int max_structs_for_matrix=(int)sqrt((float) MAX_ELEMENTS/sizeof(float2));
  int max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;

  if(max_structs >  RMSD_MAX_TR_BLOCK_SIZE) max_structs=RMSD_MAX_TR_BLOCK_SIZE;
  if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
  if(nmodels<max_structs)max_structs=nmodels;

  
  int ngrid=(nmodels%max_structs)?nmodels/max_structs+1 : nmodels/max_structs; //size of the grid of tiles - calculation is split into ngrid*(ngrid-1)/2 submatrices for large number of structures
  if (ngrid > 1)
  {
   //recalculate with MAX_ELEMENTS/2
   max_structs_for_coords=(int)((MAX_ELEMENTS/2)/pdb4_size);
   max_structs_for_matrix=(int)sqrt((float) ((MAX_ELEMENTS/2)/sizeof(float2)));
   max_structs=(max_structs_for_coords < max_structs_for_matrix)? max_structs_for_coords : max_structs_for_matrix;
   if (max_structs > RMSD_MAX_SQ_BLOCK_SIZE) max_structs=RMSD_MAX_SQ_BLOCK_SIZE;
   if(!max_structs){fprintf(stderr,"insufficient memory to load structure\n");exit(FALSE);} 
   if(nmodels<max_structs)max_structs=nmodels;
   ngrid=(nmodels%max_structs)?nmodels/max_structs+1 : nmodels/max_structs;
  }

  int block_matrix_size_tr=max_structs*(max_structs-1)/2;
  int block_matrix_size_sq=max_structs*max_structs;

  //create hosts arrays and buffers
  if (!(coords4 = (float4*)  malloc(pdb4_size * nmodels *sizeof(float4)))) exit(FALSE);
#ifdef OPENMP  
  if(nthreads >1){
   int length=nmodels;
   int max_threads=omp_get_max_threads();
   int nt=(max_threads<nthreads)? max_threads : nthreads;
   if(nt >= length) nt=1;   
   #pragma omp parallel num_threads(nt)  
   {
    int th=omp_get_thread_num();
    if(th<nt){
     int offset=th*(length/nt);
     int tlength=(th<nthreads-1)? length/nt : length-offset;
     float *tcoords=&(coords[offset*pdb_size]);
     float4 *tcoords4=&(coords4[offset*pdb4_size]);
     center_all_coords(tlength,pdb_size/3,tcoords);
     convert_coords_to_float4 (tlength,pdb_size,tcoords,tcoords4);    
    }
   }
  }
  else{ 
   center_all_coords(nmodels,pdb_size/3,coords);
   convert_coords_to_float4 (nmodels,pdb_size,coords,coords4);
  }
#else
  center_all_coords(nmodels,pdb_size/3,coords);
  convert_coords_to_float4 (nmodels,pdb_size,coords,coords4);
#endif

  center_all_coords(nmodels,pdb_size/3,coords);
  convert_coords_to_float4 (nmodels,pdb_size,coords,coords4);

  start_rmsd = get_time();  

  rmsd_matrix = clCreateKernel(program, "rmsd_matrix", &err);
  rmsd_matrix_rect = clCreateKernel(program, "rmsd_matrix_rect", &err);
  end = get_time();  
  fprintf(stderr, "%8.3f seconds elapsed for program generation\n",end-start_rmsd);
  start_rmsd = get_time();  
  coords4_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );

  if(ngrid >1)
  {
   coords42_buffer = clCreateBuffer(context,CL_MEM_READ_ONLY, max_structs*pdb4_size * sizeof(float4),NULL, NULL );
   rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_sq * sizeof(float),NULL, NULL);
   if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_sq))))exit(FALSE);  
  }
  else
  {
   rmsd_buffer = clCreateBuffer(context,CL_MEM_READ_WRITE, block_matrix_size_tr * sizeof(float),NULL, NULL);
   if((!(rmsd_scores=(float*)malloc(sizeof(float)*block_matrix_size_tr))))exit(FALSE);  
  }
  sizes.x=nats;sizes.y=nats4;
  //indices need to be worked out
  for (int ni=0;ni<ngrid;ni++)
  {
   //triangular tiles first - calculate the on diagonal submatrices
   //calculate block_size
   int block_structs=(max_structs<nmodels-ni*max_structs) ? max_structs : nmodels-ni*max_structs;
   int nwu= block_structs*(block_structs-1)/2;
   int offset=ni*max_structs;
   //fprintf(stderr," ni %d nj %d nwu %d block_size %d grid_size %d\n",ni,ni,nwu,block_structs,ngrid);

   sizes.z=block_structs;
   sizes.w=nwu;
   start_points.x=0;start_points.y=0;start_points.z=block_structs;start_points.w=block_structs;
   clSetKernelArg(rmsd_matrix, 0,sizeof(cl_int4),&sizes); 
   clSetKernelArg(rmsd_matrix, 1,sizeof(cl_int4),&start_points);
   clSetKernelArg(rmsd_matrix, 2,sizeof(coords4_buffer),&coords4_buffer);
   clSetKernelArg(rmsd_matrix, 3,sizeof(rmsd_buffer),&rmsd_buffer);
   clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
   clFinish( queue);
   size_t global,local=nt;
   global=max_nwg*local;
   clEnqueueNDRangeKernel(queue, rmsd_matrix, 1, NULL, &global, &local, 0, NULL, NULL);
   clFinish(queue);
   clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
   clFinish( queue);
   //output to matrix


   int m=0;
   for(int i=0;i<block_structs-1;i++){
    for(int j=i+1;j<block_structs;j++){
     float value=rmsd_scores[m];
     density[j+offset]+=value;
     density[i+offset]+=value;
     m++;
    }
   }
  }
  for (int ni=0;ni<ngrid-1;ni++)
   for (int nj=ni+1;nj<ngrid;nj++)
    if(ni!=nj)
    {
     //rectangular tile
     int block_structs1=(max_structs<nmodels-ni*max_structs) ? max_structs : nmodels-ni*max_structs;
     int block_structs2=(max_structs<nmodels-nj*max_structs) ? max_structs : nmodels-nj*max_structs;
     int nwu= block_structs1*block_structs2;
     int offset1=ni*max_structs;
     int offset2=nj*max_structs;
     sizes.z=block_structs1;
     sizes.w=nwu;
     //fprintf(stderr," ni %d nj %d nwu %d block_sizes %d %d grid_size %d\n",ni,nj,nwu,block_structs1,block_structs2,ngrid);

     start_points.x=0;start_points.y=0;start_points.z=block_structs1;start_points.w=block_structs2;
     clSetKernelArg(rmsd_matrix_rect, 0,sizeof(cl_int4),&sizes); 
     clSetKernelArg(rmsd_matrix_rect, 1,sizeof(cl_int4),&start_points);
     clSetKernelArg(rmsd_matrix_rect, 2,sizeof(coords4_buffer),&coords4_buffer);
     clSetKernelArg(rmsd_matrix_rect, 3,sizeof(coords42_buffer),&coords42_buffer);
     clSetKernelArg(rmsd_matrix_rect, 4,sizeof(rmsd_buffer),&rmsd_buffer);
     clEnqueueWriteBuffer(queue, coords4_buffer , CL_TRUE, 0,block_structs1*pdb4_size * sizeof(float4),&(coords4[ni*max_structs*pdb4_size]), 0, NULL, NULL); 
     clEnqueueWriteBuffer(queue, coords42_buffer, CL_TRUE, 0,block_structs2*pdb4_size * sizeof(float4),&(coords4[nj*max_structs*pdb4_size]), 0, NULL, NULL); 
     clFinish( queue );
     size_t global,local=nt;
     global=max_nwg*local;

     clEnqueueNDRangeKernel(queue, rmsd_matrix_rect, 1, NULL, &global, &local, 0, NULL, NULL);
     clFinish( queue );
     clEnqueueReadBuffer(queue, rmsd_buffer, CL_TRUE, 0,nwu*sizeof(float),rmsd_scores,0,NULL,NULL);
     clFinish( queue);

     int m=0;
     for(int i=0;i<block_structs1;i++){
      for(int j=0;j<block_structs2;j++){
       float value=(float)rmsd_scores[m];
       density[j+offset2]+=value;
       density[i+offset1]+=value;
       m++;
      }
     }
    }
  end = get_time();  
  fprintf(stderr, "%8.3f seconds elapsed for %15.0lf RMSDs at %8.3f us per RMSD\n",end-start_rmsd,(double)nmodels*(double)(nmodels-1)/2.0,(float)((end-start_rmsd)*1000000)/((float)nmodels*(float)(nmodels-1)*0.5f));

  fprintf(stderr,"finished\n");  
  clReleaseMemObject(coords4_buffer);
  clReleaseProgram(program);
  if(ngrid >1)
  {
   clReleaseMemObject(coords42_buffer);
  }
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  if(coords4)free(coords4);
  if(rmsd_scores)free(rmsd_scores);
  if(defines_string)free(defines_string);
  if(kernel_source)free(kernel_source);
 }
#endif  
 void print_density(FILE *fp,bool sort){
  float scale=(nmodels)? 1.0f/(float)nmodels : 0.0f;
  if(sort){
   int *sorted= new int [nmodels];
   sort_by_scores (nmodels,density,sorted,!greater_is_better);
   for(int i=0;i<nmodels;i++){
    int a=sorted[i];
    fprintf(fp,"%s %10.6f\n",get_names(a),density[a]*scale);   
   }
  if(sorted) delete [] sorted;
  }
  else{
   for(int i=0;i<nmodels;i++){
    fprintf(fp,"%s %10.6f\n",get_names(i),density[i]*scale);   
   } 
  }
 }
  
  private:
  int prune_to_subset(int *inverse_map){
   //inverse_map is used for in place reading of coords from files
   //forward map is needed here i.e. map=(original1,original2...originaln)
   int *map=new int[nmodels];
   int nmap=0;
   for(int i=0;i<nmodels;i++)
    if(inverse_map[i] >=0){
     map[inverse_map[i]]=i;
     nmap++;
    } 
   int new_name_length=0;
   int *new_names_offsets=new int[nmap+1];
   new_names_offsets[0]=0;
   for(int i=1;i<=nmap;i++){
    int a=map[i-1];
    new_name_length+=names_offsets[a+1]-names_offsets[a];
    new_names_offsets[i]=new_name_length;
   }

   char *new_names=new char[new_name_length];
   for(int i=0;i<nmap;i++)
    memmove(&(new_names[new_names_offsets[i]]),&(names[names_offsets[map[i]]]),new_names_offsets[i+1]-new_names_offsets[i]);
   if(names)delete [] names;
   if(names_offsets)delete [] names_offsets;
   names=new_names;
   names_offsets=new_names_offsets;
   if(coords){
    float* new_coords=(float*) memalign(32,pdb_size*nmap*sizeof(float));
    for(int i=0;i<nmap;i++)
     memmove(&(new_coords[i*pdb_size]),&(coords[map[i]*pdb_size]),pdb_size*sizeof(float));
    if(coords)free(coords);
    coords=new_coords;
   }
   return(nmap);
  }
  
  int *read_subset (char *filename){
   FILE *fp;
   int *inverse_map=new int[nmodels];
   for(int i=0;i<nmodels;i++)
    inverse_map[i]=-1;
   char line[LINE_LENGTH];
   open_file(&fp, filename, "r", "read_subset");
   int nmap=0;
   while (fgets(line, LINE_LENGTH,fp)){
    int value;
    check_eof(sscanf (line, "%d", &value), "read_subset");
    inverse_map[value]=nmap++;
   }
   close_file(&fp, filename, "read_subset");
   fprintf(stderr,"subset of %d models to be used\n",nmap); 
   return(inverse_map);
  }
       
};

template <class T> 
class mapped_cluster_models_set{ //adds a mapping to the cluster_models_set - allows for multiple threads to manipulate the same base data
 friend class mapped_cluster_set<T>;
 friend class parallel_cluster_set<T>;
 friend class cluster_partition<T>;
 friend class cluster_models_set<T>;
 public:
  int nmodels;     //indirect number of models - not necessarily same as the number of base_models
  int *map;
  float *density;   //only initialized when used for filter
  bool greater_is_better;
  cluster_models_set <T> *base_set;
  
  mapped_cluster_models_set(cluster_models_set <T> *input_base_set) : density(0){
   base_set=input_base_set;
   nmodels=base_set->nmodels;
   greater_is_better=base_set->greater_is_better;
   map=new int [nmodels];
   for (int i=0;i<nmodels;i++)
    map[i]=i;
  }
  mapped_cluster_models_set(const mapped_cluster_models_set &A) : density(0),base_set(A.base_set),nmodels(A.nmodels),greater_is_better(A.greater_is_better){
   map=new int[base_set->nmodels];
   memmove(map,A.map,nmodels*sizeof(int));
   if(A.density){
    density=new float [base_set->nmodels];
    memmove(density,A.density,base_set->nmodels*sizeof(float));
   } 
  }
  mapped_cluster_models_set & operator = (const mapped_cluster_models_set &rhs){
   if(this != &rhs){ 
    nmodels=rhs.nmodels;
    greater_is_better=rhs.greater_is_better;
    base_set=rhs.base_set;
    if(density) delete [] density;
    if(map) delete [] map;
    map=new int[base_set->nmodels];
    memmove(map,rhs.map,nmodels*sizeof(int));
    if(rhs.density){
     density=new float [nmodels];
     memmove(density,rhs.density,nmodels*sizeof(float));
    }
    else density=0;  
   }
   return *this;
  }
  ~mapped_cluster_models_set(){
    if(map) delete [] map;
    if(density) delete [] density;
  }

  float get_matrix(int i, int j){
   return(base_set->get_matrix(map[i],map[j]));
  }  
  
  T get_matrix_fast(int i, int j){
   return(base_set->get_matrix_fast(map[i],map[j]));
  }
  
  int idensity_filter(int zmode,float prune_to,float prune_stop,int prune_min_size,int prune_max_size,float ratio,FILE *output_fp){//version that removes a max percentage each iteration
   pruned_model **log=0;
   int round=0;
   float mean,stdev,bad_cutoff_value;
   if(!density)density=new float[nmodels];
   calculate_density();

   //prune_min_size && prune_stop define lower limits and will stop the pruning when reached
   //will prune until prune_max_size unless prune_stop is reached 
   //will continue pruning to until the worst value is not worse than prune_to unells prune_min_size is met
   //first prune to
   if(output_fp) log=new pruned_model* [nmodels];
   if(prune_to){
    while(nmodels>prune_min_size){  
     int nworst=(int)(ratio*nmodels);
     if(nmodels-prune_min_size < nworst)nworst=nmodels-prune_min_size;
     if(nworst <1)nworst =1;  
     if(zmode){ //check if it is OK to stop
      mean_stdev(nmodels,density,&mean,&stdev);
      bad_cutoff_value=(greater_is_better)? mean-prune_to*stdev : mean+prune_to*stdev;
     } 
     else bad_cutoff_value=prune_to*(float)(nmodels-1);
     int nremoved=remove_nworst_density_worse_than_value(bad_cutoff_value,nworst,log);
     if(!nremoved)break;
     if(log){
      for(int i=0;i<nremoved;i++){
       fprintf(output_fp,"%s pruned with score %8.5f in round %d\n",log[i]->name,log[i]->score,round);
       if(log[i]) delete log[i];
      }
     }
     round++;
    }
   }//end prune_to block
   if(prune_max_size){
    while(nmodels>prune_max_size){
     int nworst=(int)(ratio*nmodels);
     if(nmodels-prune_max_size < nworst)nworst=nmodels-prune_max_size;
     if(nworst <1) nworst =1;  
     if(prune_stop){
      if(zmode){
       mean_stdev(nmodels,density,&mean,&stdev);
       bad_cutoff_value=(greater_is_better)? mean-prune_stop*stdev : mean+prune_stop*stdev;
      } 
      else 
       bad_cutoff_value=prune_stop*(float)(nmodels-1);
      int nremoved=remove_nworst_density_worse_than_value(bad_cutoff_value,nworst,log);
      if(!nremoved)break;
      if(log){
       for(int i=0;i<nremoved;i++){
        fprintf(output_fp,"%s pruned with score %8.5f in round %d\n",log[i]->name,log[i]->score,round);
        if(log[i])delete log[i];
       }
      }
     }
     else{
      int nremoved=remove_nworst_density(nworst,log);      
      if(log){
       for(int i=0;i<nremoved;i++){
        fprintf(output_fp,"%s pruned with score %8.5f in round %d\n",log[i]->name,log[i]->score,round);
        if(log[i])delete log[i];
       }
      }      
     }
     round++;
    }
   }
   mean_stdev(nmodels,density,&mean,&stdev);
   fprintf(stderr,"pruned to %d models with mean density %8.5f +/- %-8.4f\n",nmodels,mean/(float)(nmodels-1),stdev/(float)(nmodels-1));
   if(output_fp){
    if(log) delete [] log;
   }
  }
  char* names(int i){
   if(i < 0 || i >= nmodels){
    fprintf(stderr,"names index %d out of range nmodels is %d\n",i,nmodels);
    exit(FALSE);
   } 
   return(&(base_set->names[base_set->names_offsets[map[i]]]));
  }
  
  float* coords(int i){
   //returns a pointer to the coords of structure map[i]
   if(i < 0 || i >= nmodels){
    fprintf(stderr,"names index %d out of range nmodels is %d\n",i,nmodels);
    exit(FALSE);
   }
   return(&(base_set->coords[map[i]*base_set->pdb_size]));
  }
  int remove_worst_density_worse_than_value(float value, pruned_model **log){
   float wvalue;
   int windex=find_worst_density(&wvalue);
   if(better(value,wvalue)){
    for(int i=0;i<nmodels;i++){ //adjust densities
     if(i != windex){ 
      density[i]-=get_matrix(i,windex);
     }
    }
    if(log){//output to log
     float scale=1.0f/(float)(nmodels-1);
     int b=windex;
     log[0]=new pruned_model(names(b),density[b]*scale);
    } 
    //move worst to back
    int temp=map[windex];
    map[windex]=map[nmodels-1];
    map[nmodels-1]=temp;
    density[windex]=density[nmodels-1];
    nmodels--;
    return(1);
   }
   return(0);
  }    
  int remove_worst_density(pruned_model **log){   
   float wvalue;
   
   int windex=find_worst_density(&wvalue);
   for(int i=0;i<nmodels;i++){
    if(i != windex)
     density[i]-=get_matrix(i,windex);
   }
   if(log){//output to log
    float scale=1.0f/(float)(nmodels-1);
    int b=windex;
    log[0]=new pruned_model(names(b),density[b]*scale);
   } 
   int temp=map[windex];
   map[windex]=map[nmodels-1];
   map[nmodels-1]=temp;
   density[windex]=density[nmodels-1];
   nmodels--;
   return(1);
  }  
  int remove_nworst_density_worse_than_value(float value,int nworst,pruned_model **log){
   int retvalue=0;
   if(nworst == 1){
    return(remove_worst_density_worse_than_value(value,log));
   }
   int *bad_list=new int[nmodels];
   int nbad=0;
   for(int i=0;i<nmodels;i++)
    if(better(value,density[i]))
     bad_list[nbad++]=i;
   if(nbad <= nworst){ 
    if(nbad>0){
     if(log){//output to log
      for(int i=0;i<nbad;i++){
       int b=bad_list[i];
       float scale=1.0f/(float)(nmodels-1);
       log[i]=new pruned_model(names(b),density[b]*scale);
      }
     }  
     remove_list_from_density(nbad,bad_list);
    }
    if(bad_list) delete [] bad_list;
    return(nbad);
   }
   else{
    priority_heap_list<float> my_list (density,nworst,nbad,greater_is_better,bad_list);
    if(log){//output to log
     for(int i=0;i<nworst;i++){
      int b=my_list.heap_map[i];
      float scale=1.0f/(float)(nmodels-1);
      log[i]=new pruned_model(names(b),density[b]*scale);
     }
    } 
    remove_list_from_density(nworst,my_list.heap_map);
    if(bad_list) delete [] bad_list;
    return(nworst);
   }  
  }
  int remove_nworst_density(int nworst,pruned_model **log){
   if(nworst == 1){
    remove_worst_density(log);
    return (1);
   }
   else{
    priority_heap_list<float> my_list(density,nworst,nmodels,greater_is_better,0);
    if(log){//output to log
     for(int i=0;i<nworst;i++){
      int b=my_list.heap_map[i];
      float scale=1.0f/(float)(nmodels-1);
      log[i]=new pruned_model(names(b),density[b]*scale);
     }
    } 
    remove_list_from_density(nworst,my_list.heap_map);
    return(nworst);
   }  
  }
  int remove_list_from_density(int nbad,int *bad_list){
   int ngood=0;
   bool *bad=new bool[nmodels];
   memset(bad,0,nmodels*sizeof(bool));
   for (int i=0;i<nbad;i++)
    bad[bad_list[i]]=true;
   for (int i=0;i<nmodels;i++){
    if(!bad[i])
     for (int j=0;j<nbad;j++)
      density[i]-=get_matrix(i,bad_list[j]);
   }
   for (int i=0;i<nmodels;i++){
    if(!bad[i]){
     map[ngood]=map[i];
     density[ngood]=density[i];
     ngood++;
    }
   }
   nmodels-=nbad;
   if(bad) delete [] bad;
  } 
  void print_unnormed_density(FILE *fp){
  int *sorted= new int [nmodels];
  if(!density){
   density=new float[nmodels];
   calculate_density();
  }
  sort_by_scores (nmodels,density,sorted,!greater_is_better);
  for(int i=0;i<nmodels;i++){
   int a=sorted[i];
   fprintf(fp,"%s %8.5f\n",names(a),density[a]);   
  }
  if(sorted) delete [] sorted;
 }
  void print_density(FILE *fp){
  int *sorted= new int [nmodels];
  float scale=(nmodels)? 1.0f/(float)nmodels : 0.0f;
  if(!density){
   density=new float[nmodels];
   calculate_density();
  }
  sort_by_scores (nmodels,density,sorted,!greater_is_better);
  for(int i=0;i<nmodels;i++){
   int a=sorted[i];
   fprintf(fp,"%s %8.5f\n",names(a),density[a]*scale);   
  }
  if(sorted) delete [] sorted;
 } 
  void print_density(FILE *fp,int k){
  float scale=(nmodels)? 1.0f/(float)nmodels : 0.0f;
  if(!density){
   density=new float[nmodels];
   calculate_density();
  }
  priority_heap_list<float> my_list(density,k,nmodels,!greater_is_better,0);
  //list comes off in reverse order
  int *list =new int[k];
  for(int i=k-1;i>=0;--i)
   list[i]=my_list.pop_heap();
  for(int i=0;i<k;i++)
   fprintf(fp,"%s %8.5f\n",names(list[i]),density[list[i]]*scale);
  if(list) delete [] list;
 } 
 
 
 void calculate_density(){
   for (int i=0;i<nmodels;i++)
    density[i]=0;
   for (int i=1;i<nmodels;i++)
    for(int j=0;j<i;j++){
     float value=get_matrix(i,j);
     density[i]+=value;
     density[j]+=value;
    }
  }

 private:
  int find_worst_density(float *worst){
   int windex=0; 
   float wvalue=density[0]; 
   for(int i=1;i<nmodels;i++)
   {
    if(better(wvalue,density[i])){
     windex=i;
     wvalue=density[i];
    }
   }
   *worst=wvalue;
   return(windex);
  }
  
  template <class T2>
  bool better (T2 a, T2 b){   
   return ((greater_is_better && a>b) || (!greater_is_better && a<b));
  }
};
template <class T>
class cluster_partition { //the cluster membership - with a set of models provides the entire description of the clusters
 friend class mapped_cluster_set<T>;
 friend class parallel_cluster_set<T>;
 //partition is initialized with random cluster_centers - not cluster_ids 
 public:
  int nmodels;
  int nclusters;
  int *cluster_ids;
  int *cluster_sizes;
  float *cluster_density;
  int *cluster_centers;;
  int min_cluster_size;
  int *randi;
  float recalc_density_ratio; //when to recalculate density rather than adjust density
  bool greater_is_better;
  float score;
  unsigned int seed;
  cluster_partition(int input_nmodels,int input_nclusters, int input_min_cluster_size,bool is_better,float input_score, unsigned int input_seed) : score(input_score) {
   nclusters=input_nclusters;nmodels=input_nmodels;min_cluster_size=input_min_cluster_size,seed=input_seed;greater_is_better=is_better;
   cluster_ids=new int[nmodels];
   cluster_sizes = new int[nclusters];
   cluster_centers = new int[nclusters];
   cluster_density = new float[nclusters*nmodels];
   randi=new int[nmodels];
   for(int i=0;i<nmodels;i++)
    randi[i]=i;
   recalc_density_ratio=0; 
  }   
  cluster_partition(int input_nmodels,int input_nclusters, int input_min_cluster_size,bool is_better,float input_score, unsigned int input_seed, int density_size) : score(input_score) {
   //used for agglomeration as cluster_density is not required
   nclusters=input_nclusters;nmodels=input_nmodels;min_cluster_size=input_min_cluster_size,seed=input_seed;greater_is_better=is_better;
   cluster_ids=new int[nmodels];
   cluster_sizes = new int[nclusters];
   cluster_centers = new int[nclusters];
   if(density_size)
    cluster_density = new float[density_size];
   else cluster_density=0;
   randi=new int[nmodels];
   for(int i=0;i<nmodels;i++)
    randi[i]=i;
   recalc_density_ratio=0; 
  }
  cluster_partition(const cluster_partition &A) : nmodels(A.nmodels),nclusters(A.nclusters),score(A.score),min_cluster_size(A.min_cluster_size),greater_is_better(A.greater_is_better),seed(A.seed), recalc_density_ratio(A.recalc_density_ratio){
   cluster_ids=new int[nmodels];memmove(cluster_ids,A.cluster_ids,nmodels*sizeof(int));
   cluster_sizes = new int[nclusters];memmove(cluster_sizes,A.cluster_sizes,nclusters*sizeof(int));
   cluster_centers = new int[nclusters];memmove(cluster_centers,A.cluster_centers,nclusters*sizeof(int));
   cluster_density = new float[nmodels*nclusters];memmove(cluster_density,A.cluster_density,nclusters*nmodels*sizeof(float));
   randi=new int[nmodels];memmove(randi,A.randi,nmodels*sizeof(int));
  }
  cluster_partition & operator = (const cluster_partition &rhs){
   if(this != &rhs){
    nmodels=rhs.nmodels;nclusters=rhs.nclusters;score=rhs.score;min_cluster_size=rhs.min_cluster_size;greater_is_better=rhs.greater_is_better,seed=rhs.seed;
    if(cluster_ids) delete [] cluster_ids;
    if(cluster_sizes) delete [] cluster_sizes;
    if(cluster_centers)delete [] cluster_centers;
    if(cluster_density)delete [] cluster_density;
    if(randi)delete [] randi;
    cluster_ids=new int[nmodels];memmove(cluster_ids,rhs.cluster_ids,nmodels*sizeof(int));
    cluster_sizes = new int[nclusters];memmove(cluster_sizes,rhs.cluster_sizes,nclusters*sizeof(int));
    cluster_centers = new int[nclusters];memmove(cluster_centers,rhs.cluster_centers,nclusters*sizeof(int));
    cluster_density = new float[nmodels*nclusters];memmove(cluster_density,rhs.cluster_density,nclusters*nmodels*sizeof(float));
    randi=new int[nmodels];memmove(randi,rhs.randi,nmodels*sizeof(int));
    recalc_density_ratio=rhs.recalc_density_ratio;
   }
   return *this ; 
  }
  void fast_copy(cluster_partition *dest){
    memmove(dest->cluster_ids,cluster_ids,nmodels*sizeof(int));
    memmove(dest->cluster_sizes,cluster_sizes,nclusters*sizeof(int));
    memmove(dest->cluster_centers,cluster_centers,nclusters*sizeof(int));
    memmove(dest->cluster_density,cluster_density,nmodels*nclusters*sizeof(float));
    memmove(dest->randi,randi,nmodels*sizeof(int));
    dest->recalc_density_ratio=recalc_density_ratio ; 
    dest->score=score;
  } 

  ~cluster_partition(){  
   if(cluster_ids) delete [] cluster_ids;
   if(cluster_sizes) delete [] cluster_sizes;
   if(cluster_centers)delete [] cluster_centers;
   if(cluster_density)delete [] cluster_density;
   if(randi) delete [] randi;
  }

  int kcenters (int max_initial_iterations,int max_convergence_iterations,mapped_cluster_models_set <T> *models, int (cluster_partition::*f_assign)(mapped_cluster_models_set <T>*)){
    initialize_partition_by_distance(max_initial_iterations,&seed,models,nclusters);
    //convergence is when there are no changes to clusters, target function goes up or min requirements not reached
    int niter=0;
    int nchanges=0;
    while(niter <max_convergence_iterations){
     nchanges=(this->*f_assign)(models);
     if(nchanges<=0)break;
     niter++;
    } 
    find_cluster_density(models);
    sort_clusters_insort();
   find_center_structures_using_density_of_cluster();
   fprintf(stderr,"final score %8.5f\n",score);
  }
  

  int kcluster(int max_initial_iterations,int max_convergence_iterations,mapped_cluster_models_set <T> *models, int (cluster_partition::*f_assign)(mapped_cluster_models_set<T>*),int nfixed_centers, bool initialize_flag){//generalized kcluster method
   cluster_partition p=*this;
   if(initialize_flag){
    //find inital valid partition
    if(!p.initialize_partition_by_distance(max_initial_iterations,&seed,models,nfixed_centers))return(-1); //no valid partitions
   } 
   //convergence is when there are no changes to clusters, target function goes up or min requirements not reached
   int niter=0;
   int nchanges=0;

   while(niter <max_convergence_iterations){
    nchanges=(p.*f_assign)(models);
    if(nchanges<=0)break;
    niter++;
   }
   if(better(p.score,score)){
    p.find_center_structures_using_density_of_cluster(); //match center structures to density
    *this=p;
    return(1);
   }
   return(0);
  }

  int parallel_kcluster(int nthreads, int max_initial_iterations,int max_convergence_iterations,mapped_cluster_models_set <T> *models, int (cluster_partition::*f_assign)(mapped_cluster_models_set<T>*,int),int nfixed_centers, bool initialize_flag){//generalized kcluster method
   cluster_partition p=*this;
   if(initialize_flag){
    //find inital valid partition
    if(!p.initialize_partition_by_distance(max_initial_iterations,&seed,models,nfixed_centers))return(-1); //no valid partitions
   } 
   //convergence is when there are no changes to clusters, target function goes up or min requirements not reached
   int niter=0;
   int nchanges=0;

   while(niter <max_convergence_iterations){
    nchanges=(p.*f_assign)(models,nthreads);
    if(nchanges<=0)break;
    niter++;
   }
   if(better(p.score,score)){
    p.find_center_structures_using_density_of_cluster(nthreads); //match center structures to density
    *this=p;
    return(1);
   }
   return(0);
  }
  

 int reduce_by_agglomeration_single_linkage(int final_nclusters,mapped_cluster_models_set<T> *models, int *history, bool history_only,bool initialize, int nthreads){
  //check on max threads
#ifdef OPENMP  
  int max_threads=omp_get_max_threads();
  int my_max_threads=(gcpu_info.hyperthreads)? gcpu_info.cores/2: gcpu_info.cores ;
  if(my_max_threads>max_threads) my_max_threads=max_threads;
  if(my_max_threads <1) my_max_threads=1;
#endif
  int nmap=nclusters;
  int *cmap =new int [nmap];//mapping changes to reflect joining of clusters - ids don't change but the map to them does
  for(int i =0;i<nmap;i++)
   cmap[i]=i;
  int nchanges=0;
  //initialize by asigning clusters - usually start with nclusters same as nmodels or from another cluster method
  if(initialize){
   fprintf(stderr,"initializing clusters\n");
   for(int i=0;i<nmap;i++){
    cluster_ids[i]=i;
    cluster_sizes[i]=1;
   }
  }
  //store map distances in a triangular matrix - need
  triangular_matrix<T>* cd_matrix= new triangular_matrix<T>(nclusters,0,0,greater_is_better);
  cd_matrix->min_value=models->base_set->dmatrix->min_value;
  cd_matrix->max_value=models->base_set->dmatrix->max_value;
  cd_matrix->step_size=models->base_set->dmatrix->step_size;
  cd_matrix->inv_step_size=models->base_set->dmatrix->inv_step_size;

  if(nmodels == nclusters){
#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads)  schedule(dynamic)    
#endif  
   for(int i=1;i<nclusters;i++){
    T* const matrix=cd_matrix->tmatrix[i]; //know that i,j are ordered here so access the array directly and save some cycles
    for(int j=0;j<i;j++){
     matrix[j]=models->get_matrix_fast(i,j); //for compact this also gives unmapped values
    }
   } 
  }
  else{ //find min distances   
   int *cluster_seen=new int [nclusters];
   memset(cluster_seen,0,nclusters*sizeof(int)); 
#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)     
#endif        
   for(int i=1;i<nmodels;i++){
    int ci= cluster_ids[i];
    T* const matrix= cd_matrix->tmatrix[ci];
    for(int j=0;j<i;j++){
     T dist=models->get_matrix_fast(i,j);
     if(!cluster_seen[ci]++ || better(dist,matrix[cluster_ids[j]])){
      matrix[cluster_ids[j]]=dist;
     }
    }
   }
   if(cluster_seen) delete [] cluster_seen;
  }
  fprintf(stderr,"starting agglomeration\n");

  T *min_dists= new T [nclusters];
  int *closest_cluster= new int [nclusters];
  
  memset(closest_cluster,0,nclusters*sizeof(int));
  double start = get_time();
   
  //initialize minimum dists array
  //keep track of closest cluster for each remaining cluster
  //per step recalculation de novo is O(N**2) while update is  O(N) to O(N**2) worst case when all clusters are closest to the join clusters 
  //most cases update should be very close to O(N) especially for large N - I believe that Murtagh used similar approach
  
  //there is a better algorithm from Mullner where it is not necessary to keep closest cluster but only closest cluster
  //from higher indexed clusters since the global minimum is still present in the ensemble of min_dists
  //this saves quite a few updates and searches - algorithm is still quadratic to cubic but this approach is more than three
  //times faster because of fewer comparisons and updates
  
  //I use lower triangular matrix instead instead to take advantage of array slice caching
  //It is possible for even faster access for smaller arrays using a single vector and index mapping than a triangular matrix class
  //Not sure whether this is better or more portable than array slices for very large arrays when this starts to matter
  //A heap structure for storing nn distances may be faster for large arrays but requires a lot of bookkeepping to keep track of indices 
  //The linear search is not a large factor even for small arrays (10% of CPU at 10K) and becomes less of a factor for large arrays as the other elements grow N**2
  //perhaps a heap of heaps structure or paged heap structure might be worthwhile for very large sets as that would decrease the time of the updates and
  //is parallelizable
  
  //for single linkage the minimum distance of elements in two cluster determines the cluster distance
  //cluster min_dists don't change after merge
  //cluster nearest neighbour doesn't change unless cidx2 is one of the clusters in which case nearest neigbour becomes cidx1
  //still must update distances and nearest neighbour for cidx1
  
  fprintf(stderr,"finding closest clusters\n");
#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)      
#endif    
  for(int i=1;i<nclusters;i++){
   T* const matrix=cd_matrix->tmatrix[i];
   min_dists[i]=matrix[0];
   closest_cluster[i]=0;       
   for(int j=1;j<i;j++){
    if(better(matrix[j],min_dists[i])){
     min_dists[i]=matrix[j];
     closest_cluster[i]=j;
    }  
   }
  }   
  
  fprintf(stderr,"%8.5f ms to find closest clusters\n",(get_time()-start)*1000.0f);

   //agglomeration loop
   //closest two clusters are joined
   //cluster sizes updated -
   //all cluster distances involving joined cluster are updated and single comparison with closest cluster and new distance compared
   //if the closest cluster is the joined cluster then the entire set of N distances must be compared again 
   //adjust cmap (change pointer to deleted points to what the last pointer points to and eliminate last pointer)   
   //min_dist[i] has min(dist(i,j)j<i)


   double min_time=0,update_min_time=0,update_matrix_time=0,check_matrix_time=0,update_bad_matrix_time=0,delete_time=0;
   while(nmap > final_nclusters){ 
    int idx1=nmap-1;;
    T min_dist=min_dists[cmap[nmap-1]]; 
    double start_min=get_time();
    //not worth parallelizing at 10,000 decoys - maybe at 100000
    for(int i=nmap-2;i>0;i--){
     if(better(min_dists[cmap[i]],min_dist)){
      min_dist=min_dists[cmap[i]];
      idx1=i;
     }
    }

    min_time+=(get_time()-start_min);

    //global minimum dist is correct - the upper index will have the min distance
    //cmaps are adjusted so that the mapping is always monotonic i.e. i>j => cmap[i] > cmap[j]
    //cmaps hold the original indexes - this is where all the bookeeping of nearest neighbours is done
    
    int cidx1=cmap[idx1];
    int cidx2=closest_cluster[cidx1];
 
    int idx2=0;
    while(cmap[idx2] != cidx2) idx2++;
     
    double part_time=get_time();
    T* const matrix1=cd_matrix->tmatrix[cidx1];
    T* const matrix2=cd_matrix->tmatrix[cidx2];
#ifdef OPENMP
    T tmin[nthreads];
    int tcc[nthreads];
    int nt = (nthreads >my_max_threads) ? my_max_threads : nthreads;    
    if(nt >1){
     #pragma omp parallel num_threads(nt)
     {
      int t=omp_get_thread_num();
      int k=t;
      T merged_min_dist=0;
      int merged_cc=-1;
        
      for(;k<idx2;k+=nt){
       int i=cmap[k];
       if(better(matrix2[i],matrix1[i])){
        matrix1[i] = matrix2[i];
       }
       if(merged_cc<0 || better(matrix1[i],merged_min_dist)){ 
        merged_min_dist=matrix1[i];
        merged_cc=i;
       } 
      }
      if(k == idx2)k+=nt; 
      for(;k<idx1;k+=nt){
       int i=cmap[k];
       if(better(cd_matrix->tmatrix[i][cidx2],matrix1[i])){
        matrix1[i] = cd_matrix->tmatrix[i][cidx2];
        //cidx2 can only be nearest neighbour when the if condition is met   
       }
       //can be more than one equally cluster so cannot put the next if statement within
       //previous one 
       if(merged_cc<0  || better(matrix1[i],merged_min_dist)){ 
        merged_min_dist=matrix1[i];
        merged_cc=i;
       }
       if(closest_cluster[i]==cidx2){
        closest_cluster[i]=cidx1;
       }
      }
      tmin[t]=merged_min_dist; //for reduction make a copy
      tcc[t]=merged_cc; 
      if(k==idx1)k+=nt;  
      for(;k<nmap;k+=nt){
       int i=cmap[k];    
       T* const matrix=cd_matrix->tmatrix[i];
       if((better(matrix[cidx2],matrix[cidx1]))){
        matrix[cidx1]=matrix[cidx2];       
       }
       if(closest_cluster[i] == cidx2) closest_cluster[i]=cidx1;
      }      
    }//end parallel
    T min_dist1=0;
    int cc1 =-1;
    for(int th=0;th<nt;th++){
     //reduction condition
     //first condition is that the thread actually ran through and got initiated
     //second condition is that first valid thread or min_dist1 is worse than thread min
     //third condition is so that lowest index of equal distances is used so that result is independent of threads usd
     if(tcc[th] >=0 
       && (cc1 < 0 || better(tmin[th],min_dist1) ) || (tmin[th] == min_dist1 && cc1 > tcc[th])){ 
      min_dist1=tmin[th];
      cc1=tcc[th];       
     }
    }
    min_dists[cidx1]=min_dist1;
    closest_cluster[cidx1]=cc1;      
   }
   else
#endif     
    { 
     int k=0;
     T merged_min_dist=0;
     int merged_cc=-1;
     for(;k<idx2;k++){
      int i=cmap[k];
      if(better(matrix2[i],matrix1[i])){
       matrix1[i] = matrix2[i];
      }
      if(merged_cc<0 || better(matrix1[i],merged_min_dist)){ 
       merged_min_dist=matrix1[i];
       merged_cc=i;
      } 
     }
     k++;
     for(;k<idx1;k++){
      int i=cmap[k];
      if(better(cd_matrix->tmatrix[i][cidx2],matrix1[i])){
       matrix1[i] = cd_matrix->tmatrix[i][cidx2];  
      }
      if(closest_cluster[i] == cidx2) closest_cluster[i]=cidx1;
      if(merged_cc <0  || better(matrix1[i],merged_min_dist)){ 
       merged_min_dist=matrix1[i];
       merged_cc=i;
      }
     }
     min_dists[cidx1]=merged_min_dist;
     closest_cluster[cidx1]=merged_cc; 
     k++;  
     for(;k<nmap;k++){
      int i=cmap[k];
      int change_flag=0;      
      T* const matrix=cd_matrix->tmatrix[i];
      if(better(matrix[cidx2],matrix[cidx1])){
       matrix[cidx1]=matrix[cidx2];       
      }
      if(closest_cluster[i] == cidx2) closest_cluster[i]=cidx1;
     }
    }
    update_min_time+=get_time()-part_time;   
    //remove cidx1 from list
    part_time=get_time();
    cluster_sizes[cidx1]+=cluster_sizes[cidx2];
    for(int k=idx2;k<nmap-1;k++)
     cmap[k]=cmap[k+1];
    if(history){
     history[2*nchanges]=cidx2;
     history[2*nchanges+1]=cidx1;
     nchanges++;    
    }
    delete_time+=get_time()-part_time;  
    nmap--;
   }//end while
   
   if(!history_only && final_nclusters >1){
    int *pamc= new int[nclusters];
    memset(pamc,-1,nclusters*sizeof(int));
    for(int i=0;i<nmap;i++)
     pamc[cmap[i]]=i;
    //commit changes to partition
    //allocate new density centers and sizes

    int* final_cluster_sizes=new int[nmap];
    int* final_cluster_centers=new int[nmap];
    float* final_cluster_density=new float [nmap*nmodels];
    for (int i=0;i<nmap;i++){
     final_cluster_sizes[i]=cluster_sizes[cmap[i]];
    }
    fprintf(stderr,"%8.5f ms to cluster \n",(get_time()-start)*1000.0f);
 
    if(cluster_density)delete [] cluster_density;
    if(cluster_sizes)delete [] cluster_sizes;
    if(cluster_centers)delete [] cluster_centers; 
    start = get_time();
    adjust_cluster_ids_for_agglomerations(nchanges,history);// cluster_ids are in original
    for(int i=0;i<nmodels;i++){
     //fprintf(stderr,"%d %d %d\n",i,cluster_ids[i],pamc[cluster_ids[i]]);
     cluster_ids[i]=pamc[cluster_ids[i]];
    }  
    fprintf(stderr,"%8.5f ms to calculate final cluster membership \n",(get_time()-start)*1000.0f); 
    cluster_density=final_cluster_density;
    cluster_sizes=final_cluster_sizes;
    cluster_centers=final_cluster_centers;
    nclusters=nmap;
    //adjust cluster_ids    
    find_cluster_density(models);
    sort_clusters_insort();
    find_center_structures_using_density_of_cluster(nthreads);
    //this->print(stderr);  
    if(pamc) delete [] pamc;
   }
   if(cmap) delete [] cmap; 
   if(cd_matrix) delete cd_matrix;
   if(min_dists)delete [] min_dists;
   if(closest_cluster)delete [] closest_cluster;
   return(nchanges/2);
 }

 int reduce_by_agglomeration_complete_linkage(int final_nclusters,mapped_cluster_models_set <T> *models, int *history, bool history_only,bool initialize, int nthreads){

#ifdef OPENMP  
  int max_threads=omp_get_max_threads();
  int my_max_threads=(gcpu_info.hyperthreads)? gcpu_info.cores/2: gcpu_info.cores ;
  if(my_max_threads>max_threads) my_max_threads=max_threads;
  if(my_max_threads <1) my_max_threads=1;
#endif  
  int nmap=nclusters;
  int *cmap =new int [nmap];//mapping changes to reflect joining of clusters - ids don't change but the map to them does
  for(int i =0;i<nmap;i++)
   cmap[i]=i;
  int nchanges=0;  
  //initialize by asigning clusters - usually start with nclusters same as nmodels or from another cluster method
  if(initialize){
   fprintf(stderr,"initializing clusters\n");
   for(int i=0;i<nmap;i++){
    cluster_ids[i]=i;
    cluster_sizes[i]=1;
   }
  }
  //store map distances in a triangular matrix - nee
  triangular_matrix<T>* cd_matrix= new triangular_matrix<T>(nclusters,0,0,greater_is_better);
  cd_matrix->min_value=models->base_set->dmatrix->min_value;
  cd_matrix->max_value=models->base_set->dmatrix->max_value;
  cd_matrix->step_size=models->base_set->dmatrix->step_size;
  cd_matrix->inv_step_size=models->base_set->dmatrix->inv_step_size;

  if(nmodels == nclusters){
#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)  
#endif  
   for(int i=1;i<nclusters;i++){
    T* const matrix=cd_matrix->tmatrix[i]; //know that i,j are ordered here so access the array directly and save some cycles
    for(int j=0;j<i;j++){
     matrix[j]=models->get_matrix_fast(i,j);
    }
   } 
  }
  else{ //find max distances
   int *cluster_seen=new int [nclusters];
   memset(cluster_seen,0,nclusters*sizeof(int));

#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)    
#endif        
   for(int i=1;i<nmodels;i++){
    int ci= cluster_ids[i];
    T* const matrix= cd_matrix->tmatrix[ci];
    for(int j=0;j<i;j++){
     T dist=models->get_matrix_fast(i,j);
     if(!cluster_seen[ci]++ || better(matrix[cluster_ids[j]],dist)){
      matrix[cluster_ids[j]]=dist;
      cluster_seen[i]++;
     }
    }
   }
   if(cluster_seen) delete [] cluster_seen;
  }
  
  fprintf(stderr,"starting agglomeration\n");
  T *min_dists= new T [nclusters];
  int *closest_cluster= new int [nclusters];
  
  memset(closest_cluster,0,nclusters*sizeof(int));
  double start = get_time();
  T min_dist;
   
  //initialize minimum dists array
  //keep track of closest cluster for each remaining cluster
  //per step recalculation de novo is O(N**2) while update is  O(N) to O(N**2) worst case when all clusters are closest to the join clusters 
  //most cases update should be very close to O(N) especially for large N - I believe that Murtagh used similar approach
  
  //there is a better algorithm from Mullner where it is not necessary to keep closest cluster but only closest cluster
  //from higher indexed clusters since the global minimum is still present in the ensemble of min_dists
  //this saves quite a few updates and searches - algorithm is still quadratic to cubic but this approach is more than three
  //times faster because of fewer comparisons and updates
  
  //I use lower triangular matrix instead instead to take advantage of array slice caching
  //It is possible for even faster access for smaller arrays using a single vector and index mapping than a triangular matrix class
  //Not sure whether this is better or more portable than array slices for very large arrays when this starts to matter
  //A heap structure for storing nn distances may be faster for large arrays but requires a lot of bookkeepping to keep track of indices 
  //The linear search is not a large factor even for small arrays (10% of CPU at 10K) and becomes less of a factor for large arrays as the other elements grow N**2
  //perhaps a heap of heaps structure or paged heap structure might be worthwhile for very large sets as that would decrease the time of the updates and
  //is parallelizable
  
  fprintf(stderr,"finding closest clusters\n");
#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)  
#endif    
  for(int i=1;i<nclusters;i++){    
   T* const matrix=cd_matrix->tmatrix[i];
   min_dists[i]=matrix[0];
   closest_cluster[i]=0;
   for(int j=1;j<i;j++){
    if(better(matrix[j],min_dists[i])){
     min_dists[i]=matrix[j];
     closest_cluster[i]=j;
    }  
   }
  }
  fprintf(stderr,"%8.5f ms to find closest clusters\n",(get_time()-start)*1000.0f);
   //agglomeration loop
   //closest two clusters are joined
   //cluster sizes updated -
   //all cluster distances involving joined cluster are updated and single comparison with closest cluster and new distance compared
   //if the closest cluster is the joined cluster then the entire set of N distances must be compared again 
   //adjust cmap (change pointer to deleted points to what the last pointer points to and eliminate last pointer)   
   //min_dist[i] has min(dist(i,j)j<i)

   double min_time=0,update_min_time=0,update_matrix_time=0,check_matrix_time=0,update_bad_matrix_time=0,delete_time=0;
   while(nmap > final_nclusters){
    T min_dist=min_dists[cmap[nmap-1]];
    int idx1=nmap-1;   
    double start_min=get_time();
    
    //not worth parallelizing at 10,000 decoys - maybe at 100000
   
    for(int i=nmap-2;i>0;i--){
     if(better(min_dists[cmap[i]],min_dist)){
      min_dist=min_dists[cmap[i]];
      idx1=i;
     }
    }

    min_time+=(get_time()-start_min);

    //global minimum dist is correct - the upper index will have the min distance
    //cmaps are adjusted so that the mapping is always monotonic i.e. i>j => cmap[i] > cmap[j]
    //cmaps hold the original indexes - this is where all the bookeeping of nearest neighbours is done
    //eliminate lower index keeping higher index allows for more efficient caching as it points to bigger array slice and alsp fewer updates to matrix
    
    //update min_dists[idx1] and update cd_matrix for indices > idx1 - for complete linkage - it is the greater of cd_matrix(j,idx2) cd_matrix(j,idx1)
    //at same time 
    int cidx1=cmap[idx1];
    int cidx2=closest_cluster[cidx1];
 
    int idx2=0;
    while(cmap[idx2] != cidx2) idx2++;
     
    T* const matrix1=cd_matrix->tmatrix[cidx1];
    T* const matrix2=cd_matrix->tmatrix[cidx2];
    
    
#ifdef OPENMP
    T tmin[nthreads];
    int tcc[nthreads];
    int nt = (nthreads >my_max_threads) ? my_max_threads : nthreads;    
    if(nt >1){
     #pragma omp parallel num_threads(nt)
     {
      int t=omp_get_thread_num();
      int k=t;
      T merged_min_dist=0;
      int merged_cc=-1;
      for(;k<idx2;k+=nt){
       int i=cmap[k];
       if(better(matrix1[i],matrix2[i])){
        matrix1[i] = matrix2[i];
       }
       if(merged_cc<0 || better(matrix1[i],merged_min_dist)){ 
        merged_min_dist=matrix1[i];
        merged_cc=i;
       } 
      }
      if(k == idx2)k+=nt; 
      for(;k<idx1;k+=nt){
       int i=cmap[k];
       if(better(matrix1[i],cd_matrix->tmatrix[i][cidx2])){
        matrix1[i] = cd_matrix->tmatrix[i][cidx2];
       }
       if(merged_cc<0 || better(matrix1[i],merged_min_dist)){ 
        merged_min_dist=matrix1[i];
        merged_cc=i;
       }
       if(closest_cluster[i]==cidx2){//this is going to be eliminated
        T min_dist1=0;
        int cc1=-1;
        T* const matrixi=cd_matrix->tmatrix[i];
        int m=0;
        for(;m<idx2;m++){
         int j=cmap[m];
         if(cc1 <0 || better(matrixi[j],min_dist1)){
          min_dist1=matrixi[j];
          cc1=j;      
         }
        }
        m++;
        for(;m<k;m++){
         int j=cmap[m];
         if(cc1 <0 || better(matrixi[j],min_dist1)){
          min_dist1=matrixi[j];
          cc1=j;      
         }
        }     
        min_dists[i]=min_dist1;
        closest_cluster[i]=cc1;
       }      
      }
      tmin[t]=merged_min_dist;
      tcc[t]=merged_cc; 
      if(k==idx1)k+=nt;  
      for(;k<nmap;k+=nt){
       int i=cmap[k];
       int change_flag=0;      
       T* const matrix=cd_matrix->tmatrix[i];
       T v2=matrix[cidx2];
       if((better(matrix[cidx1],v2))){
        matrix[cidx1]=v2;
        change_flag=1;
       }
       if(closest_cluster[i] == cidx2){
        closest_cluster[i]=cidx1;
        if(min_dists[i] != matrix[cidx1]) change_flag=1;
       }
       //if the closest cluster is cmin_i or merged_cluster then re-update unless the distance to cidx1 is same as min_dists[i]
       if((closest_cluster[i] == cidx1 && change_flag)){
        int cc1=cidx1;
        T min_dist1=matrix[cidx1];
        int m=0;
        for(;m<idx2;m++){
         int j=cmap[m];
         if(better(matrix[j],min_dist1)){
          min_dist1=matrix[j];
          cc1=j;      
         }
        }
        m++;
        for(;m<idx1;m++){
         int j=cmap[m];
          if(better(matrix[j],min_dist1)){
          min_dist1=matrix[j];
          cc1=j;      
         }
        }
        m++;  
        for(;m<k;m++){
         int j=cmap[m];
         if(better(matrix[j],min_dist1)){
          min_dist1=matrix[j];
          cc1=j;      
         }
        }     
       min_dists[i]=min_dist1;
       closest_cluster[i]=cc1;
      }
     }      
    }//end parallel
    T min_dist1=0;
    int cc1=-1;
    for(int th=0;th<nt;th++){
     if(tcc[th] >=0 && (cc1 < 0 || better(tmin[th],min_dist1)) || (tmin[th] == min_dist1 && cc1 > tcc[th])){ //the second condition makes the result the same regardless of nt
      min_dist1=tmin[th];
      cc1=tcc[th];       
     }
    }
    min_dists[cidx1]=min_dist1;
    closest_cluster[cidx1]=cc1;;          
   }
   else
#endif     
    { 
     int k=0;
     T merged_min_dist=0;
     int merged_cc=-1;
     for(;k<idx2;k++){
      int i=cmap[k];
      if(better(matrix1[i],matrix2[i])){
       matrix1[i] = matrix2[i];
      }
      if(merged_cc <0 || better(matrix1[i],merged_min_dist)){ 
       merged_min_dist=matrix1[i];
       merged_cc=i;
      } 
     }
     k++;
     for(;k<idx1;k++){
      int i=cmap[k];
      if(better(matrix1[i],cd_matrix->tmatrix[i][cidx2])){
       matrix1[i] = cd_matrix->tmatrix[i][cidx2];
      }
      if(merged_cc <0 || better(matrix1[i],merged_min_dist)){ 
       merged_min_dist=matrix1[i];
       merged_cc=i;
      }
      if(closest_cluster[i]==cidx2){//this is going to be eliminated
       T min_dist1=0;
       int cc1=-1;
       T* const matrixi=cd_matrix->tmatrix[i];
       int m=0;
       for(;m<idx2;m++){
        int j=cmap[m];
        if(cc1 <0 || better(matrixi[j],min_dist1)){
         min_dist1=matrixi[j];
         cc1=j;      
        }
       }
       m++;
       for(;m<k;m++){
        int j=cmap[m];
        if(cc1 <0 || better(matrixi[j],min_dist1)){
         min_dist1=matrixi[j];
         cc1=j;      
        }
       }     
       min_dists[i]=min_dist1;
       closest_cluster[i]=cc1;
      }      
     }
     min_dists[cidx1]=merged_min_dist;
     closest_cluster[cidx1]=merged_cc; 
     k++;  
     for(;k<nmap;k++){
      int i=cmap[k];
      int change_flag=0;      
      T* const matrix=cd_matrix->tmatrix[i];
      T v2=matrix[cidx2];
      if((better(matrix[cidx1],v2))){
       matrix[cidx1]=v2;
       change_flag=1;
      }
      if(closest_cluster[i] == cidx2){
       closest_cluster[i]=cidx1;
       if(min_dists[i] != matrix[cidx1]) change_flag=1;
      }
      //if the closest cluster is cmin_i or merged_cluster then re-update unless the distance to cidx1 is same as min_dists[i]
      if((closest_cluster[i] == cidx1 && change_flag)){
       int cc1=cidx1;
       T min_dist1=matrix[cidx1];
       int m=0;
       for(;m<idx2;m++){
        int j=cmap[m];
        if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }
       m++;
       for(;m<idx1;m++){
        int j=cmap[m];
         if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }
       m++;  
       for(;m<k;m++){
        int j=cmap[m];
        if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }     
       min_dists[i]=min_dist1;
       closest_cluster[i]=cc1;
      }
     }
    }
   
    //remove cidx1 from list
    cluster_sizes[cidx1]+=cluster_sizes[cidx2];
    for(int k=idx2;k<nmap-1;k++)
     cmap[k]=cmap[k+1];
    if(history){
     history[2*nchanges]=cidx2;
     history[2*nchanges+1]=cidx1;
     nchanges++;    
    }
    nmap--;
   }//end while
   
   if(!history_only && final_nclusters >1){
    int *pamc= new int[nclusters];
    memset(pamc,-1,nclusters*sizeof(int));
    for(int i=0;i<nmap;i++)
     pamc[cmap[i]]=i;
    //commit changes to partition
    //allocate new density centers and sizes
    
    int* final_cluster_sizes=new int[nmap];
    int* final_cluster_centers=new int[nmap];
    float* final_cluster_density=new float [nmap*nmodels];
    for (int i=0;i<nmap;i++){
     final_cluster_sizes[i]=cluster_sizes[cmap[i]];
    }
    fprintf(stderr,"%8.5f ms to cluster \n",(get_time()-start)*1000.0f);
 
    if(cluster_density)delete [] cluster_density;
    if(cluster_sizes)delete [] cluster_sizes;
    if(cluster_centers)delete [] cluster_centers; 
    start = get_time();
    adjust_cluster_ids_for_agglomerations(nchanges,history);// cluster_ids are in original
    for(int i=0;i<nmodels;i++){
     //fprintf(stderr,"%d %d %d\n",i,cluster_ids[i],pamc[cluster_ids[i]]);
     cluster_ids[i]=pamc[cluster_ids[i]];
    }  
    fprintf(stderr,"%8.5f ms to calculate final cluster membership \n",(get_time()-start)*1000.0f); 
    cluster_density=final_cluster_density;
    cluster_sizes=final_cluster_sizes;
    cluster_centers=final_cluster_centers;
    nclusters=nmap;
    //adjust cluster_ids    
    find_cluster_density(models);
    sort_clusters_insort();
    find_center_structures_using_density_of_cluster(nthreads);
    //this->print(stderr);  
    if(pamc) delete [] pamc;
   }
   if(cmap) delete [] cmap; 
   if(cd_matrix) delete cd_matrix;
   if(min_dists)delete [] min_dists;
   if(closest_cluster)delete [] closest_cluster;
   return(nchanges/2);
 }
      
 int reduce_by_agglomeration_average(int final_nclusters,mapped_cluster_models_set <T> *models, int *history, bool history_only,bool initialize, int nthreads){ //uses average cluster_distance
#ifdef OPENMP  
  int max_threads=omp_get_max_threads();
  int my_max_threads=(gcpu_info.hyperthreads)? gcpu_info.cores/2: gcpu_info.cores ;
  if(my_max_threads>max_threads) my_max_threads=max_threads;
  if(my_max_threads <1) my_max_threads=1;
#endif
  int nmap=nclusters;
  int *cmap =new int [nmap];//mapping changes to reflect joining of clusters - ids don't change but the map to them does
  for(int i =0;i<nmap;i++)
   cmap[i]=i;
  int nchanges=0;
  float worst_value=better(1,0)? -FLT_MAX : FLT_MAX;
  float best_value=better (1,0)? FLT_MAX : -FLT_MAX;

  //initialize by asigning clusters - usually start with nclusters same as nmodels or from another cluster method
  if(initialize){
   fprintf(stderr,"initializing clusters\n");
   for(int i=0;i<nmap;i++){
    cluster_ids[i]=i;
    cluster_sizes[i]=1;
   }
  }
  //store map distances in a triangular matrix - these are sums of all the distances between members of the two clusters

  triangular_matrix<float>* cd_matrix= new triangular_matrix<float>(nclusters,0,0,greater_is_better);
  cd_matrix->min_value=models->base_set->dmatrix->min_value;
  cd_matrix->max_value=models->base_set->dmatrix->max_value;
  cd_matrix->step_size=models->base_set->dmatrix->step_size;
  cd_matrix->inv_step_size=models->base_set->dmatrix->inv_step_size;
  
  if(nmodels == nclusters){
#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)     
#endif  
   for(int i=1;i<nclusters;i++){
    float* const matrix=cd_matrix->tmatrix[i]; //know that i,j are ordered here so access the array directly and save some cycles
    for(int j=0;j<i;j++){
     matrix[j]=models->get_matrix_fast(i,j);
    }
   } 
  }
  else{ //find average distances
   float *finv_cluster_sizes=new float[nclusters];
#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)      
#endif  
   for(int i=1;i<nclusters;i++){
    float* const matrix=cd_matrix->tmatrix[i];
    finv_cluster_sizes[i]=1.0f/(float)cluster_sizes[i];
    for(int j=0;j<i;j++){
     matrix[j]=0.0f;
    }
   }
   for(int i=1;i<nmodels;i++){
    int a=cluster_ids[i];
    for(int j=0;j<i;j++){
     int b=cluster_ids[j];
     float value=models->get_matrix_fast(i,j);
     cd_matrix->set_matrix(a,b,cd_matrix->get_matrix(a,b)+value);
    }
   }
   for(int i=1;i<nclusters;i++){
    float finv_csize=finv_cluster_sizes[i];
    float* const matrix=cd_matrix->tmatrix[i];
    for(int j=0;j<i;j++){
     matrix[j]*=finv_csize*finv_cluster_sizes[j];
    }
   }
   delete[] finv_cluster_sizes;
  }
  fprintf(stderr,"starting agglomeration\n");
  float *min_dists= new float [nclusters];
  int *closest_cluster= new int [nclusters];
  
  memset(closest_cluster,0,nclusters*sizeof(int));
  for(int i=0;i<nclusters;i++){
   min_dists[i]=worst_value;
  }  
  double start = get_time();
  float min_dist;
   
  //initialize minimum dists array
  //keep track of closest cluster for each remaining cluster
  //per step recalculation de novo is O(N**2) while update is  O(N) to O(N**2) worst case when all clusters are closest to the join clusters 
  //most cases update should be very close to O(N) especially for large N - I believe that Murtagh used similar approach
  
  //there is a better algorithm from Mullner where it is not necessary to keep closest cluster but only closest cluster
  //from higher indexed clusters since the global minimum is still present in the ensemble of min_dists
  //this saves quite a few updates and searches - algorithm is still quadratic to cubic but this approach is more than two
  //times faster because of fewer comparisons
  
  //I use lower triangular matrix instead instead to take advantage of array slice caching
  //It is possible for even faster access for smaller arrays using a single vector and index mapping than a triangular matrix class
  //Not sure whether this is better or more portable than array slices for very large arrays when this starts to matter
  //A heap structure for storing nn distances may be faster for large arrays but requires a lot of bookkeepping to keep track of indices 
  //The linear search is not a large factor even for small arrays (5-10% of CPU at 10K) and becomes less of a factor for large arrays as the other elements grow N**2
  //perhaps a heap of heaps structure or paged heap structure might be worthwhile for very large sets as that would decrease the time of the updates and
  //is parallelizable
  
  //update for average is weighted by the two relative sizes of the merged cluster
  //dist= dist(cidx1,i)*cluster_size[cidx1]/(cluster_size[cidx1]+cluster_size[cidx2]) + dist(cidx2,i)*cluster_size[cidx2]/(cluster_size[cidx1]+cluster_size[cidx2])
  //since new dist can be bigger 

  //the alternative method which is may be less prone to round off errors but a bit slower is maintain the sums and generate the distances de novo by maintaining the sums of the distances
  //and dividng by the product of the clusters to generate the distances for the min_dists array

  fprintf(stderr,"finding closest clusters\n");
#ifdef OPENMP        
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)   
#endif    
  for(int i=1;i<nclusters;i++){    
   float* const matrix=cd_matrix->tmatrix[i];
   for(int j=0;j<i;j++){
    if(better(matrix[j],min_dists[i])){
     min_dists[i]=matrix[j];
     closest_cluster[i]=j;
    }  
   }
  }   
  
  fprintf(stderr,"%8.5f ms to find closest clusters\n",(get_time()-start)*1000.0f);

  double min_time=0,update_min_time=0,update_matrix_time=0,check_matrix_time=0,update_bad_matrix_time=0,delete_time=0;
  while(nmap > final_nclusters){
   int my_nthreads=nthreads;
   float min_dist=worst_value;
   int idx1=0;
   double start_min=get_time();
   
   //not worth parallelizing at 10,000 decoys - maybe at 100,000
   for(int i=nmap-1;i>0;i--){
    if(better(min_dists[cmap[i]],min_dist)){
     min_dist=min_dists[cmap[i]];
     idx1=i;
    }
   }

   min_time+=(get_time()-start_min);    
   int cidx1=cmap[idx1];
   int cidx2=closest_cluster[cidx1];

   int idx2=0;
   while(cmap[idx2] != cidx2) idx2++;
   int new_cluster_sizes;
   float f1=(float)cluster_sizes[cidx1];
   float f2=(float)cluster_sizes[cidx2];
   {
    float total=f1+f2;
    f1/=total;
    f2/=total;
   }
   float* const matrix1=cd_matrix->tmatrix[cidx1];
   float* const matrix2=cd_matrix->tmatrix[cidx2];
#ifdef OPENMP
   float tmin[nthreads];
   int tcc[nthreads];
   int nt = (nthreads >my_max_threads) ? my_max_threads : nthreads;
   if(nt >1){
    #pragma omp parallel num_threads(nt)
    {
     int t=omp_get_thread_num();
     int k=t;
     float merged_min_dist=worst_value;
     int merged_cc=0;
     for(;k<idx2;k+=nt){
      int i=cmap[k];
      matrix1[i] = f1*matrix1[i]+f2*matrix2[i];
      if(better(matrix1[i],merged_min_dist)){ 
       merged_min_dist=matrix1[i];
       merged_cc=i;
      } 
     }
     if(k == idx2)k+=nt; 
     for(;k<idx1;k+=nt){
      int i=cmap[k];
      matrix1[i] = f1*matrix1[i]+f2*cd_matrix->tmatrix[i][cidx2];
      if(better(matrix1[i],merged_min_dist)){ 
       merged_min_dist=matrix1[i];
       merged_cc=i;
      }
      if(closest_cluster[i]==cidx2){//this is going to be eliminated
       float min_dist1=worst_value;
       int cc1=0;
       float* const matrixi=cd_matrix->tmatrix[i];
       int m=0;
       for(;m<idx2;m++){
        int j=cmap[m];
        if(better(matrixi[j],min_dist1)){
         min_dist1=matrixi[j];
         cc1=j;      
        }
       }
       m++;
       for(;m<k;m++){
        int j=cmap[m];
        if(better(matrixi[j],min_dist1)){
         min_dist1=matrixi[j];
         cc1=j;      
        }
       }     
       min_dists[i]=min_dist1;
       closest_cluster[i]=cc1;
      }      
     }
     tmin[t]=merged_min_dist;
     tcc[t]=merged_cc;
     if(k==idx1)k+=nt;  
     for(;k<nmap;k+=nt){
      int i=cmap[k];
      float* const matrix=cd_matrix->tmatrix[i];
      matrix[cidx1]=f1*matrix[cidx1]+f2*matrix[cidx2];
      if(!better(matrix[cidx1], min_dists[i])){
       min_dists[i]=matrix[cidx1];
       closest_cluster[i]=cidx1;
      }
      else if(closest_cluster[i] == cidx1 || closest_cluster[i] == cidx2){
       int cc1=cidx1;
       float min_dist1=matrix[cidx1];
       int m=0;
       for(;m<idx2;m++){
        int j=cmap[m];
        if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }
       m++;
       for(;m<idx1;m++){
        int j=cmap[m];
         if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }
       m++;  
       for(;m<k;m++){
        int j=cmap[m];
        if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }     
      min_dists[i]=min_dist1;
      closest_cluster[i]=cc1;
     }
    }      
   }//end parallel
   float min_dist1=tmin[0];
   int cc1 =tcc[0];
   for(int th=1;th<nt;th++){
    if(better(tmin[th],min_dist1) || (tmin[th] == min_dist1 && cc1 > tcc[th])){ //the second condition makes the result the same regardless of nt
     min_dist1=tmin[th];
     cc1=tcc[th];       
    }
   }
   min_dists[cidx1]=min_dist1;
   closest_cluster[cidx1]=cc1;      
  }
  else
#endif     
   { 
    int k=0;
    float merged_min_dist=worst_value;
    int merged_cc=0;
    for(;k<idx2;k++){
     int i=cmap[k];
     matrix1[i] = f1*matrix1[i]+f2*matrix2[i];
     if(better(matrix1[i],merged_min_dist)){ 
      merged_min_dist=matrix1[i];
      merged_cc=i;
     } 
    }
    k++;
    for(;k<idx1;k++){
     int i=cmap[k];
     matrix1[i] = f1*matrix1[i]+f2*cd_matrix->tmatrix[i][cidx2];
     if(better(matrix1[i],merged_min_dist)){ 
      merged_min_dist=matrix1[i];
      merged_cc=i;
     }
     if(closest_cluster[i]==cidx2){//this is going to be eliminated
      float min_dist1=worst_value;
      int cc1=0;
      float* const matrixi=cd_matrix->tmatrix[i];
      int m=0;
      for(;m<idx2;m++){
       int j=cmap[m];
       if(better(matrixi[j],min_dist1)){
        min_dist1=matrixi[j];
        cc1=j;      
       }
      }
      m++;
      for(;m<k;m++){
       int j=cmap[m];
       if(better(matrixi[j],min_dist1)){
        min_dist1=matrixi[j];
        cc1=j;      
       }
      }     
      min_dists[i]=min_dist1;
      closest_cluster[i]=cc1;
     }     
    }
    min_dists[cidx1]=merged_min_dist;
    closest_cluster[cidx1]=merged_cc; 
    k++;  
    for(;k<nmap;k++){
      int i=cmap[k];
      float* const matrix=cd_matrix->tmatrix[i];
      matrix[cidx1]=f1*matrix[cidx1]+f2*matrix[cidx2];
      if(!better(matrix[cidx1], min_dists[i])){
       min_dists[i]=matrix[cidx1];
       closest_cluster[i]=cidx1;
      }
      else if(closest_cluster[i] == cidx1 || closest_cluster[i] == cidx2){
       int cc1=cidx1;
       float min_dist1=matrix[cidx1];
       int m=0;
       for(;m<idx2;m++){
        int j=cmap[m];
        if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }
       m++;
       for(;m<idx1;m++){
        int j=cmap[m];
         if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }
       m++;  
       for(;m<k;m++){
        int j=cmap[m];
        if(better(matrix[j],min_dist1)){
         min_dist1=matrix[j];
         cc1=j;      
        }
       }     
      min_dists[i]=min_dist1;
      closest_cluster[i]=cc1;
     }
    }
   }
 
   //remove cidx1 from list

   cluster_sizes[cidx1]+=cluster_sizes[cidx2];
   for(int k=idx2;k<nmap-1;k++)
    cmap[k]=cmap[k+1];
   if(history){
    history[2*nchanges]=cidx2;
    history[2*nchanges+1]=cidx1;
    nchanges++;    
   }
   nmap--;
  }//end while

  
  if(!history_only && final_nclusters >1){
   int *pamc= new int[nclusters];
    memset(pamc,-1,nclusters*sizeof(int));
    for(int i=0;i<nmap;i++)
     pamc[cmap[i]]=i;
    //commit changes to partition
    //allocate new density centers and sizes

    int* final_cluster_sizes=new int[nmap];
    int* final_cluster_centers=new int[nmap];
    float* final_cluster_density=new float [nmap*nmodels];
    for (int i=0;i<nmap;i++){
     final_cluster_sizes[i]=cluster_sizes[cmap[i]];
    }
    fprintf(stderr,"%8.5f ms to cluster \n",(get_time()-start)*1000.0f);
 
    if(cluster_density)delete [] cluster_density;
    if(cluster_sizes)delete [] cluster_sizes;
    if(cluster_centers)delete [] cluster_centers; 
    start = get_time();
    adjust_cluster_ids_for_agglomerations(nchanges,history);// cluster_ids are in original
    for(int i=0;i<nmodels;i++){
     //fprintf(stderr,"%d %d %d\n",i,cluster_ids[i],pamc[cluster_ids[i]]);
     cluster_ids[i]=pamc[cluster_ids[i]];
    }  
    fprintf(stderr,"%8.5f ms to calculate final cluster membership \n",(get_time()-start)*1000.0f); 
    cluster_density=final_cluster_density;
    cluster_sizes=final_cluster_sizes;
    cluster_centers=final_cluster_centers;
    nclusters=nmap;
    //adjust cluster_ids    
    find_cluster_density(models);
    sort_clusters_insort();
    find_center_structures_using_density_of_cluster(nthreads);
    //this->print(stderr);  
    if(pamc) delete [] pamc;
   }
   if(cmap) delete [] cmap; 
   if(cd_matrix) delete cd_matrix;
   if(min_dists)delete [] min_dists;
   if(closest_cluster)delete [] closest_cluster;
   return(nchanges/2);
 }
 void adjust_cluster_ids_for_agglomerations(int njoins, int *joins){
  //first join the clusters 
  for(int n=0;n<njoins;n++){
   int source=joins[2*n];
   int dest=joins[2*n+1];
   for(int i=0;i<nmodels;i++){
    if(cluster_ids[i]==source)cluster_ids[i]=dest;
   } 
  }
 }     
  void test_adjust_density(mapped_cluster_models_set <T> *models, float *slope, float *intercept){
   cluster_partition new_partition=*this;
   //figure out timings for 10, 30 ,50, 100
   float y[4];
   float x[4]={100.0f,50.0f,300.0f,500.0f};
   int ix[4]={100,50,300,500};
   new_partition.cluster_ids[0]=0;
   for(int k=0;k<4;k++){
    double start = get_time();
    for (int i=0;i<ix[k];i++){
     int s=new_partition.cluster_ids[0];
     int d=(s)? 0 : 1;
     adjust_density_for_cluster_change(0,s,d,models);
    }
    y[k]=1000.0f*(get_time()-start);
   }
   linear_regression(4,x,y,slope,intercept);
  }
  
  float test_find_density(mapped_cluster_models_set <T> *models){
   cluster_partition p=*this;
   for(int i=0;i<nmodels;i++){
    p.cluster_ids[i]=i%nclusters;
   } 
   double start = get_time();
   p.find_cluster_density(models);
   return((1000.0f)*(float)(get_time()-start));
  } 
  
  void linear_regression(int n, float *x, float *y, float *slope, float *intercept){
   float sxx=0,sxy=0,sx=0,sy=0;
   for(int i=0;i<n;i++){
    sx+=x[i];
    sy+=y[i];
    sxx+=x[i]*x[i];
    sxy+=x[i]*y[i];
   }
   float fn=(float) n;
   float m=(fn*sxy-sx*sy)/(fn*sxx-sx*sx);
   float b=(sy-m*sx)/fn;
   *slope=m;
   *intercept=b;
  }

  int parallel_assign_to_lowest_average_distance(mapped_cluster_models_set <T> *models,int nthreads){ //requires that an existing partition in ids and density cluster has been done
 #ifdef OPENMP
   if(nthreads <2)
     return (assign_to_lowest_average_distance(models));
   else{
   int retvalue=0;
   for (int i=0;i<nclusters;i++){
    if(cluster_sizes[i]<2)return(-1);
   }
   int *changes= new int [nmodels];
   int *new_ids= new int [nmodels];
   int *new_cluster_sizes= new int [nclusters];
   memset(changes,0,nmodels*sizeof(int));
   memmove(new_cluster_sizes,cluster_sizes,nclusters*sizeof(int));
   memmove(new_ids,cluster_ids,nmodels*sizeof(int));

   int nchanges=0;
   //cluster_partition new_partition=*this;
   
   float new_score=0;
   #pragma omp parallel for num_threads(nthreads)  schedule(dynamic) reduction(+:new_score)
   for(int i=0;i<nmodels;i++){
    int c=cluster_ids[i];
    int newc=c;
    float best_value=cdensity(i);
    int j=0;
    for( ;j< c ;j++){ 
     if( better(cdensity(i,j),best_value)){
      best_value=cdensity(i,j);
      newc=j;
     }
    }
    for(j=c+1 ;j< nclusters ;j++){ 
     if( better(cdensity(i,j),best_value)){
      best_value=cdensity(i,j);
      newc=j;
     }
    }    
    new_score+=best_value;
    //keep track of changes
    if(newc != c){
     #pragma omp critical
     {
      changes[nchanges++]=i;
      new_cluster_sizes[c]--;
      new_cluster_sizes[newc]++;
     }
     new_ids[i]=newc;
    }
   }
   //check cluster_sizes
   for (int j=0;j<nclusters;j++){
    if(new_cluster_sizes[j] < min_cluster_size){
     if(changes) delete [] changes;
     if(new_cluster_sizes) delete [] new_cluster_sizes;
     if(new_ids) delete [] new_ids;
     return(-1);
    }
   }

   if(nchanges){
   //check new score
    new_score/=(float)nmodels;
    if(better(new_score,score)){
     retvalue=1;
     score=new_score;
     //update cluster here
     //decide whether to update existing densities or start from scratch
     //from scratch takes nmodels * (nmodels-1) additions but relativelu cache friendly
     //each update takes 2*(nmodels-1) lookups and  updates so update if less than nmodels/2 
     //in reality closer n/10 - can change to dynamically optimise
     memmove(cluster_sizes,new_cluster_sizes,nclusters*sizeof(int));
     if(!recalc_density_ratio)recalc_density_ratio=(nmodels>2000)? 1000.0f/nmodels : 0.5;
     if(nchanges < nmodels*recalc_density_ratio){
      for(int k=0;k<nchanges;k++){
       int i=changes[k];
       adjust_density_for_cluster_change(i,cluster_ids[i],new_ids[i],models);
      }
     }
     else{
      memmove(cluster_ids,new_ids,nmodels*sizeof(int));
      find_cluster_density(models);
     }
    }
   }
   if(changes) delete [] changes;
   if(new_cluster_sizes) delete [] new_cluster_sizes;
   if(new_ids) delete [] new_ids;
   return(retvalue);
   }
    #else
   return (assign_to_lowest_average_distance(models));
   #endif
  } 

    
  int assign_to_lowest_average_distance(mapped_cluster_models_set <T> *models){ //requires that an existing partition in ids and density cluster has been done
   int retvalue=0;
   for (int i=0;i<nclusters;i++){
    if(cluster_sizes[i]<2)return(-1);
   }
   int *changes= new int [nmodels];
   int *new_ids= new int [nmodels];
   int *new_cluster_sizes= new int [nclusters];
   memset(changes,0,nmodels*sizeof(int));
   memmove(new_cluster_sizes,cluster_sizes,nclusters*sizeof(int));
   memmove(new_ids,cluster_ids,nmodels*sizeof(int));

   int nchanges=0;
   //cluster_partition new_partition=*this;
   
   float new_score=0;
   for(int i=0;i<nmodels;i++){
    int c=cluster_ids[i];
    int newc=c;
    float best_value=cdensity(i);
    int j=0;
    for( ;j< c ;j++){ 
     if( better(cdensity(i,j),best_value)){
      best_value=cdensity(i,j);
      newc=j;
     }
    }
    for(j=c+1 ;j< nclusters ;j++){ 
     if( better(cdensity(i,j),best_value)){
      best_value=cdensity(i,j);
      newc=j;
     }
    }    
    new_score+=best_value;
    //keep track of changes
    if(newc != c){
     changes[nchanges++]=i;
     new_cluster_sizes[c]--;
     new_cluster_sizes[newc]++;
     new_ids[i]=newc;
    }
   }
   //check cluster_sizes
   for (int j=0;j<nclusters;j++){
    if(new_cluster_sizes[j] < min_cluster_size){
     if(changes)delete [] changes;
     if(new_cluster_sizes) delete [] new_cluster_sizes;
     if(new_ids) delete [] new_ids;
     return(-1);
    }
   }

   if(nchanges){
   //check new score
    new_score/=(float)nmodels;
    if(better(new_score,score)){
     retvalue=1;
     score=new_score;
     //update cluster here
     //decide whether to update existing densities or start from scratch
     //from scratch takes nmodels * (nmodels-1) additions but relativelu cache friendly
     //each update takes 2*(nmodels-1) lookups and  updates so update if less than nmodels/2 
     //in reality closer n/10 - can change to dynamically optimise
     memmove(cluster_sizes,new_cluster_sizes,nclusters*sizeof(int));
     if(!recalc_density_ratio)recalc_density_ratio=(nmodels>2000)? 1000.0f/nmodels : 0.5;
     if(nchanges < nmodels*recalc_density_ratio){
      for(int k=0;k<nchanges;k++){
       int i=changes[k];
       adjust_density_for_cluster_change(i,cluster_ids[i],new_ids[i],models);
      }
     }
     else{
      memmove(cluster_ids,new_ids,nmodels*sizeof(int));
      find_cluster_density(models);
     }
    }
   }
   if(changes) delete [] changes;
   if(new_cluster_sizes) delete [] new_cluster_sizes;
   if(new_ids) delete [] new_ids;
   return(retvalue);
  } 
  int assign_to_closest_center(mapped_cluster_models_set <T> *models){//assign each model to closest partition->cluster_center - used to bootstrap average distance partition->clustering too
   for(int i=0;i<nmodels;i++)
    cluster_ids[i]=-1;
   for(int i =0;i<nclusters;i++){
    cluster_sizes[i]=1;
    cluster_ids[cluster_centers[i]]=i;
   }
   for(int i=0;i<nmodels;i++){
    if(cluster_ids[i]<0){
     float min_dist=models->get_matrix(i,cluster_centers[0]);
     cluster_ids[i]=0;
     for(int j=1;j<nclusters;j++){
      int c=cluster_centers[j];
      if(better(models->get_matrix(i,c),min_dist)){
       min_dist=models->get_matrix(i,c);
       cluster_ids[i]=j;
      }
     }
     cluster_sizes[cluster_ids[i]]++;
    }
   } 
   for(int i =0;i<nclusters;i++)
    if(cluster_sizes[i]<min_cluster_size) return(0); //check if it meets the size criteria
   return(1);
  }
  int assign_to_closest_center(mapped_cluster_models_set <T> *models,int nthreads){
  //assign each model to closest partition->cluster_center - used to bootstrap average distance partition->clustering too
   #ifdef OPENMP
   int tcluster_sizes[nthreads*nclusters];
   memset(tcluster_sizes,0,sizeof(int)*nthreads*nclusters);
   for(int i=0;i<nmodels;i++)
    cluster_ids[i]=-1;
   for(int i =0;i<nclusters;i++){
    tcluster_sizes[i]=1;
    cluster_ids[cluster_centers[i]]=i;
   }
   #pragma omp parallel for num_threads(nthreads)  schedule(dynamic)
   for(int i=0;i<nmodels;i++){
    if(cluster_ids[i]<0){
     float min_dist=models->get_matrix(i,cluster_centers[0]);
     int ci=0;
     for(int j=1;j<nclusters;j++){
      int c=cluster_centers[j];
      if(better(models->get_matrix(i,c),min_dist)){
       min_dist=models->get_matrix(i,c);
       ci=j;
      }
     }
     cluster_ids[i]=ci;
     tcluster_sizes[ci]++;
    }
   } 
   for(int i=1;i<nthreads;i++){
    const int* th_cluster_sizes=&(tcluster_sizes[i*nclusters]);
    for(int c=0;c<nclusters;c++){
     tcluster_sizes[c]+=th_cluster_sizes[c];
    }
   }
   for(int i =0;i<nclusters;i++)
    if(tcluster_sizes[i]<min_cluster_size) return(0); //check if it meets the size criteria
    memmove(cluster_sizes,tcluster_sizes,nclusters*sizeof(int));
   return(1);
   #else
   assign_to_closest_center(models);
   #endif
  } 
  int idensity_filter(int zmode,float prune_to,float prune_stop,int prune_min_size,int prune_max_size,float ratio,bool initialize,int max_initial_iterations, int max_convergence_iterations,
                     mapped_cluster_models_set <T> *models, int (cluster_partition<T>::*f_assign)(mapped_cluster_models_set<T> *),FILE *output_fp){
   
   if(kcluster(max_initial_iterations,max_convergence_iterations,models,f_assign,nclusters,initialize) ==-1){
    fprintf(stderr,"unable to find initial cluster of %d clusters of minimum %d members after %d iterations\n",nclusters,min_cluster_size,max_initial_iterations);
    return(0); 
   }

   pruned_model **log=0;
   int round=0;
   float mean,stdev,bad_cutoff_value;
   
   //prune_min_size && prune_stop define lower limits and will stop the pruning when reached
   //will prune until prune_max_size unless prune_stop is reached 
   //will continue pruning to until the worst value is not worse than prune_to unells prune_min_size is met
   //first prune to
   if(output_fp) log=new pruned_model* [nmodels];
   
   if(prune_to){
    while(nmodels>prune_min_size) {  
     int nworst=ratio*nmodels;
     if(nmodels-prune_min_size < nworst)nworst=nmodels-prune_min_size;
     if(nworst <1)nworst =1;  
     if(zmode){ //check if it is OK to stop
      cdensity_mean_stdev(&mean,&stdev);
      bad_cutoff_value=(greater_is_better)? mean-prune_to*stdev : mean+prune_to*stdev;
     } 
     else bad_cutoff_value=prune_to*(float)(nmodels-1);
     int nremoved=remove_nworst_cdensity_worse_than_value(models,f_assign,bad_cutoff_value,nworst,log);
    
     if(!nremoved)break;
     if(log){
      for(int i=0;i<nremoved;i++){
       fprintf(output_fp,"%s pruned with score %8.5f in round %d\n",log[i]->name,log[i]->score,round);
       if(log[i])delete log[i];
      }
     }
     round++;
    }
   }//end prune_to block
   if(prune_max_size){
    while(nmodels>prune_max_size){
     int nworst=ratio*nmodels;
     if(nmodels-prune_max_size < nworst)nworst=nmodels-prune_max_size;
     if(nworst <1) nworst =1;  
     if(prune_stop){
      if(zmode){
       cdensity_mean_stdev(&mean,&stdev);
       bad_cutoff_value=(greater_is_better)? mean-prune_stop*stdev : mean+prune_stop*stdev;
      } 
      else 
       bad_cutoff_value=prune_stop*(float)(nmodels-1);
      int nremoved=remove_nworst_cdensity_worse_than_value(models,f_assign,bad_cutoff_value,nworst,log);
      if(!nremoved)break;
      if(log){
       for(int i=0;i<nremoved;i++){
        fprintf(output_fp,"%s pruned with score %8.5f in round %d\n",log[i]->name,log[i]->score,round);
        if(log[i])delete log[i];
       }
      }
     }
     else{

      int nremoved=remove_nworst_cdensity(models,f_assign,nworst,log);
      if(log){
       for(int i=0;i<nremoved;i++){
        fprintf(output_fp,"%s pruned with score %8.5f in round %d\n",log[i]->name,log[i]->score,round);
        if(log[i])delete log[i];
       }
      }      
     }
     round++;
    }
   }
   cdensity_mean_stdev(&mean,&stdev);
   fprintf(stderr,"pruned to %d models with mean density %8.4f +/- %-8.4f\n",nmodels,mean,stdev);
   if(output_fp){
    if(log) delete [] log;
   }
  }

 int check_cluster_density(mapped_cluster_models_set <T> *models){ 
  float *copy= new float[nclusters*nmodels];
  memset(copy,0,nmodels*nclusters*sizeof(float));
  for(int i=1;i<nmodels;i++){
   int ci= cluster_ids[i];
   float* const idensity=&(copy[ci*nmodels]);
   for(int j=0;j<i;j++){ 
    int cj= cluster_ids[j];
    float value=models->get_matrix(i,j);
    copy[cj*nmodels+i]+=value;
    idensity[j]+=value;
   }
  }
  int flag=1;
  for(int i=0;i<nmodels;i++){
   for(int c=0;c<nclusters;c++){
    int a=c*nmodels+i;
    float scale=1.0f/(float)(cluster_sizes[c]-1);
    if(fabs(copy[a]*scale-cdensity(i,c)) > 0.001f){
     fprintf(stderr,"%d %d %8.5f %8.5f %8.5f %8.5f\n",i,c,copy[a]*scale,cdensity(i,c),copy[a],unnormed_cdensity(i,c));
     flag=0;
    }  
   }
  }
  return(flag);
 }
  bool better (float a, float b){   
   return ((greater_is_better && a>b) || (!greater_is_better && a<b));
  }
  float cdensity(int i){
   int c=cluster_ids[i];
   float inv_fcluster_size = (cluster_sizes[c] <=1 )? 0 :  1.0f/(float)(cluster_sizes[c]-1);
   return(cluster_density[c*nmodels+i]*inv_fcluster_size);
  }  
  float cdensity(int i,int c){
   float scale =(c==cluster_ids[i])? 1.0f/(float)(cluster_sizes[c]-1): 1.0f/(float)(cluster_sizes[c]);
   return(cluster_density[c*nmodels+i]*scale);
  }
  float unnormed_cdensity(int i){
   int c=cluster_ids[i];
   return(cluster_density[c*nmodels+i]);
  }
  float unnormed_cdensity(int i,int c){
   return(cluster_density[c*nmodels+i]);
  }
  int sort_clusters_insort(){
   float *density_of_cluster=density_of_clusters();
   int temp_size,temp_index;
   int *cluster_map=new int[nclusters];
   int *reverse_cluster_map=new int[nclusters]; //needed for in place substitutions
   for (int i=0;i<nclusters;i++)
    cluster_map[i]=i;
   for (int i=0;i<nclusters;i++){
    int j=i;
    temp_size=cluster_sizes[j];
    temp_index=cluster_map[j];
    while (j>0 && (cluster_sizes[j-1] < temp_size || (cluster_sizes[j-1]== temp_size && better(density_of_cluster[j],density_of_cluster[j-1]) > 0)))  {
     cluster_sizes[j]=cluster_sizes[j-1];
     cluster_map[j]=cluster_map[j-1];
     j--;
    }
    cluster_sizes[j]=temp_size;
    cluster_map[j]=temp_index;
   }
   for (int i=0;i<nclusters;i++)
    reverse_cluster_map[cluster_map[i]]=i;
   for(int i=0;i<nmodels;i++)
    cluster_ids[i]=reverse_cluster_map[cluster_ids[i]];
   if(cluster_centers){
    int *temp_cluster_centers=new int[nclusters];
    for(int i=0;i<nclusters;i++)
     temp_cluster_centers[i]=cluster_centers[cluster_map[i]];
    for(int i=0;i<nclusters;i++)
     cluster_centers[i]=temp_cluster_centers[i];
    if(temp_cluster_centers)delete []temp_cluster_centers;
   }
   float *temp_density= new float[nmodels*nclusters];
   memmove(temp_density,cluster_density,nmodels*nclusters*sizeof(float));
   for(int i=0;i<nclusters;i++){
    int r=cluster_map[i];
    if(r!=i)memmove(&(cluster_density[i*nmodels]),&(temp_density[r*nmodels]),nmodels*sizeof(float));
   }
   if(temp_density)delete [] temp_density;
   if(density_of_cluster)delete [] density_of_cluster;
   if(cluster_map) delete [] cluster_map;
   if(reverse_cluster_map) delete [] reverse_cluster_map;
   return(1);
  }
  float* density_of_clusters(){
   float *scores=new float[nclusters];
   memset(scores,0,nclusters*sizeof(float));
   for(int i=0;i<nmodels;i++){
    int c=cluster_ids[i];
    scores[cluster_ids[i]]+=cluster_density[c*nmodels+i];
   }
   for (int i=0;i<nclusters;i++){
    if(cluster_sizes[i]-1) scores[i]/=((float)cluster_sizes[i]*(float)(cluster_sizes[i]-1));
    else scores[i]=0.0f;
   }
   return(scores);
  } 
  void randomize_centers(unsigned int *seed){
   kshuffle(nclusters,randi,nmodels,seed);
   for(int i=0;i<nclusters;i++)
    cluster_centers[i]=randi[i];
  }  
  
  void random_distant_centers(unsigned int *seed,int nrandom, mapped_cluster_models_set<T> *models){
   if(nrandom){
    kshuffle(nrandom,randi,nmodels,seed);
    for(int i=0;i<nrandom;i++)
     cluster_centers[i]=randi[i];
   }
   find_most_distant_models(nrandom,nclusters,cluster_centers,models);
  }
  
  void find_most_distant_models(int nstart, int nfinal, int *starting_models,mapped_cluster_models_set<T> *models){
   bool *seen = new bool [nmodels];
   memset(seen,0, nmodels);
   int m=0;
   int my_nstart=nstart;
   if(nstart ==0){
    float *density= new float[nmodels];
    memset(density,0,nmodels*sizeof(float));
    for (int i=1;i<nmodels;i++){
     for (int j=0;j<i;j++){
      float value=models->get_matrix(i,j);
      density[i]+=value;
      density[j]+=value;
     }
    }
    float max_density=density[0];
    int max_i=0;
    for(int i=1;i<nmodels;i++){
     if(better(max_density,density[i])){
      max_density=density[i];
      max_i=i;
     } 
    }
    starting_models[0]=max_i; 
    if(density)delete [] density;
    my_nstart=1;
   }
   
   for( ;m<my_nstart;m++)
    seen[starting_models[m]]=true;
   for ( ;m<nfinal;m++){
    float max_dist=better(1,0)? FLT_MAX : -FLT_MAX;
    int farj=0;
    for(int j=0;j<nmodels;j++){
     if(!seen[j]){
      float dist=0;
      for(int k=0;k<m;k++){
       float value=models->get_matrix(starting_models[k],j);
       dist+=value*value;
      }
      if(better(max_dist,dist)){
       max_dist=dist;
       farj=j;
      }  
     } 
    }
    starting_models[m]=farj;
    seen[farj]=true; 
   }
   if(seen) delete [] seen;
  }
  void print(FILE *fp){
   for(int i=0;i<nmodels;i++){
    if(i==cluster_centers[cluster_ids[i]])
     fprintf(fp,"%d*:%-d ",i,cluster_ids[i]);
    else
     fprintf(fp,"%d :%-d ",i,cluster_ids[i]); 
    int c=cluster_ids[i];
    for(int j=0;j<c;j++){
     fprintf(fp,"%d :%-8.3f  ",j,cluster_density[nmodels*j+i]/(float)cluster_sizes[j]);
    }
    fprintf(fp,"%d*:%-8.3f ",c,cluster_density[nmodels*c+i]/(float)(cluster_sizes[c]-1));
    for(int j=c+1;j<nclusters;j++){
     fprintf(fp,"%d :%-8.3f  ",j,cluster_density[nmodels*j+i]/(float)cluster_sizes[j]);
    }
    fprintf(fp,"\n");
   }
  }

  int find_cluster_density(mapped_cluster_models_set <T> *models){
   for (int i=0;i<nmodels*nclusters;i++)
    cluster_density[i]=0;
   float sums[nclusters];
   for(int i=1;i<nmodels;i++){
    int ci= cluster_ids[i];
    for(int j=0;j<nclusters;j++)
     sums[j]=0;
    float* const idensity=&(cluster_density[ci*nmodels]);
    int j=0;
    for(;j<i;j++){
     int cj=cluster_ids[j];
     float value=models->get_matrix(i,j);
     idensity[j]+=value;
     sums[cj]+=value;
    } 
    int index=i;
    for(int c=0;c<nclusters;c++){
     cluster_density[index]+=sums[c];
     index+=nmodels;    
    } 
   }
  }
  int find_cluster_density(mapped_cluster_models_set <T> *models,int nthreads){
  #ifdef OPENMP   
   if(nthreads<2)return(find_cluster_density(models));
    
   else{
    for (int i=0;i<nmodels*nclusters;i++)
     cluster_density[i]=0;
    float tsums[nclusters*nthreads]; 
    for(int i=0;i<nmodels;i++){
     float sums[nclusters];
     int ci= cluster_ids[i];
 
     for(int j=0;j<nclusters;j++)
      tsums[j]=0;
     float* const idensity=&(cluster_density[ci*nmodels]);
     #pragma omp parallel for num_threads(nthreads)  schedule(dynamic) 
     for(int j=0 ;j<i;j++){
      int cj=cluster_ids[j];
      float value=models->get_matrix(i,j);
      idensity[j]+=value;//threadsafe - depends on j
      #pragma omp critical
      cluster_density[nmodels*cj+i]+=value;//not thread safe depends on cj
     }
    }
   }
#else
  return(find_cluster_density(models));
#endif    
  }

 
  float cluster_density_homogeneity(){
   float score=0;
   for(int i=0;i<nmodels;i++){
    score+=cdensity(i);
   }
   return(score/(float)nmodels);
  }
  int find_center_structures_using_density_of_cluster(){//uses density of cluster
   int nchanges=0;
   float best_density[nclusters];
   for  (int i=0;i<nclusters;i++)
    best_density[i]=(greater_is_better)? -FLT_MAX : FLT_MAX;
   for (int i=0;i<nmodels;i++){
    float d=cdensity(i);
    int c=cluster_ids[i];
    if (better(d,best_density[c])){
     best_density[c]=d;
     cluster_centers[c]=i;
     nchanges++;
    } 
   }
   return(nchanges);
  }
  
  int find_center_structures_using_density_of_cluster(int nthreads){//uses density of cluster
   #ifdef OPENMP
   if(nthreads <2)
    return(find_center_structures_using_density_of_cluster());
   else{ 
    int nchanges=0;
    float tdensity[nthreads*nclusters];
    int tcluster_centers[nthreads*nclusters];
    float worst_value=(greater_is_better)? -FLT_MAX : FLT_MAX;
    for  (int i=0;i<nclusters*nthreads;i++){
     tdensity[i]=worst_value;
    } 
    #pragma omp parallel for num_threads(nthreads)  schedule(dynamic)
    for (int i=0;i<nmodels;i++){
     int th=omp_get_thread_num();
     int offset=th*nclusters;
     float d=cdensity(i);
     int c=cluster_ids[i];
     if (better(d,tdensity[offset+c])){
      tdensity[offset+c]=d;
      tcluster_centers[offset+c]=i;
     } 
    }
    
    memmove(cluster_centers,tcluster_centers,nclusters*sizeof(int));
    for (int t=1;t<nthreads;t++){
     const float* th_density=&(tdensity[t*nclusters]);
     for(int c=0;c<nclusters;c++){
      float d=th_density[c];
      if(better(d,tdensity[c])){
       tdensity[c]=d;
       cluster_centers[c]=tcluster_centers[t*nclusters+c];
      } 
     }
    }
   }
   #else
    return(find_center_structures_using_density_of_cluster());
   #endif
  } 
  
   
  int initialize_partition_by_distance(int max_iterations,unsigned int *seed,mapped_cluster_models_set <T> *models, int nfixed){//intialize_valid_partition   
   int niter=0;
   score=(greater_is_better)? -FLT_MAX :FLT_MAX;
   int n=nclusters-nfixed;
   while(n <= nclusters){
    while(niter<max_iterations){
     random_distant_centers(seed,n,models);
     if(assign_to_closest_center(models) != -1){
      find_cluster_density(models);
      score=cluster_density_homogeneity();
      return(1);
     }
     niter++;
    }
    n++;
   }
   return(0);
  }   
  int initialize_partition_by_distance(int max_iterations,unsigned int *seed,mapped_cluster_models_set <T> *models, int nfixed,int nthreads){//intialize_valid_partition   
   int niter=0;
   score=(greater_is_better)? -FLT_MAX :FLT_MAX;
   int n=nclusters-nfixed;
   while(n <= nclusters){
    while(niter<max_iterations){
     random_distant_centers(seed,n,models);
     if(assign_to_closest_center(models,nthreads) != -1){
      find_cluster_density(models,nthreads);
      score=cluster_density_homogeneity();
      return(1);
     }
     niter++;
    }
    n++;
   }
   return(0);
  } 
  void adjust_density_for_cluster_change(int a,int source_cluster, int dest_cluster,mapped_cluster_models_set<T> *models){
   float* const source=&(cluster_density[source_cluster*nmodels]);
   float* const dest=&(cluster_density[dest_cluster*nmodels]);
   cluster_ids[a]=dest_cluster;
   for (int i=0;i<a;i++){
    float value=models->get_matrix(i,a);
    source[i]-=value;
    dest[i]+= value;
   }
   for (int i=a+1;i<nmodels;i++){
    float value=models->get_matrix(a,i);
    source[i]-=value;
    dest[i]+= value;
   }
  }  



   void cdensity_mean_stdev(float *mean, float *stdev){
   float sum=0,ssq=0;
   for (int i=0;i<nmodels;i++){
    float value=cdensity(i);
    sum+=value;
    ssq+=value*value;
   }
   float my_mean=sum/(float)nmodels;
   float my_stdev=sqrt(ssq/(float)nmodels-my_mean*my_mean);
   *mean=my_mean;
   *stdev=my_stdev;
  }  
 };
template <class T>
class mapped_cluster_set{ //combination of partition and models - complete description of clustered set
 //requires that you construct base sets first
 public:
  bool greater_is_better;
  mapped_cluster_models_set<T> *models;  //cluster_variables
  cluster_partition<T> *best_partition;
  mapped_cluster_set(cluster_models_set<T> *input_base_set,int input_nclusters,int input_min_cluster_size,unsigned int input_seed){
   models = new mapped_cluster_models_set<T>(input_base_set);
   greater_is_better=models->greater_is_better;
   float start_score=(greater_is_better)? -FLT_MAX : FLT_MAX;
   best_partition=new cluster_partition<T>(models->nmodels,input_nclusters,input_min_cluster_size,greater_is_better,start_score,input_seed);
  }  
  mapped_cluster_set(cluster_models_set<T> *input_base_set,int input_nclusters,int input_min_cluster_size,unsigned int input_seed,int density_size){
   //used for hierarchical clustering - density not needed until the very end
   models = new mapped_cluster_models_set<T>(input_base_set);
   greater_is_better=models->greater_is_better;
   float start_score=(greater_is_better)? -FLT_MAX : FLT_MAX;
   best_partition=new cluster_partition<T>(models->nmodels,input_nclusters,input_min_cluster_size,greater_is_better,start_score,input_seed,density_size);
  }
  mapped_cluster_set (const mapped_cluster_set<T> &A) : greater_is_better(A.greater_is_better) {
   models=new mapped_cluster_models_set<T>(*(A.models));
   best_partition=new cluster_partition<T>(*(A.best_partition));
  }
  mapped_cluster_set &operator = (const mapped_cluster_set<T> &rhs){
   if(this !=&rhs){
    *models=*(rhs.models);
    *best_partition=*(rhs.best_partition);
    greater_is_better=rhs.greater_is_better;
   }
   return *this; 
  }
 ~mapped_cluster_set(){
   if (models) delete models;
   if(best_partition) delete best_partition;
  }
  void filter_by_average_centroid_distance(int npruned){
   //calculate density
   float *centroid_density=new float[models->nmodels];
   float invfnclusters=1.0f/(float)best_partition->nclusters;
   memset(centroid_density,0,models->nmodels*sizeof(float));
   for (int i=0;i<models->nmodels;i++){
    for(int c=0;c<best_partition->nclusters;c++){
     int j=best_partition->cluster_centers[c];
     if(j!=i){
      float value=models->get_matrix(i,j);
      centroid_density[i]+=value*invfnclusters;
     }
    }
   }

   //heap_list puts nworst in my_list.heap_map   
   priority_heap_list<float> my_list (centroid_density,models->nmodels-npruned,models->nmodels,greater_is_better,0);
   //make mask list to track the decoys to be eliminated
   int* bad_list=new int[models->nmodels];
   memset(bad_list,0,models->nmodels*sizeof(int));
   for(int i=0;i<models->nmodels-npruned;i++){
    bad_list[my_list.heap_map[i]]=1;
   }
   {
    int m=0;
    for(int i=0;i<models->nmodels;i++){
     if(!bad_list[i]){
      models->map[m++]=models->map[i];
     }
    }
   }
   models->nmodels=npruned;
   if(bad_list) delete [] bad_list;
   if(centroid_density) delete [] centroid_density;
  }
	  
  void print_density (FILE *fp,cluster_partition<T> *p){  
   int nmodels=p->nmodels;int nclusters=p->nclusters;
   int* const ids=p->cluster_ids;int* const centers=p->cluster_centers;int* const sizes=p->cluster_sizes;
   float* const density=p->cluster_density; 
   for(int i=0;i<nmodels;i++){
    fprintf(fp,"%s %d ",models->names(i),ids[i]);
    int c=ids[i];
    for(int j=0;j<c;j++){
     if (sizes[j]) fprintf(fp,"%8.4f ",density[best_partition->nmodels*j+i]/(float)sizes[j]);
     else fprintf(fp,"%8.4f ",0);
    }
    if (sizes[c]-1)
     fprintf(fp,"%8.4f ",density[best_partition->nmodels*c+i]/(float)(sizes[c]-1));
    else
     fprintf(fp,"%8.4f ",0);
    for(int j=c+1;j<nclusters;j++){
       if (sizes[j]) fprintf(fp,"%8.4f ",density[best_partition->nmodels*j+i]/(float)sizes[j]);
     else fprintf(fp,"%8.4f ",0);    
   }
   fprintf(fp,"\n");
   }
  }

  void print_centers(FILE *fp,cluster_partition<T> *p){
   float *densities=p->density_of_clusters();
   int * const centers=p->cluster_centers;
   for (int i=0;i<p->nclusters;i++){
    fprintf(fp,"Cluster:  %d of size %d with centroid %s with cluster density score %8.4f\n",i,p->cluster_sizes[i],models->names(centers[i]),densities[i]);
   }
   if(densities) delete [] densities;
  }
  void print_cluster_stats(FILE *fp,cluster_partition<T> *p){
   int * const centers=p->cluster_centers;
   int nclusters=p->nclusters;
   for (int i=0;i<nclusters-1;i++)
    for (int j=i+1;j<nclusters;j++)
     fprintf(fp,"cluster center %d to %d distance: %8.4f\n",i,j,models->get_matrix(centers[i],centers[j]));
     
  } 
};
template <class T>
class parallel_cluster_set{ //a set of clustered sets maintained by separate threads for parallel 
 public:
  int nthreads;
  unsigned int seed;
  bool greater_is_better;
  mapped_cluster_set<T> *best_cluster_set; //best global cluster
  mapped_cluster_set<T> **tbest_cluster_sets;   //best clusters in each thread
  parallel_cluster_set(int input_nthreads, cluster_models_set <T> *input_base_set, int input_seed, int input_nclusters,int input_min_cluster_size){ //remember to set up the base classes first first
   greater_is_better=(input_base_set->greater_is_better);
#ifdef OPENMP   
   int max_threads = omp_get_max_threads();     //this is set by the eviroment variable
#else
   int max_threads=1;
#endif
   nthreads=(input_nthreads)? input_nthreads : max_threads;
   nthreads= (nthreads > max_threads)? max_threads : nthreads;   
   if(!input_seed){
    srand(time(0));
    seed=rand();
   }
   else seed=input_seed;
   best_cluster_set = new mapped_cluster_set<T>(input_base_set,input_nclusters,input_min_cluster_size,seed);
   tbest_cluster_sets=new mapped_cluster_set<T>* [nthreads];
   //set up thread variables
   for (int i=0;i<nthreads;i++)
    tbest_cluster_sets[i] = new mapped_cluster_set<T>(input_base_set, input_nclusters,input_min_cluster_size,seed+(i+1)*1000);
   }
  ~parallel_cluster_set(){
   for (int i=0;i<nthreads;i++){
    if(tbest_cluster_sets[i])delete tbest_cluster_sets[i];
   }
   if(tbest_cluster_sets) delete [] tbest_cluster_sets;
   if(best_cluster_set) delete best_cluster_set;
  }
  int parallel_cluster(int (cluster_partition<T>::* f_assign)(mapped_cluster_models_set<T>*), int total_iterations,int max_initial_iterations,int max_convergence_iterations,int nfixed_centers){//total_iterations is stopping criterion
   //this is much simpler than keeping track of number of changes between scores 
   //only need to keep track of the number of iterations and local best scores
   
   //returns cluster_ids and cluster_density
   int retvalue=0;                                   //returns 1 if better score than original solution found - 0 if not and -1 if no solutions matching cluster size and number criterion found
   
   //variables maintained by master thread (thread 0)
   float best_score=(greater_is_better)? -FLT_MAX : FLT_MAX;     //absolute best_score as a function of densities - maintained by master thread
   float *tbest_scores=new float [nthreads];                     //best score in thread - kept in partition too but copy is kept for quick global access
   int *titer_state= new int[nthreads];			                       //number of iterations done by thread - so that they each iterate at own rate - also defines the state that is "frozen" and evaluated by the master thread
   int best_t=0;                                                 //thread with best_score 
   //variables to initialize random numbers for threads
   //initialize the thread variables
  int nt=(nthreads < total_iterations) ? nthreads : total_iterations;
   for(int i=0;i<nt;i++){
    titer_state[i]=0;
    tbest_scores[i]=best_score;
   }
   int done=0;
   int iterations=0;
   int last_output=0;
   //tune timing to find best optimal ratio for large datasets
   if(best_cluster_set->models->nmodels >= 10000 && (total_iterations >8 || max_convergence_iterations >8)){
    float slope,intercept,recalc;
    best_cluster_set->best_partition->test_adjust_density(best_cluster_set->models,&slope,&intercept);
    recalc=best_cluster_set->best_partition->test_find_density(best_cluster_set->models);
    float ratio=((float)best_cluster_set->models->nmodels)/((recalc- intercept)/slope);
    best_cluster_set->best_partition->recalc_density_ratio=1.0f/ratio;
    for(int t=0;t<nt;t++)
     tbest_cluster_sets[t]->best_partition->recalc_density_ratio=1.0f/ratio;
    fprintf(stderr,"tuned sets\n"); 
   }
  ;
#ifdef OPENMP       
   #pragma omp parallel num_threads(nt)
#endif       
   {
    //parallel section
#ifdef OPENMP  
    int t=omp_get_thread_num();
#else
    int t=0;
#endif
    float my_best_score=tbest_scores[t];
    while(titer_state[t]<total_iterations && !done){ //loop exits after total_iterations or when the done flag is set if sufficient number of iterations pass with improvement in scores or the iterations done by all threads exceeds max_iterations
     int rvalue=tbest_cluster_sets[t]->best_partition->kcluster(max_initial_iterations,max_convergence_iterations,tbest_cluster_sets[t]->models,f_assign,nfixed_centers,true);
     if(rvalue != -1){
      if(rvalue && better(tbest_cluster_sets[t]->best_partition->score,my_best_score)){
       my_best_score=tbest_cluster_sets[t]->best_partition->score;
       tbest_scores[t]=my_best_score;
      }
     }
     titer_state[t]++; //update the counter right after the updates  if(ret_value && ret_value != -1) tchanges[t]=0;
    //master thread now does the global update of best score
     if(t==0){
      iterations=0;
      for(int th=0;th<nt;th++){
       iterations+=titer_state[th];
      }
      if(iterations>total_iterations)done=1;
      else if(iterations-last_output >= 50){
       for(int th=0;th<nt;th++){
        best_score=(better(tbest_scores[th],best_score))?tbest_scores[th]:best_score;
       }
       fprintf(stderr,"best cluster score %9.5f after %d iterations\n",best_score,iterations);
       last_output=iterations;    
      }
     }
    }
   } //end parallel section     
   best_score=tbest_scores[0];

   for(int th=1;th<nt;th++){
    if(better(tbest_scores[th], best_score)){
     best_score=tbest_scores[th];
     best_t=th;
    }
   } 
   if(best_score != FLT_MAX &&  best_score !=  -FLT_MAX){//at least one good score has been found
    *best_cluster_set=*(tbest_cluster_sets[best_t]);
    fprintf(stderr,"score converged %9.5f after %d iterations\n",best_cluster_set->best_partition->score,iterations);
    best_cluster_set->best_partition->sort_clusters_insort();
    best_cluster_set->best_partition->find_center_structures_using_density_of_cluster();   
    retvalue=1;
   }
   else{
    retvalue=0;
   }
   //deallocate
   if(tbest_scores)delete [] tbest_scores;
   if(titer_state) delete [] titer_state;
   return(retvalue); 
  }
  
  int parallel_cluster(int (cluster_partition<T>::* f_assign)(mapped_cluster_models_set<T>*), int max_iterations,int max_initial_iterations,int max_convergence_iterations, int nsolutions_after_best_score,int nfixed_centers){
   //returns cluster_ids and cluster_density
   int retvalue=0;                                   //returns 1 if better score than original solution found - 0 if not and -1 if no solutions matching cluster size and number criterion found
   
   //variables maintained by master thread (thread 0)
   float best_score=(greater_is_better)? -FLT_MAX : FLT_MAX;     //absolute best_score as a function of densities - maintained by master thread
   float *tbest_scores=new float [nthreads];                    //best score in thread - kept in partition too but copy is kept for quick global access
   int best_t=0;                                                //thread that has the best score
   bool change_flag=false;                                      //has there been a change in best score 
   float *temp_scores=new float [nthreads];                     //used to make local copies during the global update
   int *titer_state= new int[nthreads];			                      //number of iterations done by thread - so that they each iterate at own rate - also defines the state that is "frozen" and evaluated by the master thread
   int iterations=0,old_iterations=0,iterations_between_best_scores=0; //these variables control termination

   int nt=(nthreads < max_iterations)? nthreads:max_iterations;//case when there are very few iterations 
   //variables to initialize random numbers for threads
   //initialize the thread variables
   for(int i=0;i<nt;i++){
    titer_state[i]=0;
    tbest_scores[i]=best_score;
   }
   int done=0;
   //tune timing to find best optimal ratio for large datasets
   if(best_cluster_set->models->nmodels > 10000 && max_iterations >8){
    float slope,intercept,recalc;
    best_cluster_set->best_partition->test_adjust_density(best_cluster_set->models,&slope,&intercept);
    recalc=best_cluster_set->best_partition->test_find_density(best_cluster_set->models);
    float ratio=((float)best_cluster_set->models->nmodels)/((recalc- intercept)/slope);
    best_cluster_set->best_partition->recalc_density_ratio=1.0f/ratio;
    for(int t=0;t<nthreads;t++)
     tbest_cluster_sets[t]->best_partition->recalc_density_ratio=1.0f/ratio;
   }
   
#ifdef OPENMP   
   #pragma omp parallel num_threads(nt)
#endif       
   {
    //parallel section
#ifdef OPENMP    
    int t=omp_get_thread_num();
#else
    int t=0;
#endif
    float my_best_score=best_score;		     //my_best_score holds the threads copy of the current best_score - it is copied to tbest_scores[t] for the global comparison by the master thread
    tbest_scores[t]=my_best_score;
    while(titer_state[t]<max_iterations && !done){ //loop exits after max_iterations or when the done flag is set if sufficient number of iterations pass with improvement in scores or the iterations done by all threads exceeds max_iterations
     int rvalue=tbest_cluster_sets[t]->best_partition->kcluster(max_initial_iterations,max_convergence_iterations,tbest_cluster_sets[t]->models,f_assign,nfixed_centers,true);
     if(rvalue != -1){
      if(rvalue && better(tbest_cluster_sets[t]->best_partition->score,my_best_score)){
       my_best_score=tbest_cluster_sets[t]->best_partition->score;
       tbest_scores[t]=my_best_score;
      }
     }
     titer_state[t]++; //update the counter right after the updates  if(ret_value && ret_value != -1) tchanges[t]=0;
    //master thread now does the global update of best score
     if(t==0){
      float temp_best_score=best_score;
      iterations=0;
      for(int th=0;th<nt;th++){;  
       temp_scores[th]= tbest_scores[th];
       iterations+=titer_state[th];
      }
      change_flag=false;   
      for(int th=0;th<nt;th++){
       if(better(temp_scores[th],temp_best_score)){
        change_flag=true;
        temp_best_score=temp_scores[th];
        best_t=th;
       }
      }
       //if best score has not changed then check whether the number of iterations since the last best score is enough to stop the loop
      if (change_flag){
       best_score=temp_best_score;    
       fprintf(stderr,"new best cluster score %9.5f after %d iterations\n",best_score,iterations);
       old_iterations=iterations;
      } 
      if(!change_flag){
       iterations_between_best_scores=iterations-old_iterations;
       if(iterations_between_best_scores > nsolutions_after_best_score) done=1;1;
      }
      if(iterations > max_iterations) done=1;
     }
    }
   } //end parallel section 
   //redo reduction in case there has been a change in the final updates
   best_score=tbest_scores[0];
   iterations=titer_state[0];
   for(int th=1;th<nt;th++){
    if(better(tbest_scores[th], best_score)){
     best_score=tbest_scores[th];
     best_t=th;
    }
    iterations+=titer_state[th];
   }
   float score=best_score;
   if(best_score != FLT_MAX &&  best_score !=  -FLT_MAX){ //this checks that at least one good answer has been obtained 
    *best_cluster_set=*(tbest_cluster_sets[best_t]);
    if(iterations_between_best_scores > nsolutions_after_best_score)
     fprintf(stderr,"score converged %9.5f after %d iterations\n",best_cluster_set->best_partition->score,iterations);
    else
     fprintf(stderr,"final score %9.5f after %d iterations\n",best_cluster_set->best_partition->score,iterations);
    best_cluster_set->best_partition->sort_clusters_insort();
    best_cluster_set->best_partition->find_center_structures_using_density_of_cluster();
    retvalue=1;
   }
   else{
    //did not beat original best_score of starting structure
    retvalue=0;
   }
   //deallocate
   if(tbest_scores)delete [] tbest_scores;
   if(titer_state)delete [] titer_state;
   if(temp_scores)delete [] temp_scores;
   return(retvalue); 
  }

  void print_cluster_summary(FILE *fp){
   best_cluster_set->best_partition->find_center_structures_using_density_of_cluster(); 
   best_cluster_set->print_centers(fp,best_cluster_set->best_partition);
   best_cluster_set->print_cluster_stats(fp,best_cluster_set->best_partition); 
  }
  void print_clusters(FILE *fp){
   best_cluster_set->print_density(fp,best_cluster_set->best_partition);
  }
  
 
  
  private:
  bool better (float a, float b){   
   return ((greater_is_better && a>b) || (!greater_is_better && a<b));
  }    
};
//misc routines
void mean_stdev(int narray,float *array, float *mean, float *stdev)
{
 float sum=0,ssq=0;
 for(int i=0;i<narray;i++)
 {
  sum+=array[i];
  ssq+=array[i]*array[i];
 } 
 float my_mean=sum/(float)narray;
 float my_var=ssq/(float)narray-my_mean*my_mean;
 float my_stdev=(my_var > 0)? sqrt(my_var) : 0;
 *mean=my_mean;
 *stdev=my_stdev;
}
void kshuffle(int k, int *arr, int narr, unsigned int *seed){
 for (int i=0;i<k;i++){
  int swap=rand_r(seed)%(narr-i);
  int temp=arr[swap+i];
  arr[swap]=arr[i];
  arr[i]=temp;
 }
}
//sort routines
int sort_by_scores (int nstructs,float *scores, int *sorted,int min_first_flag)
{
 KEY_T *vk;
 if(!(vk=(KEY_T*)malloc((nstructs+1)*sizeof(KEY_T))))exit(FALSE);
 if(min_first_flag)
 {
  for (int i=0;i<nstructs;i++)
  {
   vk[i].value=scores[i];
   vk[i].key=i;
  }
  vk[nstructs].key=nstructs;
  vk[nstructs].value=DBL_MAX;
  sedgesort(vk,nstructs);
 }
 else 
 {
  for (int i=0;i<nstructs;i++)
  {
   vk[i].value=-scores[i];
   vk[i].key=i;
  }
  vk[nstructs].key=nstructs;
  vk[nstructs].value=-DBL_MAX;
  sedgesort(vk,nstructs);
 }
 for(int i=0;i<nstructs;i++)
  sorted[i]=vk[i].key;
 if(vk)free(vk);
 return(TRUE); 
}
void  sedgesort (KEY_T  *array, int len)
{
   partial_quickersort (array, 0, len - 1);
   insort (array, len);
}

void  insort (KEY_T  *array, int len)
{
	register int	i, j;
	register KEY_T	temp;

	for (i = 1; i < len; i++) {
		/* invariant:  array[0..i-1] is sorted */
		j = i;
		/* customization bug: SWAP is not used here */
		temp = array[j];
		while (j > 0 && GT(array[j-1], temp)) {
			array[j] = array[j-1];
			j--;
		}
		array[j] = temp;
	}
}
void  partial_quickersort (KEY_T *array, int lower, int upper)
{
    register int	i, j;
    register KEY_T	temp, pivot;

    if (upper - lower > 15) {
	SWAP(array[lower], array[(upper+lower)/2]);
	i = lower;  j = upper + 1;  pivot = array[lower];
	while (1) {
	    /*
	     * ------------------------- NOTE --------------------------
	     * ignoring BIG NOTE above may lead to an infinite loop here
	     * ---------------------------------------------------------
	     */
	    do i++; while (LT(array[i], pivot));
	    do j--; while (GT(array[j], pivot));
	    if (j < i) break;
	    SWAP(array[i], array[j]);
	}
	SWAP(array[lower], array[j]);
	partial_quickersort (array, lower, j - 1);
	partial_quickersort (array, i, upper);
    }
}
double get_time(){
 timespec ts;
 clock_gettime(CLOCK_REALTIME, &ts);
 return ((double) ts.tv_sec+ (double) ts.tv_nsec/1000000000.0);
} 

//parallel density routines
float rmsd_cpu_par(int nthreads,int nat,int nmodels,float *coords, float *density){
 const double inv3=1.0/3.0;
 const float invfnat=1.0f/(float)nat;
 float ssqs[nmodels],centroids[nmodels][3];

#ifdef OPENMP
 //store first thread results in density
 //store other thread results in my_density;
 float *my_density[nthreads];
 my_density[0]=density;
 for(int i=1;i<nthreads;i++){
  my_density[i]=new float[nmodels];
 } 
 for(int i=0;i<nthreads;i++)
  for(int j=0;j<nmodels;j++)
   my_density[i][j]=0.0f;

 #pragma omp parallel for  num_threads(nthreads) schedule(dynamic)
#endif  
 for (int a=0 ;a<nmodels;a++){
  float* const coords1=&(coords[3*nat*a]);  
  float s1x=0,s1y=0,s1z=0,ssq1=0; 
  for (int i=0;i<nat;i++){
   int m=i*3; 
   float c1x=coords1[m];
   float c1y=coords1[m+1];
   float c1z=coords1[m+2];
   s1x+=c1x;s1y+=c1y;s1z+=c1z;
   ssq1+=c1x*c1x+c1y*c1y+c1z*c1z;  
  }
  ssqs[a]=ssq1;centroids[a][0]=s1x;centroids[a][1]=s1y; centroids[a][2]=s1z;
 }
  
#ifdef OPENMP 
  #pragma omp parallel for num_threads(nthreads)  schedule(dynamic)
#endif    
  for (int a=1 ;a<nmodels;a++){
   for (int b=0;b<a;b++){
    float* const coords1=&(coords[3*nat*a]);
    float* const coords2=&(coords[3*nat*b]);
    float r[9]={0,0,0,0,0,0,0,0,0};
    float s1x=centroids[a][0],s1y=centroids[a][1],s1z=centroids[a][2]; 
    float s2x=centroids[b][0],s2y=centroids[b][1],s2z=centroids[b][2];

    float sxx=0,sxy=0,sxz=0,syx=0,syy=0,syz=0,szx=0,szy=0,szz=0,ssq=ssqs[a]+ssqs[b];

    for (int i=0;i<nat;i++){
     int m=3*i;
     float c1x=coords1[m];
     float c1y=coords1[m+1];
     float c1z=coords1[m+2];
     float c2x=coords2[m];
     float c2y=coords2[m+1];
     float c2z=coords2[m+2];
     r[0]+=c1x*c2x; r[1]+=c1x*c2y; r[2]+=c1x*c2z; r[3]+=c1y*c2x; r[4]+=c1y*c2y; r[5]+=c1y*c2z; r[6]+=c1z*c2x; r[7]+=c1z*c2y; r[8]+=c1z*c2z;
    }
    r[0]-=s1x*s2x*invfnat;
    r[1]-=s1x*s2y*invfnat;
    r[2]-=s1x*s2z*invfnat;
    r[3]-=s1y*s2x*invfnat;
    r[4]-=s1y*s2y*invfnat;
    r[5]-=s1y*s2z*invfnat;
    r[6]-=s1z*s2x*invfnat;
    r[7]-=s1z*s2y*invfnat;
    r[8]-=s1z*s2z*invfnat;
    double det= r[0] * ( (r[4]*r[8]) - (r[5]*r[7]) )- r[1] * ( (r[3]*r[8]) - (r[5]*r[6]) ) + r[2] * ( (r[3]*r[7]) - (r[4]*r[6]) );
    //for symmetric matrix
    //lower triangular matrix rr
    double detsq=det*det;
    //lower triangular matrix rr 

   
    double rr[6]={r[0]*r[0]+ r[1]*r[1]+ r[2]*r[2],
                 r[3]*r[0]+ r[4]*r[1]+ r[5]*r[2],
                 r[3]*r[3]+ r[4]*r[4]+ r[5]*r[5],
                 r[6]*r[0]+ r[7]*r[1]+ r[8]*r[2],
                 r[6]*r[3]+ r[7]*r[4]+ r[8]*r[5],
                 r[6]*r[6]+ r[7]*r[7]+ r[8]*r[8]};

    double spur=((double)(rr[0]+rr[2]+rr[5]))*inv3;
    double cof=((double)(rr[2]*rr[5] - rr[4]*rr[4] + rr[0]*rr[5]- rr[3]*rr[3] + rr[0]*rr[2] - rr[1]*rr[1])) *inv3;
    double e[3] ={spur,spur,spur};
    double h=( spur > 0 )? spur*spur-cof : -1.0;
    if(h>0)
    {
     double g = (spur*cof - detsq)*0.5 - spur*h;
     double sqrth = sqrt(h);
     double d1 = h*h*h - g*g;
     d1= ( d1<0 ) ? atan2(0,-g)*inv3 : atan2(sqrt(d1),-g)*inv3;
     double cth = sqrth * cos(d1);
     double sth = sqrth*SQRT3*sin(d1);
     e[0]+=  cth+cth;
     e[1]+= -cth+sth;
     e[2]+= -cth-sth;
    }
    e[0]=(e[0] < 0) ? 0 : sqrt(e[0]);
    e[1]=(e[1] < 0) ? 0 : sqrt(e[1]);
    e[2]=(e[2] < 0) ? 0 : sqrt(e[2]);
    
    double d=(det<0)? e[0] + e[1] -e[2] : e[0] + e[1]+e[2];
    float xm=s1x*s1x+s1y*s1y*+s1z*s1z+s2x*s2x+s2y*s2y+s2z*s2z;
    float rms=(ssq-xm*invfnat-d-d)*invfnat;
          rms=(rms>1e-8)?sqrt(rms) : 0.0f;
#ifdef OPENMP
    int th=omp_get_thread_num();
    my_density[th][a]+=rms;
    my_density[th][b]+=rms;
#else
    density[a]+=rms;
    density[b]+=rms;
#endif
   }
  }
 //reduction step
#ifdef OPENMP 
 for(int i=1;i<nthreads;i++)
  for(int j=0;j<nmodels;j++)
   density[i]+=my_density[i][j];
 for(int i=1;i<nthreads;i++)
   if(my_density[i])delete [] my_density[i];
#endif
}
template <class T> float rmsd_cpu_par(int nthreads,int nat,int nmodels,float *coords,triangular_matrix<T> *matrix){
 const double inv3=1.0/3.0;
 const float invfnat=1.0f/(float)nat; 
 float ssqs[nmodels],centroids[nmodels][3];
#ifdef OPENMP 
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
 for (int a=0 ;a<nmodels;a++){
  float* const coords1=&(coords[3*nat*a]);  
  float s1x=0,s1y=0,s1z=0,ssq1=0; 
  for (int i=0;i<nat;i++){
   int m=i*3; 
   float c1x=coords1[m];
   float c1y=coords1[m+1];
   float c1z=coords1[m+2];
   s1x+=c1x;s1y+=c1y;s1z+=c1z;
   ssq1+=c1x*c1x+c1y*c1y+c1z*c1z;  
  }
  ssqs[a]=ssq1;centroids[a][0]=s1x;centroids[a][1]=s1y; centroids[a][2]=s1z;
 }

#ifdef OPENMP 
  #pragma omp parallel for num_threads(nthreads)  schedule(dynamic)
#endif    
  for (int a=1 ;a<nmodels;a++){
   for (int b=0;b<a;b++){
    float* const coords1=&(coords[3*nat*a]);
    float* const coords2=&(coords[3*nat*b]);
    float r[9]={0,0,0,0,0,0,0,0,0};
    float s1x=centroids[a][0],s1y=centroids[a][1],s1z=centroids[a][2]; 
    float s2x=centroids[b][0],s2y=centroids[b][1],s2z=centroids[b][2];

    float sxx=0,sxy=0,sxz=0,syx=0,syy=0,syz=0,szx=0,szy=0,szz=0,ssq=ssqs[a]+ssqs[b];

    for (int i=0;i<nat;i++){
     int m=3*i;
     float c1x=coords1[m];
     float c1y=coords1[m+1];
     float c1z=coords1[m+2];
     float c2x=coords2[m];
     float c2y=coords2[m+1];
     float c2z=coords2[m+2];
     r[0]+=c1x*c2x; r[1]+=c1x*c2y; r[2]+=c1x*c2z; r[3]+=c1y*c2x; r[4]+=c1y*c2y; r[5]+=c1y*c2z; r[6]+=c1z*c2x; r[7]+=c1z*c2y; r[8]+=c1z*c2z;
    }
    r[0]-=s1x*s2x*invfnat;
    r[1]-=s1x*s2y*invfnat;
    r[2]-=s1x*s2z*invfnat;
    r[3]-=s1y*s2x*invfnat;
    r[4]-=s1y*s2y*invfnat;
    r[5]-=s1y*s2z*invfnat;
    r[6]-=s1z*s2x*invfnat;
    r[7]-=s1z*s2y*invfnat;
    r[8]-=s1z*s2z*invfnat;
    double det= r[0] * ( (r[4]*r[8]) - (r[5]*r[7]) )- r[1] * ( (r[3]*r[8]) - (r[5]*r[6]) ) + r[2] * ( (r[3]*r[7]) - (r[4]*r[6]) );
    //for symmetric matrix
    //lower triangular matrix rr
    double detsq=det*det;
    //lower triangular matrix rr 

   
    double rr[6]={r[0]*r[0]+ r[1]*r[1]+ r[2]*r[2],
                 r[3]*r[0]+ r[4]*r[1]+ r[5]*r[2],
                 r[3]*r[3]+ r[4]*r[4]+ r[5]*r[5],
                 r[6]*r[0]+ r[7]*r[1]+ r[8]*r[2],
                 r[6]*r[3]+ r[7]*r[4]+ r[8]*r[5],
                 r[6]*r[6]+ r[7]*r[7]+ r[8]*r[8]};

    double spur=((double)(rr[0]+rr[2]+rr[5]))*inv3;
    double cof=((double)(rr[2]*rr[5] - rr[4]*rr[4] + rr[0]*rr[5]- rr[3]*rr[3] + rr[0]*rr[2] - rr[1]*rr[1])) *inv3;
    double e[3] ={spur,spur,spur};
    double h=( spur > 0 )? spur*spur-cof : -1.0;
    if(h>0)
    {
     double g = (spur*cof - detsq)*0.5 - spur*h;
     double sqrth = sqrt(h);
     double d1 = h*h*h - g*g;
     d1= ( d1<0 ) ? atan2(0,-g)*inv3 : atan2(sqrt(d1),-g)*inv3;
     double cth = sqrth * cos(d1);
     double sth = sqrth*SQRT3*sin(d1);
     e[0]+=  cth+cth;
     e[1]+= -cth+sth;
     e[2]+= -cth-sth;
    }
    e[0]=(e[0] < 0) ? 0 : sqrt(e[0]);
    e[1]=(e[1] < 0) ? 0 : sqrt(e[1]);
    e[2]=(e[2] < 0) ? 0 : sqrt(e[2]);
    
    double d=(det<0)? e[0] + e[1] -e[2] : e[0] + e[1]+e[2];
    float xm=s1x*s1x+s1y*s1y+s1z*s1z+s2x*s2x+s2y*s2y+s2z*s2z;
    float r2=(ssq-xm*invfnat-d-d)*invfnat;
    r2=(r2>1e-8)?sqrt(r2) : 0.0;
    matrix->set_matrix(a,b,(float)r2);
   }
  }
 }


//parallel rmsd matrix routines
#ifdef SSE2
float rmsd_sse2_par(int nthreads,int nat,int nmodels,float *coords,float *density){
 const int nat4=(nat%4)? nat/4+1 : nat/4;
 const int pdb_size=3*nat;
 const double inv3=1.0/3.0;
 const float invfnat=1.0f/(float)nat;
 const int pdb4_size=nat4*12;
 float *shuffled_coords= (float*) memalign(16,nmodels*pdb4_size*sizeof(float));
 float *ssqs=(float*) memalign(16,nmodels*sizeof(float));
#ifdef OPENMP 
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
 for (int p=0;p<nmodels;p++){
  float* const new_coords=&(shuffled_coords[p*pdb4_size]);
  float* const old_coords=&(coords[p*pdb_size]);
  for (int i=0;i<pdb4_size;i++)
   new_coords[i]=0; 
  ssqs[p]=shuffle_center_coords4_sse(nat,old_coords,new_coords,0);
 }
#ifdef OPENMP 
 //store first thread results in density
 //store other thread results in my_density;
 float *my_density[nthreads];
 my_density[0]=density;
 for(int i=1;i<nthreads;i++){
  my_density[i]=new float[nmodels];
 }
 for(int i=0;i<nthreads;i++)
  for(int j=0;j<nmodels;j++)
   my_density[i][j]=0.0f;  
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
 for (int a=1;a<nmodels;a++){
  for (int b=0;b<a;b++){
   float rr[8] __attribute__ ((aligned (16))); 
   float ssq=ssqs[a]+ssqs[b];
   const float *coords1=&(shuffled_coords[pdb4_size*a]);
   const float *coords2=&(shuffled_coords[pdb4_size*b]);  
   //SSE2 block
   {
    __m128 mxx = _mm_setzero_ps(); 
    __m128 mxy = _mm_setzero_ps();
    __m128 mxz = _mm_setzero_ps();
    //these will be reused later - others go into block to be release
    {    
     __m128 myx = _mm_setzero_ps(); 
     __m128 myy = _mm_setzero_ps();
     __m128 myz = _mm_setzero_ps();  
     __m128 mzx = _mm_setzero_ps(); 
     __m128 mzy = _mm_setzero_ps();
     __m128 mzz = _mm_setzero_ps();  

     size_t i=0;
     for(i=0;i<pdb4_size;i+=12){
    //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
      __m128 mc2x=_mm_load_ps(&coords2[i]);
      __m128 mc2y=_mm_load_ps(&coords2[i+4]);
      __m128 mc2z=_mm_load_ps(&coords2[i+8]);
      __m128 mc1=_mm_load_ps(&coords1[i]);
     //generate the 4 products that are saved in mr0
      mxx=_mm_add_ps(mxx,_mm_mul_ps(mc1,mc2x));
      mxy=_mm_add_ps(mxy,_mm_mul_ps(mc1,mc2y));
      mxz=_mm_add_ps(mxz,_mm_mul_ps(mc1,mc2z));
      mc1=_mm_load_ps(&coords1[i+4]);
      myx=_mm_add_ps(myx,_mm_mul_ps(mc1,mc2x));
      myy=_mm_add_ps(myy,_mm_mul_ps(mc1,mc2y));
      myz=_mm_add_ps(myz,_mm_mul_ps(mc1,mc2z));
      mc1=_mm_load_ps(&coords1[i+8]);  
      mzx=_mm_add_ps(mzx,_mm_mul_ps(mc1,mc2x));
      mzy=_mm_add_ps(mzy,_mm_mul_ps(mc1,mc2y));
      mzz=_mm_add_ps(mzz,_mm_mul_ps(mc1,mc2z));
     }
     //write out the components to the temp variables
     //sum up the components and write to mr1 - might be a little faster in SSE3 using hadd
     //reuses registers xy and zx as temps - probably not necessary
     mzy = _mm_add_ps(_mm_unpacklo_ps(mzy,mxy),_mm_unpackhi_ps(mzy,mxy));
     mxy = _mm_add_ps(_mm_unpacklo_ps(mxx,mxz),_mm_unpackhi_ps(mxx,mxz));
     mxx = _mm_add_ps(_mm_unpacklo_ps(mzy,mxy),_mm_unpackhi_ps(mzy,mxy)); //mxx holds sums for zy,xx,xy,xz 

     mzy = _mm_add_ps(_mm_unpacklo_ps(mzz,myy),_mm_unpackhi_ps(mzz,myy));
     mxy = _mm_add_ps(_mm_unpacklo_ps(myx,myz),_mm_unpackhi_ps(myx,myz));
     mxy = _mm_add_ps(_mm_unpacklo_ps(mzy,mxy),_mm_unpackhi_ps(mzy,mxy)); //mzy holds sums for zz,yx,yy,yz
   
     mxz = _mm_add_ps(mzx, _mm_movehl_ps(mzx, mzx));
     mxz = _mm_add_ps(mxz, _mm_unpacklo_ps(mxz,mxz)); //(junk,zx,junk,junk);
     __m128 mask = _mm_set_ss(1);
     mzy = _mm_setzero_ps();
     mask=_mm_cmpeq_ps(mask,mzy);
     mxz=_mm_and_ps(mask,mxz);
     mxz=_mm_movelh_ps(mxz,_mm_unpacklo_ps(mxx,mxy));//(0,zx,zy,zz);
     mxy=_mm_and_ps(mask,mxy);
     mxx=_mm_and_ps(mask,mxx);
    }//end block - only mxx,mxy,mxz remain - correspond to R matrix r0,r1,r2
  
    //calculate the determinant using triple product - do addition when calculating rest of dot products
    __m128 mdet = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(1,3,2,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(2,1,3,0))),
                             _mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(2,1,3,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(1,3,2,0))));//cross_product
    mdet=_mm_mul_ps(mxx,mdet); //sum to get dot product

    //calculate the necessary 6 dot products - do additions in groups of 4 for horizontal
    {
     __m128 mt0=_mm_mul_ps(mxx,mxx);
     __m128 mt1=_mm_mul_ps(mxx,mxy);
     __m128 mt2=_mm_mul_ps(mxy,mxy);
     __m128 mt3=_mm_mul_ps(mxx,mxz);
     mxx = _mm_add_ps(_mm_unpacklo_ps(mt0,mt2),_mm_unpackhi_ps(mt0,mt2));
     mt0 = _mm_add_ps(_mm_unpacklo_ps(mt1,mt3),_mm_unpackhi_ps(mt1,mt3));
     _mm_store_ps(rr,_mm_add_ps(_mm_unpacklo_ps(mxx,mt0),_mm_unpackhi_ps(mxx,mt0))); 
    } 
    mxx=_mm_mul_ps(mxz,mxy);
    mxy=_mm_mul_ps(mxz,mxz);
   
    mxx = _mm_add_ps(_mm_unpacklo_ps(mxx,mdet),_mm_unpackhi_ps(mxx,mdet));
    mxy = _mm_add_ps(_mm_unpacklo_ps(mxy,mdet),_mm_unpackhi_ps(mxy,mdet));
    _mm_store_ps(&(rr[4]),_mm_add_ps(_mm_unpacklo_ps(mxx,mxy),_mm_unpackhi_ps(mxx,mxy))); 
   }
   //convert to double - can get by with floats except for g and h but the double is faster 
 
   double detsq=rr[6]*rr[6];
 
   //lower triangular matrix rr
   double spur=((double)(rr[0]+rr[2]+rr[5]))*inv3;
   double cof=((double)(rr[2]*rr[5] - rr[4]*rr[4] + rr[0]*rr[5]- rr[3]*rr[3] + rr[0]*rr[2] - rr[1]*rr[1])) *inv3;
   double e[3] ={spur,spur,spur};
   double h=( spur > 0 )? spur*spur-cof : -1.0;
   if(h>0){
    double g = (spur*cof - detsq)*0.5 - spur*h;
    double sqrth = sqrt(h);
    double d1 = h*h*h - g*g;
    d1= ( d1<0 ) ? atan2(0,-g)*inv3 : atan2(sqrt(d1),-g)*inv3;
    double cth = sqrth * cos(d1);
    double sth = sqrth*SQRT3*sin(d1);
    e[0]+=  cth+cth;
    e[1]+= -cth+sth;
    e[2]+= -cth-sth;
   }
   e[0]=(e[0] < 0) ? 0 : sqrt(e[0]);
   e[1]=(e[1] < 0) ? 0 : sqrt(e[1]);
   e[2]=(e[2] < 0) ? 0 : sqrt(e[2]);
   double d=(rr[6]<0)? e[0] + e[1] -e[2] : e[0] + e[1]+e[2];
   float rms=(ssq-d-d)*invfnat;
   rms=(rms>1e-8)?sqrt(rms) : 0.0f;
#ifdef OPENMP
    int th=omp_get_thread_num();
    my_density[th][a]+=rms;
    my_density[th][b]+=rms;
#else
    density[a]+=rms;
    density[b]+=rms;
#endif   
  }
 }
 //reduction step
#ifdef OPENMP 
 for(int i=1;i<nthreads;i++)
  for(int j=0;j<nmodels;j++)
   density[i]+=my_density[i][j];
 for(int i=1;i<nthreads;i++)
   if(my_density[i])delete [] my_density[i];
#endif
 if(shuffled_coords)free(shuffled_coords);
 if(ssqs) free(ssqs); 
}
template <class T> float rmsd_sse2_par(int nthreads,int nat,int nmodels,float *coords,triangular_matrix<T> *matrix){
 const int nat4=(nat%4)? nat/4+1 : nat/4;
 const int pdb_size=3*nat;
 const double inv3=1.0/3.0;
 const float invfnat=1.0f/(float)nat;
 const int pdb4_size=nat4*12;
 float *shuffled_coords= (float*) memalign(16,nmodels*pdb4_size*sizeof(float));
 float *ssqs=(float*) memalign(16,nmodels*sizeof(float));
 #ifdef OPENMP 
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
 for (int p=0;p<nmodels;p++){
  float* const new_coords=&(shuffled_coords[p*pdb4_size]);
  float* const old_coords=&(coords[p*pdb_size]);
  for (int i=0;i<pdb4_size;i++)
   new_coords[i]=0; 
  ssqs[p]=shuffle_center_coords4_sse(nat,old_coords,new_coords,0);
 }
#ifdef OPENMP 
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
  for (int a=1;a<nmodels;a++){
   for (int b=0;b<a;b++){
    float rr[8] __attribute__ ((aligned (16))); 
    float ssq=ssqs[a]+ssqs[b];
    const float *coords1=&(shuffled_coords[pdb4_size*a]);
    const float *coords2=&(shuffled_coords[pdb4_size*b]);   
    //SSE2 block
    {
     __m128 mxx = _mm_setzero_ps(); 
     __m128 mxy = _mm_setzero_ps();
     __m128 mxz = _mm_setzero_ps();
     //these will be reused later - others go into block to be released
     {    
      __m128 myx = _mm_setzero_ps(); 
      __m128 myy = _mm_setzero_ps();
      __m128 myz = _mm_setzero_ps();  
      __m128 mzx = _mm_setzero_ps(); 
      __m128 mzy = _mm_setzero_ps();
      __m128 mzz = _mm_setzero_ps();  

      size_t i=0;
      for(i=0;i<pdb4_size;i+=12){
      //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
       __m128 mc2x=_mm_load_ps(&coords2[i]);
       __m128 mc2y=_mm_load_ps(&coords2[i+4]);
       __m128 mc2z=_mm_load_ps(&coords2[i+8]);
       __m128 mc1=_mm_load_ps(&coords1[i]);
      //generate the 4 products that are saved in mr0
       mxx=_mm_add_ps(mxx,_mm_mul_ps(mc1,mc2x));
       mxy=_mm_add_ps(mxy,_mm_mul_ps(mc1,mc2y));
       mxz=_mm_add_ps(mxz,_mm_mul_ps(mc1,mc2z));
       mc1=_mm_load_ps(&coords1[i+4]);
       myx=_mm_add_ps(myx,_mm_mul_ps(mc1,mc2x));
       myy=_mm_add_ps(myy,_mm_mul_ps(mc1,mc2y));
       myz=_mm_add_ps(myz,_mm_mul_ps(mc1,mc2z));
       mc1=_mm_load_ps(&coords1[i+8]);  
       mzx=_mm_add_ps(mzx,_mm_mul_ps(mc1,mc2x));
       mzy=_mm_add_ps(mzy,_mm_mul_ps(mc1,mc2y));
       mzz=_mm_add_ps(mzz,_mm_mul_ps(mc1,mc2z));
      }
     //write out the components to the temp variables
      //sum up the components and write to mr1 - might be a little faster in SSE3 using hadd
      //reuses registers xy and zx as temps - probably not necessary
      mzy = _mm_add_ps(_mm_unpacklo_ps(mzy,mxy),_mm_unpackhi_ps(mzy,mxy));
      mxy = _mm_add_ps(_mm_unpacklo_ps(mxx,mxz),_mm_unpackhi_ps(mxx,mxz));
      mxx = _mm_add_ps(_mm_unpacklo_ps(mzy,mxy),_mm_unpackhi_ps(mzy,mxy)); //mxx holds sums for zy,xx,xy,xz 

      mzy = _mm_add_ps(_mm_unpacklo_ps(mzz,myy),_mm_unpackhi_ps(mzz,myy));
      mxy = _mm_add_ps(_mm_unpacklo_ps(myx,myz),_mm_unpackhi_ps(myx,myz));
      mxy = _mm_add_ps(_mm_unpacklo_ps(mzy,mxy),_mm_unpackhi_ps(mzy,mxy)); //mzy holds sums for zz,yx,yy,yz
   
      mxz = _mm_add_ps(mzx, _mm_movehl_ps(mzx, mzx));
      mxz = _mm_add_ps(mxz, _mm_unpacklo_ps(mxz,mxz)); //(junk,zx,junk,junk);
      __m128 mask = _mm_set_ss(1);
      mzy = _mm_setzero_ps();
      mask=_mm_cmpeq_ps(mask,mzy);
      mxz=_mm_and_ps(mask,mxz);
      mxz=_mm_movelh_ps(mxz,_mm_unpacklo_ps(mxx,mxy));//(0,zx,zy,zz);
      mxy=_mm_and_ps(mask,mxy);
      mxx=_mm_and_ps(mask,mxx);
     }//end block - only mxx,mxy,mxz remain - correspond to R matrix r0,r1,r2
  
     //calculate the determinant using triple product - do addition when calculating rest of dot products
     __m128 mdet = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(1,3,2,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(2,1,3,0))),
                             _mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(2,1,3,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(1,3,2,0))));//cross_product
     mdet=_mm_mul_ps(mxx,mdet); //sum to get dot product

     //calculate the necessary 6 dot products - do additions in groups of 4 for horizontal
     {
      __m128 mt0=_mm_mul_ps(mxx,mxx);
      __m128 mt1=_mm_mul_ps(mxx,mxy);
      __m128 mt2=_mm_mul_ps(mxy,mxy);
      __m128 mt3=_mm_mul_ps(mxx,mxz);
      mxx = _mm_add_ps(_mm_unpacklo_ps(mt0,mt2),_mm_unpackhi_ps(mt0,mt2));
      mt0 = _mm_add_ps(_mm_unpacklo_ps(mt1,mt3),_mm_unpackhi_ps(mt1,mt3));
      _mm_store_ps(rr,_mm_add_ps(_mm_unpacklo_ps(mxx,mt0),_mm_unpackhi_ps(mxx,mt0))); 
     } 
     mxx=_mm_mul_ps(mxz,mxy);
     mxy=_mm_mul_ps(mxz,mxz);
   
     mxx = _mm_add_ps(_mm_unpacklo_ps(mxx,mdet),_mm_unpackhi_ps(mxx,mdet));
     mxy = _mm_add_ps(_mm_unpacklo_ps(mxy,mdet),_mm_unpackhi_ps(mxy,mdet));
     _mm_store_ps(&(rr[4]),_mm_add_ps(_mm_unpacklo_ps(mxx,mxy),_mm_unpackhi_ps(mxx,mxy))); 
    }
    //convert to double - can get by with floats except for g and h but the double is faster 
 
    double detsq=rr[6]*rr[6];
 
    //lower triangular matrix rr
    double spur=((double)(rr[0]+rr[2]+rr[5]))*inv3;
    double cof=((double)(rr[2]*rr[5] - rr[4]*rr[4] + rr[0]*rr[5]- rr[3]*rr[3] + rr[0]*rr[2] - rr[1]*rr[1])) *inv3;
    double e[3] ={spur,spur,spur};
    double h=( spur > 0 )? spur*spur-cof : -1.0;
    if(h>0){
     double g = (spur*cof - detsq)*0.5 - spur*h;
     double sqrth = sqrt(h);
     double d1 = h*h*h - g*g;
     d1= ( d1<0 ) ? atan2(0,-g)*inv3 : atan2(sqrt(d1),-g)*inv3;
     double cth = sqrth * cos(d1);
     double sth = sqrth*SQRT3*sin(d1);
     e[0]+=  cth+cth;
     e[1]+= -cth+sth;
     e[2]+= -cth-sth;
    }
    e[0]=(e[0] < 0) ? 0 : sqrt(e[0]);
    e[1]=(e[1] < 0) ? 0 : sqrt(e[1]);
    e[2]=(e[2] < 0) ? 0 : sqrt(e[2]);
    double d=(rr[6]<0)? e[0] + e[1] -e[2] : e[0] + e[1]+e[2];
    float rms=(ssq-d-d)*invfnat;
    rms=(rms>1e-8)?sqrt(rms) : 0.0f;
    matrix->set_matrix(a,b,rms);
   }
  }
 if(ssqs)free(ssqs);
 if(shuffled_coords)free(shuffled_coords);
}
#endif

#ifdef SSE3
float rmsd_sse3_par(int nthreads,int nat,int nmodels,float *coords,float *density){
 const int nat4=(nat%4)? nat/4+1 : nat/4;
 const int pdb_size=3*nat;
 const double inv3=1.0/3.0;
 const float invfnat=1.0f/(float)nat;
 const int pdb4_size=nat4*12;
 float *shuffled_coords= (float*) memalign(16,nmodels*pdb4_size*sizeof(float));
 float *ssqs=(float*) memalign(16,nmodels*sizeof(float));
#ifdef OPENMP 
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
 for (int p=0;p<nmodels;p++){
  float* const new_coords=&(shuffled_coords[p*pdb4_size]);
  float* const old_coords=&(coords[p*pdb_size]);
  for (int i=0;i<pdb4_size;i++)
   new_coords[i]=0; 
  ssqs[p]=shuffle_center_coords4_sse(nat,old_coords,new_coords,0);
 }
#ifdef OPENMP 
 //store first thread results in density
 //store other thread results in my_density;
 float *my_density[nthreads];
 my_density[0]=density;
 for(int i=1;i<nthreads;i++){
  my_density[i]=new float[nmodels];
 }
 for(int i=0;i<nthreads;i++)
  for(int j=0;j<nmodels;j++)
   my_density[i][j]=0.0f;  
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif 
 for (int a=1;a<nmodels;a++){
  for (int b=0;b<a;b++){
   float rr[8] __attribute__ ((aligned (16))); 
   float ssq=ssqs[a]+ssqs[b];
   const float *coords1=&(shuffled_coords[pdb4_size*a]);
   const float *coords2=&(shuffled_coords[pdb4_size*b]);    
    //SSE3 block - not much faster than SSE2 on Phenom but might be better on Intel
    {
     __m128 mxx = _mm_setzero_ps(); 
     __m128 mxy = _mm_setzero_ps();
     __m128 mxz = _mm_setzero_ps();
     //these will be reused later - others go into block to be released
     {    
      __m128 myx = _mm_setzero_ps(); 
      __m128 myy = _mm_setzero_ps();
      __m128 myz = _mm_setzero_ps();  
      __m128 mzx = _mm_setzero_ps(); 
      __m128 mzy = _mm_setzero_ps();
      __m128 mzz = _mm_setzero_ps();  

      size_t i=0;
      for(i=0;i<pdb4_size;i+=12){
     //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
       __m128 mc2x=_mm_load_ps(&coords2[i]);
       __m128 mc2y=_mm_load_ps(&coords2[i+4]);
       __m128 mc2z=_mm_load_ps(&coords2[i+8]);
       __m128 mc1 =_mm_load_ps(&coords1[i]);
    
      //generate the 4 sets of products 
       mxx=_mm_add_ps(mxx,_mm_mul_ps(mc1,mc2x));
       mxy=_mm_add_ps(mxy,_mm_mul_ps(mc1,mc2y));
       mxz=_mm_add_ps(mxz,_mm_mul_ps(mc1,mc2z));
       mc1=_mm_load_ps(&coords1[i+4]);
       myx=_mm_add_ps(myx,_mm_mul_ps(mc1,mc2x));
       myy=_mm_add_ps(myy,_mm_mul_ps(mc1,mc2y));
       myz=_mm_add_ps(myz,_mm_mul_ps(mc1,mc2z));
       mc1=_mm_load_ps(&coords1[i+8]);  
       mzx=_mm_add_ps(mzx,_mm_mul_ps(mc1,mc2x));
       mzy=_mm_add_ps(mzy,_mm_mul_ps(mc1,mc2y));
       mzz=_mm_add_ps(mzz,_mm_mul_ps(mc1,mc2z));
      }
      //write out the components to the temp variables
      //reuses registers xy and zx as temps - probably not necessary
   
      mxx =  _mm_hadd_ps(mzy,mxx);
      mzy =  _mm_hadd_ps(mxy,mxz);
      mxx =  _mm_hadd_ps(mxx,mzy); //mxx holds sums for zy,xx,xy,xz

      mzy =  _mm_hadd_ps(mzz,myx);
      myx =  _mm_hadd_ps(myy,myz);
      mxy =  _mm_hadd_ps(mzy,myx); //mxy holds sums for zz,yx,yy,yz
   
      mzx = _mm_hadd_ps(mzx,mzx);
      mxz = _mm_hadd_ps(mzx,mzx);   //zx,zx,zx,zx
      __m128 mask = _mm_set_ss(1);
      mzy = _mm_setzero_ps();
      mask=_mm_cmpeq_ps(mask,mzy);
      mxz=_mm_and_ps(mask,mxz);
      mxz=_mm_movelh_ps(mxz,_mm_unpacklo_ps(mxx,mxy));//(0,zx,zy,zz);
      mxy=_mm_and_ps(mask,mxy);
      mxx=_mm_and_ps(mask,mxx);
   
     }//end block - only mxx,mxy,mxz remain - correspond to R matrix r0,r1,r2
  
    //calculate the determinant using triple product - do addition when calculating rest of dot products
     __m128 mdet = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(1,3,2,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(2,1,3,0))),
                             _mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(2,1,3,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(1,3,2,0))));//cross_product
     mdet=_mm_mul_ps(mxx,mdet); //sum to get dot product

    //calculate the necessary 6 dot products - do additions in groups of 4
     {
      __m128 mt0=_mm_mul_ps(mxx,mxx);
      __m128 mt1=_mm_mul_ps(mxx,mxy);
      __m128 mt2=_mm_mul_ps(mxy,mxy);
      __m128 mt3=_mm_mul_ps(mxx,mxz);
      mxx = _mm_hadd_ps(mt0,mt1);
      mt0 = _mm_hadd_ps(mt2,mt3);
      _mm_store_ps(rr,_mm_hadd_ps(mxx,mt0)); 
     } 
     mxx = _mm_mul_ps(mxz,mxy);
     mxy = _mm_mul_ps(mxz,mxz);
     mxx = _mm_hadd_ps(mxx,mxy);
     mxy = _mm_hadd_ps(mdet,mdet);
     _mm_store_ps(&(rr[4]),_mm_hadd_ps(mxx,mxy)); 
    }
    //convert to double - can get by with floats except for g and h but the double is faster 

    double detsq=rr[6]*rr[6]; 
    //lower triangular matrix rr
    double inv3=1.0/3.0;
    double spur=((double)(rr[0]+rr[2]+rr[5]))*inv3;
    double cof=((double)(rr[2]*rr[5] - rr[4]*rr[4] + rr[0]*rr[5]- rr[3]*rr[3] + rr[0]*rr[2] - rr[1]*rr[1])) *inv3;
    double e[3] ={spur,spur,spur};
    double h=( spur > 0 )? spur*spur-cof : -1.0;
    if(h>0)
    {
     double g = (spur*cof - detsq)*0.5 - spur*h;
     double sqrth = sqrt(h);
     double d1 = h*h*h - g*g;
     d1= ( d1<0 ) ? atan2(0,-g)*inv3 : atan2(sqrt(d1),-g)*inv3;
     double cth = sqrth * cos(d1);
     double sth = sqrth*SQRT3*sin(d1);
     e[0]+=  cth+cth;
     e[1]+= -cth+sth;
     e[2]+= -cth-sth;
    }
    e[0]=(e[0] < 0) ? 0 : sqrt(e[0]);
    e[1]=(e[1] < 0) ? 0 : sqrt(e[1]);
    e[2]=(e[2] < 0) ? 0 : sqrt(e[2]);
    double d=(rr[6]<0)? e[0] + e[1] -e[2] : e[0] + e[1]+e[2];
    float rms=(ssq-d-d)*invfnat;
    rms=(rms>1e-8)?sqrt(rms) : 0.0f;
#ifdef OPENMP
    int th=omp_get_thread_num();
    my_density[th][a]+=rms;
    my_density[th][b]+=rms;
#else
    density[a]+=rms;
    density[b]+=rms;
#endif   
   }
  }
 //reduction step
#ifdef OPENMP 
 for(int i=1;i<nthreads;i++)
  for(int j=0;j<nmodels;j++)
   density[i]+=my_density[i][j];
 for(int i=1;i<nthreads;i++)
   if(my_density[i])delete [] my_density[i];
#endif
}
template <class T> float rmsd_sse3_par(int nthreads,int nat,int nmodels,float *coords, triangular_matrix<T> *matrix){
const int nat4=(nat%4)? nat/4+1 : nat/4;
 const int pdb_size=3*nat;
 const double inv3=1.0/3.0;
 const float invfnat=1.0f/(float)nat;
 const int pdb4_size=nat4*12;
 float *shuffled_coords= (float*) memalign(16,nmodels*pdb4_size*sizeof(float));
 float *ssqs=(float*) memalign(16,nmodels*sizeof(float));
 #ifdef OPENMP 
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
 for (int p=0;p<nmodels;p++){
  float* const new_coords=&(shuffled_coords[p*pdb4_size]);
  float* const old_coords=&(coords[p*pdb_size]);
  for (int i=0;i<pdb4_size;i++)
   new_coords[i]=0; 
  ssqs[p]=shuffle_center_coords4_sse(nat,old_coords,new_coords,0);
 }
#ifdef OPENMP 
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
  for (int a=1;a<nmodels;a++){
   for (int b=0;b<a;b++){
    float rr[8] __attribute__ ((aligned (16))); 
    float ssq=ssqs[a]+ssqs[b];
    const float *coords1=&(shuffled_coords[pdb4_size*a]);
    const float *coords2=&(shuffled_coords[pdb4_size*b]);   
    //SSE3 block - not much faster than SSE2 on Phenom but might be better on Intel
    {
     __m128 mxx = _mm_setzero_ps(); 
     __m128 mxy = _mm_setzero_ps();
     __m128 mxz = _mm_setzero_ps();
     //these will be reused later - others go into block to be released
     {    
      __m128 myx = _mm_setzero_ps(); 
      __m128 myy = _mm_setzero_ps();
      __m128 myz = _mm_setzero_ps();  
      __m128 mzx = _mm_setzero_ps(); 
      __m128 mzy = _mm_setzero_ps();
      __m128 mzz = _mm_setzero_ps();  

      size_t i=0;
      for(i=0;i<pdb4_size;i+=12){
     //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
       __m128 mc2x=_mm_load_ps(&coords2[i]);
       __m128 mc2y=_mm_load_ps(&coords2[i+4]);
       __m128 mc2z=_mm_load_ps(&coords2[i+8]);
       __m128 mc1 =_mm_load_ps(&coords1[i]);
    
      //generate the 4 sets of products 
       mxx=_mm_add_ps(mxx,_mm_mul_ps(mc1,mc2x));
       mxy=_mm_add_ps(mxy,_mm_mul_ps(mc1,mc2y));
       mxz=_mm_add_ps(mxz,_mm_mul_ps(mc1,mc2z));
       mc1=_mm_load_ps(&coords1[i+4]);
       myx=_mm_add_ps(myx,_mm_mul_ps(mc1,mc2x));
       myy=_mm_add_ps(myy,_mm_mul_ps(mc1,mc2y));
       myz=_mm_add_ps(myz,_mm_mul_ps(mc1,mc2z));
       mc1=_mm_load_ps(&coords1[i+8]);  
       mzx=_mm_add_ps(mzx,_mm_mul_ps(mc1,mc2x));
       mzy=_mm_add_ps(mzy,_mm_mul_ps(mc1,mc2y));
       mzz=_mm_add_ps(mzz,_mm_mul_ps(mc1,mc2z));
      }
      //write out the components to the temp variables
      //reuses registers xy and zx as temps - probably not necessary
   
      mxx =  _mm_hadd_ps(mzy,mxx);
      mzy =  _mm_hadd_ps(mxy,mxz);
      mxx =  _mm_hadd_ps(mxx,mzy); //mxx holds sums for zy,xx,xy,xz

      mzy =  _mm_hadd_ps(mzz,myx);
      myx =  _mm_hadd_ps(myy,myz);
      mxy =  _mm_hadd_ps(mzy,myx); //mxy holds sums for zz,yx,yy,yz
   
      mzx = _mm_hadd_ps(mzx,mzx);
      mxz = _mm_hadd_ps(mzx,mzx);   //zx,zx,zx,zx
      __m128 mask = _mm_set_ss(1);
      mzy = _mm_setzero_ps();
      mask=_mm_cmpeq_ps(mask,mzy);
      mxz=_mm_and_ps(mask,mxz);
      mxz=_mm_movelh_ps(mxz,_mm_unpacklo_ps(mxx,mxy));//(0,zx,zy,zz);
      mxy=_mm_and_ps(mask,mxy);
      mxx=_mm_and_ps(mask,mxx);
   
     }//end block - only mxx,mxy,mxz remain - correspond to R matrix r0,r1,r2
  
    //calculate the determinant using triple product - do addition when calculating rest of dot products
     __m128 mdet = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(1,3,2,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(2,1,3,0))),
                             _mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(2,1,3,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(1,3,2,0))));//cross_product
     mdet=_mm_mul_ps(mxx,mdet); //sum to get dot product

    //calculate the necessary 6 dot products - do additions in groups of 4
     {
      __m128 mt0=_mm_mul_ps(mxx,mxx);
      __m128 mt1=_mm_mul_ps(mxx,mxy);
      __m128 mt2=_mm_mul_ps(mxy,mxy);
      __m128 mt3=_mm_mul_ps(mxx,mxz);
      mxx = _mm_hadd_ps(mt0,mt1);
      mt0 = _mm_hadd_ps(mt2,mt3);
      _mm_store_ps(rr,_mm_hadd_ps(mxx,mt0)); 
     } 
     mxx = _mm_mul_ps(mxz,mxy);
     mxy = _mm_mul_ps(mxz,mxz);
     mxx = _mm_hadd_ps(mxx,mxy);
     mxy = _mm_hadd_ps(mdet,mdet);
     _mm_store_ps(&(rr[4]),_mm_hadd_ps(mxx,mxy)); 
    }
    //convert to double - can get by with floats except for g and h but the double is faster 

    double detsq=rr[6]*rr[6]; 
    //lower triangular matrix rr
    double inv3=1.0/3.0;
    double spur=((double)(rr[0]+rr[2]+rr[5]))*inv3;
    double cof=((double)(rr[2]*rr[5] - rr[4]*rr[4] + rr[0]*rr[5]- rr[3]*rr[3] + rr[0]*rr[2] - rr[1]*rr[1])) *inv3;
    double e[3] ={spur,spur,spur};
    double h=( spur > 0 )? spur*spur-cof : -1.0;
    if(h>0)
    {
     double g = (spur*cof - detsq)*0.5 - spur*h;
     double sqrth = sqrt(h);
     double d1 = h*h*h - g*g;
     d1= ( d1<0 ) ? atan2(0,-g)*inv3 : atan2(sqrt(d1),-g)*inv3;
     double cth = sqrth * cos(d1);
     double sth = sqrth*SQRT3*sin(d1);
     e[0]+=  cth+cth;
     e[1]+= -cth+sth;
     e[2]+= -cth-sth;
    }
    e[0]=(e[0] < 0) ? 0 : sqrt(e[0]);
    e[1]=(e[1] < 0) ? 0 : sqrt(e[1]);
    e[2]=(e[2] < 0) ? 0 : sqrt(e[2]);
    double d=(rr[6]<0)? e[0] + e[1] -e[2] : e[0] + e[1]+e[2];
    float rms=(ssq-d-d)*invfnat;
    rms=(rms>1e-8)?sqrt(rms) : 0.0f; 
    matrix->set_matrix(a,b,rms);
   }
  }
 if(ssqs)free(ssqs);
 if(shuffled_coords)free(shuffled_coords);
}
#endif

#ifdef AVX
float rmsd_avx_par(int nthreads,int nat,int nmodels,float *coords, float *density){ 
 const double inv3=1.0/3.0;
 const float invfnat=1.0f/(float)nat;
 const int upper8= (nat%8)? (nat/8)*8+8 : nat;
 const int pdb8_size=upper8*3;
 const int pdb_size=nat*3;
 float *shuffled_coords= (float*) memalign(32,nmodels*pdb8_size*sizeof(float));
 float *ssqs=(float*) memalign(16,nmodels*sizeof(float));
#ifdef OPENMP      
 int max_threads=omp_get_max_threads();
 nthreads=(max_threads<nthreads)? max_threads : nthreads;
 #pragma omp parallel for num_threads(nthreads) schedule (dynamic)
#endif  
 for (int p=0;p<nmodels;p++){
  float* const new_coords=&(shuffled_coords[p*pdb8_size]);  
  float* const old_coords=&(coords[p*pdb_size]);
  for (int i=0;i<pdb8_size;i++)
   new_coords[i]=0;
  ssqs[p]=shuffle_center_coords8_unaligned_avx(nat,old_coords,new_coords,0);
 } 
#ifdef OPENMP 
 //store first thread results in density
 //store other thread results in my_density;
 float *my_density[nthreads];
 my_density[0]=density;
 for(int i=1;i<nthreads;i++){
  my_density[i]=new float[nmodels];
 }
 for(int i=0;i<nthreads;i++)
  for(int j=0;j<nmodels;j++)
   my_density[i][j]=0.0f;  
  #pragma omp parallel for num_threads(nthreads) schedule (dynamic)
#endif 
 for (int a=1;a<nmodels;a++){
  for (int b=0;b<a;b++){ 
   const float *coords1=&(shuffled_coords[pdb8_size*a]);
   const float *coords2=&(shuffled_coords[pdb8_size*b]);
   float r[16];
   double ssq=ssqs[a]+ssqs[b];
  {
    __m256 mxx = _mm256_setzero_ps(); 
    __m256 mxy = _mm256_setzero_ps();
    __m256 mxz = _mm256_setzero_ps();  
    __m256 myx = _mm256_setzero_ps(); 
    __m256 myy = _mm256_setzero_ps();
    __m256 myz = _mm256_setzero_ps();  
    __m256 mzx = _mm256_setzero_ps(); 
    __m256 mzy = _mm256_setzero_ps();
    __m256 mzz = _mm256_setzero_ps();  

     size_t i=0;
     for( ;i<pdb8_size;i+=24){
      //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
      __m256 mc2x=_mm256_load_ps(&coords2[i]);
      __m256 mc2y=_mm256_load_ps(&coords2[i+8]);
      __m256 mc2z=_mm256_load_ps(&coords2[i+16]);
      __m256 mc1 =_mm256_load_ps(&coords1[i]);

      //generate the 8 sets of products 
      mxx=_mm256_add_ps(mxx,_mm256_mul_ps(mc1,mc2x));
      mxy=_mm256_add_ps(mxy,_mm256_mul_ps(mc1,mc2y));
      mxz=_mm256_add_ps(mxz,_mm256_mul_ps(mc1,mc2z));
      mc1=_mm256_load_ps(&coords1[i+8]);  

      myx=_mm256_add_ps(myx,_mm256_mul_ps(mc1,mc2x));
      myy=_mm256_add_ps(myy,_mm256_mul_ps(mc1,mc2y));
      myz=_mm256_add_ps(myz,_mm256_mul_ps(mc1,mc2z));
      mc1=_mm256_load_ps(&coords1[i+16]);    

      mzx=_mm256_add_ps(mzx,_mm256_mul_ps(mc1,mc2x));
      mzy=_mm256_add_ps(mzy,_mm256_mul_ps(mc1,mc2y));
      mzz=_mm256_add_ps(mzz,_mm256_mul_ps(mc1,mc2z));
     }
     //8 sums at once with AVX hadd instructions and 2 permutes
     mxx =_mm256_hadd_ps(_mm256_hadd_ps(mxx,mxy),_mm256_hadd_ps(mxz,myx)); //contains upper and lower half sums of 4 vectors
     mxy =_mm256_hadd_ps(_mm256_hadd_ps(myy,myz),_mm256_hadd_ps(mzx,mzy)); //contains upper and lower half sums of 4 vectors
     mxz =_mm256_permute2f128_ps(mxx,mxy,0x20);//lower half of first 4  lower half of second 4  bit 1:0  =00 bit3 =0 bit 5:4=10  00100000
     mxx =_mm256_permute2f128_ps(mxx,mxy,0x31);//upper half of first 4   upper half of second 4 bit 1:0 = 01 bit3 =0 bit 5:4=11 gives mask   00110001
     myz =_mm256_add_ps(mxx,mxz);
     _mm256_storeu_ps(r,myz);
     //do 9th sum
     mzx=_mm256_permute2f128_ps(mzz,mzz,0x01);
     mzz=_mm256_add_ps(mzz,mzx);  
     mzz=_mm256_hadd_ps(mzz,mzz);
     mzz=_mm256_hadd_ps(mzz,mzz);
     _mm_store_ss(&(r[8]),_mm256_castps256_ps128(mzz));
    }
    //end AVX block - do the rest in scalar 
  
    //convert to double - can get by with floats except for g and h but the double is faster 
    double det=(double)( r[0] * ( (r[4]*r[8]) - (r[5]*r[7]) )- r[1] * ( (r[3]*r[8]) - (r[5]*r[6]) ) + r[2] * ( (r[3]*r[7]) - (r[4]*r[6]) ));
    double detsq=det*det;
    double rr[6];
    //for symmetric matrix
    //lower triangular matrix rr
    rr[0]=r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    rr[1]=r[3]*r[0]+r[4]*r[1]+r[5]*r[2];
    rr[2]=r[3]*r[3]+r[4]*r[4]+r[5]*r[5];
    rr[3]=r[6]*r[0]+r[7]*r[1]+r[8]*r[2];
    rr[4]=r[6]*r[3]+r[7]*r[4]+r[8]*r[5];
    rr[5]=r[6]*r[6]+r[7]*r[7]+r[8]*r[8]; 

    //lower triangular matrix rr

    double spur=(rr[0]+rr[2]+rr[5])*inv3;
    double cof= (rr[2]*rr[5] - rr[4]*rr[4] + rr[0]*rr[5]- rr[3]*rr[3] + rr[0]*rr[2] - rr[1]*rr[1]) *inv3;
    double e[3] ={spur,spur,spur};
    double h=( spur > 0 )? spur*spur-cof : -1.0;
    if(h>0)
    {
     double g = (spur*cof - detsq)*0.5 - spur*h;
     double sqrth = sqrt(h);
     double d1 = h*h*h - g*g;
     d1= ( d1<0 ) ? atan2(0,-g)*inv3 : atan2(sqrt(d1),-g)*inv3;
     double cth = sqrth * cos(d1);
     double sth = sqrth*SQRT3*sin(d1);
     e[0]+=  cth+cth;
     e[1]+= -cth+sth;
     e[2]+= -cth-sth;
    }
    e[0]=(e[0] < 0) ? 0 : sqrt(e[0]);
    e[1]=(e[1] < 0) ? 0 : sqrt(e[1]);
    e[2]=(e[2] < 0) ? 0 : sqrt(e[2]);
    double d=(det<0)? e[0] + e[1] -e[2] : e[0] + e[1]+e[2];
    float rms=(ssq-d-d)*invfnat;
    rms=(rms>1e-8)?sqrt(rms) : 0.0f;
#ifdef OPENMP
    int th=omp_get_thread_num();
    my_density[th][a]+=rms;
    my_density[th][b]+=rms;
#else
    density[a]+=rms;
    density[b]+=rms;
#endif      
   }
  }

 //reduction step
 #ifdef OPENMP 
 for(int i=1;i<nthreads;i++)
  for(int j=0;j<nmodels;j++)
   density[i]+=my_density[i][j];
 for(int i=1;i<nthreads;i++)
   if(my_density[i])delete [] my_density[i];
 #endif
 
 if(shuffled_coords)free(shuffled_coords);
 if(ssqs)free(ssqs); 
}

template <class T> float rmsd_avx_par(int nthreads,int nat,int nmodels,float *coords,triangular_matrix<T> *matrix){
 const int upper8= (nat%8)? (nat/8)*8+8 : nat;
 const int pdb8_size=upper8*3;
 const int pdb_size=nat*3;
 const double inv3=1.0/3.0;
 const float invfnat=1.0f/(float)nat;

 float *shuffled_coords= (float*) memalign(32,nmodels*pdb8_size*sizeof(float));
 float *ssqs=(float*) memalign(16,nmodels*sizeof(float));

#ifdef OPENMP      
 int max_threads=omp_get_max_threads();
 nthreads=(max_threads<nthreads)? max_threads : nthreads;  
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif  
 for (int p=0;p<nmodels;p++){
  float* const new_coords=&(shuffled_coords[p*pdb8_size]);  
  float* const old_coords=&(coords[p*pdb_size]);
  for (int i=0;i<pdb8_size;i++)
   new_coords[i]=0;
  ssqs[p]=shuffle_center_coords8_unaligned_avx(nat,old_coords,new_coords,0);
 } 

#ifdef OPENMP 
 #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif
 for (int a=1;a<nmodels;a++){
  for (int b=0;b<a;b++){
   float r[16];//seems to be a problem making 32 bit aligned arrays with dynamic OpenMP scheduling
   const float *coords1=&(shuffled_coords[pdb8_size*a]);
   const float *coords2=&(shuffled_coords[pdb8_size*b]);
   double ssq=ssqs[a]+ssqs[b];
   {
    __m256 mxx = _mm256_setzero_ps(); 
    __m256 mxy = _mm256_setzero_ps();
    __m256 mxz = _mm256_setzero_ps();  
    __m256 myx = _mm256_setzero_ps(); 
    __m256 myy = _mm256_setzero_ps();
    __m256 myz = _mm256_setzero_ps();  
    __m256 mzx = _mm256_setzero_ps(); 
    __m256 mzy = _mm256_setzero_ps();
    __m256 mzz = _mm256_setzero_ps();  

     size_t i=0;
     for( ;i<pdb8_size;i+=24){
      //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
      __m256 mc2x=_mm256_load_ps(&coords2[i]);
      __m256 mc2y=_mm256_load_ps(&coords2[i+8]);
      __m256 mc2z=_mm256_load_ps(&coords2[i+16]);
      __m256 mc1 =_mm256_load_ps(&coords1[i]);

      //generate the 8 sets of products 
      mxx=_mm256_add_ps(mxx,_mm256_mul_ps(mc1,mc2x));
      mxy=_mm256_add_ps(mxy,_mm256_mul_ps(mc1,mc2y));
      mxz=_mm256_add_ps(mxz,_mm256_mul_ps(mc1,mc2z));
      mc1=_mm256_load_ps(&coords1[i+8]);  

      myx=_mm256_add_ps(myx,_mm256_mul_ps(mc1,mc2x));
      myy=_mm256_add_ps(myy,_mm256_mul_ps(mc1,mc2y));
      myz=_mm256_add_ps(myz,_mm256_mul_ps(mc1,mc2z));
      mc1=_mm256_load_ps(&coords1[i+16]);    

      mzx=_mm256_add_ps(mzx,_mm256_mul_ps(mc1,mc2x));
      mzy=_mm256_add_ps(mzy,_mm256_mul_ps(mc1,mc2y));
      mzz=_mm256_add_ps(mzz,_mm256_mul_ps(mc1,mc2z));
     }
     //8 sums at once with AVX hadd instructions and 2 permutes
     mxx =_mm256_hadd_ps(_mm256_hadd_ps(mxx,mxy),_mm256_hadd_ps(mxz,myx)); //contains upper and lower half sums of 4 vectors
     mxy =_mm256_hadd_ps(_mm256_hadd_ps(myy,myz),_mm256_hadd_ps(mzx,mzy)); //contains upper and lower half sums of 4 vectors
     mxz =_mm256_permute2f128_ps(mxx,mxy,0x20);//lower half of first 4  lower half of second 4  bit 1:0  =00 bit3 =0 bit 5:4=10  00100000
     mxx =_mm256_permute2f128_ps(mxx,mxy,0x31);//upper half of first 4   upper half of second 4 bit 1:0 = 01 bit3 =0 bit 5:4=11 gives mask   00110001
     myz =_mm256_add_ps(mxx,mxz);
     _mm256_storeu_ps(r,myz); //aligned write doesn't work with OpenMP dynamic for some reason
     //do 9th sum
     mzx=_mm256_permute2f128_ps(mzz,mzz,0x01);
     mzz=_mm256_add_ps(mzz,mzx);  
     mzz=_mm256_hadd_ps(mzz,mzz);
     mzz=_mm256_hadd_ps(mzz,mzz);
     _mm_store_ss(&(r[8]),_mm256_castps256_ps128(mzz));
    }
    //end AVX block - do the rest in scalar 
  
    //convert to double - can get by with floats except for g and h but the double is faster 
    double det=(double)( r[0] * ( (r[4]*r[8]) - (r[5]*r[7]) )- r[1] * ( (r[3]*r[8]) - (r[5]*r[6]) ) + r[2] * ( (r[3]*r[7]) - (r[4]*r[6]) ));
    double detsq=det*det;
    double rr[6];
    //for symmetric matrix
    //lower triangular matrix rr
    rr[0]=r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
    rr[1]=r[3]*r[0]+r[4]*r[1]+r[5]*r[2];
    rr[2]=r[3]*r[3]+r[4]*r[4]+r[5]*r[5];
    rr[3]=r[6]*r[0]+r[7]*r[1]+r[8]*r[2];
    rr[4]=r[6]*r[3]+r[7]*r[4]+r[8]*r[5];
    rr[5]=r[6]*r[6]+r[7]*r[7]+r[8]*r[8]; 

    //lower triangular matrix rr

    double spur=(rr[0]+rr[2]+rr[5])*inv3;
    double cof= (rr[2]*rr[5] - rr[4]*rr[4] + rr[0]*rr[5]- rr[3]*rr[3] + rr[0]*rr[2] - rr[1]*rr[1]) *inv3;
    double e[3] ={spur,spur,spur};
    double h=( spur > 0 )? spur*spur-cof : -1.0;
    if(h>0)
    {
     double g = (spur*cof - detsq)*0.5 - spur*h;
     double sqrth = sqrt(h);
     double d1 = h*h*h - g*g;
     d1= ( d1<0 ) ? atan2(0,-g)*inv3 : atan2(sqrt(d1),-g)*inv3;
     double cth = sqrth * cos(d1);
     double sth = sqrth*SQRT3*sin(d1);
     e[0]+=  cth+cth;
     e[1]+= -cth+sth;
     e[2]+= -cth-sth;
    }
    e[0]=(e[0] < 0) ? 0 : sqrt(e[0]);
    e[1]=(e[1] < 0) ? 0 : sqrt(e[1]);
    e[2]=(e[2] < 0) ? 0 : sqrt(e[2]);
    double d=(det<0)? e[0] + e[1] -e[2] : e[0] + e[1]+e[2];
    float rms=(ssq-d-d)*invfnat;
    rms=(rms>1e-8)?sqrt(rms) : 0.0f;
    matrix->set_matrix(a,b,rms);
   }
 }

 if(shuffled_coords)free(shuffled_coords);
 if(ssqs)free(ssqs);
}
#endif
