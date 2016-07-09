//Ling-Hong Hung Aug 2013
//routines for CPU RMSD/TMscore - no OpenCL

#include "lite.h"
#include "error_handlers.h"
#ifdef OPENMP
#include <omp.h>
#endif
#include <iostream>
#define PI 3.1415926535897932
#define SQRT3 1.732050807568877
#define COORDS_BUFFER_SIZE 768

extern double gtimer1,gtimer2;
using namespace std;


//subroutines for all v all TMscore
//CPU version of RMSD/TMScore using hybrid Kabsch/quaternion 
float rmsd_cpu(int nat,float *coords1,float *coords2,float *rmatrix);
float rmsd_cpu(int nat,float *coords1,float *coords2);
float tmscore_rmsd_cpu(int nat,float *coords1,float *coords2,float bR[3][3], float bt[3],float *rmsd);

//SSE/AVX accelerated RMSD/TMScore

#ifdef SSE2
//sse2
void print_m128(__m128 p);
//rmsd

//The SSE/AVX routines use pre-shuffled coordinates for the SOA ie. split into separate x, y and z streams and padded with zeros to fit boundaries 
//This is normally less or equally efficient than just transforming on the fly but for all vs all matrices
//In addition for RMSD the coordinates are all pre-centered which eliminates a lot of the complexity
//this needs only to be done once for the entire ensemble of coordinates so costs little for RMSD
//Using AVX there are enough registers to do also the centering efficiently in place 
//For TMScore new centers need to be calculated for each subset of atoms used to seed the comparison so this useful for the first subset only
int shuffle_coords4_sse (int nstructs,int pdb_size, float *coords, float *shuffled_coords,float *ssqs);
int shuffle_coords4_sse (int nstructs,int pdb_size, float *coords, float *shuffled_coords,float *ssqs, float *centroids);

float center_coords4_sse(int nat, float *coords);  

//TMscore
void shuffle_tmscore_coords_soa_sse(int nstructs, int nat, float *coords,float **x,float **y,float **z);
void split_coords_sse(int nat, float *coords, float *x, float *y,float *z);

//main TMscore routine
float tmscore_cpu_soa_sse2(int nat, float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3],float *rmsd);

//score fun routines - applies matrix and determines next set of atoms for RMSD in next iteration
int score_fun_soa_sse(int nat, float d0, float d, float *r,float *x1, float *y1, float *z1, float *x2, float *y2, float *z2, int *ialign,int *nalign,float *tm_score);

//Kabsch-quatXXX organizes the coordinates to a contiguous cache friendly form 
//for SSE there are not enough registers to calculate the centers and sum of squares in the same pass as the covariances
float kabsch_quat_soa_sse2(int nat, int *map, float *x1, float *y1,float *z1, float *x2,float *y2, float *z2,float *r); //given a subset of coords - finds optimal rotation and transformation to apply to whole set of coords

//coords_sum_ssqXXX are used by Kabsch-quat routines to determine center and sum of squares separately for each subset of atoms
//AVX code uses both halves of the 256 bit registers to be able to do both calculations in one pass
float coords_sum_ssq_xyz_sse2(int nat, float *x, float *y,float *z,float center[3]);

//calculates optimal matrix for starting subsets of matched atoms 
float rmsd_sse2_matrix_xyz (int nat,float *c1x,float *c1y, float *c1z,float *c2x,float *c2y, float *c2z,float center1[3], float center2[3],double ssq,float u[3][3]);

//simple sse accelerated rotation multiplication
void R34v4_sse2 (float r[16],float *x,float *Rx); 

//LG_score_ Levitt-Gerstein metric scoring routines - i.e. calculate and apply rotation matrix and return LG score
//This is the main routine in the TMscore algorithm
//The SOA form is faster and is easily extended to 256-wide AVX - However the AOS form can almost be as fast for SSE depending on HADD performance
float LG_score_soa_sse (float r[16],int nat, float *x1,float *y1,float *z1, float *x2, float *y2, float *z2, float *d,float invd0d0);
#endif

#ifdef SSE3
//SSE3 versions - only slightly faster using HADD
float rmsd_sse3 (int nat,float *coords1,float *coords2,double ssq,float *rmatrix); 
float rmsd_sse3 (int nat,float *coords1,float *coords2,double ssq,float u[3][3]); //returns rotation matrix

float tmscore_cpu_soa_sse3(int nat, float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3],float *rmsd);

float kabsch_quat_soa_sse3(int nat, int *map, float *x1, float *y1,float *z1, float *x2,float *y2, float *z2,float *r,float *coords_buffer);
float coords_sum_ssq_xyz_sse3(int nat, float *x, float *y,float *z,float center[3]);
void R34v4_sse3 (float r[16],float *x,float *Rx); 
float rmsd_sse3_matrix_xyz (int nat,float *c1x,float *c1y, float *c1z,float *c2x,float *c2y, float *c2z,float center1[3], float center2[3],double ssq,float u[3][3]);  //uses precalculated ssq and split coords for TMScore coords subsets 
#endif
#ifdef AVX  
//AVX versions - much larger changes - use of half-registers - hadds are trickier 
//speed gain offset by - emulated AVX instructions - cache pressure and loading of wide data  
//also previously mino non-vectorized code becomes limiting factor - see about 5-20% increase on Bulldozers and laptop i7 overall

void print_m256(__m256 p);

//use memalign to allocate non-local arrays __attribute__ does not work for heap storage
int shuffle_coords8_avx (int nstructs,int pdb_size, float *coords, float *shuffled_coords,float *ssqs);
int shuffle_coords8_avx (int nstructs,int pdb_size, float *coords, float *shuffled_coords,float *ssqs,float *centroids); 
float shuffle_center_coords8_avx(int nat, float *coords, float *shuffled_coords,float centroid[3]);

//rmsd using precentered coords
float rmsd_avx  (int nat,float *coords1,float *coords2,double ssq,float u[3][3]);
float rmsd_avx (int nat,float *coords1,float *coords2,double ssq);
//rmsd using uncentered coords using wide registers to do centering in same pass as calculation of covariacnes 
//used for TMScore but can be modified for normal usage 
float rmsd_uncentered_avx (int nat,float *c1x,float *c1y, float *c1z,float *c2x,float *c2y, float *c2z,float *rm);  

//earlier version using precalculated ssq sse
float rmsd_avx_matrix_xyz (int nat,float *c1x,float *c1y, float *c1z,float *c2x,float *c2y, float *c2z,float center1[3], float center2[3],double ssq,float u[3][3]); 

void shuffle_tmscore_coords_soa_avx(int nstructs, int nat, float *coords,float **x,float **y,float **z);
void split_coords_avx(int nat, float *coords, float *x, float *y,float *z);
float tmscore_cpu_soa_avx(int nat, float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3],float *rmsd);

float kabsch_quat_soa_avx(int nat, int *map, float *x1, float *y1,float *z1, float *x2,float *y2, float *z2,float *r,float *coords_buffer);
float coords_sum_ssq_avx(int nat, float *x, float *y,float *z,float center[3]);
int score_fun_soa_avx(int nat, float d0, float d, float *r,float *x1, float *y1, float *z1, float *x2, float *y2, float *z2, int *ialign,int *nalign,float *tm_score);
float LG_score_soa_avx (float r[16],int nat, float *x1,float *y1,float *z1, float *x2, float *y2, float *z2, float *d,float invd0d0);

#endif
//scalar routines
void center_all_coords(int nstructs,int nat,float *coords,float *centered_coords);

//iterative TMscore routine
int score_fun_dcoords(int nat, float d0, float d, double R[3][3], double t[3],double *coords1, double *coords2,double *acoords,int *ialign,int *nalign,float *tm_score);

//eigenvector routine from quaternion method

template <class T> void rmatrix (T ev,T r[3][3],T u[3][3]);
template <class T> void rmatrix (T ev,T r[9],T u[3][3]);

//optimised Kabsch routine to be used with eigenvector matrix calculation
double rmsd_svd(int nat,double *my_coords,double u[3][3],double t[3],bool rmsd_flag);

void R34v4(float *r,float *x, float *Rx);


template <class T> void dump_matrix(T u[3][3]);
template <class T> void dump_vector(T u[3]);


//scalar

void center_all_coords(int nstructs,int nat,float *coords,float *centered_coords){
 for (int p=0;p<nstructs;p++){
  float sums[3]={0.0f,0.0f,0.0f};
  float invnat=1.0f/(float) nat;
  float const *pcoords=&(coords[p*nat*3]);
  float *cpcoords=&(centered_coords[p*nat*3]);
  for(int i=0;i<nat;i++){
   sums[0]+=pcoords[3*i];
   sums[1]+=pcoords[3*i+1];
   sums[2]+=pcoords[3*i+2];
  }
  for(int i=0;i<3;i++){
   sums[i]*=invnat; 
  }
  for(int i=0;i<nat;i++){
   cpcoords[3*i]=pcoords[3*i]-sums[0];
   cpcoords[3*i+1]=pcoords[3*i+1]-sums[1];
   cpcoords[3*i+2]=pcoords[3*i+2]-sums[2];
  }
 }
}
template <class T> void rmatrix (T ev,T r[3][3],T u[3][3]){   
 //calculate rotation matrix
 
 T a00=(r[0][0]+r[1][1]+r[2][2]);
 T a01=(r[1][2]-r[2][1]);
 T a02=(r[2][0]-r[0][2]);
 T a03=(r[0][1]-r[1][0]);
 T a11=(r[0][0]-r[1][1]-r[2][2]);
 T a12=(r[0][1]+r[1][0]);
 T a13=(r[2][0]+r[0][2]);
 T a22=(-r[0][0]+r[1][1]-r[2][2]);
 T a23=(r[1][2]+r[2][1]);
 T a33=(-r[0][0]-r[1][1]+r[2][2]);

 //from Theobald
 a00-=ev;a11-=ev;a22-=ev;a33-=ev; 
 T a2233_3223 = a22 * a33 - a23 * a23; 
 T a1233_3123 = a12 * a33-a13*a23;
 T a1223_3122 = a12 * a23 - a13 * a22; 
 T a0232_3022 = a02 * a23-a03*a22;
 T a0233_3023 = a02 * a33 - a03 * a23;
 T a0231_3021 = a02 * a13-a03*a12;

 T q[4]={a11*a2233_3223-a12*a1233_3123+a13*a1223_3122, -a01*a2233_3223+a12*a0233_3023-a13*a0232_3022,a01*a1233_3123-a11*a0233_3023+a13*a0231_3021,-a01*a1223_3122+a11*a0232_3022-a12*a0231_3021};
 
 T invlen2q=1.0f/(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
 T aj=q[0]*q[0]*invlen2q;
 T xj=q[1]*q[1]*invlen2q;
 T yj=q[2]*q[2]*invlen2q;
 T zj=q[3]*q[3]*invlen2q;
 T  xy = q[1] * q[2]*invlen2q;
 T  az = q[0] * q[3]*invlen2q;
 T  zx = q[3] * q[1]*invlen2q;
 T  ay = q[0] * q[2]*invlen2q;
 T  yz = q[2] * q[3]*invlen2q;
 T  ax = q[0] * q[1]*invlen2q; 

 u[0][0]= aj + xj - yj - zj; u[0][1]= 2.0f * (xy + az); u[0][2]= 2.0f * (zx - ay); 
 u[1][0]= 2.0f * (xy - az);  u[1][1]=aj - xj + yj - zj; u[1][2]= 2.0f * (yz + ax); 
 u[2][0]= 2.0f * (zx + ay),  u[2][1]= 2.0f * (yz - ax); u[2][2]= aj - xj - yj + zj;
}
template <class T> void rmatrix (T ev,T r[9],T u[3][3]){   
 //calculate rotation matrix
 
 T a00=(r[0]+r[4]+r[8]);
 T a01=(r[5]-r[7]);
 T a02=(r[6]-r[2]);
 T a03=(r[1]-r[3]);
 T a11=(r[0]-r[4]-r[8]);
 T a12=(r[1]+r[3]);
 T a13=(r[6]+r[2]);
 T a22=(-r[0]+r[4]-r[8]);
 T a23=(r[5]+r[7]);
 T a33=(-r[0]-r[4]+r[8]);

 //from Theobald
 a00-=ev;a11-=ev;a22-=ev;a33-=ev; 
 T a2233_3223 = a22 * a33 - a23 * a23; 
 T a1233_3123 = a12 * a33-a13*a23;
 T a1223_3122 = a12 * a23 - a13 * a22; 
 T a0232_3022 = a02 * a23-a03*a22;
 T a0233_3023 = a02 * a33 - a03 * a23;
 T a0231_3021 = a02 * a13-a03*a12;

 T q[4]={a11*a2233_3223-a12*a1233_3123+a13*a1223_3122, -a01*a2233_3223+a12*a0233_3023-a13*a0232_3022,a01*a1233_3123-a11*a0233_3023+a13*a0231_3021,-a01*a1223_3122+a11*a0232_3022-a12*a0231_3021};
 
 T len2q=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
 if(!len2q){
  //return the identity matrix
  u[0][0]= 1.0f; u[0][1]= 0.0f; u[0][2]= 0.0f; 
  u[1][0]= 0.0f; u[1][1]= 1.0f; u[1][2]= 0.0f; 
  u[2][0]= 0.0f; u[2][1]= 0.0f; u[2][2]= 1.0f;
 }
 else{
  T invlen2q=1.0/len2q;
  T aj=q[0]*q[0]*invlen2q;
  T xj=q[1]*q[1]*invlen2q;
  T yj=q[2]*q[2]*invlen2q;
  T zj=q[3]*q[3]*invlen2q;
  T  xy = q[1] * q[2]*invlen2q;
  T  az = q[0] * q[3]*invlen2q;
  T  zx = q[3] * q[1]*invlen2q;
  T  ay = q[0] * q[2]*invlen2q;
  T  yz = q[2] * q[3]*invlen2q;
  T  ax = q[0] * q[1]*invlen2q; 

  u[0][0]= aj + xj - yj - zj; u[0][1]= 2.0 * (xy + az); u[0][2]= 2.0 * (zx - ay); 
  u[1][0]= 2.0 * (xy - az);  u[1][1]=aj - xj + yj - zj; u[1][2]= 2.0 * (yz + ax); 
  u[2][0]= 2.0 * (zx + ay);  u[2][1]= 2.0 * (yz - ax); u[2][2]= aj - xj - yj + zj;
 }
}  
void rmatrix_d(double ev,double r[3][3],double u[3][3]){   
 //calculate rotation matrix
 
 double a00=(r[0][0]+r[1][1]+r[2][2]);
 double a01=(r[1][2]-r[2][1]);
 double a02=(r[2][0]-r[0][2]);
 double a03=(r[0][1]-r[1][0]);
 double a11=(r[0][0]-r[1][1]-r[2][2]);
 double a12=(r[0][1]+r[1][0]);
 double a13=(r[2][0]+r[0][2]);
 double a22=(-r[0][0]+r[1][1]-r[2][2]);
 double a23=(r[1][2]+r[2][1]);
 double a33=(-r[0][0]-r[1][1]+r[2][2]);

 //from Theobald
 a00-=ev;a11-=ev;a22-=ev;a33-=ev; 
 double a2233_3223 = a22 * a33 - a23 * a23; 
 double a1233_3123 = a12 * a33-a13*a23;
 double a1223_3122 = a12 * a23 - a13 * a22; 
 double a0232_3022 = a02 * a23-a03*a22;
 double a0233_3023 = a02 * a33 - a03 * a23;
 double a0231_3021 = a02 * a13-a03*a12;

 double q[4]={a11*a2233_3223-a12*a1233_3123+a13*a1223_3122, -a01*a2233_3223+a12*a0233_3023-a13*a0232_3022,a01*a1233_3123-a11*a0233_3023+a13*a0231_3021,-a01*a1223_3122+a11*a0232_3022-a12*a0231_3021};
 
 double len2q=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
 if(!len2q){
  //return the identity matrix
  u[0][0]= 1.0f; u[0][1]= 0.0f; u[0][2]= 0.0f; 
  u[1][0]= 0.0f; u[1][1]= 1.0f; u[1][2]= 0.0f; 
  u[2][0]= 0.0f; u[2][1]= 0.0f; u[2][2]= 1.0f;
 }
 else{
  double invlen2q=1.0/len2q;
  double aj=q[0]*q[0]*invlen2q;
  double xj=q[1]*q[1]*invlen2q;
  double yj=q[2]*q[2]*invlen2q;
  double zj=q[3]*q[3]*invlen2q;
  double  xy = q[1] * q[2]*invlen2q;
  double  az = q[0] * q[3]*invlen2q;
  double  zx = q[3] * q[1]*invlen2q;
  double  ay = q[0] * q[2]*invlen2q;
  double  yz = q[2] * q[3]*invlen2q;
  double  ax = q[0] * q[1]*invlen2q; 

  u[0][0]= aj + xj - yj - zj; u[0][1]= 2.0 * (xy + az); u[0][2]= 2.0 * (zx - ay); 
  u[1][0]= 2.0 * (xy - az);  u[1][1]=aj - xj + yj - zj; u[1][2]= 2.0 * (yz + ax); 
  u[2][0]= 2.0 * (zx + ay);  u[2][1]= 2.0 * (yz - ax); u[2][2]= aj - xj - yj + zj;
 }
} 
int score_fun_dcoords(int nat, float d0, float d, double R[3][3], double t[3],double *coords1, double *coords2,double *acoords,int *ialign,int *nalign,float *tm_score){
 //ialign points to atom number
 int k,ncut=0,nchange=0,my_nalign=*nalign;
 double invd0d0=1.0/(double)(d0*d0);
 double d2,dist;
 double *my_dist=0,my_score=0;
 if(!(my_dist=(double*)malloc(nat*sizeof(double))))exit(FALSE);
 //keep nmin smallest distances && distances < dtmp
 for(k=0;k<nat;k++){
  double u[3];
  int m=3*k;
  u[0]=t[0]+R[0][0]*coords1[m]+R[1][0]*coords1[m+1]+R[2][0]*coords1[m+2]-coords2[m];
  u[1]=t[1]+R[0][1]*coords1[m]+R[1][1]*coords1[m+1]+R[2][1]*coords1[m+2]-coords2[m+1];
  u[2]=t[2]+R[0][2]*coords1[m]+R[1][2]*coords1[m+1]+R[2][2]*coords1[m+2]-coords2[m+2];
  dist=u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
  my_score+=1.0/(1.0+dist*invd0d0);
  my_dist[k]=dist;
 }

 //adjust d until there are at least 3 the same
 while(ncut <3)
 {
  d2=d*d;
  ncut=0;
  for(k=0;k<nat;k++)
   if(my_dist[k]<d2) ncut++;
  d+=.5;
 }
 ncut=0;
 for(k=0;k<nat;k++)
  if(my_dist[k]<d2)
  {  
   if(ncut < my_nalign && ialign[ncut] == k)ncut++;
   else
   {
    nchange=1;
    ialign[ncut++]=k;
   }
  }
 if(my_dist)free(my_dist);
 *tm_score=my_score/(double)nat;
 if(!nchange)return(0);
 int m=0;
 for(k=0;k<ncut;k++)
 {
  int n=ialign[k];
  acoords[m++]=coords1[3*n];
  acoords[m++]=coords1[3*n+1];
  acoords[m++]=coords1[3*n+2];
  acoords[m++]=coords2[3*n];
  acoords[m++]=coords2[3*n+1];
  acoords[m++]=coords2[3*n+2];
 }
 *nalign=ncut;
 return(1);
}
float rmsd_cpu(int nat,float *coords1,float *coords2){
 double e0=0,d,rr[6], ss[6], e[3],r[3][3],rms=0;
 double spur, det, cof, h, g, cth, sth, sqrth, p, sigma;
 float s1x=0,s1y=0,s1z=0,s2x=0,s2y=0,s2z=0,ssq=0;
 float sxx=0,sxy=0,sxz=0,syx=0,syy=0,syz=0,szx=0,szy=0,szz=0;

 for(int i=0;i<3;i++)
  for(int j=0;j<3;j++)
   r[i][j]=0;
 for (int i=0;i<nat;i++){
  int m=3*i;
  float c1x=coords1[m];
  float c1y=coords1[m+1];
  float c1z=coords1[m+2];
  float c2x=coords2[m];
  float c2y=coords2[m+1];
  float c2z=coords2[m+2];
  s1x+=c1x;s1y+=c1y;s1z+=c1z;s2x+=c2x;s2y+=c2y;s2z+=c2z;
  sxx+=c1x*c2x; sxy+=c1x*c2y; sxz+=c1x*c2z; syx+=c1y*c2x; syy+=c1y*c2y; syz+=c1y*c2z;szx+=c1z*c2x; szy+=c1z*c2y; szz+=c1z*c2z;
  ssq+=c1x*c1x+c1y*c1y+c1z*c1z+c2x*c2x+c2y*c2y+c2z*c2z;
 }
 float invfnat=1.0/(float) nat;

 r[0][0]=sxx-s1x*s2x*invfnat;
 r[0][1]=sxy-s1x*s2y*invfnat;
 r[0][2]=sxz-s1x*s2z*invfnat;
 r[1][0]=syx-s1y*s2x*invfnat;
 r[1][1]=syy-s1y*s2y*invfnat;
 r[1][2]=syz-s1y*s2z*invfnat;
 r[2][0]=szx-s1z*s2x*invfnat;
 r[2][1]=szy-s1z*s2y*invfnat;
 r[2][2]=szz-s1z*s2z*invfnat;

 det= r[0][0] * ( (r[1][1]*r[2][2]) - (r[1][2]*r[2][1]) )- r[0][1] * ( (r[1][0]*r[2][2]) - (r[1][2]*r[2][0]) ) + r[0][2] * ( (r[1][0]*r[2][1]) - (r[1][1]*r[2][0]) );
 sigma=det;
 //for symmetric matrix
 //lower triangular matrix rr
 {
  int m=0;
  for(int i=0;i<3;i++)
   for(int j=0;j<=i;j++)
    rr[m++]= r[i][0]*r[j][0]+ r[i][1]*r[j][1]+ r[i][2]*r[j][2];
 }      
 spur=(rr[0]+rr[2]+rr[5]) / 3.0; //average of diagonal sum
 cof=(((((rr[2]*rr[5] - rr[4]*rr[4]) + rr[0]*rr[5])- rr[3]*rr[3]) + rr[0]*rr[2]) - rr[1]*rr[1]) / 3.0;
 for(int i=0;i<3;i++)
  e[i]=spur;  
 h=( spur > 0 )? spur*spur-cof : -1;
 if(h>0)
 {
  det*=det;
  g = (spur*cof - det)/2.0 - spur*h;
  sqrth = sqrt(h);
  d = h*h*h - g*g;
  d= ( d<0 ) ? atan2(0,-g) / 3.0 : atan2(sqrt(d),-g)/3.0;
  cth = sqrth * cos(d);
  sth = sqrth*SQRT3*sin(d);
  e[0] = (spur + cth) + cth;
  e[1] = (spur - cth) + sth;
  e[2] = (spur - cth) - sth;
 }
 for(int i=0;i<3;i++)
  e[i]=(e[i] < 0) ? 0 : sqrt(e[i]);
 d=e[2];
 if(sigma < 0) d=-d;
 d+=e[1] + e[0];
 //translation for 1 to 2;
 //calculate R vectors - d is the ev;

  double xm=s1x*s1x+s1y*s1y+s1z*s1z+s2x*s2x+s2y*s2y+s2z*s2z;
  double r2=(ssq-xm*invfnat-d-d)*invfnat;
  rms=(r2>0.0) ? sqrt(r2) : 0.0;
 return(rms);
}

double rmsd_svd(int nat,double *dcoords,double u[3][3], double t[3],bool rmsd_flag){
 const double inv3=1.0/3.0;
 const double dnat=(double)nat; 
 const double invdnat=1.0/(double)nat;  
  double drms=0,r[9]={0,0,0,0,0,0,0,0,0};
  double s1x=0,s1y=0,s1z=0,s2x=0,s2y=0,s2z=0,ssq=0;
  int m=0;
  for (int i=0;i<nat;i++){
   double c1x=dcoords[m++];
   double c1y=dcoords[m++];
   double c1z=dcoords[m++];
   double c2x=dcoords[m++];
   double c2y=dcoords[m++];
   double c2z=dcoords[m++];
   r[0]+=c1x*c2x; r[1]+=c1x*c2y; r[2]+=c1x*c2z; r[3]+=c1y*c2x; r[4]+=c1y*c2y; r[5]+=c1y*c2z; r[6]+=c1z*c2x; r[7]+=c1z*c2y; r[8]+=c1z*c2z;
   s1x+=c1x;s1y+=c1y;s1z+=c1z;s2x+=c2x;s2y+=c2y;s2z+=c2z;
   if(rmsd_flag) ssq+=c1x*c1x+c1y*c1y+c1z*c1z+c2x*c2x+c2y*c2y+c2z*c2z;
  }
  r[0]-=s1x*s2x*invdnat;
  r[1]-=s1x*s2y*invdnat;
  r[2]-=s1x*s2z*invdnat;
  r[3]-=s1y*s2x*invdnat;
  r[4]-=s1y*s2y*invdnat;
  r[5]-=s1y*s2z*invdnat;
  r[6]-=s1z*s2x*invdnat;
  r[7]-=s1z*s2y*invdnat;
  r[8]-=s1z*s2z*invdnat;
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
  if(rmsd_flag){
   double xm=s1x*s1x+s1y*s1y*+s1z*s1z+s2x*s2x+s2y*s2y+s2z*s2z;
   drms=(ssq-xm*dnat-d-d)*invdnat;
   drms=(drms>1e-8)?sqrt(drms) : 0.0f;
  }
  if(rmsd_flag){
   double xm=s1x*s1x+s1y*s1y+s1z*s1z+s2x*s2x+s2y*s2y+s2z*s2z;
   double r2=(ssq-xm*invdnat-d-d)*invdnat;
   drms=(r2>1.0e-8) ? sqrt(r2) : 0.0;
  }
 if(u && t){
  rmatrix(d,r,u);	 
  t[0] =s2x*invdnat - (u[0][0]*s1x*invdnat + u[1][0]*s1y*invdnat + u[2][0]*s1z*invdnat);
  t[1]= s2y*invdnat - (u[0][1]*s1x*invdnat + u[1][1]*s1y*invdnat + u[2][1]*s1z*invdnat); 
  t[2]= s2z*invdnat - (u[0][2]*s1x*invdnat + u[1][2]*s1y*invdnat + u[2][2]*s1z*invdnat);
 }
 return(drms);
}
 
float tmscore_rmsd_cpu(int nat,float *coords1,float *coords2, float bR[3][3], float bt[3], float *rmsd){
 int nalign;
 float max_score=-1,rms;
 double R[3][3],t[3];
 int ialign[nat];
 double acoords[6*nat],dcoords1[3*nat],dcoords2[3*nat];  
 for(int i=0;i<nat*3;i++){
  dcoords1[i]=coords1[i];
  dcoords2[i]=coords2[i];
 }  
 //d0
 float d0=1.24*pow((nat-15),(1.0/3.0))-1.8;
 if(d0< 0.5)d0=0.5;
 //d0_search ----->
 float d,d0_search=d0;
 if(d0_search > 8)d0_search=8;
 if(d0_search <4.5)d0_search=4.5;
 //iterative parameters ----->

 int n_it=20;      //maximum number of iterations
 int n_init_max=6; //maximum number of L_init
 int n_init=0;
 int L_ini_min=4;
 int L_ini[6];

 if(nat < 4) L_ini_min=nat;
 int len=nat;
 int divisor=1;
 while(len > L_ini_min && n_init <5)
 {
  L_ini[n_init++]=len;
  divisor*=2;
  len=nat/divisor;
 }
 L_ini[n_init++]=4;
 if (L_ini[n_init-1] > L_ini_min)L_ini[n_init++]=L_ini_min;;

 // find the maximum score starting from local structures superposition
 float score; //TM-score
 for (int seed=0;seed<n_init;seed++)
 {
  //find the initial rotation matrix using the initial seed residues
  int L_init=L_ini[seed];
  for(int istart=0;istart<=nat-L_init;istart++)
  {
   int nchanges=1;
   int nalign=L_init;
   {
    int m=0;
    int n=0;
    for(int i=0;i<nat;i++)
    {
     if(i>=istart && i<istart+L_init)
     {
      ialign[n++]=i;
      int p=3*i;
      acoords[m++]=dcoords1[p];
      acoords[m++]=dcoords1[p+1];
      acoords[m++]=dcoords1[p+2];
      acoords[m++]=dcoords2[p];
      acoords[m++]=dcoords2[p+1];
      acoords[m++]=dcoords2[p+2]; 
     }
    }
   }
   if(!seed && rmsd)
    *rmsd=(float)rmsd_svd(nalign,acoords,R, t,1);
   else
    rmsd_svd(nalign,acoords,R, t,0);
   score_fun_dcoords(nat, d0, d0_search-1,R,t,dcoords1,dcoords2,acoords,ialign,&nalign,&score);

   d=d0_search+1;
   if(score > max_score)
   {
    max_score=score;
    if(bR && bt){
     for(int j=0;j<3;j++)
     {
      bt[j]=t[j];
      for(int k=0;k<3;k++)
       bR[j][k]=R[j][k];
     }
    }
   }    
   //extend search from seed
   for (int iter=0;iter<n_it && nchanges;iter++)
   {
    rmsd_svd(nalign,acoords,R, t,0);
    nchanges=score_fun_dcoords(nat, d0,d,R,t,dcoords1,dcoords2,acoords,ialign,&nalign,&score);

    if(score > max_score){
     max_score=score;
     if(bR && bt){
      for(int j=0;j<3;j++){
       bt[j]=t[j];
       for(int k=0;k<3;k++)
        bR[j][k]=R[j][k];
      }
     }
    }
   }
  }
 }
 return(max_score);
}

#ifdef SSE2
//sse2
void print_m128(__m128 p){
 float t[4];
 _mm_storeu_ps(t,p);
 for (int i=0;i<4;i++)
  fprintf(stderr,"%10.3f ",t[i]);
 fprintf(stderr,"\n");
}

float shuffle_center_coords4_sse(int nat, float *coords, float *shuffled_coords,float centroid[3]){ //returns ssq - does not use aligned coords - but outputs aligned coords
 float invnat=1.0f/(float)nat;
 float sums[4] __attribute__ ((aligned (16)));
 float ssq __attribute__ ((aligned (16)));
 int nat4=(nat%4)? nat/4+1 : nat/4;
 int padded_nat=nat4*4;
 int i=0;
 int lower_nat4=(nat/4)*4;

 //first pass reads and calculates sums
 {
  __m128 sum0 = _mm_setzero_ps();
  __m128 sum1 = _mm_setzero_ps();
  __m128 sum2 = _mm_setzero_ps();
  __m128 ssq0 = _mm_setzero_ps();
  
  for(;i<lower_nat4*3;i+=12){
   __m128 p0 = _mm_loadu_ps(&(coords[i]));   // x0y0z0x1
   sum0= _mm_add_ps(sum0,p0);
   ssq0= _mm_add_ps(ssq0,_mm_mul_ps(p0,p0));
   __m128 p1 = _mm_loadu_ps(&(coords[i+4])); // y1z1x2y2
   sum1 = _mm_add_ps(sum1,p1);
   ssq0 = _mm_add_ps(ssq0,_mm_mul_ps(p1,p1)); 
   __m128 p2 = _mm_loadu_ps(&(coords[i+8])); // z2x3y3z3
   sum2= _mm_add_ps(sum2,p2);
   ssq0= _mm_add_ps(ssq0,_mm_mul_ps(p2,p2));
  } 
  __m128 t = _mm_shuffle_ps(sum0, sum1, _MM_SHUFFLE(0, 1, 0, 3));
  sum0=_mm_add_ps(sum0,_mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 2, 3, 0)));
  sum0=_mm_add_ps(sum0,_mm_shuffle_ps(sum1, sum2, _MM_SHUFFLE(0, 0, 3, 2)));  
  sum0=_mm_add_ps(sum0,_mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(0, 3, 2, 1)));  
  _mm_store_ps(sums,sum0);
    
  t=_mm_add_ps(ssq0,_mm_movehl_ps(ssq0,ssq0));
  ssq0=_mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
  _mm_store_ss(&ssq,ssq0);
 }
 //finish sums in scalar
 //subtract and write out
 
 for(;i<nat*3;i+=3){
  sums[0]+=coords[i];
  ssq+=coords[i]*coords[i];
  sums[1]+=coords[i+1];
  ssq+=coords[i+1]*coords[i+1];
  sums[2]+=coords[i+2];
  ssq+=coords[i+2]*coords[i+2];
 }

 float invfnat=1.0f/(float)nat;
 ssq-=sums[0]*sums[0]*invfnat+sums[1]*sums[1]*invfnat+sums[2]*sums[2]*invfnat;
 sums[0]*=invfnat;sums[1]*=invfnat;sums[2]*=invfnat;
  //correct ssq for centered coords
 //subtract from coords 
 i=0;
 //subtract from coords 
 __m128 s0 =_mm_load1_ps(sums);
 __m128 s1 =_mm_load1_ps(&(sums[1]));
 __m128 s2 =_mm_load1_ps(&(sums[2]));
 for(;i<lower_nat4*3;i+=12){
  __m128 x0y0z0x1 = _mm_loadu_ps(&(coords[i]));
  __m128 y1z1x2y2 = _mm_loadu_ps(&(coords[i+4]));  
  __m128 z2x3y3z3 = _mm_loadu_ps(&(coords[i+8]));
  __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2,z2x3y3z3,_MM_SHUFFLE( 2,1,3,2));  
  __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1,y1z1x2y2,_MM_SHUFFLE( 1,0,2,1)); 
  _mm_store_ps (&(shuffled_coords[i]),_mm_sub_ps(_mm_shuffle_ps(x0y0z0x1,x2y2x3y3,_MM_SHUFFLE( 2,0,3,0)),s0));
  _mm_store_ps (&(shuffled_coords[i+4]),_mm_sub_ps(_mm_shuffle_ps(y0z0y1z1,x2y2x3y3,_MM_SHUFFLE( 3,1,2,0)),s1));
  _mm_store_ps (&(shuffled_coords[i+8]),_mm_sub_ps(_mm_shuffle_ps(y0z0y1z1,z2x3y3z3,_MM_SHUFFLE( 3,0,3,1)),s2));
 }

 if(nat%4){
  int k=i;
  for(;i<nat*3;i+=3){
   shuffled_coords[k] =coords[i]-sums[0];
   shuffled_coords[k+4] =coords[i+1]-sums[1];
   shuffled_coords[k+8] =coords[i+2]-sums[2];
   k++;
  }
 }
 if(centroid){
  centroid[0]=sums[0];
  centroid[1]=sums[1];
  centroid[2]=sums[2];
 }
 return(ssq);
}
float center_coords4_sse(int nat, float *coords){//rearranged 4x 4y 4z - must be aligned - uses sse2
 float invnat=1.0f/(float)nat;
 float out[4] __attribute__ ((aligned (16)));
 int nat4=(nat%4)? nat/4+1 : nat/4;
 int padded_nat=nat4*4;
 int i=0;
 __m128 sumx = _mm_setzero_ps(); 
 {
  __m128 sumy = _mm_setzero_ps();
  __m128 sumz = _mm_setzero_ps();
  __m128 ssx  = _mm_setzero_ps(); 

  for(;i<padded_nat*3;i+=12){
  //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
   __m128 mx=_mm_load_ps(&(coords[i]));
   sumx=_mm_add_ps(sumx,mx);
   ssx=_mm_add_ps(ssx,_mm_mul_ps(mx,mx));
   __m128 my=_mm_load_ps(&(coords[i+4]));
   sumy=_mm_add_ps(sumy,my);
   ssx=_mm_add_ps(ssx,_mm_mul_ps(my,my));
   __m128 mz=_mm_load_ps(&(coords[i+8]));    
   sumz=_mm_add_ps(sumz,mz);
   ssx=_mm_add_ps(ssx,_mm_mul_ps(mz,mz));
  }
  __m128 t1 = _mm_add_ps(_mm_unpacklo_ps(ssx,sumy), _mm_unpackhi_ps(ssx,sumy));
  __m128 t2 = _mm_add_ps(_mm_unpacklo_ps(sumx,sumz),_mm_unpackhi_ps(sumx,sumz));
  sumx=_mm_add_ps(_mm_unpacklo_ps(t1,t2),_mm_unpackhi_ps(t1,t2));//(ssq,sumx,sumy,sumz)
 }
 {
  __m128 mnat = _mm_set_ss(invnat);
  mnat = _mm_shuffle_ps(mnat,mnat,0x0000);
  sumx=_mm_mul_ps(mnat,sumx);
  _mm_store_ps(out,sumx);;
 } 
 __m128 sumy=_mm_shuffle_ps(sumx,sumx,0x00AA);
 __m128 sumz=_mm_shuffle_ps(sumx,sumx,0x00ff); 
  sumx=_mm_shuffle_ps(sumx,sumx,0x0055);

  int pad=padded_nat*3-nat%4*3;
  for(i=0;i<pad;i+=12){
   __m128 mx=_mm_load_ps(&(coords[i]));
   mx=_mm_sub_ps(mx,sumx);
   __m128 my=_mm_load_ps(&(coords[i+4]));
   my=_mm_sub_ps(my,sumy);
   __m128 mz=_mm_load_ps(&(coords[i+8]));    
   mz=_mm_sub_ps(mz,sumz);
   _mm_store_ps(&(coords[i]),mx);   
   _mm_store_ps(&(coords[i+4]),my);   
   _mm_store_ps(&(coords[i+8]),mz);
  }
  //do the last set of 12 in scalar
  int j=i/3;
  int m=0;
  for(;j<nat;j++){
   coords[i+m]   -=out[1];
   coords[i+m+4] -=out[2];
   coords[i+m+8] -=out[3];
   m++;
  }
 return(out[0]/invnat-out[1]*out[1]-out[2]*out[2]-out[3]*out[3]); 
}
void shuffle_tmscore_coords_soa_sse(int nstructs, int nat, float *coords,float **x,float **y,float **z){
 //need to ensure alignment 16 nat must be multiple of 4
 int anat= (nat%4) ? (nat/4)*4+4 : nat;
 *x=(float*)memalign(16,anat*nstructs*sizeof(float));
 *y=(float*)memalign(16,anat*nstructs*sizeof(float));
 *z=(float*)memalign(16,anat*nstructs*sizeof(float));
 for(int p=0;p<nstructs;p++){ 
  float *const c=&(coords[p*nat*3]); 
  float *const mx=&((*x)[p*anat]);
  float *const my=&((*y)[p*anat]);
  float *const mz=&((*z)[p*anat]);
  split_coords_sse(nat,c,mx,my,mz);
  if(nat != anat){
   for(int i=nat;i<anat;i++){
    mx[i]=0.0f;
    my[i]=0.0f;
    mz[i]=0.0f;
   }
  }
 }
}
float tmscore_cpu_soa_sse2(int nat,float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3],float *rmsd){
 int nalign=0,best_nalign=0;
 int *ialign=new int[nat];
 int *best_align=new int[nat];
 float max_score=-1,rms;
 float r[16] __attribute__ ((aligned (16)));
 //d0
 float d0=1.24*pow((nat-15),(1.0/3.0))-1.8;
      if(d0< 0.5)d0=0.5;
 //d0_search ----->
 float d,d0_search=d0;
      if(d0_search > 8)d0_search=8;
      if(d0_search <4.5)d0_search=4.5;
 //iterative parameters ----->
 int n_it=20;      //maximum number of iterations
 int n_init_max=6; //maximum number of L_init
 int n_init=0;
 int L_ini_min=4;
 int L_ini[6];

 if(nat < 4) L_ini_min=nat;
 int len=nat;
 int divisor=1;
 while(len > L_ini_min && n_init <5){
  L_ini[n_init++]=len;
  divisor*=2;
  len=nat/divisor;
 }
 L_ini[n_init++]=4;
 if (L_ini[n_init-1] > L_ini_min)L_ini[n_init++]=L_ini_min;;

 // find the maximum score starting from local structures superposition
 float score; //TM-score
 for (int seed=0;seed<n_init;seed++)
 {
  //find the initial rotation matrix using the initial seed residues
  int L_init=L_ini[seed];
  for(int istart=0;istart<=nat-L_init;istart++)
  {
   int nchanges=1;
   int nalign=L_init;
   {
    int m=0;
    int n=0;
    for(int i=0;i<nat;i++){
     if(i>=istart && i<istart+L_init){
      ialign[n++]=i;
     }
    }
   }
   if(rmsd && !seed)
    *rmsd=kabsch_quat_soa_sse2(nalign,ialign,x1,y1,z1,x2,y2,z2,r);
   else
    kabsch_quat_soa_sse2(nalign,ialign,x1,y1,z1,x2,y2,z2,r);
   score_fun_soa_sse(nat, d0, d0_search-1,r,x1,y1,z1,x2,y2,z2,ialign,&nalign,&score);
   d=d0_search+1.0f;
   if(score > max_score){
    max_score=score;
    memmove(best_align,ialign,nalign*sizeof(int));
    best_nalign=nalign;
   }
   //extend search from seed
   for (int iter=0;iter<n_it && nchanges;iter++){
    kabsch_quat_soa_sse2(nalign,ialign,x1,y1,z1,x2,y2,z2,r);
    nchanges=score_fun_soa_sse(nat, d0, d,r,x1,y1,z1,x2,y2,z2,ialign,&nalign,&score);
    if(score > max_score){
     max_score=score;
     memmove(best_align,ialign,nalign*sizeof(int));
     best_nalign=nalign;
    }
   }
  }
 }
 
 //for best frame re-calculate matrix with double precision
 double R[3][3],t[3];
 double *acoords=new double [best_nalign*6];
 for(int k=0;k<best_nalign;k++){
  int i=best_align[k];
  acoords[6*k]  =x1[i];
  acoords[6*k+1]=y1[i];
  acoords[6*k+2]=z1[i];
  acoords[6*k+3]=x2[i];
  acoords[6*k+4]=y2[i];
  acoords[6*k+5]=z2[i];
 }
rmsd_svd(best_nalign,acoords,R, t,0);
 if(bR){
  for(int i=0;i<3;i++)
   for(int j=0;j<3;j++)
    bR[i][j]=R[i][j];
 } 
 if(bt){ 
  for(int i=0;i<3;i++)
   bt[i]=t[i];
 } 
 double invd0d0=1.0/(double)(d0*d0);
 double dist;
 float invdnat=1.0/(double)nat;
 double my_score=0;
 for(int k=0;k<nat;k++){
  double u[3];
  double v[3]={x1[k],y1[k],z1[k]};
  double w[3]={x2[k],y2[k],z2[k]};
  u[0]=t[0]+R[0][0]*v[0]+R[1][0]*v[1]+R[2][0]*v[2]-w[0];
  u[1]=t[1]+R[0][1]*v[0]+R[1][1]*v[1]+R[2][1]*v[2]-w[1];
  u[2]=t[2]+R[0][2]*v[0]+R[1][2]*v[1]+R[2][2]*v[2]-w[2];
  dist=u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
  my_score+=1.0/(1.0+dist*invd0d0);
 }
 
 delete [] best_align;
 delete [] ialign;
 delete [] acoords;
 return(my_score*invdnat);
}
int score_fun_soa_sse(int nat, float d0, float d, float *r,float *x1, float *y1, float *z1, float *x2, float *y2, float *z2, int *ialign,int *nalign,float *tm_score){
 //ialign points to atom number
 int k,ncut=0,nchange=0,my_nalign=*nalign,upper_nat= (nat%4)?(nat/4)*4+4 : nat;
 float d2=d*d,invfnat=1.0f/(float)nat;
 float invd0d0=1.0f/(d0*d0);
 float* dist=(float*)memalign(16,upper_nat*sizeof(float));
 //keep nmin smallest distances && distances < dtmp
 float my_score=LG_score_soa_sse(r,nat,x1,y1,z1,x2,y2,z2,dist,invd0d0);
 for(int k=0;k<nat;k++){
  if(dist[k]<d2) ncut++;  
 }

 //adjust d until there are at least 3 the same - rare use another routine for this
 while(ncut <3){
  d2=d*d;
  ncut=0;
  for(k=0;k<nat;k++)
   if(dist[k]<d2) ncut++;
  d+=0.5f;
 }
 ncut=0;
 for(k=0;k<nat;k++){
  if(dist[k]<d2){  
   if(ncut < my_nalign && ialign[ncut] == k)ncut++;
   else{
    nchange=1;
    ialign[ncut++]=k;
   }
  }
 }
 free (dist);
 
 *tm_score=my_score*invfnat;
 if(!nchange)return(0);
 *nalign=ncut;
 return(1);
}
void split_coords_sse(int nat, float *coords, float *x, float *y,float *z){ //returns ssq - use aligned coords - do once
 int i=0,k=0;
 int lower_nat4=(nat/4)*4;
 for(;i<lower_nat4*3;i+=12){
  __m128 x0y0z0x1 = _mm_loadu_ps(&(coords[i]));
  __m128 y1z1x2y2 = _mm_loadu_ps(&(coords[i+4]));  
  __m128 z2x3y3z3 = _mm_loadu_ps(&(coords[i+8]));
  __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2,z2x3y3z3,_MM_SHUFFLE( 2,1,3,2));  
  __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1,y1z1x2y2,_MM_SHUFFLE( 1,0,2,1)); 
  _mm_storeu_ps (&x[k],_mm_shuffle_ps(x0y0z0x1,x2y2x3y3,_MM_SHUFFLE( 2,0,3,0)));
  _mm_storeu_ps (&y[k],_mm_shuffle_ps(y0z0y1z1,x2y2x3y3,_MM_SHUFFLE( 3,1,2,0)));
  _mm_storeu_ps (&z[k],_mm_shuffle_ps(y0z0y1z1,z2x3y3z3,_MM_SHUFFLE( 3,0,3,1)));
  k+=4;
 }
 //do the last set in scalar
 i=lower_nat4;
 for(;i<nat;i++){
  int p=i*3;
  x[k] =coords[p];
  y[k] =coords[p+1];
  z[k] =coords[p+2];
  k++;;
 }
}
float kabsch_quat_soa_sse2(int nat, int *map, float *x1, float *y1,float *z1, float *x2,float *y2, float *z2,float *r){
 int nat4=(nat%4) ? nat/4+1 : nat/4; 
 float center1[3],center2[3];
 float u[3][3];
 float *mem = (float*)memalign(16,6*nat4*4*sizeof(float));
 
 memset(mem,0,nat4*24*sizeof(float));
 float* c1x= mem;
 float* c1y= &mem[nat4*4];
 float* c1z= &mem[nat4*8]; 
 float* c2x= &mem[nat4*12];
 float* c2y= &mem[nat4*16];
 float* c2z= &mem[nat4*20];

 for(int i=0;i<nat;i++){
  int n=map[i];
  c1x[i]=x1[n];
  c1y[i]=y1[n];
  c1z[i]=z1[n];
  c2x[i]=x2[n];
  c2y[i]=y2[n];
  c2z[i]=z2[n];  
 }
 float ssq=coords_sum_ssq_xyz_sse2(nat,c1x,c1y,c1z,center1);
       ssq+=coords_sum_ssq_xyz_sse2(nat,c2x,c2y,c2z,center2);          
 float rms=rmsd_sse2_matrix_xyz(nat,c1x,c1y,c1z,c2x,c2y,c2z,center1,center2,(double)ssq,u);
 float rr[16] __attribute__ ((aligned (16)))=
  {-u[0][0],-u[1][0],-u[2][0],center2[0],
   -u[0][1],-u[1][1],-u[2][1],center2[1],
   -u[0][2],-u[1][2],-u[2][2],center2[2]};
 float w[4]__attribute__ ((aligned (16)));
 float v[4]__attribute__ ((aligned (16)))={center1[0],center1[1],center1[2],1.0f};
 R34v4_sse2(rr,v,w);
 r[0] =u[0][0]; r[1]=u[1][0];r[2] =u[2][0];r[3]=w[0];
 r[4] =u[0][1]; r[5]=u[1][1];r[6] =u[2][1];r[7]=w[1];
 r[8] =u[0][2]; r[9]=u[1][2];r[10]=u[2][2];r[11]=w[2];
 r[12]=0.0f;r[13]=0.0f;r[14]=0.0f;r[15]=1.0f;
 free (mem);
 return(rms);
}
float coords_sum_ssq_xyz_sse2(int nat, float *x, float *y,float *z,float center[3]){
 int lower_nat=(nat/4)*4;
 float invfnat=1.0/(float)nat;
 int i=0;
 float sums[4] __attribute__ ((aligned (16)));
 { 
  __m128 sumx = _mm_setzero_ps();
  __m128 sumy = _mm_setzero_ps();
  __m128 sumz = _mm_setzero_ps();
  __m128 ssq  = _mm_setzero_ps();
  for(;i<lower_nat;i+=4){
   __m128 p0 = _mm_load_ps(&(x[i]));  
   sumx= _mm_add_ps(sumx,p0);
   ssq= _mm_add_ps(ssq,_mm_mul_ps(p0,p0));
   __m128 p1 = _mm_load_ps(&(y[i])); 
   sumy = _mm_add_ps(sumy,p1);
   ssq = _mm_add_ps(ssq,_mm_mul_ps(p1,p1)); 
   __m128 p2 = _mm_load_ps(&(z[i])); 
   sumz = _mm_add_ps(sumz,p2);
   ssq = _mm_add_ps(ssq,_mm_mul_ps(p2,p2)); 
  }
 __m128 t1  = _mm_add_ps(_mm_unpacklo_ps(sumx,sumz),_mm_unpackhi_ps(sumx,sumz));
 __m128 t2  = _mm_add_ps(_mm_unpacklo_ps(sumy,ssq),_mm_unpackhi_ps(sumy,ssq));
 __m128 sum0= _mm_add_ps(_mm_unpacklo_ps(t1,t2),_mm_unpackhi_ps(t1,t2));  
  _mm_store_ps(sums,sum0);
 }
 for(;i<nat;i++){
  sums[0]+=x[i];
  sums[3]+=x[i]*x[i];
  sums[1]+=y[i];
  sums[3]+=y[i]*y[i];
  sums[2]+=z[i];
  sums[3]+=z[i]*z[i];;
 }
 for(int i=0;i<3;i++)
  center[i]=sums[i]*invfnat;
 return(sums[3]-center[0]*sums[0]-center[1]*sums[1]-center[2]*sums[2]);
} 
float LG_score_soa_sse (float r[16],int nat, float *x1,float *y1,float *z1, float *x2, float *y2, float *z2, float *d,float invd0d0){//coords organized x4,y4,z4
 //no hadds - compatible with SSE2
 float fsum=0;
 __m128 r0 = _mm_load_ps(r);
 __m128 r1 = _mm_load_ps(&(r[4]));
 __m128 r2 = _mm_load_ps(&(r[8]));
 __m128 one=_mm_set_ps(1.0f,1.0f,1.0f,1.0f);
 //4th multiplication unecessary as it is all zeros
 __m128 d0 = _mm_load1_ps(&invd0d0);
 __m128 sum=_mm_setzero_ps();    
 int i=0, lower_nat4=(nat/4)*4;
 //four points at a time - otherwise do it scalar
 for( ;i <lower_nat4; i+=4){
  __m128 mx1 = _mm_load_ps(&(x1[i]));
  __m128 my1 = _mm_load_ps(&(y1[i]));
  __m128 mz1 = _mm_load_ps(&(z1[i]));
  __m128 mx2 = _mm_load_ps(&(x2[i])); 

  __m128 tx1= _mm_add_ps(_mm_mul_ps(my1,_mm_shuffle_ps(r0,r0,0x55)),_mm_mul_ps(mx1,_mm_shuffle_ps(r0,r0,0x00)));
  __m128 tx2= _mm_add_ps(_mm_mul_ps(mz1,_mm_shuffle_ps(r0,r0,0xAA)),_mm_shuffle_ps(r0,r0,0xFF));
  tx2= _mm_add_ps(tx2,tx1);
  tx2= _mm_sub_ps(tx2,mx2);
   mx2 = _mm_load_ps(&(y2[i])); 
  __m128 d1 = _mm_mul_ps(tx2,tx2);

  tx1= _mm_add_ps(_mm_mul_ps(my1,_mm_shuffle_ps(r1,r1,0x55)),_mm_mul_ps(mx1,_mm_shuffle_ps(r1,r1,0x00)));
  tx2= _mm_add_ps(_mm_mul_ps(mz1,_mm_shuffle_ps(r1,r1,0xAA)),_mm_shuffle_ps(r1,r1,0xFF));
  tx2= _mm_add_ps(tx2,tx1);  
  tx2= _mm_sub_ps(tx2,mx2);
  mx2= _mm_load_ps(&(z2[i])); 
  d1 = _mm_add_ps(d1,_mm_mul_ps(tx2,tx2));
  
  tx1= _mm_add_ps(_mm_mul_ps(my1,_mm_shuffle_ps(r2,r2,0x55)),_mm_mul_ps(mx1,_mm_shuffle_ps(r2,r2,0x00)));
  tx2= _mm_add_ps(_mm_mul_ps(mz1,_mm_shuffle_ps(r2,r2,0xAA)),_mm_shuffle_ps(r2,r2,0xFF));
  tx2= _mm_add_ps(tx2,tx1);   
  tx2= _mm_sub_ps(tx2,mx2);
  d1 = _mm_add_ps(d1,_mm_mul_ps(tx2,tx2));
  _mm_store_ps(&(d[i]),d1); //write out 4 differences            
  mx1= _mm_mul_ps(d1,d0);
  mx1= _mm_add_ps(mx1,one);
#ifdef FAST_DIVISION
  mx1= _mm_rcp_ps(mx1);
#else
  mx1=_mm_div_ps(one,mx1);
#endif
  sum=_mm_add_ps(sum,mx1); 
 }

 for( ;i <nat; i++){
  float v[4] __attribute__ ((aligned (16))) ={x1[i],y1[i],z1[i],1}; 
  float w[4] __attribute__ ((aligned (16))) ={x2[i],y2[i],z2[i],1}; 
  __m128 x1 = _mm_load_ps(v);
  __m128 y1 = _mm_load_ps(w);  
  __m128 a1 = _mm_mul_ps(r0,x1);
  __m128 b1 = _mm_mul_ps(r1,x1);
  
  x1 = _mm_mul_ps(r2,x1);
  
  a1 = _mm_add_ps(_mm_unpacklo_ps(a1,x1),_mm_unpackhi_ps(a1,x1));
  x1 = _mm_add_ps(_mm_unpacklo_ps(b1,one),_mm_unpackhi_ps(b1,one));
  x1 = _mm_add_ps(_mm_unpacklo_ps(a1,x1),_mm_unpackhi_ps(a1,x1)); 
  
  x1 = _mm_sub_ps(x1,y1); 
  x1 = _mm_mul_ps(x1,x1);
 
  x1 = _mm_add_ps(x1, _mm_movehl_ps(x1, x1));
  x1 = _mm_add_ss(x1, _mm_shuffle_ps(x1, x1, 1));
  
  _mm_store_ss(&(d[i]),x1);
  x1= _mm_mul_ss(x1,d0);
  x1= _mm_add_ss(x1,one);
#ifdef FAST_DIVISION
   x1= _mm_rcp_ps(x1);
#else
   x1=_mm_div_ps(one,x1);
#endif
  sum=_mm_add_ss(sum,x1);
 } 
 //not worth writing two versions for SSE/SSE3 - avoid the hadds
 sum=_mm_add_ps(sum,_mm_movehl_ps(sum, sum));
 sum=_mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
 _mm_store_ss(&(fsum),sum);
 return(fsum);
}

void R34v4_sse2 (float r[12],float *x,float *Rx){
 __m128 r0 = _mm_load_ps(r);
 __m128 r1 = _mm_load_ps(&(r[4]));
 __m128 r2 = _mm_load_ps(&(r[8])); 
 __m128 one= _mm_set_ss(1.0f); 
 __m128 mu = _mm_load_ps(x);
 __m128 m0 = _mm_mul_ps(r0,mu);
 __m128 m1 = _mm_mul_ps(r1,mu);
 __m128 m2 = _mm_mul_ps(r2,mu);

  __m128 s1 =_mm_add_ps(_mm_unpacklo_ps(m0,m2),_mm_unpackhi_ps(m0,m2));
  __m128 s2 =_mm_add_ps(_mm_unpacklo_ps(m1,one),_mm_unpackhi_ps(m1,one));
 _mm_store_ps(Rx,_mm_add_ps(_mm_unpacklo_ps(s1,s2),_mm_unpackhi_ps(s1,s2)));
}

float rmsd_sse2_matrix_xyz (int nat,float *c1x,float *c1y, float *c1z,float *c2x,float *c2y, float *c2z,float center1[3], float center2[3],double ssq,float u[3][3]){
 //c1 has (0,sum(x1),sum(y1),sum(z1));
 //c2 has (0,mean(x1),mean(y2),mean(z2));
 float fnat=(float)nat;
 float c1[4] __attribute__ ((aligned (16))) ={0.0f,center1[0]*fnat,center1[1]*fnat,center1[2]*fnat};
 float c2[4] __attribute__ ((aligned (16))) ={0.0f,center2[0],center2[1],center2[2]};  
 float r0[4] __attribute__ ((aligned (16)));  
 float r1[4] __attribute__ ((aligned (16)));  
 float r2[4] __attribute__ ((aligned (16))); 
 float rr[8] __attribute__ ((aligned (16)));
 int nat4=(nat%4)? nat/4+1 : nat/4;
 int padded_nat=nat4*4;
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
   for(i=0;i<padded_nat;i+=4){
   //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
    __m128 mc2x=_mm_load_ps(&(c2x[i]));
    __m128 mc2y=_mm_load_ps(&(c2y[i]));
    __m128 mc2z=_mm_load_ps(&(c2z[i]));
    __m128 mc1 =_mm_load_ps(&(c1x[i]));
    
   //generate the 4 sets of products 
    mxx=_mm_add_ps(mxx,_mm_mul_ps(mc1,mc2x));
    mxy=_mm_add_ps(mxy,_mm_mul_ps(mc1,mc2y));
    mxz=_mm_add_ps(mxz,_mm_mul_ps(mc1,mc2z));
    mc1=_mm_load_ps(&(c1y[i]));
    myx=_mm_add_ps(myx,_mm_mul_ps(mc1,mc2x));
    myy=_mm_add_ps(myy,_mm_mul_ps(mc1,mc2y));
    myz=_mm_add_ps(myz,_mm_mul_ps(mc1,mc2z));
    mc1=_mm_load_ps(&(c1z[i]));
    mzx=_mm_add_ps(mzx,_mm_mul_ps(mc1,mc2x));
    mzy=_mm_add_ps(mzy,_mm_mul_ps(mc1,mc2y));
    mzz=_mm_add_ps(mzz,_mm_mul_ps(mc1,mc2z));
   }
   //write out the components to the temp variables
   
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
  __m128 mc1=_mm_load_ps(c1);
  __m128 mc2=_mm_load_ps(c2);
  __m128 mt1=_mm_shuffle_ps(mc1,mc1,0x55);//s1x s1x s1x s1x
  mxx=_mm_sub_ps(mxx,_mm_mul_ps(mt1,mc2)); 
  mt1=_mm_shuffle_ps(mc1,mc1,0xAA);//s1y s1y s1y s1y 
  mxy=_mm_sub_ps(mxy,_mm_mul_ps(mt1,mc2));
  mt1=_mm_shuffle_ps(mc1,mc1,0xFF);//s1y s1y s1y s1y   
  mxz=_mm_sub_ps(mxz,_mm_mul_ps(mt1,mc2));         
  //write out the matrix
 _mm_store_ps(r0,mxx); _mm_store_ps(r1,mxy); _mm_store_ps(r2,mxz);
  //calculate the determinant using triple product - do addition when calculating rest of dot products
    __m128 mdet = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(1,3,2,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(2,1,3,0))),
                             _mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(2,1,3,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(1,3,2,0))));//cross_product
    mdet=_mm_mul_ps(mxx,mdet); //sum to get dot product

   //calculate the necessary 6 dot products - do additions in groups of 4 for sse2
   {
    __m128 mt0=_mm_mul_ps(mxx,mxx);
    __m128 mt1=_mm_mul_ps(mxx,mxy);
    __m128 mt2=_mm_mul_ps(mxy,mxy);
    __m128 mt3=_mm_mul_ps(mxx,mxz);

    mxx = _mm_add_ps(_mm_unpacklo_ps(mt0,mt2),_mm_unpackhi_ps(mt0,mt2));
    mt0 = _mm_add_ps(_mm_unpacklo_ps(mt1,mt3),_mm_unpackhi_ps(mt1,mt3));
    _mm_store_ps(rr,_mm_add_ps(_mm_unpacklo_ps(mxx,mt0),_mm_unpackhi_ps(mxx,mt0)));   
   } 
   mxx = _mm_mul_ps(mxz,mxy);
   mxy = _mm_mul_ps(mxz,mxz);

   mxx = _mm_add_ps(_mm_unpacklo_ps(mxx,mdet),_mm_unpackhi_ps(mxx,mdet));
   mxy = _mm_add_ps(_mm_unpacklo_ps(mxy,mdet),_mm_unpackhi_ps(mxy,mdet));
   _mm_store_ps(&(rr[4]),_mm_add_ps(_mm_unpacklo_ps(mxx,mxy),_mm_unpackhi_ps(mxx,mxy)));
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
 double rms=(ssq-d-d)/(double)nat;
 float r[3][3]={{r0[1],r0[2],r0[3]},
                {r1[1],r1[2],r1[3]},
                {r2[1],r2[2],r2[3]}};
 rmatrix((float)d,r,u);
  
 rms=(rms>1e-12)?sqrt(rms) : 0.0f;
 return(rms);
}

#endif

#ifdef SSE3
//sse3
float tmscore_cpu_soa_sse3(int nat,float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3],float *rmsd){
 int nalign=0,best_nalign=0;
 int *ialign=new int[nat];
 int *best_align=new int[nat];
 float max_score=-1,rms;
 int upper_nat8=(nat%8)? (nat/8*8)+8 : nat;
 float *coords_buffer=(float*)memalign(16,6*upper_nat8*sizeof(float));
 float r[16] __attribute__ ((aligned (16)));
 //d0
 float d0=1.24*pow((nat-15),(1.0/3.0))-1.8;
      if(d0< 0.5)d0=0.5;
 //d0_search ----->
 float d,d0_search=d0;
      if(d0_search > 8)d0_search=8;
      if(d0_search <4.5)d0_search=4.5;
 //iterative parameters ----->
 int n_it=20;      //maximum number of iterations
 int n_init_max=6; //maximum number of L_init
 int n_init=0;
 int L_ini_min=4;
 int L_ini[6];

 if(nat < 4) L_ini_min=nat;
 int len=nat;
 int divisor=1;
 while(len > L_ini_min && n_init <5){
  L_ini[n_init++]=len;
  divisor*=2;
  len=nat/divisor;
 }
 L_ini[n_init++]=4;
 if (L_ini[n_init-1] > L_ini_min)L_ini[n_init++]=L_ini_min;;

 // find the maximum score starting from local structures superposition
 float score; //TM-score
 for (int seed=0;seed<n_init;seed++)
 {
  //find the initial rotation matrix using the initial seed residues
  int L_init=L_ini[seed];
  for(int istart=0;istart<=nat-L_init;istart++)
  {
   int nchanges=1;
   int nalign=L_init;
   {
    int m=0;
    int n=0;
    for(int i=0;i<nat;i++){
     if(i>=istart && i<istart+L_init){
      ialign[n++]=i;
     }
    }
   }
   if(rmsd && !seed)
    *rmsd=kabsch_quat_soa_sse3(nalign,ialign,x1,y1,z1,x2,y2,z2,r,coords_buffer);
   else
    kabsch_quat_soa_sse3(nalign,ialign,x1,y1,z1,x2,y2,z2,r,coords_buffer);

   score_fun_soa_sse(nat, d0, d0_search-1,r,x1,y1,z1,x2,y2,z2,ialign,&nalign,&score);
   d=d0_search+1.0f;
   if(score > max_score){
    max_score=score;
    memmove(best_align,ialign,nalign*sizeof(int));
    best_nalign=nalign;
   }
   //extend search from seed
   for (int iter=0;iter<n_it && nchanges;iter++){
    kabsch_quat_soa_sse3(nalign,ialign,x1,y1,z1,x2,y2,z2,r,coords_buffer);
    nchanges=score_fun_soa_sse(nat, d0, d,r,x1,y1,z1,x2,y2,z2,ialign,&nalign,&score);;
    if(score > max_score){
     max_score=score;
     memmove(best_align,ialign,nalign*sizeof(int));
     best_nalign=nalign;
    }
   }
  }
 }
 //calculate matrix - use double precision here
 double R[3][3],t[3];
 double *acoords=new double [best_nalign*6];
 for(int k=0;k<best_nalign;k++){
  int i=best_align[k];
  acoords[6*k]  =x1[i];
  acoords[6*k+1]=y1[i];
  acoords[6*k+2]=z1[i];
  acoords[6*k+3]=x2[i];
  acoords[6*k+4]=y2[i];
  acoords[6*k+5]=z2[i];
 }
 rmsd_svd(best_nalign,acoords,R, t,0);
 if(bR){
  for(int i=0;i<3;i++)
   for(int j=0;j<3;j++)
    bR[i][j]=R[i][j];
 } 
 if(bt){ 
  for(int i=0;i<3;i++)
   bt[i]=t[i];
 } 
 double invd0d0=1.0/(double)(d0*d0);
 double dist;
 float invdnat=1.0/(double)nat;
 double my_score=0;

 for(int k=0;k<nat;k++){
  double u[3];
  double v[3]={x1[k],y1[k],z1[k]};
  double w[3]={x2[k],y2[k],z2[k]};
  u[0]=t[0]+R[0][0]*v[0]+R[1][0]*v[1]+R[2][0]*v[2]-w[0];
  u[1]=t[1]+R[0][1]*v[0]+R[1][1]*v[1]+R[2][1]*v[2]-w[1];
  u[2]=t[2]+R[0][2]*v[0]+R[1][2]*v[1]+R[2][2]*v[2]-w[2];
  dist=u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
  my_score+=1.0/(1.0+dist*invd0d0);
 }
 
 if(coords_buffer)free(coords_buffer);
 delete [] ialign;
 delete [] acoords;
 delete [] best_align;
 return(my_score*invdnat);
}
float coords_sum_ssq_xyz_sse3(int nat, float *x, float *y,float *z,float center[3]){
 int lower_nat=(nat/4)*4;
 float invfnat=1.0/(float)nat;
 int i=0;
 float sums[4] __attribute__ ((aligned (16)));
 { 
  __m128 sumx = _mm_setzero_ps();
  __m128 sumy = _mm_setzero_ps();
  __m128 sumz = _mm_setzero_ps();
  __m128 ssq  = _mm_setzero_ps();
  for(;i<lower_nat;i+=4){
   __m128 p0 = _mm_load_ps(&(x[i]));  
   sumx= _mm_add_ps(sumx,p0);
   ssq= _mm_add_ps(ssq,_mm_mul_ps(p0,p0));
   __m128 p1 = _mm_load_ps(&(y[i])); 
   sumy = _mm_add_ps(sumy,p1);
   ssq = _mm_add_ps(ssq,_mm_mul_ps(p1,p1)); 
   __m128 p2 = _mm_load_ps(&(z[i])); 
   sumz = _mm_add_ps(sumz,p2);
   ssq = _mm_add_ps(ssq,_mm_mul_ps(p2,p2)); 
  } 
  __m128 t1=_mm_hadd_ps (sumx,sumy);
  __m128 t2=_mm_hadd_ps (sumz,ssq);
  __m128 sum0=_mm_hadd_ps(t1,t2);
  _mm_store_ps(sums,sum0);
 }
 for(;i<nat;i++){
  sums[0]+=x[i];
  sums[3]+=x[i]*x[i];
  sums[1]+=y[i];
  sums[3]+=y[i]*y[i];
  sums[2]+=z[i];
  sums[3]+=z[i]*z[i];;
 }
 for(int i=0;i<3;i++)
  center[i]=sums[i]*invfnat;
 return(sums[3]-center[0]*sums[0]-center[1]*sums[1]-center[2]*sums[2]);
} 
float kabsch_quat_soa_sse3(int nat, int *map, float *x1, float *y1,float *z1, float *x2,float *y2, float *z2,float *r,float *coords_buffer){
 int nat4=(nat%4) ? nat/4+1 : nat/4; 
 float center1[3],center2[3];
 float u[3][3];
 float fixed_mem[COORDS_BUFFER_SIZE]__attribute__ ((aligned (16)));
 float *mem;
 int upper_nat8= (nat%8)? (nat/8)*8+8: nat;
 int size= 6*upper_nat8*sizeof(float);
 if(size > COORDS_BUFFER_SIZE)
  mem=coords_buffer;
 else
  mem=fixed_mem;
 memset(mem,0,size);
 float* c1x= mem;
 float* c1y= &mem[nat4*4];
 float* c1z= &mem[nat4*8]; 
 float* c2x= &mem[nat4*12];
 float* c2y= &mem[nat4*16];
 float* c2z= &mem[nat4*20];
 for(int i=0;i<nat;i++){
  int n=map[i];
  c1x[i]=x1[n];
  c1y[i]=y1[n];
  c1z[i]=z1[n];
  c2x[i]=x2[n];
  c2y[i]=y2[n];
  c2z[i]=z2[n];  
 }
 float ssq=coords_sum_ssq_xyz_sse3(nat,c1x,c1y,c1z,center1);
       ssq+=coords_sum_ssq_xyz_sse3(nat,c2x,c2y,c2z,center2);               
 float rms=rmsd_sse3_matrix_xyz(nat,c1x,c1y,c1z,c2x,c2y,c2z,center1,center2,(double)ssq,u);

 float rr[16] __attribute__ ((aligned (16)))=
  {-u[0][0],-u[1][0],-u[2][0],center2[0],
   -u[0][1],-u[1][1],-u[2][1],center2[1],
   -u[0][2],-u[1][2],-u[2][2],center2[2]};
 float w[4]__attribute__ ((aligned (16)));
 float v[4]__attribute__ ((aligned (16)))={center1[0],center1[1],center1[2],1.0f};
 R34v4_sse3(rr,v,w);
 r[0] =u[0][0]; r[1]=u[1][0];r[2] =u[2][0];r[3]=w[0];
 r[4] =u[0][1]; r[5]=u[1][1];r[6] =u[2][1];r[7]=w[1];
 r[8] =u[0][2]; r[9]=u[1][2];r[10]=u[2][2];r[11]=w[2];
 r[12]=0.0f;r[13]=0.0f;r[14]=0.0f;r[15]=1.0f;
 return(rms);
}
float rmsd_sse3_matrix_xyz (int nat,float *c1x,float *c1y, float *c1z,float *c2x,float *c2y, float *c2z, float center1[3], float center2[3],double ssq,float u[3][3]){
 //c1 has (0,sum(x1),sum(y1),sum(z1));
 //c2 has (0,mean(x1),mean(y2),mean(z2));
 float fnat=(float)nat;
 float c1[4] __attribute__ ((aligned (16))) ={0.0f,center1[0]*fnat,center1[1]*fnat,center1[2]*fnat};
 float c2[4] __attribute__ ((aligned (16))) ={0.0f,center2[0],center2[1],center2[2]};  
 float r0[4] __attribute__ ((aligned (16)));  
 float r1[4] __attribute__ ((aligned (16)));  
 float r2[4] __attribute__ ((aligned (16))); 
 float rr[8] __attribute__ ((aligned (16)));
 int nat4=(nat%4)? nat/4+1 : nat/4;
 int padded_nat=nat4*4;
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
   for(i=0;i<padded_nat;i+=4){
   //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
    __m128 mc2x=_mm_load_ps(&(c2x[i]));
    __m128 mc2y=_mm_load_ps(&(c2y[i]));
    __m128 mc2z=_mm_load_ps(&(c2z[i]));
    __m128 mc1 =_mm_load_ps(&(c1x[i]));
    
   //generate the 4 products that are saved in mr0
    mxx=_mm_add_ps(mxx,_mm_mul_ps(mc1,mc2x));
    mxy=_mm_add_ps(mxy,_mm_mul_ps(mc1,mc2y));
    mxz=_mm_add_ps(mxz,_mm_mul_ps(mc1,mc2z));
    mc1=_mm_load_ps(&(c1y[i]));
    myx=_mm_add_ps(myx,_mm_mul_ps(mc1,mc2x));
    myy=_mm_add_ps(myy,_mm_mul_ps(mc1,mc2y));
    myz=_mm_add_ps(myz,_mm_mul_ps(mc1,mc2z));
    mc1=_mm_load_ps(&(c1z[i]));
    mzx=_mm_add_ps(mzx,_mm_mul_ps(mc1,mc2x));
    mzy=_mm_add_ps(mzy,_mm_mul_ps(mc1,mc2y));
    mzz=_mm_add_ps(mzz,_mm_mul_ps(mc1,mc2z));
   }
   //write out the components to the temp variables
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

  __m128 mc1=_mm_load_ps(c1);
  __m128 mc2=_mm_load_ps(c2);
  __m128 mt1=_mm_shuffle_ps(mc1,mc1,0x55);//s1x s1x s1x s1x
  mxx=_mm_sub_ps(mxx,_mm_mul_ps(mt1,mc2)); 
  mt1=_mm_shuffle_ps(mc1,mc1,0xAA);//s1y s1y s1y s1y 
  mxy=_mm_sub_ps(mxy,_mm_mul_ps(mt1,mc2));
  mt1=_mm_shuffle_ps(mc1,mc1,0xFF);//s1y s1y s1y s1y   
  mxz=_mm_sub_ps(mxz,_mm_mul_ps(mt1,mc2));         
  //write out the matrix
 _mm_store_ps(r0,mxx); _mm_store_ps(r1,mxy); _mm_store_ps(r2,mxz);
  //calculate the determinant using triple product - do addition when calculating rest of dot products
    __m128 mdet = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(1,3,2,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(2,1,3,0))),
                             _mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(2,1,3,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(1,3,2,0))));//cross_product
    mdet=_mm_mul_ps(mxx,mdet); //sum to get dot product

   //calculate the necessary 6 dot products - do additions in groups of 4 for sse2
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
 double rms=(ssq-d-d)/(double)nat;
 float r[3][3]={{r0[1],r0[2],r0[3]},
                {r1[1],r1[2],r1[3]},
                {r2[1],r2[2],r2[3]}};
 rmatrix((float)d,r,u);
  
 rms=(rms>1e-8)?sqrt(rms) : 0.0f;
 return(rms);
}
void R34v4_sse3 (float r[12],float *x,float *Rx){
 __m128 r0 = _mm_load_ps(r);
 __m128 r1 = _mm_load_ps(&(r[4]));
 __m128 r2 = _mm_load_ps(&(r[8])); 
 __m128 one= _mm_set_ss(1.0f); 
 __m128 mu = _mm_load_ps(x);
 __m128 m0 = _mm_mul_ps(r0,mu);
 __m128 m1 = _mm_mul_ps(r1,mu);
 __m128 m2 = _mm_mul_ps(r2,mu);
 __m128 s1 = _mm_hadd_ps(m0,m1); 
 __m128 s2 = _mm_hadd_ps(m2,one);
 _mm_store_ps(Rx,_mm_hadd_ps(s1,s2));
}
void R34v4(float *r,float *x, float *Rx){
 float x0=x[0],x1=x[1],x2=x[2];	
 Rx[0]=r[0]*x0+r[1]*x1+r[2]*x2+r[3];
 Rx[1]=r[4]*x0+r[5]*x1+r[6]*x2+r[7];
 Rx[2]=r[8]*x0+r[9]*x1+r[10]*x2+r[11];
}	

#endif

#ifdef AVX
//avx
void print_m256(__m256 p){
 float t[8];
 _mm256_storeu_ps(t,p);
 for (int i=0;i<8;i++)
  fprintf(stderr,"%10.3f ",t[i]);
 fprintf(stderr,"\n");
}
int shuffle_coords8_avx (int nstructs,int pdb_size, float *coords, float *shuffled_coords,float *ssqs){
 //just rearrange and pad with zeros to 8;
 int natoms=pdb_size/3;
 int nat8=(natoms%8)? natoms/8+1 : natoms/8;
 int pdb8_size=nat8*24;
 float centroid[3];
 float* aligned_coords = (float*) memalign(32,pdb8_size*sizeof(float));
 memset (aligned_coords,0,sizeof(float)*pdb8_size);
 memset(shuffled_coords,0,sizeof(float)*pdb8_size*nstructs);
 for (int p=0;p<nstructs;p++){
  float* const new_coords=&(shuffled_coords[p*pdb8_size]);
  memmove(aligned_coords,&(coords[p*pdb_size]),pdb_size*sizeof(float));
  ssqs[p]=shuffle_center_coords8_avx(natoms,aligned_coords,new_coords,centroid);
 }
 if(aligned_coords)free(aligned_coords);
 return(pdb8_size*nstructs);
}
int shuffle_coords8_avx (int nstructs,int pdb_size, float *coords, float *shuffled_coords,float *ssqs,float *centroids){
 //just rearrange and pad with zeros to 8;
 int natoms=pdb_size/3;
 int nat8=(natoms%8)? natoms/8+1 : natoms/8;
 int pdb8_size=nat8*24;
 float* aligned_coords = (float*) memalign(32,pdb8_size*sizeof(float));
 memset (aligned_coords,0,sizeof(float)*pdb8_size);
 memset(shuffled_coords,0,sizeof(float)*pdb8_size*nstructs);
 for (int p=0;p<nstructs;p++){
  float* const new_coords=&(shuffled_coords[p*pdb8_size]);
  memmove(aligned_coords,&(coords[p*pdb_size]),pdb_size*sizeof(float));
  ssqs[p]=shuffle_center_coords8_avx(natoms,aligned_coords,new_coords,&(centroids[p*3]));
 }
 if(aligned_coords)free(aligned_coords);
 return(pdb8_size*nstructs);
}
float shuffle_center_coords8_avx(int nat, float *coords, float *shuffled_coords,float centroid[3]){ //returns ssq - does not use aligned coords - but outputs aligned coords
 float invnat=1.0f/(float)nat;
 float test[8];
 float sums[4] __attribute__ ((aligned (16)));
 float ssq __attribute__ ((aligned (16)));
 int lower_nat4=(nat/4)*4;
 int nat8=nat/8;
 int padded_nat=(nat%8)? nat8*8+8: nat;
 memset(shuffled_coords,0,padded_nat*sizeof(float)*3);
 int i=0;
 //first pass reads and calculates sums
//bug with OpenMP and aligned 32 coords 
//  __m128 *m = (__m128*) coords;
 {
   __m128 msum,mssq;
  {
   __m256 sum0 = _mm256_setzero_ps();
   __m256 sum1 = _mm256_setzero_ps();
   __m256 sum2 = _mm256_setzero_ps();
   __m256 ssq0 = _mm256_setzero_ps();
   for(;i<nat8*6;i+=6){
    __m256 m03;
    __m256 m14; 
    __m256 m25;
    
    { 
     int p=i*4;
     __m256 m01=_mm256_loadu_ps(&(coords[p])); 
     __m256 m23=_mm256_loadu_ps(&(coords[p+8]));
     __m256 m45=_mm256_loadu_ps(&(coords[p+16])); 
     m03= _mm256_permute2f128_ps(m01,m23,0x30);//lower first upper second 0011  0000     
     m14= _mm256_permute2f128_ps(m01,m45,0x21);//upper first lower second 0010  0001
     m25= _mm256_permute2f128_ps(m23,m45,0x30);//lower first upper second
    } 
    sum0 = _mm256_add_ps(sum0,m03);
    ssq0 = _mm256_add_ps(ssq0,_mm256_mul_ps(m03,m03));
    sum1 = _mm256_add_ps(sum1,m14);
    ssq0 = _mm256_add_ps(ssq0,_mm256_mul_ps(m14,m14));
    sum2 = _mm256_add_ps(sum2,m25);
    ssq0 = _mm256_add_ps(ssq0,_mm256_mul_ps(m25,m25));
   }
   __m256 t = _mm256_shuffle_ps(sum0, sum1, _MM_SHUFFLE(0, 1, 0, 3));

   sum0=_mm256_add_ps(sum0,_mm256_shuffle_ps(t, t, _MM_SHUFFLE(0, 2, 3, 0)));
   sum0=_mm256_add_ps(sum0,_mm256_shuffle_ps(sum1, sum2, _MM_SHUFFLE(0, 0, 3, 2)));  
   sum0=_mm256_add_ps(sum0,_mm256_shuffle_ps(sum2, sum2, _MM_SHUFFLE(0, 3, 2, 1)));
   sum0=_mm256_add_ps(sum0,_mm256_permute2f128_ps(sum0,sum0,0x01));//add two halves together a
   ssq0=_mm256_add_ps(ssq0,_mm256_permute2f128_ps(ssq0,ssq0,0x01));//add two halves together
   _mm256_zeroupper();
   msum= _mm256_castps256_ps128(sum0); 
   mssq= _mm256_castps256_ps128(ssq0); 

  }//end read 8
  if(nat%8)
  {
   __m128 sum0 = _mm_setzero_ps();
   __m128 sum1 = _mm_setzero_ps();
   __m128 sum2 = _mm_setzero_ps();
   i=i*4;
   for(;i<lower_nat4*3;i+=12){
    __m128 p0 = _mm_loadu_ps(&(coords[i]));   // x0y0z0x1
    sum0= _mm_add_ps(sum0,p0);
    mssq= _mm_add_ps(mssq,_mm_mul_ps(p0,p0));
    __m128 p1 = _mm_loadu_ps(&(coords[i+4])); // y1z1x2y2
    sum1 = _mm_add_ps(sum1,p1);
    mssq = _mm_add_ps(mssq,_mm_mul_ps(p1,p1)); 
    __m128 p2 = _mm_loadu_ps(&(coords[i+8])); // z2x3y3z3
    sum2= _mm_add_ps(sum2,p2);
    mssq= _mm_add_ps(mssq,_mm_mul_ps(p2,p2));
   } 
   __m128 t = _mm_shuffle_ps(sum0, sum1, _MM_SHUFFLE(0, 1, 0, 3));
   sum0=_mm_add_ps(sum0,_mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 2, 3, 0)));
   sum0=_mm_add_ps(sum0,_mm_shuffle_ps(sum1, sum2, _MM_SHUFFLE(0, 0, 3, 2)));  
   sum0=_mm_add_ps(sum0,_mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(0, 3, 2, 1)));
   msum=_mm_add_ps(sum0,msum);

  }
  mssq=_mm_hadd_ps(mssq,mssq);
  mssq=_mm_hadd_ps(mssq,mssq);
  _mm_store_ps(sums,msum); 
  _mm_store_ss(&ssq,mssq);
 }
 //end read 4
 //finish sums in scalar
 //subtract and write out
 if(nat%4){
  int j=i/3;
  int m=0;
  for(;i<nat*3;i+=3){
   sums[0]+=coords[i];
   ssq+=coords[i]*coords[i];
   sums[1]+=coords[i+1];
   ssq+=coords[i+1]*coords[i+1];
   sums[2]+=coords[i+2];
   ssq+=coords[i+2]*coords[i+2];
  }
 }
 float invfnat=1.0/(float)nat;
 ssq-=sums[0]*sums[0]*invfnat+sums[1]*sums[1]*invfnat+sums[2]*sums[2]*invfnat;//modify ssq to reflect the centered coords
 sums[0]*=invfnat;sums[1]*=invfnat;sums[2]*=invfnat;
 //fprintf(stderr,"%f %f %f %f\n",sums[0],sums[1],sums[2],ssq);
 //subtract from coords 
 i=0;
 int k=0;
 __m256 s0=_mm256_broadcast_ss(&(sums[0]));
 __m256 s1=_mm256_broadcast_ss(&(sums[1]));
 __m256 s2=_mm256_broadcast_ss(&(sums[2]));  
 for(;i<nat8*6;i+=6){
   __m256 m03;
   __m256 m14; 
   __m256 m25; 
    { 
     int p=i*4;
     __m256 m01=_mm256_loadu_ps(&(coords[p])); 
     __m256 m23=_mm256_loadu_ps(&(coords[p+8]));
     __m256 m45=_mm256_loadu_ps(&(coords[p+16])); 
     m03= _mm256_permute2f128_ps(m01,m23,0x30);//lower first upper second 0011  0000     
     m14= _mm256_permute2f128_ps(m01,m45,0x21);//upper first lower second 0010  0001
     m25= _mm256_permute2f128_ps(m23,m45,0x30);//lower first upper second
    } 
  __m256 mxy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE( 2,1,3,2)); // upper x's and y's 
  __m256 myz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE( 1,0,2,1)); // lower y's and z's
  _mm256_store_ps (&(shuffled_coords[k]),_mm256_sub_ps(_mm256_shuffle_ps(m03, mxy, _MM_SHUFFLE( 2,0,3,0)),s0)); 
  _mm256_store_ps (&(shuffled_coords[k+8]),_mm256_sub_ps(_mm256_shuffle_ps(myz ,mxy, _MM_SHUFFLE( 3,1,2,0)),s1)); 
  _mm256_store_ps (&(shuffled_coords[k+16]),_mm256_sub_ps(_mm256_shuffle_ps(myz ,m25, _MM_SHUFFLE( 3,0,3,1)),s2));
  k+=24;
 }
 i=4*i;
 _mm256_zeroupper();
 for(;i<lower_nat4*3;i+=12){
  __m128 x0y0z0x1 = _mm_load_ps(&(coords[i]));
  __m128 y1z1x2y2 = _mm_load_ps(&(coords[i+4]));  
  __m128 z2x3y3z3 = _mm_load_ps(&(coords[i+8]));
  __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2,z2x3y3z3,_MM_SHUFFLE( 2,1,3,2));  
  __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1,y1z1x2y2,_MM_SHUFFLE( 1,0,2,1)); 
  _mm_store_ps (&(shuffled_coords[k]),_mm_sub_ps(_mm_shuffle_ps(x0y0z0x1,x2y2x3y3,_MM_SHUFFLE( 2,0,3,0)), _mm256_castps256_ps128(s0)));
  _mm_store_ps (&(shuffled_coords[k+8]),_mm_sub_ps(_mm_shuffle_ps(y0z0y1z1,x2y2x3y3,_MM_SHUFFLE( 3,1,2,0)),_mm256_castps256_ps128(s1)));
  _mm_store_ps (&(shuffled_coords[k+16]),_mm_sub_ps(_mm_shuffle_ps(y0z0y1z1,z2x3y3z3,_MM_SHUFFLE( 3,0,3,1)),_mm256_castps256_ps128(s2)));
  k+=4;
 }
 for(;i<nat*3;i+=3){
  shuffled_coords[k] =coords[i]-sums[0];
  shuffled_coords[k+8] =coords[i+1]-sums[1];
  shuffled_coords[k+16] =coords[i+2]-sums[2];
  k++;
 }

 __m256 *mshuf=(__m256*)shuffled_coords;
 if(centroid){
  centroid[0]=sums[0];
  centroid[1]=sums[1];
  centroid[2]=sums[2];
 }
 return(ssq);
}

float shuffle_center_coords8_unaligned_avx(int nat, float *coords, float *shuffled_coords,float centroid[3]){ //returns ssq - does not use aligned coords - but outputs aligned coords
 float invnat=1.0f/(float)nat;
 float sums[4] __attribute__ ((aligned (16)));
 float ssq __attribute__ ((aligned (16)));
 int lower_nat4=(nat/4)*4;
 int nat8=nat/8;
 int padded_nat=(nat%8)? nat8*8+8: nat;
 memset(shuffled_coords,0,padded_nat*sizeof(float)*3);
 int i=0;
 //first pass reads and calculates sums
 
//  __m128 *m = (__m128*) coords;
 {
   __m128 msum,mssq;
  {
   __m256 sum0 = _mm256_setzero_ps();
   __m256 sum1 = _mm256_setzero_ps();
   __m256 sum2 = _mm256_setzero_ps();
   __m256 ssq0 = _mm256_setzero_ps();
   for(;i<nat8*6;i+=6){
    __m256 m03;
    __m256 m14; 
    __m256 m25;
    
    { 
     int p=i*4;
     __m256 m01=_mm256_loadu_ps(&(coords[p])); 
     __m256 m23=_mm256_loadu_ps(&(coords[p+8]));
     __m256 m45=_mm256_loadu_ps(&(coords[p+16])); 
     m03= _mm256_permute2f128_ps(m01,m23,0x30);//lower first upper second 0011  0000     
     m14= _mm256_permute2f128_ps(m01,m45,0x21);//upper first lower second 0010  0001
     m25= _mm256_permute2f128_ps(m23,m45,0x30);//lower first upper second
    }  
       
    sum0 = _mm256_add_ps(sum0,m03);
    ssq0 = _mm256_add_ps(ssq0,_mm256_mul_ps(m03,m03));
    sum1 = _mm256_add_ps(sum1,m14);
    ssq0 = _mm256_add_ps(ssq0,_mm256_mul_ps(m14,m14));
    sum2 = _mm256_add_ps(sum2,m25);
    ssq0 = _mm256_add_ps(ssq0,_mm256_mul_ps(m25,m25));
   }
   __m256 t = _mm256_shuffle_ps(sum0, sum1, _MM_SHUFFLE(0, 1, 0, 3));

   sum0=_mm256_add_ps(sum0,_mm256_shuffle_ps(t, t, _MM_SHUFFLE(0, 2, 3, 0)));
   sum0=_mm256_add_ps(sum0,_mm256_shuffle_ps(sum1, sum2, _MM_SHUFFLE(0, 0, 3, 2)));  
   sum0=_mm256_add_ps(sum0,_mm256_shuffle_ps(sum2, sum2, _MM_SHUFFLE(0, 3, 2, 1)));
   sum0=_mm256_add_ps(sum0,_mm256_permute2f128_ps(sum0,sum0,0x01));//add two halves together a
   ssq0=_mm256_add_ps(ssq0,_mm256_permute2f128_ps(ssq0,ssq0,0x01));//add two halves together
   _mm256_zeroupper();
   msum= _mm256_castps256_ps128(sum0); 
   mssq= _mm256_castps256_ps128(ssq0); 

  }//end read 8
  if(nat%8)
  {
   __m128 sum0 = _mm_setzero_ps();
   __m128 sum1 = _mm_setzero_ps();
   __m128 sum2 = _mm_setzero_ps();
   i=i*4;
   for(;i<lower_nat4*3;i+=12){
    __m128 p0 = _mm_loadu_ps(&(coords[i]));   // x0y0z0x1
    sum0= _mm_add_ps(sum0,p0);
    mssq= _mm_add_ps(mssq,_mm_mul_ps(p0,p0));
    __m128 p1 = _mm_loadu_ps(&(coords[i+4])); // y1z1x2y2
    sum1 = _mm_add_ps(sum1,p1);
    mssq = _mm_add_ps(mssq,_mm_mul_ps(p1,p1)); 
    __m128 p2 = _mm_loadu_ps(&(coords[i+8])); // z2x3y3z3
    sum2= _mm_add_ps(sum2,p2);
    mssq= _mm_add_ps(mssq,_mm_mul_ps(p2,p2));
   } 
   __m128 t = _mm_shuffle_ps(sum0, sum1, _MM_SHUFFLE(0, 1, 0, 3));
   sum0=_mm_add_ps(sum0,_mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 2, 3, 0)));
   sum0=_mm_add_ps(sum0,_mm_shuffle_ps(sum1, sum2, _MM_SHUFFLE(0, 0, 3, 2)));  
   sum0=_mm_add_ps(sum0,_mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(0, 3, 2, 1)));
   msum=_mm_add_ps(sum0,msum);

  }
  mssq=_mm_hadd_ps(mssq,mssq);
  mssq=_mm_hadd_ps(mssq,mssq);
  _mm_store_ps(sums,msum); 
  _mm_store_ss(&ssq,mssq);
 }
 //end read 4
 //finish sums in scalar
 //subtract and write out
 if(nat%4){
  int j=i/3;
  int m=0;
  for(;i<nat*3;i+=3){
   sums[0]+=coords[i];
   ssq+=coords[i]*coords[i];
   sums[1]+=coords[i+1];
   ssq+=coords[i+1]*coords[i+1];
   sums[2]+=coords[i+2];
   ssq+=coords[i+2]*coords[i+2];
  }
 }
 float invfnat=1.0/(float)nat;
 ssq-=sums[0]*sums[0]*invfnat+sums[1]*sums[1]*invfnat+sums[2]*sums[2]*invfnat;//modify ssq to reflect the centered coords
 sums[0]*=invfnat;sums[1]*=invfnat;sums[2]*=invfnat;
 //fprintf(stderr,"%f %f %f %f\n",sums[0],sums[1],sums[2],ssq);
 //subtract from coords 
 i=0;
 int k=0;
 __m256 s0=_mm256_broadcast_ss(&(sums[0]));
 __m256 s1=_mm256_broadcast_ss(&(sums[1]));
 __m256 s2=_mm256_broadcast_ss(&(sums[2]));  
 for(;i<nat8*6;i+=6){
   __m256 m03;
   __m256 m14; 
   __m256 m25; 
    { 
     int p=i*4;
     __m256 m01=_mm256_loadu_ps(&(coords[p])); 
     __m256 m23=_mm256_loadu_ps(&(coords[p+8]));
     __m256 m45=_mm256_loadu_ps(&(coords[p+16])); 
     m03= _mm256_permute2f128_ps(m01,m23,0x30);//lower first upper second 0011  0000     
     m14= _mm256_permute2f128_ps(m01,m45,0x21);//upper first lower second 0010  0001
     m25= _mm256_permute2f128_ps(m23,m45,0x30);//lower first upper second
    } 
  __m256 mxy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE( 2,1,3,2)); // upper x's and y's 
  __m256 myz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE( 1,0,2,1)); // lower y's and z's
  _mm256_store_ps (&(shuffled_coords[k]),_mm256_sub_ps(_mm256_shuffle_ps(m03, mxy, _MM_SHUFFLE( 2,0,3,0)),s0)); 
  _mm256_store_ps (&(shuffled_coords[k+8]),_mm256_sub_ps(_mm256_shuffle_ps(myz ,mxy, _MM_SHUFFLE( 3,1,2,0)),s1)); 
  _mm256_store_ps (&(shuffled_coords[k+16]),_mm256_sub_ps(_mm256_shuffle_ps(myz ,m25, _MM_SHUFFLE( 3,0,3,1)),s2));
  k+=24;
 }
 i=4*i;
 _mm256_zeroupper();
 for(;i<lower_nat4*3;i+=12){
  __m128 x0y0z0x1 = _mm_loadu_ps(&(coords[i]));
  __m128 y1z1x2y2 = _mm_loadu_ps(&(coords[i+4]));  
  __m128 z2x3y3z3 = _mm_loadu_ps(&(coords[i+8]));
  __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2,z2x3y3z3,_MM_SHUFFLE( 2,1,3,2));  
  __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1,y1z1x2y2,_MM_SHUFFLE( 1,0,2,1)); 
  _mm_store_ps (&(shuffled_coords[k]),_mm_sub_ps(_mm_shuffle_ps(x0y0z0x1,x2y2x3y3,_MM_SHUFFLE( 2,0,3,0)), _mm256_castps256_ps128(s0)));
  _mm_store_ps (&(shuffled_coords[k+8]),_mm_sub_ps(_mm_shuffle_ps(y0z0y1z1,x2y2x3y3,_MM_SHUFFLE( 3,1,2,0)),_mm256_castps256_ps128(s1)));
  _mm_store_ps (&(shuffled_coords[k+16]),_mm_sub_ps(_mm_shuffle_ps(y0z0y1z1,z2x3y3z3,_MM_SHUFFLE( 3,0,3,1)),_mm256_castps256_ps128(s2)));
  k+=4;
 }
 for(;i<nat*3;i+=3){
  shuffled_coords[k] =coords[i]-sums[0];
  shuffled_coords[k+8] =coords[i+1]-sums[1];
  shuffled_coords[k+16] =coords[i+2]-sums[2];
  k++;
 }

 __m256 *mshuf=(__m256*)shuffled_coords;
 if(centroid){
  centroid[0]=sums[0];
  centroid[1]=sums[1];
  centroid[2]=sums[2];
 }
 return(ssq);
}
int score_fun_soa_avx(int nat, float d0, float d, float *r,float *x1, float *y1, float *z1, float *x2, float *y2, float *z2, int *ialign,int *nalign,float *tm_score){
 //ialign points to atom number
 int k,ncut=0,nchange=0,my_nalign=*nalign,upper_nat= (nat%8)?(nat/8)*8+8 : nat;
 float d2=d*d,invfnat=1.0f/(float)nat;
 float invd0d0=1.0f/(d0*d0);

 
 float* dist = (float*) memalign(32,upper_nat*sizeof(float));
 //keep nmin smallest distances && distances < dtmp
 float my_score=LG_score_soa_avx(r,nat,x1,y1,z1,x2,y2,z2,dist,invd0d0);
 for(int k=0;k<nat;k++){
  if(dist[k]<d2) ncut++;  
 }

 //adjust d until there are at least 3 the same - rare use another routine for this
 while(ncut <3){
  d2=d*d;
  ncut=0;
  for(k=0;k<nat;k++)
   if(dist[k]<d2) ncut++;
  d+=0.5f;
 }
 ncut=0;
 for(k=0;k<nat;k++){
  if(dist[k]<d2){  
   if(ncut < my_nalign && ialign[ncut] == k)ncut++;
   else{
    nchange=1;
    ialign[ncut++]=k;
   }
  }
 }
 if(dist) free(dist);
 
 *tm_score=my_score*invfnat;
 if(!nchange)return(0);
 *nalign=ncut;
 return(1);
}
void shuffle_tmscore_coords_soa_avx(int nstructs, int nat, float *coords,float **x,float **y,float **z){
 int upper_nat8=(nat%8)? (nat/8+1)*8 : nat;
 int pdb8_size=3*upper_nat8;
 *x=(float*) memalign(32,sizeof(float)*upper_nat8*nstructs);
 *y=(float*) memalign(32,sizeof(float)*upper_nat8*nstructs);
 *z=(float*) memalign(32,sizeof(float)*upper_nat8*nstructs);
 float* aligned_coords = (float*) memalign(32,pdb8_size*sizeof(float));
 memset (aligned_coords,0,sizeof(float)*pdb8_size);
 if(upper_nat8 > nat){
  memset(*x,0,sizeof(float)*upper_nat8*nstructs);
  memset(*y,0,sizeof(float)*upper_nat8*nstructs);
  memset(*z,0,sizeof(float)*upper_nat8*nstructs);
 }

 for(int p=0;p<nstructs;p++){ 
  float *const c=&(coords[p*nat*3]);
  memmove(aligned_coords,&(coords[p*nat*3]),nat*3*sizeof(float));
  float *const mx=&((*x)[p*upper_nat8]);
  float *const my=&((*y)[p*upper_nat8]);
  float *const mz=&((*z)[p*upper_nat8]);
  split_coords_avx(nat,aligned_coords,mx,my,mz); 
 }
 if(aligned_coords)free (aligned_coords); 
}
void split_coords_avx(int nat, float *coords, float *x, float *y,float *z){ //needs aligned coords
 int i=0,k=0;
 int nat8=nat/8;
 //from INTEL 
 __m128 *m = (__m128*) coords; //this actually requires coords to be aligned - at least for Bulldozers
 for(;i<nat8*6;i+=6){
   __m256 m03;
   __m256 m14; 
   __m256 m25; 
   m03  = _mm256_castps128_ps256(m[i]); // load lower halves
   m14  = _mm256_castps128_ps256(m[i+1]);
   m25  = _mm256_castps128_ps256(m[i+2]);
   m03  = _mm256_insertf128_ps(m03 ,m[i+3],1);  // load upper halves
   m14  = _mm256_insertf128_ps(m14 ,m[i+4],1);
   m25  = _mm256_insertf128_ps(m25 ,m[i+5],1);
  __m256 mxy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE( 2,1,3,2)); // upper x's and y's 
  __m256 myz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE( 1,0,2,1)); // lower y's and z's
  _mm256_store_ps (&(x[k]),_mm256_shuffle_ps(m03, mxy, _MM_SHUFFLE( 2,0,3,0))); 
  _mm256_store_ps (&(y[k]),_mm256_shuffle_ps(myz ,mxy, _MM_SHUFFLE( 3,1,2,0))); 
  _mm256_store_ps (&(z[k]),_mm256_shuffle_ps(myz ,m25, _MM_SHUFFLE( 3,0,3,1)));
  k+=8;
 }

 //do the last set in scalar
 for(i=nat8*8;i<nat;i++){
  int p=i*3;
  x[k] =coords[p];
  y[k] =coords[p+1];
  z[k] =coords[p+2];
  k++;;
 }
}
float tmscore_cpu_soa_avx(int nat,float *x1, float *y1,float *z1,float *x2,float *y2,float *z2,float bR[3][3], float bt[3],float *rmsd){
 float r[16] __attribute__ ((aligned (32)));
 int upper_nat8=(nat%8)?(nat/8)*8+8: nat;
 float *coords_buffer=(float*)memalign(32,6*upper_nat8*sizeof(float));
 int nalign=0,best_nalign=0;
 int *ialign=new int[nat];
 int *best_align=new int[nat];
 float max_score=-1,rms;
 //d0
 float d0=1.24*pow((nat-15),(1.0/3.0))-1.8;
      if(d0< 0.5)d0=0.5;
 //d0_search ----->
 float d,d0_search=d0;
      if(d0_search > 8)d0_search=8;
      if(d0_search <4.5)d0_search=4.5;
 //iterative parameters ----->
 int n_it=20;      //maximum number of iterations
 int n_init_max=6; //maximum number of L_init
 int n_init=0;
 int L_ini_min=4;
 int L_ini[6];

 if(nat < 4) L_ini_min=nat;
 int len=nat;
 int divisor=1;
 while(len > L_ini_min && n_init <5){
  L_ini[n_init++]=len;
  divisor*=2;
  len=nat/divisor;
 }
 L_ini[n_init++]=4;
 if (L_ini[n_init-1] > L_ini_min)L_ini[n_init++]=L_ini_min;;

 // find the maximum score starting from local structures superposition
 float score; //TM-score
 for (int seed=0;seed<n_init;seed++)
 {
  //find the initial rotation matrix using the initial seed residues
  int L_init=L_ini[seed];
  for(int istart=0;istart<=nat-L_init;istart++)
  {
   int nchanges=1;
   int nalign=L_init;
   {
    int m=0;
    int n=0;
    for(int i=0;i<nat;i++){
     if(i>=istart && i<istart+L_init){
      ialign[n++]=i;
     }
    }
   }
   if(rmsd && !seed)
    *rmsd=kabsch_quat_soa_avx(nalign,ialign,x1,y1,z1,x2,y2,z2,r,coords_buffer);
   else
    kabsch_quat_soa_avx(nalign,ialign,x1,y1,z1,x2,y2,z2,r,coords_buffer);
   score_fun_soa_avx(nat, d0, d0_search-1,r,x1,y1,z1,x2,y2,z2,ialign,&nalign,&score);
   d=d0_search+1.0f;
   if(score > max_score){
    max_score=score;
    memmove(best_align,ialign,nalign*sizeof(int));
    best_nalign=nalign;
   }
   //extend search from seed
   for (int iter=0;iter<n_it && nchanges;iter++){
    kabsch_quat_soa_avx(nalign,ialign,x1,y1,z1,x2,y2,z2,r,coords_buffer);
    nchanges=score_fun_soa_avx(nat, d0, d,r,x1,y1,z1,x2,y2,z2,ialign,&nalign,&score);
    if(score > max_score){
     max_score=score;
     memmove(best_align,ialign,nalign*sizeof(int));
     best_nalign=nalign;
    }
   }
  }
 }
 //calculate matrix - use double precision here
 double R[3][3],t[3];
 double *acoords=new double [best_nalign*6];
 for(int k=0;k<best_nalign;k++){
  int i=best_align[k];
  acoords[6*k]  =x1[i];
  acoords[6*k+1]=y1[i];
  acoords[6*k+2]=z1[i];
  acoords[6*k+3]=x2[i];
  acoords[6*k+4]=y2[i];
  acoords[6*k+5]=z2[i];
 }
 rmsd_svd(best_nalign,acoords,R,t,0);
 if(bR){
  for(int i=0;i<3;i++)
   for(int j=0;j<3;j++)
    bR[i][j]=R[i][j];
 } 
 if(bt){ 
  for(int i=0;i<3;i++)
   bt[i]=t[i];
 } 
 double invd0d0=1.0/(double)(d0*d0);
 double dist;
 float invdnat=1.0/(double)nat;
 double my_score=0;


 for(int k=0;k<nat;k++){
  double u[3];
  double v[3]={x1[k],y1[k],z1[k]};
  double w[3]={x2[k],y2[k],z2[k]};
  u[0]=t[0]+R[0][0]*v[0]+R[1][0]*v[1]+R[2][0]*v[2]-w[0];
  u[1]=t[1]+R[0][1]*v[0]+R[1][1]*v[1]+R[2][1]*v[2]-w[1];
  u[2]=t[2]+R[0][2]*v[0]+R[1][2]*v[1]+R[2][2]*v[2]-w[2];
  dist=u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
  my_score+=1.0/(1.0+dist*invd0d0);
 }

 if(coords_buffer)free(coords_buffer);
 delete [] best_align;
 delete [] ialign;
 delete [] acoords;
 return(my_score*invdnat);
}

float coords_sum_ssq_avx(int nat, float *x, float *y,float *z,float center[3]){
 int lower_nat4=(nat/4)*4,lower_nat8=(nat/8)*8;
 float invfnat=1.0/(float)nat;
 int i=0;
 float sums[4] __attribute__ ((aligned (16)));
 {
  __m128 msumx,msumy,msumz,mssq;
  {
   __m256 sumx = _mm256_setzero_ps();
   __m256 sumy = _mm256_setzero_ps();
   __m256 sumz = _mm256_setzero_ps();
   __m256 ssq  = _mm256_setzero_ps();
   for(;i<lower_nat8;i+=8){
    __m256 p0 = _mm256_load_ps(&(x[i]));  
    sumx= _mm256_add_ps(sumx,p0);
    ssq= _mm256_add_ps(ssq,_mm256_mul_ps(p0,p0));
    __m256 p1 = _mm256_load_ps(&(y[i])); 
    sumy = _mm256_add_ps(sumy,p1);
    ssq = _mm256_add_ps(ssq,_mm256_mul_ps(p1,p1)); 
    __m256 p2 = _mm256_load_ps(&(z[i])); 
    sumz = _mm256_add_ps(sumz,p2);
    ssq = _mm256_add_ps(ssq,_mm256_mul_ps(p2,p2)); 
   }
   msumx= _mm256_castps256_ps128(_mm256_permute2f128_ps(sumx,sumx,0x01)); //add two halves together
   msumy= _mm256_castps256_ps128(_mm256_permute2f128_ps(sumy,sumy,0x01)); //add two halves together
   msumz= _mm256_castps256_ps128(_mm256_permute2f128_ps(sumz,sumz,0x01)); //add two halves together
   mssq = _mm256_castps256_ps128(_mm256_permute2f128_ps(ssq,ssq,0x01)); //add two halves together
   _mm256_zeroupper();
   msumx= _mm_add_ps(msumx,_mm256_castps256_ps128(sumx));
   msumy= _mm_add_ps(msumy,_mm256_castps256_ps128(sumy));
   msumz= _mm_add_ps(msumz,_mm256_castps256_ps128(sumz));
   mssq=  _mm_add_ps(mssq,_mm256_castps256_ps128(ssq));
  } //end read 8
  for(;i<lower_nat4;i+=4){
   __m128 p0 = _mm_load_ps(&(x[i]));  
   msumx= _mm_add_ps(msumx,p0);
   mssq= _mm_add_ps(mssq,_mm_mul_ps(p0,p0));
   __m128 p1 = _mm_load_ps(&(y[i])); 
   msumy = _mm_add_ps(msumy,p1);
   mssq = _mm_add_ps(mssq,_mm_mul_ps(p1,p1)); 
   __m128 p2 = _mm_load_ps(&(z[i])); 
   msumz = _mm_add_ps(msumz,p2);
   mssq = _mm_add_ps(mssq,_mm_mul_ps(p2,p2)); 
  }
  __m128 t1=_mm_hadd_ps (msumx,msumy);
  __m128 t2=_mm_hadd_ps (msumz,mssq);
  __m128 sum0=_mm_hadd_ps(t1,t2);
  _mm_store_ps(sums,sum0);
 }
 for(;i<nat;i++){
  sums[0]+=x[i];
  sums[3]+=x[i]*x[i];
  sums[1]+=y[i];
  sums[3]+=y[i]*y[i];
  sums[2]+=z[i];
  sums[3]+=z[i]*z[i];;
 }
 for(int i=0;i<3;i++)
  center[i]=sums[i]*invfnat;
 return(sums[3]-center[0]*sums[0]-center[1]*sums[1]-center[2]*sums[2]);
}
float kabsch_quat_soa_avx(int nat, int *map, float *x1, float *y1,float *z1, float *x2,float *y2, float *z2,float *r,float *coords_buffer){
 //most of time will be small number of coords
 float fixed_mem[COORDS_BUFFER_SIZE]__attribute__ ((aligned (32)));
 float *mem;
 int upper_nat8= (nat%8)? (nat/8)*8+8: nat;
 int size= 6*upper_nat8*sizeof(float);
 if(size > COORDS_BUFFER_SIZE)
  mem=coords_buffer;
 else
  mem=fixed_mem;
 memset(mem,0,size);
 float* c1x= mem;
 float* c1y= &(mem[upper_nat8]);
 float* c1z= &(mem[upper_nat8*2]); 
 float* c2x= &(mem[upper_nat8*3]);
 float* c2y= &(mem[upper_nat8*4]);
 float* c2z= &(mem[upper_nat8*5]);
 for(int i=0;i<nat;i++){
  int n=map[i];
  c1x[i]=x1[n];
  c1y[i]=y1[n];
  c1z[i]=z1[n];
  c2x[i]=x2[n];
  c2y[i]=y2[n];
  c2z[i]=z2[n];  
 }
 float rms=rmsd_uncentered_avx(nat,c1x,c1y,c1z,c2x,c2y,c2z,r);
 return(rms);
} 
float LG_score_soa_avx (float r[16],int nat, float *x1,float *y1,float *z1, float *x2, float *y2, float *z2, float *d,float invd0d0){//coords organized x4,y4,z4
 float mask_array[56]__attribute__ ((aligned (32)))=
 {1,0,0,0,0,0,0,0,
  1,1,0,0,0,0,0,0,
  1,1,1,0,0,0,0,0,
  1,1,1,1,0,0,0,0,
  1,1,1,1,1,0,0,0,
  1,1,1,1,1,1,0,0,
  1,1,1,1,1,1,1,0};
    
 float fsum=0;
 int i=0,lower_nat8=(nat/8)*8;
 //arrange r0 in duplicates for easy loading into hi and low word
 __m128 *mr= (__m128*)r;
 __m256 sum= _mm256_setzero_ps(); 
 {
  __m256 r0 = _mm256_castps128_ps256(mr[0]);
  __m256 r1 = _mm256_castps128_ps256(mr[1]);
  __m256 r2 = _mm256_castps128_ps256(mr[2]);
  r0=_mm256_insertf128_ps(r0,_mm256_castps256_ps128(r0),1);
  r1=_mm256_insertf128_ps(r1,_mm256_castps256_ps128(r1),1);
  r2=_mm256_insertf128_ps(r2,_mm256_castps256_ps128(r2),1);

  //4th multiplication unecessary as it is all zeros
  __m256 d0 = _mm256_broadcast_ss(&invd0d0);
  __m256 one=_mm256_set1_ps(1.0f);  
  //8 points at a time then mask out
  for( ;i <lower_nat8; i+=8){
   __m256 mx1 = _mm256_load_ps(&(x1[i]));
   __m256 my1 = _mm256_load_ps(&(y1[i]));
   __m256 mz1 = _mm256_load_ps(&(z1[i]));
   __m256 mx2 = _mm256_load_ps(&(x2[i])); 

   __m256 tx1= _mm256_add_ps(_mm256_mul_ps(my1,_mm256_shuffle_ps(r0,r0,0x55)),_mm256_mul_ps(mx1,_mm256_shuffle_ps(r0,r0,0x00)));
   __m256 tx2= _mm256_add_ps(_mm256_mul_ps(mz1,_mm256_shuffle_ps(r0,r0,0xAA)),_mm256_shuffle_ps(r0,r0,0xFF));
   
   tx2 = _mm256_add_ps(tx2,tx1);
   tx2 = _mm256_sub_ps(tx2,mx2);
   mx2 = _mm256_load_ps(&(y2[i])); 
   __m256 d1 = _mm256_mul_ps(tx2,tx2);

   tx1= _mm256_add_ps(_mm256_mul_ps(my1,_mm256_shuffle_ps(r1,r1,0x55)),_mm256_mul_ps(mx1,_mm256_shuffle_ps(r1,r1,0x00)));
   tx2= _mm256_add_ps(_mm256_mul_ps(mz1,_mm256_shuffle_ps(r1,r1,0xAA)),_mm256_shuffle_ps(r1,r1,0xFF));
   tx2= _mm256_add_ps(tx2,tx1);
   tx2= _mm256_sub_ps(tx2,mx2);
   mx2= _mm256_load_ps(&(z2[i])); 
   d1 = _mm256_add_ps(d1,_mm256_mul_ps(tx2,tx2));
   
   tx1= _mm256_add_ps(_mm256_mul_ps(my1,_mm256_shuffle_ps(r2,r2,0x55)),_mm256_mul_ps(mx1,_mm256_shuffle_ps(r2,r2,0x00)));
   tx2= _mm256_add_ps(_mm256_mul_ps(mz1,_mm256_shuffle_ps(r2,r2,0xAA)),_mm256_shuffle_ps(r2,r2,0xFF));
   tx2= _mm256_add_ps(tx2,tx1);
   tx2= _mm256_sub_ps(tx2,mx2);
   d1 = _mm256_add_ps(d1,_mm256_mul_ps(tx2,tx2));
   _mm256_store_ps(&(d[i]),d1); //write out 8 differences            
   mx1= _mm256_mul_ps(d1,d0);
   mx1= _mm256_add_ps(mx1,one);
#ifdef FAST_DIVISION     
   mx1= _mm256_rcp_ps(mx1);
#else
   mx1= _mm256_div_ps(one,mx1);
#endif
   sum= _mm256_add_ps(sum,mx1); 
  
  } 
  for( ;i<nat; i+=8){
   __m256 mask=_mm256_load_ps(&(mask_array[(nat-lower_nat8-1)*8]));   
   __m256 mx1 = _mm256_load_ps(&(x1[i]));
   __m256 my1 = _mm256_load_ps(&(y1[i]));
   __m256 mz1 = _mm256_load_ps(&(z1[i]));
   __m256 mx2 = _mm256_load_ps(&(x2[i])); 

   __m256 tx1= _mm256_add_ps(_mm256_mul_ps(my1,_mm256_shuffle_ps(r0,r0,0x55)),_mm256_mul_ps(mx1,_mm256_shuffle_ps(r0,r0,0x00)));
   __m256 tx2= _mm256_add_ps(_mm256_mul_ps(mz1,_mm256_shuffle_ps(r0,r0,0xAA)),_mm256_shuffle_ps(r0,r0,0xFF));
   
   tx2 = _mm256_add_ps(tx2,tx1);
   tx2 = _mm256_sub_ps(tx2,mx2);
   mx2 = _mm256_load_ps(&(y2[i])); 
   __m256 d1 = _mm256_mul_ps(tx2,tx2);

   tx1= _mm256_add_ps(_mm256_mul_ps(my1,_mm256_shuffle_ps(r1,r1,0x55)),_mm256_mul_ps(mx1,_mm256_shuffle_ps(r1,r1,0x00)));
   tx2= _mm256_add_ps(_mm256_mul_ps(mz1,_mm256_shuffle_ps(r1,r1,0xAA)),_mm256_shuffle_ps(r1,r1,0xFF));
   tx2= _mm256_add_ps(tx2,tx1);
   tx2= _mm256_sub_ps(tx2,mx2);
   mx2= _mm256_load_ps(&(z2[i])); 
   d1 = _mm256_add_ps(d1,_mm256_mul_ps(tx2,tx2));
   
   tx1= _mm256_add_ps(_mm256_mul_ps(my1,_mm256_shuffle_ps(r2,r2,0x55)),_mm256_mul_ps(mx1,_mm256_shuffle_ps(r2,r2,0x00)));
   tx2= _mm256_add_ps(_mm256_mul_ps(mz1,_mm256_shuffle_ps(r2,r2,0xAA)),_mm256_shuffle_ps(r2,r2,0xFF));
   tx2= _mm256_add_ps(tx2,tx1);
   tx2= _mm256_sub_ps(tx2,mx2);
   d1 = _mm256_add_ps(d1,_mm256_mul_ps(tx2,tx2));
   
   _mm256_store_ps(&(d[i]),d1); //write out 8 differences            
   mx1= _mm256_mul_ps(d1,d0);
   mx1= _mm256_add_ps(mx1,one);

#ifdef FAST_DIVISION     
   mx1= _mm256_rcp_ps(mx1);
#else
   mx1= _mm256_div_ps(one,mx1);
#endif
   sum= _mm256_add_ps(sum, _mm256_mul_ps(mx1,mask)); //multiply by mask
  }
 }//end summation

 sum=_mm256_add_ps(sum,_mm256_permute2f128_ps(sum,sum,0x01));  
 sum=_mm256_hadd_ps(sum,sum);
 sum=_mm256_hadd_ps(sum,sum); 
 _mm_store_ss(&(fsum),_mm256_castps256_ps128(sum));
 return(fsum);
}
float rmsd_uncentered_avx (int nat,float *c1x,float *c1y, float *c1z,float *c2x,float *c2y, float *c2z, float *rm){

 float u[3][3];
 double invdnat=1.0/(double)nat;
 float fnat=(float)nat;
 float invfnat=1.0f/fnat;
 float c0[8] __attribute__ ((aligned (32)));
 float c1[16] __attribute__ ((aligned (32)));
 float frr[8] __attribute__ ((aligned (32)));
 int lower8= (nat%8)? (nat/8)*8 : nat;
 int upper8= (nat%8)? (nat/8)*8+8 : nat;
 {
  //use half registers to do everything at once rather than trying for different products
  //essentially SSE3 recipe but with twice as many registers to save one load pass to center coords
  //the 4 hadds will split the lower and upper halves into a variable with 8 sums
  // A E
  // B F
  // C G
  // D H -> sumA sumB sumC sumD sumE sumF sumG sumH
  //these two vectors/registers will also hold final products after HADDS
   __m256 sxxssq = _mm256_setzero_ps();  // hold cross sums 
   __m256 s1xs2x = _mm256_setzero_ps();  // hold sums
  {
   __m256 s1ys2y = _mm256_setzero_ps();  
   __m256 s1zs2z = _mm256_setzero_ps();  
   __m256 sxysyz = _mm256_setzero_ps();
   __m256 sxzszx = _mm256_setzero_ps();
   __m256 syxszy = _mm256_setzero_ps(); 
   __m256 syyszz = _mm256_setzero_ps();  
   
   size_t i=0;
   for( ;i<upper8;i+=8){
   //load the 4 sets of coords from molecule 2 and then load x,y,z of molecule 1
    __m256 mmc2x = _mm256_load_ps(&(c2x[i]));
    __m256 mmc2y = _mm256_load_ps(&(c2y[i]));
    __m256 mmc2z = _mm256_load_ps(&(c2z[i]));
    __m256 t1    = _mm256_load_ps(&(c1x[i])); //do everything necessary with c1x and then use as temp register 
    __m256 t2    = _mm256_mul_ps(t1,t1);

    //block with extra temp
    __m256 mmc1y= _mm256_mul_ps(t1,mmc2x); //used as a temp
    s1xs2x      = _mm256_add_ps(s1xs2x,_mm256_permute2f128_ps(t1,mmc2x,0x20));
    s1xs2x      = _mm256_add_ps(s1xs2x,_mm256_permute2f128_ps(t1,mmc2x,0x31));      
    sxxssq      = _mm256_add_ps(sxxssq,_mm256_permute2f128_ps(mmc1y,t2,0x20));
    sxxssq      = _mm256_add_ps(sxxssq,_mm256_permute2f128_ps(mmc1y,t2,0x31));     
    __m256 mmc1z= _mm256_mul_ps(t1,mmc2y);
    mmc1y       = _mm256_load_ps(&(c1y[i]));
    t2          = _mm256_mul_ps(mmc1y,mmc2z);
    sxysyz      = _mm256_add_ps(sxysyz,_mm256_permute2f128_ps(mmc1z,t2,0x20));
    sxysyz      = _mm256_add_ps(sxysyz,_mm256_permute2f128_ps(mmc1z,t2,0x31));
    mmc1z = _mm256_load_ps(&(c1z[i]));
    t1          = _mm256_mul_ps(t1,mmc2z); //t1 can be used freely now
    t2          = _mm256_mul_ps(mmc1z,mmc2x);
    sxzszx      = _mm256_add_ps(sxzszx,_mm256_permute2f128_ps(t1,t2,0x20));  
    sxzszx     = _mm256_add_ps(sxzszx,_mm256_permute2f128_ps(t1,t2,0x31));
  
    //finish calculating ssq
    t1=_mm256_mul_ps(mmc2x,mmc2x);
    t2=_mm256_mul_ps(mmc2y,mmc2y);
    t1=_mm256_add_ps(t1,_mm256_mul_ps(mmc2z,mmc2z));
    t2=_mm256_add_ps(t2,_mm256_mul_ps(mmc1y,mmc1y));
    t1=_mm256_add_ps(t1,t2);
    t1=_mm256_add_ps(t1,_mm256_mul_ps(mmc1z,mmc1z));
    sxxssq    = _mm256_add_ps(sxxssq,_mm256_permute2f128_ps(t1,t1,0x08));//pass lower 128 and zero lower  0000 1000
    sxxssq    = _mm256_add_ps(sxxssq,_mm256_permute2f128_ps(t1,t1,0x18));//pass upper 128 and zero lower; 0001 1000  
    
    //do other sums
    s1ys2y     =_mm256_add_ps(s1ys2y,_mm256_permute2f128_ps(mmc1y,mmc2y,0x20));
    s1ys2y     =_mm256_add_ps(s1ys2y,_mm256_permute2f128_ps(mmc1y,mmc2y,0x31));
    s1zs2z     =_mm256_add_ps(s1zs2z,_mm256_permute2f128_ps(mmc1z,mmc2z,0x20));
    s1zs2z     =_mm256_add_ps(s1zs2z,_mm256_permute2f128_ps(mmc1z,mmc2z,0x31));  
   
    //do other cross_sums 
    //registers start freeing up
    t1 =  _mm256_mul_ps(mmc1y,mmc2x);
    mmc1y=_mm256_mul_ps(mmc1y,mmc2y);
    t2 =  _mm256_mul_ps(mmc1z,mmc2y); 
    mmc1z=_mm256_mul_ps(mmc1z,mmc2z);
    syxszy     = _mm256_add_ps(syxszy,_mm256_permute2f128_ps(t1,t2,0x20));
    syxszy     = _mm256_add_ps(syxszy,_mm256_permute2f128_ps(t1,t2,0x31));
    syyszz     = _mm256_add_ps(syyszz,_mm256_permute2f128_ps(mmc1y,mmc1z,0x20));
    syyszz     = _mm256_add_ps(syyszz,_mm256_permute2f128_ps(mmc1y,mmc1z,0x31));           
   }//end loop
   //now do two groups of 4 hadds
   //      a1 a2 a3 a4 b1 b2 b3 b4
   //      c1 c2 c3 c4 d1 d2 d3 d4
   //      a12 a34 c12 c34 b12 b34 d12 d34
   //
   //      e1 e2 e3 e4 f1 f2 f3 f4
   //      g1 g2 g3 g4 h1 h2 h3 h4
   //      e12 e34 g12 g34 f12 f34 h12 h34
   //
   //      a12 a34 c12 c34 b12 b34 d12 d34
   //      e12 e34 g12 g34 f12 f34 h12 h34
   //      A C E G B D F H - even/odd
   s1xs2x =_mm256_hadd_ps(_mm256_hadd_ps(sxxssq,s1xs2x),_mm256_hadd_ps(s1ys2y,s1zs2z)); 
   sxxssq =_mm256_hadd_ps(_mm256_hadd_ps(sxysyz,sxzszx),_mm256_hadd_ps(syxszy,syyszz));
//   print_m256(s1xs2x);
//   print_m256(sxxssq); 
  }//no need for other cross sums anymore

  __m256 normed_sums=_mm256_mul_ps(s1xs2x,_mm256_broadcast_ss(&invfnat));
  _mm256_store_ps(c0,normed_sums);

  //sums  sxx s1x s1y s1z ssq s2x s2y s2z  
  //csums sxy sxz syx syy syz szx szy szz
  //      s1x s1x s1y s1y s2z s2x s2y s2z
  //      1   1   2   2   3   1   2   3 
  //      01  01  10  10  11  01  10  11  ->            
  //      1110 0111 1010 0101 (leftmost number are least significant 2 bits)  
  //       E     7    A    5 // A5 E7 in 2 separate permutes to generate the halves and a permute2f to blend them or can load separate mask 
  // want low of first word and high of second word   00 11  00 00 = 0x30 
  // 

  __m256 t1=_mm256_permute2f128_ps(_mm256_permute_ps(normed_sums,0xA5),_mm256_permute_ps(normed_sums,0xE7),0x30);
  __m256 t2=_mm256_permute2f128_ps(s1xs2x,s1xs2x,1);
  // t1   (s1x s1x s1y s1y s2z s2x s2y s2z)*invfnat
  // t2   (ssq s2x s2y s2z sxx s1x s1y s1z) 
  
  //do sxx first
  //second element of t1*s1xs2x has desired element 
  //move to first element - rest is junk
  s1xs2x=_mm256_sub_ps(s1xs2x,_mm256_permute_ps(_mm256_mul_ps(t1,t2),0x01));
  s1xs2x=_mm256_permute_ps(s1xs2x,0);
  s1xs2x=_mm256_permute2f128_ps(s1xs2x,s1xs2x,0);  
  _mm256_store_ps(c1,s1xs2x);
  //       s2y s2z s2x s2y s1y s1z s1z s1z  <- want this for csums subtraction
  //       2   3   1   2   2   3   3   3
  //      10   11  01  10  10  11  11  11
  //      1111 1110 1001 1110
  //        F   E    9    E
 
  t2=_mm256_permute2f128_ps(_mm256_permute_ps(t2,0x9E),_mm256_permute_ps(t2,0xFE),0x30);  
  sxxssq=_mm256_sub_ps(sxxssq,_mm256_mul_ps(t1,t2));
  _mm256_store_ps(&(c1[8]),sxxssq);
  __m256 zero = _mm256_setzero_ps();
  //put into 3 vectors for calculation of cross_products and determinant
  //sxy sxz syx syy syz szx szy szz
  //        sxy sxz
  //    syx syy 
  //syz ...........................
  //
  // 
  //sxxssq    sxy sxz syx syy syz szx szy szz
  //t1        sxy sxy sxy sxz syz syz syz szz  permute mask 0001 ->  01 00 00 00 -> 0x40
  //s1xs2x    sxx sxx sxx sxx sxx sxx sxx sxx  blend mask 00000010
  //r0r0      sxy sxx sxy sxz syz syz syz szz 
  //r0r0       0  sxx sxy sxz  0   0   0   0
  //r0r0       0  sxx sxy sxz  0  sxx sxy sxz
  //sxxssq    sxy sxz syx syy syz szx szy szz   
  //t2        sxy syx syy sxy syz szy szz syz  permute mask 0230   0011 1000 -> 0x38
  //          syz szx szy szz sxy sxz syx syy  swapped sxxssq 0000 0001 
  //t1        syz syz syz syz sxy sxy sxy sxy  permute 
  //t2        sxy syx syy syz syz szy szz syz  blend mask 00001000 -> 0x08
  //r1r2      sxy syx syy syz syz szx szy szz  low word t2 high word sxxssq 0010 0000
  //r1r2      0   syx syy syz  0  szx szy szz
  t1=_mm256_permute_ps(sxxssq,0x40);
  __m256 r0r0=_mm256_blend_ps(t1,s1xs2x,0x02); 
  r0r0 =_mm256_blend_ps(r0r0,zero,0xF1); //mask 11110001
  r0r0 =_mm256_permute2f128_ps(r0r0,r0r0,0);
  t2=_mm256_permute_ps(sxxssq,0x38); 
  t1=_mm256_permute_ps(_mm256_permute2f128_ps(sxxssq,sxxssq,0x01),0);
  t2=_mm256_blend_ps(t2,t1,0x08);
  __m256 r1r2 =_mm256_permute2f128_ps(t2,sxxssq,0x30);//00110000
  r1r2=_mm256_blend_ps(r1r2,zero,0x11); //mask 00010001  
  //do determinant cross
  //r1r2    0 syx syy syz  0  szx szy szz
  //t1      macromask 1 3 2 0    01 11 10 00  78
  //t2      macromask 2 1 3 0    10 01 11 00  9C  
  //        swap 
  //  multiply - lower - upper is cross prodduct
  //  multiply by r0r0  subtract two halves and do hadds
  t1=_mm256_permute_ps(r1r2,0x78);
  t2=_mm256_permute_ps(r1r2,0x9C);
  t1=_mm256_mul_ps(t1,_mm256_permute2f128_ps(t2,t2,0x01));
  t1=_mm256_sub_ps(t1,_mm256_permute2f128_ps(t1,t1,0x01));
  __m256 det=_mm256_mul_ps(r0r0,t1);//det sum is in lower half -sum with other dot products
  __m256 r0r1r0r2=_mm256_mul_ps(r0r0,r1r2);
  __m256 r1r1r2r2=_mm256_mul_ps(r1r2,r1r2);
  r0r0=_mm256_mul_ps(r0r0,r0r0);
  r1r2=_mm256_mul_ps(r1r2,_mm256_permute2f128_ps(r1r2,r1r2,0x01));
  r0r0=_mm256_permute2f128_ps(r0r0,r1r2,0x30);//lower half r0r0 upper half r1r2
  //hadds
  //r0r0 r1r2
  //r0r1 r0r2
  //r1r1 r2r2
  // det -det  -> r0r0 r0r1 r1r1 det r1r2 r0r2 r2r2 -det
  _mm256_store_ps(frr,_mm256_hadd_ps(_mm256_hadd_ps(r0r0,r0r1r0r2),_mm256_hadd_ps(r1r1r2r2,det)));

  
// note that the _MM_SHUFFLE macro reverses the order so that lower integer is first   
//                                                   (lower word t1 upperword of t2  -upperword t1 lower word t2)
//  __m256 cross = _mm_sub_ps (_mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(1,3,2,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(2,1,3,0))), -
//                             _mm_mul_ps(_mm_shuffle_ps(mxy, mxy, _MM_SHUFFLE(2,1,3,0)), _mm_shuffle_ps(mxz, mxz, _MM_SHUFFLE(1,3,2,0))));//cross_product

 } //end AVX section

 const float *r = &(c1[7]); 
 //c0=(ssq s1x s1y s1z sxx s2x s2y s2z)*invfnat  
 const float *center1 = &(c0[1]);
 const float *center2 = &(c0[5]);
 double ssq=(double)((c0[4]-c0[1]*c0[1]-c0[2]*c0[2]-c0[3]*c0[3]-c0[5]*c0[5]-c0[6]*c0[6]-c0[7]*c0[7])*fnat); 
 //normalize ssq
 //double det= (double)( r[0] * ( (r[4]*r[8]) - (r[5]*r[7]) )- r[1] * ( (r[3]*r[8]) - (r[5]*r[6]) ) + r[2] * ( (r[3]*r[7]) - (r[4]*r[6])));
 double det= (double)frr[3];
 double detsq=det*det;
 double rr[6]={(double) frr[0],(double) frr[1],(double) frr[2],(double) frr[5],(double) frr[4],(double) frr[6]};
 
 //lower triangular matrix rr
 double inv3=1.0/3.0;
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
 
 double rms=(ssq-d-d)*invdnat;
 rms=(rms>1e-8)?sqrt(rms) : 0.0f;
 float mr[3][3]={{r[0],r[1],r[2]},
                {r[3],r[4],r[5]},
                {r[6],r[7],r[8]}};
 rmatrix<float>(d,mr,u);
 float mrr[16] __attribute__ ((aligned (16)))=
 {-u[0][0],-u[1][0],-u[2][0],center2[0],
  -u[0][1],-u[1][1],-u[2][1],center2[1],
  -u[0][2],-u[1][2],-u[2][2],center2[2]};
 float w[4]__attribute__ ((aligned (16)));
 float v[4]__attribute__ ((aligned (16)))={center1[0],center1[1],center1[2],1.0f};
 R34v4_sse3(mrr,v,w);
 rm[0] =u[0][0]; rm[1]=u[1][0];rm[2] =u[2][0];rm[3]=w[0];
 rm[4] =u[0][1]; rm[5]=u[1][1];rm[6] =u[2][1];rm[7]=w[1];
 rm[8] =u[0][2]; rm[9]=u[1][2];rm[10]=u[2][2];rm[11]=w[2];
// rm[12]=0.0f;rm[13]=0.0f;rm[14]=0.0f;rm[15]=1.0f; 
 return((float) rms);
}

#endif
template <class T> void dump_matrix(T u[3][3]){
 for(int i=0;i<3;i++){
  for (int j=0;j<3;j++){
   fprintf(stderr,"%8.4f ",u[i][j]);
  }
  fprintf(stderr,"\n");
 } 
}
template <class T> void dump_vector(T u[3]){
fprintf(stderr,"%8.4f %8.4f %8.4f\n",u[0],u[1],u[2]);	
}
#include <iostream>
#include <string>

using namespace std;

void center_all_coords(int nstructs,int nat,float *coords){
 for (int p=0;p<nstructs;p++){
  float sums[3]={0.0f,0.0f,0.0f};
  float invnat=1.0f/(float) nat;
  float *pcoords=&(coords[p*nat*3]);
  for(int i=0;i<nat;i++){
   int m=3*i; 
   sums[0]+=pcoords[m];
   sums[1]+=pcoords[m+1];
   sums[2]+=pcoords[m+2];
  }
  for(int i=0;i<3;i++){
   sums[i]*=invnat; 
  }
  for(int i=0;i<nat;i++){   
   int m=3*i; 
   pcoords[m]-=sums[0];
   pcoords[m+1]-=sums[1];
   pcoords[m+2]-=sums[2];
  }
 }
}
