//Ling-Hong Hung 2013
#define zero4 ((float4)(0.0f,0.0f,0.0f,0.0f)) 
#define sqrt3 1.73205080756888
#define inv3  0.33333333333333
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

float svd(float4 r0, float4 r1, float4 r2);

__kernel void rmsd_matrix (int4 sizes,int4 start_points,__global float4 *coords4,__global float *rmsds)
{
 //outputs all vs all matrix
 
 //stuff that doesn't change with the threads
 const int ng=get_global_size(0);
 const int nt=get_local_size(0);
 const int nat=sizes.x,nat4=sizes.y,nwu=sizes.w;
 const float invfnat=1.0f/(float)nat;

 //initializing private thread dependent variables
 
 int gid=get_global_id(0)/nt;      //workgroup id
 int t=get_local_id(0);            //thread in workgroup
 int nwg=ng/nt;                    //number of workgroups - stride for sampling different pairs of structures
 local float4 lcoords[PDB4SIZE];   //stores coords of first structure using an interweaved pattern 
 
 int4 index=start_points;
 int pstartx=index.x;
 index=(int4)(index.x+gid,index.x+gid+1,index.z,index.w);
 int work_i=index.z*gid-(gid*(gid+1)/2);
 float ssq1;
 //main loop - threads will continue until the workunits are done or all the structures are looped through
 while(index.x < pstartx +index.z -1 && work_i < nwu)
 {
  //read coords
  {
   int i=t;
   ssq1=0;
   while(i<nat4*3)
   {
    float4 pcoords1=coords4[nat4*3*index.x+i];
    ssq1+=dot(pcoords1,pcoords1);
    lcoords[i]=pcoords1;
    i+=nt;
   }
   barrier(CLK_LOCAL_MEM_FENCE);
  }
  //reduce ssq
  {
   local float local_sum[NTHREADS];
   int max_items=nt;
   int diff=max_items/2;
   local_sum[t]=ssq1;
   barrier(CLK_LOCAL_MEM_FENCE);
   while(max_items >1)
   {
    if(t+diff<max_items)
     local_sum[t]+=local_sum[t+diff];
    max_items-=diff;
    diff=max_items/2;
    barrier(CLK_LOCAL_MEM_FENCE);
   }
   ssq1=local_sum[0];
  } 
  work_i+=t;
  index.y=index.x+1+t;
  while(index.y <index.w && work_i <nwu)
  {
   //now read in j - each thread does this individually
   int m=0;
   int offset=nat4*3*index.y;
   float4 r0=zero4,r1=zero4, r2=zero4;
   float ssq=0.0f;
   while(m<3*nat4)
   {
    float4 c1x=lcoords[m++];float4 c1y=lcoords[m++];float4 c1z=lcoords[m++]; 
    float4 c2x=coords4[offset++]; float4 c2y=coords4[offset++]; float4 c2z=coords4[offset++]; 
    ssq+=dot(c2x,c2x)+dot(c2y,c2y)+dot(c2z,c2z); 
    r0+=(float4)(dot(c1x,c2x),dot(c1x,c2y),dot(c1x,c2z),0);
    r1+=(float4)(dot(c1y,c2x),dot(c1y,c2y),dot(c1y,c2z),0);
    r2+=(float4)(dot(c1z,c2x),dot(c1z,c2y),dot(c1z,c2z),0);
   } 
   float ev=svd(r0,r1,r2);
   float rms=(ssq+ssq1-2.0f*ev)*invfnat;
   rmsds[work_i]=(rms>0) ? sqrt(rms) : 0;
   work_i+=nt;
   index.y+=nt;
  }
  index.x+=nwg;
  barrier(CLK_LOCAL_MEM_FENCE);
  {
   int h=index.x-pstartx;
   work_i=index.z*h-(h*(h+1)/2);
  }
 } 
} 

__kernel void rmsd_matrix_rect (int4 sizes,int4 start_points,__global float4 *coords4,__global float4 *coords42,__global float *rmsds)
{
 //outputs all vs all matrix
 
 //stuff that doesn't change with the threads
 const int ng=get_global_size(0);
 const int nt=get_local_size(0);
 const int nat=sizes.x,nat4=sizes.y,nwu=sizes.w; //nstructs is kept in sizes.z - not used
 const float invfnat=1.0f/(float)nat;

 //initializing private thread dependent variables
 
 int gid=get_global_id(0)/nt;      //workgroup id
 int t=get_local_id(0);            //thread in workgroup
 int nwg=ng/nt;                    //number of workgroups - stride for sampling different pairs of structures
 local float4 lcoords[PDB4SIZE];   //stores coords of first structure using an interweaved pattern 
 
 int4 index=start_points;
 int pstartx=index.x;
 int pstarty=index.y;
 index=(int4)(index.x+gid,index.y,index.z,index.w);
 int work_i=index.w*gid;
 float ssq1;
 //main loop - threads will continue until the workunits are done or all the structures are looped through
 while(index.x < pstartx +index.z && work_i < nwu)
 {

  //read coords
  {
   int i=t;
   ssq1=0;
   while(i<nat4*3)
   {
    float4 pcoords1=coords4[nat4*3*index.x+i];
    ssq1+=dot(pcoords1,pcoords1);
    lcoords[i]=pcoords1;
    i+=nt;
   }
   barrier(CLK_LOCAL_MEM_FENCE);
  }
  //reduce ssq
  {
   local float local_sum[NTHREADS];
   int max_items=nt;
   int diff=max_items/2;
   local_sum[t]=ssq1;
   barrier(CLK_LOCAL_MEM_FENCE);
   while(max_items >1)
   {
    if(t+diff<max_items)
     local_sum[t]+=local_sum[t+diff];
    max_items-=diff;
    diff=max_items/2;
    barrier(CLK_LOCAL_MEM_FENCE);
   }
   ssq1=local_sum[0];
  } 
  work_i+=t;
  index.y=pstarty+t;
  while(index.y <pstarty+index.w && work_i <nwu)
  {
   //now read in j - each thread does this individually
   int m=0;
   int offset=nat4*3*index.y;
   float4 r0=zero4,r1=zero4, r2=zero4;
   float ssq=0.0f;
   while(m<3*nat4)
   {
    float4 c1x=lcoords[m++];float4 c1y=lcoords[m++];float4 c1z=lcoords[m++]; 
    float4 c2x=coords42[offset++]; float4 c2y=coords42[offset++]; float4 c2z=coords42[offset++]; 
    ssq+=dot(c2x,c2x)+dot(c2y,c2y)+dot(c2z,c2z); 
    r0+=(float4)(dot(c1x,c2x),dot(c1x,c2y),dot(c1x,c2z),0);
    r1+=(float4)(dot(c1y,c2x),dot(c1y,c2y),dot(c1y,c2z),0);
    r2+=(float4)(dot(c1z,c2x),dot(c1z,c2y),dot(c1z,c2z),0);
   } 
   float ev=svd(r0,r1,r2);
   float rms=(ssq+ssq1-2.0f*ev)*invfnat;
   rmsds[work_i]=(rms>0) ? sqrt(rms) : 0;
   work_i+=nt;
   index.y+=nt;
  }
  index.x+=nwg;
  barrier(CLK_LOCAL_MEM_FENCE);
  work_i=index.w*(index.x-pstartx);
 }
} 

float svd(float4 r0, float4 r1, float4 r2)
{
 double det= r0.x * ( (r1.y*r2.z) - (r1.z*r2.y) )- r0.y * ( (r1.x*r2.z) - (r1.z*r2.x) ) + r0.z * ( (r1.x*r2.y) - (r1.y*r2.x) );
 float d;
 //lower triangular matrix rr
 float rr0=dot(r0,r0); float rr1=dot(r1,r0);float rr2= dot(r1,r1);float rr3= dot(r2,r0);float rr4= dot(r2,r1);float rr5=dot(r2,r2);
 double spur=(rr0+rr2+rr5)/ 3.0;
 double cof=(rr2*rr5 - rr4*rr4 + rr0*rr5- rr3*rr3 + rr0*rr2 - rr1*rr1)*inv3;
 double4 e;
 e.x=spur; e.y=spur;e.z=spur;
 double h=( spur > 0 )? spur*spur-cof : -1.0;
 if(h>0)
 {
  double g = (spur*cof - det*det)/2.0 - spur*h;
  double sqrth = sqrt(h);
  double d1 = h*h*h - g*g;
  d1= ( d1<0 ) ? atan2(0,-g) / 3.0 : atan2(sqrt(d1),-g)*inv3;
  double cth = sqrth * cos(d1);
  double sth = sqrth*sqrt3*sin(d1);
  e.x=spur + cth + cth;e.y=spur - cth + sth;e.z=spur - cth - sth;
 }
 e.x=(e.x < 0) ? 0 : sqrt((float)e.x);
 e.y=(e.y < 0) ? 0 : sqrt((float)e.y);
 e.z=(e.z < 0) ? 0 : sqrt((float)e.z);
 d=(det<0)? e.x + e.y -e.z : e.x + e.y+e.z;
 return(d);
}
