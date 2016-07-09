#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define one4 ((float4)(1.0f,1.0f,1.0f,1.0f))
#define zero4 ((float4)(0.0f,0.0f,0.0f,0.0f)) 
#define sqrt3 1.73205080756888
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

int2 start_finish(int nat,int index);
float2 best_tm(int t,int nt,int pnat, int pnat4,int nseeds,float d0,float d0_search,float4 cmask,__local float4 *lcoords);
float svd(float4 r0, float4 r1, float4 r2);
void rmatrix (float ev,float4 r0,float4 r1,float4 r2,float4 *Rx,float4 *Ry,float4 *Rz);

__kernel void tmscore_matrix (int4 sizes,int input_nseeds,int4 start_points,__global float4 *coords4,__global float2 *tmscores)
{
 //outputs all vs all matrix
 
 //stuff that doesn't change with the threads
 const int ng=get_global_size(0);
 const int nt=get_local_size(0);
 const int nat=sizes.x,nat4=sizes.y,nstructs=sizes.z,nwu=sizes.w;
 const float d0=(1.24*pow((nat-15),(1.0/3.0))-1.8 <0.5) ? 0.5 : 1.24*pow((nat-15),(1.0/3.0))-1.8;
 const float d0_search_max =(d0 > 8)? 8: d0;
 const float d0_search =(d0_search_max < 4.5)? 4.5: d0_search_max;
 const int nseeds=input_nseeds;

 float4 cmask= (float4)(1.0f,1.0f,1.0f,1.0f);
 {
  int rm=nat%4;
  cmask=(rm==1)? (float4)(1.0f,0.0f,0.0f,0.0f) : cmask;
  cmask=(rm==2)? (float4)(1.0f,1.0f,0.0f,0.0f) : cmask;
  cmask=(rm==3)? (float4)(1.0f,1.0f,1.0f,0.0f) : cmask;
 }
 //initializing private thread dependent variables
 
 int gid=get_global_id(0)/nt;      //workgroup id
 int t=get_local_id(0);            //thread in workgroup
 int nwg=ng/nt;                    //number of workgroups - stride for sampling different pairs of structures
 int ichange=1,jchange=1;          //flags to determine whether new coordinates need to be read in
 int work_i=gid;                   //counter to keep track of number of Tmscores calculated by this kernel
 local float4 lcoords[PDB4SIZE*2]; //stores coords of two structures using an interweaved pattern to allow coalesced local memory access - slightly more efficient 
 local float2 local_scores[NTHREADS];
 
 //calculate the starting indices for upper triangular matrix given that workgroup 0 starts at i=start.x j=start.y
 //there is a formula for this but the calculation is unstable due to roundoff for very large indices

 //find number of comparisons assigned to each group
 //give extra ones to first workitems

 //starting pair
 int4 index=start_points; //(starti,startj,ni,nj) i,j are indices of outer and inner loops, ni, nj, number of structures to be looped through

 //initial estimate of workunits assigned 
 int workunits_assigned=nwu/nwg;
 
 //initial workunit is gid*workunits_assigned + min(gid,nwu % nwg);
 
 work_i=gid*workunits_assigned + min(gid,nwu % nwg);
 
 //save start point to calculate the proper indices
 int pstartx=index.x
; 
 //assign extra pairs to first groups
 workunits_assigned=(gid < nwu % nwg)? workunits_assigned+1 : workunits_assigned;

 //loop through the workunits assigned until just before the index of the first assigned workunit is exceeded
 //this allows calculation of outer index from which the inner index is then calculated
 //formula exists for a different triangular pattern but is unstable for large triangular arrays even in double precision
 //constant memory lookup might be a bit faster for very large matrices or might not but is more work on the host side
 //loop is only executed once 
 {
  int in=0;
  while(in < nwu && in+index.w-index.x-1<= work_i)
  {
   in+=index.w-index.x-1; 
   index.x++;
  }
  index.y=work_i-in+index.x+1; //inner loop resets to index.x+1 - calculate index from there
 }

 //main loop - threads will continue until the workunits are done or all the structures are looped through
 for(int iter=0; iter<workunits_assigned && index.x < pstartx+ index.z -1; iter++)
 {
  float2 pbest_score=(float2)(-1,-1);
  //read coords
  if(ichange)
  {
   int i=t;
   int stride=nt;
   while(i<PDB4SIZE)
   {
    lcoords[2*i]=coords4[PDB4SIZE*index.x+i];
    i+=stride;
   }
   ichange=0;
   barrier(CLK_LOCAL_MEM_FENCE);
  } 
  if(jchange)
  {
   int i=t;
   int stride=nt;
   while(i<PDB4SIZE)
   {
    lcoords[2*i+1]=coords4[PDB4SIZE*index.y+i];
    i+=stride;
   }
   jchange=0;
   barrier(CLK_LOCAL_MEM_FENCE);
  }
  //evaluate different starting seeds - loop is in the subroutine
  {
   float2 my_score=  best_tm(t,nt,nat,nat4,nseeds,d0,d0_search,cmask,lcoords);
   pbest_score=(my_score.x>pbest_score.x)? my_score : pbest_score;
   barrier(CLK_LOCAL_MEM_FENCE);
  } 
  //parallel reduction
   {
   int max_items=nt;
   int diff=max_items/2;
   local_scores[t]=pbest_score;
   barrier(CLK_LOCAL_MEM_FENCE);
   while(max_items >1)
   {
    if(t+diff<max_items)
     local_scores[t]=(local_scores[t+diff].x>local_scores[t].x) ? local_scores[t+diff]:local_scores[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    max_items-=diff;
    diff=max_items/2;
   }
   if(t==0)
    tmscores[work_i]=local_scores[0];
  }
  //find next set of upper triangular indices 
  {
   int2 old_index=(int2)(index.x,index.y);
   index=(index.y +1 < index.w)? (int4) (index.x,index.y+1,index.z,index.w) :(int4)(index.x+1,index.x+2,index.z,index.w); 
   ichange=(index.x !=old_index.x) ? 1 : 0;
   jchange=(index.y !=old_index.y) ? 1 : 0;
  }
  work_i++; 
 }
}

__kernel void tmscore_matrix_rect (int4 sizes,int input_nseeds,int4 start_points,__global float4 *coords41,__global float4 *coords42,__global float2 *tmscores)
{

 //stuff that doesn't change with the threads
 const int ng=get_global_size(0);
 const int nt=get_local_size(0);
 const int nwg=ng/nt;
 const int nat=sizes.x,nat4=sizes.y,nstructs=sizes.z,nwu=sizes.w;
 const float d0=(1.24*pow((nat-15),(1.0/3.0))-1.8 <0.5) ? 0.5 : 1.24*pow((nat-15),(1.0/3.0))-1.8;
 const float d0_search_max =(d0 > 8)? 8: d0;
 const float d0_search =(d0_search_max < 4.5)? 4.5: d0_search_max;
 const int nseeds=input_nseeds;
 float4 cmask= (float4)(1.0f,1.0f,1.0f,1.0f);
 {
  int rm=nat%4;
  cmask=(rm==1)? (float4)(1.0f,0.0f,0.0f,0.0f) : cmask;
  cmask=(rm==2)? (float4)(1.0f,1.0f,0.0f,0.0f) : cmask;
  cmask=(rm==3)? (float4)(1.0f,1.0f,1.0f,0.0f) : cmask;
 }

 //initializing private thread dependent variables
 
 int gid=get_global_id(0)/nt;      //workgroup id
 int t=get_local_id(0);            //thread in workgroup
 int ichange=1,jchange=1;          //flags to determine whether new coordinates need to be read in
 int work_i=gid;                   //counter to keep track of number of Tmscores calculated by this kernel
 local float4 lcoords[PDB4SIZE*2]; //stores coords of two structures using an interweaved pattern to allow coalesced local memory access - slightly more efficient 
 local float2 local_scores[NTHREADS];
 //use simple pattern for now - can optimize to decrease reads later if necessary 

 //calculate the starting indices for upper triangular matrix given that workgroup 0 starts at i=start.x j=start.y
 //there is a formula for this but the calculation is unstable due to roundoff for very large indices

 //find number of comparisons assigned to each group
 //give extra ones to first workitems

 //starting pair
 int4 index=start_points; //(starti,startj,ni,nj) i,j are indices of outer and inner loops, ni, nj, number of structures to be looped through

 //initial estimate of workunits assigned 
 int workunits_assigned=nwu/nwg;
 
 //initial workunit is gid*workunits_assigned + min(gid,nwu % nwg);
 
 work_i=gid*workunits_assigned + min(gid,nwu % nwg);
 
 //save both start points to calculate the proper indices
 int pstartx=index.x; int pstarty=index.y;
 
 //assign extra pairs to first groups
 workunits_assigned=(gid < nwu % nwg)? workunits_assigned+1 : workunits_assigned;

 index.x=work_i/index.w;
 index.y=work_i-index.x*index.w;
 //main loop - threads will continue until the workunits are done or all the structures are looped through

 for (int iter=0; iter <workunits_assigned && index.x < pstartx +index.z;iter++)
 {
  float2 pbest_score=(float2)(-1,-1);
  //read coords
  if(ichange)
  {
   int i=t;
   int stride=nt;
   while(i<PDB4SIZE)
   {
    lcoords[2*i]=coords41[PDB4SIZE*index.x+i];
    i+=stride;
   }
   ichange=0;
   barrier(CLK_LOCAL_MEM_FENCE);
  } 
  if(jchange)
  {
   int i=t;
   int stride=nt;
   while(i<PDB4SIZE)
   {
    lcoords[2*i+1]=coords42[PDB4SIZE*index.y+i];
    i+=stride;
   }
   jchange=0;
   barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  //evaluate different starting seeds - loop is in the subroutine
  {
   float2 my_score=  best_tm(t,nt,nat,nat4,nseeds,d0,d0_search,cmask,lcoords);
   pbest_score=(my_score.x>pbest_score.x)? my_score : pbest_score;
   barrier(CLK_LOCAL_MEM_FENCE);
  } 
  //parallel reduction
  //version using half-size local array and private variables doesn't seem to work - maybe some caching of privates not properly synched?
  {
   int max_items=nt;
   int diff=max_items/2;
   local_scores[t]=pbest_score;
   barrier(CLK_LOCAL_MEM_FENCE);
   while(max_items >1)
   {
    if(t+diff<max_items)
     local_scores[t]=(local_scores[t+diff].x>local_scores[t].x) ? local_scores[t+diff]:local_scores[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    max_items-=diff;
    diff=max_items/2;
   }
   if(t==0)
    tmscores[work_i]=local_scores[0];
  }
  {
   int2 old_index=(int2)(index.x,index.y);
   index=(index.y +1 < pstarty+index.w)? (int4) (index.x,index.y+1,index.z,index.w) :(int4)(index.x+1,pstarty,index.z,index.w); 
   ichange=(index.x !=old_index.x) ? 1 : 0;
   jchange=(index.y !=old_index.y) ? 1 : 0;
  }
  work_i++; 
 }
}

float2 best_tm(int t,int nt,int pnat, int pnat4,int nseeds,float d0,float d0_search,float4 cmask,__local float4 *lcoords)
{
 int iseed=t;
 float2 my_pbest_score=(float2)(0.0f,0.0f);
 while(iseed<nseeds)
 {
  int rnat=0;
  int2 endpoints=start_finish(pnat,iseed);
  int start_nat=endpoints.x;  
  int end_nat=endpoints.y; 
  if(start_nat >=0)
  {
   float4 c1x,c1y,c1z,c2x,c2y,c2z;
   float4 s1x=zero4,s1y=zero4,s1z=zero4,s2x=zero4,s2y=zero4,s2z=zero4;
   float4 r0=zero4,r1=zero4,r2=zero4;

   //special case of same 
   int rm=start_nat%4;
   int rm_last=(end_nat+1)%4;
   int m=(start_nat/4)*6;

   //read starting block 
   if (rm)
   {
    float4 mask = (float4)(0.0f,1.0f,1.0f,1.0f);
    mask= (rm==2) ? (float4)(0.0f,0.0f,1.0f,1.0f) : mask;
    mask= (rm==3) ? (float4)(0.0f,0.0f,0.0f,1.0f) : mask;
    c1x=lcoords[m++]*mask;c2x=lcoords[m++]*mask;c1y=lcoords[m++]*mask;c2y=lcoords[m++]*mask;c1z=lcoords[m++]*mask;c2z=lcoords[m++]*mask;
    s1x=c1x,s1y=c1y,s1z=c1z,s2x=c2x,s2y=c2y,s2z=c2z;
    r0+=(float4)(dot(c1x,c2x),dot(c1x,c2y),dot(c1x,c2z),0);
    r1+=(float4)(dot(c1y,c2x),dot(c1y,c2y),dot(c1y,c2z),0);
    r2+=(float4)(dot(c1z,c2x),dot(c1z,c2y),dot(c1z,c2z),0);
    rnat+=4-rm;
   }
 
   //do middle to end part of block - need middle brackets to force round down
   while(m<6*((end_nat+1)/4))
   {  
    c1x=lcoords[m++];c2x=lcoords[m++];c1y=lcoords[m++];c2y=lcoords[m++];c1z=lcoords[m++];c2z=lcoords[m++];
    s1x+=c1x; s1y+=c1y;s1z+=c1z;s2x+=c2x;s2y+=c2y;s2z+=c2z;
    r0+=(float4)(dot(c1x,c2x),dot(c1x,c2y),dot(c1x,c2z),0);
    r1+=(float4)(dot(c1y,c2x),dot(c1y,c2y),dot(c1y,c2z),0);
    r2+=(float4)(dot(c1z,c2x),dot(c1z,c2y),dot(c1z,c2z),0);
    rnat+=4; 
   }
   
   //do the last block  
   if(rm_last)
   {
    float4 last_mask= (float4)(1.0f,0.0f,0.0f,0.0f);
    last_mask= (rm_last==2) ? (float4)(1.0f,1.0f,0.0f,0.0f):last_mask;
    last_mask= (rm_last==3) ? (float4)(1.0f,1.0f,1.0f,0.0f):last_mask;
    c1x=lcoords[m++]*last_mask;c2x=lcoords[m++]*last_mask;c1y=lcoords[m++]*last_mask;c2y=lcoords[m++]*last_mask;c1z=lcoords[m++]*last_mask;c2z=lcoords[m++]*last_mask;
    s1x+=c1x; s1y+=c1y;s1z+=c1z;s2x+=c2x;s2y+=c2y;s2z+=c2z;
    r0+=(float4)(dot(c1x,c2x),dot(c1x,c2y),dot(c1x,c2z),0);
    r1+=(float4)(dot(c1y,c2x),dot(c1y,c2y),dot(c1y,c2z),0);
    r2+=(float4)(dot(c1z,c2x),dot(c1z,c2y),dot(c1z,c2z),0);
    rnat+=rm_last;
   }

   //define rotational vectors.
   float4 Rx,Ry,Rz; 

   //define key matrix - use double precision here here for degenerate cases - only very occasional loss of accuracy  using single precision
   
   float ds1x= (float) (s1x.x+s1x.y+s1x.z+s1x.w),ds1y=(float)(s1y.x+s1y.y+s1y.z+s1y.w),ds1z=(float)(s1z.x+s1z.y+s1z.z+s1z.w),ds2x=(float)(s2x.x+s2x.y+s2x.z+s2x.w),ds2y=(float)(s2y.x+s2y.y+s2y.z+s2y.w),ds2z=(float)(s2z.x+s2z.y+s2z.z+s2z.w);
   float fnat=(float)rnat;
   r0-= (float4) (ds1x*ds2x,ds1x*ds2y,ds1x*ds2z,0)/fnat;
   r1-= (float4) (ds1y*ds2x,ds1y*ds2y,ds1y*ds2z,0)/fnat;
   r2-= (float4) (ds1z*ds2x,ds1z*ds2y,ds1z*ds2z,0)/fnat;
   
   //the order is off diagonal elements, padding, diagonal elements - save a couple of cycles by using 4-vector calculatons in the Jacobi with this ordering
   {
    float ev=svd(r0,r1,r2);
    rmatrix(ev,r0,r1,r2,&Rx,&Ry,&Rz);
   }
   //define translation vectors for center of mass
   float4  tr1x= one4*ds1x/fnat,tr1y= one4*ds1y/fnat,tr1z= one4*ds1z/fnat,tr2x= one4*ds2x/fnat,tr2y =one4*ds2y/fnat,tr2z= one4*ds2z/fnat;
   //calculate rotational vectors by calculating eigenvectors
   //re-initialize the accumulators

   //the following loop reads into private memory the local array and rearranges the coordinates
   //it then updates the accumulators for RMSD calculation immediately
   //this allows us to discard the values and read in new ones

   //two masks are used - mask masks out the contribution to the next rotation matrix from atoms greater than d
   //cmask masks out hte padding zero coordinates used to fill out the 4-vectors

   //set up the initial matrix using the cutoff d=d0_search-1
   //then try to extend the matching subset using d=d0_search+1 - 20 tries;

   for (int s=0;s<2;s++)
   {
    int n=0;
    int niter=(s==0) ? 1 :20;
    float d= (s==0) ? d0_search-1.0f : d0_search+1.0f;    
    int4 drx=(int4)(-1,-1,-1,-1);
    int4 dry=(int4)(-1,-1,-1,-1);
    int4 drz=(int4)(-1,-1,-1,-1);
    float4 min_dif= (float4)(1e-4,1e-4,1e-4,1e-4);
    while( (any(drx) || any(dry) || any(drz))  && n< niter)
    {
     float4 newRx,newRy,newRz;
     int j;
     float score=0;
     float4 score4=zero4;
     s1x=zero4;s1y=zero4;s1z=zero4;s2x=zero4; s2y=zero4;s2z=zero4;
     
     rnat=0;  
     r0=zero4;r1=zero4;r2=zero4;
     for(j=0;j<pnat4;j++)
     {
      int p=j*6;
      float4 mask;
      c1x=lcoords[p++]-tr1x;c2x=lcoords[p++]-tr2x;c1y=lcoords[p++]-tr1y;c2y=lcoords[p++]-tr2y;c1z=lcoords[p++]-tr1z;c2z=lcoords[p]-tr2z;

      float4 v0=(float4)(c1x.x,c1y.x,c1z.x,0);
      float4 v1=(float4)(c1x.y,c1y.y,c1z.y,0);
      float4 v2=(float4)(c1x.z,c1y.z,c1z.z,0);
      float4 v3=(float4)(c1x.w,c1y.w,c1z.w,0);

      float4 u0=(float4)(c2x.x,c2y.x,c2z.x,0);
      float4 u1=(float4)(c2x.y,c2y.y,c2z.y,0);
      float4 u2=(float4)(c2x.z,c2y.z,c2z.z,0);
      float4 u3=(float4)(c2x.w,c2y.w,c2z.w,0);
     
      u0=(float4)(dot(Rx,u0),dot(Ry,u0),dot(Rz,u0),0);
      u1=(float4)(dot(Rx,u1),dot(Ry,u1),dot(Rz,u1),0);
      u2=(float4)(dot(Rx,u2),dot(Ry,u2),dot(Rz,u2),0);
      u3=(float4)(dot(Rx,u3),dot(Ry,u3),dot(Rz,u3),0);

      float4 dist= (float4) (distance(v0,u0),distance(v1,u1),distance(v2,u2),distance(v3,u3));
      float4 dist2= dist/d0;
      mask=step(-d,-dist);
      mask=(j==pnat4-1)?cmask*mask : mask;
      score4=(j==pnat4-1)?score4+cmask*(one4/(one4+dist2*dist2)):score4+one4/(one4+dist2*dist2);
      int temp_atoms=round(dot(mask,mask));
      if(temp_atoms)
      {
       c1x*=mask;c2x*=mask;c1y*=mask;c2y*=mask;c1z*=mask;c2z*=mask;
       s1x+=c1x; s1y+=c1y;s1z+=c1z;s2x+=c2x;s2y+=c2y;s2z+=c2z;
       r0+=(float4)(dot(c1x,c2x),dot(c1x,c2y),dot(c1x,c2z),0);
       r1+=(float4)(dot(c1y,c2x),dot(c1y,c2y),dot(c1y,c2z),0);
       r2+=(float4)(dot(c1z,c2x),dot(c1z,c2y),dot(c1z,c2z),0);
       rnat+=temp_atoms;
      }
     }
     score=(score4.x+score4.y+score4.z+score4.w)/(float)pnat;
     my_pbest_score=(score>my_pbest_score.x) ?(float2)(score,iseed) : my_pbest_score;
     if(rnat >=3)
     { 
      float ds1x= (float) (s1x.x+s1x.y+s1x.z+s1x.w),ds1y=(float)(s1y.x+s1y.y+s1y.z+s1y.w),ds1z=(float)(s1z.x+s1z.y+s1z.z+s1z.w),ds2x=(float)(s2x.x+s2x.y+s2x.z+s2x.w),ds2y=(float)(s2y.x+s2y.y+s2y.z+s2y.w),ds2z=(float)(s2z.x+s2z.y+s2z.z+s2z.w);
      float fnat=(float)rnat;
      r0-= (float4) (ds1x*ds2x,ds1x*ds2y,ds1x*ds2z,0)/fnat;
      r1-= (float4) (ds1y*ds2x,ds1y*ds2y,ds1y*ds2z,0)/fnat;
      r2-= (float4) (ds1z*ds2x,ds1z*ds2y,ds1z*ds2z,0)/fnat;
      {
       float ev=svd(r0,r1,r2);
       rmatrix(ev,r0,r1,r2,&newRx,&newRy,&newRz);
      }

     //the translation vector must take into account the buffered coordinates were already translated using the tr vectors so the new vectors are
     //added to the old vectors to calculate what must be added when reading in the original coordinates
 
      tr1x+= one4*ds1x/fnat;tr1y+= one4*ds1y/fnat;tr1z+= one4*ds1z/fnat;tr2x+=one4*ds2x/fnat;tr2y+=one4*ds2y/fnat,tr2z+= one4*ds2z/fnat;
      drx=convert_int4_sat_rtz((Rx-newRx)/min_dif); //checks if R matrices have changed - the explicit convert function handles nans
      dry=convert_int4_sat_rtz((Ry-newRy)/min_dif);
      drz=convert_int4_sat_rtz((Rz-newRz)/min_dif);    
      Rx=newRx;Ry=newRy;Rz=newRz;
      n++;
     }
     d=(rnat<3)?d+.5:d;
    }
   }
  } 
  iseed+=nt;
 }//end frame loop 
 return(my_pbest_score);
}

void rmatrix(float ev,float4 r0,float4 r1,float4 r2,float4 *Rx,float4 *Ry,float4 *Rz)
{   
 float a00=(r0.x+r1.y+r2.z)-ev;
 float a01=(r1.z-r2.y);
 float a02=(r2.x-r0.z);
 float a03=(r0.y-r1.x);
 float a11=(r0.x-r1.y-r2.z)-ev;
 float a12=(r0.y+r1.x);
 float a13=(r2.x+r0.z);
 float a22=(-r0.x+r1.y-r2.z)-ev;
 float a23=(r1.z+r2.y);
 float a33=(-r0.x-r1.y+r2.z)-ev;

 //from Theobald
 float a2233_3223 = a22 * a33 - a23 * a23; 
 float a1233_3123 = a12 * a33-a13*a23;
 float a1223_3122 = a12 * a23 - a13 * a22; 
 float a0232_3022 = a02 * a23-a03*a22;
 float a0233_3023 = a02 * a33 - a03 * a23;
 float a0231_3021 = a02 * a13-a03*a12;

 float4 q= (float4) (a11*a2233_3223-a12*a1233_3123+a13*a1223_3122, -a01*a2233_3223+a12*a0233_3023-a13*a0232_3022,a01*a1233_3123-a11*a0233_3023+a13*a0231_3021,-a01*a1223_3122+a11*a0232_3022-a12*a0231_3021);
 q=normalize(q);

 float aj=q.s0*q.s0;
 float xj=q.s1*q.s1;
 float yj=q.s2*q.s2;
 float zj=q.s3*q.s3;
 float  xy = q.s1 * q.s2;
 float  az = q.s0 * q.s3;
 float  zx = q.s3 * q.s1;
 float  ay = q.s0 * q.s2;
 float  yz = q.s2 * q.s3;
 float  ax = q.s0 * q.s1; 

 *Rx= (float4) (aj + xj - yj - zj,     2.0f * (xy + az),    2.0f * (zx - ay),0); 
 *Ry= (float4) (2.0f * (xy - az), aj - xj + yj - zj,     2.0f * (yz + ax),0); 
 *Rz= (float4) (2.0f * (zx + ay),     2.0f * (yz - ax), aj - xj - yj + zj,0);
} 

int2 start_finish(int nat,int index)
{
 //given the number of atoms and  - find the start and end points of the subset of atoms to seed
 //might be faster to use image but not as flexible 

 //algorithm for seed length
 //divide length by 1,2,4,8,16
 //if len goes below 4 do seeds of length 4 and stop
 //if divided by 16 and still len not less than 4 do one more round with len ==4

 int len=nat;
 int nseeds=0;
 int divisor=1;
 while(len > 4 && divisor<=16 && index >= nseeds+nat-len+1)
 {
  nseeds+=nat-len+1;
  divisor*=2;
  len=nat/divisor;
 }
 len=(len<4 || divisor > 16)? 4 : len;
 return((int2)(index-nseeds,index-nseeds+len-1));  
}

float svd(float4 r0, float4 r1, float4 r2)
{
 double det= r0.x * ( (r1.y*r2.z) - (r1.z*r2.y) )- r0.y * ( (r1.x*r2.z) - (r1.z*r2.x) ) + r0.z * ( (r1.x*r2.y) - (r1.y*r2.x) );
 float d;
 //lower triangular matrix rr
 float rr0=dot(r0,r0); float rr1=dot(r1,r0);float rr2= dot(r1,r1);float rr3= dot(r2,r0);float rr4= dot(r2,r1);float rr5=dot(r2,r2);
 double spur=(rr0+rr2+rr5)/ 3.0;
 double cof=(rr2*rr5 - rr4*rr4 + rr0*rr5- rr3*rr3 + rr0*rr2 - rr1*rr1) / 3.0;
 double4 e;
 e.x=spur; e.y=spur;e.z=spur;
 double h=( spur > 0 )? spur*spur-cof : -1.0;
 if(h>0)
 {
  double g = (spur*cof - det*det)/2.0 - spur*h;
  double sqrth = sqrt(h);
  double d1 = h*h*h - g*g;
  d1= ( d1<0 ) ? atan2(0,-g) / 3.0 : atan2(sqrt(d1),-g)/3.0;
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
