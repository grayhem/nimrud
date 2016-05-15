# functions using pycuda to access the GPU

import pycuda.autoinit
import pycuda.driver as cuda
import numpy
import time
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

#-------------------------------------------------------------------------------

def dmat_mean(dmat,feats,thresh):
	# g mills 9/10/14
	# calculates the mean values of features associated with distance matrix
	# values below a given threshold (for use in vector field multiscale
	# operators)
	
	# INPUT
	# ~gpuarrays for all~
	# dmat = distance matrix. assuming query set on y axis, search space on
	# x axis.
	# feats = features associated with the search space points, with search
	# space on x axis.
	# thresh = distance threshold
	
	# PARAMETERS
	qserows=numpy.uint32(dmat.shape[0])
	ssprows=numpy.uint32(dmat.shape[1])
	featrows=numpy.uint32(feats.shape[0])
	blockx=int(32)	# update static shared allocations in kernel if you change
	blocky=int(4)	# also update static shared allocations for this
	blockz=int(1)	# pattern depends on this being 1
	gridx=int(1)	# pattern depends on this being 1
	gridy=int(numpy.ceil(qserows/blocky))
	gridz=int(featrows)
	thresh=numpy.float32(thresh)

	
	# OUTPUT
	# acc = means of each feature for each qse point
	# cmat = count of points in each qse neighborhood (as float for friendly
	# division)
	
	
	# allocate accumulator and counter
	acc=gpuarray.zeros((qserows,featrows),dtype=numpy.float32)
	cmat=gpuarray.zeros(qserows,dtype=numpy.float32)
	
	mod=SourceModule("""
	
	__global__ void dmat_mean(float *dmat, float *feats, float *acc, float *cmat, float thresh, int qserows, int ssprows, int featrows)
	{
		// calculates mean feature vectors for each point in the query set. tiles over distance
		// and feat matrices in search space direction (x), and if a thread finds the distance 
		// it reads from *dmat* is less than *thresh*, it adds that value to its accumulator in
		// shared memory. last thread in the x dim sums over shared and writes to *acc*. only 
		// one block in the x dim. last threads in x dim in last block in z dim write to *cmat*.
		// this works because each block is dim 1 in z. 
		
		// note that even though we calculate the mean, we also return the count of entries
		// so that we're set up to calculate the standard deviation on the same dataset next.
		
		// note also the neighborhood point count is a float, because we use it in float math 
		// later.
		
		// thread coords
		int tx = threadIdx.x;							// tiles ssp
		int ty = blockIdx.y*blockDim.y+threadIdx.y;		// spans qse
		int tz = blockIdx.z;							// spans feats
		
		// shared memory declaration- block level accumulator-- assuming 32x, 4y yields 128
		__shared__ float bacc[128];
		
		// block level counter-- 32x, 4y yields 128 (only used if blockIdx.z==num feats)
		__shared__ float bcount[128];		
		
		// shared accumulator write position
		int saw = threadIdx.y*blockDim.x+threadIdx.x;
		
		// shared counter write position
		int scw = threadIdx.y*blockDim.x+threadIdx.x;
		
		// let's go ahead and initialize those shared arrays to zero just to be safe.
		bacc[saw]=0.f;
		bcount[scw]=0.f;
		
		// global accumulator write position
		int gaw = ty*featrows+tz;
		
		// global counter write position
		int gcw = ty;

		// thread read condition (qse only, since we re-evaluate ssp condition on each tile)
		int rc = 0;
		if (ty < qserows) {rc=1;}
				
		// accumulator write condition
		int ac = 0;
		if (tx == blockDim.x-1) {ac=1;}		
		
		// counter write condition
		int cc = 0;
		if ( (tz == gridDim.z-1) && (ac==1) ) {cc=1;}
		
		// *dmat* read position-- add tile offset in tile loop
		int dr = ty*ssprows+tx;
		
		// *feats* read position-- again, tile offset (z dim check not needed)
		int fr = tz*ssprows+tx;
		
		// tile over ssp dim
		for (int i=0; i<ssprows; i+=blockDim.x)
		{
			// check we haven't overshot the edge of *dmat*
			if ( (tx+i<ssprows) && (rc==1) )
			{
				// load this distance from *dmat* and compare to *thresh*
				if (dmat[dr+i]<thresh)
				{
					// throw the feature value on the block accumulator
					bacc[saw]=__fadd_rn(feats[fr+i],bacc[saw]);
					
					// increment the block counter-- probably faster not to check if this thread
					// is in a position to ever use this value and just do it anyway.
					bcount[scw]=__fadd_rn(bcount[scw],1.f);
				}
			}
		}
		
		// synchronize before we sum up and write to *acc* and *cmat*
		__syncthreads();
		
		
		// block accumulator sum
		float featsum=0.f;
		// block counter sum
		float csum=0.f;
		
		
		// here we're going to condense two potential operations into one-- we'll only write to 
		// the global counter if necessary, though
		if (ac==1)
		{
						
			// shared sum position
			int bp=threadIdx.y*blockDim.x;
			
			// sum over the block accumulator and counter arrays
			for (int i=0; i<blockDim.x; i++)
			{
				featsum=__fadd_rn(bacc[bp+i],featsum);
				csum+=bcount[bp+i];
			}
			
			// write the mean to the global accumulator
			acc[gaw]=__fdiv_rn(featsum,(csum+0.0000000001f));
		}
		
		// write the counter sum to the global counter
		if (cc==1)
		{
			cmat[gcw]=csum;
		}
		
	}
	
	
	
	
	""")
	
	
	func=mod.get_function('dmat_mean')
	
	
	func(dmat,feats,acc,cmat,thresh,qserows,ssprows,featrows,block=(blockx,blocky,blockz),grid=(gridx,gridy,gridz))
		
	return acc


#-------------------------------------------------------------------------------

def make_dmat(qse,ssp,measure):
	# g mills 10/10/14
	# makes a distance matrix corresponding to a search space and query set.
	# for vector field multiscale operators.
	
	# INPUT
	# all gpuarray float32
	# ssp = search space 
	# qse = query set
	# measure = 'euclid' or 'cheby'
	
	# PARAMETERS
	ssprows = numpy.uint32(ssp.shape[0])
	qserows = numpy.uint32(qse.shape[0])
	blockx = int(16)
	blocky = int(16)
	gridx = int(numpy.ceil(ssprows/blockx))
	gridy = int(numpy.ceil(qserows/blocky))
		
	# OUTPUT
	dmat=gpuarray.GPUArray((qserows,ssprows),dtype=numpy.float32)
	
	
	mod=SourceModule("""
	
	__global__ void distmat(float *qse, float *ssp, float *dmat, int qserows, int ssprows)
	{
		// builds a distance matrix from a *qse* and *ssp*, storing it in *dmat*
		
		// thread coords
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		int ty = blockIdx.y*blockDim.y+threadIdx.y;
		
		// protect read and write access
		if ( (tx<ssprows) && (ty<qserows) )
		{
			// load the coordinates 
			float x = ssp[3*tx]-qse[3*ty];
			float y = ssp[3*tx+1]-qse[3*ty+1];
			float z = ssp[3*tx+2]-qse[3*ty+2];
			
			// norm of vector
			dmat[ty*ssprows+tx] = __fsqrt_rn(__fmaf_rn(x,x,__fmaf_rn(y,y,__fmul_rn(z,z))));
		}
	}
	
	//------------------------------------------------------------------------------------	
		
	__global__ void distmat_cheby(float *qse, float *ssp, float *dmat, int qserows, int ssprows)
	{
		// builds a distance matrix from a *qse* and *ssp*, storing it in *dmat*. this uses
		// chebyshev norm instead of euclid.
		
		// thread coords
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		int ty = blockIdx.y*blockDim.y+threadIdx.y;
		
		// protect read and write access
		if ( (tx<ssprows) && (ty<qserows) )
		{
			// coordinate spans
			float a = fabsf(__fadd_rn(ssp[3*tx],-qse[3*ty]));
			float b = fabsf(__fadd_rn(ssp[3*tx+1],-qse[3*ty+1]));
			float c = fabsf(__fadd_rn(ssp[3*tx+2],-qse[3*ty+2]));
			
			// get the max for chebyshev distance
			dmat[ty*ssprows+tx] = fmaxf(a,fmaxf(b,c));
		
		}
	}	
	
	""")

	if measure=='cheby':
		func=mod.get_function('distmat_cheby')
	else:
		func=mod.get_function('distmat')
	
	
	func(qse,ssp,dmat,qserows,ssprows,block=(blockx,blocky,1),grid=(gridx,gridy))
	
	return dmat



#-------------------------------------------------------------------------------

def vox_vf_interp(ssp,sspfeats,qse,thresh,measure):
	# g mills 11/10/14	
	# use distance mat kernels to do a basic interpolation of a point cloud
	# vector field to its voxelized representation. just use the mean of all
	# the points in range.
	
	# INPUT
	# gpuarrays all
	# ssp = search space
	# sspfeats = features corresponding to search space
	# qse = query set to receive new features
	# thresh = mean-taking distance
	# measure = 'euclid' or 'cheby'. exactly what it is.
	
	# PARAMETERS
	qserows=numpy.uint32(qse.shape[0])
	ssprows=numpy.uint32(ssp.shape[0])
	featrows=numpy.uint32(sspfeats.shape[0])
	blockx=int(32)	# update static shared allocations in kernel if you change 
	blocky=int(4)	# also update static shared allocations for this
	blockz=int(1)	# pattern depends on this being 1
	gridx=int(1)	# pattern depends on this being 1
	gridy=int(numpy.ceil(qserows/blocky))
	gridz=int(featrows)
	thresh=numpy.float32(thresh)
	
	# OUTPUT
	# new feature vectors for the voxel representation
	qsefeats = gpuarray.zeros((qserows,featrows),dtype=numpy.float32)	
	
	
	mod=SourceModule("""
	
	__global__ void dmat_mean_blind(float *dmat, float *feats, float *acc, float thresh, int qserows, int ssprows, int featrows)
	{
		// this version is for interpolating a vector field point cloud over its voxel 
		// representation, and is not concerned with returning the number of neighboring 
		// observations.
		
		// in this case the search space is the original point cloud and the query set is the
		// voxelized representation of it-- the search space we'll use later.
		
		// calculates mean feature vectors for each point in the query set. tiles over distance
		// and feat matrices in search space direction (x), and if a thread finds the distance 
		// it reads from *dmat* is less than *thresh*, it adds that value to its accumulator in
		// shared memory. last thread in the x dim sums over shared and writes to *acc*. only 
		// one block in the x dim. 
		
				
		// thread coords
		int tx = threadIdx.x;							// tiles ssp
		int ty = blockIdx.y*blockDim.y+threadIdx.y;		// spans qse
		int tz = blockIdx.z;							// spans feats
		
		// shared memory declaration- block level accumulator-- assuming 32x, 4y yields 128
		__shared__ float bacc[128];
		
		// block level counter-- 32x, 4y yields 128 (only used if blockIdx.z==num feats)
		__shared__ float bcount[128];		
		
		// shared accumulator write position
		int saw = threadIdx.y*blockDim.x+threadIdx.x;
		
		// shared counter write position
		int scw = threadIdx.y*blockDim.x+threadIdx.x;
		
		// let's go ahead and initialize those shared arrays to zero just to be safe.
		bacc[saw]=0.f;
		bcount[scw]=0.f;
		
		// global accumulator write position
		int gaw = ty*featrows+tz;
		
		// thread read condition (qse only, since we re-evaluate ssp condition on each tile)
		int rc = 0;
		if (ty < qserows) {rc=1;}
				
		// accumulator write condition
		int ac = 0;
		if (tx == blockDim.x-1) {ac=1;}		
				
		// *dmat* read position-- add tile offset in tile loop
		int dr = ty*ssprows+tx;
		
		// *feats* read position-- again, tile offset (z dim check not needed)
		int fr = tz*ssprows+tx;
		
		// tile over ssp dim
		for (int i=0; i<ssprows; i+=blockDim.x)
		{
			// check we haven't overshot the edge of *dmat*
			if ( (tx+i<ssprows) && (rc==1) )
			{
				// load this distance from *dmat* and compare to *thresh*
				if (dmat[dr+i]<thresh)
				{
					// throw the feature value on the block accumulator
					bacc[saw]=__fadd_rn(feats[fr+i],bacc[saw]);
					
					// increment the block counter-- probably faster not to check if this thread
					// is in a position to ever use this value and just do it anyway.
					bcount[scw]=__fadd_rn(bcount[scw],1.f);
				}
			}
		}
		
		// synchronize before we sum up and write to *acc* and *cmat*
		__syncthreads();
		
		
		// block accumulator sum
		float featsum=0.f;
		// block counter sum
		float csum=0.f;		
		
		// sum over the block accumulator and get the mean
		if (ac==1)
		{
						
			// shared sum position
			int bp=threadIdx.y*blockDim.x;
			
			// sum over the block accumulator and counter arrays
			for (int j=0; j<blockDim.x; j++)
			{
				featsum=__fadd_rn(bacc[bp+j],featsum);
				csum=__fadd_rn(bcount[bp+j],csum);
			}
			
			// write the mean to the global accumulator
			acc[gaw]=__fdiv_rn(featsum,(csum+0.0000000001f));
		}
				
	}
	
	
	
	""")

	func=mod.get_function('dmat_mean_blind')
	
	# build the appropriate distance matrix
	dmat=make_dmat(qse,ssp,measure)
	
	# use the distance matrix to interpolate the vector field observations
	func(dmat,sspfeats,qsefeats,thresh,qserows,ssprows,featrows,block=(blockx,blocky,blockz),grid=(gridx,gridy,gridz))

	return qsefeats




#-------------------------------------------------------------------------------

def rule_threshold(inc, rule):
	# g mills 23/8/15
	# return a boolean mask of points in a plane-bounded region in input point 
	# cloud.
	
	# INPUT
	# inc = incoming point cloud-- gpuarray([[x,y,z],...])
	# rule = ndarray of bounding planes describing the desired partition
	
	# PARAMETERS
	rows = numpy.uint32(inc.shape[0])
	blockx = int(64)
	blocks = (blockx,int(1),int(1))
	gridx = int(numpy.ceil(rows/blockx))
	grids = (gridx,int(1))
	
	# OUTPUT
	# bmask = boolean mask of partition as gpuarray
	bmask = gpuarray.zeros(rows,dtype=numpy.uint32)
	
	
	mod=SourceModule("""
	
	__global__ void thresh(float *inc, int *outc, float xi, float xa, float yi, float ya, float zi, float za, int rows)
	{
		// quick rule-based coordinate thresholding
		int tx = threadIdx.x+blockIdx.x*blockDim.x;

		// check if we're in bounds
		if (tx<rows)
		{
			// output marker
			int outbool = 1;
			
			// load this point
			float x = inc[tx*3];
			float y = inc[tx*3+1];
			float z = inc[tx*3+2];
		
			// check it against each threshold in turn
			if ( (x<xi) || (x>xa) ) {outbool=0;}
			if ( (y<yi) || (y>ya) ) {outbool=0;}
			if ( (z<zi) || (z>za) ) {outbool=0;}
			
			// done
			outc[tx]=outbool;
		}
	}
	""")
	
	thresh = mod.get_function("thresh")
	# parse the input rule
	xi=numpy.float32(rule[0,0])
	xa=numpy.float32(rule[0,1])
	yi=numpy.float32(rule[1,0])
	ya=numpy.float32(rule[1,1])
	zi=numpy.float32(rule[2,0])
	za=numpy.float32(rule[2,1])
	
	# call the kernel
	thresh(inc,bmask,xi,xa,yi,ya,zi,za,rows,block=blocks,grid=grids)
	
	return bmask
	
	

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def PT_cov(a,cents,irows):
	# g mills 29/9/14
	# get covariance matrix of *a* with centroids *cents* over rows *irows*.

	# INPUT
	# a = incoming neighborhood tensor
	# cents = array of centroids of *a*
	# irows = interesting rows array

	# PARAMETERS
	rows=numpy.uint32(a.shape[1])
	pags=numpy.uint32(a.shape[0])
	blockx=int(3)
	blocky=int(3)
	blockz=int(1)
	gridx=int(1)
	gridy=int(1)
	gridz=int(pags)


	# OUTPUT
	c = gpuarray.zeros((pags,3,3),dtype=numpy.float32)


	mod=SourceModule("""

	__global__ void PT_cov(float *a, float *c, float *cents, int *irows, int rows)
	{
		// matrix multiplication: c=a.T x a, on pages of a tensor. each thread
		// here aims at one entry in output c. assuming j=3. also assuming
		// grid z dim = pages. subtracts the centroid as it loads the point.
		// a: i=rows, j=cols, k=pags
		// aT: j=rows, i=cols, k=pags

		// shared memory allocation. block size here is static-- 3x3x1.
		__shared__ float smat[9];
		__shared__ float cen[3];

		// thread coordinates
		int tx = threadIdx.x;		// traverses i	(3)
		int ty = threadIdx.y;		// traverses j	(3)
		int tz = blockIdx.z;		// traverses k

		// page offsets
		int apo = tz*rows*3;
		int cpo = tz*9;

		// write index
		int wi = cpo+tx*3+ty;

		// shared memory write coordinate
		int wsmc = tx*3+ty;

		// absolute read position for *a*, modified each pass in the loop
		int ra;

		// row index in *a* for clarity, modified each pass
		int ta;

		// sum to be written to *c*
		float csum = 0.f;

		// stopping point for looping over the page
		int stopping=irows[tz];

		// centroid for this neighborhood
		if (tx==0)
		{
			cen[ty]=cents[tz*3+ty];
		}
		__syncthreads();

		// loop over the page
		for (int t=0; t<((2+stopping)/3);t++)
		{

			// find this thread's row index in a
			ta = t*3+tx;

			// find this thread's read index in a
			ra = apo+t*9+tx*3+ty;

			// if thread is in domain of *a* then load its point in the shared
			// memory blocks.
			if (ta<stopping) 	// read from *a*
			{
				smat[wsmc]=__fsub_rn(a[ra],cen[ty]);
			}
			else
			{
				smat[wsmc]=0.f;
			}

			__syncthreads();


			// transpose, multiply, add to the sum. nested intrinsics in place
			// of a loop.
			csum=__fmaf_rn(smat[tx],smat[ty],csum);
			csum=__fmaf_rn(smat[tx+3],smat[ty+3],csum);
			csum=__fmaf_rn(smat[tx+6],smat[ty+6],csum);

		}

		// write to *c*
		c[wi]=csum;

	}

	""")

	cov=mod.get_function('PT_cov')

	cov(a,c,cents,irows,rows,block=(blockx,blocky,blockz),grid=(gridx,gridy,gridz))

	return c


#-------------------------------------------------------------------------------

def PTcentroid(gnb,irows):
	# g mills 15/7/14
	# calculates the centroid of each neighborhood in a tensor and the norm
	# to its qse point.

	# INPUT
	# gnb = neighborhood tensor on gpu
	# irows = irows data for gnb

	# PARAMETERS
	rows=numpy.uint32(gnb.shape[1])
	pags=numpy.uint32(gnb.shape[0])
	dom=numpy.uint32(64)			# number of points each thread reduces
	arows=numpy.uint32(numpy.ceil(rows/dom))	# num rows in accumulator array
	blocksx=int(1)		# blocks are skinny in sum kernel
	blocksy=int(512)	# 512 gives a max neighborhood of 32768 points w/ 64 dom
	blockdx=int(64)		# division kernel
	gridsx=int(pags)
	gridsy=int(1)		# grid y dim in sum kernel
	griddx=int(numpy.ceil(pags/blockdx))

	# OUTPUT
	mvec = gpuarray.zeros((pags),dtype=numpy.float32)	# norms of mean vectors
	cent = gpuarray.zeros((pags,3),dtype=numpy.float32)	# centroids


	mod=SourceModule("""

	__global__ void PTcolsum(float *gnb, float *acc, float *cent, int *irows, int pags, int rows, int arows, int dom)
	{
		// sum up each column in *gnb*, accumulating to *acc*. last thread of each block sums
		// *acc* into *cent*. this is intended for neighborhood tensors and therefore each
		// block accounts for an entire page instead of a column.

		// we have 2**31-1 blocks in X and 65535 in Y and Z. each block is long and skinny and
		// covers one page, with each thread covering *dom* rows.

		// thread coordinates
		int tx = blockDim.x*blockIdx.x+threadIdx.x;
		int ty = blockDim.y*blockIdx.y+threadIdx.y;

		// page offset in *inc*. ydim spans the page, xdim spans the set of pages.
		int ipo = tx*rows*3;

		// page offset in *acc*
		int apo = tx*arows*3;

		// starting offset
		int iro = ipo+ty*3*dom;		// read *inc*
		int aro = apo+ty*3;			// write *acc*

		// local accumulators
		float x = 0.f;
		float y = 0.f;
		float z = 0.f;

		// number of interesting rows
		int ir=irows[tx];

		if ( (ty*dom)<(ir) )
		{
			// loop over thread's domain
			for (int f=0; f<dom; f++)
			{
				// make sure this loop doesn't push us to the next page
				if ((ty*dom+f)<ir)
				{
					x+=gnb[iro+3*f];
					y+=gnb[iro+3*f+1];
					z+=gnb[iro+3*f+2];
				}
			}
			// write to the big accumulator
			acc[aro]=x;
			acc[aro+1]=y;
			acc[aro+2]=z;
		}

		__syncthreads();

		// first thread writes results to output
		if (ty==0)
		{
			// reset accumulators
			x=0.f;
			y=0.f;
			z=0.f;

			// loop over the interesting part of the page in *acc*
			for (int f=0; f<arows; f++)
			{
				x+=acc[apo+f*3];
				y+=acc[apo+f*3+1];
				z+=acc[apo+f*3+2];
			}

			// fill output
			cent[tx*3]=x;
			cent[tx*3+1]=y;
			cent[tx*3+2]=z;
		}


	}


	//--------------------------------------------------------------------------

	__global__ void PTdivide_norm(float *cent, float *mvec, int *irows, int pags)
	{
		// intended for use on output of PTcolsum. divide the sum of each col
		// of each page by the *irows* value associated with that page. then
		// produce a euclidian norm for each centroid.

		// thread coordinate
		int tx = blockDim.x*blockIdx.x+threadIdx.x;

		// read coordinates
		int ini = tx*3;		// for cent

		// make sure we're in bounds
		if (tx<pags)
		{
			// divide
			float x=__fdiv_rn(cent[ini],(irows[tx]+0.f));
			float y=__fdiv_rn(cent[ini+1],(irows[tx]+0.f));
			float z=__fdiv_rn(cent[ini+2],(irows[tx]+0.f));

			// get the norm
			mvec[tx]=__fsqrt_rn(__fmaf_rn(x,x,__fmaf_rn(y,y,__fmul_rn(z,z))));

			// we're also going to output the centroid itself here to save some
			// launches and allocations
			cent[ini]=x;
			cent[ini+1]=y;
			cent[ini+2]=z;

		}
	}
	""")

	colsum=mod.get_function('PTcolsum')
	div=mod.get_function('PTdivide_norm')

	# initialize accumulator
	acc=gpuarray.zeros((pags,arows,3),dtype=numpy.float32)

	# first sum up each col of each page
	colsum(gnb,acc,cent,irows,pags,rows,arows,dom,block=(blocksx,blocksy,1),grid=(gridsx,gridsy))

	# now divide and get norm
	div(cent,mvec,irows,pags,block=(blockdx,1,1),grid=(griddx,1))

	return mvec, cent




#-------------------------------------------------------------------------------

def row_norm_sort(inc):
	# g mills 30/3/15
	# normalizes the rows of *inc* to sum to 1, and then sorts them in
	# descending order. used in mso workflow on gpu.

	# INPUT
	# inc = nx3 float32 gpuarray.

	# PARAMETERS
	rows = numpy.uint32(inc.shape[0])
	blockx = int(256)
	gridx = int(numpy.ceil(rows/blockx))

	# OUTPUT
	# inc with rows normalized to 1.f, on gpu


	mod=SourceModule("""

	__global__ void row_normalize(float *inc, int rows)
	{
		// normalize the rows in *inc* to sum to 1

		// thread coords
		int tx = blockDim.x*blockIdx.x+threadIdx.x;

		// accumulator
		float acc=0.f;

		// starting read position
		int rp=tx*3;

		// in bounds check
		if (tx<rows)
		{

			// read off the values in this row
			float a = inc[rp];
			float b = inc[rp+1];
			float c = inc[rp+2];

			// sum to the accumulator
			acc=a+b+c;

			// normalize by the accumulator
			a=__fdiv_rn(a,acc);
			b=__fdiv_rn(b,acc);
			c=__fdiv_rn(c,acc);

			// get first and second eigvals
			float first=fmaxf(a,fmaxf(b,c));
			float third=fminf(a,fminf(b,c));
			float second=1-(first+third);

			// send em home
			inc[rp]=first;
			inc[rp+1]=second;
			inc[rp+2]=third;

		}
	}


	""")

	func=mod.get_function('row_normalize')

	func(inc,rows,block=(blockx,1,1),grid=(gridx,1))

	return inc


#-------------------------------------------------------------------------------

def block_eigvals(a):
	# g mills 30/3/15
	# calculates eigenvalues for a set of 3x3 covariance matrices.

	# INPUT
	# a = float32 gpuarray 3-tensor of 3x3 covariance matrices

	# PARAMETERS
	pags=numpy.uint32(a.shape[0])
	blockx=int(1)
	blocky=int(3)
	blockz=int(1)
	gridx=int(pags)


	# OUTPUT
	eigs = gpuarray.zeros((pags,3),dtype=numpy.float32)


	mod=SourceModule("""
	__global__ void pt_eigvals(float *cov, float *eigs, int pags)
	{
		// direct calculation of eigenvalues for 3x3 real valued, symmetric
		// matrices. each block of 3 threads takes on one matrix; each thread
		// takes a row and writes one eigval to output.

		// thread coords
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		int ty = threadIdx.y;

		// shared memory declarations
		__shared__ float A[9];	// the input data
		__shared__ float B[9];	// an intermediate step

		// check if we're in bounds
		if (tx<pags)
		{
			// starting read index
			int sri = tx*9+ty*3;

			// starting shared index-- this thread deals with 3 shared mem indices
			int ssi = ty*3;

			// write index
			int wi = tx*3+ty;

			// read from global into A
			A[ssi]=cov[sri];
			A[ssi+1]=cov[sri+1];
			A[ssi+2]=cov[sri+2];

			// catch up
			__syncthreads();

			// calculate p1- sum of squares of upper triangle
			// higher precision:
			float p1 = fmaf(A[5],A[5],fmaf(A[2],A[2],A[1]*A[1]));

			// if it's diagonal, we're in luck
			if (p1<=0.00000001f) {eigs[wi]=A[4*ty];}

			// otherwise we have a little more math to do
			else
			{
				float q = fdividef(A[0]+A[4]+A[8],3.0f);

				float p2 = fmaf(A[0]-q,A[0]-q,fmaf(A[4]-q,A[4]-q,fmaf(A[8]-q,A[8]-q,2.0f*p1)));
				float p = sqrtf(fdividef(p2,6.0f));

				// fill B
				if (ty==0) {B[ssi]=fdividef(A[ssi]-q,p);}
				else {B[ssi]=fdividef(A[ssi],p);}
				if (ty==1) {B[ssi+1]=fdividef(A[ssi+1]-q,p);}
				else {B[ssi+1]=fdividef(A[ssi+1],p);}
				if (ty==2) {B[ssi]=fdividef(A[ssi+2]-q,p);}
				else {B[ssi+2]=fdividef(A[ssi+2],p);}


				// catch up
				__syncthreads();

				// now get the determinant.

				float r = fmaf(B[0]*B[4],B[8],
							fmaf(B[1]*B[5],B[6],
							B[3]*B[7]*B[2]))
							-fmaf(B[2]*B[4],B[6],
							fmaf(B[1]*B[3],B[8],
							B[0]*B[5]*B[7]));

				float phi;
				if (r<=-1.0f) { phi = fdividef(3.141592654f,3); }
				else if (r>=1.0f) { phi = 0.f; }
				else { phi = fdividef(acosf(r),3); }

				// write back the eigvals
				float eig1 = fmaf(2.0f*p,cosf(phi),q);
				float eig3 = fmaf(2.0f*p,cosf(phi+2.0f*fdividef(3.141592654f,3.0f)),q);
				float eig2 = 3.0f*q-eig1-eig3;

				if (ty==0) {eigs[wi]=eig1;}
				else if (ty==1) {eigs[wi]=eig2;}
				else {eigs[wi]=eig3;}
			}
		}
	}


	""")

	func=mod.get_function('pt_eigvals')

	func(a,eigs,pags,block=(blockx,blocky,blockz),grid=(gridx,1))

	return eigs




#-------------------------------------------------------------------------------

def PTshrink(gnb,irows,i2,val):
	# g mills 9/6/14
	# multi-step process to de-sparsify search space point clouds in a
	# neighborhood tensor for multiscale analysis. performs a scan over the
	# interesting region of each neighborhood to calculate new indices for the
	# points therein (while also checking if they meet the max distance
	# requirements and dropping if necessary), then scatters the points that
	# pass to their new indices at the head of the array. it then determines
	# the size of the new interesting region of each neighborhood point cloud.

	# interesting region (irows) refers to the part of the page of the
	# tensor that contains data to be worked on. a page's entry in *irows*
	# is the SIZE of the interesting region, not the last entry in it.

	# INPUT
	# notice: everything will be modified except *val*
	# gnb = neighborhood tensor.
	# irows = set of interesting region sizes.
	# i2 = max row-wise length allowable for output neighborhood tensor.
	# call with zero on second half of the mspca pipeline.
	val = numpy.float32(val)	# new maximum radius constraint for the nbhd


	# PARAMETERS

	# blocks
	blockpx=8			# block dimension for scan prep and scatter kernels
	blocksx=512			# block dimension for scan kernels-- 512 works.
	blockix=512			# block dimension for irowset kernel

	# dimensions
	rows=numpy.uint32(gnb.shape[1])			# rows in neighborhood tensor/ 'i'
	pags=numpy.uint32(gnb.shape[0])			# pages in neighborhood tensor/ 'k'

	# grids
	gridpx=int(numpy.ceil(rows/blockpx))	# neighborhood spanning dimension in 
												# prep/ scatter
	gridpy=int(numpy.ceil(pags/blockpx))	# query set spanning dimension in 
												# same
	gridsx=int(numpy.ceil(rows/blocksx))	# neighborhood spanning dim in scan 
												# kernels
	gridsy=int(pags)						# query set spanning dim in same
	gridix=int(numpy.ceil(pags/blockix))	# query set spanning dimension in 
												# irowset

	# launch parameters
	blockps=(blockpx,blockpx,1)				# block dimensions for prep/ scatter
	gridps=(gridpx,gridpy)					# grid for same
	blockss=(blocksx,1,1)					# block dimensions for scan kernels
	gridss=(gridsx,gridsy)					# grid for same
	blockis=(blockix,1,1)					# block dimensions for irowset
	gridis=(gridix,1)						# grid for same


	# OUPUT
	# gnb-- with all relevant entries scattered to the head of each page
	# irows-- updated with new indices reflecting the densification


	mod=SourceModule("""


	__global__ void PTDscanprepFP( int *scanlist, float *inc, int *irows, float val, int pags, int rows)
	{
		// preps a scanlist for neighborhood tensor *inc* based on *irows*
		// and *val*.

		// this version is used on the first pass.

		// coordinates-- x dim spans the neighborhood/ rows, y dim spans
		// qset points/ pages
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		int ty = blockIdx.y*blockDim.y+threadIdx.y;

		// page offset from beginning of tensor
		int pag = ty*rows*3;

		// tensor read start position for this thread
		int rs = pag + tx*3;

		// scanlist initial write position
		int sw = ty*rows+tx;

		// number of rows we're interested in on this page-- note we have to
		// protect this read.
		int ir = 0;
		if (ty<pags) {ir = irows[ty];}

		// storage for distance to the origin-- used to nullify non-writing scan indices
		float d = 0.f;

		// load the point and initialize the scan list
		if ( (tx<ir) && (ty<pags) )
		{
			// load the point
			float x = inc[rs];
			float y = inc[rs+1];
			float z = inc[rs+2];

			// calculate the point distance to the origin
			d=__fsqrt_rn(__fmaf_rn(x,x,__fmaf_rn(y,y,__fmul_rn(z,z))));

			// check if it's the first point or not at the origin and closer
			// than the max distance.
			if ( (tx==0) || ( (d>0.0001)  && (d<val) ) )
			{
				scanlist[sw]=1;
			}

			else
			{
				scanlist[sw]=0;
			}
		}
	}


	__global__ void PTDscanprep( int *scanlist, float *inc, int *irows, float val, int pags, int rows)
	{
		// preps a scanlist for neighborhood tensor *inc* based on *irows* and
		// *val*.

		// coordinates-- x dim spans the neighborhood/ rows, y dim spans qset
		// points/ pages
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		int ty = blockIdx.y*blockDim.y+threadIdx.y;

		// page offset from beginning of tensor
		int pag = ty*rows*3;

		// tensor read start position for this thread
		int rs = pag + tx*3;

		// scanlist initial write position
		int sw = ty*rows+tx;

		// number of rows we're interested in on this page-- note we have to
		// protect this read.
		int ir = 0;
		if (ty<pags) {ir = irows[ty];}

		// storage for distance to the origin-- used to nullify non-writing
		// scan indices
		float d = 0.f;

		// load the point and initialize the scan list
		if ( (tx<ir) && (ty<pags) )
		{
			// load the point
			float x = inc[rs];
			float y = inc[rs+1];
			float z = inc[rs+2];

			// calculate the point distance to the origin
			d=__fsqrt_rn(__fmaf_rn(x,x,__fmaf_rn(y,y,__fmul_rn(z,z))));

			// check if it's not the origin and closer than the max distance
			if ( (d>0.0001) && (d<val) )	// adjust for tiny distances.
			{
				scanlist[sw]=1;
			}

			else
			{
				scanlist[sw]=0;
			}
		}
	}


	//--------------------------------------------------------------------------


	__global__ void PTsegscan( int *scanlist, int *metascan, int *irows, int rows, int pags)
	{
		// segment scan for the rows of *scanlist*, up to *irows*. the last
		// working thread in each block of the scan also writes its value to
		// *metascan* so the next kernel can unify the scan. finally, we'll use
		// the membership (0/1 val) of the thread in inbound *scanlist* to
		// mask non-writing entries as the final step.

		// coordinates-- note we're expecting blockDim.y to be 1.
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		int ty = blockIdx.y*blockDim.y+threadIdx.y;

		// protected read for "interesting rows" (in this case columns) value
		int ir = 0;
		if (ty<pags) {ir = irows[ty];}

		// shared memory declaration
		extern __shared__ int ssg[];

		// thread position in global memory
		int gm = ty*rows+tx;

		// thread position in shared memory
		int sm = threadIdx.x;

		// thread position in *metascan* (will be pags * ceil(rows/blockDim.x))
		int mm = ty*gridDim.x+blockIdx.x;

		// number of iterations required for the scan-- equivalent to log2 of
		// blockDim.x. which is going to be a power of 2
		int iters = __ffs(blockDim.x)-1;

		// scan offset
		int space;
		// short storage for mid-scan
		int sval=0;
		// membership of this thread's entry (for masking)
		int membership=0;

		// copy this thread's value into shared memory
		if (tx<ir) {ssg[sm]=scanlist[gm];}

		__syncthreads();

		// record the initial value for masking purposes later
		membership = ssg[sm];

		// iterate the scan
		for (int i=0; i<iters; i++)
		{
			// scan offset
			space=1<<i;

			if ( (sm>=space) && (tx<ir) )		// add entries for this part of the scan
			{
				sval=ssg[sm]+ssg[sm-space];
			}
			else if ( (sm<space) && (tx<ir)	) 	// or copy down if no add takes place.
			{
				sval=ssg[sm];
			}

			__syncthreads();

			// now we can copy back to shared memory

			if (tx<ir)
			{
				ssg[sm]=sval;
			}

			__syncthreads();	// and back to the top

		}
		// write the last value in the segment to *metascan*.
		// if thread is the last one in a block in addition to being
		// within the domain of *metascan*, then write.

		if ( (sm==blockDim.x-1) && (ty<pags) ) {metascan[mm]=sval;}


		// and write back to global if the thread is within scan domain,
		// masking if necessary.
		if (membership==0) {sval+=1000000;}
		if ( (tx<ir) && (ty<pags) )	{scanlist[gm]=sval;}
	}


	//--------------------------------------------------------------------------

	__global__ void PTmetascan( int *scanlist, int *metascan, int *irows, int rows, int pags)
	{
		// broadcasts *metascan* onto *scanlist* to make a complete scan.
		// comes after PTsegscan. should be launched with same block/grid
		// parameters as PTsegscan.

		// NOTE: each block will scan *metascan*, instead of launching
		// PTsegscan a second time for the same purpose. since the segments in
		// PTsegscan are so large, this kernel will likely only be responsible
		// for scanning 16 entries, at most.

		// coordinates-- note we're expecting blockDim.y to be 1.
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		int ty = blockIdx.y*blockDim.y+threadIdx.y;

		// starting read coordinate for *metascan*
		int msread = ty*gridDim.x;

		// read coordinate for *scanlist*
		int scread = ty*rows+tx;

		// protected read for interesting row domain
		int ir = 0;
		if (ty<pags) {ir=irows[ty];}


		// scan and broadcast --------------------------------------------------

		// shared memory to scan *metascan*
		__shared__ int mts[1];

		// scan accumulator
		int acc = 0;

		// load and scan *metascan* up to the value needed for the block.
		if ( ( (ty<pags) && (tx<ir) ) && ( (threadIdx.x==0) && (tx!=0) ) )
		{
			for (int b=0; b<blockIdx.x; b++)
			{
				acc+=metascan[msread+b];
			}
			mts[0]=acc;
		}
		else if (tx==0) {mts[0]=0;}

		__syncthreads();

		// scan and broadcast completed ----------------------------------------


		// map the scan values onto the segment scan----------------------------

		if ( ( (blockIdx.x !=0) && (ty<pags) ) && (tx<ir) )
		{
			scanlist[scread]+=mts[0];
		}
	}

	//--------------------------------------------------------------------------

	__global__ void PTshrink(int *scanlist, float *target, float *inc, int *irows, int pags, int rows, int newrows)
	{
		// scatter the entries of *inc* to locations specified by *scanlist*
		// in *target*.


		// coordinates-- x dim spans the neighborhood/ rows,
		// y dim spans qset points/ pages
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		int ty = blockIdx.y*blockDim.y+threadIdx.y;

		// page offset from beginning of tensor- on input
		int pag = ty*rows*3;

		// page offset from beginning of tensor- on output
		int npag = ty*newrows*3;

		// read start position for this thread
		int rs = pag + tx*3;

		// scanlist read position
		int sw = ty*rows+tx;

		// number of rows we're interested in on this page-- protected
		// read required
		int ir = 0;

		// scatter location from *scanlist*-- also protected read
		int sl = 1000000;

		// protected read for the above
		if (ty<pags)
		{
			ir = irows[ty];

			if (tx<ir)
			{
				sl = scanlist[sw];
			}
		}

		// check if this thread will copy its entry, and load if so. this one
		// check is all that's necessary, since we'll only have a read index
		// less than 1 mil if we're in domain AND the point should be read.
		if (sl<1000000)
		{

			// calculate the write index for this thread-- INCLUSIVE scan
			int idx = 3*(sl-1)+npag;

			// and scatter
			target[idx]=inc[rs];
			target[idx+1]=inc[rs+1];
			target[idx+2]=inc[rs+2];

		}
	}


	//--------------------------------------------------------------------------

	__global__ void PTirowset(int *scanlist, int *irows, int pags, int rows)
	{
		// grabs the entry at *irows* for each page in *scanlist* and uses it
		// to update irows. a max reduction is not necessary, as the last value
		// in the interesting part of the *scanlist* will be the number of
		// rows with data after shrinking. note we're using *irows* to hold the
		// size of the interesting region, not point to the last entry.

		// note that the convention used with *scanlist* is that it's assumed
		// to be a 3-tensor with 1 column. hence pages and rows instead of rows
		// and columns.

		// this is a small launch. max 1024 threads per block on cc 3.5.
		int tx = blockIdx.x*blockDim.x+threadIdx.x;

		// page offset
		int pago = tx*rows;

		// check if we're in the domain
		if (tx<pags)
		{
			// use *irows* to find the entry in *scanlist* we're looking for.
			// (it's the region size - 1).
			int ir = irows[tx];
			if (ir>0) {ir-=1;}
			int nir = scanlist[pago+ir];

			// write this entry back to *irows*. since this is an inclusive
			// scan, the value at the entry is the same as the size of the
			// interesting region. this requires a check if the last entry is
			// masked-- in which case we have to subtract 1 mil from it to
			// get the right index.
			if (nir>=1000000) {nir-=1000000;}
			irows[tx]=nir;
		}
	}

	""")

	# get functions
	if i2==0:
		scanprep=mod.get_function('PTDscanprep')
	else:
		scanprep=mod.get_function('PTDscanprepFP')
	segscan=mod.get_function('PTsegscan')
	metascan=mod.get_function('PTmetascan')
	scatter=mod.get_function('PTshrink')
	irowset=mod.get_function('PTirowset')

	# initialize the scanlist
	scanlist=gpuarray.GPUArray((int(pags),int(rows)),dtype=numpy.uint32)
	scanlist.fill(0)

	# initialize the meta scanlist
	metasl=gpuarray.zeros((pags,gridsx),dtype=numpy.uint32)

	# load the scanlist
	scanprep(scanlist, gnb, irows, val, pags, rows, block=blockps,grid=gridps)

	# calculate shared memory for use in segment scan
	shares=int(blocksx*numpy.dtype(numpy.uint32).itemsize)

	# two step scan
	segscan(scanlist, metasl, irows, rows, pags, block=blockss, grid=gridss, shared=shares)
	metascan(scanlist, metasl, irows, rows, pags, block=blockss, grid=gridss)

	# initialize the target neighborhood tensor-- for this we need to predict
	# the interesting row count, but we still need the earlier interesting row
	# count, too.
	shrinkrows=irows.copy()
	irowset(scanlist, shrinkrows, pags, rows,block=blockis,grid=gridis)
	newrows=numpy.uint32(gpuarray.max(shrinkrows).get()/1)
	shrinkrows=0

	# make sure the new size won't run over memory constraints
	warnstring='need ' + str(newrows)+' rows in the shrunken neighborhood tensor. hint: this means the search space is more dense than anticipated and PTshrink is protecting us from possibly exceeding RAM on the GPU.'
	assert not (newrows>i2 and i2 !=0), warnstring

	# scatter the entries in *gnb*
	target=gpuarray.zeros((pags,newrows,3),dtype=numpy.float32)
	scatter(scanlist, target, gnb, irows, pags, rows, newrows,block=blockps,grid=gridps)

	# get new interesting region sizes
	irowset(scanlist, irows, pags, rows,block=blockis,grid=gridis)

	return target, irows




#-------------------------------------------------------------------------------

def ngrab(ssp,qse,val):
	# g mills 22/5/14
	# pulls the neighborhoods of points in a query set from the accompanying
	# search space, puts them in a tensor, and centers them.

	# each page of the output tensor is a point cloud. rows and cols in shape
	# of *inc*, number of pages equal to number of points in the query set.


	# INPUT
	# all gpuarrays of floats.
	# ssp = search space point cloud
	# qse = query set point cloud
	# val = search radius

	# PARAMETERS
	pags=numpy.uint32(qse.shape[0])		# each output page is a neighborhood
	rows=numpy.uint32(ssp.shape[0]+1)	# each page has each possible search 
											# space point
	blockx=int(16)			# don't make it too big since we're using shared mem
	blocky=int(16)			# and also if it changes here then change shared 
								# declaration
	gridx=int(numpy.ceil(rows/blockx))	# grid over output tensor
	gridy=int(numpy.ceil(pags/blocky))	# each thread spans the col dimension

	# OUTPUT
	outc = gpuarray.GPUArray((pags,rows,3),dtype=numpy.float32)	# 3 tensor
	#  wherein each page is a neighborhood point cloud.


	mod=SourceModule("""

	__global__ void ngrab( float *ssp, float *qse, float val, float *outc, int rows, int pags)
	{
		// generates a neighborhood 3-tensor from a query set, search space,
		// and search radius. at the head of each neighborhood is its
		// geometrical center coordinate in full pointcloud coordinate system--
		// i.e. the query set point the neighborhood is centered on.

		// thread coordinates
		int tx = blockDim.x*blockIdx.x+threadIdx.x;
		int tz = blockDim.y*blockIdx.y+threadIdx.y;

		// shared mem coordinates
		int sx = threadIdx.x*3;
		int sz = threadIdx.y*3;

		// declare shared memory for search space access
		__shared__ float sspmat[48];	// 3 floats per thread on a side of the block
		__shared__ float qsemat[48];	// same

		// output write position-- tx+1 accounts for the first thread in the
		// neighborhood writing the query set point in addition to copying
		// the first point in the search space.
		int ow = tz*3*rows+(tx+1)*3;

		// work flag
		int wf = 0;

		// make sure thread is in output domain-- again, note rows-1
		if ( (tx<rows-1) && (tz<pags) ) { wf = 1;}

		// load search space point into shared memory if on first page
		// covered by block
		if ( (wf==1) & (threadIdx.y==0) )
		{
			sspmat[sx]=ssp[tx*3];
			sspmat[sx+1]=ssp[tx*3+1];
			sspmat[sx+2]=ssp[tx*3+2];
		}

		// load query set point into shared memory if on first row
		// covered by block
		if ( (wf==1) && (threadIdx.x==0) )
		{
			qsemat[sz]=qse[tz*3];
			qsemat[sz+1]=qse[tz*3+1];
			qsemat[sz+2]=qse[tz*3+2];
		}

		__syncthreads();

		// if this is the first thread in the neighborhood, write the search
		// set point. this will access the first row of the neighborhood page.
		if ( (wf==1) && (tx==0) )
		{
			outc[ow-3]=qsemat[sz];
			outc[ow-2]=qsemat[sz+1];
			outc[ow-1]=qsemat[sz+2];
		}

		// calculate the euclidian distance of the point from the center.
		if (wf==1)
		{

			// calculate distance from this point to the center
			float a = fabsf(__fadd_rn(sspmat[sx],-qsemat[sz]));
			float b = fabsf(__fadd_rn(sspmat[sx+1],-qsemat[sz+1]));
			float c = fabsf(__fadd_rn(sspmat[sx+2],-qsemat[sz+2]));
			float d=__fsqrt_rn(__fmaf_rn(a,a,__fmaf_rn(b,b,__fmul_rn(c,c))));

			// compare distance to search radius
			if (d<val)
			{
				// subtract center and write the search space entry to output
				outc[ow]=sspmat[sx]-qsemat[sz];
				outc[ow+1]=sspmat[sx+1]-qsemat[sz+1];
				outc[ow+2]=sspmat[sx+2]-qsemat[sz+2];
			}
			else
			{
				// write a zero.
				outc[ow]=0.;
				outc[ow+1]=0.;
				outc[ow+2]=0.;
			}
		}
	}

	""")

	ng=mod.get_function('ngrab')

	ng(ssp,qse,val,outc,rows,pags,block=(blockx,blocky,1),grid=(gridx,gridy))

	return outc



#-------------------------------------------------------------------------------

def cuvox(inc, edge):
	# g mills 20/5/14
	# voxelizes a point cloud. useful as a point density equalizer.
	# returns a unique list of voxel center points.

	# INPUT
	# inc = input point cloud on the host. numpy array.
	# edge = edge length of a voxel.
	edge=numpy.float32(edge)

	# PARAMETERS
	rows=numpy.uint32(inc.shape[0])
	blockx=512		# one dimensional blocks
	gridx=int(numpy.ceil(rows/blockx))

	# OUTPUT
	# outc = unique list of voxel center coordinates in input coordinate system.


	# get the minimum corner of *inc*
	corner=numpy.float32(inc.min(0))

	# get the physical dimensions of the input cloud
	xmin,ymin,zmin=inc.min(0)
	xmax,ymax,zmax=inc.max(0)
	span=numpy.array([xmax-xmin,ymax-ymin,zmax-zmin])
	spandim=span.max()		# this is the largest dimension of the point cloud

	# convert the span to voxel coordinates
	spandim=2**(numpy.ceil(numpy.log2(spandim/edge)))

	# check that we're not exceeding the descriptive capacity of a 32 bit 
	# unsigned int. we have a total of 2^32 voxels which can possibly be 
	# represented. assuming that the point cloud completely fills a cubic space,
	# we can represent up to 2^10 along each dimension, or 1024 voxels on an 
	# edge of the volume.
	if spandim>1024:
		print(str(spandim)+" voxels on a side. that's too many, unfortunately")
		return 0


	# voxel coordinates
	vc=gpuarray.zeros((rows,1),dtype=numpy.uint32)

	# kernel- calculate the number of voxel edges each point is from the corner
	# as a uint and represent as a bitstring-- remember <1024 voxels on a side
	mod=SourceModule("""
	__global__ void vox(float *inc, int *voxcoord, float xm, float ym, float zm, float edge, int rows)
	{
		// input row
		int ty=blockDim.x*blockIdx.x+threadIdx.x;

		// make sure thread is in domain
		if (ty<rows)
		{
			// get point coordinates
			float x = inc[ty*3];
			float y = inc[ty*3+1];
			float z = inc[ty*3+2];

			// calculate distance from corner
			uint xc = __float2uint_rn(__fdiv_rn(__fadd_rn(x,-xm),edge));
			uint yc = __float2uint_rn(__fdiv_rn(__fadd_rn(y,-ym),edge));
			uint zc = __float2uint_rn(__fdiv_rn(__fadd_rn(z,-zm),edge));

			// intercalate the array coordinates via bitwise operators.
			uint arc=(xc<<20)|(yc<<10)|zc;

			voxcoord[ty]=arc;
		}
	}

	__global__ void devox(int *voxcoord, float *outc, float xm, float ym, float zm, float edge, int rows)
	{
		// input row
		int ty=blockDim.x*blockIdx.x+threadIdx.x;

		// make sure thread is in domain
		if (ty<rows)
		{
			// get intercalated coordinate string
			uint vc=voxcoord[ty];

			// decalate the array coordinates using bitmasks
			float xc = (vc & 1072693248)>>20;
			float yc = (vc & 1047552)>>10;
			float zc = (vc & 1023);

			// calculate original coordinates and write back
			outc[ty*3] = __fmaf_rn(xc,edge,xm);
			outc[ty*3+1] = __fmaf_rn(yc,edge,ym);
			outc[ty*3+2] = __fmaf_rn(zc,edge,zm);
		}
	}



	""")

	vox=mod.get_function('vox')
	devox=mod.get_function('devox')

	# send *inc* to the gpu
	ginc=gpuarray.to_gpu(inc.astype(numpy.float32))

	# voxelize
	vox(ginc,vc,corner[0],corner[1],corner[2],edge,rows,block=(blockx,1,1),grid=(gridx,1))

	# get the voxel coordinate list back
	vc=vc.get()

	# unique it and back to gpu
	vc=gpuarray.to_gpu(numpy.unique(vc).astype(numpy.uint32))

	# get new size
	rows=numpy.uint32(vc.size)
	gridx=int(numpy.ceil(rows/blockx))

	# build output point cloud
	outc=gpuarray.zeros((rows,3),dtype=numpy.float32)

	devox(vc,outc,corner[0],corner[1],corner[2],edge,rows,block=(blockx,1,1),grid=(gridx,1))

	return outc
	
	

#-------------------------------------------------------------------------------

def cu_natural_vox(inc, edge):
	# g mills 19/1/15
	# "natural" voxel pattern-- returns the index of the first point in each 
	# voxel, instead of the coordinates of the voxel center.
	
	# NOTE ndarray in, ndarray out
	
	# INPUT
	# inc = input point cloud- on the host. numpy array. not a gpuarray.
	# edge = edge length of a voxel. NOT a cubic radius. cubic diameter. 
	edge=numpy.float32(edge)
	
	# PARAMETERS
	rows=numpy.uint32(inc.shape[0])
	blockx=512		# one dimensional blocks
	gridx=int(numpy.ceil(rows/blockx))
	
	# OUTPUT
	# vci = indices; one point per voxel 
	
	
	# get the minimum corner of *inc*
	corner=numpy.float32(inc.min(0))

	# get the physical dimensions of the input cloud
	xmin,ymin,zmin=inc.min(0)
	xmax,ymax,zmax=inc.max(0)	
	span=numpy.array([xmax-xmin,ymax-ymin,zmax-zmin])
	spandim=span.max()		# this is the largest dimension of the point cloud

	# convert the span to voxel coordinates
	spandim=2**(numpy.ceil(numpy.log2(spandim/edge)))
	
	# check that we're not exceeding the descriptive capacity of a 32 bit 
	# unsigned int. we have a total of 2^32 voxels which can possibly be 
	# represented. assuming that the point cloud completely fills a cubic space,
	# we can represent up to 2^10 along each dimension, or 1024 voxels on an 
	# edge of the volume.
	if spandim>1024:
		print(str(spandim)+" voxels on a side. that's too many, unfortunately")
		return 0

	
	# voxel coordinates
	vc=gpuarray.zeros((rows,1),dtype=numpy.uint32)
	
	# kernel- calculate the number of voxel edges each point is from the corner 
	# as a uint and represent as a bitstring-- remember <1024 voxels on a side
	mod=SourceModule("""	
	__global__ void vox(float *inc, int *voxcoord, float xm, float ym, float zm, float edge, int rows)
	{
		// input row
		int ty=blockDim.x*blockIdx.x+threadIdx.x;
		
		// make sure thread is in domain
		if (ty<rows)
		{
			// get point coordinates
			float x = inc[ty*3];
			float y = inc[ty*3+1];
			float z = inc[ty*3+2];
			
			// calculate distance from corner
			uint xc = __float2uint_rn(__fdiv_rn(__fadd_rn(x,-xm),edge));
			uint yc = __float2uint_rn(__fdiv_rn(__fadd_rn(y,-ym),edge));
			uint zc = __float2uint_rn(__fdiv_rn(__fadd_rn(z,-zm),edge));
			
			// intercalate the array coordinates via bitwise operators.
			uint arc=(xc<<20)|(yc<<10)|zc;
			
			voxcoord[ty]=arc; 
		}
	}
	
	""")
	
	vox=mod.get_function('vox')
	
	# send *inc* to the gpu
	ginc=gpuarray.to_gpu(inc.astype(numpy.float32))
	
	# voxelize
	vox(ginc,vc,corner[0],corner[1],corner[2],edge,rows,block=(blockx,1,1),grid=(gridx,1))
	
	# get the voxel coordinate list back
	vc=vc.get()
	
	# get the indices of the first uniques
	ignore,vci=numpy.unique(vc,return_index=True)
	
	return vci

#-------------------------------------------------------------------------------

def gpu_rigid_tree(inc,minpoints,maxrad,cen,rad,partlist,roll):
	# g mills 3/2/15
	# octree on the gpu. another solution to partitioning very large (>30mil) point clouds.
	# this version outputs partitions of uniform size. useful for voxel filter setup. buffer
	# functionality is disabled.
	
	# INPUT
	# inc = input point cloud, gpuarray (needs to be ndarray on first call)
	# minpoints = minimum number of points in a partition
	# maxrad = maximum radius of a partition-- all output partitions will be this size.
	# cen = center of partition in consideration-- enter 0 on first call
	# rad = radius of partition in consideration-- enter 0 on first call
	# partlist = ndarray of partition centers and radii-- enter 0 on first call
	# roll = gpuarray of binary membership markers-- enter 0 on first call
	
	# PARAMETERS
	rows=numpy.uint32(inc.shape[0])
	blockx=int(256)
	gridx=int(numpy.ceil(rows/blockx))
	cen=numpy.float32(cen)
	rad=numpy.float32(rad)
	
	# OUTPUT
	# partlist = center x,y,z,radius,point count
	
	
	# check if this is first call-- note that depending on how this is used, any, all or none of 
	# these may be calculated externally.
	if not isinstance(cen,numpy.ndarray):
		cen=inc.mean(0)
	if not isinstance(partlist,numpy.ndarray):
		partlist=numpy.zeros((0,5))
	if not isinstance(roll,gpuarray.GPUArray):
		roll=gpuarray.zeros(rows,dtype=numpy.float32)
	if rad==0:
		imi=numpy.abs(cen-inc.min(0)).max()
		ima=numpy.abs(cen-inc.max(0)).max()
		rad=max(imi,ima)		
	# check if *inc* already on gpu for some reason
	if not isinstance(inc,gpuarray.GPUArray):
		inc=gpuarray.to_gpu(inc.astype(numpy.float32))	

	# modified version of cu_query_neighborhood kernel	
	mod=SourceModule("""
	
		
	__global__ void QNC(float *inc, float *outc, float x, float y, float z, float rad, int rows)
	{
		// if a given point in *inc* is within *rad* of *x,y,z*, write a 1 to *outc*.
		// chebyshev distance metric.
		
		// coords
		int tx = blockIdx.x*blockDim.x+threadIdx.x;
		
		// in bounds check
		if (tx<rows)
		{
			// get the coordinate spans
			float a = fabsf(inc[3*tx]-x);
			float b = fabsf(inc[3*tx+1]-y);
			float c = fabsf(inc[3*tx+2]-z);

			// find the maximum coordinate span
			float mcs = fmaxf(a,fmaxf(b,c));
			
			
			// write back if necessary
			if (mcs<rad)
			{
				outc[tx]=1.f;
			}
			
		}
	}
	
	
	""")

	func=mod.get_function('QNC')
	
	# halve the bounding radius to get the partition radius
	prad=rad/2	
	
	# find the bounding box face coordinates:
	smin=cen-rad
	smax=cen+rad
	
	yr=numpy.array([-1,1])
	ssc=numpy.asarray([[x,y,z] for x in yr for y in yr for z in yr])
	
	ssc=(ssc*prad+cen).astype(numpy.float32)

	for s in ssc:
		func(inc,roll,s[0],s[1],s[2],numpy.float32(prad),rows,block=(blockx,1,1),grid=(gridx,1,1))
		pop=gpuarray.sum(roll).get()/1
		#print(pop)
		roll.fill(numpy.float32(0))	
		
		# this partition is a keeper if the partition radius is under *maxrad* and there's
		# enough points left.
		if pop>minpoints and prad<maxrad:
			partlist=numpy.vstack((partlist,numpy.hstack((s,prad,pop))))
		
		# otherwise we recurse
		elif pop>minpoints:
			partlist=gpu_rigid_tree(inc,minpoints,maxrad,s,prad,partlist,roll)		
			
	return partlist
	

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def cu_query_neighborhood(inc,cen,rad,measure):
	# g mills 20/10/14
	# finds all points in given point cloud within given range of given point.
	# useful for very large point clouds. note this function doesn't work for
	# point clouds with more than ~4.294 billion entries because that maxes out
	# the uint32.


	# INPUT
	# inc = input point cloud, float32 gpuarray, nx3
	# cen = target point
	# rad = neighborhood radius
	# measure = 'euclid' or 'cheby'

	# PARAMETERS
	rows=numpy.uint32(inc.shape[0])
	blockx=int(256)
	gridx=int(numpy.ceil(rows/blockx))
	cen=numpy.float32(cen)
	rad=numpy.float32(rad)
	umax=numpy.uint32(4294967295)

	# OUTPUT
	# outc = unique idx array to the points in *inc* within *rad* of *cen*
	outc=gpuarray.GPUArray(rows,dtype=numpy.uint32)
	outc.fill(umax)



	mod=SourceModule("""

	__global__ void query_neighborhood_cheby(float *inc, int *outc, float x, float y, float z, float rad, int rows)
	{
		// if a given point in *inc* is within *rad* of *x,y,z*, write its row
		// to *outc*.
		// chebyshev distance metric.

		// coords
		int tx = blockIdx.x*blockDim.x+threadIdx.x;

		// in bounds check
		if (tx<rows)
		{
			// get the coordinate spans
			float a = fabsf(inc[3*tx]-x);
			float b = fabsf(inc[3*tx+1]-y);
			float c = fabsf(inc[3*tx+2]-z);

			// find the maximum coordinate span
			float mcs = fmaxf(a,fmaxf(b,c));


			// write back if necessary
			if (mcs<rad)
			{
				outc[tx]=tx;
			}

		}
	}

	__global__ void query_neighborhood_euclid(float *inc, int *outc, float x, float y, float z, float rad, int rows)
	{
		// if a given point in *inc* is within *rad* of *x,y,z*, write its row
		// to *outc*.
		// euclidian distance metric.

		// coords
		int tx = blockIdx.x*blockDim.x+threadIdx.x;

		// in bounds check
		if (tx<rows)
		{
			// get the coordinate spans
			float a = fabsf(inc[3*tx]-x);
			float b = fabsf(inc[3*tx+1]-y);
			float c = fabsf(inc[3*tx+2]-z);

			// find the euclidian norm
			float norm = __fsqrt_rn(__fmaf_rn(a,a,__fmaf_rn(b,b,__fmul_rn(c,c))));

			// write back if necessary
			if (norm<rad)
			{
				outc[tx]=tx;
			}
		}
	}


	""")

	# decide on the kernel we need
	if measure=='cheby':
		func=mod.get_function('query_neighborhood_cheby')
	elif measure=='euclid':
		func=mod.get_function('query_neighborhood_euclid')
	else:
		print("sorry, only euclidian and chebyshev distance metrics are supported at this point")
		return

	func(inc,outc,cen[0],cen[1],cen[2],rad,rows,block=(blockx,1,1),grid=(gridx,1,1))

	# take the list back, unique it, discard the null value
	outc=numpy.unique(outc.get())[:-1]

	return outc


#-------------------------------------------------------------------------------

def gpu_tree(inc,maxpoints,buffer,minrad,cen,rad,partlist,roll):
	# g mills 21/10/14
	# octree on the gpu. for partitioning large (>30mil) point clouds.

	# INPUT
	# inc = input point cloud, gpuarray (needs to be ndarray on first call)
	# maxpoints = max number of points in a partition
	# buffer = buffer distance: base the partition size on the population of a
	# slightly larger concentric partition
	# minrad = minimum radius of a partition
	# cen = center of partition in consideration-- enter 0 on first call
	# rad = radius of partition in consideration-- enter 0 on first call
	# partlist = ndarray of partition centers and radii-- enter 0 on first call
	# roll = gpuarray of binary membership markers-- enter 0 on first call

	# PARAMETERS
	minpoints=100		# minimum partition population
	rows=numpy.uint32(inc.shape[0])
	blockx=int(256)
	gridx=int(numpy.ceil(rows/blockx))
	cen=numpy.float32(cen)
	rad=numpy.float32(rad)
	buffer=numpy.float32(buffer)

	# OUTPUT
	# partlist = center x,y,z,radius,point count


	# check if this is first call-- note that depending on how this is used,
	# any, all or none of these may be calculated externally.
	if not isinstance(cen,numpy.ndarray):
		cen=inc.mean(0)
	if not isinstance(partlist,numpy.ndarray):
		partlist=numpy.zeros((0,5))
	if not isinstance(roll,gpuarray.GPUArray):
		roll=gpuarray.zeros(rows,dtype=numpy.float32)
	if rad==0:
		imi=numpy.abs(cen-inc.min(0)).max()
		ima=numpy.abs(cen-inc.max(0)).max()
		rad=max(imi,ima)
	# check if *inc* already on gpu
	if not isinstance(inc,gpuarray.GPUArray):
		inc=gpuarray.to_gpu(inc.astype(numpy.float32))

	# modified version of cu_query_neighborhood kernel
	mod=SourceModule("""


	__global__ void QNC(float *inc, float *outc, float x, float y, float z, float rad, int rows)
	{
		// if a given point in *inc* is within *rad* of *x,y,z*, write a 1 to *outc*.
		// chebyshev distance metric.

		// coords
		int tx = blockIdx.x*blockDim.x+threadIdx.x;

		// in bounds check
		if (tx<rows)
		{
			// get the coordinate spans
			float a = fabsf(inc[3*tx]-x);
			float b = fabsf(inc[3*tx+1]-y);
			float c = fabsf(inc[3*tx+2]-z);

			// find the maximum coordinate span
			float mcs = fmaxf(a,fmaxf(b,c));

			// write back if necessary
			if (mcs<rad)
			{
				outc[tx]=1.f;
			}

		}
	}


	""")

	func=mod.get_function('QNC')

	# halve the bounding radius to get the partition radius
	prad=rad/2

	# find the bounding box face coordinates:
	smin=cen-rad
	smax=cen+rad

	yr=numpy.array([-1,1])
	ssc=numpy.asarray([[x,y,z] for x in yr for y in yr for z in yr])

	ssc=(ssc*prad+cen).astype(numpy.float32)

	for s in ssc:
		func(inc,roll,s[0],s[1],s[2],numpy.float32(prad+buffer),rows,block=(blockx,1,1),grid=(gridx,1,1))
		pop=gpuarray.sum(roll).get()/1
		roll.fill(numpy.float32(0))

		# this partition is a keeper if the pop is low enough or the next
		# partition radius is too small
		if pop>minpoints and (pop<=maxpoints or prad/2<minrad):
			if buffer>0:
				# if the buffer is nonzero, then the actual cube we're looking
				# for could still be empty. check on that.
				func(inc,roll,s[0],s[1],s[2],numpy.float32(prad),rows,block=(blockx,1,1),grid=(gridx,1,1))
				innerpop=gpuarray.sum(roll).get()/1
				roll.fill(numpy.float32(0))
				if innerpop>minpoints:
					partlist=numpy.vstack((partlist,numpy.hstack((s,prad,pop))))
			else:
				partlist=numpy.vstack((partlist,numpy.hstack((s,prad,pop))))


		# otherwise we recurse
		elif pop>minpoints:
			partlist=gpu_tree(inc,maxpoints,buffer,minrad,s,prad,partlist,roll)

	return partlist



