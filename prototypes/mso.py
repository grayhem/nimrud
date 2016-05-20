# multiscale feature generation algorithms

import numpy
import pycuda.gpuarray as gpua
import ch
import time
from numpy.linalg import eigh as eig


#-------------------------------------------------------------------------------

def V_MSO(qse,qseidx,ssp,sspvec,sspedge,scales,imax=20000):
	# g mills 10/12/14
	# vector multiscale operator processing chain handler. takes vectors of
	# float values for each point, and returns for each scale the mean vector
	# of the neighborhood. 
	
	# INPUT
	# qse = query set point cloud 
	# qseidx = indices of query set points in original point cloud
	# ssp = search space point cloud
	# sspvec = set of cluster distance vectors for ssp
	# sspedge = edge length for search space voxelization. use 0 to skip vox. 
	# scales = numpy array of spherical neighborhood radii to use
	# imax = maximum number of points in a search space partition
	
	# FIRST: put scales in descending order and make float32
	scales=numpy.float32(numpy.sort(scales)[::-1])
	
	# PARAMETERS
	leaf=10000
	#imax=20000			# maximum number of points in a search space partition
	qsebite=17000		# maximum number of query set points to process at once
	minrad=scales[-1]	# absolute minimum radius of a query set partition
	ominrad=minrad*3	# minimum radius of a query set partition in octree
	buffer=scales[0]	# difference between query set and search space radii
	ivt = 10			# ignore voxel threshold- number of points needed in a
						# partition in order to justify work on it
	imeasure='cheby'	# distance metric for search space feature interpolation
	vdim=sspvec.shape[1]	# feature vector dimension
	
	# OUTPUT
	# outc = indices to point cloud (query set points) with multiscale vectors
	# appended. scales in descending order. points in no guaranteed order.
	# [IDX, mean (vdim*numscales)]
	outc=numpy.zeros((0,1+scales.size*vdim),dtype=numpy.float32)
	
	
	# memento mori	
	alltime=time.time()
		
	# manually set types
	ssp=ssp.astype(numpy.float32)
	sspvec=sspvec.astype(numpy.float32)
	qse=qse.astype(numpy.float32)
	
	# voxelize the search space point cloud
	if sspedge!=0:
		sspv=double_vox(ssp,sspedge).astype(numpy.float32)		
		# interpolate the vector field to the search space
		ssp,sspvec=vec_field_interp(sspv,ssp,sspvec,sspedge,imeasure)
	
	
	# partition the search space
	partset = Partitions(ssp,imax,sspedge,ominrad,minrad,ivt)
	# put the qse and ssp on the gpu for faster partitioning
	g_qse=gpua.to_gpu(qse)
	g_ssp=gpua.to_gpu(ssp)
	
	# iterate over the set of partitions
	for qse_mask, ssp_mask in partset.partition_generator(g_qse,g_ssp):
		# make sure this volume should be processed
		if qse_mask.sum() > ivt:
		
			# get the subset of query set points we need on this partition
			lqse=numpy.compress(qse_mask,qse,axis=0)
			
			# and search space with features
			lssp=numpy.compress(ssp_mask,ssp,axis=0)
			lsspvec=numpy.compress(ssp_mask,sspvec,axis=0)
			# get indices of query set points in the original point cloud 
			oIDX=numpy.extract(qse_mask,qseidx)
			
			# process the points
			soutc=V_MSO_process(lssp,lqse,oIDX,lsspvec,scales,qsebite)
			
			# concatenate to output
			outc=numpy.vstack((outc,soutc))			
		
			
	finaltime=time.time()-alltime
	finalpoints=outc.shape[0]
	pointsec=finalpoints/finaltime
	print( 'total time in vmso at ' + str(sspedge) + 'm voxel edge length: ' + str(int(finaltime)) + 's')
	print( str(finalpoints) + ' points processed at an overall rate of ' + str(int(pointsec)) + ' points per second')
	
	
	return outc
	

#-------------------------------------------------------------------------------

def V_MSO_process(ssp,qse,qseidx,sspfeats,scales,qsebite):
	# g mills 12/10/14
	# processing pipeline for vector multiscale operator: mean of each dim of 
	# the feature vector attached to each search space point
	
	# INPUT
	# ssp = search space point cloud
	# qse = query set point cloud 
	# qseidx = indices of points in qse
	# sspfeats = feature vectors for ssp
	# scales = analysis scales: descending order, float 32
	# qsebite = max number of query set points to process at once
	
	# PARAMETERS
	eps = numpy.spacing(1)	# tiny number to protect against division by zero
	ns = scales.size
	nf = sspfeats.shape[1]
	ne = qse.shape[0]
	measure = 'euclid'		# not like we're going to use chebyshev distance here
		
	# OUTPUT
	# outc = indices point cloud (query set points) with multiscale vectors
	# appended. scales in descending order. mean only.
	# [idx, mean (feats*scales)]
	outc=numpy.zeros((0,1+ns*nf),dtype=numpy.float32)
	
	
	# transpose the features and make sure they're C-contiguous
	sspfeats=sspfeats.T.astype(numpy.float32,order='C')
	
	# put search space and feats on GPU
	gssp=gpua.to_gpu(ssp.astype(numpy.float32))
	gsspfeats=gpua.to_gpu(sspfeats)

	# loop over chunks of query set
	for d in range(0,ne,qsebite):
	
		# load query set chunk
		lqse=gpua.to_gpu(qse[d:d+qsebite])
		
		# initialize output chunk
		soutc=numpy.zeros((lqse.shape[0],1+ns*nf),dtype=numpy.float32)
		
		# slot in the indices
		soutc[:,0]=qseidx[d:d+qsebite]
		# build the distance matrix
		dmat=ch.make_dmat(lqse,gssp,measure)
					
		# loop over scales
		for s in enumerate(scales):
			# calculate the mean feature vector of each query set point's
			# neighborhood
			mean=ch.dmat_mean(dmat,gsspfeats,s[1])						
			# find the starting index in output wherein the values slot
			ms=1+s[0]*nf			
			# slot em in
			soutc[:,ms:ms+nf]=mean.get()
			
			
		# purge the query set from the GPU
		dmat.gpudata.free()
		lqse.gpudata.free()
		
		# concatenate output chunk to *outc*
		outc=numpy.vstack((outc,soutc))
					
	# just to be safe
	gssp.gpudata.free()
	gsspfeats.gpudata.free()
	
	return outc


#-------------------------------------------------------------------------------

def vec_field_interp(qse,ssp,sspfeats,qseedge,measure):
	# g mills 12/10/14
	# interpolate a vector field represented as a point cloud to a voxel grid.
	
	
	# INPUT
	# qse = query set: in this case the voxel set
	# ssp = coordinates of observations in...
	# sspfeats = set of feature vectors
	# qseedge = voxel edge length
	# measure = 'euclid' or 'cheby'
	
	# FIRST OFF:
	# make sure the feats are row vectors, even if they're scalars. NxM.
	sspfeats=sspfeats.reshape(ssp.shape[0],-1).astype(numpy.float32)
	# we also need to make sure they're all float32
	qse=qse.astype(numpy.float32)
	ssp=ssp.astype(numpy.float32)
	ivt = 10	# ignore voxel threshold
	
	
	# PARAMETERS
	leaf = 100000
	imax = 30000		# maximum number of points in a search space partition
	qsebite = 10000		# maximum number of query set points to search at a time
	minrad=qseedge*2
	ominrad=minrad*3
	
	# OUTPUT
	# outqse = RE-ORDERED coordinates of observations in...
	# qsefeats = set of feature vectors 
	outqse=numpy.zeros((0,3),dtype=numpy.float32)
	qsefeats=numpy.zeros((0,sspfeats.shape[1]))
	
		
	# interpolate input features to query set points-- note we're doing this on
	# a separate partition array from V_MSO because the search space/ query
	# set designation is flipped from the main process and the partitions would
	# be the wrong size if we did them together.
	
	# partition the search space
	partset = Partitions(ssp,imax,qseedge,ominrad,minrad,ivt)
	# put the qse and ssp on the gpu for faster partitioning
	g_qse=gpua.to_gpu(qse)
	g_ssp=gpua.to_gpu(ssp)
	
	# iterate over the set of partitions
	for qse_mask, ssp_mask in partset.partition_generator(g_qse,g_ssp):
	
		# check for sufficient membership
		if qse_mask.sum()>=ivt:
	
			# get the partitions and features
			gsspfeats=gpua.to_gpu(numpy.compress(ssp_mask,sspfeats,axis=0).T.astype(numpy.float32,order='C'))
			gpssp=gpua.to_gpu(numpy.compress(ssp_mask,ssp,axis=0))
			
			# we need to process smaller pieces of the query set at once, which
			# we can't do all on gpu.
			lqse=numpy.compress(qse_mask,qse,axis=0)
			
			# preserve the ordering of query set points and features as they go
			# through the blender 
			outqse=numpy.vstack((outqse,lqse))
			
			# break off chunks of query set in this partition
			for b in range(0,lqse.shape[0],qsebite):						
				# put on gpu
				gpqse=gpua.to_gpu(lqse[b:b+qsebite].copy())
				# interpolate
				featchunk=ch.vox_vf_interp(gpssp,gsspfeats,gpqse,qseedge,measure).get()
				# concatenate to output
				qsefeats=numpy.vstack((qsefeats,featchunk))
				
	
	# clean up 		
	gpqse=0
	gpssp=0
	gsspfeats=0
			
	return outqse,qsefeats



#-------------------------------------------------------------------------------

def G_MSO(qse,ssp,sspedge,scales,imax=20000):
	# g mills 30/9/14
	# first order (pure geometry) multiscale operator processing chain. 
	# voxelizes and processes MSOs for input point cloud at given scales, then 
	# returns feature vectors with their indices in the original point cloud.
	
	# 29/8/15 modification: outgoing features are grouped by scale.
	
	# FIRST: put scales in descending order and make float32
	scales=numpy.float32(numpy.sort(scales)[::-1])
	
	# INPUT
	# qse = query set
	# ssp = search space
	# sspedge = edge length for search space voxelization. 0 skips subsampling.
	# scales = numpy array of spherical neighborhood radii to use
	# imax= maximum number of points in a ssp partition
	
	# PARAMETERS
	inrows=qse.shape[0]
	#imax=20000			# maximum number of points in a ssp partition
	minrad=scales[-1]	# absolute minimum radius of a qse partition
	ominrad=minrad*3	# minimum radius of a query set partition in octree
	buffer=scales[0]	# size difference between query set and search space rad
	ivt = 10			# ignore voxel threshold- number of points needed in a 
						# search space partition in order to justify work on it
	pdir='nbtemp/nb'	# directory name and prefix for temp storage of tensors
	pidx=0				# starting save file index
	
	# OUTPUT
	# outc = point cloud (query set points) with multiscale vectors appended. 
	# scales in descending order. points in no guaranteed order.
	# [index, (density, centroid, eigval x2) x num scales]
	outc=numpy.zeros((0,1+scales.size*4),dtype=numpy.float32)
	
				
	# some timers		
	alltime=time.time()
	
	# voxelize the search space point cloud if necessary
	if sspedge!=0:
		v=time.time()
		ssp=double_vox(ssp,sspedge).astype(numpy.float32)	
		v2=time.time()-v
	else:
		ssp=ssp.astype(numpy.float32)
					
	# query set won't be voxelized since we're returning indices in original
	# point cloud.
	qse=qse.astype(numpy.float32)
	# index the query set
	all_qse_index=numpy.arange(qse.shape[0])
	
	# partition the search space
	partset = Partitions(ssp,imax,buffer,ominrad,minrad,ivt)
	# put the qse and ssp on the gpu for faster partitioning
	g_qse=gpua.to_gpu(qse)
	g_ssp=gpua.to_gpu(ssp)
	
	# iterate over the set of partitions
	for qse_mask, ssp_mask in partset.partition_generator(g_qse,g_ssp):
		pt=time.time()
		
		# reset the storage file index
		pidx=0					
			
		# make sure this volume should be processed
		if qse_mask.sum()>ivt:
		
			# get the search space and query set partitions
			lqse = numpy.compress(qse_mask,qse,axis=0)
			lssp = numpy.compress(ssp_mask,ssp,axis=0)
			# pass the partitions to the neighborhood construction pipeline
			nb,pidx=NB_build(lqse,lssp,buffer,pdir,pidx)
			# retrieve the indices associated with the qse points 
			qseidx=numpy.extract(qse_mask,all_qse_index)
									
			# loop over all the tensors in the temp directory if necessary
			if pidx:
				for pd in range(pidx):	
					nb=numpy.load(pdir+str(pd)+'.npy')
					# carve off the first however many query set indices
					uqseidx=qseidx[:nb.shape[0]]
					qseidx=qseidx[nb.shape[0]:]
					outvec=NB_process(nb,scales,uqseidx)
					outc=numpy.vstack((outc,outvec))
					nb=0	
			else:				
				# pass the only neighborhood to the process pipeline
				outvec=NB_process(nb,scales,qseidx)
				outc=numpy.vstack((outc,outvec))
				nb=0	
		
		partime=(time.time()-pt)
				
			
	finaltime=time.time()-alltime
	finalpoints=outc.shape[0]
	pointsec=finalpoints/finaltime
	print( 'total time in gmso at ' + str(sspedge) + 'm voxel edge length: ' + str(int(finaltime)) + 's')
	print( str(finalpoints) + ' points processed at an overall rate of ' + str(int(pointsec)) + ' points per second')
	
	
	return outc






#-------------------------------------------------------------------------------

def NB_build(qse,ssp,rad,pdir,pidx):
	# g mills 30/9/14-- original version 13/6/14
	# associates a query set point cloud with neighbors in search space,
	# returning a neighborhood tensor with associated query set points at the
	# head of each page in the tensor. if the tensor grows large enough to 
	# crowd system RAM, we'll write it to the HDD in chunks and process them
	# individually.

	# INPUT
	# qse = query set, on the host
	# ssp = search space, on the host
	# rad = starting neighborhood search radius. float32.
	# pdir = directory where we save the tensors for later processing-- 
			# including file prefix
	# pidx = temporary storage file index
	
	# PARAMETERS
	ikmax=60000000		# maximum size of number of points in search space * 
						# number of points in query set. beyond 60 mil, we risk
						# overfilling gpu ram on a 2gb board
	i=ssp.shape[0]					# num points in search space
	k=int(numpy.floor(ikmax/i))		# num points in query set-- per dwell cycle
	j=3								# point cloud dimensionality 
	pthresh=750000000	# when a tensor reaches this size, dump to the hdd. 3gb.

		
	# OUTPUT
	# a neighborhood tensor for the entire query set and search space given.
	ia = []	# this is just an intermediary list.
	# pidx = index following the last save file that was generated-- return 0 
	# if we didn't write to the hdd.
	
	# start a timer
	star=time.clock()
	
	# calculate the dwell
	dwell=int(numpy.ceil(qse.shape[0]/k))	# at least one loop
	if dwell==1:			# is it only one loop?
		k=qse.shape[0]		# if so, k is size of entire query set
		klast=k				# and identical to the last pass k value. 
	else:
		# last pass k value
		klast=qse.shape[0]%k		

	# instantiate a NBtensor object with the search space
	gen=NBtensor(ssp)
	
	# set up our first query set
	lqse=qse[:k,:]

	# loop over dwell and use *gen* to build neighborhoods
	for d in range(dwell):

		# build a neighborhood with this query set and add to the intermediary
		ia+=[gen.fill(lqse,rad)]
		
		# set up the next query set
		if d<dwell-1:		# get new query set if in the middle of the dwell
			lqse=qse[(d+1)*k:(d+2)*k,:]
			# if the intermediary array is getting too big then dump to hdd
			maxrows=max(x.shape[1] for x in ia)	
			pags=sum(x.shape[0] for x in ia)	
			if maxrows*pags*3>pthresh:	# time to dump it out
				print('saving contents of intermediarray to ' + pdir + str(pidx))
				onb=numpy.zeros((pags,maxrows,3),dtype=numpy.float32)
				for x in enumerate(ia):
					rows=x[1].shape[1]
					onb[k*x[0]:k*(x[0]+1),:rows,:]=x[1]
				# save the data
				numpy.save(pdir+str(pidx),onb)
				pidx+=1		# don't forget to increment the out index counter
				ia=[]		# and empty out the intermediary
			
		elif d==dwell-1:	# destroy all allocations if last or only pass
			gen.purge()
								
	# condense a single neighborhood tensor from the intermediary
	maxrows=max(x.shape[1] for x in ia)	
	pags=sum(x.shape[0] for x in ia)	# the number of pages
	onb=numpy.zeros((pags,maxrows,3),dtype=numpy.float32)
	for x in enumerate(ia):
		rows=x[1].shape[1]
		onb[k*x[0]:k*(x[0]+1),:rows,:]=x[1]

	
	# if we've written nbhds to the HDD, put the last one on there too and
	# return zero instead.
	if pidx!=0:	
		numpy.save(pdir+str(pidx),onb)
		pidx+=1		# don't forget to increment the out index counter
		onb=0
	return onb, pidx
	
	

#-------------------------------------------------------------------------------

def NB_process(inb,scales,qseidx):
	# g mills 30/9/14
	# processing pipeline for eigvals, density and centroid displacement.
	
	# INPUT
	# inb = input numpy 3-array
	# scales = list of analysis scales. should be descending order and float32 
	# qseidx = list of indices associated with the points we will be processing
	
	# PARAMETERS
	ikmax=50000000		# max value of i*k in the processing pipeline
	k = inb.shape[0]	# pages in bulk neighborhood/ query set points
	i = inb.shape[1]	# rows per page in same/ search space points
	eps=numpy.spacing(1)	# tiny number to protect against division by zero
	ns = scales.size
	rad=180/numpy.pi	
	conv=100*100*100	# 1 million cubic centimeters in a cubic meter.	
	ydimmax=65535		# this is the largest possible grid y-dim in CUDA. 
	# refactor *segscan* and we won't have to place this artificial limitation
	# on the chunk size here. see *PTshrink* comments for details. of course,
	# if we load TOO big a tensor we could risk a kernel hang.
		
	# OUTPUT
	# outc = point cloud (query set points) with multiscale vectors appended. 
	# scales in descending order. points in no guaranteed order.
	# [index, (density, centroid, eigval x2) x num scales]
	outc=numpy.zeros((0,1+ns*4),dtype=numpy.float32)
	
	
	# decide how to partition the tensor into manageable chunks
	kmax=min(ydimmax,int(numpy.floor(ikmax/i)))
	klast=k%kmax					# last chunk
	dwell=int(numpy.ceil(k/kmax))
	
	# loop over chunks
	for d in range(dwell):
		# initialize output chunk
		if inb.shape[1]>1:
			if d==dwell-1:	# if last pass
				# take the last or only piece
				gen=NBtensor(inb[-klast:,1:,:])	
				# strip those query set indices and compose to a col vector
				soutc=numpy.zeros((klast,1+ns*4),dtype=numpy.float32)
				soutc[:,0]=qseidx[-klast:]	

			else:			
				# take a piece from the front or middle
				gen=NBtensor(inb[kmax*d:kmax*(d+1),1:,:]) 
				# strip those query set indices and compose to a col vector
				soutc=numpy.zeros((kmax,1+ns*4),dtype=numpy.float32)
				soutc[:,0]=qseidx[kmax*d:kmax*(d+1)]
		
			# loop over scales
			for s in enumerate(scales):
				# calculate offset to starting column in feature block
				s_off=1+s[0]*4
				# drop neighborhood to this scale (no points should be dropped
				# on first pass)
				irows=gen.drop(s[1])				
				# calculate the volume of the neighborhood-- points per cm^3
				vol=conv*(4/3)*numpy.pi*s[1]**3
				# calculate the density of the neighborhoods
				soutc[:,s_off]=irows/vol
				# get the mean point displacements
				soutc[:,s_off+1]=gen.MP_displacement()	
				# get the write position for eigenvalues
				es=s_off+2
				# get the eigenvalues (GPU)
				soutc[:,es:es+2]=gen.MSPCA_eigs()				
			
			# purge all that stuff from the GPU
			gen.purge()
			# concatenate output chunk to *outc*
			outc=numpy.vstack((outc,soutc))
				
	
	return outc

	




#-------------------------------------------------------------------------------

class NBtensor(object):
	# g mills 23/6/14
	# object oriented solution for general work with multiscale neighborhood
	# tensors on the GPU. this is a dual purpose class: it can be used to create
	# a tensor from space and set, or can be initialized with a tensor and then
	# used to process on it. methods of this class may deal with MSPCA
	# processing or other chains. note we're allocating and deallocating a lot
	# of memory here. this is to make the process outside the object cleaner,
	# but it incurs a slight performance hit.
	
	
	def __init__(self,space):
		# prep the tensor for filling by supplying the search space OR
		# initialize the object for use with a full tensor. if passing in a
		# tensor, make sure the query set points have been stripped off first.
		
		# INPUT
		# space = numpy array of points in search space OR an extant tensor
		
		# PARAMETERS
		#self.ikmax=XX000000	# max size for search space * set for a gpu
								# with 2gb of ram.
		#self.i2=3000			# max size constraint for PTshrink on gpu 
								# with 2gb of ram.
		

		assert len(space.shape)==2 or len(space.shape)==3, "need a 2- or 3-array as input"
		
		# for error handling in other methods: 
		self.gssp=0
		self.gnb=0
		
		# if we put in a search space:
		if len(space.shape)==2:
			# get the search space size
			self.i=space.shape[0]
			# set other parameters
			self.ikmax=80000000
			self.i2=5000
			# send *space* to the gpu and include in *self*
			self.gssp=gpua.to_gpu(space.astype(numpy.float32))			
		
		# if we put in a tensor:	
		elif len(space.shape)==3:
			# set misc parameters
			self.ikmax=60000000
			self.i2=0
			# get the dimensions
			self.i=space.shape[1]
			self.k=space.shape[0]
			# make sure we're not oversized here
			assert self.i*self.k<self.ikmax, "the tensor is too big, sorry"			
			# build *irows*
			self.irows=gpua.zeros(self.k,dtype=numpy.uint32)
			self.irows.fill(self.i)
			# send the neighborhood to the gpu
			self.gnb=gpua.to_gpu(space.astype(numpy.float32))
			
	#=========================
		
	def fill(self, qset, rad):
		# create a neighborhood tensor for general multiscale processing.
		# note we're not defining any more *self* variables in here because 
		# this method directly returns the tensor we're interested in
		
		# TAGS: general, build
		
		# INPUT
		# qset = query set, on host
		# rad = search radius
		
		# OUTPUT
		# neighborhood tensor
		
		# make sure the object has been initialized appropriately
		assert isinstance(self.gssp, gpua.GPUArray), "this tensor is already full"
		
		# get the remaining dimension of the tensor
		k=qset.shape[0]
		
		# more safety
		assert k*self.i<self.ikmax, "the proposed tensor is too big, sorry"
		
		# build irows
		irows=gpua.zeros(k,dtype=numpy.uint32)
		irows.fill(self.i+1)	
		
		# put the query set on the gpuarray
		gqse=gpua.to_gpu(qset.astype(numpy.float32))
		
		# compose the tensor
		gnb=ch.ngrab(self.gssp,gqse,rad)
		
		# get rid of the query set on gpu
		gqse.gpudata.free()
		
		# shrink and return the tensor
		gnb, _ = ch.PTshrink(gnb,irows,self.i2,rad)
		nb=gnb.get()

		# salt the earth
		gnb.gpudata.free()
		irows.gpudata.free()
		
		# return the neighborhood
		return nb
		
	#=========================
	
	def MP_displacement(self):
		# get the norm of the vector between the origin of each neighborhood
		# and its centroid, and send back to host
		
		# TAGS: MP, process
		
		# OUTPUT
		# vector of mean point displacements
		
		
		# make sure we're on the processing side of the workflow
		assert isinstance(self.gnb,gpua.GPUArray),"this method requires a tensor"	
		
		# get the norms
		norms,self.cents=ch.PTcentroid(self.gnb,self.irows) 
		
		# pass norms back to host and purge
		n=norms.get()
		norms.gpudata.free()		
		
		return n
		
	#=========================		
	
	def SAZO(self):
		# Signed mAximum Z Offset multiscale operator. returns the maximum
		# Z offset between the evaluation point and any point in the
		# neighborhood, with sign preserved.
		
		# OUTPUT
		# vector of z offset values
		
		
		assert isinstance(self.gnb,gpua.GPUArray),"this method requires a tensor"	
		pass	# for now

	#=========================		
	
	def MSPCA_cov(self):
		# generate a covariance matrix for each page of the neighborhood and
		# send back to host.
				
		# TAGS: MSPCA, process
		
		# OUTPUT
		# covariance matrices in tensor form
		
		assert isinstance(self.gnb,gpua.GPUArray),"this method requires a tensor"
		assert isinstance(self.cents,gpua.GPUArray),"this method requires MP_displacement to be run first"
		
		# generate covariance matrix
		gc=ch.PT_cov(self.gnb,self.cents,self.irows)
		
		c=gc.get()
		gc.gpudata.free()
		self.cents.gpudata.free()
		
		return c
		
	#=========================		
	
	def MSPCA_eigs(self):
		# generate a covariance matrix for each page of the neighborhood,
		# calculate eigvals, normalize and then send the biggest 2 back to host.
		
		# TAGS: MSPCA, process
		
		# OUTPUT
		# 2-array of eigvals
		
		assert isinstance(self.gnb,gpua.GPUArray),"this method requires a tensor"
		assert isinstance(self.cents,gpua.GPUArray),"this method requires MP_displacement to be run first"
		
		# generate covariance matrix
		gc=ch.PT_cov(self.gnb,self.cents,self.irows)
		try:
			self.cents.gpudata.free()
		except pycuda._driver.LogicError:
			print('failed to dump original centroid locations in NBtensor.MSPCA_eigs')
			print(cuda.mem_get_info())
			
		# get eigvals
		eig=ch.block_eigvals(gc)
		# prep em
		eig=ch.row_norm_sort(eig)
		
		c=eig.get()
		eig.gpudata.free()
		gc.gpudata.free()
		self.cents=0
		
		return c.take([0,1],axis=1)
				
	#=========================		
	
	def drop(self,rad):
		# drop points out of neighborhoods in the tensor
		# TAGS: process
		
		# INPUT
		# rad = new exclusion radius
		
		# OUTPUT
		# modifies state (neighborhood tensor and irows) and returns *irows*
		# to host
		
		
		# make sure we're on the processing side of the workflow
		assert isinstance(self.gnb,gpua.GPUArray),"this method requires a tensor" 
		
		# shrink out points past the radius
		self.gnb,self.irows=ch.PTshrink(self.gnb,self.irows,self.i2,rad)
		
		return self.irows.get()
		
	#=========================
		
	def purge(self):
		# deallocates whatever's left on the GPU when done with the object
		
		# TAGS: general, build, process
		
		# if a tensor was supplied we take one path
		if isinstance(self.gnb,gpua.GPUArray):
			self.gnb.gpudata.free()
			self.irows.gpudata.free()
		# and if we started from a search space we take another
		elif isinstance(self.gssp,gpua.GPUArray):
			self.gssp.gpudata.free()	
		

		

#-------------------------------------------------------------------------------

class Partitions(object):
	# g mills 27/6/14
	# severe refactor 22/8/15
	# self-contained solution to the density partition problem. produces a 
	# generator which can be used to index nested partitions in one or two
	# co-located point clouds.	
	
	# this implementation is designed to have a small memory footprint. it 
	# leaves the original point cloud in place and just paqses around a set of
	# rules for extracting (mostly) cubic pieces which allow independent 
	# processing. 
	
	#=========================
	
	def __init__(self,inc,imax,bufferrad,omrad,rmrad,minpop):
		# initialize the object and perform the partition.
		
		# INPUT 
		# inc = search space point cloud-- gpuarray or ndarray is fine
		# imax = max population of a search space partition
		# bufferrad = difference between search space and query set radii
		# omrad = minimum acceptable radius of an octree partition
		# rmrad = minimum acceptable final output radius-- note edge partitions
				# may come out smaller than this.
		# minpop = minimum population of a partition to be stored 
				
		
		# PARAMETERS
		cpumax=3000000	# maximum size of a point cloud we want to work with
						# on the host. if more than this, switch to gpu.
		
		# OUTPUT
		# builds and stores self.rulebook, describing all the partitions.
		
		
		# if we took in an array on the host, check and make sure we don't
		# need to send to gpu.
		if isinstance(inc,numpy.ndarray):
		
			if inc.shape[0]>cpumax:
				#print("host-side pointcloud size limit exceeded-- using GPU to partition")
				inc=gpua.to_gpu(inc.astype(numpy.float32))
		
		# set the attributes
		self.pop=inc.shape[0]		# total population
		self.buffer=bufferrad		# this will be the same for all methods
		self.minpop=minpop
		self.imax=imax								
		self.omrad=omrad
		self.rmrad=rmrad
		self.rulebook=[]			# list of final partitioning rules dividing
							# the space into acceptable partitions. rules are
							# of the format ndarray([[xi,xa],...])
		self.radreduce=.95			# rigid partitioning radius reduction factor
							# this influences the rate at which rigid converges.
							# if it's smaller it'll go faster, but we'll tend to
							# find smaller partitions.
		
		# finding the faces of the bounding cube is a little more complicated.
		if isinstance(inc,gpua.GPUArray): 		# first get center
			ig=inc.get()		# TODO...
			center=ig.mean(0)	 
			imi=numpy.abs(center-ig.min(0)).max()
			ima=numpy.abs(center-ig.max(0)).max()
		else:
			center=inc.mean(0)
			imi=numpy.abs(center-inc.min(0)).max()
			ima=numpy.abs(center-inc.max(0)).max()
		rad=max(imi,ima) 		# all points will fall within this radius
		iface = center-rad 		# minimum face
		aface = center+rad		# maximum face
		
		# build the initial rule to start the octree partition
		rule = numpy.column_stack((iface,aface))
		
		
		# check if partitioning is needed at all-- it's possible that we don't
		# have enough points to worry about.
		if self.pop<=self.imax:
			self.rulebook+=[rule]
		else:
			# then we should partition. 
			if rad<=omrad:
				# if the point cloud's domain is small, then start with rigid.
				self._rigid(inc,rule,self.pop)
			else:
				# otherwise launch the octree
				self._octree(inc,rule)
		
	#=========================

	def partition_generator(self,qse,ssp=[]):
		# g mills 22/8/15
		# yields each qse/ssp partition pair in turn, given a qse and ssp.
		# note that the choice of gpuarray or ndarray here will influence 
		# performance.
		
		# INPUT
		# qse = query set, gpuarray or ndarray
		# ssp = search space, gpuarray or ndarray. if not supplied, will use
				# the point cloud provided as first input for both qse and ssp.
		
		# OUTPUT
		# yields qse and ssp boolean masks-- gpuarray/ndarray same as input.
		
		
		# iterate over the rulebook
		for rule in self.rulebook:
			q=self._rule_threshold(qse,rule,False)
			rp=rule.copy()
			
			rp[:,0]-=self.buffer
			rp[:,1]+=self.buffer
			# if supplied, use the specified ssp to generate the surrounding 
			# partition's boolean mask
			if len(ssp):
				s=self._rule_threshold(ssp,rule)
			else:
				s=self._rule_threshold(qse,rule)
				
			# type check for gpuarray
			if isinstance(q,gpua.GPUArray):
				q=q.get()
			if isinstance(s,gpua.GPUArray):
				s=s.get()
				
			yield q,s
	
	#=========================

	def _rigid(self,inc,rule,pop):
		# makes small (non-dyadic) adjustments in partition size. the output
		# partition set tiles the input partition with cubes, with rectangular
		# prisms touching three of its faces.
		
		# an alternative approach would be to divide the partition into n*n
		# cubes and increase n until all pass, but my intuition is we would end
		# up with more, smaller partitions that way.
	
		# starts with a large radius and then works downwards. the factor by 
		# which the points in the oversized proposed partition exceeds *imax*
		# becomes a power to which we raise *radreduce* when calculating the
		# next reduction in radius. we to start from the assumption that points
		# are distributed uniformly throughout the volume and base our first
		# radius on that. we start large because if we started at a small
		# radius then we'd need to check every partition to make the decision
		# to increase the radius each time. 
	
		# INPUT
		# inc = incoming point cloud
		# rule = rule for the partition to process
		# pop = population of the inbound partition
	
		# PARAMETERS
		
		runflag=True		# running condition for rule finding loop
		runtime=60			# number of seconds this process is permitted to run
	
		# OUTPUT
		# par = partition array
	
	
		# find the radius of the input region
		irad = (rule[0,1]-rule[0,0])/2		
		# find its center
		center = rule[:,0]+irad
		# calculate the volume of the partition bounding box
		bvol=(irad*2)**3		# calculate the point density
		bden=pop/bvol
		
		# find the minimum and maximum faces of this partition
		smin=rule.min(1)
		smax=rule.max(1)
	
		# find the cubic radius corresponding to the volume of space which 
		# contains *imax* points on average
		tvol=self.imax/bden		# target volume of this box
		srad=tvol**(1/3)*.5		# and its radius
		rad=srad-self.buffer	# and the query set radius corresponding to that
									# search space radius
		
		# final safety switch
		start=time.clock()
	
		# try and find that optimal radius.
		while runflag:
			
			# check that we haven't run out of time
			if time.clock()-start > runtime:
				print("no acceptable partition found in Partitions._rigid")
				break	# if we let it run one more, it will take even longer
						# than the last. the ruleset list comprehension below
						# can get very heavy so that might be a bad idea.
				
			# make sure that this partitioning scheme will result in large-
			# enough partitions
			assert rad>=self.rmrad, 'search space is too dense for the radius specified'
		
			# now build a set of partitioning rules to divide the space.
			# minimum faces in each coordinate direction
			xir=numpy.arange(smin[0],smax[0],2*rad)
			yir=numpy.arange(smin[1],smax[1],2*rad)
			zir=numpy.arange(smin[2],smax[2],2*rad)
			# and the corresponding maximum faces
			xar=xir+2*rad
			yar=yir+2*rad	# not super pleased with this.
			zar=zir+2*rad 
			# set the last max face equal to the max face of the outer
			# partition
			xar[-1]=smax[0]
			yar[-1]=smax[1]
			yar[-1]=smax[2]
			# compose min, max face pairs
			xr = numpy.column_stack((xir,xar))
			yr = numpy.column_stack((yir,yar))
			zr = numpy.column_stack((zir,zar))
			# obtain every possible permutation of these rules
			ruleset = [numpy.vstack((x,y,z)) for x in xr for y in yr for z in zr]
			# and we'll also take note of their population
			popset = numpy.zeros(len(ruleset))
		
			# iterate over these rules and find their populations
			for num,rule in enumerate(ruleset): 
				pop=self._population(inc,rule)
				popset[num]=pop
				# check if oversize
				if pop>self.imax:
					# reduce the radius and break to try again
					rad*=radreduce**(pop/self.imax)
					break
			# if we made it through the loop, we have an acceptable set of rules
			else:
				# filter the empty partitions out of the ruleset
				ruleset=[r for r,p in zip(ruleset, popset) if p>=self.minpop]
				self.rulebook+=ruleset
				break
			
	#=========================
	
	def _octree (self, inc, rule):
		# recursive octree partitioning. runs until the target point count or 
		# minimum partition radius is reached. in the latter case, calls rigid
		# to finish partitioning.
		
		# INPUT
		# inc = point cloud to be partitioned
		# rule = the rule defining the region to be partitioned
	
		# OUTPUT
		# writes cubic partitions to self.rulebook if they pass the size or 
		# population tests.
		
		
		# come up with 8 new rules to divide this partition--		
		# find the new radius
		rad = (rule[0,1]-rule[0,0])/4		
		# find the current center
		center = rule[:,0]+rad*2		
		# build the set of new centers
		cr=numpy.array([-1,1])
		centerset=numpy.asarray([[x,y,z] for x in cr for y in cr for z in cr])
		centerset=(centerset*rad+center).astype(numpy.float32)		
		# to each center, add and subtract the radius to get its faces
		r_hi = [cen+rad for cen in centerset]
		r_lo = [cen-rad for cen in centerset]
		# compose into a single list of rules
		ruleset = [numpy.column_stack((lo,hi)) for lo,hi in zip(r_lo,r_hi)]

		# iterate over the rules and check their populations
		for rule in ruleset:
			pop=self._population(inc,rule)
			# first make sure it's sufficiently populated
			if pop>= self.minpop:			
				# if the population of this partition is small enough,
				# save its rule
				if pop <= self.imax:
					self.rulebook+=[rule]
				# otherwise, we'll have to continue partitioning	
				else:
					# if the radius is small, use the rigid partition method
					if rad <= self.omrad:
						self._rigid(inc,rule,pop)
					# if its radius is still large, call octree on it
					else:
						self._octree(inc,rule)
	
	#=========================
					
	def _rule_threshold(self, inc, rule, use_buffer=True):
		# g mills 22/8/15
		# threshold the data using a set of min- and max- bounding planes
		
		# INPUT
		# inc = search space point cloud
		# rule = set of bounding coordinates- ndarray([	 [xmin,xmax],
														#[yi,ya],
														#[zi,za]])
		# use_buffer = incorporate the buffer into the rule?
		
		# OUTPUT
		# a boolean mask corresponding to all the points within the region
		# described by the rule
		
		
		# copy the rule
		userule=rule.copy()

		# modify if necessary
		if use_buffer:
			userule[:,0]-=self.buffer
			userule[:,1]+=self.buffer
			
		# either threshold it in-house or send to the cuda kernel
		if isinstance(inc,gpua.GPUArray):
			bmask=ch.rule_threshold(inc,userule)
		else:
			# perform thresholds on each dimension in turn
			bmask=numpy.ones(inc.shape[0],dtype=numpy.uint32)
			for d in range(3):
				bmask*=inc[:,d]>userule[d,0]	
				bmask*=inc[:,d]<userule[d,1]		
		return bmask

	#=========================

	def _population(self, inc, rule):
		# g mills 22/8/15
		# calculates the population of a search space partition. to be used in 
		# partition construction methods.
		
		# INPUT
		# inc = point cloud, gpuarray or ndarray
		# rule = rule describing the partition
		
		# OUTPUT
		# population of the partition
		
		
		# get the boolean mask
		bmask = self._rule_threshold(inc,rule)
		
		# we can get the population two different ways: this is why this method
		# exists.
		if isinstance(bmask, numpy.ndarray):
			return bmask.sum()
		else:
			return gpua.sum(bmask).get()/1
		
		
#-------------------------------------------------------------------------------

def double_vox(inc,edge):
	# g mills 19/2/15
	# voxel filter using the basic voxel filter function to perform the first
	# order partition. 40% faster than octree_vox. 
	
	# INPUT
	# inc = input point cloud, ndarray
	# edge = edge length of voxel filter, i.e. distance between centers.
	
	# PARAMETERS
	minvox=10		# if there aren't this many points in the partition, ignore
	measure='cheby'	# we voxelize cubic partitions, of course
		
	# OUTPUT
	# vox = voxelized version of *inc*
	ivox=numpy.zeros((0,3))
	
	
	# calculate the maximum radius of a cube that we can voxelize at once
	dim=1024*edge*0.98		# small safety factor here for cuvox
	rad=dim/2
	
	# make sure double_vox will work here, and if not then outsource it
	span=inc.max(0)-inc.min(0)
	if span.max()>1000000*edge:
		print('using octree voxel pattern')
		ivox=octree_vox(inc,edge)
		return ivox
	
	# get partition centers by voxelizing the original point cloud at that edge 
	# length
	partlist=ch.cuvox(inc,dim).get()
	
	# load *inc* on the gpu
	ginc=gpua.to_gpu(inc.astype(numpy.float32))
	for p in partlist:
		# indices of the partition members in the original point cloud
		idx=ch.cu_query_neighborhood(ginc,p,rad,measure)
		# make sure it's worth our time
		if idx.size>minvox:
			part=inc.take(idx,axis=0)
			# apply voxel filter and concatenate to output
			ivox=numpy.vstack((ivox,ch.cuvox(part,edge).get()))
	
	return ivox
	

#-------------------------------------------------------------------------------

def octree_vox(inc,edge):
	# g mills 3/2/15
	# basic voxel filter launcher
	
	# INPUT
	# inc = input point cloud as ndarray of any floats
	# edge = voxel edge length, i.e. the space between centers. 
	
	# PARAMETERS
	minpoints=100	# ignore any branch smaller than
	minvox=10		# ignore any leaf smaller than
	measure='cheby'	# we voxelize cubic partitions, of course
	
	# OUTPUT
	# vox = voxelized version of *inc*
	ivox=numpy.zeros((0,3))
	
	
	# calculate the maximum radius of a cube that we can voxelize at once
	dim=1024*edge*0.98		# small safety factor here for cuvox
	rad=dim/2
	
	# load *inc* on the gpu
	ginc=gpua.to_gpu(inc.astype(numpy.float32))
	
	# use an octree to partition the point cloud
	partlist=ch.gpu_rigid_tree(inc,minpoints,rad,0,0,0,0)
	
	# we only want the partition centers
	partlist=partlist[:,:3]
	
	# loop over the partition array and voxelize each one in turn
	for p in partlist:
		# indices of the partition members in the original point cloud
		idx=ch.cu_query_neighborhood(ginc,p,rad,measure)
		# make sure it's worth our time
		if idx.size>minvox:
			part=inc.take(idx,axis=0)
			# apply voxel filter and concatenate to output
			ivox=numpy.vstack((ivox,ch.cuvox(part,edge).get()))
		
	return ivox



#-------------------------------------------------------------------------------

def natural_vox(inc,edge):
	# g mills 19/1/15
	# natural voxel filter: returns an index set corresponding to one point in 
	# each voxel cell, instead of the coordinates of each voxel cell.
	
	# INPUT
	# inc = input pointcloud
	# edge = voxel edge length- a whole side of the cube
	
	# PARAMETERS
	octree=False	# if true, we'll use the octree pattern for first pass
					# partition.
	
	# OUTPUT
	# vci = indices of original points; one per voxel
	
	
	# calculate the maximum radius of a cube that we can voxelize at once
	dim=1024*edge*0.98		# small safety factor here for cuvox
	rad=dim/2
	
	# find the point cloud spanning dimensions
	span=inc.max(0)-inc.min(0)
	
	# make sure double vox pattern will work here, else use an octree
	span=inc.max(0)-inc.min(0)
	if span.max()>1000000*edge:
		print('using octree pattern')
		octree=True
	
	# voxelize search space-- 
	# if smaller than max voxel dimensions, do it all in one pass
	if span.max() < dim:
		vci=ch.cu_natural_vox(inc,edge)
	else:
		# initialize search space output array
		vci=numpy.zeros(0,dtype=numpy.uint32)
		
		# load *inc* on the gpu
		ginc=gpua.to_gpu(inc.astype(numpy.float32))
		
		if octree:
			# use an octree to partition the point cloud
			partlist=ch.gpu_rigid_tree(inc,minpoints,rad,0,0,0,0)
	
			# we only want the partition centers
			partlist=partlist[:,:3]
		
		else:
			# get partition centers by voxelizing the original point cloud at
			# that edge length
			partlist=ch.cuvox(inc,dim).get()	
	
		for p in partlist:
			# indices of the partition members in the original point cloud
			idx=ch.cu_query_neighborhood(ginc,p,rad,measure)
			# pull the points, voxelize, index to original point cloud index, 
			# and stack to output
			if len(idx)>10:
				vidx=ch.cu_natural_vox(inc.take(idx,axis=0),edge)
				vci=numpy.hstack((vci,numpy.take(idx,vidx)))
	
	return vci
	

#-------------------------------------------------------------------------------

def OG_MSO(qse,ssp,sspedge,scales,imax=20000):
	# g mills 24/8/15
	# first order (pure geometry) multiscale operator processing chain. 
	# voxelizes and proceqses MSOs for input point cloud at given scales, then 
	# returns feature vectors with their indices in the original point cloud.
	# this version calculates eigenvectors as well in order to preserve
	# neighborhood orientation information
	
	# 29/8/15 rework: feature-major ordering, like gmso and vmso
	
	# FIRST: put scales in descending order and make float32
	scales=numpy.float32(numpy.sort(scales)[::-1])
	
	# INPUT
	# qse = query set
	# ssp = search space
	# sspedge = edge length for search space voxelization. 0 skips subsampling.
	# scales = numpy array of spherical neighborhood radii to use
	# imax = maximum number of points in a ssp partition
	
	# PARAMETERS
	inrows=qse.shape[0]
	#imax=20000			# maximum number of points in a ssp partition
	minrad=scales[-1]	# absolute minimum radius of a qse partition
	ominrad=minrad*3	# minimum radius of a query set partition in octree
	buffer=scales[0]	# size difference between query set and search space rad
	ivt = 10			# ignore voxel threshold- number of points needed in a 
						# partition in order to justify work on it
	pdir='nbtemp/nb'	# directory name and prefix for temp storage of tensors
	pidx=0				# starting save file index
	
	# OUTPUT
	# outc = point cloud indices (query set points) with multiscale vectors
	# appended. scales in descending order.
	# [IDX, (density, centroid, eigval x2, vec x4) x num scales]
	outc=numpy.zeros((0,1+scales.size*8),dtype=numpy.float32)
	
				
	# some timers		
	alltime=time.time()
	
	# voxelize the search space point cloud if necessary
	if sspedge!=0:
		v=time.time()
		ssp=double_vox(ssp,sspedge).astype(numpy.float32)	
		v2=time.time()-v
	else:
		ssp=ssp.astype(numpy.float32)
					
	# query set won't be voxelized since we're returning indices in original
	# point cloud.
	qse=qse.astype(numpy.float32)
	# index the query set
	all_qse_index=numpy.arange(qse.shape[0])
	
	# partition the search space
	partset = Partitions(ssp,imax,buffer,ominrad,minrad,ivt)
	# put the qse and ssp on the gpu for faster partitioning
	g_qse=gpua.to_gpu(qse)
	g_ssp=gpua.to_gpu(ssp)
	
	# iterate over the set of partitions
	for qse_mask, ssp_mask in partset.partition_generator(g_qse,g_ssp):
		pt=time.time()
		
		# reset the storage file index
		pidx=0					
			
		# make sure this volume should be processed
		if qse_mask.sum()>ivt:
		
			# get the search space and query set partitions
			lqse = numpy.compress(qse_mask,qse,axis=0)
			lssp = numpy.compress(ssp_mask,ssp,axis=0)
			# pass the partitions to the neighborhood construction pipeline
			nb,pidx=NB_build(lqse,lssp,buffer,pdir,pidx)
			# retrieve the indices associated with the qse points 
			qseidx=numpy.extract(qse_mask,all_qse_index)
									
			# loop over all the tensors in the temp directory if necessary
			if pidx:
				for pd in range(pidx):	
					nb=numpy.load(pdir+str(pd)+'.npy')
					# carve off the first however many query set indices
					uqseidx=qseidx[:nb.shape[0]]
					qseidx=qseidx[nb.shape[0]:]
					outvec=OGNB_process(nb,scales,uqseidx)
					outc=numpy.vstack((outc,outvec))
					nb=0	
			else:				
				# pass the only neighborhood to the process pipeline
				outvec=OGNB_process(nb,scales,qseidx)
				outc=numpy.vstack((outc,outvec))
				nb=0	
		
		partime=(time.time()-pt)
				
			
	finaltime=time.time()-alltime
	finalpoints=outc.shape[0]
	pointsec=finalpoints/finaltime
	print( 'total time in gmso at ' + str(sspedge) + 'm voxel edge length: ' + str(int(finaltime)) + 's')
	print( str(finalpoints) + ' points processed at an overall rate of ' + str(int(pointsec)) + ' points per second')
	
	
	return outc




	
#-------------------------------------------------------------------------------

def OGNB_process(inb,scales,qseidx):
	# g mills 24/8/15
	# processing pipeline for eigvals, density, centroid and eigvecs.
	
	# INPUT
	# inb = input numpy 3-array
	# scales = list of analysis scales. should be descending order and float32 
	# qseidx = list of indices associated with the points we will be processing
	
	# PARAMETERS
	ikmax=50000000		# max value of i*k in the processing pipeline
	k = inb.shape[0]	# pages in bulk neighborhood/ query set points
	i = inb.shape[1]	# rows per page in same/ search space points
	eps=numpy.spacing(1)	# tiny number to protect against division by zero
	ns = scales.size
	rad=180/numpy.pi	
	conv=100*100*100	# 1 million cubic centimeters in a cubic meter.	
	ydimmax=65535		# this is the largest possible grid y-dim in CUDA. 
	# refactor *segscan* and we won't have to place this artificial limitation
	# on the chunk size here. see *PTshrink* comments for details. of course,
	# if we load TOO big a tensor we could risk a kernel hang.
	outwidth=8			# number of output args per scale (4 for basic gmso, 
						# 8 for vecs)	
	
	# OUTPUT
	# outc = point cloud indices (query set points) with multiscale vectors
	# appended. scales in descending order.
	# [IDX, (density, centroid, eigval x2, vec x4) x num scales]
	outc=numpy.zeros((0,1+ns*outwidth),dtype=numpy.float32)
	
	
	# decide how to partition the tensor into manageable chunks
	kmax=min(ydimmax,int(numpy.floor(ikmax/i)))
	klast=k%kmax					# last chunk
	dwell=int(numpy.ceil(k/kmax))
	
	# loop over chunks
	for d in range(dwell):
		# initialize output chunk
		if inb.shape[1]>1:
			if d==dwell-1:	# if last pass
				# take the last or only piece
				gen=NBtensor(inb[-klast:,1:,:])	
				# strip those query set indices and compose to a col vector
				soutc=numpy.zeros((klast,1+ns*8),dtype=numpy.float32)
				soutc[:,0]=qseidx[-klast:]	

			else:			
				# take a piece from the front or middle
				gen=NBtensor(inb[kmax*d:kmax*(d+1),1:,:]) 
				# strip those query set indices and compose to a col vector
				soutc=numpy.zeros((kmax,1+ns*8),dtype=numpy.float32)
				soutc[:,0]=qseidx[kmax*d:kmax*(d+1)]
		
			# loop over scales
			for s in enumerate(scales):
				# calculate output block offset
				s_off = 1+s[0]*outwidth
				# drop neighborhood to this scale (no points should be dropped
				# on first pass)
				irows=gen.drop(s[1])				
				# calculate the volume of the neighborhood-- points per cm^3
				vol=conv*(4/3)*numpy.pi*s[1]**3
				# calculate the density of the neighborhoods
				soutc[:,s_off]=irows/vol			
				# get the mean point displacements
				soutc[:,s_off+1]=gen.MP_displacement()			
				# get the write position for eigenvalues
				es=s_off+2
				
				# we're using numpy to perform the eigendecomposition since we
				# want the eigvecs now. the vectors will be normalized and
				# correspond to the same ordering as the values. since they're
				# normalized the first two entries of the vector will uniquely
				# define it. likewise, we don't really care about the least
				# vector so we will take the first two of the first two. 
				cov=numpy.nan_to_num(gen.MSPCA_cov())
				vals,vecs=eig(cov)
				
				# normalize the eigenvalues to 1-- normalize all three before
				# taking off the two largest. 
				vals/=(vals.sum(1).reshape(-1,1)+eps)
				
				# we want to (efficiently) sort the rows of *vals* and chop off
				# the third of each. so we'll do an argsort over the rows, add
				# an offset to each row of the argsort output, chop out the
				# last column, flatten it, and then do a flat .take on *vals*,
				# and then reshape *vals* into 2 cols.
				sorter=numpy.argsort(vals,axis=1)
				valoff=numpy.arange(sorter.shape[0])*3
				valsorter=sorter+valoff.reshape(-1,1)
				valsorter=valsorter[:,:2].ravel()
				soutc[:,es:es+2]=vals.take(valsorter).reshape(-1,2)
				
				# now let's get a write position for the eigenvectors:
				vs=s_off+4
				# now we have a 3-array containing our eigenvalues. the 2nd and
				# 3rd axes will be transposed and we'll cut out the Z axis
				vecs=numpy.transpose(vecs,(0,2,1))[:,:,:2]
				# now we can take our sorting array and chop it down to the
				# first 2 vectors' addreqses before we start transforming it
				vecsorter=sorter[:,:2]
				# it will now reference the first entry of the first vector
				# of each qse point
				vecsorter=vecsorter*2+(numpy.arange(sorter.shape[0])*6).reshape(-1,1)
				# we want to take both entries of each vector so we'll flatten
				# it and interleave it with a version that has had 1 added 
				vecsorter1=vecsorter.ravel()
				vecsorter2=vecsorter1+1
				vecsorter=numpy.column_stack((vecsorter1,vecsorter2)).ravel()
				# now take out the partial vectors and shape into a 2-array
				soutc[:,vs:vs+4]=vecs.take(vecsorter).reshape(-1,4)
							
			
			# purge all that stuff from the GPU
			gen.purge()
			# concatenate output chunk to *outc*
			outc=numpy.vstack((outc,soutc))
				
	
	return outc
	
								
				
				
#-------------------------------------------------------------------------------

def C_MSO(qse,ssp,sspedge,scales,imax=20000):
	# g mills 31/8/15
	# first order geometric multiscale operator processing chain. 
	# voxelizes and proceqses MSOs for input point cloud at given scales, then 
	# returns feature vectors with their indices in the original point cloud.
	# this version returns the upper triangular portion of the covariance 
	# matrix of each neighborhood, rather than the eigenfeatures.
	
	
	# FIRST: put scales in descending order and make float32
	scales=numpy.float32(numpy.sort(scales)[::-1])
	
	# INPUT
	# qse = query set
	# ssp = search space
	# sspedge = edge length for search space voxelization. 0 skips subsampling.
	# scales = numpy array of spherical neighborhood radii to use
	# imax = maximum number of points in a ssp partition
	
	# PARAMETERS
	inrows=qse.shape[0]
	#imax=20000			# maximum number of points in a ssp partition
	minrad=scales[-1]	# absolute minimum radius of a qse partition
	ominrad=minrad*3	# minimum radius of a query set partition in octree
	buffer=scales[0]	# size difference between query set and search space rad
	ivt = 10			# ignore voxel threshold- number of points needed in a 
						# partition in order to justify work on it
	pdir='nbtemp/nb'	# directory name and prefix for temp storage of tensors
	pidx=0				# starting save file index
	
	# OUTPUT
	# outc = point cloud indices (query set points) with multiscale vectors
	# appended. scales in descending order.
	# [IDX, (density, centroid, cov x 6) x num scales]
	outc=numpy.zeros((0,1+scales.size*8),dtype=numpy.float32)
	
				
	# some timers		
	alltime=time.time()
	
	# voxelize the search space point cloud if necessary
	if sspedge!=0:
		v=time.time()
		ssp=double_vox(ssp,sspedge).astype(numpy.float32)	
		v2=time.time()-v
	else:
		ssp=ssp.astype(numpy.float32)
					
	# query set won't be voxelized since we're returning indices in original
	# point cloud.
	qse=qse.astype(numpy.float32)
	# index the query set
	all_qse_index=numpy.arange(qse.shape[0])
	
	# partition the search space
	partset = Partitions(ssp,imax,buffer,ominrad,minrad,ivt)
	# put the qse and ssp on the gpu for faster partitioning
	g_qse=gpua.to_gpu(qse)
	g_ssp=gpua.to_gpu(ssp)
	
	# iterate over the set of partitions
	for qse_mask, ssp_mask in partset.partition_generator(g_qse,g_ssp):
		pt=time.time()
		
		# reset the storage file index
		pidx=0					
			
		# make sure this volume should be processed
		if qse_mask.sum()>ivt:
		
			# get the search space and query set partitions
			lqse = numpy.compress(qse_mask,qse,axis=0)
			lssp = numpy.compress(ssp_mask,ssp,axis=0)
			# pass the partitions to the neighborhood construction pipeline
			nb,pidx=NB_build(lqse,lssp,buffer,pdir,pidx)
			# retrieve the indices associated with the qse points 
			qseidx=numpy.extract(qse_mask,all_qse_index)
									
			# loop over all the tensors in the temp directory if necessary
			if pidx:
				for pd in range(pidx):	
					nb=numpy.load(pdir+str(pd)+'.npy')
					# carve off the first however many query set indices
					uqseidx=qseidx[:nb.shape[0]]
					qseidx=qseidx[nb.shape[0]:]
					outvec=CNB_process(nb,scales,uqseidx)
					outc=numpy.vstack((outc,outvec))
					nb=0	
			else:				
				# pass the only neighborhood to the process pipeline
				outvec=OGNB_process(nb,scales,qseidx)
				outc=numpy.vstack((outc,outvec))
				nb=0	
		
		partime=(time.time()-pt)
				
			
	finaltime=time.time()-alltime
	finalpoints=outc.shape[0]
	pointsec=finalpoints/finaltime
	print( 'total time in cmso at ' + str(sspedge) + 'm voxel edge length: ' + str(int(finaltime)) + 's')
	print( str(finalpoints) + ' points processed at an overall rate of ' + str(int(pointsec)) + ' points per second')
	
	
	return outc




	
#-------------------------------------------------------------------------------

def CNB_process(inb,scales,qseidx):
	# g mills 24/8/15
	# processing pipeline for density, centroid and covariance matrix.
	
	# INPUT
	# inb = input numpy 3-array
	# scales = list of analysis scales. should be descending order and float32 
	# qseidx = list of indices associated with the points we will be processing
	
	# PARAMETERS
	ikmax=50000000		# max value of i*k in the processing pipeline
	k = inb.shape[0]	# pages in bulk neighborhood/ query set points
	i = inb.shape[1]	# rows per page in same/ search space points
	eps=numpy.spacing(1)	# tiny number to protect against division by zero
	ns = scales.size
	rad=180/numpy.pi	
	conv=100*100*100	# 1 million cubic centimeters in a cubic meter.	
	ydimmax=65535		# this is the largest possible grid y-dim in CUDA. 
	# refactor *segscan* and we won't have to place this artificial limitation
	# on the chunk size here. see *PTshrink* comments for details. of course,
	# if we load TOO big a tensor we could risk a kernel hang.
	outwidth=8			# number of output args per scale (4 for basic gmso, 
						# 8 for vecs)	
	
	# OUTPUT
	# outc = point cloud indices (query set points) with multiscale vectors
	# appended. scales in descending order.
	# [IDX, (density, centroid, cov x 6) x num scales]
	outc=numpy.zeros((0,1+ns*outwidth),dtype=numpy.float32)
	
	
	# decide how to partition the tensor into manageable chunks
	kmax=min(ydimmax,int(numpy.floor(ikmax/i)))
	klast=k%kmax					# last chunk
	dwell=int(numpy.ceil(k/kmax))
	
	# loop over chunks
	for d in range(dwell):
		# initialize output chunk
		if inb.shape[1]>1:
			if d==dwell-1:	# if last pass
				# take the last or only piece
				gen=NBtensor(inb[-klast:,1:,:])	
				# strip those query set indices and compose to a col vector
				soutc=numpy.zeros((klast,1+ns*8),dtype=numpy.float32)
				soutc[:,0]=qseidx[-klast:]	

			else:			
				# take a piece from the front or middle
				gen=NBtensor(inb[kmax*d:kmax*(d+1),1:,:]) 
				# strip those query set indices and compose to a col vector
				soutc=numpy.zeros((kmax,1+ns*8),dtype=numpy.float32)
				soutc[:,0]=qseidx[kmax*d:kmax*(d+1)]
		
			# loop over scales
			for s in enumerate(scales):
				# calculate output block offset
				s_off = 1+s[0]*outwidth
				# drop neighborhood to this scale (no points should be dropped
				# on first pass)
				irows=gen.drop(s[1])				
				# calculate the volume of the neighborhood-- points per cm^3
				vol=conv*(4/3)*numpy.pi*s[1]**3
				# calculate the density of the neighborhoods
				soutc[:,s_off]=irows/vol			
				# get the mean point displacements
				soutc[:,s_off+1]=gen.MP_displacement()			
				
				# take the covariance matrices
				cov=numpy.nan_to_num(gen.MSPCA_cov())				
				# the shape of the covariance matrix-holding tensor is (k,3,3)
				# so we can flatten the last 2 dimensions to make it (k,9).
				# then we can use triu_indices to index the upper triangle.
				tridx=numpy.triu_indices(3)
				tridx=tridx[0]*3+tridx[1]
				cov=cov.reshape(-1,9).take(tridx,axis=1)				
				# now let's get a write position:
				cs=s_off+1
				# and slide in
				soutc[:,cs:cs+6]=cov
							
			
			# purge all that stuff from the GPU
			gen.purge()
			# concatenate output chunk to *outc*
			outc=numpy.vstack((outc,soutc))
				
	
	return outc
	
								
				
				

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
