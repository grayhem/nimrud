# point cloud archiving tools

import numpy
import os
import shutil
import mso 
import ch
import time
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import pycuda.gpuarray as gpua
import pycuda.driver as cuda
import gc
import ml
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.decomposition import FactorAnalysis as FA
from sklearn import preprocessing
from sklearn import manifold

#-------------------------------------------------------------------------------

def quick_builder(fnames,indir,aname,ext,delimiter,fcols):
	# g mills 28/5/15
	# builds label APCs from qse and ssp files with additional features. the 
	# point cloud will be qse and SSP from all classes with 1 index set. 
	
	# INPUT
	# fnames = file name label load list (L3): [(label,[qse],[ssp])]
	# indir = directory for files in fnames
	# aname = output APC label
	# ext = extension for fnames
	# fcols = number of columns of features
	
	# PARAMETERS
	ssplabnum = 999			# search space point identifier
	hltag = 'hand labeled'	# hand labeled points tag
	ssptag = 'ssp'			# search space tag
	
	# OUTPUT
	# saves a new label apc
	
	
	# start the point cloud
	inc = numpy.zeros((0,3))
	
	# start the label sets
	elabels = numpy.zeros(0)
	plabels = numpy.zeros(0)
	
	# start the feature sets
	efeats = numpy.zeros((0,fcols))
	pfeats = numpy.zeros((0,fcols))
	
	# start the index sets
	eidx = numpy.zeros(0,dtype=numpy.uint32)
	pidx = numpy.zeros(0,dtype=numpy.uint32)
	
	# load loop
	for labelset in fnames:
		qse=numpy.zeros((0,3+fcols))
		ssp=qse.copy()
		# acquire qse and ssp for this class
		for name in labelset[1]:
			qse=numpy.vstack((qse,numpy.genfromtxt(indir+name+ext,delimiter=delimiter)))
		for name in labelset[2]:
			ssp=numpy.vstack((ssp,numpy.genfromtxt(indir+name+ext,delimiter=delimiter)))
		# build up labels
		qselabs=numpy.zeros(qse.shape[0])+labelset[0]
		ssplabs=numpy.zeros(ssp.shape[0])+ssplabnum
		# build up indices
		qseidx=numpy.arange(qse.shape[0])
		sspidx=numpy.arange(ssp.shape[0])
		
		# put qse in inc
		qseidx+=inc.shape[0]
		inc=numpy.vstack((inc,qse[:,:3]))
		# and store its features, labels and indices
		elabels=numpy.hstack((elabels,qselabs))
		efeats=numpy.vstack((efeats,qse[:,3:]))
		eidx=numpy.hstack((eidx,qseidx))
		
		# and now for ssp
		sspidx+=inc.shape[0]
		inc=numpy.vstack((inc,ssp[:,:3]))
		elabels=numpy.hstack((elabels,ssplabs))
		efeats=numpy.vstack((efeats,ssp[:,3:]))
		eidx=numpy.hstack((eidx,sspidx))
			
	# make an APC
	apc=APC(inc,0,aname)
	
	# save index sets
	apc.add_idx(hltag,eidx,elabels,0,efeats,0)
	
	# bless this mess
	closer(apc,aname)
			

#-------------------------------------------------------------------------------

def opener(apcname):
	# g mills 26/5/15
	# opens an APC on the hdd
	
	# INPUT
	# apcname = name of the apc
	
	# PARAMETERS
	apc_dir='APC/'			# directory where the APCs live
	
	# OUTPUT
	# apc = the apc object
		
	apc=pickle.load(open(apc_dir+apcname+'.pkl','rb'))
	
	return apc


#-------------------------------------------------------------------------------

def closer(apc,apcname):
	# g mills 26/5/15
	# returns an apc to the hdd
	
	# INPUT
	# apcname = name of the apc
	
	# PARAMETERS
	apc_dir='APC/'			# APC directory
	
	# OUTPUT
	# n/a
	
	apc.gpu_inc_purge()
	pickle.dump(apc,open(apc_dir+apcname+'.pkl','wb'),pickle.HIGHEST_PROTOCOL)	
	

#-------------------------------------------------------------------------------

def get_feats(apc,tag=None):
	# g mills 27/5/15
	# just return features and tag we used.
	
	if tag is None:
		apc.query_keys()
		tag=input("use which feature set? >>> ")
	return apc.pull_feats(tag), tag
	
	
#-------------------------------------------------------------------------------

def get_idx(apc,tag=None):
	# g mills 29/5/15
	# just return indices corresponding to an index set, and the tag.
	
	if tag is None:
		apc.query_keys()
		tag=input("use which feature set? >>> ")
	return apc.items[tag][0], tag




#-------------------------------------------------------------------------------

class APC(object):
	# g mills 10/16/14
	# refactor 20/8/15
	# mediates access to point clouds with associated features stored on the
	# hard drive. 
	
	def __init__(self, fpath, vox, aname):
		# loads the point cloud for the first time, downsamples it, saves it as 
		# a .npy, recenters and builds a partition array. supports incoming 
		# feature vectors using natural_vox to avoid interpolation. APC will 
		# center the point cloud it contains to limit precision errors if it is
		# far from the origin (> 100 km). so please use self.get_inc when
		# exporting the point cloud.
		
		# INPUT
		# fpath = incoming point cloud. any attached features are OK as long as
		# they come after XYZ. 2 intake options are available:
				# path to point cloud, including filename. ascii or npy.
				# ndarray
		# vox = voxel edge length. set 0 to not voxelize. 
		# aname = output filename
		
		# PARAMETERS
		self.demodir = 'APC/demos/'	# directory to write demo point clouds to
		self.expodir = 'APC/expos/'	# export directory 
		self.items = {}			# dictionary for index and feature sets
		asciis=["txt","pts","ascii","csv","tsv"]	# expected ascii extensions
		kdthresh=50000000		# if > this number of points, don't use kdtree
		imax = 3000000			# max number of points in a partition
		minrad = .5				# minimum cubic radius of a partition (m)
		far_thresh = 100000		# point cloud centering threshold (m)
				
		# ATTRIBUTES
		# partlist = [center coordinates, radius, point count] of each partition


		# make the feature array directory
		self.featdir = 'APC/feats/'	+ aname +'/'
		while os.path.exists(self.featdir):
			# make sure it doesn't already exist
			self.featdir = 'APC/feats/'	+ aname + str(numpy.random.randint(100000))+'/'
		os.mkdir(self.featdir)	
			
		# memento mori
		mm=time.time()

		if isinstance(fpath,numpy.ndarray):
			inc=fpath
		else:
			if fpath.split(".")[-1]=="npy":
				# load it up in one piece if it's a numpy binary
				inc=numpy.load(fpath)
			elif fpath.split(".")[-1] in asciis:			
				# safely load point cloud (numpy chokes on big ascii files)
				inc=dainty_loader(fpath)
			else:
				print('what kind of file is this? aborting')
				return 
				
		print('------------- finished loading -------------' )
		print('time elapsed: ' + str(int(time.time()-mm)) + ' s')
		
		
		# determine what to do with the features, if they are present
		if inc.shape[1]>3:
			if int(input('keep the features? 1/0 >>> ')):
				fname = input('then name them >>> ')
			else:
				inc=inc[:,:3]
			
		# center it, if we're far from center.
		self.centroid = inc[:,:3].mean(0,dtype=numpy.float64)
		if numpy.linalg.norm(self.centroid)>far_thresh:
			inc[:,:3]-=self.centroid
			self.center_flag=True
		else:
			self.center_flag=False
		
		# and convert to float32
		inc=inc.astype(numpy.float32)
		
		# voxel filter?	
		if vox:
			vox_idx=mso.natural_vox(inc[:,:3],vox)
			inc=inc.take(vox_idx,axis=0).astype(numpy.float32)
			
		# store points	
		self.inc=inc[:,:3]		
			
		if inc.shape[1]>3:
			# save the feature vectors as well
			self.add_idx(fname,numpy.arange(inc.shape[0]),0,0,inc[:,3:],0)
			
			
		print('------------- finished voxelizing -------------' )
		print('time elapsed: ' + str(int(time.time()-mm)) + ' s')
		print(self.inc.shape)
		
		# build a partition array for the point cloud		
		self.partlist=ch.gpu_tree(self.inc,imax,0,minrad,0,0,0,0)
		# this will be used to query the partition array 
		self.inc_g=0
		
		print('------------- finished partitioning -------------' )
		print('time elapsed: ' + str(int(time.time()-mm)) + ' s')
		
	#=========================

	def gpu_inc(self):
		# put *inc* on the gpu
		self.inc_g=gpua.to_gpu(self.inc)

	#=========================
	
	def gpu_inc_purge(self):
		# remove *inc* from the gpu. run this before putting the object away.
		
		# note that gpuarrays will decide to destroy themselves pretty 
		# arbitrarily once they think they're out of scope. so just to be safe 
		# we'll allow for that.
		
		try:
			self.inc_g.gpudata.free()
		except AttributeError:
			pass	
		self.inc_g=0
		
	#=========================
	
	def get_inc(self):
		# return *inc* in original coordinates.
		if self.center_flag:
			return self.inc.astype(numpy.float64)+self.centroid
		else:
			return self.inc
	
	#=========================

	def pull_feats(self,tag):
		# return a feature set
		while True:
			try:
				out=numpy.load(self.items[tag][3][0])
				return numpy.nan_to_num(out)
			except FileNotFoundError:
				print('no such feature set.')
				tag=input('let us try that one again >>> ')
				
	#=========================
		
	def axe_idx(self,tag):
		# safely remove one or all idx sets from self.items
		
		# INPUT
		# tag = tag to remove; enter 0 to remove all.
		
		if tag==0:
			# find all the feature sets
			files=os.listdir(self.featdir)
			for file in files:
				os.remove(self.featdir+file)
			self.items.clear()
			
		# just in case we specified a tag that doesn't exist
		elif tag in self.items:
			# it's possible that the idx set we're removing doesn't have a 
			# feature set stored.
			try:
				os.remove(self.items[tag][3][0])
			except TypeError:
				pass
			# delete just this item
			del self.items[tag]
	
	#=========================
		
	def add_idx(self,tag,idx,labels,clusters,feats,scaleset):
		# add whatever new features, indices and labels to the object's 
		# dictionary. feel free to replace *labels* or *feats* with a 0 if one 
		# doesn't apply. if *labels* == 0, we assume that all points 
		# corresponding to the index set have the same label or we don't know 
		# their labels yet. 'unknown' is a label, too.
		
		# INPUT
		# tag = a string, whatever you want to call this index set
		# idx = 1d numpy array of indices in original point cloud
		# labels = 1d numpy array of labels corresponding to those indices
		# clusters = 1d numpy array of cluster IDs corresponding to same
		# feats = path to a binary of a numpy array of features corresponding
				# to those indices
		# scaleset = list of tuples of vox edges and analysis scales associated
				# with those feats
		
		# DICT FORMAT
		# tag, (idx, labels, cluster labels, feats, numscales, scales)
		# feats is stored as either a 0 (there are no features) or a tuple 
				# (feat path, num feats)
		
		# calculate the number of analysis scales
		if isinstance(scaleset,list):
			numscales=sum([len(x[1]) for x in scaleset])
		else:
			numscales=0
		
		# get a unique version of the indices and use it to re-sort the content 
		# arrays. this is really just insurance against any funny business
		# resulting from the MSO code so that we know it's safe to use the 
		# assume_unique criterion below. which we want to do because it speeds 
		# up intersect1d.
		idx,idxmap=numpy.unique(idx,return_index=True)
		if isinstance(labels,numpy.ndarray):
			labels=labels[idxmap]
		if isinstance(clusters,numpy.ndarray):
			clusters=clusters[idxmap]
		if isinstance(feats,numpy.ndarray):
			feats=feats[idxmap]
			numpy.save(self.featdir+tag+'.npy',feats)
			feats=(self.featdir+tag+'.npy',feats.shape[1])
		
		# load em up
		self.items.update([(tag,(idx,labels,clusters,feats,numscales,scaleset))])
	
	#=========================	
	
	def query_partlist(self,p,buffer,tag):
		# returns the ssp/qse indices associated with partition *p* and buffer 
		# radius *buffer* within the point cloud subset associated with *tag*.
		
		# INPUT
		# p = the partition
		# buffer = buffer radius
		# tag = tag of index set to pull output point clouds from-- if 0, use 
				# entire point cloud
		
		# OUTPUT
		# sspidx, qseidx = indices in the whole point cloud
		
		
		# make sure inc is on the gpu
		if not isinstance(self.inc_g,gpua.GPUArray):
			self.gpu_inc()
		
		# query the point cloud with the partition array
		cen=self.partlist[p,:3]		# centroid
		rad=self.partlist[p,3]		# radius	
		sspidx=ch.cu_query_neighborhood(self.inc_g,cen,rad+buffer,'cheby')
		qseidx=ch.cu_query_neighborhood(self.inc_g,cen,rad,'cheby')
				
		# mask these index sets with the index set associated with the tag
		if tag!=0:
			sspidx=numpy.intersect1d(sspidx,self.items[tag][0],assume_unique=True)
			qseidx=numpy.intersect1d(qseidx,self.items[tag][0],assume_unique=True)
		
		return sspidx,qseidx
	
	#=========================
	
	def query_keys(self):
		# prints off the names of the index sets, whether they have features
		# and/or labels associated and how populous they are		
		
		if self.items=={}:
			print('this APC contains no index sets')
		
		else:
			for k in self.items.keys():
				print('----------------------------------------')
				print(k)
				print('has ' + str(self.items[k][0].size)+' points')
				if isinstance(self.items[k][1],numpy.ndarray):
					print('has labels')
				if isinstance(self.items[k][2],numpy.ndarray):
					print('has clusters')
				if isinstance(self.items[k][3],tuple):
					print('has ' + str(self.items[k][3][1]) + ' features')	
				print('----------------------------------------')
				
				
#-------------------------------------------------------------------------------				
				
def dainty_loader(filename):
	# g mills 15/10/15
	# load a too-big point cloud in pieces

	# INPUT
	# filename = where to find the file

	# PARAMETERS
	middir='pointclouds/temp/'	# save segments here
	de=input("what's the delimiter? >>> ")

	# scour out the midfile directory in case we crashed last time
	midfiles=os.listdir(middir)
	for midfile in midfiles:
		os.remove(middir+midfile)

	mm=time.time()

	# split the file (15 million point segments)
	os.system('split -l 15000000 ' + filename + ' ' + middir+'temp')

	# go into *middir* and join the segments
	midfiles=os.listdir(middir)
	for midfile in enumerate(midfiles):
		print('------------- working on ' + midfile[1] + ' -------------' )
		print('time elapsed: ' + str(int(time.time()-mm)) + ' s')
		# load and stack up
		ina=numpy.genfromtxt(middir+midfile[1],delimiter=de,dtype=numpy.float32)
		if midfile[0]==0:
			inc=ina.copy()
		else:
			inc=numpy.vstack((inc,ina))

		# remove file
		os.remove(middir+midfile[1])

	return inc
				
				
				
#-------------------------------------------------------------------------------	

def gmso_APC(apcname, scaleset):
	# g mills 17/10/14
	# build multiscale geometric features for the point cloud, or a subset
	# thereof.	
	# note if you're using this on an idx set in a point cloud, you will not
	# get any points outside of the idx contributing to the features.
	
	# 9/1/15
	# all labeled points will end up in the qse (if there are labels) and the
	# skip will be applied to unlabeled points only. i want to be completely
	# clear: all labeled points will be taken in addition to taking 1/skip of
	# the unlabeled points. for now, cluster labels are not preserved.
		
	# INPUT
	# apcname = name of an APC object to load
	# skip = build the features for 1 in every *skip* points. if 1 or 0, all
			# points.
	# scaleset = list of tuples of analysis scales and ssp voxel edges like:
	# [ (biggest vox, array[biggest scales in descending order...])]
	#   (next biggest vox, array[next biggest scale after first set...])]
	# ...
	#   ( smallest vox, array[small scale... smallest scale]) ]
	
	# PARAMETERS
	apc_dir='APC/'
	minpoints=100		# minimum number of points in a metapartition
	budget=1700000000	# space (bytes) needed on the GPU for G_MSO
	ssplabelnum=999		# this label means a point is unlabeled. 
	imax=13000			# maximum size of a partition in G_MSO function
	
	# OUTPUT
	# saves features and their indices in the point cloud to *apc*'s dictionary
	
	
	# load up the APC
	apc=pickle.load(open(apc_dir+apcname+'.pkl','rb'))
	
	# pick an idx set to which to limit the analysis and figure out our skip
	# setting. if there are labels, get em.
	apc.query_keys()
	tag=input("use which idx set? 0 to use none of em >>> ")
	if tag=='0':
		tag=0
		labels=0
		outlabels=0	# output; ignore me
		print('there are ' + str(apc.inc.shape[0]) + ' unlabeled points available')
	else:
		# get labels. if no labels in the idx set, no big deal.
		labels=apc.items[tag][1]
		if isinstance(labels,numpy.ndarray):
			indices=apc.items[tag][0]
			lsum=(labels!=ssplabelnum).sum()
			# just to be safe, if a point doesn't get a label it'll get the
			# non-label
			relabels=numpy.zeros(apc.inc.shape[0])+ssplabelnum
			numpy.put(relabels,indices,labels,mode='wrap')
			# initialize label output
			outlabels=numpy.zeros(0)
			print('features will be generated for ' + str(lsum) + ' labeled points.')
			print('there are ' + str(apc.inc.shape[0]-lsum) + ' unlabeled points.')
	
		
	skip=int(input('enter skip number for unlabeled points >>> '))
			
	# name the output feature set
	featname = input('name the new feature set >>> ')
	
	# calculate size of the features
	fsize=sum([len(x[1]) for x in scaleset])*4
			
	# initialize output
	feats=numpy.zeros((0,fsize+1))
	if isinstance(labels,numpy.ndarray):
		outlabels=numpy.zeros(0)
	else:
		outlabels=0
		
	# get the largest scale
	smax = scaleset[0][1][0]
	
	# load the point cloud on the GPU so we can query the part list with it
	apc.gpu_inc()
	
	# check if there's enough space left over for G_MSO to do its job
	if cuda.mem_get_info()[0]<budget:
		shuffler=True
	else:
		shuffler=False	
	
	# loop over the partition array 
	prange=apc.partlist.shape[0]
	
	# time the whole thing
	outime=time.clock()
	
	for p in range(prange):
		print('starting metapartition ' + str(p+1) + ' of ' + str(prange))
		# pull the search space and query set indices 
		sspidx,qseidx=apc.query_partlist(p,smax,tag)
	
		# skip the requisite entries in query set partition index
		if skip>1:
			# if we have labels then we definitely want to get features for all
			# labeled points.
			if isinstance(labels,numpy.ndarray):
				# get a label set local to this metapartition
				mplabs=relabels.take(qseidx)
				# this takes the indices to labeled query set points in the
				# metapartition
				labqseidx=numpy.extract(mplabs!=ssplabelnum,qseidx)
				# this takes indices to all the unlabeled query set points
				unqseidx=numpy.extract(mplabs==ssplabelnum,qseidx)
				# now take a uniform random subset of them
				grabidx=numpy.random.permutation(unqseidx.size)[:int(unqseidx.size/skip)]
				unqseidx=unqseidx.take(grabidx)
				# and concatenate the labeled and unlabeled point indices
				qseidx=numpy.append(unqseidx,labqseidx)
			else:
			# otherwise just get a random subset of all the indices
				grabidx=numpy.random.permutation(qseidx.size)[:int(qseidx.size/skip)]
				qseidx=qseidx.take(grabidx)
		
		if sspidx.size>minpoints and qseidx.size>minpoints:
			
			sspidx=sspidx.astype(numpy.uint32)
			qseidx=qseidx.astype(numpy.uint32)
			# get query set and search space partitions-- POINTS ONLY
			ssp=apc.inc.take(sspidx,axis=0)
			qse=apc.inc.take(qseidx,axis=0)
		
			# build output vector holding array
			ovh=numpy.zeros((qse.shape[0],fsize+1))
			rrl=numpy.zeros(qse.shape[0])
			
			# index holder for the starting write column of output block
			dc=1
			
			# clear the GPU if we decided the point cloud it holds is too big
			if shuffler:
				apc.gpu_inc_purge()
			print("processing "+str(qse.shape[0])+" query set points")
			
			# loop over scale/ voxel edge combinations to fill holding array
			for s in scaleset:
				# get the scales and voxel edge length to use
				vxl=s[0]
				sc=s[1]
				ss=sc.size	# num scales being processed in this instance
			
				# build features
				oc=mso.G_MSO(qse,ssp,vxl,sc,imax=imax)
		
				# split off the indices (int type needed)
				oci=numpy.int64(oc[:,0])
		
				# slot in the features-- scale-major now
				ovh[oci,dc:dc+ss*4] = oc[:,1:]
		
				# now slot in the indices we built features for: if a point gets
				# a feature from at least one pass in g_mso it will be
				# represented in the index set. use the qse index set to map
				# the given indices back to the original point cloud
				ovh[oci,0]=qseidx.take(oci)
				if isinstance(labels,numpy.ndarray):
					rrl[oci]=relabels.take(qseidx.take(oci))
			
				dc+=ss*4		# update starting index
		
			# stack to output
			feats=numpy.vstack((feats,ovh))
			if isinstance(labels,numpy.ndarray):
				outlabels=numpy.hstack((outlabels,rrl))		
			
	# clean up the gpu
	apc.gpu_inc_purge()
	
	num=feats.shape[0]
	outime=time.clock()-outime
	rate=num/outime
	print(str(num) + ' points processed at a total, final rate of ' +str(rate)+' points/sec.')
	
	# save these results to the APC and put back on HDD
	apc.add_idx(featname,feats[:,0],outlabels,0,feats[:,1:],scaleset)
	closer(apc,apcname)
			
#-------------------------------------------------------------------------------				

def voxel_gang(apcs):
	# g mills 4/12/14
	# does sequential voxel analysis, as below. maintains 2 numpy binaries of
	# results (both population numbers and percent) to save time when building
	# figures. plots cumulative results of both.
	
	# INPUT
	# apcs = python list of apc names. ['apc1','apc2'], like that. or [] if you
	# just want to view results.
	
	# PARAMETERS
	apc_dir='APC/'	
	save_dir='APC/demos/voxes/'	# put binaries here
	pop_bin='vox_pops.npy'		# binary names
	per_bin='vox_pers.npy'
	name_bin='vox_names.npy'
	voxes=numpy.linspace(0.05,1.5,30)
	
	# OUTPUT
	# saves numpy binary files and draws plots
	
	
	# first we'll try and load up the extant results (and start up if not there)
	try: 
		pop=numpy.load(save_dir+pop_bin)
		per=numpy.load(save_dir+per_bin)
		names=numpy.load(save_dir+name_bin)
	except FileNotFoundError:
		print('files not found; making fresh')
		pop=numpy.zeros((0,voxes.size))
		per=pop.copy()
		names=numpy.zeros(0)
	
	# assuming we have work to do, do it.
	if apcs:
		# hold the populations of the inputs
		pops=numpy.zeros(len(apcs))
	
		# hold the scores
		rpop=numpy.zeros((len(apcs),voxes.size))
	
	
		# use voxeltest to get the results
		for a in enumerate(apcs):
			rpop[a[0]],pops[a[0]]=voxeltest(a[1],voxes)
	
		# process the percentages of the populations
		rper=(rpop/pops.reshape(-1,1))*100
	
		# save results
		pop=numpy.vstack((pop,rpop))
		per=numpy.vstack((per,rper))
		names=numpy.hstack((names,numpy.array(apcs)))
		numpy.save(save_dir+pop_bin,pop)
		numpy.save(save_dir+per_bin,per)
		numpy.save(save_dir+name_bin,names)
	
	# view results
	font={'size':'17'}
	plt.figure()
	voxes*=100
	
	# population first
	xnew=numpy.linspace(voxes[0],voxes[-1],300)	
	for p in enumerate(pop):
		plt.plot(xnew,spline(voxes,p[1],xnew),label=names[p[0]])

	plt.ylabel('Voxel Population',**font)
	plt.xlabel('Voxel Edge Length (cm)',**font)	
	plt.xticks(voxes[numpy.arange(0,voxes.size,2)].round(2))#astype(numpy.uint))	
	plt.legend()#loc=8)	
	plt.grid(linestyle='-')	
	plt.savefig(save_dir+'voxel_population.png',dpi=300)
	
	plt.figure()
	# now percentage
	for p in enumerate(per):
		plt.plot(xnew,spline(voxes,p[1],xnew),label=names[p[0]])
		
	plt.ylabel('Voxel Population (% Original)',**font)
	plt.xlabel('Voxel Edge Length (cm)',**font)	
	plt.xticks(voxes[numpy.arange(0,voxes.size,2)].round(2))#astype(numpy.uint))	
	plt.legend(loc=8)	
	plt.grid(linestyle='-')	
	plt.ylim(0,100)
	plt.savefig(save_dir+'voxel_percentage.png',dpi=300)
		
	


def voxeltest(apcname,voxes):
	# g mills 4/12/14
	# calculates population of the point cloud in *apcname* at given voxel
	# edge lengths.
	
	# INPUT
	# apcname = name of apc
	# voxes = numpy 1d array of voxel edge lengths. linspace, arange etc
	
	# PARAMETERS
	apc_dir='APC/'	
		
	# OUTPUT
	# outv = numpy array of point cloud sizes
	# pop = population of point cloud in natural form
	outv=numpy.zeros_like(voxes)
	
	
	# load up the APC
	apc=pickle.load(open(apc_dir+apcname+'.pkl','rb'))
	
	# just get the point cloud
	inc=apc.inc	
	
	for v in enumerate(voxes):
		ssp=mso.double_vox(inc,v[1])
		outv[v[0]]=ssp.shape[0]
	
	return outv,inc.shape[0]


#-------------------------------------------------------------------------------				

def multiclass_self(apcname):
	# g mills 10/1/15
	# multiclass supervised learning using an apc with a labeled, featured IDX
	# set. takes all null-labeled points and uses a classifier to assign a label
	# to them, then saves to a new idx set if desired. 
	
	# note that as it stands now, hand labeled points do not migrate to the
	# new idx set.
	
	# 29/5/15 UPDATE:
	# return a classifier for sake of composability. allow training on
	# arbitrary unbalanced subset of labels. scale/ feature reduction cut.
	
	# INPUT
	# apcname = only one APC 
	
	# PARAMETERS
	apc_dir='APC/'			# directory where the APCs live
	demo_dir='APC/demos/'	# demo file directory
	runflag=1		 
	null_label=999			# non-label label
	
	# OUTPUT
	# clf = the trained classifier
	
	
	# load up the test point cloud--------------------------
	apc=opener(apcname)
	
	# get features
	feats, fname = get_feats(apc)
	
	# get their size
	scs=apc.items[fname][4]
	
	# get out the index sets
	incidx=apc.items[fname][0].astype(numpy.uint32)
	
	# make local indices
	localidx=numpy.arange(apc.items[fname][0].size)	
	
	# get the label set
	labels=apc.items[fname][1].astype(numpy.uint32)
	
	# local index set for label points (to get features ONLY)
	lidx=numpy.extract(labels!=null_label,localidx)
	
	# local index set for test points (to get features)
	tidx=numpy.extract(labels==null_label,localidx)
	
	# global index set for test points (to view point cloud)
	gtidx=incidx.take(lidx)
	points=apc.inc.take(gtidx,axis=0)
	
	# get the labels belonging to observations in the labeled set
	labels=labels.take(lidx)
	# indices to the label features
	lfidx=numpy.arange(labels.size)
	
	# tidy up
	gc.collect()
	
	# figure out how many labels there are
	numlabs=labels.max()+1
	if numlabs>10:
		print(numlabs)
		print((labels>10).sum())
	
	
	# segregate the indices belonging to different labels
	labset=[]
	for n in range(numlabs):
		labset+=[numpy.extract(labels==n,lfidx)]
	# get the label populations
	labpops=numpy.asarray([s.size for s in labset])
	print('label populations:')
	for n in range(numlabs):
		print(str(n)+': ' + str(labpops[n]))
	

	while runflag==1:
		
		# if we're using composite features, we won't have any null-labeled
		# points in test data because we don't have gmso data for them. so use
		# the label points instead.
		lfeats=feats.take(lidx,axis=0)
				
				
		# BALANCED SAMPLING------------------------------------	
		if int(input('balanced sampling? 1/0 >>> ')):
			# figure out how many points will be in training and validation sets
			tnum=int(numpy.floor(.5*labpops.min()))
			vnum=tnum	
			# shuffle the index sets
			for n in range(numlabs):
				numpy.random.shuffle(labset[n])
			# get an index set to the label index sets (will be same size as
			# smaller index sets)
			perm=numpy.random.permutation(tnum*2)
			# take first half as the training set and the second as validation
			tgrab=perm[:tnum]
			vgrab=perm[tnum:]
			tset=numpy.zeros((0,lfeats.shape[1]))
			tlabels=numpy.zeros(0)
			vset=numpy.zeros((0,lfeats.shape[1]))
			vlabels=numpy.zeros(0)
			for n in range(numlabs):
				tset=numpy.vstack((tset,lfeats.take(labset[n].take(tgrab),axis=0)))
				tlabels=numpy.hstack((tlabels,numpy.ones(tnum)*n))
				vset=numpy.vstack((vset,lfeats.take(labset[n].take(vgrab),axis=0)))
				vlabels=numpy.hstack((vlabels,numpy.ones(tnum)*n))
				
		else:
			print('manually designating training set sizes per class')
			
			# initialize feature and label sets
			tset=numpy.zeros((0,lfeats.shape[1]))
			tlabels=numpy.zeros(0)
			vset=numpy.zeros((0,lfeats.shape[1]))
			vlabels=numpy.zeros(0)
			
			# fill from each class
			for n in range(numlabs):
				print('label ' +str(n)+': ' + str(labpops[n]))
				tnum = int(input('train on how many? >>> '))
				
				# random index to label set
				perm=numpy.random.permutation(labpops[n])
				tgrab=perm[:tnum]
				vgrab=perm[tnum:tnum*2]
				
				# obtain the features and labels
				tset=numpy.vstack((tset,lfeats.take(labset[n].take(tgrab),axis=0)))
				tlabels=numpy.hstack((tlabels,numpy.ones(tnum)*n))
				vset=numpy.vstack((vset,lfeats.take(labset[n].take(vgrab),axis=0)))
				vlabels=numpy.hstack((vlabels,numpy.ones(tnum)*n))
		
		
		# BUILD CLASSIFIER------------------------------------
		# now with safe(ish) input handling
		clf=False
		while isinstance(clf,bool):
			classifier=input('what classifier to use? svm/rf/erf/nb/knn/sgd/rpte >>> ')
			clf=param_classifier(classifier)
		
		# a kernel map for the feature space might be appropriate
		if classifier in ['svm','knn','sgd','rpte']:
			kern=int(input('try a kernel approximation on the data? 1/0 >>> '))
			if kern:
				kmap=input('use [n]ystroem or [r]bf sampler to approximate a kernel mapping? >>> ')
				comps=int(input('how many components? >>> '))
				# try somewhere 10-1000 ish?
				if kmap=='n':
					fmap=Nystroem(gamma=5,random_state=1,n_components=comps)
				elif kmap=='r':
					fmap=RBFSampler(gamma=.2,random_state=1,n_components=comps)
				tset=fmap.fit_transform(tset)
				vset=fmap.transform(vset)
				lfeats=fmap.transform(lfeats)
				
				
		# fit the classifier
		clf.fit(tset,tlabels)
		

		# RF importance-- note this doesn't work with composite features!
		if classifier in ["rf","erf"]:
			if int(input("view feature importance? 1/0 >>> ")):
				# get number of features
				numfeats=int(lfeats.shape[1]/scs)
				# get importances
				rfi=clf.feature_importances_.copy()
				# break up by scale
				scale_importances=rfi.reshape(-1,numfeats).sum(1)
				# plot
				plt.figure()
				plt.bar(numpy.arange(scale_importances.size),scale_importances)
				plt.xlabel("scale index")
				plt.ylabel("total importance")
				plt.show()
				# drop scales?
				if int(input("remove some scales from analysis? 1/0 >>> ")):
					drops=input("enter scale indices to remove, sep by space >>> ")
					drops=numpy.asarray([int(drop) for drop in drops.split(" ")])
					# find the scales to keep
					keeps=numpy.setdiff1d(numpy.arange(scale_importances.size),drops)
					# transform to feature indices
					keeps=numpy.asarray([ks*numfeats+numpy.arange(numfeats) for ks in keeps]).flatten()
					# take these scales
					tset=tset.take(keeps.astype(numpy.int64),axis=1)
					vset=vset.take(keeps.astype(numpy.int64),axis=1)
					lfeats=lfeats.take(keeps.astype(numpy.int64),axis=1)
					# retrain classifier
					clf=param_classifier(classifier)
					clf.fit(tset,tlabels)
					
		# and clean up
		del tset
		
		# VALIDATE RESULTS------------------------------------
		# classify validation set
		l=clf.predict(vset)
				
		# CONFUSION RESULTS------------------------------------
		conf=ml.mc_confusion(l,vlabels)
		conf_plotter(conf)
		
		# VIEW RESULTS------------------------------------
		if int(input('print a cloud to view classification results? 1/0 >>> ')):
			sr = int(input('enter a skip ratio: 1-whatever >>> '))
			skipidx=numpy.arange(0,lfeats.shape[0],sr)
			vis_labels(clf,points.take(skipidx,axis=0),lfeats.take(skipidx,axis=0),demo_dir+'zap_mc_test.txt')
		
		# SAVE RESULTS------------------------------------
		if int(input('save these labels back to the test APC? 1/0 >>> ')):
			assigned=clf.predict(lfeats)	
			del clf		# we'll go ahead and dump out the potentially huge clf;
			# feel free to comment out if you want it.
			out_fname=input('name a new label set >>> ')
			if int(input('include original features? 1/0 >>> ')):
				apc.add_idx(out_fname,gtidx,assigned,0,feats.take(tidx,axis=0),apc.items[fname][5])
			else:
				apc.add_idx(out_fname,gtidx,assigned,0,0,apc.items[fname][5])
			closer(apc,apcname)			
			
		if int(input('save these label probabilities to the test APC? 1/0 >>> ')):
			try:
				probs=clf.predict_proba(lfeats)
				out_fname=input('name the new feature set >>> ')
				apc.add_idx(out_fname,gtidx,labels,0,probs,apc.items[fname][5])
				closer(apc,apcname)
			except AttributeError:
				print("we can't predict probability with this classifier. try another.")
						
		gc.collect()
		runflag=int(input('keep testing? 1/0 >>> '))	
	
	"""	
	saveflag=int(input('save this classifier? 1/0 >>> '))
	if saveflag==1:
		return clf
	#"""		
	return clf	
	
	
			
#-------------------------------------------------------------------------------
				
def multiclass_blind(apcname):
	# g mills 30/8/15
	# multiclass supervised learning using an apc with a labeled, featured IDX
	# set. takes all null-labeled points and uses a classifier to assign a label
	# to them, then saves to a new idx set if desired. 
	
	# this version does not perform cross-validation.

	# choices short-circuited 1/9/15 for evaluation scripting
	
	# INPUT
	# apcname = only one APC 
	
	# PARAMETERS
	apc_dir='APC/'			# directory where the APCs live
	demo_dir='APC/demos/'	# demo file directory
	runflag=1		 
	null_label=999			# non-label label
	#fname="o4-v-vm2-1g-merge"		# do we already know the feature name?
	#fname="o4-v-2g-merge"
	
	# OUTPUT
	# clf = the trained classifier
	
	
	# load up the test point cloud--------------------------
	apc=opener(apcname)
	
	# get features
	feats, fname = get_feats(apc)#,tag=fname)
	
	# get their size
	scs=apc.items[fname][4]
	
	# get out the index sets
	incidx=apc.items[fname][0].astype(numpy.uint32)
	
	# make local indices
	localidx=numpy.arange(apc.items[fname][0].size)	
	
	# get the label set
	labels=apc.items[fname][1].astype(numpy.uint32)
	
	# local index set for label points (to get features ONLY)
	lidx=numpy.extract(labels!=null_label,localidx)
	
	# local index set for test points (to get features)
	tidx=numpy.extract(labels==null_label,localidx)
	
	# global index set for test points (to view point cloud)
	gtidx=incidx.take(lidx)
	points=apc.inc.take(gtidx,axis=0)
	
	# get the labels belonging to observations in the labeled set
	labels=labels.take(lidx)
	# indices to the label features
	lfidx=numpy.arange(labels.size)
	
	# tidy up
	gc.collect()
	
	# figure out how many labels there are
	numlabs=labels.max()+1
	if numlabs>10:
		print(numlabs)
		print((labels>10).sum())
	
	
	# segregate the indices belonging to different labels
	labset=[]
	for n in range(numlabs):
		labset+=[numpy.extract(labels==n,lfidx)]
	# get the label populations
	labpops=numpy.asarray([s.size for s in labset])
	#print('label populations:')
	#for n in range(numlabs):
	#	print(str(n)+': ' + str(labpops[n]))
	

	while runflag==1:
		
		# if we're using composite features, we won't have any null-labeled
		# points in test data because we don't have gmso data for them. so use
		# the label points instead.
		lfeats=feats.take(lidx,axis=0)
				
				
		# BALANCED SAMPLING------------------------------------	
		if True: #int(input('balanced sampling? 1/0 >>> ')):
			# figure out how many points will be in training and validation sets
			tnum=int(labpops.min())
			# the way i'm doing this is actually redundant. it should be safe
			# to EITHER shuffle the index sets or use numpy.random.perm to
			# index the unshuffled index sets. not sure which way would be 
			# more efficient but i am sure it doesn't realistically matter.
			# shuffle the index sets
			for n in range(numlabs):
				numpy.random.shuffle(labset[n])
			# get an index set to the label index sets (will be same size as
			# smaller index sets)
			perm=numpy.random.permutation(tnum)
			# take first half as the training set and the second as validation
			tgrab=perm[:tnum]
			tset=numpy.zeros((0,lfeats.shape[1]))
			tlabels=numpy.zeros(0)
			for n in range(numlabs):
				tset=numpy.vstack((tset,lfeats.take(labset[n].take(tgrab),axis=0)))
				tlabels=numpy.hstack((tlabels,numpy.ones(tnum)*n))
				
		else:
			print('manually designating training set sizes per class')
			
			# initialize feature and label sets
			tset=numpy.zeros((0,lfeats.shape[1]))
			tlabels=numpy.zeros(0)
			
			# fill from each class
			for n in range(numlabs):
				print('label ' +str(n)+': ' + str(labpops[n]))
				tnum = int(input('train on how many? >>> '))
				
				# random index to label set
				perm=numpy.random.permutation(labpops[n])
				tgrab=perm[:tnum]
				
				# obtain the features and labels
				tset=numpy.vstack((tset,lfeats.take(labset[n].take(tgrab),axis=0)))
				tlabels=numpy.hstack((tlabels,numpy.ones(tnum)*n))
		
		
		# BUILD CLASSIFIER------------------------------------
		# now with safe(ish) input handling
		clf=False
		while isinstance(clf,bool):
			#classifier=input('what classifier to use? svm/rf/erf/nb/knn/sgd/rpte >>> ')
			#clf=param_classifier(classifier)
			clf=ExtraTreesClassifier(n_jobs=4,n_estimators=30, criterion="gini", bootstrap=False)
				
		# fit the classifier
		clf.fit(tset,tlabels)

		# and clean up
		del tset
		
		
		# VIEW RESULTS------------------------------------
		if False: #int(input('print a cloud to view classification results? 1/0 >>> ')):
			sr = int(input('enter a skip ratio: 1-whatever >>> '))
			skipidx=numpy.arange(0,lfeats.shape[0],sr)
			vis_labels(clf,points.take(skipidx,axis=0),lfeats.take(skipidx,axis=0),demo_dir+'zap_mc_test.txt')
		
		# SAVE RESULTS------------------------------------
		if False: #int(input('save these labels back to the test APC? 1/0 >>> ')):
			assigned=clf.predict(lfeats)	
			del clf		# we'll go ahead and dump out the potentially huge clf;
			# feel free to comment out if you want it.
			out_fname=input('name a new label set >>> ')
			if int(input('include original features? 1/0 >>> ')):
				apc.add_idx(out_fname,gtidx,assigned,0,feats.take(tidx,axis=0),apc.items[fname][5])
			else:
				apc.add_idx(out_fname,gtidx,assigned,0,0,apc.items[fname][5])
			closer(apc,apcname)			
			
		if False: #int(input('save these label probabilities to the test APC? 1/0 >>> ')):
			try:
				probs=clf.predict_proba(lfeats)
				out_fname=input('name the new feature set >>> ')
				apc.add_idx(out_fname,gtidx,labels,0,probs,apc.items[fname][5])
				closer(apc,apcname)
			except AttributeError:
				print("we can't predict probability with this classifier. try another.")
						
		gc.collect()
		runflag=False #int(input('keep testing? 1/0 >>> '))	
	
	"""	
	saveflag=int(input('save this classifier? 1/0 >>> '))
	if saveflag==1:
		return clf
	#"""		
	return clf	
	
	
			
#-------------------------------------------------------------------------------

def binary_self(apcname):
	# g mills 24/8/15
	# binary supervised learning using an apc with a labeled, featured IDX
	# set. takes all null-labeled points and uses a classifier to assign a label
	# to them, then saves to a new idx set if desired. 
	
	# note that as it stands now, hand labeled points do not migrate to the
	# new idx set.
	
	# INPUT
	# apcname = only one APC 
	
	# PARAMETERS
	apc_dir='APC/'			# directory where the APCs live
	demo_dir='APC/demos/'	# demo file directory
	runflag=1		 
	null_label=999			# non-label label
	
	# OUTPUT
	# clf = the trained classifier
	
	
	# load up the test point cloud--------------------------
	apc=opener(apcname)
	
	# get features
	feats, fname = get_feats(apc)
	
	# get their size
	scs=apc.items[fname][4]
	
	# get out the index sets
	incidx=apc.items[fname][0].astype(numpy.uint32)
	
	# make local indices
	localidx=numpy.arange(apc.items[fname][0].size)	
	
	# get the label set
	labels=apc.items[fname][1].astype(numpy.uint32)
	
	# local index set for label points (to get features ONLY)
	lidx=numpy.extract(labels!=null_label,localidx)
	
	# local index set for test points (to get features)
	tidx=numpy.extract(labels==null_label,localidx)
	
	# global index set for test points (to view point cloud)
	gtidx=incidx.take(lidx)
	points=apc.inc.take(gtidx,axis=0)
	
	# get the labels belonging to observations in the labeled set
	labels=labels.take(lidx)
	# indices to the label features
	lfidx=numpy.arange(labels.size)
	
	# tidy up
	gc.collect()

	# figure out how many labels there are
	numlabs=labels.max()+1
	if numlabs>10:
		print(numlabs)
		print((labels>10).sum())
	
	
	# segregate the indices belonging to different labels
	labset=[]
	for n in range(numlabs):
		labset+=[numpy.extract(labels==n,lfidx)]
	# get the label populations
	labpops=numpy.asarray([s.size for s in labset])
	print('label populations:')
	for n in range(numlabs):
		print(str(n)+': ' + str(labpops[n]))
	

	while runflag==1:
		
		# if we're using composite features, we won't have any null-labeled
		# points in test data because we don't have gmso data for them. so use
		# the label points instead.
		efeats=feats.take(lidx,axis=0)
		lfeats=feats.take(lidx,axis=0)
				
				
		# SAMPLING------------------------------------	
		print('manually designating training set sizes per class')
			
		# initialize feature and label sets
		tset=numpy.zeros((0,lfeats.shape[1]))
		tlabels=numpy.zeros(0)
		vset=numpy.zeros((0,lfeats.shape[1]))
		vlabels=numpy.zeros(0)
		
		positive_ID=int(input("which class is the positive ID? >>> "))
		
		# fill from each class
		for n in range(numlabs):
			print('label ' +str(n)+': ' + str(labpops[n]))
			tnum = int(input('train on how many? >>> '))
			
			# random index to label set
			perm=numpy.random.permutation(labpops[n])
			tgrab=perm[:tnum]
			vgrab=perm[tnum:tnum*2]
			
			# obtain the features
			tset=numpy.vstack((tset,lfeats.take(labset[n].take(tgrab),axis=0)))
			vset=numpy.vstack((vset,lfeats.take(labset[n].take(vgrab),axis=0)))
			
			# if this is the positive ID, our label is 1. otherwise it's 0
			if n==positive_ID:
				nlabels=numpy.ones(tnum)
			else:
				nlabels=numpy.zeros(tnum)
				
			tlabels=numpy.hstack((tlabels,nlabels))
			vlabels=numpy.hstack((vlabels,nlabels))
	
		
		# BUILD CLASSIFIER------------------------------------
		# now with safe(ish) input handling
		clf=False
		while isinstance(clf,bool):
			classifier=input('what classifier to use? svm/rf/erf/nb/knn/sgd/rpte >>> ')
			clf=param_classifier(classifier)
		
		# a kernel map for the feature space might be appropriate
		if classifier in ['svm','knn','sgd','rpte']:
			kern=int(input('try a kernel approximation on the data? 1/0 >>> '))
			if kern:
				kmap=input('use [n]ystroem or [r]bf sampler to approximate a kernel mapping? >>> ')
				comps=int(input('how many components? >>> '))
				# try somewhere 10-1000 ish?
				if kmap=='n':
					fmap=Nystroem(gamma=5,random_state=1,n_components=comps)
				elif kmap=='r':
					fmap=RBFSampler(gamma=.2,random_state=1,n_components=comps)
				tset=fmap.fit_transform(tset)
				vset=fmap.transform(vset)
				
		# or else a random trees embedding 
		elif classifier=='nb':
			trees=int(input('how many trees in the embedding? >>> '))
			depth=int(input('how deep should the trees go? >>> '))
			emb=RandomTreesEmbedding(n_estimators=trees,n_jobs=6,max_depth=depth)
			emb.fit(alfeats)
			tset=emb.transform(tset)
			vset=emb.transform(vset)
			efeats=emb.transform(efeats)
				
		# fit the classifier and clean up
		clf.fit(tset,tlabels)
		del tset

		# VALIDATE RESULTS------------------------------------
		# classify validation set
		l=clf.predict(vset)
				
		# CONFUSION RESULTS------------------------------------
		conf=ml.mc_confusion(l,vlabels)
		conf_plotter(conf)
		
		
		# VIEW RESULTS------------------------------------
		if int(input('print a cloud to view classification results? 1/0 >>> ')):
			sr = int(input('enter a skip ratio: 1-whatever >>> '))
			skipidx=numpy.arange(0,efeats.shape[0],sr)
			# predict classes
			assigned=clf.predict(efeats.take(skipidx,axis=0)).reshape(-1,1)			
			# associate points with their classes
			p=numpy.hstack((points.take(skipidx,axis=0),assigned))
			try:
				probs=clf.predict_proba(efeats)
				pw=ml.colorize_mc_prob(p,probs)
			except AttributeError:
				pw=ml.colorize_multiclass(p)
			numpy.savetxt(demo_dir+'zap_mc_test.txt',pw,delimiter=',')
			
		
		# SAVE RESULTS------------------------------------
		if int(input('save these labels back to the test APC? 1/0 >>> ')):
			assigned=clf.predict(efeats)	
			del clf		# we'll go ahead and dump out the potentially huge clf;
			# feel free to comment out if you want it.
			out_fname=input('name a new label set >>> ')
			if int(input('include original features? 1/0 >>> ')):
				apc.add_idx(out_fname,gtidx,assigned,0,feats.take(tidx,axis=0),apc.items[fname][5])
			else:
				apc.add_idx(out_fname,gtidx,assigned,0,0,apc.items[fname][5])
			closer(apc,apcname)				
		
		runflag=int(input('keep testing? 1/0 >>> '))	
	
	"""	
	saveflag=int(input('save this classifier? 1/0 >>> '))
	if saveflag==1:
		return clf
	#"""		
	return clf	

#-------------------------------------------------------------------------------				

def param_classifier(classifier):
	# g mills 24/8/15
	# classifier selection and parameterization, factored out of multiclass
	# scripts.
	
	# INPUT
	# classifier = abbreviated classifier name, as below
	
	# OUTPUT
	# returns a parameterized classifier
	

	if classifier=='svm':
		clf=svm.LinearSVC()
		
	elif classifier=='rf':
		trees=int(input('how many trees in the forest? >>> '))
		crit=input('which criterion? gini or entropy >>> ')
		boot=bool(int(input('bootstrap samples? 1/0 >>> ')))
		clf=RandomForestClassifier(n_jobs=6,n_estimators=trees, criterion=crit, bootstrap=boot)
		
	elif classifier=='erf':
		trees=int(input('how many trees in the forest? >>> '))
		crit=input('which criterion? gini or entropy >>> ')
		boot=bool(int(input('bootstrap samples? 1/0 >>> ')))
		clf=ExtraTreesClassifier(n_jobs=6,n_estimators=trees, criterion=crit, bootstrap=boot)
		
	elif classifier=='nb':
		clf=BernoulliNB()

	elif classifier=='knn':
		# k nearest neighbors with a ball tree. this one might hurt to train.
		K=int(input('how many neighbors? >>> '))
		leaf = int(input('how many obs in a leaf? >>> '))
		# rest of the default parameters set it up for euclidian distance metric
		clf=neighbors.KNeighborsClassifier(n_neighbors=K,algorithm='ball_tree',leaf_size=leaf)

	elif classifier=='sgd':
		# stochastic gradient descent with a linear SVM style decision function
		pen=input('what regularization term? l2, l1 or elasticnet >>> ')
		fint=int(input('fit intercept? (1 if you did not sphere/pca the data) 1/0 >>> '))
		clf=SGDClassifier(loss='hinge',penalty=pen,n_jobs=6,fit_intercept=fint)
		
	elif classifier=='rpte':
		trees=int(input('how many trees would you like to train? >>> '))
		df=input('which decision function? wmean or wmax >>> ')
		pur=input('set purity target. one or two floats 0-1 >>> ')
		try:
			pur=float(pur)
		except TypeError:
			pur=tuple([float(pu) for pu in pur.split()])
		clf=ml.RPT_ensemble(n_estimators=trees,d_func=df, impurity=pur)
		
	else:
		print('unrecognized input; try again')
		return False
			
	return clf

#-------------------------------------------------------------------------------				

def conf_plotter(conf):
	# g mills 24/8/15
	# plot confusion statistics
	
	# INPUT
	# conf = a confusion matrix
	
	# OUTPUT
	# prints to command line and makes a plot
	
	
	print(conf)
	
	# user and producer scores
	user,prod=ml.user_producer(conf)
	user=numpy.around(user,decimals=2)
	prod=numpy.around(prod,decimals=2)
	
	print('------------------------------------------')
	print('user averages:')
	for av in enumerate(user):
		print('label ' + str(av[0]) + ' : ' + str(av[1]))
	print('------------------------------------------')
	print('producer averages:')
	for av in enumerate(prod):
		print('label ' + str(av[0]) + ' : ' + str(av[1]))
	print('------------------------------------------')
	
	# plot confusion matrix
	conf=ml.dilate_scale(conf,75)		# dilate and scale it for plotting
	plt.imshow(conf)
	plt.show()
	
	
	
#-------------------------------------------------------------------------------				

def three_printer(conf):
	# g mills 1/9/15
	# print tp/fp/fn metrics
	
	# INPUT
	# conf = a confusion matrix
	
	# OUTPUT
	# prints to command line
	
	
	#print(conf)
	
	# user and producer scores
	block=ml.three_metrics(conf)*100
	block=numpy.around(block,decimals=2)
	
	print('true positive:')
	for b in range(block.shape[0]):
		print('label ' + str(b) + ' : ' + str(block[b,0]))
	print('------------------------------------------')
	print('false positive:')
	for b in range(block.shape[0]):
		print('label ' + str(b) + ' : ' + str(block[b,1]))
	print('------------------------------------------')
	print('false negative:')
	for b in range(block.shape[0]):
		print('label ' + str(b) + ' : ' + str(block[b,2]))
	print('------------------------------------------')
	
	

#-------------------------------------------------------------------------------				

def balance_resampler(feats,labels,clf,trials,blind=False):
	# g mills 29/8/15
	# repeatedly performs balanced sampling and validation on a set of labeled
	# feature vectors, returning mean and std confusion matrices.
	
	# INPUT
	# feats = feature set
	# labels = label set corresponding to feats
	# clf = classifier
	# trials = number of trials 
	# blind = operate in non-user-input-mode
	
	# OUTPUT
	# mean = mean confusion matrix
	# std = std confusion matrix
	
	
	# figure out how many labels there are
	numlabs=int(labels.max()+1)
	if numlabs>10:
		print(numlabs)
		print((labels>10).sum())
	
	# index the features and labels
	idx=numpy.arange(labels.size,dtype=numpy.int32)


	# segregate the indices belonging to different labels
	labset=[]
	for n in range(numlabs):
		labset+=[numpy.extract(labels==n,idx)]
	# get the label populations
	labpops=numpy.asarray([s.size for s in labset])
	if not blind:
		print('label populations:')
		for n in range(numlabs):
			print(str(n)+': ' + str(labpops[n]))
		
	# figure out how many points in validation set
	vnum=int(numpy.floor(.5*labpops.min()))
	
	# initialize output accumulator
	cmat = numpy.zeros((numlabs,numlabs,trials))
	
	for t in range(trials):	
		# shuffle the index sets
		for n in range(numlabs):
			numpy.random.shuffle(labset[n])
		# get an index set to the label index sets 
		grab=numpy.random.permutation(vnum)
		vfeats=numpy.zeros((0,feats.shape[1]))
		vlabels=numpy.zeros(0)
		for n in range(numlabs):
			vfeats=numpy.vstack((vfeats,feats.take(labset[n].take(grab),axis=0)))
			vlabels=numpy.hstack((vlabels,numpy.ones(vnum)*n))
		# classify
		assigned=clf.predict(vfeats)
		# confusion matrix
		cmat[:,:,t]=ml.mc_confusion(assigned,vlabels)
	mean=numpy.mean(cmat,2)
	std=numpy.std(cmat,2)
	return mean, std

#-------------------------------------------------------------------------------				

def apc_factor_analysis(apcname,model=None):
	# g mills 30/8/15
	# use factor analysis to derive and/or apply a low-dimensional
	# approximation to the features in an APC
	
	# INPUT
	# apcname = apc to work with
	# model = factor analysis object, or nothing
	
	# PARAMETERS
	apc_dir='APC/'	
	demo_dir=apc_dir+'demos/'
	
	# OUTPUT
	# model = factor analysis object 
	
	# load up the APC
	apc=opener(apcname)
	
	# get features
	feats, fname = get_feats(apc)

	# get the relevant indices and labels
	# tag, (idx, labels, cluster labels, feats, numscales, scales)
	labels=apc.items[fname][1]
	idx=apc.items[fname][0].astype(numpy.uint32)
	
	# decide if we need to train a model
	if model is None:
		tp = int(input("train on how many points? >>> "))
		tfeats = numpy.random.permutation(feats.shape[0])[:tp]
		tfeats = feats.take(tfeats,axis=0)
		n_components = int(input("how many factors do you want? >>> "))
		model=FA(n_components=n_components).fit(tfeats)
		
	# apply the model to the features
	newfeats=model.transform(feats)
	
	outname=input("name the new feature set >>> ")
	apc.add_idx(outname,idx,labels,0,newfeats,apc.items[fname][5])
	closer(apc,apcname)	
	return model
	
	
#-------------------------------------------------------------------------------		
	
def apply_clf_APC(apcname,clf):
	# g mills 28/10/14
	# use a given clf to label a feature set in an APC. predicted labels are
	# put under cluster labels in the same idx set the features came from.
	# first slot labels are preserved.
	
	# choices short-circuited 1/9/15
	
	# INPUT
	# clf = a classifier object with a .predict_proba method
	# apcname = name of the apc to work on
	
	# PARAMETERS
	apc_dir='APC/'	
	runflag=True
	demo_dir=apc_dir+'demos/'
	fname="o4-v-vm2-1g-merge"		# do we already know the feature name?
	#fname="o4-v-2g-merge"
	
	# OUTPUT
	# applies labels to the feature set and saves the APC
	
	
	# load up the APC
	apc=opener(apcname)
	
	# get features
	feats, fname = get_feats(apc)#,tag=fname)
	
	# get the relevant indices and labels
	# tag, (idx, labels, cluster labels, feats, numscales, scales)
	labels=apc.items[fname][1]
	idx=apc.items[fname][0].astype(numpy.uint32)
	
	# we haven't predicted any new labels yet
	assigned = None

	# what are we doing with the clf? we can:
	while runflag:
		choice=input("would you like to [e]valuate accuracy, [v]iew a point cloud, save [l]abels, save [p]robability or [q]uit? >>> ")
	
		# evaluate classification score
		if choice=='e' and isinstance(labels,numpy.ndarray):
			# confusion stats
			trials=5#int(input("how many validation trials? >>> "))
			conf,std=balance_resampler(feats,labels,clf,trials,blind=True)
			return conf
			print("std of trial confusion matrix")
			print(std)
			conf_plotter(conf)
			
		# view a point cloud
		elif choice=='v':
			vis_labels(clf,apc.inc.take(idx,axis=0),feats,demo_dir+'zap_mc_test.txt')
			
		# save labels
		elif choice=='l':
			if assigned is None:
				assigned = clf.predict(feats)
			outname=input("name the new label set >>> ")
			apc.add_idx(outname,idx,assigned,0,0,apc.items[fname][5])
		# save probability as features

		elif choice=='p':
			try:
				proba=clf.predict_proba(feats)
				outname=input("name the new feature set >>> ")
				apc.add_idx(outname,idx,labels,0,proba,apc.items[fname][5])
			except AttributeError:
				print("sorry, this classifier can't estimate probability")

		elif choice=='q':
			runflag=False
		else:
			print("invalid selection. please try again")

	closer(apc,apcname)	


#-------------------------------------------------------------------------------				

def vis_labels(clf,xyz,feats,outpath,precision=3):
	# g mills 27/8/15
	# applies a classifier to a point cloud and prints it out, colorized.
	# if a predict_proba method is available, color will reflect class 
	# probability.
	
	# INPUT
	# clf = classifier
	# xyz = point cloud geometry, in same order as
	# feats = features pertaining to the point cloud
	# outpath = full path and filename of point cloud
	# precision = points past the decimal in output file	
	
	# PARAMETERS
	delimiter=','
	
	# OUTPUT
	# just writes the file where you tell it to
	
	
	# set the format specifier
	fs = '%.'+str(precision)+'f'		
	
	# predict labels/ probabilities
	assigned=clf.predict(feats).reshape(-1,1)		
	# associate points with their classes
	p=numpy.hstack((xyz,assigned))
	# predict probability?
	try:
		probs=clf.predict_proba(feats)
		pw=ml.colorize_mc_prob(p,probs)
	except AttributeError:
		pw=ml.colorize_multiclass(p)
	# save
	numpy.savetxt(outpath,pw,delimiter=delimiter,fmt=fs)

#-------------------------------------------------------------------------------				
#-------------------------------------------------------------------------------				
#-------------------------------------------------------------------------------				
#-------------------------------------------------------------------------------				

#-------------------------------------------------------------------------------				

def embed_plot(apcname):
	# g mills 31/12/14
	# try a 2D tsne embedding of a feature set in *apcname*.

	# INPUT
	# apcname = apc with a featured idx set
	
	# PARAMETERS
	apc_dir='APC/'
	ssplabelnum=999		# labels associated with search space points
	colors=['#FF1493','#00BFFF','#00FF7F','#8A2BE2','#FF8C00']	
	rcolors=numpy.array([[255,20,147],[0,191,255],[0,255,127],[138,43,226],[255,140,0],[192,0,0]])/255
	runflag=True
	lw = '0'	# zero linewidth on the marker borders so they're more legible
	msize = 30	# marker size for legibility
	null_label=999
	
	# OUTPUT
	# outv = a vector of analysis scales
	
	
	# load up the test point cloud--------------------------
	apc=opener(apcname)
	
	# get features
	feats, fname = get_feats(apc)
	
	# get their size
	scs=apc.items[fname][4]
	
	# get out the index sets
	incidx=apc.items[fname][0].astype(numpy.uint32)
	
	# make local indices
	localidx=numpy.arange(apc.items[fname][0].size)	
	
	# get the label set
	labels=apc.items[fname][1].astype(numpy.uint32)
	
	# local index set for label points (to get features ONLY)
	lidx=numpy.extract(labels!=null_label,localidx)
	
	# local index set for test points (to get features)
	tidx=numpy.extract(labels==null_label,localidx)
	
	# global index set for test points (to view point cloud)
	gtidx=incidx.take(lidx)
	points=apc.inc.take(gtidx,axis=0)
	
	# get the labels belonging to observations in the labeled set
	labels=labels.take(lidx)
	# indices to the label features
	lfidx=numpy.arange(labels.size)
	
	# tidy up
	gc.collect()

	# figure out how many labels there are
	numlabs=labels.max()+1
	if numlabs>10:
		print(numlabs)
		print((labels>10).sum())
	
	
	# segregate the indices belonging to different labels
	labset=[]
	for n in range(numlabs):
		labset+=[numpy.extract(labels==n,lfidx)]
	# get the label populations
	labpops=numpy.asarray([s.size for s in labset])
	print('label populations:')
	for n in range(numlabs):
		print(str(n)+': ' + str(labpops[n]))
	

	while runflag:
		
		# if we're using composite features, we won't have any null-labeled
		# points in test data because we don't have gmso data for them. so use
		# the label points instead.
		efeats=feats.take(lidx,axis=0)
				
				
		# SAMPLING------------------------------------	
		print('manually designating training set sizes per class')
			
		# initialize feature and label sets
		tset=numpy.zeros((0,efeats.shape[1]))
		tlabels=numpy.zeros((0,3))

		# fill from each class
		for n in range(numlabs):
			print('label ' +str(n)+': ' + str(labpops[n]))
			tnum = int(input('model with how many? >>> '))
			
			# random index to label set
			perm=numpy.random.permutation(labpops[n])
			tgrab=perm[:tnum]
			
			# obtain the features
			tset=numpy.vstack((tset,efeats.take(labset[n].take(tgrab),axis=0)))
			
			# colorize labels
			nlabels=numpy.tile(rcolors[n],(tnum,1))
			tlabels=numpy.vstack((tlabels,nlabels))


		# sphere
		sphere=int(input("sphere? 1/0 >>> "))
		if sphere:
			whitener=preprocessing.StandardScaler().fit(tset)
			feats=whitener.transform(feats)
	

		n_components = 2
		fig = plt.figure()
		if int(input("use a random state? 1/0 >>> ")):
			random_state=int(input("enter now >>> "))
		else:
			random_state=None
		init=input("initiate with pca or random? pca >>> ")
		n_iter=int(input("iterations? 1000 >>> "))
		alpha=float(input("learning rate? 1000 >>> "))
			
		tsne = manifold.TSNE(n_components=n_components, n_iter=n_iter, init=init, random_state=random_state,learning_rate = alpha)
		Y = tsne.fit_transform(tset)
		plt.scatter(Y[:, 0], Y[:, 1],c=tlabels,linewidth=lw,s=msize)
		plt.title("t-SNE")

		plt.show()	
		
		runflag= int(input("keep embedding? 1/0 >>> "))

#-------------------------------------------------------------------------------
	
def merge_features(apcname):
	# g mills 28/5/15
	# compose multiple feature sets into one. note that we will only output
	# intact feature vectors. if an index is missing from one or more feature
	# sets, it will be dropped.
	
	# INPUT
	# apcname = name of the apc to use
	
	# PARAMETERS
	apc_dir='APC/'			# directory where the APCs live
	demo_dir='APC/demos/'	# demo file directory
	
	# OUTPUT
	# adds a new idx set to an APC
	
	
	# load it
	apc=opener(apcname)
	
	# get the first feature set. it doesn't matter if it's the biggest or the
	# smallest of the feature sets to be merged, since the output will be the
	# same cardinality as the smallest.
	outfeats, fname = get_feats(apc)
	# labels
	try:
		outlabs=apc.items[fname][1].astype(numpy.int64)
		labflag=True
	except AttributeError:
		outlabs=0
		labflag=False
		
	# indices
	outidx=apc.items[fname][0].astype(numpy.int64)
	
	# work loop
	while int(input('add features? 1/0 >>> ')): 
		# get new feature vectors
		feats, fname = get_feats(apc)
		
		# get new indices
		idx = apc.items[fname][0].astype(numpy.int64)
		
		# get new labels
		if labflag:
			labels = apc.items[fname][1].astype(numpy.int64)
		
		# find the intersection of this idx set and the output idx set
		inter = numpy.intersect1d(outidx,idx,assume_unique=True)
		
		# get boolean masks for this idx set and the output idx set
		outmask = numpy.in1d(outidx,inter,assume_unique=True)
		mask = numpy.in1d(idx,inter,assume_unique=True)
		
		# note that the idx set we get from the APC's dictionary is already
		# sorted since it was uniqued when it was first stored. so after using
		# the intersection masks, the input and output features will just match.
		feats=numpy.compress(mask,feats,axis=0)		
		outfeats = numpy.compress(outmask,outfeats,axis=0)
		outfeats = numpy.hstack((outfeats,feats))
		outidx = numpy.extract(outmask,outidx)
		if labflag:
			outlabs = numpy.extract(outmask,outlabs)
		
	
	# add the new features to the APC
	newname = input('name the new feature set >>> ')
	apc.add_idx(newname,outidx,outlabs,0,outfeats,0)
	
	# save it
	closer(apc,apcname)
	

def chop_features(apcname):
	# g mills 26/8/15
	# delete columns of a feature array in an apc
	
	# INPUT
	# apcname = name of the apc to modify
	
	# PARAMETERS
	apc_dir='APC/'			# directory where the APCs live
	demo_dir='APC/demos/'	# demo file directory
	
	# OUTPUT
	# adds a new idx set to an APC
	
	
	# load it
	apc=opener(apcname)
	
	# get the feature set
	outfeats, fname = get_feats(apc)
	# labels
	try:
		outlabs=apc.items[fname][1].astype(numpy.int64)
	except AttributeError:
		outlabs=0
		
	# indices
	outidx=apc.items[fname][0].astype(numpy.int64)
	
	# get the std of each feature and print it w/ index
	s=numpy.std(outfeats,0)
	print("std of features:")
	for num,s in enumerate(s):
		print("feature "+str(num)+" : " + str(s))
	
	if not int(input("remove any features? 1/0 >>> ")):
		return
		
	# find indices to remove
	rlist=input("which indices? separate by space >>> ")
	rlist=numpy.asarray([int(n) for n in rlist.split(" ")])
	
	# do a difference on the undesired indices and all indices
	keep=numpy.setdiff1d(numpy.arange(outfeats.shape[1]),rlist)
	
	# keep those features
	outfeats=outfeats.take(keep,axis=1)
	
	# add the mutated features to the APC
	newname = input('name the new feature set >>> ')
	apc.add_idx(newname,outidx,outlabs,0,outfeats,0)
	
	# save it
	closer(apc,apcname)	


def snipper(apcname):
	# g mills 5/11/14
	# convenience function to fiddle with the contents of APCs by deleting 
	# index sets
	
	apc_dir='APC/'			# directory where the APCs live
	
	apc=pickle.load(open(apc_dir+apcname+'.pkl','rb'))
	
	# find the feature set
	apc.query_keys()
	
	while True:
		fname=input("axe which feature set? all, name, none. >>> ")
		if fname == 'all':
			apc.axe_idx(0)
		elif fname != 'none':
			apc.axe_idx(fname)
		else:
			break
			
	pickle.dump(apc,open(apc_dir+apcname+'.pkl','wb'),pickle.HIGHEST_PROTOCOL)	
	
	
def murk(apcname):
	# g mills 14/12/14
	# convenience function to delete APCs.
	
	apc_dir='APC/'			# directory where the APCs live
	
	apc=pickle.load(open(apc_dir+apcname+'.pkl','rb'))
	
	# display contents
	apc.query_keys()
	
	check=input('are you really sure? type out yes >>> ')
	
	if check=='yes':
		apc.axe_idx(0)
		os.rmdir(apc.featdir)
		os.remove(apc_dir+apcname+'.pkl')
		

def collapse(apcname):
	# g mills 16/12/14
	# replaces two labels in a hand labeled dataset with one label. 
	
	
	ssplabelnum=999
	demodir='APC/demos/'	# demo file directory
	
		
	# load it
	apc=opener(apcname)
	
	# get the feature set
	feats, fname = get_feats(apc)
	
	# get the labels out
	# tag, (idx, labels, cluster labels, feats, numscales, scales)
	labels=apc.items[fname][1].copy()	
	indices=apc.items[fname][0].astype(numpy.uint32)
	
	# figure out how many labels there are
	numlabs=int(labels.max()+1)
	if numlabs>10:
		print(numlabs)
		print((labels>10).sum())
	
	# print the label populations
	print('label populations:')
	for n in range(numlabs):
		print(str(n)+': ' + str((labels==n).sum()))	

	# figure out the merger
	mergeset = input("merge which labels? all will get the first label in the merged set >>> ")
	mergeset = [int(m) for m in mergeset.split(" ")]
	first=mergeset[0]
	
	# do it
	for m in mergeset[1:]:
		numpy.putmask(labels,labels==m,first)
		
	newname=input('name the new idx set >>> ')
	apc.add_idx(newname,indices,labels,0,feats,apc.items[fname][5])
		
	# save it
	closer(apc,apcname)	
	
	
#-------------------------------------------------------------------------------				

def ogmso_APC(apcname, scaleset):
	# g mills 24/8/15
	# build oriented geometric multiscale features for the point cloud, or a
	# subset thereof.
	
	# 31/8/15 update: covariance matrix option
		
	# INPUT
	# apcname = name of an APC object to load
	# skip = build the features for 1 in every *skip* points. if 1 or 0 take all
	# scaleset = list of tuples of analysis scales and ssp voxel edges like:
	# [ (biggest vox, array[biggest scales in descending order...])]
	#   (next biggest vox, array[next biggest scale after first set...])]
	# ...
	#   ( smallest vox, array[small scale... smallest scale]) ]
	
	# PARAMETERS
	apc_dir='APC/'
	minpoints=100		# minimum number of points in a metapartition
	budget=1700000000	# space (bytes) needed on the GPU for OG_MSO
	ssplabelnum=999		# this label means a point is unlabeled. makes sense.
	outwidth=8			# 8 for gmso + eigvecs
	imax = 14000		# max partition size in ogmso
	
	# OUTPUT
	# saves features and their indices in the point cloud to *apc*'s dictionary
	
	
	# load up the APC
	apc=opener(apcname)	
	
	# pick an idx set to which to limit the analysis and figure out our skip
	# setting. if there are labels, get em.
	indices,tag=get_idx(apc)
	
	# get labels. if no labels in the idx set, no big deal.
	labels=apc.items[tag][1]
	if isinstance(labels,numpy.ndarray):
		lsum=(labels!=ssplabelnum).sum()
		# just to be safe, if a point doesn't get a label it'll get non-label
		relabels=numpy.zeros(apc.inc.shape[0])+ssplabelnum
		numpy.put(relabels,indices,labels,mode='wrap')
		# initialize label output
		outlabels=numpy.zeros(0)
		print('features will be generated for ' + str(lsum) + ' labeled points.')
		print('there are ' + str(apc.inc.shape[0]-lsum) + ' unlabeled points.')
	
		
	skip=int(input('enter skip number for unlabeled points (zero to skip all) >>> '))
			
	cov_switch=int(input("use covariance matrix instead of eigenfeatures? 1/0 >>> "))		
			
	# name the output feature set
	featname = input('name the new feature set >>> ')
	
	# calculate size of the features
	fsize=sum([len(x[1]) for x in scaleset])*outwidth
			
	# initialize output
	feats=numpy.zeros((0,fsize+1))
	if isinstance(labels,numpy.ndarray):
		outlabels=numpy.zeros(0)
	else:
		outlabels=0
		
	# get the largest scale
	smax = scaleset[0][1][0]
	
	# load the point cloud on the GPU so we can query the part list with it
	apc.gpu_inc()
	
	# check if there's enough space left over for OG_MSO to do its job
	if cuda.mem_get_info()[0]<budget:
		shuffler=True
	else:
		shuffler=False	
	
	# loop over the partition array 
	prange=apc.partlist.shape[0]
	
	# time the whole thing
	outime=time.clock()
	
	for p in range(prange):
		print('starting metapartition ' + str(p+1) + ' of ' + str(prange))
		# pull the search space and query set indices 
		sspidx,qseidx=apc.query_partlist(p,smax,tag)
	
		# skip the requisite entries in query set partition index
		if skip>1:
			# if we have labels then we definitely want to get features for all
			# labeled points.
			if isinstance(labels,numpy.ndarray):
				# get a label set local to this metapartition
				mplabs=relabels.take(qseidx)
				# this takes the indices to labeled query set points in the
				# metapartition
				labqseidx=numpy.extract(mplabs!=ssplabelnum,qseidx)
				# this takes indices to all the unlabeled query set points
				unqseidx=numpy.extract(mplabs==ssplabelnum,qseidx)
				# now take a uniform random subset of them
				grabidx=numpy.random.permutation(unqseidx.size)[:int(unqseidx.size/skip)]
				unqseidx=unqseidx.take(grabidx)
				# and concatenate the labeled and unlabeled point indices
				qseidx=numpy.append(unqseidx,labqseidx)
			else:
			# otherwise just get a random subset of all the indices
				grabidx=numpy.random.permutation(qseidx.size)[:int(qseidx.size/skip)]
				qseidx=qseidx.take(grabidx)
	
		elif skip==0:
			if isinstance(labels,numpy.ndarray):
				# get a label set local to this metapartition
				mplabs=relabels.take(qseidx)
				# this takes the indices to labeled query set points in the metapartition
				qseidx=numpy.extract(mplabs!=ssplabelnum,qseidx)
			else:
				print('you just indicated that you want to skip all unlabeled points, but all points are unlabeled. hence there is nothing to be done here.')
				return 0
			
				
		if sspidx.size>minpoints and qseidx.size>minpoints:
			
			sspidx=sspidx.astype(numpy.uint32)
			qseidx=qseidx.astype(numpy.uint32)
			# get query set and search space partitions
			ssp=apc.inc.take(sspidx,axis=0)
			qse=apc.inc.take(qseidx,axis=0)
		
			# build output vector holding array
			ovh=numpy.zeros((qse.shape[0],fsize+1))
			rrl=numpy.zeros(qse.shape[0])
			
			# holder to first write index in output block
			dc=1
			
			# clear the GPU if we decided the point cloud it holds is too big
			if shuffler:
				apc.gpu_inc_purge()
			print("processing "+str(qse.shape[0])+" query set points")
			# loop over scale/ voxel edge combinations to fill holding array
			for s in scaleset:
				# get the scales and voxel edge length to use
				vxl=s[0]
				sc=s[1]
				ss=sc.size
			
				# build features
				if cov_switch:
					oc=mso.C_MSO(qse,ssp,vxl,sc,imax=imax)
				else:
					oc=mso.OG_MSO(qse,ssp,vxl,sc,imax=imax)
		
				# split off the indices (int type needed)
				oci=numpy.int64(oc[:,0])
		
				# slot in the features in scale-major order
				ovh[oci,dc:dc+ss*outwidth] = oc[:,1:]
				
				# now slot in the indices we built features for: if a point
				# gets a feature from at least one pass in og_mso it will be
				# represented in the index set. 
				# use the qse index set to map the given indices back to the
				# original point cloud.
				ovh[oci,0]=qseidx.take(oci)
				if isinstance(labels,numpy.ndarray):
					rrl[oci]=relabels.take(qseidx.take(oci))
			
				dc+=ss*outwidth		# update starting index
		
			# stack to output
			feats=numpy.vstack((feats,ovh))
			if isinstance(labels,numpy.ndarray):
				outlabels=numpy.hstack((outlabels,rrl))		
			
	# clean up the gpu
	apc.gpu_inc_purge()
	
	num=feats.shape[0]
	outime=time.clock()-outime
	rate=num/outime
	print(str(num) + ' points processed at a total, final rate of ' +str(rate)+' points/sec.')
	
	# save these results to the APC and put back on HDD
	apc.add_idx(featname,feats[:,0],outlabels,0,feats[:,1:],scaleset)
	pickle.dump(apc,open(apc_dir+apcname+'.pkl','wb'),pickle.HIGHEST_PROTOCOL)


#-------------------------------------------------------------------------------				

def vmso_APC(apcname, skip, scaleset):
	# g mills 13/12/14
	# build multiscale vector features for the point cloud, or a subset thereof.
	
	# hand-applied labels pass through safely. 
		
	# INPUT
	# apcname = name of an APC object to load
	# skip = build the features for 1 in every *skip* points. if 1 or 0, all
	# points.
	# scaleset = list of tuples of analysis scales and ssp voxel edges like:
	# [ (biggest vox, array[biggest scales in descending order...])]
	#   (next biggest vox, array[next biggest scale after first set...])]
	# ...
	#   ( smallest vox, array[small scale... smallest scale]) ]
	
	# PARAMETERS
	apc_dir='APC/'
	minpoints=100		# minimum number of points in a metapartition
	budget=1700000000	# space (bytes) needed on the GPU for F_MSO
	hltag='deep labels'	# if the input has this label, then we'll use its
							# search space points as the search space.
	ssplabelnum=999		# labels associated with search space points
	sspex=False			# if true build features for search space points
	imax=18000			# maximum size of a partition in V_MSO
	
	
	# OUTPUT
	# saves features and their indices in the point cloud to *apc*'s dictionary
	
	
	# load up the APC
	apc=pickle.load(open(apc_dir+apcname+'.pkl','rb'))
	
	# figure out which feature set of the point cloud to process
	apc.query_keys()
	tag=input("use which feature set? >>> ")
		
	# get the labels and indices
	# tag, (idx, labels, clusters, feats, numscales, scales)
	indices=apc.items[tag][0].astype(numpy.uint32)
	labels=apc.items[tag][1]
	clabels=apc.items[tag][2]	
	
	# get feats
	feats=apc.pull_feats(tag)
	nc=feats.shape[1]
	
	# figure out what the ssp/ qse special situation is
	dssp=int(input('is there a special ssp we should use? 1/0 >>> '))
	if dssp:
		sspex=int(input('should we build features for it? 1/0 >>> '))

	# reorder the labels and cluster labels so we can use the metapartition's
	# search space index list to index em
	reclabels=numpy.zeros(apc.inc.shape[0])
	relabels=numpy.zeros_like(reclabels)+ssplabelnum
	refeats=numpy.zeros((apc.inc.shape[0],nc))
	reclabels[indices]=clabels
	relabels[indices]=labels
	refeats[indices]=feats
	
	# name the output feature set
	featname = input('name the output feature set >>> ')
	
	# calculate size of the output features
	scs=sum([len(x[1]) for x in scaleset])
	fsize=scs*nc
	fsmall=fsize
	
	# initialize outputs
	feats=numpy.zeros((0,fsize+1))
	outlabels=numpy.zeros(0)
		
	# get the largest scale
	smax = scaleset[0][1][0]
	
	# loop over the metapartition array 
	prange=apc.partlist.shape[0]
	
	# load the point cloud on the GPU so we can query the part list with it
	apc.gpu_inc()
	
	# check if there's enough space left over for H_MSO to do its job
	if cuda.mem_get_info()[0]<budget:
		shuffler=True
	else:
		shuffler=False
		
	outime=time.clock()
	
	for p in range(prange):
		print('starting metapartition ' + str(p+1) + ' of ' + str(prange))
		
		# pull the search space and query set indices 
		sspidx,qseidx=apc.query_partlist(p,smax,tag)
		if dssp:
			# if there is a designated search space, that's all we use as the
			# search space.
			sspidx=numpy.unique((relabels[sspidx]==ssplabelnum)*sspidx).astype(numpy.uint32)
			# if we're not building features for the search space, then we can
			# boot them from the query set here.
			if not sspex:
				qseidx=numpy.unique((relabels[qseidx]!=ssplabelnum)*qseidx).astype(numpy.uint32)
		# skip the requisite entries in query set partition index
		if skip>1:
			# get a random subset of the indices
			grabidx=numpy.random.permutation(qseidx.size)[:int(qseidx.size/skip)]
			qseidx=qseidx[grabidx]
		
		if sspidx.size>minpoints and qseidx.size>minpoints:
			sspidx=sspidx.astype(numpy.uint32)
			qseidx=qseidx.astype(numpy.uint32)
			# get query set and space metapartitions-- points, indices, features
			ssp=apc.inc.take(sspidx,axis=0)
			qse=apc.inc.take(qseidx,axis=0)
			sspvec=refeats.take(sspidx,axis=0)
			# index to the query set metapartition
			mapidx=numpy.arange(qseidx.size)
			
			# clear the GPU if we decided the point cloud it holds is too big
			if shuffler:
				apc.gpu_inc_purge()
			
			# build output vector holding array
			ovh=numpy.zeros((qse.shape[0],fsize+1))
			# re-reordered label array
			rrl=numpy.zeros(qse.shape[0])
				
			# index holder-- this is the number of scales that have been slotted
			# in so far
			dc=0
			print("processing "+str(qse.shape[0])+" query set points")
			# loop over scale/ voxel edge combinations to fill holding array
			for s in scaleset:
				# get the scales and voxel edge length to use
				vxl=s[0]
				sc=s[1]
				ss=sc.size
	
				# build features
				oc=mso.V_MSO(qse,mapidx,ssp,sspvec,vxl,sc,imax=imax)
		
				# split off the indices (int type needed)
				oci=oc[:,0].astype(numpy.int64)
				
				# slot in the mean
				ovh[oci,1+dc:1+dc+nc*ss] = oc[:,1:1+nc*ss]
				
				# now slot in the indices we built features for: if a point gets
				# a feature from at least one pass in h_mso it will be
				# represented in the index set. use the qse index set to map
				# the given indices back to the original point cloud
				ovh[oci,0]=qseidx.take(oci)				
				rrl[oci]=relabels.take(qseidx.take(oci))
				
				dc+=nc*ss		# update starting index
		
			# stack to outputs
			feats=numpy.vstack((feats,ovh))
			outlabels=numpy.hstack((outlabels,rrl))			
		
	# clean up the gpu
	apc.gpu_inc_purge()
		
	num=feats.shape[0]
	outime=time.clock()-outime
	rate=num/outime
	print(str(num) + ' points processed at a total, final rate of ' +str(rate)+' points/sec.')	
		
	# save these results to the APC and put back on HDD
	apc.add_idx(featname,feats[:,0],outlabels,0,feats[:,1:],scaleset)
	pickle.dump(apc,open(apc_dir+apcname+'.pkl','wb'),pickle.HIGHEST_PROTOCOL)		
		

#-------------------------------------------------------------------------------				

