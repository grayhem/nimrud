# testing some settings in MSO chain

import mso
import cv
import numpy
import apc
import time


def gmso_rate():
	# test the influence of max partition size on throughput of gmso with
	# CM data and scales
	
	# about 15000 seems to be best at middle scale set
	# about 14000 at biggest scale set
	# about 10000 at smallest scale set
	
	# there is a dependence on analysis scale, but it's not all that strong.
	# there's maybe 1000 pts/ sec to be gained over the 3x2 scaleset if we
	# set optimal imax for each voxel edge. maybe 5-600 if we use 12-14000 pts
	# for all voxel edges.
	
	# actually we gained almost 2k/s in gmso with first_scales!
	
	imaxes=numpy.arange(4000,20001,2000)[::-1]
	scaleset=cv.first_scales()
	vox=scaleset[2][0]
	scales=scaleset[2][1]
	
	# load up an apc
	a=apc.opener("cm_labels2")
	# get the bare point cloud
	idx,_=apc.get_idx(a)
	inc=a.inc.take(idx.astype(numpy.int64),axis=0)
	
	# iterate over partition sizes
	for imax in imaxes:
		print("max partition size: "+str(imax))
		st=time.time()
		# basic gmso call
		mso.G_MSO(inc,inc,vox,scales,imax=imax)
		ft=time.time()-st
		print("total gmso call time: " + str(ft))

def vmso_rate():
	# test the influence of max partition size on throughput of vmso with
	# CM data and scales
	# right around 18000 at 5 features, medium edge
	# 18000 at largest edge, too
	# and smallest.
	
	
	vox=0.25
	scaleset=cv.first_scales()
	vox=scaleset[2][0]
	scales=scaleset[2][1]
	imaxes=numpy.arange(4000,20001,2000)[::-1]
	
	# load up an apc
	a=apc.opener("cm_labels2")
	
	# get the point cloud and features
	feats,tag=apc.get_feats(a)
	idx=a.items[tag][0]
	inc=a.inc.take(idx.astype(numpy.int64),axis=0)
	
	
	# iterate over partition sizes
	for imax in imaxes:
		print("max partition size: "+str(imax))
		st=time.time()
		# basic gmso call
		mso.V_MSO(inc,idx,inc,feats,vox,scales,imax=imax)
		ft=time.time()-st
		print("total vmso call time: " + str(ft))
		
		
		
		
		
		
		
		
		
		
		
