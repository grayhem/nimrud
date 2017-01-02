# extra tools for machine learning


import numpy
import os
import matplotlib.pyplot as plt
import time

#------------------------------------------------------------------------------------------------

class RPT_ensemble(object):
	# g mills 21/2/15
	# ensemble of random projection tree classifiers. we train a number of RP trees on subsets
	# of the training data, then we walk test data down the trees. if an obs ends up in an
	# explicitly decided leaf, it gets the vector of pop proportions and gini impurity of that
	# leaf. if it ends up in a null leaf (which was empty in training set) then it gets the
	# vector of pop proportions and impurity of the branch it came from. these vectors are then
	# reduced over all the trees to decide the vector of class probabilities for each test obs.

	# based on:
	# Dasgupta, S., & Freund, Y. (2008, May). Random projection trees and low dimensional manifolds. In Proceedings of the fortieth annual ACM symposium on Theory of computing (pp. 537-546). ACM.

	def __init__(self, **kwargs):
		# initialize the RPT_E object.


		# KWARGS--------------------

		# d_func = decision function type. determines how we decide class probabilities from
				# population proportion and gini impurity of cells.
				# string, default 'wmean'
			# 'wmean' = weighted mean of pop proportions, using (1-gini impurity) as weight.
			# 'wmax' = max of (pop proportions * gini impurity)

		# n_estimators = number of RPTs in the ensemble.
				# int, default 10

		# impurity = gini impurity thresh-- if a cell hits this, it's a leaf. if not, split.
				# may be set for all trees, or selected at random per tree.
				# float or (float,float), default .2
			# if float, then this is the impurity thresh for all.
			# if tuple of floats, then this is the range of impurity threshes selected per tree.

		# min_obs = minimum number of training samples in a branch. will stop splitting after
				# this is reached.
				# int, default 20

		# floatype = type used for storing proportions when predicting new labels.
				# floating point type, numpy.float32 or numpy.float64. default numpy.float64

		# onepass = number of test data observations to evaluate in one pass. prevents
				# bottlenecks.
				# int, default 10000

		try:
			self.d_func=kwargs['d_func']
		except KeyError:
			self.d_func='wmean'

		try:
			self.n_estimators=kwargs['n_estimators']
		except KeyError:
			self.n_estimators=10

		try:
			self.impurity=kwargs['impurity']
		except KeyError:
			self.impurity=.2

		try:
			self.min_obs=kwargs['min_obs']
		except KeyError:
			self.min_obs=20

		try:
			self.floatype=kwargs['floatype']
		except KeyError:
			self.floatype=numpy.float64

		try:
			self.onepass=kwargs['onepass']
		except KeyError:
			self.onepass=10000

		# this is what we use to identify leaves in the tree code books
		self.dummy_split=numpy.inf



	#=========================

	def set_params(self, **kwargs):
		# (re)set parameters for the ensemble. see above.

		if 'd_func' in kwargs:
			self.d_func=kwargs['d_func']
		if 'n_estimators' in kwargs:
			self.n_estimators=kwargs['n_estimators']
		if 'impurity' in kwargs:
			self.impurity=kwargs['impurity']
		if 'min_obs' in kwargs:
			self.min_obs=kwargs['min_obs']
		if 'floatype' in kwargs:
			self.floatype=kwargs['floatype']
		if 'onepass' in kwargs:
			self.onepass=kwargs['onepass']

	#=========================

	def fit(self, data, labels):
		# fit the ensemble to some labeled training data. we'll train each tree with a balanced
		# subset of the training data. we don't assume the training data is balanced coming in.

		# INPUT
		# data = training data
		# labels = labels for training data

		# PARAMETERS
		num = data.shape[0]
		self.numlabs=int(labels.max()+1)
		self.dim=data.shape[1]


		# OUTPUT
		# makes a dictionary of tree rule dictionaries, puts in self.

		# first check if we have exactly enough labels for the data:
		assert labels.size==num, 'training set and label set do not match!'

		# build the ensemble dictionary
		self.ensemble_rule={}

		# BALANCED SAMPLING PREP
		# index the data and labels
		idx=numpy.arange(num,dtype=numpy.uint32)
		# get the indices of each class in the training set
		labset=[numpy.extract(labels==m,idx) for m in range(self.numlabs)]
		# shuffle em
		for l in labset:
			numpy.random.shuffle(l)
		# get the populations per label
		labpops=numpy.asarray([l.size for l in labset])
		# find the min population
		minpop=labpops.min()
		# divide that over the total number of estimators:
		bpop=int(numpy.floor(minpop/self.n_estimators))
		# now get a random set of indices to the label set index array
		perm=numpy.random.permutation(minpop)

		# now build the trees. they'll be identified by number. so exciting!
		for n in range(self.n_estimators):
			# assemble the training set for this tree:
			# first find the range of indices in each label we want
			nperm=perm[n*bpop:(n+1)*bpop]
			# now map those indices to the indices in the data and labels
			tgrab=numpy.r_[[l.take(nperm) for l in labset]].reshape(-1)
			#print('tgrab')
			#print(tgrab.shape)
			# and map THOSE indices to the data and labels
			tdata=data.take(tgrab,axis=0)
			tlabs=labels.take(tgrab)
			#print(tdata.shape)
			#print(data.shape)
			# figure out what the impurity threshold for this tree shall be
			if isinstance(self.impurity,tuple):
				impurity=max(self.impurity)-numpy.random.rand(1)[0]*min(self.impurity)
			else:
				impurity=self.impurity

			# train the tree and record it
			self.ensemble_rule.update({n:self._maketree(tdata,tlabs,1,impurity)})

	#=========================

	def predict(self,data):
		# evaluate the trees and return the most likely class for each observation based
		# on the desired decision function.

		# INPUT
		# data = data for which to predict labels

		# OUTPUT
		# labels = labels predicted for the data


		# almost all the necessary work has already been done in self.predict_proba().
		proba=self.predict_proba(data)

		labels=proba.argmax(axis=1)
		return labels

	#=========================

	def predict_and_proba(self,data):
		# evaluate the trees and return the most likely class for each observation based
		# on the desired decision function. and probabilities.

		# INPUT
		# data = data for which to predict labels

		# OUTPUT
		# labels = labels predicted for the data
		# proba = probabilities predicted for the data


		# almost all the necessary work has already been done in self.predict_proba().
		proba=self.predict_proba(data)

		labels=proba.argmax(axis=1)
		return labels,proba

	#=========================

	def predict_proba(self,data):
		# evaluate the trees and return the vector of probabilities for each observation
		# based on the desired decision function. we're going to break it up into small pieces.

		# INPUT
		# data = data for which to predict label probabilities

		# OUTPUT
		# proba = probabilities predicted for the data.
		# proba = numpy.ndarray((data.shape[0],self.numlabs),dtype=self.floatype)
		proba=numpy.zeros((0,self.numlabs))


		# make sure the data is aligned with the data we trained with
		assert data.shape[1]==self.dim, 'test data do not match training data dimensions!'

		# figure out how many passes we need over the trees
		passes=int(numpy.ceil(data.shape[0]/self.onepass))

		# evaluate the trees
		for p in range(passes):
			# and time how long it'll take (this thing is slow)
			if p==0:
				st=time.clock()
			chunk=data[p*self.onepass:(p+1)*self.onepass]
			proba=numpy.vstack((proba,self._evaluator(chunk)))
			if p==0:
				ot=numpy.round(time.clock()-st,decimals=2)
				print('first batch of ' + str(self.onepass) + ' samples took ' + str(ot)+'s.')
				rt=(passes-1)*ot
				print('estimated time remaining: ' +str(rt)+'s.')


		return proba

	#=========================

	def _evaluator(self, data):
		# evaluate a dataset and return probabilities for each entry

		# PARAMETERS
		pshape=[data.shape[0],self.n_estimators,self.numlabs+1]

		# OUTPUT
		# proba = probabilities predicted for the data.
		# proba = numpy.ndarray((data.shape[0],self.numlabs),dtype=self.floatype)

		# make sure the data is aligned with the data we trained with
		assert data.shape[1]==self.dim, 'test data do not match training data dimensions!'

		# run the data down the trees. big operation.
		props=numpy.zeros(pshape,dtype=self.floatype)
		#treval=time.clock()
		for n,rule in enumerate(self.ensemble_rule.values()):
			props[:,n,:]=self._evaltree(data,rule,1)
		#treval=time.clock()-treval
		#treval=numpy.round(treval,decimals=2)
		#print('running down the trees took '+str(treval)+'s.')

		# flip the gini impurity to use as weight
		weights=(1-props[:,:,0]).reshape(-1,self.n_estimators,1)

		#deval=time.clock()
		# figure out which decision function we'll use
		if self.d_func=='wmean':
			# weighted geometric mean------------------
			# first normalize the weights
			weights/=(weights.sum(1).reshape(-1,1,1)+numpy.spacing(32))
			# now weight the proportions
			props[:,:,1:]*=weights
			# sum the weighted proportions to calculate the weighted average
			proba=props[:,:,1:].sum(1)

		elif self.d_func=='wmax':
			# weighted max value------------------
			# weight the proportions
			props[:,:,1:]*=weights
			# take the max of the weighted proportions
			proba=props[:,:,1:].max(1)

		else:
			print(self.d_func + ' is not a recognized decision function. sorry!')
			return
		#deval=time.clock()-deval
		#deval=numpy.round(deval,decimals=2)
		#print('evaluating the decision function took '+str(deval)+'s.')
		return proba

	#=========================

	def _maketree(self, data, labels, tag, impurity):
		# grow a single random projection tree.

		# INPUT
		# data = training data
		# labels = labels for same
		# tag = branch code for current cell-- initialize with 1
		# impurity = gini impurity threshold

		# PARAMETERS
		num=data.shape[0]		# cell population

		# OUTPUT
		# rules = dict of rules for evaluating the tree:
		# { branch code : [ split value, ndarray[vector], gini impurity, ndarray[proportions] ] }


		# first let's find out if this cell is a leaf.
		labpops=numpy.asarray([(labels==m).sum() for m in range(self.numlabs)])
		labpops/=num
		gini=1-(labpops**2).sum()

		if gini<=impurity or num<=self.min_obs:
			# congratulations! we have a sufficiently small or pure leaf. time to quit.
			vec=numpy.zeros(self.dim)
			return {tag:[self.dummy_split,vec,gini,labpops]}

		# if we're still going, then this cell is a branch.

		# get the (unit) projection vector for this split
		vec=numpy.random.rand(self.dim)
		vec/=numpy.linalg.norm(vec)

		# project the data along the vector
		proj=numpy.dot(data,vec)

		# get the median
		med=numpy.median(proj)

		# jitter the median
		point=data[numpy.random.randint(num)]		# first find a random point
		dists=numpy.linalg.norm(data-point,axis=1)	# now find the furthest point from that point
		mdist=dists.max()
		jitter=(numpy.random.rand(1)-.5)*12*mdist/numpy.sqrt(self.dim)
		med+=jitter

		# compose the rule for this branch
		rule={tag:[med,vec,gini,labpops]}

		# split this branch
		left=proj<=med
		ltag=tag<<1
		if left.sum():
			lrule=self._maketree(numpy.compress(left,data,axis=0),numpy.extract(left,labels),ltag,impurity)
			rule.update(lrule)
		right=proj>med
		rtag=ltag|1
		if right.sum():
			rrule=self._maketree(numpy.compress(right,data,axis=0),numpy.extract(right,labels),rtag,impurity)
			rule.update(rrule)
		return rule


	#=========================

	def _evaltree(self,data,rule,tag):
		# evaluate a tree for given data, returning gini impurity and proportions for each obs.

		# INPUT
		# data = test data
		# rule = rulebook defining the tree
		# tag = branch code, call with 1.

		# PARAMETERS
		num=data.shape[0]

		# OUTPUT
		# props = ndarray of gini impurity and proportions aligned with data
		# props = [gini, props....]


		# pull this cell's instructions from the rulebook
		# { branch code : [ split value, ndarray[vector], gini impurity, ndarray[proportions] ] }
		try:
			todo=rule[tag]
		except KeyError:
			# if we end up here, then this is a dead leaf which was unrepresented in the
			# training data. meaning we have no idea what's in it. best we can do is back off
			# to the last branch and use that code to represent the contents here.
			tag=tag>>1
			todo=rule[tag]
			props=numpy.hstack((todo[2],todo[3]))
			props=numpy.tile(props,(num,1))
			return props.astype(self.floatype)

		# if we're on a leaf, go ahead and return props. we'll know because leaves don't have a
		# splitting value.
		if todo[0]==self.dummy_split:
			props=numpy.hstack((todo[2],todo[3]))
			props=numpy.tile(props,(num,1))
			return props.astype(self.floatype)

		# otherwise, project the data and keep going
		med=todo[0]
		vec=todo[1]
		proj=numpy.dot(data,vec)

		# split er and use the conditionals to map the props predicted downstream back into the
		# output.
		props=numpy.zeros((num,1+self.numlabs),dtype=self.floatype)
		left=proj<=med
		ltag=tag<<1
		if left.sum():
			lprops=self._evaltree(numpy.compress(left,data,axis=0),rule,ltag)
			# apparently there isn't a numpy indexing routine that maps into 2D arrays, but
			# fancy indexing supports it. ~ugh~
			props[left.astype(bool)]=lprops
			del lprops
		right=proj>med
		rtag=ltag|1
		if right.sum():
			rprops=self._evaltree(numpy.compress(right,data,axis=0),rule,rtag)
			props[right.astype(bool)]=rprops
			del rprops

		return props


#------------------------------------------------------------------------------------------------

def dilate_scale(inc, factor):
	# g mills 6/12/14
	# dilates a matrix by isotropically replicating the entries, and scales to (0,1).

	# INPUT
	# inc = input matrix, numpy array
	# factor = dilation factor. it should be an int.

	# OUTPUT
	# outc = dilated version of input matrix as float32.


	# initialize *outc*
	outc=numpy.empty((inc.shape[0]*factor,inc.shape[1]*factor))

	# scale input
	inc/=inc.max()

	# loop over rows in *inc*
	for rows in range(inc.shape[0]):
		# loop over cols in *inc*
		for cols in range(inc.shape[1]):
			# write to a patch in *outc*
			outc[rows*factor:(rows+1)*factor,cols*factor:(cols+1)*factor]=inc[rows,cols]

	return outc.astype(numpy.float32)



#------------------------------------------------------------------------------------------------

def user_producer(conf):
	# g mills 3/12/14
	# user and producer averages on the supplied confusion matrix. rows should be assigned
	# classes, cols should be known classes.

	# INPUT
	# conf = confusion matrix like from mc_confusion

	# PARAMETERS
	# none that i can think of

	# OUTPUT
	# user = user average (over cols/ down rows)
	# prod = producer average (over rows/ down cols)


	# user
	user=numpy.asarray([conf[x,x]/conf.sum(1)[x] for x in range(conf.shape[0])])*100

	# producer
	prod=numpy.asarray([conf[x,x]/conf.sum(0)[x] for x in range(conf.shape[0])])*100

	return user, prod

#------------------------------------------------------------------------------------------------

def three_metrics(conf):
	# g mills 1/9/15
	# output true positive, false positive and false negative %
	# scores for each class
	
	# INPUT
	# conf = confusion matrix
	
	# OUTPUT
	# scores = n_classes x 4 (tp, tn, fp, fn) ndarray
	
	
	# num classes
	n_classes=conf.shape[0]
	
	# num obs per real class-- all classes should be same
	n_real = conf.sum(0)[0]
	
	# num obs per predicted class-- classes are probably different
	n_pred = conf.sum(1)
	
	# assemble the columns of the output
	tp = numpy.asarray([conf[x,x]/n_real for x in range(n_classes)])
	fp = numpy.asarray([(n_real-conf[x,x])/n_real for x in range(n_classes)])
	fn = numpy.asarray([(n_pred[x]-conf[x,x])/n_pred[x] for x in range(n_classes)])
	
	return numpy.column_stack((tp,fp,fn))
	
#------------------------------------------------------------------------------------------------

def mc_confusion(lies,truth):
	# g mills 31/10/14
	# full confusion matrix for multiclass supervised learning. we assume labels are integers
	# 0...n-1


	# INPUT
	# lies = a posteriori labels (classified)
	# truth = a priori labels (known)

	# PARAMETERS

	# OUTPUT
	# conf = confusion matrix, n classes x n classes. total number in known class (col) receiving
	# (row) label.


	# figure out how many classes we got
	nlabels=truth.max()+1
	clabels=lies.max()+1

	nlabels=max(nlabels,clabels)
	# initialize output
	conf=numpy.zeros((nlabels,nlabels))

	# start filling it
	for row in range(int(nlabels)):
		for col in range(int(nlabels)):
			num=((lies==row)*(truth==col)).sum()
			conf[row,col]=num

	return conf




#------------------------------------------------------------------------------------------------

def confusion(inc,labels):
	# g mills 17/9/14
	# confusion matrix metrics. specifically completeness, correctness, and quality.


	# INPUT
	# inc = a posteriori (classified) labels. 1 = b, 0 = a.
	# labels = 1d array of known labels corresponding to points in inc

	# OUTPUT
	# acomp	= completeness of a
	# acorr	= correctness of a
	# aqual = quality of a
	# bcomp
	# bcorr
	# bqual

	if inc.shape[1]==4:
		inc=inc[:,3]

	# get number of known a and b points
	asum=labels.size-labels.sum()
	bsum=labels.sum()

	# find the number of a and b points with correct labels
	bright=inc*labels
	bright=bright.sum()

	aright=(1-inc)*(1-labels)
	aright=aright.sum()

	# confusion matrix for A:
	aTP=((1-inc)*(1-labels)).sum()			# true positive-- reference A classed as A
	aTN=(inc*labels).sum()					# true negative-- reference B classed as B
	aFP=bsum-aTN							# false positive-- reference B classed as A
	aFN=asum-aTP							# false negative-- reference A classed as B

	# confusion matrix for B:
	bTP=aTN
	bTN=aTP
	bFP=aFN
	bFN=aFP

	acomp=100*(aTP/(aTP+aFN))
	acorr=100*(aTP/(aTP+aFP))
	aqual=100*(aTP/(aTP+aFP+aFN))

	bcomp=100*(bTP/(bTP+bFN))
	bcorr=100*(bTP/(bTP+bFP))
	bqual=100*(bTP/(bTP+bFP+bFN))

	return acomp,acorr,aqual,bcomp,bcorr,bqual


#------------------------------------------------------------------------------------------------

def colorize_mc_prob(inc,probs):
	# g mills 7/10/14
	# colorizes a point cloud based on classes and their associated probabilities. zero prob gets
	# white, 1 prob gets the real color.

	# INPUT
	# inc = input point cloud with labels. just XYZL.
	# probs = probability array corresponding to the point cloud. fresh from the classifier is
	# fine.

	# PARAMETERS
	num_points=inc.shape[0]
	num_labels=int(probs.shape[1])
	# deep pink, blue, green, violet, orange, 'free speech red', forest green, saddle brown, navy, goldenrod
	colormat=numpy.array([[255,20,147],[0,191,255],[0,255,127],[138,43,226],[255,140,0],[192,0,0],[34,139,34],[139,69,19],[0,0,128],[218,165,32]])
	white=numpy.ones((num_points,3))*255
	colors=white.copy()


	# OUTPUT
	# outc = output point cloud with an rgb value trailing the xyz.


	# cycle through each label and subtract the appropriate color adjustment from the color array
	for c in range(num_labels):
		# get an array of probabilities associated with this label for each point, iff that point
		# is the appropriate label. otherwise it gets zero.
		probarray=(inc[:,3]==c)*probs[:,c]

		# color array
		color_sub=numpy.ones((num_points,3))*colormat[c]

		# put in the gradient
		colors-=(white-color_sub)*probarray.reshape(-1,1)



	return numpy.hstack((inc[:,:3],colors))




#------------------------------------------------------------------------------------------------

def colorize_multiclass(inc):
	# g mills 27/7/14
	# colorizes a classified point cloud. 

	# messed up 9/8/14 for multiclass support
	# as of 3/11/14, supports 10 labels

	# INPUT
	# inc = input point cloud with integer label values, like from apply_svm_numeric down there.
	# up to 10 labels (0-9) are allowed.

	# PARAMETERS
	points=inc.shape[0]
	# deep pink, blue, green, violet, orange, 'free speech red', forest green, saddle brown, navy, goldenrod
	colormat=numpy.array([[255,20,147],[0,191,255],[0,255,127],[138,43,226],[255,140,0],[192,0,0],[34,139,34],[139,69,19],[0,0,128],[218,165,32]])


	# OUTPUT
	# outc = output point cloud with an rgb value trailing the xyz.
	outc=numpy.zeros((points,6))

	outc[:,:3]=inc[:,:3]

	# put colors on the points
	for p in range(points):
		outc[p,3:]=colormat[inc[p,3]]

	return outc

#------------------------------------------------------------------------------------------------

def dainty_loader(filename):
	# g mills 15/10/15
	# load a too-big point cloud in pieces

	# INPUT
	# filename = where to find the file

	# PARAMETERS
	middir='pointclouds/temp/'	# save segments here
	de=','

	# first off scour out the midfile directory in case we crashed in the middle of last run
	midfiles=os.listdir(middir)
	for midfile in midfiles:
		os.remove(middir+midfile)

	mm=time.time()

	# split the file (20 million point segments)
	os.system('split -l 20000000 ' + filename + ' ' + middir+'temp')

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
