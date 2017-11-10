# minimal

quick-n-dirty CPU implementation. no gpu required.
performance was about 5k points per second per scale on my machine, using a single core.
try using [multiprocessing](https://docs.python.org/3.5/library/multiprocessing.html) for a little more oomph. see comments in multiscale.py.

the kdtree implementation it uses might start to choke when input size reaches a few million points.



## to get started...

the only requirements here are python3, numpy, scipy and scikit-learn. any numpy after about 1.10
should work.

```
pip3 install numpy
pip3 install scipy
pip3 install sklearn

cd ~
mkdir code
cd code/
git clone git@github.com:grayhem/nimrud.git
```


append the following to .bashrc or .profile to get this module on your pythonpath:

```
CODE="$HOME/code"
for DIR in $( ls $CODE ); do
    export PYTHONPATH="$PYTHONPATH:$CODE/$DIR"
done
```

## usage
all point clouds are 2D numpy arrays, with each row representing a single point. the first three
columns are assumed to represent the geometry in 3-space, and all remaining columns are features.
we make no attempt to track what the features represent in this implementation. i highly recommend
using [pandas dataframes](http://pandas.pydata.org/).