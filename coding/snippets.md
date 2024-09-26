# snippets

## LLM prompt

### latex format

```
**Prompt:**

Convert the following mathematical expressions from backslash format to LaTeX using `$...$` for inline formulas and `$$...$$` for standalone equations.

**Text to Convert:**

1. `y = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}`
2. `E = mc^2`
3. `\int_{a}^{b} x^2 dx`
4. `x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} and y = mx + c`
5. `\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}`

**Expected Conversion:**

1. $$y = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
2. $E = mc^2$
3. $$\int_{a}^{b} x^2 dx$$
4. $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ and $y = mx + c$
5. $$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$$

**Instructions:**

Use `$...$` for expressions within a line of text and `$$...$$` for centered equations.

---

This streamlined version focuses directly on the conversion task.
```

## Bash

### Work flow

```bash
#!/bin/bash

# 原始文件名称
original_file="calc.py"

# 复制文件的数量
copies=50

for i in $(seq 1 $copies); do
    # 创建新文件的名称
    new_file="calc_$i.py"

    # 复制原始文件到新文件
    cp $original_file $new_file

    # 在第10行替换文本
    sed -i "5s/i=0/i=$i/" $new_file
done
```

### ls column by index

```bash
ls -i submit_*.sh| awk '{print $2}' | sort -V
```

### Add sbatch in `do.sh`

```bash
sed -i 's/\([^ ]*\)/sbatch \1/g' do.sh
```

### Find how many works are running

```bash
squeue -u wangyiming25 | grep ' R ' | wc -l
squeue -u wangyiming25 | grep ' PD ' | wc -l
```

### Cancel homework from one id to another id

```bash
for job_id in {244328..244422}
do
   scancel $job_id
done
```

### Find numbers of pattern in one file

```bash
grep -o 'pattern' filename | wc -l
```

### Display all the content in many files

```bash
for i in e265{507..642}; do echo "cat $i.log text:"; cat "$i.log"; done
```

### Find empty file of subdirectories

```bash
find . -type d -empty
```

### grep number

```bash
grep -o  '[0-9]\+'
```

### loop over some specific list

```bash
numbers=(340 528 650 166 430 307 394 652 160 91 311 429 95 420 425)
for i in "${numbers[@]}"; do
    ...
done
```

### hard link

```bash
ln -s /sharefs/your/path/to/directory /afs/your/path/to/directory
```

### check disk space

```bash
lfs df -h -p sharefs.alipool /sharefs/alicpt
```

### delete many files

```bash
rm {0..100}.npy
```

### Tips

* do not add space when assign value to some variable.
* `find . -type d -empty | grep -o '[0-9]\+' | tr '\n' ' ' | awk '{print "numbers=(" $0 ")"}'`

## Git

### How to add some specific file

```git
*.csv
!0.csv
```

## Python

### Algorithm

* interpolation on lg scale

```python
def log_Cubic_interpolate(xx,yy):
    logx=np.log10(xx)
    logy=np.log10(yy)
    Cubic_interp=CubicSpline(logx,logy)
    log_Cubic_interp=lambda zz: np.power(10, Cubic_interp(np.log10(zz)))
    return log_Cubic_interp
```

* check unique element

```python
def check_unique_seeds(seeds):
    # if the seeds array are unique, return True
    seen = {}
    for seed in seeds:
        if seed in seen:
            return False  # Duplicate found
        seen[seed] = True
    return True
```

* generate random seeds

```python
seeds = [secrets.randbits(32) for _ in range(seeds_number)]
```

* get suitable chi2/dof

```python
from scipy.stats import chi2

dof = N - p

lower_bound = chi2.ppf(0.00135, dof)
upper_bound = chi2.ppf(0.99865, dof)

chi2dof_lower = lower_bound / dof
chi2dof_upper = upper_bound / dof
```

### Ipython

* %pylab \[--no-import-all]

```python
import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs
```

### Healpy

* plot with 0 pad

```python
plt.title('your title', pad=0)
```

* square mask

```python
def mask_square(_mask, square_length=np.radians(6), square_width=np.radians(5),/, mask_value=1):

    # square_length = np.radians(6) # longitude
    # square_width = np.radians(5) # latitude

    p1_theta = l0 - square_width / 2
    p2_theta = l0 - square_width / 2
    p3_theta = l0 + square_width / 2
    p4_theta = l0 + square_width / 2

    p1_phi = b0 - square_length / 2
    p2_phi = b0 + square_length / 2
    p3_phi = b0 + square_length / 2
    p4_phi = b0 - square_length / 2

    p1_vec = hp.rotator.dir2vec(theta = p1_theta, phi=p1_phi)
    p2_vec = hp.rotator.dir2vec(theta = p2_theta, phi=p2_phi)
    p3_vec = hp.rotator.dir2vec(theta = p3_theta, phi=p3_phi)
    p4_vec = hp.rotator.dir2vec(theta = p4_theta, phi=p4_phi)

    vertices = np.vstack((p1_vec,p2_vec,p3_vec,p4_vec))
    print(f"{vertices.shape=}")

    ipix_strip = hp.query_polygon(nside=nside, vertices= vertices)
    _mask[ipix_strip] = mask_value
    return _mask
```

* disc mask

```python
centerVec = hp.rotator.dir2vec(theta = np.pi/2, phi = 0)
ipix_disc = hp.query_disc(nside=nside, vec=centerVec, radius=np.radians(16))
mask[ipix_disc] = disc_value
```

* strip mask

```python
ipix_strip = hp.query_strip(nside=nside, theta1=np.pi/2-np.radians(5)/2, theta2=np.pi/2+np.radians(5)/2)
mask[ipix_strip] = strip_value
```

* how to get colorbar of the plots

```python
fig = plt.gcf()
ax_list = fig.axes
cbar_ax = ax_list[-1] if len(ax_list) > 1 else None
if cbar_ax and hasattr(cbar_ax, 'collections'):
    mappable = cbar_ax.collections[1]
    vmin, vmax = mappable.get_clim()
    print("Color bar min:", vmin)
    print("Color bar max:", vmax)
```

* how to get colorbar of the plots (another way)

```python
hp.gnomview(m, return_projected_map=True)
```

### Numpy

* np.where:numpy.where(condition, \[x, y, ]/);for condition:Where True, yield x, otherwise yield y.

### Fits files operation

* check the header name of a fits file

```python
from astropy.io import fits
hdul = fits.open(filename)
print(repr(hdul[1].header))
hdul.close()
```

* check the info

```python
hdul.info()
```

* check the data and its type(planck cls fits)

```python
hdul[1].data
plt.plot(l*(l+1)*hdul[1].data["C_2_2"][0]/(2*np.pi))
```

### Matplotlib

#### colors

https://matplotlib.org/stable/gallery/color/named\_colors.html#tableau-palette

#### markers

https://matplotlib.org/stable/api/markers\_api.html

### h5 files operation

* open h5 files

```python
import h5py
def open_h5_file(filename):
    with h5py.File(filename, 'r') as f:
        dset=np.array(f['intermediate_map'])
        return dset
```

* get colomn name of h5 file

```python
list(f.keys())
dset=f['intermediate_map']
dset.attrs['colname']
```

### sav files operation

* load sav files

```python
from scipy.io import readsav
data = readsav(your_file, python_dict=True, verbose=True)
```

### Latex

* plot in latex

```python
from matplotlib.font_manager import FontProperties
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
```

### Plots

* some codes for subplots

```python
fig, axes= plt.subplots(nrows=3, ncols=2, sharex='col',figsize=(10,7))
fmt = ticker.FormatStrFormatter('%.1f')
ax1.yaxis.set_major_formatter(fmt)
ax1=axes[0,1]
ax1.set_ylabel('string',fontsize=10)
ax1.set_ylim((2,10))
ax1.tick_params(bottom=True,top=True,left=True,right=True,which='both',direction='in')
plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
fig.align_ylabels(axes[:,0])
fig.align_ylabels(axes[:,1])
plt.savefig('string',dpi=300,pad_inches = 0, bbox_inches='tight')
```

### CMB tasks

* check maps

```python
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

m = np.load('')
m = hp.read_map('',field=0)
m = hp.read_map('',field=(0,1,2))

hp.mollview(m)
hp.orthview(m, rot=[100,50,0],half_sky=True)
hp.gnomview(m, rot=[lon, lat, 0])
plt.show()
```

* get map depth for noise level

```python
def get_map_depth(hit_num:'the number hit the detector in long period',NET:'muK sqrt(s)',sampling_freq:'Hz',detector_freq:'GHz'):
    NET_set=NET*np.ones(len(hit_num))
    sampling_freq_set=sampling_freq*np.ones(len(hit_num))
    sigma=NET_set/(np.sqrt(hit_num/sampling_freq_set))
    return sigma
```

* multiply gaussian noise

```python
np.random.normal(loc=0, scale=1, size=your size)
```

* convert inf to np.inf then to 0

```python
# Create a vector with some infinite elements
x = np.array([1, 2, float('inf'), 4, float('inf'), 6])

# Replace "inf" with numpy.inf
x[x == float('inf')] = np.inf

# Check which elements are infinite
mask = np.isinf(x)

# Set infinite elements to zero
x[mask] = 0

# Print the modified vector
print(x)
```

* rotate coordinate

```python
import healpy as hp
r = hp.Rotator(coord=['C','G'])
mask_G = r.rotate_map_pixel(mask)
```

* get mask from variance map

```python
np.where(map_variace>0,1,0)
```

### Utils

* work flow

```bash
python ~/bin/change_sbatch_name.py ~/work/dc2/ILC_test/test.py
sbatch ~/bin/submit_sbatch.sh
```

* absolute path

```bash
/afs/ihep.ac.cn/users/w/wangyiming25/
```

* minimum import for test

```python
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import glob
import os,sys
from pathlib import Path
```

* logger

```python
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s -%(name)s - %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

* find files by python

```python
import glob
def find_files(directory, prefix, subfix):
    return glob.glob(f'{directory}/**/{prefix}*{subfix}', recursive=True)
```

* print options

```python
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
```

* debugger

```bash
python -m pdb your_python_code
```

* name variable

```
tmp:temporary
res:result
cnt:count
idx:index
sum:summary
avg/mean:average
max/min
prev/next:previous
init/mid/final:intermediate_results
```

* transfer input parameter names

```python
def plot_map(*args):
    stack = inspect.stack()
    frame = stack[1][0]
    locals_dict = frame.f_locals
    for var in args:
        for key, value in locals_dict.items():
            if value is var:
                hp.mollview(var,title=f'{key}')
    plt.show()
```

* save a plot to a directory in your home directory

```python
import os
import matplotlib.pyplot as plt

# Your code to generate a plot here...

# Define the directory where you want to save the file.
# This will be a directory named 'some_path' in your home directory.
directory = os.path.join(os.path.expanduser('~'), 'some_path')

# If the directory does not exist, create it.
if not os.path.exists(directory):
    os.makedirs(directory)

# Define the filename of the plot.
filename = 'my_plot.png'

# Full path to the file
full_file_path = os.path.join(directory, filename)

# Save the plot to the file.
plt.savefig(full_file_path, dpi=300)
```

* generate and save seeds

```python
import numpy as np
import secrets

def check_unique_seeds(seeds):
    # if the seeds array are unique, return True
    seen = {}
    for seed in seeds:
        if seed in seen:
            return False  # Duplicate found
        seen[seed] = True
    return True

seeds_number = 2000
seeds = [secrets.randbits(32) for _ in range(seeds_number)]
print(f'{seeds=}')
np.save('../seeds_fg_2k.npy', np.array(seeds))

seeds_arr = np.load('../seeds_fg_2k.npy', allow_pickle=True)
print(f'{seeds_arr.shape=}')

# print(f'{type(seeds_arr[0])=}')

# np.random.seed(seed=seeds_arr[0])

print(check_unique_seeds(seeds_arr))
```

### Docs

#### tips

* Argument passing sys.argv is a list in Python that contains the script name and any additional arguments passed to the script. When using the -c or -m options, sys.argv\[0] is set to -c or the full name of the module, respectively. Any options found after -c or -m are not processed by the interpreter and are left in sys.argv for the command or module to handle.
* raw strings:print(r'some string')
* range() is not a list, but an object returns the successive items of hte desired sequence when you iterate over it. Called iterable
* dir() lists all the function and variable definition
* match statements

```python
class Point:
    x: int
    y: int

def where_is(point):
    match point:
        case Point(x=0, y=0):
            print("Origin")
        case Point(x=0, y=y):
            print(f"Y={y}")
        case Point(x=x, y=0):
            print(f"X={x}")
        case Point():
            print("Somewhere else")
        case _:
            print("Not a point")
```

#### keywords argment

```python
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])
```

#### Special parameters

```python
def f(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
      -----------    ----------     ----------
        |             |                  |
        |        Positional or keyword   |
        |                                - Keyword only
         -- Positional only
```

#### walrus operator

assign expression value to a variable

```python
if (x := input("Enter a number: ")).isdigit() and (x := int(x)) > 0:
    print("You entered a positive number!")
else:
    print("You entered a non-positive number.")
```

#### Looping Techniques

```python
dictinary:
knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k, v)
sequence:
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)
many sequences:
questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
    print('What is your {0}?  It is {1}.'.format(q, a))
```

#### formatted string

```python
bugs = 'roaches'
count = 13
area = 'living room'
print(f'Debugging {bugs=} {count=} {area=}')
# Debugging bugs='roaches' count=13 area='living room'
```

#### save list and dict in a json file

```python
import json
x = ['an','example']
with open('filename','r+') as f:
    json.dump(x, f)
    copy_of_x = json.load(f)
```

#### namespace

namespace is a mapping from names to object. Most namespaces are implemented as Python dictionaries.

#### static method

a method belongs to a class rather than an instance of the class.(we don't need the self to be passed)

```python
class Calculator:

    # create addNumbers static method
    @staticmethod
    def addNumbers(x, y):
        return x + y

print('Product:', Calculator.addNumbers(15, 110))
```

#### super method

```python
class Parent:
    def __init__(self):
        print("Parent class initialization")

    def some_method(self):
        print("Parent class method")

class Child(Parent):
    def __init__(self):
        super().__init__()  # Call the __init__ method of the parent class
        print("Child class initialization")

    def some_method(self):
        super().some_method()  # Call the parent class's implementation of some_method
        print("Child class method")

# Create an instance of Child
child = Child()
# Output:
# Parent class initialization
# Child class initialization

child.some_method()
# Output:
# Parent class method
# Child class method
```

#### dataclass

dataclass is a decorator available in Python 3.7 and later versions that can be used to automatically generate special methods for classes, such as the **init**, **repr**, and **eq** methods, based on class attributes. The primary purpose of data classes is to simplify the creation of classes that mostly serve as containers for data and require minimal behavior.

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0)
print(p1)  # Output: Point(x=1.0, y=2.0)
print(p1 == p2)  # Output: True
```

#### python directory structure

```markdown
my_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── models/
│       ├── __init__.py
│       └── my_model.py
├── tests/
│   ├── __init__.py
│   ├── test_helpers.py
│   └── test_my_model.py
├── docs/
├── README.md
└── setup.py
```

#### iterable and iterator

list, set, dict.items() are all iterable Iterating over an iterable: An iterable is an object that implements the **iter**() method, which returns an iterator. When you use a for loop (or other iteration construct) on an iterable, Python automatically calls iter() on the object to get an iterator, and then calls next() on the iterator to get the values one by one. Once all values have been retrieved and next() is called again, the iterator raises a StopIteration exception, which Python catches and handles by ending the loop.

```python
my_list = [1, 2, 3]  # a list is an iterable
for item in my_list:
    print(item)  # prints 1, then 2, then 3
my_iterator = iter([1, 2, 3])  # create an iterator from a list

for item in my_iterator:
    print(item)  # prints 1, then 2, then 3

for item in my_iterator:
    print(item)  # prints nothing, because the iterator is exhausted
```

## ArchLinux

### pacman

* new server

```bash
pacman -Syu
pacman -S packagename
```

## Useful tools

### For Shell

#### Zoxide

rust cd tools, `zi` command is very useful

#### eza

rust ls tools

#### fzf

fuzzy finder

#### toolong

A terminal application to view, tail, merge, and search log files (python)

#### history command

```bash
bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'
```

#### cd git directory

```bash
alias zb='cd "$(git rev-parse --show-toplevel)"'
```
