# Faircorels

Welcome to the python binding of the FairCORELS algorithm!

## Overview

FairCORELS is a modified version of CORELS to learn fair rule list. 


## Installation
#### Ubuntu

```
sudo apt install libgmp-dev
pip install faircorels
```

#### Mac

```
# Install g++ and gmp
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install g++ gmp

pip install faircorels
```

#### Windows
Note: Python 2 is currently NOT supported on Windows.
```
pip install faircorels
```


## Examples

### Large dataset, loaded from [this file](https://raw.githubusercontent.com/fingoldin/pycorels/master/examples/data/compas.csv)

```python
from faircorels import *

# Load the dataset
X, y = load_from_csv("data/compas.csv")

# Create the model, with 10000 as the maximum number of iterations 
c = CorelsClassifier(n_iter=10000)

# Fit, and score the model on the training set
a = c.fit(X, y).score(X, y)

# Print the model's accuracy on the training set
print(a)
```

