# PyNormaliz_inequalities

[Normaliz](https://github.com/Normaliz/Normaliz) is a tool for various computations in discrete convex geometry. In particular it supports computations in polytopes and polyhedra specified by a system of linear inequalities.
For example, it allows to compute the Hilbert or Ehrhart series of a polytope or polyhedron, and thereby count the number of integer points in it.

It is possible to interact with Normaliz using the Python package [PyNormaliz](https://github.com/Normaliz/PyNormaliz), but the interface requires users to specify inequalities in a vector format. 
This package provides a more convenient interface for specifying inequalities in a natural format, and includes utility functions for interacting with PyNormaliz.

## Installation

Normaliz and PyNormaliz must be installed on your system to use this package. You can install these packages using the following commands:

```sh
git clone --depth 1 https://github.com/Normaliz/Normaliz.git
cd Normaliz
./install_normaliz.sh
./install_pynormaliz.sh
```

To install this package, use pip:

```sh
pip install PyNormaliz_inequalities
```

## Usage

The main point of the package is to allow users to specify inequalities in a natural format.
The main components are `Variable`s that can be combined into `Expression`s and `Inequality`s, which can be collected into an `InequalitySystem`, which can then be passed to PyNormaliz.

### Example: Basic Usage

Let's count the number of integer pairs `(a,b)` with `a + b = n` that satisfy the inequalities `a >= 0`, `b >= 0`, and `a >= b`, as a function of `n`.

```python
from PyNormaliz_inequalities import Variable, InequalitySystem, evaluate_quasipolynomial

a = Variable()
b = Variable()

inequalities = InequalitySystem()
inequalities.add_inequality(a >= 0)
inequalities.add_inequality(b >= 0)
inequalities.add_inequality(a >= b)

quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial()
print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
```
Output: `[0, 1, 1, 2, 2, 3, 3, 4, 4, 5]`

We can also output the resulting problem in the Normaliz input file format:

```python
with open("example.in", "w") as f:
    f.write(inequalities.as_normitz_input_file())
```
This produces a file with the content
```
amb_space 2
inequalities 3
1 0
0 1
1 -1
total_degree
```
This can then be passed to Normaliz by running `normaliz example.in`, which produces a file `example.out`.

The package also supports grading:
```python
cone = inequalities.construct_homogeneous_cone(grading=a+3*b)
with open("example.in", "w") as f:
    f.write(inequalities.as_normitz_input_file(grading=a+3*b))
```

### Example: Condorcet Paradox

Let's compute the fraction of anonymous preference profiles in which the Condorcet paradox occurs. We consider three candidates. We compute this fraction as follows: we find, for each number of voters `n`, the number of preference profiles in which a specific candidate is the Condorcet winner, multiply this number by `3` (since there are `3` possible Condorcet winners), and subtract this from the total number of preference profiles. Now we have the number of preference profiles in which the Condorcet paradox occurs. We divide this by the total number of preference profiles to get the fraction.

```python
from PyNormaliz_inequalities import Variable, InequalitySystem, evaluate_quasipolynomial
import itertools
import math
import matplotlib.pyplot as plt

candidates = [0, 1, 2]
rankings = list(itertools.permutations(candidates))

condorcet_winner = 0
voter_count = {ranking: Variable() for ranking in rankings}

margins = {}
for x, y in itertools.combinations(candidates, 2):
    # net number of voters who prefer x to y
    margins[x, y] = sum(voter_count[ranking] for ranking in rankings if ranking.index(x) < ranking.index(y)) \
        - sum(voter_count[ranking] for ranking in rankings if ranking.index(x) > ranking.index(y))

inequalities = InequalitySystem()
for ranking in rankings:
    inequalities.add_inequality(voter_count[ranking] >= 0)
for x in candidates:
    if x == condorcet_winner:
        continue
    inequalities.add_inequality(margins[condorcet_winner, x] > 0)

print(inequalities.as_normitz_input_file())

quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial()

Ns = range(100)
num_profiles = [math.comb(len(rankings) + n - 1, n) for n in Ns]
num_profiles_with_condorcet_winner = [evaluate_quasipolynomial(quasipolynomial, n) for n in Ns]
num_profiles_with_condorcet_paradox = [num_profiles[n] - 3 * num_profiles_with_condorcet_winner[n] for n in Ns]
fraction_with_condorcet_paradox = [num_profiles_with_condorcet_paradox[n] / num_profiles[n] for n in Ns]

plt.plot(Ns, fraction_with_condorcet_paradox)
plt.xlabel('Number of voters')
plt.ylabel('Fraction of profiles with Condorcet paradox')
plt.savefig("condorcet.png", dpi=300)
plt.show()
```
<img src="https://github.com/user-attachments/assets/9e3b0239-2f3f-44ef-b6b6-fb7e0c606a87" width="500">

## Explanation

The `PyNormaliz_inequalities` package provides a convenient interface to PyNormaliz, allowing users to specify inequalities in a natural format. It supports creating variables, expressions, and inequalities, and converting them to vector representations suitable for PyNormaliz. The package also includes functionality to construct homogeneous cones and compute Hilbert quasi-polynomials.

The main components of the package are:

- `Variable`: Represents a variable in an inequality.
- `Expression`: Represents a linear expression involving variables.
- `Inequality`: Represents an inequality involving expressions.
- `InequalitySystem`: Manages a system of inequalities and provides methods to interact with PyNormaliz.

The package also includes utility functions for converting inequalities to vector representations and evaluating quasi-polynomials.
