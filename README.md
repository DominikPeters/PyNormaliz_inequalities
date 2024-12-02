# PyNormaliz_inequalities

[Normaliz](https://github.com/Normaliz/Normaliz) is a tool for various computations in discrete convex geometry. In particular it supports computations in polytopes and polyhedra specified by a system of linear inequalities.
For example, it allows to compute the Hilbert or Ehrhart series of a polytope or polyhedron, and thereby count the number of integer points in it.

It is possible to interact with Normaliz using the Python package [PyNormaliz](https://github.com/Normaliz/PyNormaliz), but the interface requires users to specify inequalities in a vector format. 
This package provides a more convenient interface for specifying inequalities in a natural format, and includes utility functions for interacting with PyNormaliz.

## Installation

Normaliz and PyNormaliz must be installed on your system to use this package. You can install these packages using the following commands:

```sh
git clone https://github.com/Normaliz/Normaliz.git
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

### Example 1: Basic Usage

Let's count the number of integer pairs `(a,b)` with `a + b = n` that satisfy the inequalities `a >= 0`, `b >= 0`, and `a + b >= 1`, as a function of `n`.

```python
from PyNormaliz_inequalities import Variable, InequalitySystem, evaluate_quasipolynomial

a = Variable()
b = Variable()

inequalities = InequalitySystem()
inequalities.add_inequality(a >= 0)
inequalities.add_inequality(b >= 0)
inequalities.add_inequality(a + b >= 1)

quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial()
print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
```

### Example 2: Complex Inequalities

```python
from PyNormaliz_inequalities import Variable, InequalitySystem, evaluate_quasipolynomial

a = Variable()
b = Variable()
c = Variable()

inequalities = InequalitySystem()
inequalities.add_inequality(a >= 0)
inequalities.add_inequality(b >= 0)
inequalities.add_inequality(c >= 0)
inequalities.add_inequality(a > 2*b)
inequalities.add_inequality(b > c)

quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial()
print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
```

### Example 3: Using Inequality System

```python
from PyNormaliz_inequalities import Variable, InequalitySystem, evaluate_quasipolynomial

x = Variable()
y = Variable()

inequalities = InequalitySystem()
inequalities.add_inequality(x >= 2)
inequalities.add_inequality(y >= x)

quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial()
print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
```

## Explanation

The `PyNormaliz_inequalities` package provides a convenient interface to PyNormaliz, allowing users to specify inequalities in a natural format. It supports creating variables, expressions, and inequalities, and converting them to vector representations suitable for PyNormaliz. The package also includes functionality to construct homogeneous cones and compute Hilbert quasi-polynomials.

The main components of the package are:

- `Variable`: Represents a variable in an inequality.
- `Expression`: Represents a linear expression involving variables.
- `Inequality`: Represents an inequality involving expressions.
- `InequalitySystem`: Manages a system of inequalities and provides methods to interact with PyNormaliz.

The package also includes utility functions for converting inequalities to vector representations and evaluating quasi-polynomials.
