# pynormaliz_inequalities
Python utility package for interacting with PyNormaliz

## Installation

To install the package, use pip:

```sh
pip install pynormaliz_inequalities
```

## Usage

This package provides a convenient interface to PyNormaliz, allowing users to specify inequalities in a natural format. Below are some usage examples:

### Example 1: Basic Usage

```python
from pynormaliz_inequalities import Variable, InequalitySystem, evaluate_quasipolynomial

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
from pynormaliz_inequalities import Variable, InequalitySystem, evaluate_quasipolynomial

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
from pynormaliz_inequalities import Variable, InequalitySystem, evaluate_quasipolynomial

x = Variable()
y = Variable()

inequalities = InequalitySystem()
inequalities.add_inequality(x >= 2)
inequalities.add_inequality(y >= x)

quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial()
print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
```

## Explanation

The `pynormaliz_inequalities` package provides a convenient interface to PyNormaliz, allowing users to specify inequalities in a natural format. It supports creating variables, expressions, and inequalities, and converting them to vector representations suitable for PyNormaliz. The package also includes functionality to construct homogeneous cones and compute Hilbert quasi-polynomials.

The main components of the package are:

- `Variable`: Represents a variable in an inequality.
- `Expression`: Represents a linear expression involving variables.
- `Inequality`: Represents an inequality involving expressions.
- `InequalitySystem`: Manages a system of inequalities and provides methods to interact with PyNormaliz.

The package also includes utility functions for converting inequalities to vector representations and evaluating quasi-polynomials.
