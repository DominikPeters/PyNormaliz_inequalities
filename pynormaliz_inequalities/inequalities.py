from collections import defaultdict
from typing import Dict, List, Union
from PyNormaliz import Cone
import itertools

class Variable:
    _counter: int = 0

    def __init__(self) -> None:
        Variable._counter += 1
        self.id: int = Variable._counter

    @classmethod
    def reset_counter(cls) -> None:
        cls._counter = 0

    def __add__(self, other: Union['Variable', 'Expression', int, float]) -> 'Expression':
        return Expression({self: 1}) + other

    def __radd__(self, other: Union[int, float]) -> 'Expression':
        return Expression({self: 1}) + other

    def __mul__(self, other: Union[int, float]) -> 'Expression':
        return Expression({self: other})

    def __rmul__(self, other: Union[int, float]) -> 'Expression':
        return Expression({self: other})

    def __sub__(self, other: Union['Variable', 'Expression', int, float]) -> 'Expression':
        return self + (-1 * other)

    def __rsub__(self, other: Union[int, float]) -> 'Expression':
        return (-1 * self) + other

    def __neg__(self) -> 'Expression':
        return Expression({self: -1})

    def __ge__(self, other: Union['Variable', 'Expression', int, float]) -> 'Inequality':
        return Expression({self: 1}) >= other

    def __gt__(self, other: Union['Variable', 'Expression', int, float]) -> 'Inequality':
        return Expression({self: 1}) > other

    def __le__(self, other: Union['Variable', 'Expression', int, float]) -> 'Inequality':
        return Expression({self: 1}) <= other

    def __lt__(self, other: Union['Variable', 'Expression', int, float]) -> 'Inequality':
        return Expression({self: 1}) < other

    def __eq__(self, other: Union['Variable', 'Expression', int, float]) -> bool:
        return Expression({self: 1}) == other

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Variable({self.id})"

    def __str__(self) -> str:
        return f"x_{self.id}"

class Expression:
    def __init__(self, coeffs: Dict[Variable, Union[int, float]] = None, constant: Union[int, float] = 0) -> None:
        self.coeffs: Dict[Variable, Union[int, float]] = defaultdict(int, coeffs or {})
        self.constant: Union[int, float] = constant

    def __add__(self, other: Union['Expression', Variable, int, float]) -> 'Expression':
        if isinstance(other, (int, float)):
            return Expression(self.coeffs.copy(), self.constant + other)
        elif isinstance(other, Variable):
            new_coeffs = self.coeffs.copy()
            new_coeffs[other] += 1
            return Expression(new_coeffs, self.constant)
        elif isinstance(other, Expression):
            new_coeffs = self.coeffs.copy()
            for var, coeff in other.coeffs.items():
                new_coeffs[var] += coeff
            return Expression(new_coeffs, self.constant + other.constant)
        else:
            return NotImplemented

    def __radd__(self, other: Union[int, float]) -> 'Expression':
        return self + other

    def __mul__(self, other: Union[int, float]) -> 'Expression':
        if isinstance(other, (int, float)):
            new_coeffs = {var: coeff * other for var, coeff in self.coeffs.items()}
            return Expression(new_coeffs, self.constant * other)
        else:
            return NotImplemented

    def __rmul__(self, other: Union[int, float]) -> 'Expression':
        return self * other

    def __sub__(self, other: Union['Expression', Variable, int, float]) -> 'Expression':
        return self + (-1 * other)

    def __rsub__(self, other: Union[int, float]) -> 'Expression':
        return (-1 * self) + other

    def __neg__(self) -> 'Expression':
        return self * -1

    def __ge__(self, other: Union['Expression', Variable, int, float]) -> 'Inequality':
        return Inequality(self - other, '>=')

    def __gt__(self, other: Union['Expression', Variable, int, float]) -> 'Inequality':
        return Inequality(self - other, '>')

    def __le__(self, other: Union['Expression', Variable, int, float]) -> 'Inequality':
        return Inequality(other - self, '>=')

    def __lt__(self, other: Union['Expression', Variable, int, float]) -> 'Inequality':
        return Inequality(other - self, '>')

    def __eq__(self, other: Union['Expression', Variable, int, float]) -> bool:
        return Inequality(self - other, '==')

    def __repr__(self) -> str:
        terms = [f"{coeff}*{var}" for var, coeff in self.coeffs.items() if coeff != 0]
        if self.constant != 0 or not terms:
            terms.append(str(self.constant))
        return " + ".join(terms)

    def __str__(self) -> str:
        terms = [f"{coeff}{var}" if coeff != 1 else f"{var}" for var, coeff in self.coeffs.items() if coeff != 0]
        if self.constant != 0 or not terms:
            terms.append(str(self.constant))
        return " + ".join(terms)

class Inequality:
    def __init__(self, expr: Expression, op: str) -> None:
        self.expr = expr
        self.op = op
        if op == '<':
            self.op = '>'
            self.expr = -1 * self.expr
        elif op == '<=':
            self.op = '>='
            self.expr = -1 * self.expr

    def to_vec(self) -> List[Union[int, float]]:
        all_variables = Variable._counter  # Get the total number of variables created
        vec = [0] * all_variables  # Initialize vector with zeros for all possible variables
        for i, (var, coeff) in enumerate(self.expr.coeffs.items()):
            vec[i] = coeff
        vec.append(self.expr.constant)  # Append constant at the end
        return vec

    def is_satisfied_by(self, values: Dict[Variable, Union[int, float]]) -> bool:
        LHS_value = sum(coeff * values[var] for var, coeff in self.expr.coeffs.items()) + self.expr.constant
        if self.op == '>=':
            return LHS_value >= 0
        elif self.op == '>':
            return LHS_value > 0
        elif self.op == '==':
            return LHS_value == 0
        else:
            raise ValueError("Invalid inequality operator", self.op)

    def __eq__(self, other: 'Inequality') -> bool:
        return isinstance(other, Inequality) and self.to_vec() == other.to_vec()

    def __repr__(self) -> str:
        return f"{self.expr} {self.op} 0"

    def __str__(self) -> str:
        return f"{str(self.expr)} {self.op} 0"

class InequalitySystem:
    def __init__(self) -> None:
        self.inequalities: List[Inequality] = []

    def add_inequality(self, inequality: Inequality) -> None:
        if inequality.op != '==':
            self.inequalities.append(inequality)
        else:
            self.inequalities.append(inequality.expr >= 0)
            self.inequalities.append(inequality.expr <= 0)

    def is_homogeneous(self) -> bool:
        return all(ineq.expr.constant == 0 for ineq in self.inequalities)

    def compute_number_of_lattice_points(self, n) -> int:
        used_vars = {var.id for ineq in self.inequalities for var in ineq.expr.coeffs}
        num_points = 0
        for choice in itertools.combinations_with_replacement(used_vars, n):
            value = {var: choice.count(var) for var in used_vars}
            if all(ineq.is_satisfied_by(value) for ineq in self.inequalities):
                num_points += 1
        return num_points

    def get_vecs(self) -> List[List[Union[int, float]]]:
        """Converts inequalities to vector representation.

        Converts internal representation of inequalities to vectors suitable for further processing.
        Variables that don't appear in any inequality are omitted from the output.

        Returns:
            tuple[list[list[float]], list[list[float]]]: A tuple containing two lists:
                1. List of vectors representing weak inequalities (>=)
                2. List of vectors representing strict inequalities (>)
        """
        used_vars = sorted({var.id for ineq in self.inequalities for var in ineq.expr.coeffs})

        homogeneous = self.is_homogeneous()

        weak_inequality_vecs = []
        strict_inequality_vecs = []

        # Initialize the result vectors
        vecs = []
        for ineq in self.inequalities:
            vec = [0] * len(used_vars)
            for i, var in enumerate(used_vars):
                if var in ineq.expr.coeffs:
                    vec[i] = ineq.expr.coeffs[var]
            if not homogeneous:
                vec.append(ineq.expr.constant)
            if ineq.op == '>=':
                weak_inequality_vecs.append(vec)
            elif ineq.op == '>':
                strict_inequality_vecs.append(vec)
            else:
                raise ValueError("Invalid inequality operator", ineq.op)

        return (weak_inequality_vecs, strict_inequality_vecs)

    def construct_homogeneous_cone(self) -> Cone:
        """Constructs a homogeneous cone as a PyNormaliz Cone object.

        Constructs a homogeneous cone from the given inequalities. The inequalities should be homogeneous.

        Returns:
            Cone: A PyNormaliz Cone object representing the homogeneous cone.
        """
        homogeneous = self.is_homogeneous()
        # if not homogeneous:
        #     raise ValueError("Inequality system is not homogeneous")

        weak_inequality_vecs, strict_inequality_vecs = self.get_vecs()
        if homogeneous:
            num_vars = len((weak_inequality_vecs + strict_inequality_vecs)[0])
            return Cone(
                inequalities=weak_inequality_vecs, 
                excluded_faces=strict_inequality_vecs, 
                grading=[[1] * num_vars]
            )
        else:
            num_vars = len((weak_inequality_vecs + strict_inequality_vecs)[0]) - 1
            return Cone(
                inhom_inequalities=weak_inequality_vecs, 
                inhom_excluded_faces=strict_inequality_vecs, 
                grading=[[1] * num_vars]
            )

    def as_normitz_input_file(self) -> str:
        """Converts the inequality system to a string in Normaliz input file format.

        Converts the inequality system to a string in Normaliz input file format. The output can be used as input to the Normaliz command line tool.

        Returns:
            str: A string in Normaliz input file format.
        """
        homogeneous = self.is_homogeneous()
        weak_inequality_vecs, strict_inequality_vecs = self.get_vecs()
        num_cols = len((weak_inequality_vecs + strict_inequality_vecs)[0])
        num_vars = num_cols if homogeneous else num_cols - 1 # last column is not a variable if homogeneous
        num_weak_inequalities = len(weak_inequality_vecs)
        num_strict_inequalities = len(strict_inequality_vecs)

        # to simplify output, check if constraints include non-negativity constraints for all variables
        unit_vectors = [tuple(1 if i == j else 0 for j in range(num_cols)) for i in range(num_vars)]
        contains_non_negativity_constraints = all(vec in weak_inequality_vecs for vec in unit_vectors)
        if contains_non_negativity_constraints:
            # delete non-negativity constraints from weak_inequality_vecs
            weak_inequality_vecs = [vec for vec in weak_inequality_vecs if vec not in unit_vectors]

        lines = []
        lines.append(f"amb_space {num_vars}")

        if homogeneous:
            if weak_inequality_vecs:
                lines.append(f"inequalities {num_weak_inequalities}")
                for vec in weak_inequality_vecs:
                    lines.append(" ".join(str(x) for x in vec))
            if strict_inequality_vecs:
                lines.append(f"excluded_faces {num_strict_inequalities}")
                for vec in strict_inequality_vecs:
                    lines.append(" ".join(str(x) for x in vec))
        else:
            if weak_inequality_vecs:
                lines.append(f"inhom_inequalities {num_weak_inequalities}")
                for vec in weak_inequality_vecs:
                    lines.append(" ".join(str(x) for x in vec))
            if strict_inequality_vecs:
                lines.append(f"inhom_excluded_faces {num_strict_inequalities}")
                for vec in strict_inequality_vecs:
                    lines.append(" ".join(str(x) for x in vec))

        if contains_non_negativity_constraints:
            lines.append("nonnegative")

        lines.append("total_degree") # grading = all 1s

        return "\n".join(lines)


####################
#### Example use ###
####################

def evaluate_quasipolynomial(qp, n):
    # e.g. qp == [[384, 640, 408, 124, 18, 1], [135, 297, 234, 86, 15, 1], 384]
    period = len(qp) - 1
    denominator = qp[-1]
    polynomial = qp[n % period]
    return round(sum((c / denominator) * n**i for i, c in enumerate(polynomial)))

if __name__ == "__main__":

    import itertools
    A = [1,2,3]
    perms = list(itertools.permutations(A))
    profile = {perm : Variable() for perm in perms}

    def better(perm, x, y):
        return perm.index(x) < perm.index(y)

    inequalities = InequalitySystem()

    # 1 should be a Condorcet winner

    margin = {(x, y) : sum([profile[perm] for perm in perms if better(perm, x, y)]) - sum([profile[perm] for perm in perms if better(perm, y, x)]) for x in A for y in A if x != y}

    # there should be more voters who prefer 1 to 2 than voters who prefer 2 to 1
    ineq1 = margin[(1, 2)] > 0

    # there should be more voters who prefer 1 to 3 than voters who prefer 3 to 1
    ineq2 = margin[(1, 3)] > 0

    inequalities.add_inequality(ineq1)
    inequalities.add_inequality(ineq2)

    # non-negativity
    for perm in perms:
        ineq = profile[perm] >= 0
        inequalities.add_inequality(ineq)

    # polyhedron = Cone(inequalities=to_vecs(inequalities, homogeneous=True), grading=[[1, 1, 1, 1, 1, 1]], excluded_faces=[[1, 1, -1, -1, 1, -1], [1, 1, 1, -1, -1, -1]])
    polyhedron = inequalities.construct_homogeneous_cone()
    # print(polyhedron.Multiplicity())
    quasipolynomial = polyhedron.HilbertQuasiPolynomial()

    print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
    print([inequalities.compute_number_of_lattice_points(n) for n in range(10)])

    inequalities = InequalitySystem()
    a = Variable()
    b = Variable()
    c = Variable()
    inequalities.add_inequality(a >= 0)
    inequalities.add_inequality(b >= 0)
    inequalities.add_inequality(c >= 0)
    inequalities.add_inequality(a > 2*b)
    inequalities.add_inequality(b > c)
    quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial()
    print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
    print([inequalities.compute_number_of_lattice_points(n) for n in range(10)])
    print(inequalities.as_normitz_input_file())

    inequalities = InequalitySystem()
    a = Variable()
    b = Variable()
    inequalities.add_inequality(a >= 0)
    inequalities.add_inequality(a - 2*b <= 0)
    inequalities.add_inequality(b >= 0)
    quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial(NoGradingDenom=True)
    print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
    print([inequalities.compute_number_of_lattice_points(n) for n in range(10)])
    print(inequalities.as_normitz_input_file())

    inequalities = InequalitySystem()
    x = Variable()
    y = Variable()
    inequalities.add_inequality(x >= 2)
    inequalities.add_inequality(y >= x)
    quasipolynomial = inequalities.construct_homogeneous_cone().HilbertQuasiPolynomial(NoGradingDenom=True)
    print([evaluate_quasipolynomial(quasipolynomial, n) for n in range(10)])
    print([inequalities.compute_number_of_lattice_points(n) for n in range(10)])
    print(inequalities.as_normitz_input_file())
