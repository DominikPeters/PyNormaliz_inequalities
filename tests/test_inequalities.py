import unittest
from PyNormaliz_inequalities import Variable, Expression, Inequality, InequalitySystem

class TestVariable(unittest.TestCase):
    def setUp(self):
        Variable.reset_counter()

    def test_variable_creation(self):
        v1 = Variable()
        v2 = Variable()
        self.assertEqual(v1.id, 1)
        self.assertEqual(v2.id, 2)

    def test_variable_operations(self):
        v1 = Variable()
        v2 = Variable()
        expr = v1 + v2
        self.assertEqual(str(expr), "x_1 + x_2")
        expr = v1 - v2
        self.assertEqual(str(expr), "x_1 + -1*x_2")
        expr = v1 * 2
        self.assertEqual(str(expr), "2*x_1")
        expr = 3 * v2
        self.assertEqual(str(expr), "3*x_2")

class TestExpression(unittest.TestCase):
    def setUp(self):
        Variable.reset_counter()

    def test_expression_creation(self):
        v1 = Variable()
        v2 = Variable()
        expr = Expression({v1: 1, v2: 2}, 3)
        self.assertEqual(str(expr), "x_1 + 2*x_2 + 3")

    def test_expression_operations(self):
        v1 = Variable()
        v2 = Variable()
        expr1 = Expression({v1: 1, v2: 2}, 3)
        expr2 = Expression({v1: 2, v2: 1}, 1)
        expr = expr1 + expr2
        self.assertEqual(str(expr), "3*x_1 + 3*x_2 + 4")
        expr = expr1 - expr2
        self.assertEqual(str(expr), "-1*x_1 + x_2 + 2")
        expr = expr1 * 2
        self.assertEqual(str(expr), "2*x_1 + 4*x_2 + 6")

class TestInequality(unittest.TestCase):
    def setUp(self):
        Variable.reset_counter()

    def test_inequality_creation(self):
        v1 = Variable()
        v2 = Variable()
        expr = v1 + v2
        ineq = Inequality(expr, ">=")
        self.assertEqual(str(ineq), "x_1 + x_2 >= 0")

    def test_inequality_operations(self):
        v1 = Variable()
        v2 = Variable()
        ineq = (v1 + v2) >= 0
        self.assertEqual(str(ineq), "x_1 + x_2 >= 0")
        ineq = (v1 - v2) > 0
        self.assertEqual(str(ineq), "x_1 + -1*x_2 > 0")

class TestInequalitySystem(unittest.TestCase):
    def test_add_inequality(self):
        v1 = Variable()
        v2 = Variable()
        ineq1 = (v1 + v2) >= 0
        ineq2 = (v1 - v2) > 0
        system = InequalitySystem()
        system.add_inequality(ineq1)
        system.add_inequality(ineq2)
        self.assertEqual(len(system.inequalities), 2)

    def test_is_homogeneous(self):
        v1 = Variable()
        v2 = Variable()
        ineq1 = (v1 + v2) >= 0
        ineq2 = (v1 - v2) > 0
        system = InequalitySystem()
        system.add_inequality(ineq1)
        system.add_inequality(ineq2)
        self.assertTrue(system.is_homogeneous())

    def test_get_vecs(self):
        v1 = Variable()
        v2 = Variable()
        ineq1 = (v1 + v2) >= 0
        ineq2 = (v1 - v2) > 0
        system = InequalitySystem()
        system.add_inequality(ineq1)
        system.add_inequality(ineq2)
        weak_vecs, strict_vecs = system.get_vecs()
        self.assertEqual(len(weak_vecs), 1)
        self.assertEqual(len(strict_vecs), 1)

    def test_construct_homogeneous_cone(self):
        v1 = Variable()
        v2 = Variable()
        ineq1 = (v1 + v2) >= 0
        ineq2 = (v1 - v2) > 0
        system = InequalitySystem()
        system.add_inequality(ineq1)
        system.add_inequality(ineq2)
        cone = system.construct_homogeneous_cone()
        self.assertIsNotNone(cone)

    def test_as_normitz_input_file(self):
        v1 = Variable()
        v2 = Variable()
        ineq1 = (v1 + v2) >= 0
        ineq2 = (v1 - v2) > 0
        system = InequalitySystem()
        system.add_inequality(ineq1)
        system.add_inequality(ineq2)
        input_file = system.as_normitz_input_file()
        self.assertIn("amb_space 2", input_file)
        self.assertIn("inequalities 1", input_file)
        self.assertIn("excluded_faces 1", input_file)

if __name__ == "__main__":
    unittest.main()
