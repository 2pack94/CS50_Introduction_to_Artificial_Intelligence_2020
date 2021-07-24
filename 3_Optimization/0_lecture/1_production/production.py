import scipy.optimize

# Linear Programming:
# Optimize a linear equation by minimizing a cost function.

# Problem:
# Two machines, x1 and x2. x1 costs $50/hour to run, x2 costs $80/hour to run. The goal is to minimize cost.
# x1 requires 5 units of labor per hour. x2 requires 2 units of labor per hour. Total of 20 units of labor to spend.
# x1 produces 10 units of output per hour. x2 produces 12 units of output per hour. Company needs 90 units of output.

# Cost Function: 50x_1 + 80x_2
# Constraint 1: 5x_1 + 2x_2 <= 20
# Constraint 2: -10x_1 + -12x_2 <= -90

result = scipy.optimize.linprog(
    [50, 80],  # Coefficients for Cost function
    A_ub=[[5, 2], [-10, -12]],  # Coefficients for inequalities (ub: upper bound)
    b_ub=[20, -90],  # Constraints for inequalities
)

if result.success:
    print(f"X1: {round(result.x[0], 2)} hours")
    print(f"X2: {round(result.x[1], 2)} hours")
else:
    print("No solution")
