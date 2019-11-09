import warnings

import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def is_less_than_zero(x):
    result = (x < 0)
    return result.astype(int)

def is_greater_than_or_equal_to_zero(x):
    result = (x >= 0)
    return result.astype(int)

is_lt_zero = make_function(is_less_than_zero, "is_lt_zero", arity=1)
is_gte_zero = make_function(is_greater_than_or_equal_to_zero, "is_gte_zero", arity=1)

function_set = [is_lt_zero, is_gte_zero, "mul", "add", "neg"]

X = np.arange(-10, 11).reshape(-1, 1)
y = np.abs(X).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y.tolist())

my_abs_gp = SymbolicRegressor(function_set=function_set,
                              init_method="grow",
                              parsimony_coefficient=0.0625,
                              verbose=True)

my_abs_gp.fit(X_train, y_train)

print(my_abs_gp.score(X_test, y_test))
print(my_abs_gp._program)
