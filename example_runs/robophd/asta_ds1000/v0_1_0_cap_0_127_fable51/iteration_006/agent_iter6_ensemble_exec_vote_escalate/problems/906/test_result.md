## Test output:
```

```
## Test error:
```
Traceback (most recent call last):
  File "/ds1000/test_906.py", line 87, in <module>
    test_execution(solution)
  File "/ds1000/test_906.py", line 83, in test_execution
    exec(code, test_env)
  File "<string>", line 16, in <module>
  File "<string>", line 12, in fit_col
  File "/usr/local/lib/python3.11/site-packages/sklearn/base.py", line 1351, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/sklearn/linear_model/_base.py", line 578, in fit
    X, y = self._validate_data(
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/sklearn/base.py", line 650, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/sklearn/utils/validation.py", line 1192, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/sklearn/utils/validation.py", line 787, in check_array
    raise TypeError(
TypeError: np.matrix is not supported. Please convert to a numpy array with np.asarray. For more information see: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html

```