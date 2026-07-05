## Test output:
```

```
## Test error:
```
Traceback (most recent call last):
  File "/ds1000/test_812.py", line 47, in <module>
    test_execution(solution)
  File "/ds1000/test_812.py", line 44, in test_execution
    assert exec_test(test_env["result"], expected_result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ds1000/test_812.py", line 24, in exec_test
    assert np.allclose(result, ans, atol=1e-2)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numpy/core/numeric.py", line 2241, in allclose
    res = all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numpy/core/numeric.py", line 2351, in isclose
    return within_tol(x, y, atol, rtol)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numpy/core/numeric.py", line 2332, in within_tol
    return less_equal(abs(x-y), atol + rtol * abs(y))
                          ~^~
ValueError: operands could not be broadcast together with shapes (7,) (12,) 

```