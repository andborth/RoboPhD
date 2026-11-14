## Test output:
```

```
## Test error:
```
/ds1000/test_723.py:28: DeprecationWarning: Please import `csr_matrix` from the `scipy.sparse` namespace; the `scipy.sparse.csr` namespace is deprecated and will be removed in SciPy 2.0.0.
  assert type(result) == sparse.csr.csr_matrix
Traceback (most recent call last):
  File "/ds1000/test_723.py", line 52, in <module>
    test_execution(solution)
  File "/ds1000/test_723.py", line 49, in test_execution
    assert exec_test(test_env["result"], expected_result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ds1000/test_723.py", line 28, in exec_test
    assert type(result) == sparse.csr.csr_matrix
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

```