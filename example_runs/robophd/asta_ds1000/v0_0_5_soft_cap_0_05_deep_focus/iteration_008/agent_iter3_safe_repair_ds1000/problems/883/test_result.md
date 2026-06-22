## Test output:
```

```
## Test error:
```
/ds1000/test_883.py:17: ClusterWarning: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix
  Z = linkage(np.array(data_matrix), "ward")
Traceback (most recent call last):
  File "/ds1000/test_883.py", line 73, in <module>
    test_execution(solution)
  File "/ds1000/test_883.py", line 63, in test_execution
    assert exec_test(test_env["result"], expected_result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

```