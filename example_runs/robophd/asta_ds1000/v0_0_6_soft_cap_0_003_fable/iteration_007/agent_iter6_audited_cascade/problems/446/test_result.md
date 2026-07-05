## Test output:
```

```
## Test error:
```
Traceback (most recent call last):
  File "/ds1000/test_446.py", line 48, in <module>
    test_execution(solution)
  File "/ds1000/test_446.py", line 45, in test_execution
    assert exec_test(test_env["result"], expected_result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ds1000/test_446.py", line 27, in exec_test
    np.testing.assert_array_equal(result, ans)
  File "/usr/local/lib/python3.11/site-packages/numpy/testing/_private/utils.py", line 920, in assert_array_equal
    assert_array_compare(operator.__eq__, x, y, err_msg=err_msg,
  File "/usr/local/lib/python3.11/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numpy/testing/_private/utils.py", line 797, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Arrays are not equal

Mismatched elements: 6 / 8 (75%)
Max absolute difference: 3
Max relative difference: 3.
 x: array([7, 5, 2, 4, 6, 3, 1, 0])
 y: array([7, 6, 4, 1, 3, 5, 2, 0])

```