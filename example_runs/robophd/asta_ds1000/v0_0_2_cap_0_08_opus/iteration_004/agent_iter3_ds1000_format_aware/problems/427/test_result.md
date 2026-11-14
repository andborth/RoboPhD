## Test output:
```

```
## Test error:
```
Traceback (most recent call last):
  File "/ds1000/test_427.py", line 52, in <module>
    test_execution(solution)
  File "/ds1000/test_427.py", line 49, in test_execution
    assert exec_test(test_env["result"], expected_result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ds1000/test_427.py", line 32, in exec_test
    np.testing.assert_array_equal(result, ans)
  File "/usr/local/lib/python3.11/site-packages/numpy/testing/_private/utils.py", line 920, in assert_array_equal
    assert_array_compare(operator.__eq__, x, y, err_msg=err_msg,
  File "/usr/local/lib/python3.11/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numpy/testing/_private/utils.py", line 713, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Arrays are not equal

(shapes (6,), (1, 6) mismatch)
 x: array([0, 0, 0, 0, 0, 1], dtype=uint8)
 y: array([[0, 0, 0, 0, 0, 1]])

```