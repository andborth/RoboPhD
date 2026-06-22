## Test output:
```

```
## Test error:
```
Traceback (most recent call last):
  File "/ds1000/test_763.py", line 51, in <module>
    test_execution(solution)
  File "/ds1000/test_763.py", line 48, in test_execution
    assert exec_test(test_env["result"], expected_result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ds1000/test_763.py", line 28, in exec_test
    np.testing.assert_allclose(result, ans)
  File "/usr/local/lib/python3.11/site-packages/numpy/testing/_private/utils.py", line 1504, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/usr/local/lib/python3.11/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numpy/testing/_private/utils.py", line 797, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 50 / 50 (100%)
Max absolute difference: 5.07087283e-07
Max relative difference: 30.88943351
 x: array([-2.807769e-13,  1.451280e-07,  3.754214e-07,  4.827918e-07,
        4.373139e-07,  3.066597e-07,  1.900776e-07,  1.669176e-07,
        2.597018e-07,  4.117408e-07,  4.792955e-07,  2.382970e-07,...
 y: array([-2.807287e-13, -9.466243e-09, -1.928667e-08, -2.429550e-08,
       -2.237269e-08, -1.025980e-08,  1.657809e-08,  6.004413e-08,
        1.114551e-07,  1.395406e-07,  7.439562e-08, -2.126215e-07,...

```