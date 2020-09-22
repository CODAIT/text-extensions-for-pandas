# Features to Add to Pandas

## Band joins
Allow joining dataframes using join predicates in the form
```sql
R.a - S.b BETWEEN delta_1 AND delta_2
```
or its interval equivalent
```sql
R.a IN S.b -- where S.b is of type INTERVAL
```
Two algorithms possible:
* Index nested loops: Build a B-tree index on the smaller dataframe and probe ranges based on values of the outer tuples.
* Sort merge: Sort both dataframes and walk through the sorted lists, backing up as needed.


## `DataFrame.insert()` doesn't handle anonymous series gracefully
Example:
```python
>>> df = pd.DataFrame([[1,3], [2, 4]])
>>> df
   0  1
0  1  3
1  2  4
>>> df.insert(0, 0, np.array([5, 6]))
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-7-4d659b01127f> in <module>
----> 1 df.insert(0, 0, np.array([5, 6]))

~/opt/miniconda3/envs/pd/lib/python3.7/site-packages/pandas/core/frame.py in insert(self, loc, column, value, allow_duplicates)
   3589         self._ensure_valid_index(value)
   3590         value = self._sanitize_column(column, value, broadcast=False)
-> 3591         self._data.insert(loc, column, value, allow_duplicates=allow_duplicates)
   3592 
   3593     def assign(self, **kwargs):

~/opt/miniconda3/envs/pd/lib/python3.7/site-packages/pandas/core/internals/managers.py in insert(self, loc, item, value, allow_duplicates)
   1171         if not allow_duplicates and item in self.items:
   1172             # Should this be a different kind of error??
-> 1173             raise ValueError("cannot insert {}, already exists".format(item))
   1174 
   1175         if not isinstance(loc, int):

ValueError: cannot insert 0, already exists
```

## Pandas series type for Numpy record arrays

At the moment, Pandas won't let you set a column of a DataFrame to a numpy record array.
```python
>>> repr(rec)
"array([([3, 4): 'maker', 'NOUN', 'maker', [0, 1): 'The', [0, 1): 'The'),\n       ([3, 4): 'maker', 'NOUN', 'maker', [1, 2): 'luxury', [0, 1): 'The'),\n       ([3, 4): 'maker', 'NOUN', 'maker', [2, 3): 'auto', [0, 1): 'The')],\n      dtype=(numpy.record, [('head', 'O'), ('headPOS', 'O'), ('headNF', 'O'), ('child', 'O'), ('determiner', 'O')]))"
>>> df = pd.DataFrame()
>>> df["col"] = rec
---------------------------------------------------------------------------
[...]
~/opt/miniconda3/envs/pd/lib/python3.7/site-packages/pandas/core/frame.py in _ensure_valid_index(self, value)
   3052             except (ValueError, NotImplementedError, TypeError):
   3053                 raise ValueError(
-> 3054                     "Cannot set a frame with no defined index "
   3055                     "and a value that cannot be converted to a "
   3056                     "Series"

ValueError: Cannot set a frame with no defined index and a value that cannot be converted to a Series
```

Pandas *will* allow you to insert a record array with `pd.DataFrame.insert()`, but that behavior is probably a bug:
```python
>>> df = pd.DataFrame()
>>> df[0] = [1, 2, 3]
>>> df.insert(loc=1, column=1, value=rec)
>>> repr(df)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
[...]
~/opt/miniconda3/envs/pd/lib/python3.7/site-packages/pandas/core/dtypes/missing.py in _isna_ndarraylike(obj)
    260         result = values.view("i8") == iNaT
    261     else:
--> 262         result = np.isnan(values)
    263 
    264     # box

TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
```

Pandas really ought to have series type that wraps Numpy record arrays and allows access via a custom namespace; something like:
```python
>>> df = pd.DataFrame()
>>> df["col"] = pd.RecordArray(rec)
>>> df["col"].field["head"]
array([[3, 4): 'maker', [3, 4): 'maker', [3, 4): 'maker'], dtype=object)
```
