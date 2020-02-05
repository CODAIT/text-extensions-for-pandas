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

## Use intervals to extract substrings/subsequences
For intervals that represent offsets into a string, numpy array, or `pd.Series`,
add a vectorized function that takes an `IntervalIndex` and the target object
and returns a new `pd.Series` of substrings or arrays containing the specified
intervals of the target object.

Relevant example code, which extracts the substrings corresponing to token windows:
```python
window_begins = token_begins[:num_tokens - cur_len + 1]
window_ends = token_ends[cur_len - 1:]

window_intervals = pd.Series(pd.arrays.IntervalArray.from_arrays(window_begins, window_ends))
window_substrs = window_intervals.apply(lambda x: doc_text[x.left: x.right])
```

## Vectorized version of `re.fullmatch()`

Pandas has a vectorized version of `re.match()` ("Is there a match at the 
beginning of the string?") but does not have a version of `re.fullmatch()`
("Does the entire string comprise a match?").

Equivalent non-vectorized code:
```python
my_str_series.apply(lambda s: re.fullmatch(my_regex, s) is not None)
```

## Bug in merge

Repro: 

1. Attempt to turn the code of `GraphTraversal.out()` into `GraphTraversal.in_()`:
```python
    def in_(self):
        """
        :returns: A GraphTraversal that adds the destination of any edges into
        the current traversal's last element.
        """
        if self._path_col_types[-1] != "v":
            raise ValueError(
                "Can only call out() when the last element in the path is a "
                "vertex. Last element type is {}".format(
                    self._path_col_types[-1]))

        # Column of path is a list of vertices. Join with edges table.
        p = self.paths
        new_paths = (
            p
            .merge(self.edges, left_on=p.columns[-1], right_on="to")
            .drop("to",
                  axis="columns")  # merge keeps both sides of equijoin
            .rename(columns={
                "from": len(p.columns)}))  # "from" field ==> Last element
        new_path_col_types = self._path_col_types + ["v"]
        return GraphTraversal(self.vertices, self.edges, new_paths,
                              new_path_col_types, self._aliases)
```
2. Attempt to call the `in_()` method:
```python
g = pt.token_features_to_traversal(token_features)
g.V().in_().toList()
```
You will get a strange error message about inputs to assign not being strings.

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
