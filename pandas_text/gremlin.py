#
# gremlin.py
#
# Part of pandas_text
#
# Code for running Gremlin queries against parse trees stored as DataFrames.
#

import numpy as np
import pandas as pd
from typing import *


class GraphTraversal:
    """
    Class that represents a [subset of] Gremlin graph traversal.

    For ease of implementation, the traversal is computed eagerly, and there is
    no attempt at query optimization. This may change in the future.
    """

    def __init__(self, vertices: pd.DataFrame, edges: pd.DataFrame,
                 paths: pd.DataFrame, path_col_types: List[str],
                 aliases: Dict[str, int]):
        """
        **DO NOT CALL THIS CONSTRUCTOR DIRECTLY.** Use factory methods.

        :param vertices: DataFrame of vertices that make up the graph

        :param edges: DataFrame of edges in the graph, where the "from" and "to"
        columns are references to the index of `vertices`.

        :param paths: DataFrame of paths that this traversal represents. Each
        row represents a path.

        :param path_col_types: Array of additional information about the columns
        of `paths`. Each value may be None (raw Pandas type), "v" (vertex
        reference), or "e" (edge reference)

        :param aliases: The current set of aliases for path elements, as a map
        from alias name to integer index.
        """
        self._vertices = vertices
        self._edges = edges
        self._paths = paths
        self._path_col_types = path_col_types
        self._aliases = aliases

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges

    @property
    def paths(self):
        return self._paths

    def toList(self):
        """
        :returns the rightmost element of the traversal as a Python list
        """
        col_type = self._path_col_types[-1]
        last_elems = self.paths[self.paths.columns[-1]].tolist()
        if col_type == "v":
            return ["v[{}]".format(elem) for elem in last_elems]
        elif col_type == "e":
            return ["e[{}]".format(elem) for elem in last_elems]
        else:
            return last_elems

    def V(self):
        """
        :returns a GraphTraversal that starts a new traversal with all
        vertices of the graph.
        """
        new_paths = pd.DataFrame({0: self.vertices.index})
        return GraphTraversal(self.vertices, self.edges, new_paths, ["v"], {})

    def has(self, key: str, value: Any):
        """
        :param key: Key to look for in the properties of the most recent
        vertex or edge

        :param value: Expected value the indicated field in each vertex.
        Can be either a literal value (expected value of a field) or a

        :returns: A GraphTraversal that filters out paths whose last element
        does not contain the indicated value in the indicated field.
        """
        if len(self.paths.columns) == 0:
            raise ValueError("Cannot call has() on an empty path.")
        if not isinstance(value, ColumnPredicate):
            # Not a predicate ==> assume literal value
            pred = Within(value)
        else:
            pred = value
        # Join current path back with vertices table
        vertices_to_check = self.vertices.loc[
            self.paths[self.paths.columns[-1]]]
        if key not in vertices_to_check.columns:
            # Column not present ==> empty result
            filtered_paths = self.paths.iloc[0:0]
        else:
            mask = pred.apply(vertices_to_check[key])
            filtered_paths = self.paths[mask]
        return GraphTraversal(self.vertices, self.edges, filtered_paths,
                              self._path_col_types, self._aliases)

    def as_(self, *names: str):
        """
        :param names: Alias[es] for the current element in the traversal

        :returns: A GraphTraversal that marks the current (rightmost) element
        of this traversal with the indicated short name[s].
        """
        new_aliases = self._aliases.copy()
        for name in names:
            new_aliases[name] = len(self.paths.columns) - 1
        return GraphTraversal(self.vertices, self.edges, self.paths,
                              self._path_col_types, new_aliases)

    def out(self):
        """
        :returns: A GraphTraversal that adds the destination of any edges out
        of the current traversal's last elemehasnt.
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
            .merge(self.edges, left_on=p.columns[-1], right_on="from")
            .drop("from",
                  axis="columns")  # merge keeps both sides of equijoin
            .rename(columns={
                "to": len(p.columns)}))  # "to" field ==> Last element
        new_path_col_types = self._path_col_types + ["v"]
        return GraphTraversal(self.vertices, self.edges, new_paths,
                              new_path_col_types, self._aliases)

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
        merge_tmp = p.copy()
        # Pandas doesn't like integer join keys for merge
        merge_tmp["join_key"] = merge_tmp[merge_tmp.columns[-1]]
        new_paths = (
            merge_tmp
            .merge(self.edges, left_on="join_key", right_on="to")
            .drop(["to", "join_key"], axis="columns")
            .rename(columns={"from": len(p.columns)})
        )
        new_path_col_types = self._path_col_types + ["v"]
        return GraphTraversal(self.vertices, self.edges, new_paths,
                              new_path_col_types, self._aliases)

    def select(self, *args):
        """
        :param args: List of 1 or more strings to select

        :returns: A GraphTraversal at the beginning of a `select` operation.
        """
        if 0 == len(args):
            raise ValueError(
                "No arguments passed to select(). Must select at least 1 "
                "alias from the traversal.") 
        return SelectGraphTraversal(self.vertices, self.edges, self.paths,
                                    self._path_col_types, self._aliases,
                                    list(args))


class DoubleUnderscore:
    """
    Standin for Gremlin's `__` (two underscore characters) operation, which
    means "anonymous traversal starting at the end of the current travesal"
    """
    pass


# Alias to allow "pt.__" in Gremlin expressions
__ = DoubleUnderscore()


class ColumnPredicate:
    """
    Base class for Boolean predicates applied to fields of vertices of the
    graph.
    """
    def apply(self, values: pd.Series) -> np.ndarray:
        """
        :param values: Vertex fields on which to apply the predicate
        :return: A numpy Boolean mask containing `True` in each
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Within(ColumnPredicate):
    """
    Implementation of the Gremlin `within()` predicate
    """
    def __init__(self, *args: Any):
        """
        :param args: 1 or more arguments to the predicate as Python varargs.
        Currently the class passes these objects through to Pandas' comparison
        functions, but future versions may perform some additional validation
        here.
        """
        self._args = args

    def apply(self, vertices: pd.DataFrame) -> np.ndarray:
        return vertices.isin(self._args).values


class Without(ColumnPredicate):
    """
    Implementation of the Gremlin `without()` predicate
    """
    def __init__(self, *args: Any):
        """
        :param args: 1 or more arguments to the predicate as Python varargs.
        Currently the class passes these objects through to Pandas' comparison
        functions, but future versions may perform some additional validation
        here.
        """
        self._args = args

    def apply(self, vertices: pd.DataFrame) -> np.ndarray:
        return (~vertices.isin(self._args)).values


class SelectGraphTraversal(GraphTraversal):
    """
    A traversal that ends in a `select`, possibly followed by a sequence of 
    `by` elements. 
    """

    def __init__(self, vertices: pd.DataFrame, edges: pd.DataFrame,
                 paths: pd.DataFrame, path_col_types: List[str],
                 aliases: Dict[str, int], selected_aliases: List[str],
                 by_list: List[str] = []):
        """
        :param selected_aliases: Which entries of `aliases` are mentioned in 
        the `select` statement. 
        """
        GraphTraversal.__init__(self, vertices, edges, paths, path_col_types,
                                aliases)
        self._selected_aliases = selected_aliases
        self._by_list = by_list

    def by(self, field_name: str):
        """
        :field_name: Next argument in the list of field name arguments to select
        """
        return SelectGraphTraversal(self.vertices, self.edges, self.paths,
                                    self._path_col_types,
                                    self._aliases, self._selected_aliases,
                                    self._by_list + [field_name])

    def toList(self):
        """
        Overload the superclass's implementation to generate the results of 
        the `select`. 
        """
        return self.toDataFrame().to_dict("records")

    def toDataFrame(self):
        """
        :return: A DataFrame with the schema of the `select` statement.
        Column names will be the aliases referenced in the `select()` element,
        and values will be drawn from the vertex/edge columns named in the
        `by` element.
        """
        # Schema of a Gremlin select is: (key1, value1), (key2, value2), ...
        # where key1, key2, ... are the aliases and value1, value2, ... are the
        # extracted field values.
        if 0 == len(self._by_list):
            # TODO: Implement this branch
            raise NotImplementedError("select without by not yet implemented")
        df_contents = {}
        for i in range(len(self._selected_aliases)):
            alias = self._selected_aliases[i]
            # The Gremlin select statement rotates through the by list if there
            # are more aliases than elements of the by list.
            by = self._by_list[i % len(self._by_list)]
            selected_index = self._aliases[alias]  # index into traversal
            col_type = self._path_col_types[selected_index]
            elem = self.paths[selected_index]
            if "v" == col_type:
                # Vertex element.
                if by not in self.vertices.columns:
                    raise ValueError(
                        "Column '{}' not present in vertices of '{}'"
                        "".format(by, alias))
                df_contents[alias] = self.vertices[by].loc[elem].values
            else:
                # TODO: Implement this branch
                raise NotImplementedError(
                    "select on non-vertex element not implemented")
        return pd.DataFrame(df_contents)


def token_features_to_traversal(token_features: pd.DataFrame):
    """
    Turn a DataFrame of token features in the form returned by
    `make_tokens_and_features` into an empty graph traversal.
    Similar to calling `graph.traversal()` in Gremlin.

    :param token_features: DataFrame containing information about individual
    tokens. Must contain a `head_token_num` column that is synchronized with
    the DataFrame's index.

    :returns: A traversal containing a graph version of `token_features` and
    an empty set of paths.
    """
    # Don't include token IDs in the vertex attributes
    vertices = token_features.drop(["token_num", "head_token_num"], axis=1)
    edges = pd.DataFrame(
        {"from": token_features.index, "to": token_features["head_token_num"]})
    paths = pd.DataFrame()
    path_col_types = []
    aliases = {}
    return GraphTraversal(vertices, edges, paths, path_col_types, aliases)
