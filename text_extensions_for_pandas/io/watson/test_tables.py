#
#  Copyright (c) 2020 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
import os
import textwrap
import unittest

from text_extensions_for_pandas.io.watson.tables import *


class TestTables(unittest.TestCase):

    def setUp(self):
        with open("test_data/io/test_watson_tables/archive.json", 'r') as archive_file:
            self.responses_dict = json.load(archive_file)

        self.maxDiff = None
        pd.set_option("display.max_columns", 100)  # dont have things wrapping

    def tearDown(self):
        pd.reset_option("display.max_columns")

    def test_parse_response(self):
        response = self.responses_dict["double_header_table"]
        parsed = parse_response(response)
        self.assertEqual(len(parsed), 4)
        self.assertSequenceEqual(sorted(parsed.keys()), sorted(["body_cells", "col_headers", "row_headers", "given_loc"]))
        self.assertEqual(repr(parsed),
                         """\
{'row_headers':                    text  column_index_begin  column_index_end  \\
0    Statatory tax rate                   0                 0   
1  IRS audit settlement                   0                 0   
2    Dividends received                   0                 0   
3        Total tax rate                   0                 0   

   row_index_begin  row_index_end              cell_id       text_normalized  
0                2              2  rowHeader-2810-2829    Statatory tax rate  
1                3              3  rowHeader-4068-4089  IRS audit settlement  
2                4              4  rowHeader-5329-5348    Dividends received  
3                5              5  rowHeader-6586-6601        Total tax rate  , 'col_headers':                                text  column_index_begin  column_index_end  \\
0                                                     0                 0   
1  Three months ended setptember 30                   1                 2   
2   Nine months ended setptember 30                   3                 4   
3                                                     0                 0   
4                              2005                   1                 1   
5                              2004                   2                 2   
6                              2005                   3                 3   
7                              2004                   4                 4   

   row_index_begin  row_index_end              cell_id  \\
0                0              0    colHeader-786-787   
1                0              0  colHeader-1012-1206   
2                0              0  colHeader-1444-1514   
3                1              1  colHeader-1586-1587   
4                1              1  colHeader-1813-1818   
5                1              1  colHeader-2061-2066   
6                1              1  colHeader-2305-2310   
7                1              1  colHeader-2553-2558   

                    text_normalized  
0                                    
1  Three months ended setptember 30  
2   Nine months ended setptember 30  
3                                    
4                              2005  
5                              2004  
6                              2005  
7                              2004  , 'body_cells':      text  column_index_begin  column_index_end  row_index_begin  \\
0     35%                   1                 1                2   
1     36%                   2                 2                2   
2     37%                   3                 3                2   
3     38%                   4                 4                2   
4     97%                   1                 1                3   
5   35.5%                   2                 2                3   
6     58%                   3                 3                3   
7   15.2%                   4                 4                3   
8   13.2%                   1                 1                4   
9    3.3%                   2                 2                4   
10  15.4%                   3                 3                4   
11   4.7%                   4                 4                4   
12  76.1%                   1                 1                5   
13   4.3%                   2                 2                5   
14  38.8%                   3                 3                5   
15  15.1%                   4                 4                5   

    row_index_end             cell_id  \\
0               2  bodyCell-3073-3077   
1               2  bodyCell-3320-3324   
2               2  bodyCell-3564-3568   
3               2  bodyCell-3811-3815   
4               3  bodyCell-4333-4337   
5               3  bodyCell-4579-4585   
6               3  bodyCell-4825-4829   
7               3  bodyCell-5071-5077   
8               4  bodyCell-5591-5597   
9               4  bodyCell-5838-5843   
10              4  bodyCell-6082-6088   
11              4  bodyCell-6329-6334   
12              5  bodyCell-6844-6850   
13              5  bodyCell-7091-7096   
14              5  bodyCell-7335-7341   
15              5  bodyCell-7583-7589   

                             column_header_ids  \\
0   [colHeader-1012-1206, colHeader-1813-1818]   
1   [colHeader-1012-1206, colHeader-2061-2066]   
2   [colHeader-1444-1514, colHeader-2305-2310]   
3   [colHeader-1444-1514, colHeader-2553-2558]   
4   [colHeader-1012-1206, colHeader-1813-1818]   
5   [colHeader-1012-1206, colHeader-2061-2066]   
6   [colHeader-1444-1514, colHeader-2305-2310]   
7   [colHeader-1444-1514, colHeader-2553-2558]   
8   [colHeader-1012-1206, colHeader-1813-1818]   
9   [colHeader-1012-1206, colHeader-2061-2066]   
10  [colHeader-1444-1514, colHeader-2305-2310]   
11  [colHeader-1444-1514, colHeader-2553-2558]   
12  [colHeader-1012-1206, colHeader-1813-1818]   
13  [colHeader-1012-1206, colHeader-2061-2066]   
14  [colHeader-1444-1514, colHeader-2305-2310]   
15  [colHeader-1444-1514, colHeader-2553-2558]   

                         column_header_texts         row_header_ids  \\
0   [Three months ended setptember 30, 2005]  [rowHeader-2810-2829]   
1   [Three months ended setptember 30, 2004]  [rowHeader-2810-2829]   
2    [Nine months ended setptember 30, 2005]  [rowHeader-2810-2829]   
3    [Nine months ended setptember 30, 2004]  [rowHeader-2810-2829]   
4   [Three months ended setptember 30, 2005]  [rowHeader-4068-4089]   
5   [Three months ended setptember 30, 2004]  [rowHeader-4068-4089]   
6    [Nine months ended setptember 30, 2005]  [rowHeader-4068-4089]   
7    [Nine months ended setptember 30, 2004]  [rowHeader-4068-4089]   
8   [Three months ended setptember 30, 2005]  [rowHeader-5329-5348]   
9   [Three months ended setptember 30, 2004]  [rowHeader-5329-5348]   
10   [Nine months ended setptember 30, 2005]  [rowHeader-5329-5348]   
11   [Nine months ended setptember 30, 2004]  [rowHeader-5329-5348]   
12  [Three months ended setptember 30, 2005]  [rowHeader-6586-6601]   
13  [Three months ended setptember 30, 2004]  [rowHeader-6586-6601]   
14   [Nine months ended setptember 30, 2005]  [rowHeader-6586-6601]   
15   [Nine months ended setptember 30, 2004]  [rowHeader-6586-6601]   

          row_header_texts attributes.text attributes.type  
0     [Statatory tax rate]           [35%]    [Percentage]  
1     [Statatory tax rate]           [36%]    [Percentage]  
2     [Statatory tax rate]           [37%]    [Percentage]  
3     [Statatory tax rate]           [38%]    [Percentage]  
4   [IRS audit settlement]           [97%]    [Percentage]  
5   [IRS audit settlement]         [35.5%]    [Percentage]  
6   [IRS audit settlement]           [58%]    [Percentage]  
7   [IRS audit settlement]         [15.2%]    [Percentage]  
8     [Dividends received]         [13.2%]    [Percentage]  
9     [Dividends received]          [3.3%]    [Percentage]  
10    [Dividends received]         [15.4%]    [Percentage]  
11    [Dividends received]          [4.7%]    [Percentage]  
12        [Total tax rate]         [76.1%]    [Percentage]  
13        [Total tax rate]          [4.3%]    [Percentage]  
14        [Total tax rate]         [38.8%]    [Percentage]  
15        [Total tax rate]         [15.1%]    [Percentage]  , 'given_loc': {'begin': 786, 'end': 7589}}"""
                         )
        parsed_2 = parse_response(self.responses_dict["20-populous-countries"])
        self.assertEqual(len(parsed_2), 4)
        self.assertSequenceEqual(sorted(parsed_2.keys()), sorted(['body_cells', 'col_headers', 'row_headers',"given_loc"]))
        self.assertEqual(repr(parsed_2),
                         """\
{'row_headers': None, 'col_headers':                                  text  column_index_begin  column_index_end  \\
0                                Rank                   0                 0   
1  Country (or\\ndependent\\nterritory)                   1                 1   
2                          Population                   2                 2   
3                % of worldpopulation                   3                 3   
4                                Date                   4                 4   
5                              Source                   5                 5   

   row_index_begin  row_index_end              cell_id  \\
0                0              0  colHeader-1611-1616   
1                0              0  colHeader-1859-2250   
2                0              0  colHeader-2504-2515   
3                0              0  colHeader-2758-2779   
4                0              0  colHeader-3034-3039   
5                0              0  colHeader-3284-3291   

                      text_normalized  
0                                Rank  
1  Country (or\\ndependent\\nterritory)  
2                          Population  
3                % of worldpopulation  
4                                Date  
5                              Source  , 'body_cells':                     text  column_index_begin  column_index_end  \\
0                      1                   0                 0   
1              China [b]                   1                 1   
2          1,403,627,360                   2                 2   
3                  18.0%                   3                 3   
4            21 Jul 2020                   4                 4   
..                   ...                 ...               ...   
121                World                   1                 1   
122        7,800,767,000                   2                 2   
123                 100%                   3                 3   
124          21 Jul 2020                   4                 4   
125  UN Projection [199]                   5                 5   

     row_index_begin  row_index_end               cell_id  \\
0                  1              1    bodyCell-3552-3554   
1                  1              1    bodyCell-3799-3975   
2                  1              1    bodyCell-4226-4240   
3                  1              1    bodyCell-4483-4489   
4                  1              1    bodyCell-4734-4746   
..               ...            ...                   ...   
121               21             21  bodyCell-39162-39169   
122               21             21  bodyCell-39425-39439   
123               21             21  bodyCell-39694-39699   
124               21             21  bodyCell-39954-39966   
125               21             21  bodyCell-40221-40405   

         column_header_ids                   column_header_texts  \\
0    [colHeader-1611-1616]                                [Rank]   
1    [colHeader-1859-2250]  [Country (or\\ndependent\\nterritory)]   
2    [colHeader-2504-2515]                          [Population]   
3    [colHeader-2758-2779]                [% of worldpopulation]   
4    [colHeader-3034-3039]                                [Date]   
..                     ...                                   ...   
121  [colHeader-1859-2250]  [Country (or\\ndependent\\nterritory)]   
122  [colHeader-2504-2515]                          [Population]   
123  [colHeader-2758-2779]                [% of worldpopulation]   
124  [colHeader-3034-3039]                                [Date]   
125  [colHeader-3284-3291]                              [Source]   

    row_header_ids row_header_texts  attributes.text attributes.type  
0               []               []              [1]        [Number]  
1               []               []          [China]      [Location]  
2               []               []  [1,403,627,360]        [Number]  
3               []               []          [18.0%]    [Percentage]  
4               []               []    [21 Jul 2020]      [DateTime]  
..             ...              ...              ...             ...  
121             []               []               []              []  
122             []               []  [7,800,767,000]        [Number]  
123             []               []           [100%]    [Percentage]  
124             []               []    [21 Jul 2020]      [DateTime]  
125             []               []             [UN]  [Organization]  

[126 rows x 12 columns], 'given_loc': {'begin': 1611, 'end': 40405}}\
""")

    def test_make_exploded_df(self):
        double_header = parse_response(self.responses_dict["double_header_table"])
        countries = parse_response(self.responses_dict["20-populous-countries"])

        countries_exp = make_exploded_df(countries, keep_all_cols=True)
        self.assertSequenceEqual(countries_exp[1], ['row_index'])
        self.assertSequenceEqual(countries_exp[2], ['column_header_texts_0'])
        self.assertEqual(repr(countries_exp[0]),
                         """\
                    text  column_index_begin  column_index_end  row_index_end  \\
0                      1                   0                 0              1   
1              China [b]                   1                 1              1   
2          1,403,627,360                   2                 2              1   
3                  18.0%                   3                 3              1   
4            21 Jul 2020                   4                 4              1   
..                   ...                 ...               ...            ...   
121                World                   1                 1             21   
122        7,800,767,000                   2                 2             21   
123                 100%                   3                 3             21   
124          21 Jul 2020                   4                 4             21   
125  UN Projection [199]                   5                 5             21   

                  cell_id      column_header_ids row_header_ids  \\
0      bodyCell-3552-3554  [colHeader-1611-1616]             []   
1      bodyCell-3799-3975  [colHeader-1859-2250]             []   
2      bodyCell-4226-4240  [colHeader-2504-2515]             []   
3      bodyCell-4483-4489  [colHeader-2758-2779]             []   
4      bodyCell-4734-4746  [colHeader-3034-3039]             []   
..                    ...                    ...            ...   
121  bodyCell-39162-39169  [colHeader-1859-2250]             []   
122  bodyCell-39425-39439  [colHeader-2504-2515]             []   
123  bodyCell-39694-39699  [colHeader-2758-2779]             []   
124  bodyCell-39954-39966  [colHeader-3034-3039]             []   
125  bodyCell-40221-40405  [colHeader-3284-3291]             []   

    row_header_texts  attributes.text attributes.type  \\
0                 []              [1]        [Number]   
1                 []          [China]      [Location]   
2                 []  [1,403,627,360]        [Number]   
3                 []          [18.0%]    [Percentage]   
4                 []    [21 Jul 2020]      [DateTime]   
..               ...              ...             ...   
121               []               []              []   
122               []  [7,800,767,000]        [Number]   
123               []           [100%]    [Percentage]   
124               []    [21 Jul 2020]      [DateTime]   
125               []             [UN]  [Organization]   

                  column_header_texts_0 row_index  
0                                  Rank         1  
1    Country (or\\ndependent\\nterritory)         1  
2                            Population         1  
3                  % of worldpopulation         1  
4                                  Date         1  
..                                  ...       ...  
121  Country (or\\ndependent\\nterritory)        21  
122                          Population        21  
123                % of worldpopulation        21  
124                                Date        21  
125                              Source        21  

[126 rows x 12 columns]\
""")

        double_header_exp = make_exploded_df(double_header, keep_all_cols=True, drop_original=False)
        self.assertSequenceEqual(double_header_exp[1], ['row_header_texts_0'])
        self.assertSequenceEqual(double_header_exp[2], ['column_header_texts_0', 'column_header_texts_1'])
        self.assertEqual(repr(double_header_exp[0]), """\
     text  column_index_begin  column_index_end  row_index_begin  \\
0     35%                   1                 1                2   
1     36%                   2                 2                2   
2     37%                   3                 3                2   
3     38%                   4                 4                2   
4     97%                   1                 1                3   
5   35.5%                   2                 2                3   
6     58%                   3                 3                3   
7   15.2%                   4                 4                3   
8   13.2%                   1                 1                4   
9    3.3%                   2                 2                4   
10  15.4%                   3                 3                4   
11   4.7%                   4                 4                4   
12  76.1%                   1                 1                5   
13   4.3%                   2                 2                5   
14  38.8%                   3                 3                5   
15  15.1%                   4                 4                5   

    row_index_end             cell_id  \\
0               2  bodyCell-3073-3077   
1               2  bodyCell-3320-3324   
2               2  bodyCell-3564-3568   
3               2  bodyCell-3811-3815   
4               3  bodyCell-4333-4337   
5               3  bodyCell-4579-4585   
6               3  bodyCell-4825-4829   
7               3  bodyCell-5071-5077   
8               4  bodyCell-5591-5597   
9               4  bodyCell-5838-5843   
10              4  bodyCell-6082-6088   
11              4  bodyCell-6329-6334   
12              5  bodyCell-6844-6850   
13              5  bodyCell-7091-7096   
14              5  bodyCell-7335-7341   
15              5  bodyCell-7583-7589   

                             column_header_ids  \\
0   [colHeader-1012-1206, colHeader-1813-1818]   
1   [colHeader-1012-1206, colHeader-2061-2066]   
2   [colHeader-1444-1514, colHeader-2305-2310]   
3   [colHeader-1444-1514, colHeader-2553-2558]   
4   [colHeader-1012-1206, colHeader-1813-1818]   
5   [colHeader-1012-1206, colHeader-2061-2066]   
6   [colHeader-1444-1514, colHeader-2305-2310]   
7   [colHeader-1444-1514, colHeader-2553-2558]   
8   [colHeader-1012-1206, colHeader-1813-1818]   
9   [colHeader-1012-1206, colHeader-2061-2066]   
10  [colHeader-1444-1514, colHeader-2305-2310]   
11  [colHeader-1444-1514, colHeader-2553-2558]   
12  [colHeader-1012-1206, colHeader-1813-1818]   
13  [colHeader-1012-1206, colHeader-2061-2066]   
14  [colHeader-1444-1514, colHeader-2305-2310]   
15  [colHeader-1444-1514, colHeader-2553-2558]   

                         column_header_texts         row_header_ids  \\
0   [Three months ended setptember 30, 2005]  [rowHeader-2810-2829]   
1   [Three months ended setptember 30, 2004]  [rowHeader-2810-2829]   
2    [Nine months ended setptember 30, 2005]  [rowHeader-2810-2829]   
3    [Nine months ended setptember 30, 2004]  [rowHeader-2810-2829]   
4   [Three months ended setptember 30, 2005]  [rowHeader-4068-4089]   
5   [Three months ended setptember 30, 2004]  [rowHeader-4068-4089]   
6    [Nine months ended setptember 30, 2005]  [rowHeader-4068-4089]   
7    [Nine months ended setptember 30, 2004]  [rowHeader-4068-4089]   
8   [Three months ended setptember 30, 2005]  [rowHeader-5329-5348]   
9   [Three months ended setptember 30, 2004]  [rowHeader-5329-5348]   
10   [Nine months ended setptember 30, 2005]  [rowHeader-5329-5348]   
11   [Nine months ended setptember 30, 2004]  [rowHeader-5329-5348]   
12  [Three months ended setptember 30, 2005]  [rowHeader-6586-6601]   
13  [Three months ended setptember 30, 2004]  [rowHeader-6586-6601]   
14   [Nine months ended setptember 30, 2005]  [rowHeader-6586-6601]   
15   [Nine months ended setptember 30, 2004]  [rowHeader-6586-6601]   

          row_header_texts attributes.text attributes.type  \\
0     [Statatory tax rate]           [35%]    [Percentage]   
1     [Statatory tax rate]           [36%]    [Percentage]   
2     [Statatory tax rate]           [37%]    [Percentage]   
3     [Statatory tax rate]           [38%]    [Percentage]   
4   [IRS audit settlement]           [97%]    [Percentage]   
5   [IRS audit settlement]         [35.5%]    [Percentage]   
6   [IRS audit settlement]           [58%]    [Percentage]   
7   [IRS audit settlement]         [15.2%]    [Percentage]   
8     [Dividends received]         [13.2%]    [Percentage]   
9     [Dividends received]          [3.3%]    [Percentage]   
10    [Dividends received]         [15.4%]    [Percentage]   
11    [Dividends received]          [4.7%]    [Percentage]   
12        [Total tax rate]         [76.1%]    [Percentage]   
13        [Total tax rate]          [4.3%]    [Percentage]   
14        [Total tax rate]         [38.8%]    [Percentage]   
15        [Total tax rate]         [15.1%]    [Percentage]   

               column_header_texts_0 column_header_texts_1  \\
0   Three months ended setptember 30                  2005   
1   Three months ended setptember 30                  2004   
2    Nine months ended setptember 30                  2005   
3    Nine months ended setptember 30                  2004   
4   Three months ended setptember 30                  2005   
5   Three months ended setptember 30                  2004   
6    Nine months ended setptember 30                  2005   
7    Nine months ended setptember 30                  2004   
8   Three months ended setptember 30                  2005   
9   Three months ended setptember 30                  2004   
10   Nine months ended setptember 30                  2005   
11   Nine months ended setptember 30                  2004   
12  Three months ended setptember 30                  2005   
13  Three months ended setptember 30                  2004   
14   Nine months ended setptember 30                  2005   
15   Nine months ended setptember 30                  2004   

      row_header_texts_0  
0     Statatory tax rate  
1     Statatory tax rate  
2     Statatory tax rate  
3     Statatory tax rate  
4   IRS audit settlement  
5   IRS audit settlement  
6   IRS audit settlement  
7   IRS audit settlement  
8     Dividends received  
9     Dividends received  
10    Dividends received  
11    Dividends received  
12        Total tax rate  
13        Total tax rate  
14        Total tax rate  
15        Total tax rate  \
""")

    def test_make_table(self):
        double_header_table = make_table(parse_response(self.responses_dict["double_header_table"]))
        self.assertEqual(repr(double_header_table), """\
                     Three months ended setptember 30        \\
                                                 2005  2004   
Statatory tax rate                               35.0  36.0   
IRS audit settlement                             97.0  35.5   
Dividends received                               13.2   3.3   
Total tax rate                                   76.1   4.3   

                     Nine months ended setptember 30        
                                                2005  2004  
Statatory tax rate                              37.0  38.0  
IRS audit settlement                            58.0  15.2  
Dividends received                              15.4   4.7  
Total tax rate                                  38.8  15.1  \
""")

        countries_table = make_table(parse_response(self.responses_dict["20-populous-countries"]))
        self.assertEqual(repr(countries_table), """\
    Rank Country (or\\ndependent\\nterritory)    Population  \\
1      1                          China [b]  1.403627e+09   
2      2                          India [c]  1.364965e+09   
3      3                  United States [d]  3.299913e+08   
4      4                          Indonesia  2.696034e+08   
5      5                       Pakistan [e]  2.208923e+08   
6      6                             Brazil  2.118221e+08   
7      7                            Nigeria  2.061396e+08   
8      8                         Bangladesh  1.689908e+08   
9      9                         Russia [f]  1.467486e+08   
10    10                             Mexico  1.277923e+08   
11    11                              Japan  1.259300e+08   
12    12                        Philippines  1.089213e+08   
13    13                              Egypt  1.006445e+08   
14    14                           Ethiopia  9.866500e+07   
15    15                            Vietnam  9.620898e+07   
16    16                           DR Congo  8.956140e+07   
17    17                               Iran  8.363117e+07   
18    18                            Germany  8.316671e+07   
19    19                             Turkey  8.315500e+07   
20    20                         France [g]  6.708100e+07   
21  <NA>                              World  7.800767e+09   

    % of worldpopulation         Date                            Source  
1                  18.00  21 Jul 2020      National populationclock [3]  
2                  17.50  21 Jul 2020      National populationclock [4]  
3                   4.23  21 Jul 2020    National population\\nclock [5]  
4                   3.46   1 Jul 2020     National annualprojection [6]  
5                   2.83   1 Jul 2020                 UN Projection [2]  
6                   2.72  21 Jul 2020      National populationclock [7]  
7                   2.64   1 Jul 2020                 UN Projection [2]  
8                   2.17  21 Jul 2020      National populationclock [8]  
9                   1.88   1 Jan 2020             National estimate [9]  
10                  1.64   1 Jul 2020    National annualprojection [10]  
11                  1.61   1 Jun 2020  Monthly provisionalestimate [11]  
12                  1.40  21 Jul 2020     National populationclock [12]  
13                  1.29  21 Jul 2020     National populationclock [13]  
14                  1.26   1 Jul 2019    National annualprojection [14]  
15                  1.23   1 Apr 2019            2019 censusresult [15]  
16                  1.15   1 Jul 2020                 UN Projection [2]  
17                  1.07  21 Jul 2020     National populationclock [16]  
18                  1.07  31 Dec 2019            National estimate [17]  
19                  1.07  31 Dec 2019      National annualestimate [18]  
20                  0.86   1 Jun 2020     Monthly nationalestimate [19]  
21                100.00  21 Jul 2020               UN Projection [199]  \
""")





