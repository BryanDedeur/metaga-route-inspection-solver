# Truss Bridge Problem Instances
 
This is a set of 40 undirected graphs all representing realistic truss bridge structures. There are 20 small and 20 large instances of 4 different truss structures, howe, ktruss, pratt, warren. Foreach instance there are two representations, a .csv adjacency matrix and .obj vertex coordinates (can be imported into 3d modeling software).

| Id | Problem Instance | Num Edges | Num Vertices |
|----|------------------|-----------|--------------|
| 1  | howe1            | 37        | 18           |
| 2  | howe2            | 63        | 28           |
| 3  | howe3            | 89        | 38           |
| 4  | howe4            | 115       | 48           |
| 5  | howe5            | 141       | 58           |
| 6  | ktruss1          | 45        | 22           |
| 7  | ktruss2          | 79        | 36           |
| 8  | ktruss3          | 113       | 50           |
| 9  | ktruss4          | 147       | 64           |
| 10 | ktruss5          | 181       | 78           |
| 11 | pratt1           | 37        | 18           |
| 12 | pratt2           | 63        | 28           |
| 13 | pratt3           | 89        | 38           |
| 14 | pratt4           | 115       | 48           |
| 15 | pratt5           | 141       | 58           |
| 16 | warren1          | 59        | 26           |
| 17 | warren2          | 85        | 36           |
| 18 | warren3          | 111       | 46           |
| 19 | warren4          | 137       | 56           |
| 20 | warren5          | 163       | 66           |

## Large bridges

| Problem Instance | Num Edges | Num Vertices |
| ---------------- | --------- | ------------ |
| howe-l1          | 245       | 98           |
| howe-l2          | 505       | 198          |
| howe-l3          | 765       | 298          |
| howe-l4          | 999       | 388          |
| pratt-l1         | 245       | 98           |
| pratt-l2         | 505       | 198          |
| pratt-l3         | 765       | 298          |
| pratt-l4         | 999       | 388          |
| ktruss-l1        | 249       | 106          |
| ktruss-l2        | 487       | 204          |
| ktruss-l3        | 759       | 316          |
| ktruss-l4        | 997       | 414          |
| warren-l1        | 241       | 96           |
| warren-l2        | 501       | 196          |
| warren-l3        | 761       | 296          |
| warren-l4        | 995       | 386          |

To add to your project I recommend [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

## Reading .csv files

Each instance is stored in a .csv adjacency matrix format. 

Here is a formatted example of pratt1.csv:

The first row and the first column represents the vertex ids. You can determine connecting edges if the value in the row and column is greater than 0.

|      | v_0      | v_1      | v_2      | v_3      | v_4 | v_5 | v_6 | v_7 | v_8      | v_9      | v_10     | v_11     | v_12     | v_13     | v_14     | v_15     | v_16     | v_17     |
|------|----------|----------|----------|----------|-----|-----|-----|-----|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| v_0  | 0        | 0        | 6.5      | 0        | 0   | 4   | 4   | 0   | 0        | 7.632169 | 7.632169 | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| v_1  | 0        | 0        | 0        | 6.5      | 4   | 0   | 0   | 4   | 7.632169 | 0        | 0        | 7.632169 | 0        | 0        | 0        | 0        | 0        | 0        |
| v_2  | 6.5      | 0        | 0        | 6.4384   | 0   | 0   | 0   | 0   | 0        | 4        | 4        | 0        | 3.789888 | 3.789888 | 0        | 0        | 0        | 0        |
| v_3  | 0        | 6.5      | 6.4384   | 0        | 0   | 0   | 0   | 0   | 4        | 0        | 0        | 4        | 3.789888 | 3.789888 | 0        | 0        | 0        | 0        |
| v_4  | 0        | 4        | 0        | 0        | 0   | 0   | 0   | 0   | 6.5      | 0        | 0        | 0        | 0        | 0        | 4        | 0        | 0        | 0        |
| v_5  | 4        | 0        | 0        | 0        | 0   | 0   | 0   | 0   | 0        | 6.5      | 0        | 0        | 0        | 0        | 0        | 4        | 0        | 0        |
| v_6  | 4        | 0        | 0        | 0        | 0   | 0   | 0   | 0   | 0        | 0        | 6.5      | 0        | 0        | 0        | 0        | 0        | 4        | 0        |
| v_7  | 0        | 4        | 0        | 0        | 0   | 0   | 0   | 0   | 0        | 0        | 0        | 6.5      | 0        | 0        | 0        | 0        | 0        | 4        |
| v_8  | 0        | 7.632169 | 0        | 4        | 6.5 | 0   | 0   | 0   | 0        | 0        | 6.4384   | 0        | 3.789888 | 0        | 7.632169 | 0        | 0        | 0        |
| v_9  | 7.632169 | 0        | 4        | 0        | 0   | 6.5 | 0   | 0   | 0        | 0        | 0        | 6.4384   | 0        | 3.789888 | 0        | 7.632169 | 0        | 0        |
| v_10 | 7.632169 | 0        | 4        | 0        | 0   | 0   | 6.5 | 0   | 6.4384   | 0        | 0        | 0        | 3.789888 | 0        | 0        | 0        | 7.632169 | 0        |
| v_11 | 0        | 7.632169 | 0        | 4        | 0   | 0   | 0   | 6.5 | 0        | 6.4384   | 0        | 0        | 0        | 3.789888 | 0        | 0        | 0        | 7.632169 |
| v_12 | 0        | 0        | 3.789888 | 3.789888 | 0   | 0   | 0   | 0   | 3.789888 | 0        | 3.789888 | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| v_13 | 0        | 0        | 3.789888 | 3.789888 | 0   | 0   | 0   | 0   | 0        | 3.789888 | 0        | 3.789888 | 0        | 0        | 0        | 0        | 0        | 0        |
| v_14 | 0        | 0        | 0        | 0        | 4   | 0   | 0   | 0   | 7.632169 | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| v_15 | 0        | 0        | 0        | 0        | 0   | 4   | 0   | 0   | 0        | 7.632169 | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| v_16 | 0        | 0        | 0        | 0        | 0   | 0   | 4   | 0   | 0        | 0        | 7.632169 | 0        | 0        | 0        | 0        | 0        | 0        | 0        |
| v_17 | 0        | 0        | 0        | 0        | 0   | 0   | 0   | 4   | 0        | 0        | 0        | 7.632169 | 0        | 0        | 0        | 0        | 0        | 0        |

## Reading .obj files

Here is a formatted pratt1.obj file indicating vertex id's and associated 2D coordinates

|      | x  | y       | z |
|------|----|---------|---|
| v_0  | 0  | -9.7192 | 0 |
| v_1  | 0  | 9.7192  | 0 |
| v_2  | 0  | -3.2192 | 0 |
| v_3  | 0  | 3.2192  | 0 |
| v_4  | -4 | 9.7192  | 0 |
| v_5  | 4  | -9.7192 | 0 |
| v_6  | -4 | -9.7192 | 0 |
| v_7  | 4  | 9.7192  | 0 |
| v_8  | -4 | 3.2192  | 0 |
| v_9  | 4  | -3.2192 | 0 |
| v_10 | -4 | -3.2192 | 0 |
| v_11 | 4  | 3.2192  | 0 |
| v_12 | -2 | 0       | 0 |
| v_13 | 2  | 0       | 0 |
| v_14 | -8 | 9.7192  | 0 |
| v_15 | 8  | -9.7192 | 0 |
| v_16 | -8 | -9.7192 | 0 |
| v_17 | 8  | 9.7192  | 0 |
