# robust-mixed-dist

Data scientists address real-world problems using multivariate and heterogeneous
datasets, characterized by multiple variables of different natures. Selecting a suitable
distance function between units is crucial, as many statistical techniques and machine
learning algorithms depend on this concept. Traditional distances, such as Euclidean
or Manhattan, are unsuitable for mixed-type data, and although Gower distance was
designed to handle this kind of data, it may lead to suboptimal results in the presence
of outlying units or underlying correlation structure.

In the paper ***Grané , Aurea; Scielzo-Ortiz, Fabio. “On generalized Gower distance for mixed-type data: extensive simulation study and new software tools”. SORT-Statistics and Operations Research Transactions, pp. 213-44, doi:10.57645/20.8080.02.28.*** robust distances for mixed-type data are defined and explored, namely **robust generalized Gower** and **robust related metric scaling**. In addition,  the new Python package `robust-mixed-dist` is developed, which enables to
compute these robust proposals as well as classical ones.

The package is located in Python Package Index (PyPI), the standard repository of packages for the Python programming language: https://pypi.org/project/robust_mixed_dist/

- **Package documentation:** https://fabioscielzoortiz.github.io/robust-mixed-dist-docu/intro.html

- **Paper link:** https://raco.cat/index.php/SORT/article/view/9900373