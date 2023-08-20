# Replication Code for "Land Offers and Fiscal Competition Between City Governments in China"

# Note
- [Numba](https://numba.pydata.org/) and 
[Ray](https://www.ray.io/) are needed to run the code.
- [Pyecharts (version 0.5.11)](https://05x-docs.pyecharts.org/#/) is needed for making the map (Figure 2 in the paper).
And the [map packages](https://05x-docs.pyecharts.org/#/zh-cn/customize_map) should also be installed.

# Structure of the code
- Data: The data sets used for estimation.

- Modules: Python code of Bertrand game solver, model simulator, and MSM estimator.

- Tables: The folder saves all the table outputs of the code.

- Graphs: The folder saves all the graph outputs of the code.

- data_description.py: The code used to make descriptive statistics and visualization of data.

- estimation.py: The code to implement estimations in the paper (Section 6.1 and Appendix B).

- model_fitness.py: The code used to generate Figure 4 and Table 4 in Section 6.3 (Fit of the model).

- counterfactuals.py: The code used to generate counterfactuals in Section 7.

- main.py: The main program generates all the tables and plots
in the paper.


# How to run the code
- run main.py to get all the tables and graphs in the paper.
(If you want to suppress the ray information, you may use python ``main.py 2>&1 > out.log`` to redirect the output to a log file.)

&copy; 2023 Wending Liu <Wending.Liu@anu.edu.au>
