"""Code of the paper "Land Offers and Fiscal Competition Between City Governments in China" """
__author__ = "Wending Liu"
__copyright__ = "Copyright (C) 2023 Wending Liu"
__license__ = "Academic Free"
__version__ = "1.0"

import data_description
import estimation
import model_fitness
import counterfactuals


# data description (Section 4.2)
data_description.draw_map()
data_description.describe_data()

# main results of estimation (Section 6)
estimation.main(margin=2000, size_list=[(3, 5)])

# robustness checks for different settings of choice sets (Appendix B.1)
estimation.main(margin=2000, size_list=[(3, 4), (4, 5)])

# robustness checks for different participation constraints (Appendix B.2)
for margin in [1000, 3000]:
    estimation.main(margin, size_list=[(3, 5)])


# fit of model (Section 6.3)
model_fitness.main()

# counterfactuals (Section 7)
counterfactuals.main()
