from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools


#QUIZ Q3:
data =np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])

model = UnivariateGaussian().fit(data)
print(np.round(model.log_likelihood(1,1,data),2))
print(np.round(model.log_likelihood(10,1,data),2))