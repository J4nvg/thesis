import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata, studentized_range

q = studentized_range.ppf(1 - 0.05, 6, np.inf) / np.sqrt(2)
print(q)