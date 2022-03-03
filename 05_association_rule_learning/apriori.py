import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)


# the apyori only accepts list of lists so we convert dataframe as such
vals = dataset.values.tolist()

for i in range(len(vals)):
    for j in range(len(vals[i])):
        vals[i][j] = str(vals[i][j])

from apyori import apriori
rules = apriori(transactions=vals,min_support = 0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2)
results = list(rules)

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Unsorted
print(resultsinDataFrame)
print()
# sorted
print(resultsinDataFrame.nlargest(n = 15, columns = 'Lift'))