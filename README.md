# spread_dimension

## The Spread Dimension of a Metric Space

Spread dimension of a metric space was defined by Willerton in
<i>Spread: A measure of the size of metric spaces. International
Journal of ComputationalGeometry & Applications, 25(03):207–225,
2015.</i> <https://doi.org/10.48550/arXiv.1209.2300>.

The spread and spread dimension are also introduced in the n-Catgory
Cafe blog post:

<https://golem.ph.utexas.edu/category/2012/09/the_spread_of_a_metric_space.html>

## Estimating Intrinsic Dimension of Data

Spread dimension can be used to estimate the intrinsic dimension of
data <https://doi.org/10.48550/arXiv.2308.01382>

## Using spread_dimension

Installation

## Examples

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
 
from spread_dimension import EuclideanSubspace
 
swiss_roll_points = make_swiss_roll(1000)[0]
 
swiss_roll = EuclideanSubspace(swiss_roll_points)

swiss_roll.compute_metric()

T = swiss_roll.find_operative_range()

X = np.linspace(0, T, 100)
Y = [swiss_roll.spread_dimension(t) for t in X]

plt.plot(X,Y)
plt.show()
```
