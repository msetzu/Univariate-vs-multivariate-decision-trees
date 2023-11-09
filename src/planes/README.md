# Hyperplanes
`Hyperplane` is an abstract class representing hyperplanes, and is implemented by
`APHyperplane` and `OHyperplane`, impelentig axis-parallel and oblique hyperplanes, respectively.
Both can be:
- copied
- deep-copied
- hashed
- compared for equality
- accessed through standard array notation, e.g., `hyperplane[feature]` and `hyperplane[feature] = 0.`
- iterated over, yielding their values, coefficients, e.g., `for coefficient in ohyperplane`
- invoked to know whether some data lies within the hyperplane, e.g., `hyperplane(x)`

```python
from planes.planes import OHyperplane, APHyperplane

import numpy


ohyperplane = OHyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
aphyperplane = APHyperplane(1, 10., 30.)  # x[4] in [0, 20)

x = numpy.array([[5., 2., 1.],      # 5*1 -0.5*2 <= 10
                 [22., 20., 0.]])   # 22*1 -0.5*21 <= 10
print(ohyperplane(x))
# [ True False]
print(aphyperplane(x))
# [ False True]
```

## Oblique hyperplanes
The `OHyperplane` class support basic ring operations, which are defined **only on hyperplanes of the same size**.
Applying them to `Hyperplanes`s of different sizes will raise an exception.

**All operations are stateless**, that is, they **do not** modify the original `Hyperplane`, rather they return a copy
on which the operation is performed.
```python
from planes.planes import OHyperplane

import numpy

hyperplane_1 = OHyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
hyperplane_2 = OHyperplane(numpy.array([-2., +0.5, 1.]), 10.)  # 1*x1 -0.5*x2 <= 10.

sum_hyperplanes = hyperplane_1 + hyperplane_2   # sums coefficients and bounds
sub_hyperplanes = hyperplane_1 - hyperplane_2   # subtracts coefficients and bounds
mul_hyperplanes = hyperplane_1 * hyperplane_2   # multiplies coefficients and bounds
inverted_hyperplane = ~hyperplane_1             # negates coefficients and bounds
negated_hyperplane = -hyperplane_1              # negates coefficients and bounds
```
Moreover, we can induce an oblique hyperplane from an axis-parallel one with `OHyperplane.from_aphyperplane`:
```python
from planes.planes import OHyperplane, APHyperplane

import numpy


ohyperplane = OHyperplane(numpy.array([1., -0.5, 0.]), 10.)  # 1*x1 -0.5*x2 <= 10.
aphyperplane = APHyperplane(1, 10., 30.)  # x[4] in [0, 20)

converted_ohyperplane = OHyperplane.from_aphyperplane(aphyperplane,
                                                      dimensionality=ohyperplane.coefficients.size)
print(converted_ohyperplane)
```


`APHyperplane`s are more limited, as they can only be inverted:
```python
from planes.planes import APHyperplane

import numpy

aphyperplane = APHyperplane(1, 10., 30.)  # x[4] in [0, 20)
print(~aphyperplane)
```
