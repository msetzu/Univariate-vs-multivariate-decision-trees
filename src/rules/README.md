# Rules
Implementing decision rules for Decision Trees

## Premises

```python
from rules.rules import APPremise, OPremise

# axis-parallel premise stating that "feature 10 should be between 0 and 20"
ap_premise = APPremise(feat=10, low=0, upp=20)

# oblique premise: coefficients * x <= bound
o_premise = OPremise(coefficients=[2., 4., -1.2, 0], bound=5.)
```

### Rules
`Rule`s are simply sets of premises which operate jointly as an intersection of premises.

```python
import numpy
from rules.rules import APPremise
from rules.rules import APRule

ap_premise = APPremise(feat=10, low=0, upp=20)
rule = APRule(head=0, body=[ap_premise])

# access by feature
rule[10]
# check if premise in rule
10 in rule
# reset premises
rule[10] = APPremise(10, 0, 10)
rule[10]
# delete premises
del rule[10]
10 in rule
# check if some data satisfies the rule
rule = APRule(head=0, body=[ap_premise])
x = numpy.random.rand(12,)
rule(x)
```

#### Rule arithmetic
We also define a rule arithmetic: rules can be added into a new rule with premises from both, provided they are 
defined on different axes.  

```python
from rules.rules import APPremise
from rules.rules import APRule

ap_premise = APPremise(feat=10, low=0, upp=20)
rule = APRule(head=0, body=[ap_premise])
other_ap_premise = APPremise(feat=12, low=50, upp=100)
other_rule = APRule(head=0, body=[other_ap_premise])

# can sum rules with premises on different axes
new_rule = rule + other_rule
10 in new_rule
12 in new_rule
print(rule)
print(other_rule)
print(new_rule)
```