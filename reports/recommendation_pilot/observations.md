# Preliminary observations

These examples describe this pilot only. Each condition has two populations.
They do not establish a population-wide effect or an optimal diversity penalty.
The complete tables include confidence intervals and all other settings.

## Full catalogue check

At K=500, all methods offer the complete eligible catalogue.
All 120 method-population comparisons matched full browsing exactly.
Consumed article IDs, utilities, fatigue penalties, and departure times were equal.
This checks the exposure-size explanation under the current reader model.

## MMR example

These means use K=10, phi=0.5, rho=0.5, and a zero departure threshold.

| History visible | MMR penalty | Category entropy | Net utility per consumed article |
| --- | ---: | ---: | ---: |
| No | 0.3 | 0.3410 | 1.9412 |
| No | 0.9 | 0.3786 | 1.7643 |
| No | 1.0 | 0.3738 | 1.6944 |
| Yes | 0.3 | 0.3684 | 1.9245 |
| Yes | 0.9 | 0.3868 | 1.9643 |
| Yes | 1.0 | 0.4080 | 2.0557 |

Higher penalties reduced this utility measure for new users in this example.
The returning-user example showed a different direction.
Thus this pilot does not support one universal penalty threshold.
Compare cumulative utility and consumption counts when readers leave early.

## History and fatigue examples

At the same settings, visible history increased content ranking's category entropy
from 0.3171 to 0.3431. Its article overlap also increased, from 0.0577 to 0.0600.
These two proxies therefore describe different changes.

For returning users, fatigue content increased category entropy from 0.3431 to 0.5292.
Its mean fatigue penalty fell from 0.5817 to 0.5303.
Its net utility per consumed article also fell, from 1.9197 to 1.8720.
These results show a trade-off within this small simulation.

Fatigue-aware popularity and MF matched their blind versions in this K=10 example.
The new feedback does not guarantee different rankings or improved outcomes.
Longer runs and more populations are needed before drawing broader conclusions.

Use the paired comparison tables for matched differences across every condition.
