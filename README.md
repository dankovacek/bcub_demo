BC Unaged Basin Database
========================

![The British Columbia Ungauged Basin
Database](content/notebooks/img/all_pts_and_stns.png)

Introduction
------------

The British Columbia Ungauged Basin database is intended to support
water resources research, namely in the optimization of the British
Columbia streamflow monitoring network.

Notes
-----

There are several ways to use the information provided in this
repository.

1.  A minimal dataset containing basin attributes and pour points for
    nearly 1 million basins in and around British Columbia. This file
    does not contain basin polygons since these require very large disk
    space and we cannot host the data.
2.  An expanded set of compressed (parquet?) files containing the above,
    plus accompanying basin polygons, for a total of approximately 60
    GB.
3.  A set of notebooks demonstrating the complete process of generating
    basin polygons for the purpose of extracting basin attributes. The
    demonstration is carried out on a smaller region (Vancouver Island).

### Setup

See the [README.md under
`content/setup_scripts/`](https://github.com/dankovacek/bcub/tree/main/content/setup_scripts/README.md)
to get started.

### References

Addor, Nans, Grey Nearing, Cristina Prieto, AJ Newman, Nataliya Le Vine,
and Martyn P Clark. 2018. “A Ranking of Hydrological Signatures Based on
Their Predictability in Space.” *Water Resources Research* 54 (11):
8792–8812.

Addor, Nans, Andrew J Newman, Naoki Mizukami, and Martyn P Clark. 2017.
“The Camels Data Set: Catchment Attributes and Meteorology for
Large-Sample Studies.” *Hydrology and Earth System Sciences* 21 (10):
5293–5313.

Alvarez-Garreton, Camila, Pablo A Mendoza, Juan Pablo Boisier, Nans
Addor, Mauricio Galleguillos, Mauricio Zambrano-Bigiarini, Antonio Lara,
et al. 2018. “The Camels-Cl Dataset: Catchment Attributes and
Meteorology for Large Sample Studies–Chile Dataset.” *Hydrology and
Earth System Sciences* 22 (11): 5817–46.

Coxon, Gemma, Nans Addor, John P Bloomfield, Jim Freer, Matt Fry, Jamie
Hannaford, Nicholas JK Howden, et al. 2020. “CAMELS-Gb:
Hydrometeorological Time Series and Landscape Attributes for 671
Catchments in Great Britain.” *Earth System Science Data* 12 (4):
2459–83.

Fowler, Keirnan JA, Suwash Chandra Acharya, Nans Addor, Chihchung Chou,
and Murray C Peel. 2021. “CAMELS-Aus: Hydrometeorological Time Series
and Landscape Attributes for 222 Catchments in Australia.” *Earth System
Science Data* 13 (8): 3847–67.

Gupta, Hoshin Vijai, C Perrin, G Blöschl, A Montanari, R Kumar, M Clark,
and Vazken Andréassian. 2014. “Large-Sample Hydrology: A Need to Balance
Depth with Breadth.” *Hydrology and Earth System Sciences* 18 (2):
463–77.

Klingler, Christoph, Karsten Schulz, and Mathew Herrnegger. 2021.
“LamaH-Ce: LArge-Sample Data for Hydrology and Environmental Sciences
for Central Europe.” *Earth System Science Data* 13 (9): 4529–65.

Kratzert, Frederik, Daniel Klotz, Mathew Herrnegger, Alden K Sampson,
Sepp Hochreiter, and Grey S Nearing. 2019. “Toward Improved Predictions
in Ungauged Basins: Exploiting the Power of Machine Learning.” *Water
Resources Research* 55 (12): 11344–54.
