<!-- ---
bibliography: docs/references/references.bib
nocite: "@*"
--- -->

BC Ungauged Basin Database
==========================

![The British Columbia Ungauged Basin
Database](notebooks/img/all_pts_and_stns.png)

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
`docs/content/setup_scripts/`](https://github.com/dankovacek/bcub_demo/tree/main/docs/setup_scripts/README.md)
to get started.

### References
