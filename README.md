# The Green Economy Diagnostics Tool


## Description


The [Green Economy Diagnostic (GED)](https://gpbp-ged.vercel.app/) is a World Bank Group tool that maps the economic and environmental performance of sub-national regions within a country. It is designed to provide a standardized, transparent, and comprehensive picture of regional performance in terms of both economic and environmental factors. It is aimed to help raise public awareness of nations' current development and the challenges they are facing. It is also meant to be a tool to aid governments in driving evidence-based policy making, managing public assets, and targeting infrastructure investments. It is part of the World Bank's [Public Investment Management - Public Assets Management platform](https://pim-pam.net/) Platform.

This repository provides the code to:
* Extract raw geospatial data which constitute the GED's scores from various public and private sources
* Process the extracted raw data to create the scores according to this [methodology](https://gpbp-ged.vercel.app/methodology)

## Data Requirements

The following table lists all sources of the raw data that is used as part of the GED. Please contact the author to gain access to the private sources

| Domain                    | Score                          | Availability  | Granularity  | Source                                   |  Public/Private      |
| :---                      |    :----                       |     :---      |     :---     |   :---                                   | :---                 |
| Nighttime Luminosity      | Economic                       | 2014-Present  |    500m      | NASA Black Marble                        | Public               |
| Population                | Economic                       | 2000-2020     |    1000m     | WorldPop                                 | Public               | 
| Land Cover                | Economic, Environmental        | 2016-Present  |    10m       | Google Dynamic World V1                  | Public               |
| Air Pollution             | Environmental                  | 2018-Present  |    44528m    | Copernicus Sentinel-5P                   | Public               |
| Air Pollution PM2.5       | Environmental                  | 2016-Present  |    1113.2m   | Copernicus Atmosphere Monitoring Service | Public               |
| Air Temperature           | Environmental                  | 2000-Present  |    27830m    | Telespazio                               | Private              |
| Precipitation             | Environmental                  | 2000-Present  |    27830m    | Telespazio                               | Private              |

## Getting Started

### Dependencies

* Python 3.8 or above

### Installing

The recommended option is to use a [miniconda](https://conda.io/miniconda.html)
environment to work in for this project, relying on conda to handle some of the
trickier library dependencies.

Create a conda environment for the project and install packages
```bash
conda create --name <env> --file requirements.txt
```

### Executing program

To generate scores, please use and following the instructions detailed in the Jupyter Notebook named _GED.ipynb_


## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

![Alt text](images/image-3.png)

![Alt text](images/image.png)
