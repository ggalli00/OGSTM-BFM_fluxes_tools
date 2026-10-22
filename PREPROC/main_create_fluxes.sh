#!/bin/bash

# executes on CINECA g100

source /g100_work/OGS23_PRACE_IT/COPERNICUS/sequence.sh
python create_fluxes.py -y create_fluxes.yaml

