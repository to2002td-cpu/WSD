#!/bin/bash

#OAR -q production 
#OAR -l {resources.gpu_mem > 20000}/host=1/gpu=1,walltime=12:0:0
#OAR -O .1.logs
#OAR -E .1.errors

bash /home/tderrien/WSD/scripts/oar_job.sh