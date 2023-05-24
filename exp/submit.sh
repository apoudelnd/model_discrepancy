#!/bin/bash
#$ -q long     # Specify queue (use ‘debug’ for development)
#$ -N multi-m      # Specify job name
if [ -r /opt/crc/Modules/current/init/bash ]; then
    source /opt/crc/Modules/current/init/bash
fi
module load python/3.7.3
cd /afs/crc.nd.edu/user/a/apoudel/projects/model_discrepancy/
source ./venv/bin/activate
python ./exp/ms_coco.py 
