#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:${PWD}/
export BASE_PATH=${PWD}
export TRANSFORMERS_CACHE=/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/cache/

if [[ "$HOSTNAME" == login*ufhpc ]]; then
    echo "Loading modules"
    module load python/3.8
    module load git
fi
