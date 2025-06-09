#!/usr/bin/env bash
set -eu

FOLDER=test_numfolio

export N_JOBS="1"
export N_BOOTSTRAPS="50"
export SEED="42"

echo "##########################################"
echo "### Execute tests for numfolio package ###"
echo "##########################################"


coverage run suite.py --test $FOLDER
coverage report -m

echo ""
echo ""
echo "Done!"
