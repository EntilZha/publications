#!/bin/bash -e


echo "PLOTTING ------------------------------------------------------"
echo $1

if [ -f $1/figures.py ]; then
        echo "Running $PYCOMMAND"
        python $1/figures.py
else
        echo "No Python scripts!"
fi
