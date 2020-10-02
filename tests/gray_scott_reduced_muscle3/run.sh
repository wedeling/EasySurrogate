#!/bin/bash

./muscle_manager reduced.ymmsl &

manager_pid=$!

echo 'Running reduced gray-scott in Python'

python3 ./micro.py --muscle-instance=micro >'micro.log' 2>&1 &
python3 ./macro.py --muscle-instance=macro >'macro.log' 2>&1 &

wait
