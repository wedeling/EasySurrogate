#!/bin/bash

./muscle_manager reduced.ymmsl &

manager_pid=$!

echo 'Running reduced gray-scott in Python'

python3 ./micro.py --muscle-instance=micro &
python3 ./macro.py --muscle-instance=macro &

wait
