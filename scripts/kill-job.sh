#!/bin/sh

ps aux | awk -v user="$USER" '$1 == user && (/python3 scripts\/compute_wct.py/ || /python3.*multiprocessing/) && !/awk/ { print $2 }' | xargs kill