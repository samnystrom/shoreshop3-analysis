#!/bin/sh
cat /sys/fs/cgroup/"$(cat /proc/self/cgroup | cut -d/ -f2-)"/memory.peak
