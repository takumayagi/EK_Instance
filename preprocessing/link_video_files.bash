#! /bin/bash
#
# link_video_files.bash
# Copyright (C) 2022 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

EPIC_ROOT=$1
OUT_DIR="data/epic_kitchens"
mkdir -p $OUT_DIR

for pid in 1 2 5 7 9 11 14 15 16 18 19 20 25 26 27 29 31 32 33 34 35 36
do
    PID=`printf "P%02d" $pid`
    mkdir -p $OUT_DIR/$PID
    if [ ! -e $EPIC_ROOT/$PID/videos ]
    then
        echo "No directory found: $EPIC_ROOT/$PID/videos"
    else
        for VIDEO_NAME in `ls -1 $EPIC_ROOT/$PID/videos`
        do
            echo "ln -s $EPIC_ROOT/$PID/$VIDEO_NAME $OUT_DIR/$PID/$VIDEO_NAME"
            ln -s $EPIC_ROOT/$PID/$VIDEO_NAME $OUT_DIR/$PID/$VIDEO_NAME
        done
    fi
done
