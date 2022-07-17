#! /bin/bash
#
# extract_all_frames.bash
# Copyright (C) 2022 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

for vid in `cat configs/video_ids.txt`
do
    bash preprocessing/extract_frames.bash $vid
done
