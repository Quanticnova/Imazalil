#!/bin/bash
# this script takes all input files in a directory and creates an mp4 movie.
# IMPORTANT NOTE: input should be something like '*.png'

ffmpeg -r 60 -f image2 -pattern_type glob -i '*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p output.mp4

exit 0 

