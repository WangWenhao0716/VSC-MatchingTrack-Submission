import os
import pandas as pd
import cv2
import argparse

path = '/raid/VSC/data/test/reference/'
ls = sorted(os.listdir(path))
ls = [i.split('.')[0] for i in ls]
path_save = '/raid/VSC/data/test/reference_one_second_ff_v2/'
os.makedirs(path_save, exist_ok=True)

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * (len(ls)//100 + 1)
end = (num+1) * (len(ls)//100 + 1)

for i in range(begin, end):
    name = ls[i]
    print(i, name)
    os.system('ffmpeg -ss 00:00:00 -i ' + path + name + '.mp4 -vf fps=2 -qscale:v 1 -start_number 0 ' + path_save + name + '_%d.jpg')
