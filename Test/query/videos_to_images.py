import os
import argparse
import pandas as pd

DATA_DIRECTORY = "/data/"
QRY_VIDEOS_DIRECTORY = DATA_DIRECTORY + "query/"
QUERY_SUBSET_FILE = DATA_DIRECTORY + "query_subset.csv"

query_subset = pd.read_csv(QUERY_SUBSET_FILE)
ls = query_subset.video_id.values.astype("U")

df = pd.read_csv(DATA_DIRECTORY + 'query_metadata.csv')
path_save = '/dev/shm/query_one_second_ff_v2/'
os.makedirs(path_save, exist_ok = True)
#ls = ([q + '.mp4' for q in query_subset_video_ids])

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * (len(ls)//2 + 1)
end = (num+1) * (len(ls)//2 + 1)

for i in range(begin, end):
    name = ls[i]
    os.system('ffmpeg -hide_banner -loglevel panic -ss 00:00:00 -i ' + QRY_VIDEOS_DIRECTORY + name + '.mp4 -vf fps=2 -qscale:v 1 -start_number 0 ' + path_save + name + '_%d.jpg > /dev/null')