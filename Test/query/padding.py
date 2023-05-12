import os
from PIL import Image
from augly.image.transforms import *
import argparse


ls = sorted(os.listdir('/dev/shm/query_one_second_ff_v2_rotate_2'))
os.makedirs('/dev/shm/query_one_second_ff_v2_rotate_2_detection_media_v3_all/', exist_ok = True)

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * (len(ls)//6 + 1)
end = (num+1) * (len(ls)//6 + 1)

class PPad:
    def __call__(self, x):
        w_factor = 0.1
        h_factor = 0.1
        color_1 = 255
        color_2 = 255
        color_3 = 255
        x = Pad(w_factor = w_factor, h_factor = h_factor, color = (color_1, color_2, color_3))(x)
        return x

for i in range(begin, end):
    name = ls[i]
    src = '/dev/shm/query_one_second_ff_v2_rotate_2/' + name
    dst = '/dev/shm/query_one_second_ff_v2_rotate_2_detection_media_v3_all/' + name
    img = Image.open(src)
    img = PPad()(img)
    img.save(dst, quality=100)