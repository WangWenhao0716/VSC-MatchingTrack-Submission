import pandas as pd
import numpy as np

import os
import shutil
from PIL import Image
from augly.image.transforms import *
from random import choice

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import Counter

from isc.io import read_descriptors, write_hdf5_descriptors
import faiss
import pickle

def videos_to_images():
    os.system('bash videos_to_images.sh')
    
    
def extract_feature_50SK_half():
    os.makedirs('./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_half/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_ff_v2 \
      --o ./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_half/query_v1_ff.hdf5 \
      --model 50SK  --GeM_p 3 --bw \
      --checkpoint train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar --imsize 256') 
    
def get_candidates():
    path = './feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_half/'
    ls = os.listdir(path)
    q_name, q_feature = read_descriptors([path + 'query_%d_v1_ff.hdf5'%i for i in range(len(ls))])
    q_name = np.array(q_name)
    series = []
    for q in q_name:
        if '__' in q:
            series.append(int(q.split('_')[0][1:])*1000000000 + int(q.split('_')[1])*1000 + int(q.split('__')[1]) + 1)
        else:
            series.append(int(q.split('_')[0][1:])*1000000000 + int(q.split('_')[1])*1000)
    q_name_new = q_name[np.argsort(series)]
    q_feature_new = q_feature[np.argsort(series)]
    timestamps = np.array([[float(q.split('_')[1])*0.5, float(q.split('_')[1])*0.5+0.5] for q in q_name_new], dtype=np.float32)
    path_to = path + 'query_v1_ff_sort.npz'
    q_name_new_clean = [q.split('_')[0] for q in q_name_new]
    
    features = q_feature_new
    video_ids = np.array(q_name_new_clean)
    timestamps = timestamps
    all_id = sorted(list(set(list(video_ids))))
    print("The length of all_id is", len(all_id))
    save = []
    for query_id in all_id:
        q_matrix = features[video_ids==query_id]
        ans = q_matrix@q_matrix.T
        scores = np.diag(ans, k=1)
        if min(scores)<0.3:
            save.append(query_id)
            
    with open('save_50SK_2_ff.pkl', 'wb') as f:
        pickle.dump(save, f)

    
def detectection():
    os.system('bash padding.sh')
    
    ls = sorted(os.listdir('/dev/shm/query_one_second_ff_v2_rotate_2')) 
    os.makedirs("/dev/shm/query_one_second_ff_v2_rotate_2_detection_v3_all", exist_ok = True)
    
    for i in range(len(ls)):
        name = ls[i]
        src = '/dev/shm/query_one_second_ff_v2_rotate_2/' + name
        dst = '/dev/shm/query_one_second_ff_v2_rotate_2_detection_v3_all/' + name
        shutil.copy(src, dst)
    
    os.system('conda run --no-capture-output -n condaenv \
    python detect.py --source /dev/shm/query_one_second_ff_v2_rotate_2_detection_media_v3_all \
    --weights best_20230101.pt --conf 0.1 > /dev/null')
    
def generation():
    path = '/dev/shm/query_one_second_ff_v2_rotate_2_detection_media_v3_all/'
    path_new = '/dev/shm/query_one_second_ff_v2_rotate_2_detection_v3_all/'
    path_ori = '/dev/shm/query_one_second_ff_v2/'
    all_detection = os.listdir(path)
    all_detection = [i for i in all_detection if 'jpg.npy' in i]
    
    for d in range(len(all_detection)):
        det = all_detection[d]
        detect = np.load(path + det)
        overylay = detect[detect[:,-1]==0]
        if len(overylay)>=1:
            for over in range(len(overylay)):
                overylay_select = overylay[over]
                old_img = Image.open(path + det[:-4])
                w, h = old_img.size
                max_length = max(w,h)
                enlarge = 640/max_length
                new_w = int(enlarge*w)
                new_h = int(enlarge*h)
                old_img = old_img.resize((new_w,new_h))
                new_img = old_img.crop(overylay_select[:-2])
                new_img.save(path_new +  det[:-8] + '__' + str(over) + '.jpg', quality=100)
    

def fix_rotate():
    os.system("conda run --no-capture-output -n condaenv python fix_rotate.py") 
            
def extract_feature_vit_bw():
    os.makedirs('./feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_ff_v2_rotate_2_detection_v3_all \
      --o ./feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/query_v1_rotate_detection_v3_all.hdf5 \
      --model vit_base  --GeM_p 3 --bw \
      --checkpoint train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN.pth.tar --imsize 224')

def extract_feature_swin_bw():
    os.makedirs('./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_ff_v2_rotate_2_detection_v3_all \
      --o ./feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/query_v1_rotate_detection_v3_all.hdf5 \
      --model swin_base  --GeM_p 3 --bw \
      --checkpoint train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN.pth.tar --imsize 224')
    
def extract_feature_t2t_bw():
    os.makedirs('./feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_ff_v2_rotate_2_detection_v3_all \
      --o ./feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/query_v1_rotate_detection_v3_all.hdf5 \
      --model t2t  --GeM_p 3 --bw \
      --checkpoint train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN.pth.tar --imsize 224')
    
def extract_feature_50SK_bw():
    os.makedirs('./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/', exist_ok = True)
    os.system('conda run --no-capture-output -n condaenv python extract_feature.py \
      --image_dir /dev/shm/query_one_second_ff_v2_rotate_2_detection_v3_all \
      --o ./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/query_v1_rotate_detection_v3_all.hdf5 \
      --model 50SK  --GeM_p 3 --bw \
      --checkpoint train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN.pth.tar --imsize 256')
    
def change(path, leng):
    q_name, q_feature = read_descriptors([path + 'query_%d_v1_rotate_detection_v3_all.hdf5'%i for i in range(leng-2)])
    q_name = np.array(q_name)
    series = []
    for q in q_name:
        if '__' in q:
            series.append(int(q.split('_')[0][1:])*1000000000 + int(q.split('_')[1])*1000 + int(q.split('__')[1]) + 1)
        else:
            series.append(int(q.split('_')[0][1:])*1000000000 + int(q.split('_')[1])*1000)
    q_name_new = q_name[np.argsort(series)]
    q_feature_new = q_feature[np.argsort(series)]
    timestamps = np.array([[float(q.split('_')[1])*0.5, float(q.split('_')[1])*0.5+0.5] for q in q_name_new], dtype=np.float32)
    path_to = path + 'query_v1_rotate_detection_v3_all_sort_3.npz'
    q_name_new_clean = [q.split('_')[0] for q in q_name_new]
    
    np.savez(
        path_to,
        video_ids = q_name_new_clean,
        timestamps= timestamps,
        features = q_feature_new,
    )
    
    
def transfer_hdf5_to_npz():
    
    paths = ['./feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/', \
             './feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/', \
             './feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/', \
             './feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/',]
    
    leng = len(os.listdir(paths[0]))
    for path in paths:
        change(path, leng)
             
def ensemble():
    os.system("bash ensemble.sh")
    
    match = pd.read_csv('./output/matches.csv')
    match_subset = pd.DataFrame()
    match_subset['query_id'] = match['query_id']
    match_subset['ref_id'] = match['ref_id']
    match_subset['query_start'] = match['query_start']
    match_subset['query_end'] = match['query_end']
    match_subset['ref_start'] = match['ref_start']
    match_subset['ref_end'] = match['ref_end']
    match_subset['score'] = match['score']
    match_subset.to_csv('subset_matches.csv', index=False)
    
    subset = pd.read_csv('subset_matches.csv')
    subset_1 = subset.groupby(by=['query_id','ref_id','query_start','query_end','ref_start','ref_end']).max()
    subset_1.to_csv('temp.csv')
    subset_2 = pd.read_csv('temp.csv')
    subset_2.to_csv('subset_matches.csv', index=False)

def main():
    import time
    begin = time.time()
    print("Install packages")
    
    os.system("conda run --no-capture-output -n condaenv python -m pip install timm-0.4.12-py3-none-any.whl")
    
    a = time.time()
    print("Transfer videos to images every one second using ff")
    videos_to_images()
    print("videos_to_images: ", (time.time() - a)/3600)
    
    print("Perform Deep Learning filter!")
    extract_feature_50SK_half()
    get_candidates()
    
    with open('save_50SK_2_ff.pkl', 'rb') as f:
        query_subset_video_ids_filter_1 = pickle.load(f)
        
    query_subset_video_ids_filter = query_subset_video_ids_filter_1 
    query_subset_video_ids_filter = sorted(list(set(query_subset_video_ids_filter)))
    
    images_all = sorted(os.listdir('/dev/shm/query_one_second_ff_v2/'))
    for i in range(len(images_all)):
        if images_all[i].split('_')[0] not in query_subset_video_ids_filter:
            os.remove('/dev/shm/query_one_second_ff_v2/' + images_all[i])
    
    b = time.time()
    print("Fix rotating")
    fix_rotate()
    print("fix_rotate: ", (time.time() - b)/3600)
    
    c = time.time()
    print("Using detection models to generate new test sets")
    detectection()
    generation()
    print("detectection_generation: ", (time.time() - c)/3600)
    
    print("Extract features")
    os.makedirs('feature/', exist_ok = True)
    
    d = time.time()
    extract_feature_vit_bw()
    extract_feature_swin_bw()
    extract_feature_t2t_bw()
    extract_feature_50SK_bw()
    print("extract_feature: ", (time.time() - d)/3600)
    
    print("Transfer hdf5 to npz")
    transfer_hdf5_to_npz()
    
    print("Get final results")
    ensemble()
    print(time.time()-begin)
    
if __name__ == "__main__":
    main()