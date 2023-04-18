from isc.io import read_descriptors
import numpy as np

paths = ['./feature/train_v1_50SK_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_2_cls_FIN/', \
        './feature/train_v1_vit_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/', \
        './feature/train_v1_swin_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/', \
        './feature/train_v1_t2t_s3_512_twolosses_com_L2_norm_100_all_tune_bw_gt_ng_1_cls_FIN/']

for path in paths:
  r_name, r_feature = \
  read_descriptors([path + 'reference_%d_v1_imageio.hdf5'%i for i in range(25)])
  r_name = np.array(r_name)
  series = [int(r.split('_')[0][1:])*1000000 + int(r.split('_')[1]) for r in r_name]
  r_name_new = r_name[np.argsort(series)]
  r_feature_new = r_feature[np.argsort(series)]
  timestamps = np.array([[float(r.split('_')[1]), float(r.split('_')[1])+1] for r in r_name_new], dtype=np.float32)
  path_to = path + 'reference_v1_sort_train.npz'
  r_name_new_clean = [r.split('_')[0] for r in r_name_new]
  np.savez(
      path_to,
      video_ids = r_name_new_clean,
      timestamps= timestamps,
      features = r_feature_new,
  )

