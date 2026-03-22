import os
import os.path as osp
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm

nusences_root = '/data/test04/datasets/08_nuScenes_kitti/'
def main(phase):
    print("Process {0} ......".format(phase))
    
    if not osp.exists('downsampled'):
        os.makedirs('downsampled')
    
    for split in ['train', 'val', 'test']:
        split_dir = osp.join('downsampled', split)
        if not osp.exists(split_dir):
            os.makedirs(split_dir)
    
    root = os.path.join(nusences_root,phase)
    subset_names = os.listdir(os.path.join(root, 'sequences'))
    print(subset_names)
    for dirname in subset_names:
        file_names = glob.glob(osp.join(nusences_root,phase,'sequences', dirname, 'velodyne', '*.bin'))
        for file_name in tqdm(file_names):
            frame = file_name.split('/')[-1][:-4]
            path = osp.join('downsampled', phase, dirname)
            if os.path.exists(path) != True:
                os.mkdir(path)
            new_file_name = osp.join(path, frame + '.npy')
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
            points = points[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = pcd.voxel_down_sample(0.2)
            points = np.array(pcd.points).astype(np.float32)
            np.save(new_file_name, points)


if __name__ == '__main__':
    for p in ['train','val','test']:
        main(p)
