import numpy as np
import torch
from ugp.modules.ops import grid_subsample, radius_search

def get_p2b(points,down_stage,min_vals,max_vals,img_size=(512,512)):
    xy_points = points[:,:2]
    xy_points = (xy_points - min_vals) / (max_vals - min_vals)
    
    # Map normalized coordinates to the target image resolution
    xy_points = (xy_points * torch.tensor(img_size, dtype=torch.float32).cuda()).long()
  
    xy_points[:, 0] = xy_points[:, 0].clamp(0, img_size[1] - 1)
    xy_points[:, 1] = xy_points[:, 1].clamp(0, img_size[0] - 1)
    
    # Compute the spatial downsampling factor for the current stage
    scaling_factor = 2 ** (down_stage - 1)
    superpoint_xy_mapped = (xy_points / scaling_factor).long()

    th = img_size[0] // scaling_factor - 1
    superpoint_xy_mapped[:, 0] = superpoint_xy_mapped[:, 0].clamp(0, th)
    superpoint_xy_mapped[:, 1] = superpoint_xy_mapped[:, 1].clamp(0, th)

    return superpoint_xy_mapped


def point_cloud_to_image(point_cloud, img_size=(512, 512)):
    """
    Project the point cloud onto the XY plane and generate a grayscale image.
    """
    xy_points = point_cloud[:, :2]

    # Normalize point coordinates to the [0, 1] range
    min_vals = torch.min(xy_points, dim=0)[0]
    max_vals = torch.max(xy_points, dim=0)[0]
    xy_points = (xy_points - min_vals) / (max_vals - min_vals)

    # Map normalized coordinates to the target image resolution
    xy_points = (xy_points * torch.tensor(img_size, dtype=torch.float32)).long()
    xy_points[:, 0] = xy_points[:, 0].clamp(0, img_size[1] - 1)
    xy_points[:, 1] = xy_points[:, 1].clamp(0, img_size[0] - 1)

    # Create an empty image
    img = torch.zeros(img_size, dtype=torch.float32)

    # Set projected point locations to 1.0
    img[xy_points[:, 1], xy_points[:, 0]] = 1.0
    img = img.unsqueeze(0)

    return img, min_vals, max_vals


def precompute_data_stack_mode_full(points, lengths, num_stages, voxel_size, radius, neighbor_limits, max_points=None, phase=None):
    assert num_stages == len(neighbor_limits)
    raw_radius = radius
    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []
    min_vals = []
    max_vals = []
    
    ref_points_img,ref_min_vals,ref_max_vals = point_cloud_to_image(points[:lengths[0]])
    src_points_img,src_min_vals,src_max_vals = point_cloud_to_image(points[lengths[0]:])

    min_vals.append(ref_min_vals)
    min_vals.append(src_min_vals)
    max_vals.append(ref_max_vals)
    max_vals.append(src_max_vals)
    
    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2
    
    if max_points is not None:  # GPU <= 24G, nuscenes datasets
        # Check and adjust the last stage point cloud if necessary
        ref_flag = 0
        src_flag = 0
        ref_length, src_length = lengths_list[-1]
        ref_points_c = points_list[-1][:ref_length]
        src_points_c = points_list[-1][ref_length:]
    
        # Process for ref points
        if ref_length > max_points:
            ref_indices = np.random.permutation(ref_points_c.shape[0])[: max_points]
            ref_points_c = ref_points_c[ref_indices]
            # Update points
            # Update lengths
            lengths_list[-1][0] = max_points
            ref_flag = 1

        ref_length, src_length = lengths_list[-1]
        # Process for src points
        if src_length > max_points:
            src_indices = np.random.permutation(src_points_c.shape[0])[: max_points]
            src_points_c = src_points_c[src_indices]
            src_flag = 1 
            # Update lengths
            lengths_list[-1][1] = max_points
    
        if ref_flag==1 or src_flag == 1:
            points_list[-1] = torch.cat([ref_points_c,src_points_c],dim=0)
            neighbors_list = []
            subsampling_list = []
            upsampling_list = []
            
            radius = raw_radius
            # radius search
            for i in range(num_stages):
                cur_points = points_list[i]
                cur_lengths = lengths_list[i]

                neighbors = radius_search(
                    cur_points,
                    cur_points,
                    cur_lengths,
                    cur_lengths,
                    radius,
                    neighbor_limits[i],
                )
                neighbors_list.append(neighbors)

                if i < num_stages - 1:
                    sub_points = points_list[i + 1]
                    sub_lengths = lengths_list[i + 1]

                    subsampling = radius_search(
                        sub_points,
                        cur_points,
                        sub_lengths,
                        cur_lengths,
                        radius,
                        neighbor_limits[i],
                    )
                    subsampling_list.append(subsampling)

                    upsampling = radius_search(
                        cur_points,
                        sub_points,
                        cur_lengths,
                        sub_lengths,
                        radius * 2,
                        neighbor_limits[i + 1],
                    )
                    upsampling_list.append(upsampling)

                radius *= 2

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
        'ref_img':ref_points_img,
        'src_img':src_points_img,
        'min_vals':min_vals,
        'max_vals':max_vals,
    }
    
def registration_collate_fn_stack_mode_full(
    data_dicts, 
    num_stages, 
    voxel_size, 
    search_radius, 
    neighbor_limits, 
    precompute_data=True,
    max_points=None,
    phase='train',
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode_full(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits, max_points, phase)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


