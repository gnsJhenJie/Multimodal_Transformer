import matplotlib.pyplot as plt
import tqdm
import numpy as np
import math
from .object_class import Lane, agent, other_error
from numpy import linalg as LA
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap


def euclideanDistance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def check_dict(lanes_points_dict):
    empty_flag = True
    for Lane_type in lanes_points_dict.keys():
        if len(lanes_points_dict[Lane_type]):
            empty_flag = False
    return empty_flag


def get_around_lane(nusc_map, lane_dict, agent, radius, resolution_meters):
    """
    Get lanes points within a radius of query point.
    :param nusc_map: nuScenes map api class
    :param pos: current x,y position.
    :param radius: Radius around point to consider.
    :param resolution_meters: the distance between each continuos lane set point
    :return: Dictionary -> {Lane_id : Lane_set_points}
    """
    pos = agent.get_attribute('cur_pos')
    x = pos[0]
    y = pos[1]

    lane_id = list()
    lanes = nusc_map.get_records_in_radius(
        x, y, radius, ['lane', 'lane_connector'])
    #     lanes = lanes['lane'] + lanes['lane_connector']

    for lane in lanes['lane']:
        lane_record = nusc_map.get_arcline_path(lane)
        poses = arcline_path_utils.discretize_lane(
            lane_record, resolution_meters=resolution_meters)
        poses = np.array(poses)[:, :2]  # :2 get x,y coordinate
        lane_dict[lane] = Lane('lane', poses)
        lane_dict[lane].get_nearest_point_index(pos)
        lane_dict[lane].get_lane_vector(pos)
        lane_id.append(lane)

    for lane in lanes['lane_connector']:
        lane_record = nusc_map.get_arcline_path(lane)
        poses = arcline_path_utils.discretize_lane(
            lane_record, resolution_meters=resolution_meters)
        poses = np.array(poses)[:, :2]  # :2 get x,y coordinate
        lane_dict[lane] = Lane('lane_connector', poses)
        lane_dict[lane].get_nearest_point_index(pos)
        lane_dict[lane].get_lane_vector(pos)
        lane_id.append(lane)

    if len(lane_id) == 0:
        raise other_error('get_around_lane empty dict')

    return lane_id


def get_closest_lane(nusc_map, x, y, radius):
    """
    Get closest lane id within a radius of query point. The distance from a point (x, y) to a lane is
    the minimum l2 distance from (x, y) to a point on the lane.
    :param x: X coordinate in global coordinate frame.
    :param y: Y Coordinate in global coordinate frame.
    :param radius: Radius around point to consider.
    :return: Lane id of closest lane within radius.
    """

    lane_dict = nusc_map.get_records_in_radius(
        x, y, radius, ['lane', 'lane_connector'])
    lanes = lane_dict['lane'] + lane_dict['lane_connector']
    discrete_points = nusc_map.discretize_lanes(lanes, 0.5)

    current_min = np.inf

    min_id = ""
    for lane_id, points in discrete_points.items():
        distance = np.linalg.norm(
            np.array(points)[:, :2] - [x, y], axis=1).min()
        if distance <= current_min:
            current_min = distance
            min_id = lane_id

    if min_id in lane_dict['lane']:
        return min_id, 'lane'
    else:
        return min_id, 'lane_connector'


def get_same_direction_lanes(lane_dict, lanes_id, agent, angle_threshold, stop_threshold):
    '''
    remain Lanes which is same direction as agent
    :method calculate the angle between lane direction vector and agent direction vector
    :param lanes_points_dict: Dictionary -> {'Lane_type': {Lane_id : Lane_class}}
    :param agent: agent_class
    :param angle_threshold: interest degree(lane and object direction)
    :param stop_threshold: a threshold to determine the agent is stop or not
    :return: Dictionary -> {'Lane_type': {Lane_id : Lane_set_points}}
    '''
    agent.get_line(lane_dict, lanes_id, stop_threshold)
    agent_vector = agent.get_attribute('agent_vector')
    same_direction_lanes_id = list()
    for lane_id in lanes_id:
        lane_class = lane_dict[lane_id]
        lane_vector = lane_class.get_attribute('lane_vector')

        # unitization lane vector and agent vector
        zero = np.zeros(2)
        unit_len = euclideanDistance(agent_vector, zero)
        nor_for_point_vec = np.true_divide(agent_vector, unit_len)
        unit_len = euclideanDistance(lane_vector, zero)
        nor_for_lane_vec = np.true_divide(lane_vector, unit_len)

        # inner product -> cos value (after unitization)
        inner_product = np.dot(nor_for_point_vec, nor_for_lane_vec)
        if inner_product > 1.0:
            inner_product = 1.0
        # get the degree
        angle = np.degrees(np.arccos(inner_product))

        if angle < angle_threshold:
            same_direction_lanes_id.append(lane_id)

    if len(same_direction_lanes_id) == 0:
        raise other_error('get_same_direction_lanes empty dict')

    return same_direction_lanes_id


def get_future_lanes(lane_dict, lanes_id, agent):
    '''
    remain lanes let have a point in front of agent (or closest lane if agent is stop)
    :param lanes_points_dict: Dictionary -> {Lane_id : Lane_class}
    :param agent: agent class
    :return: a Dictionary -> {'Lane_type': {Lane_id : Lane_class}}
    '''

    future_lanes_id = list()
    # orthogonal to agent_direction (or closest lane direction if agent is stop)
    slope = agent.get_attribute('orthogonal_slope')
    bias = agent.get_attribute('orthogonal_bias')
    past_point = agent.get_attribute('past_pos')

    # Determine the region of point (1 line divide a plane into 2 region)
    agent_flag = bias - \
        past_point[0] > 0 if slope == 0 else past_point[1] - \
        slope * past_point[0] - bias > 0
    for lane_id in lanes_id:
        # Get lane points
        lane_class = lane_dict[lane_id]
        poses = lane_class.get_attribute('lane_set_points')
        # Determine the region of point (1 line divide a plane into 2 region)
        i = lane_class.get_attribute('nearest_point_index')
        lane_flag = agent_flag
        # if there is one lane point in front of agent add in the dict
        while lane_flag == agent_flag and i in range(lane_class.get_attribute('nearest_point_index'), len(poses)):
            lane_flag = bias - \
                poses[i][0] > 0 if slope == 0 else poses[i][1] - \
                slope * poses[i][0] - bias > 0
            if lane_flag != agent_flag:
                future_lanes_id.append(lane_id)
            i = i + 1

    if len(future_lanes_id) == 0:
        raise other_error('get_future_lanes empty dict')

    return future_lanes_id


def get_nearest_lane(nusc_map, lane_dict, lanes_id, agent):
    '''
    get closest right middle left lanes
    :param nusc_map: nuScenes api
    :param lanes_points_dict: Dictionary -> {lane_id : Lane class}
    :param agent: agent class
    :return: a 1D list -> [right_id, middle_id, left_id] (if None append zero)
    '''
    # parallel to agent_direction (or closest lane direction if agent is stop)
    slope = agent.get_attribute('parallel_slope')
    bias = agent.get_attribute('parallel_bias')

    # init parameter
    max_number = -1e10
    min_number = 1e10
    right_id = 0
    left_id = 0
    nearest_lane = list()

    # get middle lane (assume the closest lane is middle lane)
    closest_lane = agent.get_closest_lane(lane_dict, lanes_id)
    out_goings = nusc_map.get_outgoing_lane_ids(closest_lane)
    if len(out_goings) == 0:
        raise other_error('get_nearest_lane no outgoing lane!')

    lane_class = lane_dict[closest_lane]

    # delete it for not effecting the region determination
    for lane_id in out_goings:
        if lane_id in lanes_id:
            lanes_id.remove(lane_id)

    # if it is lane_connector using the last point to determine the distance
    if lane_class.get_attribute('lane_type') == 'lane_connector':
        min_distance = 1e10
        for lane_id in lanes_id:
            lane_class = lane_dict[lane_id]
            if lane_class.get_attribute('lane_type') == 'lane_connector':
                poses = lane_class.get_attribute('lane_set_points')
                ep2line_dis = np.abs(
                    poses[-1][1] - slope * poses[-1][0] - bias)
                if ep2line_dis < min_distance:
                    closest_lane = lane_id
                    min_distance = ep2line_dis

    lanes_id.remove(closest_lane)

    for lane_id in lanes_id:
        lane_class = lane_dict[lane_id]
        poses = lane_class.get_attribute('lane_set_points')
        nearst_point_index = lane_class.get_attribute('nearest_point_index')

        # Determine the closest lane at the specific region (right or left)
        # if it is lane_connector using the end point to determine the region
        if lane_class.get_attribute('lane_type') == 'lane_connector':
            nearst_point_index = -1
        num = slope * poses[nearst_point_index][0] + \
            bias - poses[nearst_point_index][1]
        if num > 0 and num > max_number:
            max_number = num
            right_id = lane_id
        if num < 0 and num < min_number:
            min_number = num
            left_id = lane_id
    # if right_id == 0 or left_id == 0:
    #     raise other_error('get_nearest_lane no right or left lane!')
    nearest_lane.append(right_id)
    nearest_lane.append(closest_lane)
    nearest_lane.append(left_id)
    
    return nearest_lane


def get_future_80m_lane(nusc_map, cur_pos, lane_dict, lanes_id, resolution_meters, sample_num):
    '''
    get 80m lanes points
    :param nusc_map: nuScenes map api class
    :param lanes_id_list: a list stored right middle left lane id
    :param cur_pos: agent current position
    :param resolution_meters: the distance between sampled points
    :param sample_num: the points number you want to sample
    :return 3d np.array [lane_num, sample_num, pos]
    '''
    feature_lanes = list()
    # get lane set points after current pos
    for lane_id in lanes_id:

        if lane_id == 0:  # 0 -> no lane
            dummy_input = np.ones((sample_num, 2))*(-100)
            feature_lanes.append(dummy_input)
            continue

        # get lane set points
        poses = lane_dict[lane_id].get_attribute('lane_set_points')
        nearst_point_index = lane_dict[lane_id].get_attribute(
            'nearest_point_index')

        # will not out of index if the nearst_point_index is end_point_index
        lane_set_points = poses[nearst_point_index + 1:]

        # Determine the lane_set_points is enough or not
        if len(lane_set_points) >= sample_num:
            lane_set_points = lane_set_points[:sample_num]
        # start lane set points less than sample_num , we extend the lane set points by other lane
        elif len(lane_set_points) < sample_num:
            while len(lane_set_points) < sample_num:
                # get the extended point by the delta_pos of original lane set point
                next_x = poses[-1][0] + (poses[-1][0] - poses[-2][0])
                next_y = poses[-1][1] + (poses[-1][1] - poses[-2][1])
                # get new lane by extended point
                if lane_dict[lane_id].get_attribute('next_lane_id') is None:
                    next_lane, next_lane_type = get_closest_lane(
                        nusc_map, next_x, next_y, radius=5)
                    next_lane_record = nusc_map.get_arcline_path(next_lane)
                    poses = arcline_path_utils.discretize_lane(
                        next_lane_record, resolution_meters=resolution_meters)
                    poses = np.array(poses)[:, :2]
                    # store next_lane information (using extended point to search will always get the same result to same lane)
                    lane_dict[lane_id].set_next_lane_id(next_lane)
                    # if next_lane is not in dictionary store the info
                    if next_lane not in lane_dict.keys():
                        lane_dict[next_lane] = Lane(next_lane_type, poses)
                    lane_set_points = poses if not len(
                        lane_set_points) else np.concatenate((lane_set_points, poses))
                else:
                    next_lane = lane_dict[lane_id].get_attribute(
                        'next_lane_id')
                    poses = lane_dict[lane_id].get_attribute('lane_set_points')
                    lane_set_points = poses if not len(
                        lane_set_points) else np.concatenate((lane_set_points, poses))

                if len(lane_set_points) >= sample_num:
                    lane_set_points = lane_set_points[:sample_num]

                lane_id = next_lane
        
        feature_lanes.append(lane_set_points-cur_pos)
        
    return np.array(feature_lanes)


def get_each_timestamp_lane(nusc_map, lane_dict, agent_points, radius, angle_threshold=30, stop_threshold=0.5, resolution_meters=5, sample_num=10):
    '''
    To get each frame's lane and the groundtruth lane 
    (gt : 0,1,2,3) 0-> no lane 1 -> right 2 -> middle 3 -> left
    :param nusc_map: nuScenes API
    :param agent_points: each frame position.
    :param radius: The interest range to consider the lane.
    :param threshold: The direction angle threshold
    :return 3D array (Timestamp, Lanes, ID)
    '''

    total_lanes_set_point = list()
    total_lane_boolean = list()
    dummy_input = np.zeros((3,sample_num, 2))
    length_t = len(agent_points)
    try:
    # print("agent_points : \n", agent_points)
        for i in range(len(agent_points)):
            lane_exist = True
            # print("current index : ", i)
            if i == 0:
                total_lanes_set_point.append(dummy_input)
                total_lane_boolean.append(False)
                continue
            # i -> current pos, i-1 previous pos
            agent_class = agent(agent_points[i], agent_points[i - 1])
            lanes_id = get_around_lane(
                nusc_map, lane_dict, agent_class, radius=radius, resolution_meters=0.1)
            lanes_id = get_same_direction_lanes(
                lane_dict, lanes_id, agent_class, angle_threshold=angle_threshold, stop_threshold=stop_threshold)
            lanes_id = get_future_lanes(lane_dict, lanes_id, agent_class)
            lanes_id = get_nearest_lane(
                nusc_map, lane_dict, lanes_id, agent_class)
            total_lanes_set_point.append(get_future_80m_lane(
                nusc_map, agent_points[i],lane_dict, lanes_id, resolution_meters, sample_num))
            total_lane_boolean.append(lane_exist)
    except other_error as error:
        # print(error)
        return False, np.zeros((length_t, 3, sample_num, 2)),np.zeros((length_t),dtype=bool)
    except:
        # print("Other exceptions!")
        return False, np.zeros((length_t, 3, sample_num, 2)),np.zeros((length_t),dtype=bool)

    return True, np.array(total_lanes_set_point), np.array(total_lane_boolean)

# if name == '__main__' :

#     dataroot = 'data/sets/nuscenes'
#     map_name = 'singapore-onenorth'
#     my_patch = (440, 1320, 530, 1400)
#     nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
#     bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
#     fig, ax = nusc_map.render_map_patch(my_patch, nusc_map.non_geometric_layers, figsize=(10, 10), bitmap=bitmap)

#     point = np.array([[492,1368],[490,1365]])
#     radius = 4
#     threshold = 20
#     color = ['r','g','c','m','y','k','gold','lime']

#     lanes = get_around_lane(point[0][0], point[0][1], radius=1, nusc_map)
#     print("get_around_lane : \n",lanes)
#     lanes = get_same_direction_lanes(lanes, point, threshold)
#     print("get_same_direction_lanes : \n",lanes)
#     lanes = get_future_lanes(lanes, point, )
#     print("get_future_lanes : \n",lanes)
#     lanes = get_nearest_lane(lanes, point)
#     print("get_nearest_lane : \n",lanes)
#     future_nearest_lanes = find_future_80m_lane(lanes, nusc_map)

#     for i, feature_lane in enumerate(future_nearest_lanes):
#         clr = color[i]
#         for j, feature_lane_point in enumerate(feature_lane):
#             if not j :
#                 ax.plot(feature_lane_point[0], feature_lane_point[1],'bo', markersize=3) # start_pose
#             else:
#                 ax.plot(feature_lane_point[0], feature_lane_point[1], c=clr, marker='o', markersize=2) # remain_pose

#     ax.plot(point[1][0], point[1][1],'co') #青色實心(前一幀)
#     ax.plot(point[0][0], point[0][1],'wo') #白色實心(現在)
