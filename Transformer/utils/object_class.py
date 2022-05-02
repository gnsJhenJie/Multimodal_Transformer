import numpy as np
from numpy import linalg as LA


class Lane():
    def __init__(self, lane_type, lane_set_points):
        self.next_lane_id = None
        self.lane_type = lane_type
        self.lane_set_points = lane_set_points
        self.length = len(lane_set_points) - 1

    def set_next_lane_id(self, next_lane_id):
        self.next_lane_id = next_lane_id

    def get_nearest_point_index(self, cur_point):
        # using norm1 to get the closest index of lane set points
        # lane set points subtract current pos
        delta_pos = self.lane_set_points - cur_point
        self.nearest_point_index = np.argmin(LA.norm(delta_pos, axis=-1))
        self.nearest_point_index = self.nearest_point_index + \
            1 if self.nearest_point_index == 0 else self.nearest_point_index

    def get_lane_vector(self, cur_point):
        # if the closest index is last one, get the previous index
        if self.nearest_point_index == self.length:
            next_pt = self.nearest_point_index
            self.nearest_point_index = self.nearest_point_index - 1
        else:
            next_pt = self.nearest_point_index + 1
        # avoid repeat lane set points
        while LA.norm(self.lane_set_points[next_pt] -
                      self.lane_set_points[self.nearest_point_index], axis=-
                      1) <= 1e-5:
            next_pt = next_pt + 1
        # Get lane vector
        self.lane_vector = self.lane_set_points[next_pt] - \
            self.lane_set_points[self.nearest_point_index]

    def calc_nearest_dist(self, cur_point):
        delta_pos = self.lane_set_points[self.nearest_point_index] - cur_point
        return LA.norm(delta_pos)

    def get_attribute(self, attribute_name):
        for attribute in self.__dict__.keys():
            if attribute == attribute_name:
                value = getattr(self, attribute)
                return value
        print("attribute_name : ", attribute_name, " not found !")


class agent():
    def __init__(self, cur_pos, past_pos):
        self.cur_pos = cur_pos
        self.past_pos = past_pos
        self.agent_vector = self.cur_pos - self.past_pos

    def get_line(self, lanes_dict, cur_lanes_id, min_distance):
        closest_id = self.get_closest_lane(lanes_dict, cur_lanes_id)
        nearest_point_index = lanes_dict[closest_id].get_attribute(
            'nearest_point_index')
        lane_pose = lanes_dict[closest_id].get_attribute('lane_set_points')
        agent_distance = LA.norm(self.agent_vector, axis=-1)
        # if agent stop
        if agent_distance <= min_distance:
            # print('agent stop!')
            # print('agent dis to last frame :',agent_distance)
            self.cur_pos = lane_pose[nearest_point_index]
            self.past_pos = lane_pose[nearest_point_index - 1]
            self.agent_vector = self.cur_pos - self.past_pos
        # else:
        #     print('agent moving!')
        #     print('agent dis to last frame :',agent_distance)

        self.parallel_slope, self.parallel_bias = self.get_parallel_line()
        self.orthogonal_slope, self.orthogonal_bias = self.get_orthogonal_line()

    def get_closest_lane(self, lanes_dict, cur_lanes_id):
        minimum = 1e10
        closest_lane = None
        for lane_id in cur_lanes_id:
            distance = lanes_dict[lane_id].calc_nearest_dist(self.cur_pos)
            if distance < minimum:
                minimum = distance
                lane_type = lanes_dict[lane_id].get_attribute('lane_type')
                closest_lane = lane_id

        return closest_lane

    def get_parallel_line(self):
        # using 2 point to get a line in 2d plane(y = ax+b)
        a = 1e-10 if self.agent_vector[0] == 0 else self.agent_vector[1] / \
            self.agent_vector[0]
        a = 0 if self.agent_vector[1] == 0 else a
        b = self.cur_pos[1] - a * self.cur_pos[0]
        return a, b

    def get_orthogonal_line(self):
        # using 2 point to get a line in 2d plane(y = ax+b)
        # two orthogonal line the product of slopes will be -1
        # -1.0*agent_vector[0]/agent_vector[1] orthogonal to agent direction
        if self.parallel_slope == 1e-10:
            a = 0
        elif self.parallel_slope == 0:
            a = 1e-10
        else:
            a = -1.0 / self.parallel_slope
        b = self.cur_pos[1] - a * self.cur_pos[0]
        return a, b

    def get_attribute(self, attribute_name):
        for attribute in self.__dict__.keys():
            if attribute == attribute_name:
                value = getattr(self, attribute)
                return value
        print("attribute_name : ", attribute_name, " not found !")


class other_error(Exception):
    pass
