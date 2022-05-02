import csv
import os
import math
time_len = 10  # unit is second, CSV time length
mode = 'Sfunction'  # function mode, 'poly2' 'Sfunction'
curve_xy_axis = 'x'  # 'x' or 'y', it determines curve direction.
curve_coef = 0.03  # 0.01 to 1, it is coefficient of curve function.
curve_sign = +1  # +1 or -1
points_direction_on_curve = +1  # +1 or -1

speed = 10  # unit is m/s
fps = 10  # frames per second
path = "../" + str(fps) + "fps_csv"  # csv restore path.
init_x = 0  # x coordinat of start point
init_y = 0  # y coordinat of start point
# init_heading = 0 # heading of start point
precision = 0.001  # see below code


def generate_sfunction_coordinate():
    x_list, y_list, heading_list = [
        0] * (time_len * fps + 1), [0] * (time_len * fps + 1), [0] * (time_len * fps + 1)

    distance_per_frame = speed / fps
    cnt = 0
    x_list[0] = init_x
    y_list[0] = init_y
    heading_list[0] = 0  # init_heading
    cnt += 1
    x = 0
    y = 0
    next_x = 0
    next_y = 0
    while cnt < (time_len * fps + 1):
        next_x += precision
        next_y = 10 * math.sin(0.14 * next_x)
        if (next_x - x)**2 + (next_y - y)**2 >= distance_per_frame**2:
            x_list[cnt] = next_x + init_x
            y_list[cnt] = next_y + init_y
            heading_list[cnt] = math.atan2(next_y - y, next_x - x)
            x = next_x
            y = next_y
            cnt += 1
    return x_list, y_list, heading_list


def generate_poly2_coordinate():
    x_list, y_list, heading_list = [
        0] * (time_len * fps + 1), [0] * (time_len * fps + 1), [0] * (time_len * fps + 1)

    distance_per_frame = speed / fps
    cnt = 0
    x_list[0] = init_x
    y_list[0] = init_y
    heading_list[0] = 0  # init_heading
    cnt += 1
    x = 0
    y = 0
    next_x = 0
    next_y = 0
    while cnt < (time_len * fps + 1):
        if curve_xy_axis == 'y':
            next_x += points_direction_on_curve * precision
            next_y = curve_sign * curve_coef * next_x * next_x
        elif curve_xy_axis == 'x':
            next_y += points_direction_on_curve * precision
            next_x = curve_sign * curve_coef * next_y * next_y
        if (next_x - x)**2 + (next_y - y)**2 >= distance_per_frame**2:
            x_list[cnt] = next_x + init_x
            y_list[cnt] = next_y + init_y
            heading_list[cnt] = math.atan2(next_y - y, next_x - x)
            x = next_x
            y = next_y
            cnt += 1
    return x_list, y_list, heading_list


def generate_table():
    table = [[0] * 15 for i in range(time_len * fps + 2)]
    table[0] = [
        'frame_id',
        'type',
        'node_id',
        'dt',
        'x',
        'y',
        'z',
        'length',
        'width',
        'height',
        'heading',
        'PP_x',
        'PP_y',
        'self_x',
        'self_y']
    for i in range(1, len(table)):
        table[i][0] = i
    for i in range(1, len(table)):
        table[i][1] = 'VEHICLE'
    # for i in range(1, len(table)):
    #     table[i][2] = 1
    for i in range(1, len(table)):
        table[i][3] = 0.1
    if mode == 'poly2':
        x_list, y_list, heading_list = generate_poly2_coordinate()
    elif mode == 'Sfunction':
        x_list, y_list, heading_list = generate_sfunction_coordinate()
    for i in range(1, len(table)):
        table[i][4] = x_list[i - 1]
    for i in range(1, len(table)):
        table[i][5] = y_list[i - 1]
    for i in range(1, len(table)):
        table[i][10] = heading_list[i - 1]
    return table


with open(os.path.join(path, 'fake_testing_trajectory.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    table = generate_table()
    writer.writerows(table)
