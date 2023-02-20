import pandas as pd
import numpy as np
import math
from utils.signal_name_handler import LaneSide


class Pose:
    def __init__(self):

        self.x_pos = 0.0
        self.y_pos = 0.0
        self.theta = 0.0

    def append_(self, append_term):
        return_value = Pose()
        return_value.x_pos = self.x_pos + append_term.x_pos * math.cos(self.theta) - append_term.y_pos * math.sin(
            self.theta)
        return_value.y_pos = self.y_pos + append_term.x_pos * math.sin(self.theta) + append_term.y_pos * math.cos(
            self.theta)
        return_value.theta = self.theta_calibrator(self.theta + append_term.theta)

        return return_value

    def theta_calibrator(self, theta):
        sv_pi = math.atan(1.0) * 4.0
        sv_pi_2 = math.atan(1.0) * 8.0
        while -sv_pi >= theta or sv_pi < theta:
            if theta >= sv_pi:
                theta -= sv_pi_2
            else:
                theta += sv_pi_2

        return theta


class GroundTruth:
    def __init__(self, playlist, output_path):
        self.dir_name = playlist.recording_name
        self.output_path = output_path

    def generate_gt(self, dgps_df, me_df, sv_df):

        max_lane_change_lateral_thres = 2.5  # 2.5m
        min_lane_change_lateral_thres = 0.8  # 0.8m
        min_ego_speed = 2.7  # 10kph, 2.7m/s

        kinematic_pose_list = []

        for idx in range(1, len(dgps_df.index)):
            max_longitudinal_distance = max(50, sv_df.at[idx, 'SV.Host.LH.ViewRangeEnd'], sv_df.at[idx, 'SV.Host.RH.ViewRangeEnd']) #max(50, me_df.at[idx, 'ME.Host.LH.ViewRangeEnd'], me_df.at[idx, 'ME.Host.RH.ViewRangeEnd']) #60 #
            travel_distance = 0.0
            temp = {'x_pos': [], 'y_pos':[], 'theta':[]}
            curr_pose = Pose()
            # print(idx)
            for travel_idx in range(idx, len(dgps_df.index)):
                ego_speed = dgps_df.at[travel_idx - 1, 'VDY.EgoSpeed']  # [m/s]  #dgps_df.at[travel_idx - 1, 'WHL_SPD_FL'] / 3.6  #
                current_timestamp = dgps_df.at[travel_idx, 'DGPS.Timestamp']  # should be taken from DGPS dgps_df.at[travel_idx, 'CAN.Timestamp']  #
                prev_timestamp =  dgps_df.at[travel_idx - 1, 'DGPS.Timestamp'] #dgps_df.at[travel_idx - 1, 'CAN.Timestamp']  #
                dt = current_timestamp - prev_timestamp  # [s]
                yaw_rate = dgps_df.at[travel_idx - 1, 'AngRateZ'] #dgps_df.at[travel_idx - 1, 'YAW_RATE']  #
                if np.isnan(yaw_rate) or (yaw_rate == -1):
                    break
                update_pose = Pose()
                update_pose.theta =-math.radians(yaw_rate * dt) # math.radians(yaw_rate * dt)#

                update_pose.x_pos = ego_speed * dt * np.cos(update_pose.theta)
                update_pose.y_pos = ego_speed * dt * np.sin(update_pose.theta)

                prev_pose = curr_pose
                curr_pose = prev_pose.append_(update_pose)

                travel_distance += np.sqrt(
                    math.pow(curr_pose.x_pos - prev_pose.x_pos, 2) + math.pow(curr_pose.y_pos - prev_pose.y_pos, 2))

                # if longitudinal distance is over 50m or c0 is over 2.5m, set to invalid value to avoid lane change.
                if (travel_distance > max_longitudinal_distance) \
                        | ((np.abs(sv_df.at[travel_idx, 'SV.Host.LH.C0']) > max_lane_change_lateral_thres)
                           & (np.abs(sv_df.at[travel_idx, 'SV.Host.RH.C0']) < min_lane_change_lateral_thres))\
                        | ((np.abs(sv_df.at[travel_idx, 'SV.Host.RH.C0']) > max_lane_change_lateral_thres)
                           &(np.abs(sv_df.at[travel_idx, 'SV.Host.LH.C0']) < min_lane_change_lateral_thres)):  #| (ego_speed < min_ego_speed):
                    break
                else:
                    temp['x_pos'].append(curr_pose.x_pos)
                    temp['y_pos'].append(curr_pose.y_pos)
                    temp['theta'].append(curr_pose.theta)
            kinematic_pose_list.append(temp)

        gt_df = self.get_gt_polynomial(kinematic_pose_list, me_df, sv_df)

        # if not os.path.exists(os.path.join(self.output_path, self.dir_name)):
        #     os.makedirs(os.path.join(self.output_path, self.dir_name))
        # gt_df.to_csv(os.path.join(self.output_path, self.dir_name, self.dir_name + '_gt_sv.csv'))

        return gt_df

    def get_gt_polynomial(self, kinematic_pose_list, me_df, sv_df):
        """
        Generate coefficient values of ground truth for each of frames
        :param df: SV LD data
        :param kinematic_pose_list: x_pos, y_pos, theta contain
        :return: coefficient values of ground truth
        """

        gt_df = pd.DataFrame({'GT.Host.LH.C0': [],
                              'GT.Host.LH.C1': [],
                              'GT.Host.LH.C2': [],
                              'GT.Host.LH.C3': [],
                              'GT.Host.LH.ViewRangeEnd': [],
                              'GT.Host.RH.C0': [],
                              'GT.Host.RH.C1': [],
                              'GT.Host.RH.C2': [],
                              'GT.Host.RH.C3': [],
                              'GT.Host.RH.ViewRangeEnd': []
                              })
        invalid_value = -1
        min_rank = 5
        high_quality = 2
        c0_temp = []
        c1_temp = []
        c2_temp = []
        c3_temp = []
        view_end_temp = []
        view_start_temp = []
        quality_temp = []
        for side in LaneSide:
            for frame_idx in range(len(kinematic_pose_list)):
                if ((len(kinematic_pose_list[frame_idx]['theta']) < min_rank)
                    & (len(kinematic_pose_list[frame_idx]['x_pos']) < min_rank)
                    & (len(kinematic_pose_list[frame_idx]['y_pos']) < min_rank))\
                    | (sv_df.at[frame_idx, 'SV.Host.{}.Quality'.format(side.name)] < high_quality):
                         # Avoid RankWarning of polyfit (me_df.at[frame_idx, 'ME.Host.{}.Quality'.format(side.name)] < 2)\| (len(kinematic_pose_list[frame_idx]['x_pos']) < min_rank):
                    c0_temp.append(invalid_value)
                    c1_temp.append(invalid_value)
                    c2_temp.append(invalid_value)
                    c3_temp.append(invalid_value)
                    view_end_temp.append(invalid_value)
                    view_start_temp.append(invalid_value)
                    quality_temp.append(invalid_value)
                else:
                    x_pos = np.array(kinematic_pose_list[frame_idx]['x_pos'])
                    y_pos = np.array(kinematic_pose_list[frame_idx]['y_pos'])
                    theta = np.array(kinematic_pose_list[frame_idx]['theta'])
                    c0 = np.abs(me_df.at[frame_idx, 'ME.Host.{}.C0'.format(side.name)])
                    c0 = np.abs(sv_df.at[frame_idx, 'SV.Host.{}.C0'.format(side.name)])
                    if side.name == 'LH':
                        new_x_pos = c0 * np.sin(theta) + x_pos
                        new_y_pos = c0 * np.cos(theta) + y_pos
                    else:
                        new_x_pos = -c0 * np.sin(theta) + x_pos
                        new_y_pos = -c0 * np.cos(theta) + y_pos
                    linear_model = np.polyfit(new_x_pos, new_y_pos, 3)
                    linear_model = np.poly1d(linear_model)
                    c0_temp.append(linear_model.coefficients[3])
                    c1_temp.append(linear_model.coefficients[2])
                    c2_temp.append(linear_model.coefficients[1])
                    c3_temp.append(linear_model.coefficients[0])
                    view_end_temp.append(max(kinematic_pose_list[frame_idx]['x_pos']))
                    view_start_temp.append(0)
                    quality_temp.append(3)
            gt_df.loc[:, 'GT.Host.{}.C0'.format(side.name)] = c0_temp
            gt_df.loc[:, 'GT.Host.{}.C1'.format(side.name)] = c1_temp
            gt_df.loc[:, 'GT.Host.{}.C2'.format(side.name)] = c2_temp
            gt_df.loc[:, 'GT.Host.{}.C3'.format(side.name)] = c3_temp
            gt_df.loc[:, 'GT.Host.{}.ViewRangeEnd'.format(side.name)] = view_end_temp
            gt_df.loc[:, 'GT.Host.{}.ViewRangeStart'.format(side.name)] = view_start_temp
            gt_df.loc[:, 'GT.Host.{}.Quality'.format(side.name)] = quality_temp
            c0_temp.clear()
            c1_temp.clear()
            c2_temp.clear()
            c3_temp.clear()
            view_end_temp.clear()
            view_start_temp.clear()
            quality_temp.clear()

        # self.debug_vis(sv_log_df,gt_df, me_df)

        return gt_df

    #  on top view
    def debug_vis(self, df, gt_df, me_data):
        import utils.polynomial as polynomial
        import plotly.express as px
        sv_color = 'blue'
        gt_color = 'green'
        me_color = 'tomato'
        # Visu for Left window
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import plotly.graph_objs as go
        for frame_idx in range(len(df)):
            if (frame_idx > 100) & (frame_idx < 105):
                fig = make_subplots(rows=4, cols=2,
                                    shared_xaxes=True,
                                    vertical_spacing=0.03,
                                    horizontal_spacing=0.03,
                                    specs=[[{"type": "scatter"}, {"type": "scatter", 'rowspan':4}],
                                           [{}, None],
                                           [{}, None],
                                           [{}, None], ],
                                    column_widths=[0.8,0.2],
                                    )
                fig.add_trace(go.Scatter(
                    line=dict(color=sv_color, width=6),
                    y=np.arange(0, 50),
                    x=polynomial.polynomial(np.arange(0, 50),
                                            df.at[frame_idx, 'SV.Host.LH.C0'], df.at[frame_idx, 'SV.Host.LH.C1'],
                                            df.at[frame_idx, 'SV.Host.LH.C2'], df.at[frame_idx, 'SV.Host.LH.C3']), name='SV'),
                    row=1, col=2
                )
                fig.add_trace(go.Scatter(
                    line=dict(color=sv_color, width=6),
                    y=np.arange(0, 50),
                    x=polynomial.polynomial(np.arange(0, 50),
                                            df.at[frame_idx, 'SV.Host.RH.C0'], df.at[frame_idx, 'SV.Host.RH.C1'],
                                            df.at[frame_idx, 'SV.Host.RH.C2'], df.at[frame_idx, 'SV.Host.RH.C3']), showlegend=False),
                    row=1, col=2
                )

                fig.add_trace(go.Scatter(
                    line=dict(color=me_color, width=6),
                    y=np.arange(0, 50),
                    x=polynomial.polynomial(np.arange(0, 50),
                                            gt_df.at[frame_idx, 'GT.Host.LH.C0'], gt_df.at[frame_idx, 'GT.Host.LH.C1'],
                                            gt_df.at[frame_idx, 'GT.Host.LH.C2'], gt_df.at[frame_idx, 'GT.Host.LH.C3']), name='GT'),
                    row=1, col=2
                )
                fig.add_trace(go.Scatter(
                    line=dict(color=me_color, width=6),
                    y=np.arange(0, 50),
                    x=polynomial.polynomial(np.arange(0, 50),
                                            gt_df.at[frame_idx, 'GT.Host.RH.C0'], gt_df.at[frame_idx, 'GT.Host.RH.C1'],
                                            gt_df.at[frame_idx, 'GT.Host.RH.C2'], gt_df.at[frame_idx, 'GT.Host.RH.C3']), showlegend=False),
                    row=1, col=2
                )

                fig.add_trace(trace=go.Scatter(x=df.index, y=df['SV.Host.LH.C0'], line=dict(color=sv_color),
                                               name='SV'), row=1, col=1)

                fig.add_trace(trace=go.Scatter(x=df.index, y=df['SV.Host.RH.C0'], line=dict(color=sv_color),
                                               name='SV'), row=1, col=1)
                fig.add_trace(trace=go.Scatter(x=df.index, y=df['SV.Host.LH.C1'], line=dict(color=sv_color),
                                               showlegend=False), row=2, col=1)
                fig.add_trace(trace=go.Scatter(x=df.index, y=df['SV.Host.LH.C2'], line=dict(color=sv_color),
                                               showlegend=False), row=3, col=1)
                fig.add_trace(trace=go.Scatter(x=df.index, y=df['SV.Host.LH.C3'], line=dict(color=sv_color),
                                               showlegend=False), row=4, col=1)

                for frame_idx, side in enumerate(['LH']):
                    fig.add_trace(trace=go.Scatter(x=gt_df.index, y=gt_df['GT.Host.{}.C0'.format(side)], line=dict(color=gt_color),
                                                   name='SV'), row=1, col=1)
                    fig.add_trace(trace=go.Scatter(x=gt_df.index, y=gt_df['GT.Host.{}.C1'.format(side)], line=dict(color=gt_color),
                                                   showlegend=False), row=2, col=1)
                    fig.add_trace(trace=go.Scatter(x=gt_df.index, y=gt_df['GT.Host.{}.C2'.format(side)], line=dict(color=gt_color),
                                                   showlegend=False), row=3, col=1)
                    fig.add_trace(trace=go.Scatter(x=gt_df.index, y=gt_df['GT.Host.{}.C3'.format(side)], line=dict(color=gt_color),
                                                   showlegend=False), row=4, col=1)

                if 'ME.Host.LH.C0' in me_data.columns:
                    fig.add_trace(trace=go.Scatter(x=me_data.index, y=me_data['ME.Host.LH.C0'], line=dict(color=me_color),
                                                   name='MEyeQ4'), row=1, col=1)
                    fig.add_trace(trace=go.Scatter(x=me_data.index, y=me_data['ME.Host.LH.C1'], line=dict(color=me_color),
                                                   showlegend=False), row=2, col=1)
                    fig.add_trace(trace=go.Scatter(x=me_data.index, y=me_data['ME.Host.LH.C2'], line=dict(color=me_color),
                                                   showlegend=False), row=3, col=1)
                    fig.add_trace(trace=go.Scatter(x=me_data.index, y=me_data['ME.Host.LH.C3'], line=dict(color=me_color),
                                                   showlegend=False), row=4, col=1)


                fig.show()
