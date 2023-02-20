import numpy as np
import pandas as pd
import os
from utils.logger import logger
from pandas.core.common import SettingWithCopyWarning

import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


class Arranger:
    def __init__(self, dir_name, argument):
        self.dir_name = dir_name
        self.output_path = argument.output_path
        self.is_adaf_db = argument.is_adaf_db
        self.is_next_line_plot = argument.is_next_line_plot

    def rearrangement(self, file_path):

        if 'lda_tiny_log.csv' in file_path:
            data = self.rearrangement_log(file_path)
        elif 'adaf_log.csv' in file_path:
            data = self.rearrangement_cpp(file_path)
        elif 'me.csv' in file_path or 'pcan.csv' in file_path:
            data = self.rearrangement_mobileye(file_path)
        elif 'can.csv' in file_path or 'ccan.csv' in file_path:
            data = self.rearrangement_can(file_path)
        elif 'dgps.csv' in file_path:
            data = self.rearrangement_dgps(file_path)

        return data

    def rearrangement_log(self, log_file_path):

        logger.debug(log_file_path)

        origin_log_df = pd.read_csv(log_file_path, index_col=False, low_memory=False, error_bad_lines=False)
        origin_log_df.drop(
            labels=['rb_color', 'rb_position', 'rb_type', 'rb_trackingID', 'rb_enable', 'rb_multiple',
                    'rb_roadBoundaryType', 'rb_vcs_start', 'rb_vcs_end', 'rb_vcs_predicted_start',
                    'rb_vcs_predicted_end', 'rb_lineWidthModel_C0', 'rb_lineWidthModel_C1', 'rb_segment_num',
                    'rb_segment_start', 'rb_segment_end', 'rb_segment_C0', 'rb_segment_C1',
                    'rb_segment_C2', 'rb_segment_C3', 'drv_lines_num', 'drv_segment_start',
                    'drv_segment_end','rb_lines_num', 'num_of_merged_point', 'merge_pos_lon', 'num_of_branch_point',
                    # 'branch_pos_lon'
                    ], axis=1, inplace=True) #'rb_lines_num', 'num_of_merged_point', 'merge_pos_lon', 'num_of_branch_point',

        # Get host left line with condition that drv_position is 3.0 And drv_multiple is 1.0
        host_left_df = origin_log_df.loc[
            (origin_log_df['drv_position'] == 3.0) & (origin_log_df['drv_multiple'] == 1.0)]
        host_left_df = self.arrange(origin_log_df, host_left_df, 'Host.LH')
        # Get host right line with condition that drv_position is 4.0 And drv_multiple is 1.0
        host_right_df = origin_log_df.loc[
            (origin_log_df['drv_position'] == 4.0) & (origin_log_df['drv_multiple'] == 1.0)]
        host_right_df = self.arrange(origin_log_df, host_right_df, 'Host.RH')

        if self.is_next_line_plot:
            next_left_df = origin_log_df.loc[
                (origin_log_df['drv_position'] == 2.0) & (origin_log_df['drv_multiple'] == 1.0)]
            next_left_df = self.arrange(origin_log_df, next_left_df, 'Next.LH')
            # Get host right line with condition that drv_position is 4.0 And drv_multiple is 1.0
            next_right_df = origin_log_df.loc[
                (origin_log_df['drv_position'] == 5.0) & (origin_log_df['drv_multiple'] == 1.0)]
            next_right_df = self.arrange(origin_log_df, next_right_df, 'Next.RH')
        else:
            next_left_df = pd.DataFrame()
            next_right_df = pd.DataFrame()

        df = pd.concat(
            [host_left_df, host_right_df, next_left_df, next_right_df],
            axis=1)

        # SV Quality
        if 'SV.Host.LH.Confidence' in df.columns:
            sv_lh_confidence = df.loc[:, 'SV.Host.LH.Confidence'].to_numpy()
            sv_rh_confidence = df.loc[:, 'SV.Host.RH.Confidence'].to_numpy()
            df.loc[:, 'SV.Host.LH.Quality'] = np.where(sv_lh_confidence > 0.77, 3, 0)
            df.loc[:, 'SV.Host.RH.Quality'] = np.where(sv_rh_confidence > 0.77, 3, 0)
        else:
            sv_lh_c0 = df.loc[:, 'SV.Host.LH.C0'].to_numpy()
            sv_rh_c0 = df.loc[:, 'SV.Host.RH.C0'].to_numpy()
            df.loc[:, 'SV.Host.LH.Confidence'] = np.where(sv_lh_c0 == -1, -1, 1)
            df.loc[:, 'SV.Host.RH.Confidence'] = np.where(sv_rh_c0 == -1, -1, 1)
            df.loc[:, 'SV.Host.LH.Quality'] = np.where(sv_lh_c0 == -1, -1, 3)
            df.loc[:, 'SV.Host.RH.Quality'] = np.where(sv_rh_c0 == -1, -1, 3)

        # if not os.path.exists(os.path.join(self.output_path, self.dir_name)):
        #     os.makedirs(os.path.join(self.output_path, self.dir_name))

        # df.to_csv(os.path.join(self.output_path, self.dir_name, self.dir_name + '_log_rearranged.csv'))
        return df

    def arrange(self, origin_df, df, str):

        # origin_df.reset_index(inplace=True, drop=False)
        df.reset_index(inplace=True, drop=False)
        empty = pd.DataFrame(index=range(0, 1), columns=df.columns)
        empty.fillna(-1, inplace=True)

        for i in range(int(origin_df['frame_id'].max()) + 1):
            if i < len(df):
                if i == int(df.at[i, 'frame_id']):
                    pass
                elif i != int(df.at[i, 'frame_id']):
                    temp1 = df.iloc[df.index < i, :]
                    temp2 = df.iloc[df.index >= i, :]
                    df = temp1.append(empty, ignore_index=True).append(temp2, ignore_index=True)
                    df.at[i, 'frame_id'] = i
                    df.at[i, 'ocPitch'] = origin_df.at[origin_df.at[i, 'frame_id'], 'ocPitch']
                    df.iloc[i, 3:] = -1
            else:
                temp1 = df.iloc[df.index < i, :]
                temp2 = df.iloc[df.index >= i, :]
                df = temp1.append(empty, ignore_index=True).append(temp2, ignore_index=True)
                df.at[i, 'frame_id'] = i
                df.at[i, 'ocPitch'] = origin_df.at[origin_df.at[i, 'frame_id'], 'ocPitch']
                df.iloc[i, 3:] = -1

        df.reset_index(inplace=True, drop=True)

        df.rename(columns={'drv_color': 'SV.{}.drv_color'.format(str), 'drv_position': 'SV.{}.drv_position'.format(str),
                           'drv_type': 'SV.{}.drv_type'.format(str),
                           'drv_trackingID': 'SV.{}.drv_trackingID'.format(str),
                           'drv_enable': 'SV.{}.drv_enable'.format(str),
                           'drv_multiple': 'SV.{}.drv_multiple'.format(str),
                           'drv_roadBoundaryType': 'SV.{}.drv_roadBoundaryType'.format(str),
                           'drv_lineWidthModel_C0': 'SV.{}.drv_lineWidthModel_C0'.format(str),
                           'drv_lineWidthModel_C1': 'SV.{}.drv_lineWidthModel_C1'.format(str),
                           'drv_segment_num': 'SV.{}.drv_segment_num'.format(str),

                           'drv_vcs_start': 'SV.{}.Start'.format(str),
                           'drv_vcs_end': 'SV.{}.End'.format(str),
                           'drv_vcs_predicted_start': 'SV.{}.ViewRangeStart'.format(str),
                           'drv_vcs_predicted_end': 'SV.{}.ViewRangeEnd'.format(str),

                           'ocPitch': 'oc.Pitch',

                           'drv_segment_C0': 'SV.{}.C0'.format(str),
                           'drv_segment_C1': 'SV.{}.C1'.format(str),
                           'drv_segment_C2': 'SV.{}.C2'.format(str),
                           'drv_segment_C3': 'SV.{}.C3'.format(str),
                           'confidence': 'SV.{}.Confidence'.format(str),
                           'confidence ': 'SV.{}.Confidence'.format(str)
                           }, inplace=True)

        return df

    def rearrangement_cpp(self, log_file_path):

        logger.debug(log_file_path)
        origin_log_df = pd.read_csv(log_file_path, index_col=False, low_memory=False, error_bad_lines=False)
        origin_log_df.drop(
            labels=['obj_id', 'obj_type', 'obj_confidence', 'BB_x1', 'BB_y1', 'BB_x2', 'BB_y2', 'far_BB_x1',
                    'far_BB_x2',
                    'far_BB_x3', 'far_BB_x4', 'far_BB_y1', 'far_BB_y2', 'far_BB_y3', 'far_BB_y4', 'near_BB_x1',
                    'near_BB_x2',
                    'near_BB_x3', 'near_BB_x4', 'near_BB_y1', 'near_BB_y2', 'near_BB_y3', 'near_BB_y4', 'long_dist',
                    'lat_dist', 'dist_validity', 'rel_long_vel', 'rel_lat_vel', 'lane_assignment', 'lane_change',
                    'motion_status',
                    'motion_category', 'motion_orientation', 'cipv_id', 'cipv_tracked', 'cipv_lost', 'cinvl_id',
                    'cinvl_tracked',
                    'cinvr_id', 'cinvr_tracked', 'heading_angle', 'tracking_age', 'ttc', 'opi_headingAngle_status'],
            axis=1, inplace=True)
        df = origin_log_df.drop_duplicates(['frame_id'])
        df.reset_index(inplace=True, drop=False)

        empty = pd.DataFrame(index=range(0, 1), columns=origin_log_df.columns)
        empty.fillna(-1, inplace=True)
        for i in range(int(origin_log_df['frame_id'].max()) + 1):
            if i < len(df):
                if (i == int(df.at[i, 'frame_id'])):
                    pass
                elif (i != int(df.at[i, 'frame_id'])):
                    temp1 = df.iloc[df.index < i, :]
                    temp2 = df.iloc[df.index >= i, :]
                    df = temp1.append(empty, ignore_index=True).append(temp2, ignore_index=True)
                    df.at[i, 'frame_id'] = i

                    df.iloc[i, 3:] = -1
            else:
                temp1 = df.iloc[df.index < i, :]
                temp2 = df.iloc[df.index >= i, :]
                df = temp1.append(empty, ignore_index=True).append(temp2, ignore_index=True)
                df.at[i, 'frame_id'] = i

                df.iloc[i, 3:] = -1

        df.reset_index(inplace=True, drop=True)

        df.rename(columns={'cpp_ego_lane_confidence': 'CPP.Host.Confidence',
                           'cpp_ego_lane_view_range': 'CPP.Host.ViewRangeEnd',
                           'cpp_ego_lane_coefficient_0': 'CPP.Host.C0',
                           'cpp_ego_lane_coefficient_1': 'CPP.Host.C1',
                           'cpp_ego_lane_coefficient_2': 'CPP.Host.C2',
                           'cpp_ego_lane_coefficient_3': 'CPP.Host.C3'}, inplace=True)
        df.fillna(-1, inplace=True)
        # df.to_csv(os.path.join(self.output_path, self.dir_name, self.dir_name + '_cpp_rearranged.csv'))

        return df

    def rearrangement_mobileye(self, file_path):

        df = pd.read_csv(file_path, index_col=False, header=1, low_memory=False)
        if self.is_adaf_db == True:

            df = pd.concat([df.iloc[:, 0:8], df.iloc[:, 8:-1].fillna(method='ffill')], axis=1)

            df = df.dropna(subset=['CAM_FC'])
            df.reset_index(drop=True, inplace=True)
            df.drop(['CAM_FC', 'CAM_SR', 'CAM_SL', 'CAM_SVM_F', 'CAM_SVM_R', 'CAM_SVM_L', 'CAM_SVM_B'],
                          axis=1, inplace=True)

        elif self.is_adaf_db == False:

            df = pd.concat([df.iloc[:, 0:21], df.iloc[:, 21:-1].fillna(method='ffill')], axis=1)

            df = df.dropna(subset=['CAM_1'])
            df.reset_index(drop=True, inplace=True)
            df.drop(['CAM_{}'.format(i) for i in range(1, 21)], axis=1, inplace=True)

        df.rename(columns={'Time': 'ME.Timestamp', 'LaneMarkModelA_C2_Lh_ME': 'ME.Host.LH.C2',
                               'LaneMarkPosition_C0_Lh_ME': 'ME.Host.LH.C0',
                               'LaneMarkHeadingAngle_C1_Lh_ME': 'ME.Host.LH.C1',
                               'LaneMarkModelDerivA_C3_Lh_ME': 'ME.Host.LH.C3', 'LaneMarkModelA_C2_Rh_ME': 'ME.Host.RH.C2',
                               'LaneMarkPosition_C0_Rh_ME': 'ME.Host.RH.C0',
                               'LaneMarkHeadingAngle_C1_Rh_ME': 'ME.Host.RH.C1',
                               'LaneMarkModelDerivA_C3_Rh_ME': 'ME.Host.RH.C3',
                               'Lh_View_End_Longitudinal_Dist': 'ME.Host.LH.ViewRangeEnd',
                               'Rh_View_End_Longitudinal_Dist': 'ME.Host.RH.ViewRangeEnd',
                               'Lh_View_Start_Longitudinal_Dist': 'ME.Host.LH.ViewRangeStart',
                               'Rh_View_Start_Longitudinal_Dist': 'ME.Host.RH.ViewRangeStart',

                               'Lh_Neightbor_Avail': 'ME.Neighbor.LH.Available',

                               'Quality_Lh_ME': 'ME.Host.LH.Quality',
                               'Quality_Rh_ME': 'ME.Host.RH.Quality',

                               'Lh_Neighbor_LaneMark_Model_A_C2': 'ME.Next.LH.C2',
                               'Lh_Neighbor_LaneMark_Model_B_C1': 'ME.Next.LH.C1',
                               'Lh_Neighbor_LaneMark_Model_dA_C3': 'ME.Next.LH.C3',
                               'Lh_Neighbor_LaneMark_Pos_C0': 'ME.Next.LH.C0',
                               'Rh_Neighbor_LaneMark_Model_A_C2': 'ME.Next.RH.C2',
                               'Rh_Neighbor_LaneMark_Model_B_C1': 'ME.Next.RH.C1',
                               'Rh_Neighbor_LaneMark_Model_dA_C3': 'ME.Next.RH.C3',
                               'Rh_Neighbor_LaneMark_Pos_C0': 'ME.Next.RH.C0',
                               'Lh_Neightbor_Type': 'ME.Next.LH.Type', 'Rh_Neightbor_Type': 'ME.Next.RH.Type',
                               'Classification_Lh_ME': 'ME.Host.LH.Classification',
                               'Classification_Rh_ME': 'ME.Host.RH.Classification',
                               'Marker_Width_Lh_ME': 'ME.Host.LH.MarkerWidth',
                               'Marker_Width_Rh_ME': 'ME.Host.RH.MarkerWidth'
                               }, inplace=True)
        df.fillna(-1, inplace=True)
        df['ME.Host.LH.C0'] = -df['ME.Host.LH.C0']
        df['ME.Host.LH.C1'] = -df['ME.Host.LH.C1']
        df['ME.Host.LH.C2'] = -df['ME.Host.LH.C2']
        df['ME.Host.LH.C3'] = -df['ME.Host.LH.C3']

        df['ME.Host.RH.C0'] = -df['ME.Host.RH.C0']
        df['ME.Host.RH.C1'] = -df['ME.Host.RH.C1']
        df['ME.Host.RH.C2'] = -df['ME.Host.RH.C2']
        df['ME.Host.RH.C3'] = -df['ME.Host.RH.C3']

        df['ME.Next.LH.C0'] = -df['ME.Next.LH.C0']
        df['ME.Next.LH.C1'] = -df['ME.Next.LH.C1']
        df['ME.Next.LH.C2'] = -df['ME.Next.LH.C2']
        df['ME.Next.LH.C3'] = -df['ME.Next.LH.C3']

        df['ME.Next.RH.C0'] = -df['ME.Next.RH.C0']
        df['ME.Next.RH.C1'] = -df['ME.Next.RH.C1']
        df['ME.Next.RH.C2'] = -df['ME.Next.RH.C2']
        df['ME.Next.RH.C3'] = -df['ME.Next.RH.C3']

        df.fillna(-1, inplace=True)

        # if not os.path.exists(os.path.join(self.output_path, self.dir_name)):
        #     os.makedirs(os.path.join(self.output_path, self.dir_name))

        # df.to_csv(os.path.join(self.output_path, self.dir_name, self.dir_name + '_me_rearranged.csv'))

        return df

    def rearrangement_dgps(self, file_path):
        df = pd.read_csv(file_path, index_col=False, header=1, low_memory=False)

        if self.is_adaf_db == True:
            df = pd.concat([df.iloc[:, 0:8], df.iloc[:, 8:-1].fillna(method='ffill')], axis=1)
            df = df.dropna(subset=['CAM_FC'])
            df.reset_index(drop=True, inplace=True)
            df.drop(['CAM_FC', 'CAM_SR', 'CAM_SL', 'CAM_SVM_F', 'CAM_SVM_R', 'CAM_SVM_L', 'CAM_SVM_B'],
                          axis=1, inplace=True)

        elif self.is_adaf_db == False:
            df = pd.concat([df.iloc[:, 0:21], df.iloc[:, 21:-1].fillna(method='ffill')], axis=1)
            df = df.dropna(subset=['CAM_1'])
            df.reset_index(drop=True, inplace=True)
            df.drop(['CAM_{}'.format(i) for i in range(1, 21)], axis=1, inplace=True)

        df.rename(columns={'Time': 'DGPS.Timestamp','Speed2D': 'VDY.EgoSpeed'}, inplace=True)# (m/s)
        df.fillna(-1, inplace=True)

        # if not os.path.exists(os.path.join(self.output_path, self.dir_name)):
        #     os.makedirs(os.path.join(self.output_path, self.dir_name))

        # df.to_csv(os.path.join(self.output_path, self.dir_name, self.dir_name + '_dgps_rearranged.csv'))

        return df

    def rearrangement_can(self, file_path):
        df = pd.read_csv(file_path, index_col=False, header=1, low_memory=False)

        if self.is_adaf_db == True:
            df = pd.concat([df.iloc[:, 0:8], df.iloc[:, 8:-1].fillna(method='ffill')], axis=1)
            df = df.dropna(subset=['CAM_FC'])
            df.reset_index(drop=True, inplace=True)
            df.drop(['CAM_FC', 'CAM_SR', 'CAM_SL', 'CAM_SVM_F', 'CAM_SVM_R', 'CAM_SVM_L', 'CAM_SVM_B'],
                          axis=1, inplace=True)

        elif self.is_adaf_db == False:
            df = pd.concat([df.iloc[:, 0:21], df.iloc[:, 21:-1].fillna(method='ffill')], axis=1)

            df = df.dropna(subset=['CAM_1'])
            df.reset_index(drop=True, inplace=True)
            df.drop(['CAM_{}'.format(i) for i in range(1, 21)], axis=1, inplace=True)

        df.rename(columns={'Time': 'CAN.Timestamp','Speed2D': 'VDY.EgoSpeed',
                               # (m/s)
                               }, inplace=True)

        df.fillna(-1, inplace=True)

        # if not os.path.exists(os.path.join(self.output_path, self.dir_name)):
        #     os.makedirs(os.path.join(self.output_path, self.dir_name))

        # df.to_csv(os.path.join(self.output_path, self.dir_name, self.dir_name + '_can_rearranged.csv'))

        return df





