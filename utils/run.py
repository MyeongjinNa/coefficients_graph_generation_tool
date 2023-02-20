import os
import pandas as pd
import numpy as np
import glob

from utils.logger import logger
from utils.playlist import Playlist
from utils.visualizer import Visualize
from utils.raw_data_rearranger import Arranger


class Run:
    def __init__(self, argument):
        self.argument = argument

        # set directory path to log_sv
        self.log_sv_dir_path_1 = argument.log_path
        self.log_sv_dir_path_2 = argument.log2_path
        self.log_sv_dir_path_3 = argument.log3_path
        self.log_sv_dir_path_4 = argument.log4_path
        self.measurement_path = argument.measurement_path

        # set path to playlist csv file
        self.playlist_file_path = argument.playlist_path

    def load_raw_data(self, dir_path, current_playlist, suffix_plus_ext):
        """
        Rearrange the values by frameID from raw csv file
        :return: the data frame of sv ego or next line or cpp/can/dgps
        """
        # get directory name from playlist csv file
        dir_name = current_playlist.recording_name
        file_list = glob.glob(
            os.path.join(dir_path, '*' + dir_name + '*', '*' + dir_name + '*' + suffix_plus_ext))
        try:
            logger.debug('Rearranging..{}'.format(file_list[0]))
            # Rearrange the coefficients by frameIO
            arrg = Arranger(dir_name, self.argument)
            df = arrg.rearrangement(file_list[0])
            # Crop the data length according to start timestamp and end timestamp.
            # If end timestamp is 0, it is full length of recording.
            if current_playlist.end_timestamp == 0:
                start_timestamp = current_playlist.start_timestamp
                end_timestamp = len(df)
            else:
                start_timestamp = current_playlist.start_timestamp
                end_timestamp = current_playlist.end_timestamp
            df = df.loc[start_timestamp:end_timestamp, :]
        except:
            # if it cannot find the relevant csv file, it set empty dataFrame
            df = self.initial_data()
            logger.warning('Fail..{}'.format(current_playlist.recording_name))

        return df

    def load_gt_data(self, current_playlist):
        """
        Generate ground truth coefficients using the trajectory path of ego vehicle and lateral distance to ego line
        :return: the data frame of ground truth coefficients
        """
        from utils.groundtruth_generator import GroundTruth

        logger.info('{}:{} is running'.format(current_playlist.recording_name))
        if self.argument.is_adaf_db:
            mobileye_data = self.load_raw_data(self.measurement_path, current_playlist, suffix_plus_ext='me.csv')
            dgps_data = self.load_raw_data(self.measurement_path, current_playlist, suffix_plus_ext='dgps.csv')
        else:
            mobileye_data = self.load_raw_data(self.measurement_path, current_playlist, suffix_plus_ext='pcan.csv')
            dgps_data = self.load_raw_data(self.measurement_path, current_playlist, suffix_plus_ext='dgps.csv')

        sv_log_data1 =  self.load_raw_data(self.log_sv_dir_path_1, current_playlist, suffix_plus_ext='lda_tiny_log.csv')

        try:
            groundtruth = GroundTruth(current_playlist, self.argument.output_path)
            gt_df = groundtruth.generate_gt(dgps_data, mobileye_data, sv_log_data1)

        except Exception as e:
            gt_df = self.initial_data()
            logger.warning('gt generate : {}'.format(e))
        return gt_df

    def load_playlist_data(self):
        """
        load playlist csv file as data frame
        :return: the data frame of the recording name/scenario name/strat timestamp/end timestamp by given from playlist
        """
        if os.path.isfile(self.playlist_file_path):
            df = pd.read_csv(self.playlist_file_path)

            df['scenario'] = df['scenario1'] + '_' + df['scenario2']

        else:
            logger.warning('playlist file is missing')
            df = self.initial_data()

        return df

    def initial_data(self):
        """
        initialize dataFrame
        :return: DataFrame
        """
        df = pd.DataFrame({'NoData': np.full(1, 0)})
        return df

    def generate_coefficient_graph(self):
        playlist_df = self.load_playlist_data()

        for i in range(len(playlist_df.index)):
            current_playlist = Playlist(playlist_df, i)

            logger.debug('{}:{} is running'.format(i, current_playlist.recording_name))

            try:
                # Get data from sv log files
                sv_log_data1 = self.load_raw_data(self.log_sv_dir_path_1, current_playlist, suffix_plus_ext='lda_tiny_log.csv')
                sv_log_data2 = self.load_raw_data(self.log_sv_dir_path_2, current_playlist, suffix_plus_ext='lda_tiny_log.csv')
                sv_log_data3 = self.load_raw_data(self.log_sv_dir_path_3, current_playlist, suffix_plus_ext='lda_tiny_log.csv')
                sv_log_data4 = self.load_raw_data(self.log_sv_dir_path_4, current_playlist, suffix_plus_ext='lda_tiny_log.csv')

                # Options is about which feature is plotted.
                # The subject of file name is depends on database type
                if self.argument.is_adaf_db:
                    mobileye_data = self.load_raw_data(self.measurement_path, current_playlist, suffix_plus_ext='me.csv')
                else:
                    mobileye_data = self.load_raw_data(self.measurement_path, current_playlist, suffix_plus_ext='pcan.csv')

                # Plot ground truth coefficients graph
                if self.argument.is_gt_plot:
                    # it is not implemented yet
                    gt_data = self.load_gt_data(current_playlist)
                else:
                    gt_data = self.initial_data()

                # Plot CPP coefficients graph
                if self.argument.is_cpp_plot:
                    cpp_data = self.load_raw_data(self.log_sv_dir_path_1, current_playlist, suffix_plus_ext='adaf_log.csv')
                else:
                    cpp_data = self.initial_data()

                vis = Visualize(current_playlist, self.argument)
                vis.calculate_statistics(mobileye_data, sv_log_data1, sv_log_data2, sv_log_data3, sv_log_data4)
                vis.plots_coefficient_graph(mobileye_data, gt_data, cpp_data,
                                            sv_log_data1, sv_log_data2, sv_log_data3, sv_log_data4)

                # Plot next line coefficients graph
                if self.argument.is_next_line_plot:
                    vis.plots_coefficient_graph_next_line(sv_log_data1, sv_log_data2)

            except Exception as e:
                logger.warning(e)

    def run(self):
        self.generate_coefficient_graph()
