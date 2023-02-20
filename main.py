"""
This tool is for coefficients graph generation tool.
"""

import os
from utils.logger import logger
import argparse
from utils.run import Run

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Coefficients Graph Generation Tool")

        parser.add_argument('-op', '--output_path', dest='output_path', required=False,
                            default=r'./sample/output', help='path to output directory')

        # the graph will be generated according to playlist csv file
        parser.add_argument('-pl', '--playlist_path', dest='playlist_path', required=False,
                            default=r'./sample/input/playlist/playlist.csv',
                            help='path to playlist file path')

        # should be checked out if log csv files are included in the path
        parser.add_argument('-lp', '--log_path',  dest='log_path', required=False,
                            default=r'.\sample\input\new_log_sv',
                            help='path to log_sv directory')
        # should be checked out if log csv files are included in the path
        parser.add_argument('-lp2', '--log2_path', dest='log2_path', required=False,
                            default=r'.\sample\input\old_log_sv',
                            help='path to log_sv directory')
        # optional path
        parser.add_argument('-lp3', '--log3_path', dest='log3_path', required=False,
                            default=r'',
                            help='path to log_sv directory')
        # optional path
        parser.add_argument('-lp4', '--log4_path', dest='log4_path', required=False,
                            default=r'',
                            help='path to log_sv directory')
        # should be checked if can/dgps/pcan csv files are included in the path
        parser.add_argument('-mp', '--measurement_path', dest='measurement_path', required=False,
                            default=r'.\sample\input\measurement',
                            help='path to gt directory')
        # can be selected if ground truth coefficients plot or not
        parser.add_argument('-is_gt_plot', '--is_gt_plot', dest='is_gt_plot', required=False,
                            default=False,
                            help='can be selected if ground truth coefficients plot or not')
        # can be selected if cpp coefficients plot or not
        parser.add_argument('-is_cpp_plot', '--is_cpp_plot', dest='is_cpp_plot', required=False,
                            default=False,
                            help='can be selected if cpp coefficients plot or not')
        # can be selected if next line coefficients plot or not
        parser.add_argument('-is_next_line_plot', '--is_next_line_plot', dest='is_next_line_plot', required=False,
                            default=True,
                            help='can be selected if next line coefficients plot or not')
        # should be set if the database is from adaf or not
        parser.add_argument('-is_adaf_db', '--is_adaf_db', dest='is_adaf_db', required=False,
                            default=True,
                            help='is adaf database?')

        args = parser.parse_args()
        # if output path does not exist
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        import getpass
        logger.info(getpass.getuser())
        logger.info("Start")
        logger.info("output path       : %s" % args.output_path)
        logger.info("playlist path     : %s" % args.playlist_path)

        logger.info("1. log path       : %s" % args.log_path)
        logger.info("2. log path       : %s" % args.log2_path)
        logger.info("3. log path       : %s" % args.log3_path)
        logger.info("4. log path       : %s" % args.log4_path)

        start = Run(args)
        start.run()

    # except Exception as e:
    #     logger.critical('Failed : %s', e)
    finally:
        logger.info('Done')
