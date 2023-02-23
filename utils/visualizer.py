import os
from utils.logger import logger

from utils.polynomial import polynomial
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils.signal_name_handler import *

import pandas as pd
import numpy as np

legend_list = ['New SV', 'Old SV']

class Visualize:
    def __init__(self, playlist, argument):
        self.scenario = str(playlist.scenario+'_'+str(playlist.start_timestamp)+'_'+str(playlist.end_timestamp))
        logger.info('scenario : %s',self.scenario)
        self.recording_name = playlist.recording_name
        self.output_path = argument.output_path
        self.argument = argument

    def plots_coefficient_graph_next_line(self, me_data, *Datas):
        confi_val_high_quality = 0.77

        me_color = 'tomato'
        sv_color = ['blue', 'green', 'fuchsia', 'pink']
        gt_color = 'fuchsia'
        guide_color = 'gray'
        cpp_color = 'fuchsia'

        fig = make_subplots(rows=6, cols=3,
                            shared_xaxes=True,
                            vertical_spacing=0.03,
                            horizontal_spacing=0.03,
                            specs=[[{}, {}, {'type': 'table', 'rowspan': 6}],
                                   [{}, {}, None],
                                   [{}, {}, None],
                                   [{}, {}, None],
                                   [{}, {}, None],
                                   [{}, {}, None]],
                            column_widths=[0.4, 0.4, 0.2],
                            subplot_titles=('Next left line', 'Next right line')
                            )

        valid_log_dataset_num = 0

        for i in range(len(Datas)):
            for side in LaneSide:
                for signal in SignalName:
                    if 'SV.Next.{}.{}'.format(side.name, signal.name) in Datas[i].columns:
                        valid_log_dataset_num = valid_log_dataset_num + 1
                        fig.add_trace(trace=go.Scatter(x=Datas[i].index, y=Datas[i]['SV.Next.{}.{}'.format(side.name, signal.name)],
                                                       line=dict(color=sv_color[i]), name=legend_list[i]+'_'+side.name+signal.name), row=int(signal.value), col=side.value)
        for side in LaneSide:
            for signal in SignalName:
                if 'ME.Next.{}.{}'.format(side.name, signal.name) in me_data.columns:
                    fig.add_trace(trace=go.Scatter(x=me_data.index,
                                                   y=me_data['ME.Next.{}.{}'.format(side.name, signal.name)],
                                                   line=dict(color=me_color),
                                                   name='me'+'_' + side.name + signal.name),
                                  row=int(signal.value), col=side.value)

        length = len(Datas[0].index)

        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.01), line=dict(color=guide_color), showlegend=False),row=2, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.01), line=dict(color=guide_color), showlegend=False),
                      row=2, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.0002), line=dict(color=guide_color), showlegend=False),row=3, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.0002), line=dict(color=guide_color), showlegend=False),
                      row=3, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,confi_val_high_quality), line=dict(color=guide_color), showlegend=False),
                      row=6, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.01), line=dict(color=guide_color), showlegend=False),row=2, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.01), line=dict(color=guide_color), showlegend=False),row=2, col=2)

        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.0002), line=dict(color=guide_color), showlegend=False),row=3, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.0002), line=dict(color=guide_color), showlegend=False),
                      row=3, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.000002), line=dict(color=guide_color), showlegend=False),
                      row=4, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.000002), line=dict(color=guide_color), showlegend=False),
                      row=4, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.000002), line=dict(color=guide_color), showlegend=False),
                      row=4, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.000002), line=dict(color=guide_color), showlegend=False),
                      row=4, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,confi_val_high_quality), line=dict(color=guide_color), showlegend=False),
                      row=6, col=2)
        self.calculate_statistics('Next', me_data, Datas[0], Datas[1])

        header = ['/', 'MEyeQ4']
        sv_header = legend_list #list('SV_{}'.format(i) for i in range(valid_log_dataset_num))
        header.extend(sv_header)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header,
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=self.kpi_data.T,
                    align="left")
            ),
            row=1, col=3
        )

        if 'highway' in self.scenario:
            fig.update_layout(
                              yaxis2=dict(range=[-2.5, -1.2]),
                              yaxis3=dict(range=[-0.015, 0.015]),
                              yaxis4=dict(range=[-0.015, 0.015]),
                              yaxis5=dict(range=[-0.00019, 0.00019]),
                              yaxis6=dict(range=[-0.00019, 0.00019]),
                              yaxis7=dict(range=[-0.0000025, 0.0000025]),
                              yaxis8=dict(range=[-0.0000025, 0.0000025]),
                              font=dict(size=9))
        elif 'curve' in self.scenario:
            fig.update_layout(
                              yaxis2=dict(range=[-2, 0]),
                              yaxis3=dict(range=[-0.05, 0.05]),
                              yaxis4=dict(range=[-0.05, 0.05]),
                              yaxis5=dict(range=[-0.002, 0.002]),
                              yaxis6=dict(range=[-0.002, 0.002]),
                              yaxis7=dict(range=[-0.000025, 0.000025]),
                              yaxis8=dict(range=[-0.000025, 0.000025]),
                              font=dict(size=9))
        else:
            fig.update_layout(
                              yaxis2=dict(range=[-2.5, -1]),
                              yaxis3=dict(range=[-0.025, 0.025]),
                              yaxis4=dict(range=[-0.025, 0.025]),
                              yaxis5=dict(range=[-0.00055, 0.00055]),
                              yaxis6=dict(range=[-0.00055, 0.00055]),
                              yaxis7=dict(range=[-0.000005, 0.000005]),
                              yaxis8=dict(range=[-0.000005, 0.000005]),
                              font=dict(size=9))


        fig['layout']['xaxis7']['title'] = 'Frame'
        fig['layout']['xaxis8']['title'] = 'Frame'
        fig['layout']['xaxis11']['title'] = 'Frame'
        fig['layout']['xaxis12']['title'] = 'Frame'
        fig['layout']['yaxis']['title'] = 'C0[m]'
        fig['layout']['yaxis3']['title'] = 'C1[rad]'
        fig['layout']['yaxis5']['title'] = 'C2[1/m]'
        fig['layout']['yaxis7']['title'] = 'C3[1/m^2]'
        fig['layout']['yaxis9']['title'] = 'View range(Start/End)[m]'
        fig['layout']['yaxis2']['title'] = 'C0[m]'
        fig['layout']['yaxis4']['title'] = 'C1[rad]'
        fig['layout']['yaxis6']['title'] = 'C2[1/m]'
        fig['layout']['yaxis8']['title'] = 'C3[1/m^2]'
        fig['layout']['yaxis10']['title'] = 'View range(Start/End)[m]'
        # fig['layout']['yaxis11']['title'] = 'Ego speed[kph]'
        fig['layout']['yaxis11']['title'] = 'Confidence(Quality)'
        fig['layout']['yaxis12']['title'] = 'Confidence(Quality)'
        fig['layout']['title'] = self.recording_name + ' ' + str(self.scenario)

        if 'SV.Next.LH.C0' in Datas[0].columns:
            fig.show()
            fig.write_html(os.path.join(self.output_path, self.recording_name + '_' + str(self.scenario))
                           + '_' + legend_list[0] + '_' + legend_list[1] + '_next_line_graph.html')

    def plots_coefficient_graph(self, me_data, gt_data, cpp_data, *Datas):

        confi_val_high_quality = 0.77

        me_color = 'tomato'
        sv_color = ['blue', 'green', 'fuchsia', 'pink']
        gt_color = 'fuchsia'
        guide_color = 'gray'
        cpp_color = 'fuchsia'

        fig = make_subplots(rows=6, cols=3,
                            shared_xaxes=True,
                            vertical_spacing=0.03,
                            horizontal_spacing=0.03,
                            specs=[[{}, {}, {'type': 'table', 'rowspan': 6}],
                                   [{}, {}, None],
                                   [{}, {}, None],
                                   [{}, {}, None],
                                   [{}, {}, None],
                                   [{}, {}, None]],
                            column_widths=[0.4, 0.4, 0.2],
                            subplot_titles=('Host left line', 'Host right line')
                            )

        valid_log_dataset_num = 0

        for i in range(len(Datas)):
            for side in LaneSide:
                for signal in SignalName:
                    if 'SV.Host.{}.{}'.format(side.name, signal.name) in Datas[i].columns:
                        valid_log_dataset_num = valid_log_dataset_num + 1
                        fig.add_trace(trace=go.Scatter(x=Datas[i].index, y=Datas[i]['SV.Host.{}.{}'.format(side.name, signal.name)],
                                                       line=dict(color=sv_color[i]), name=legend_list[i]+'_'+side.name+signal.name), row=int(signal.value), col=side.value)

        for side in LaneSide:
            for signal in SignalName:
                if 'ME.Host.{}.{}'.format(side.name, signal.name) in me_data.columns:
                    fig.add_trace(trace=go.Scatter(x=me_data.index,
                                                   y=me_data['ME.Host.{}.{}'.format(side.name, signal.name)],
                                                   line=dict(color=me_color),
                                                   name='me'+'_' + side.name + signal.name),
                                  row=int(signal.value), col=side.value)

        for side in LaneSide:
            for signal in SignalName:
                if 'GT.Host.{}.{}'.format(side.name, signal.name) in gt_data.columns:
                    fig.add_trace(trace=go.Scatter(x=gt_data.index,
                                                   y=gt_data['GT.Host.{}.{}'.format(side.name, signal.name)],
                                                   line=dict(color=gt_color),
                                                   name='gt'+'_' + side.name + signal.name),
                                  row=int(signal.value), col=side.value)

        for side in LaneSide:
            for signal in SignalName:
                if 'CPP.Host.{}'.format(signal.name) in cpp_data.columns:
                    fig.add_trace(trace=go.Scatter(x=cpp_data.index,
                                                   y=cpp_data['CPP.Host.{}'.format(signal.name)],
                                                   line=dict(color=cpp_color),
                                                   name='cpp' + '_' + signal.name),
                                  row=int(signal.value), col=side.value)

        length = len(Datas[0].index)

        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.01), line=dict(color=guide_color), showlegend=False),row=2, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.01), line=dict(color=guide_color), showlegend=False),
                      row=2, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.0002), line=dict(color=guide_color), showlegend=False),row=3, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.0002), line=dict(color=guide_color), showlegend=False),
                      row=3, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,confi_val_high_quality), line=dict(color=guide_color), showlegend=False),
                      row=6, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.01), line=dict(color=guide_color), showlegend=False),row=2, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.01), line=dict(color=guide_color), showlegend=False),row=2, col=2)

        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.0002), line=dict(color=guide_color), showlegend=False),row=3, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.0002), line=dict(color=guide_color), showlegend=False),
                      row=3, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.000002), line=dict(color=guide_color), showlegend=False),
                      row=4, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,0.000002), line=dict(color=guide_color), showlegend=False),
                      row=4, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.000002), line=dict(color=guide_color), showlegend=False),
                      row=4, col=1)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,-0.000002), line=dict(color=guide_color), showlegend=False),
                      row=4, col=2)
        fig.add_trace(trace=go.Scatter(x=Datas[0].index, y=np.full(length,confi_val_high_quality), line=dict(color=guide_color), showlegend=False),
                      row=6, col=2)

        self.calculate_statistics('Host', me_data, Datas[0], Datas[1], Datas[2], Datas[3])

        header = ['/', 'MEyeQ4']
        sv_header = legend_list #list('SV_{}'.format(i) for i in range(valid_log_dataset_num))
        header.extend(sv_header)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header,
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=self.kpi_data.T,
                    align="left")
            ),
            row=1, col=3
        )
        if 'highway' in self.scenario:
            fig.update_layout(
                              yaxis2=dict(range=[-2.5, -1.2]),
                              yaxis3=dict(range=[-0.015, 0.015]),
                              yaxis4=dict(range=[-0.015, 0.015]),
                              yaxis5=dict(range=[-0.00019, 0.00019]),
                              yaxis6=dict(range=[-0.00019, 0.00019]),
                              yaxis7=dict(range=[-0.0000025, 0.0000025]),
                              yaxis8=dict(range=[-0.0000025, 0.0000025]),
                              font=dict(size=9))
        elif 'curve' in self.scenario:
            fig.update_layout(
                              yaxis2=dict(range=[-2, 0]),
                              yaxis3=dict(range=[-0.05, 0.05]),
                              yaxis4=dict(range=[-0.05, 0.05]),
                              yaxis5=dict(range=[-0.002, 0.002]),
                              yaxis6=dict(range=[-0.002, 0.002]),
                              yaxis7=dict(range=[-0.000025, 0.000025]),
                              yaxis8=dict(range=[-0.000025, 0.000025]),
                              font=dict(size=9))
        else:
            fig.update_layout(
                              yaxis2=dict(range=[-2.5, -1]),
                              yaxis3=dict(range=[-0.025, 0.025]),
                              yaxis4=dict(range=[-0.025, 0.025]),
                              yaxis5=dict(range=[-0.00055, 0.00055]),
                              yaxis6=dict(range=[-0.00055, 0.00055]),
                              yaxis7=dict(range=[-0.000005, 0.000005]),
                              yaxis8=dict(range=[-0.000005, 0.000005]),
                              font=dict(size=9))


        fig['layout']['xaxis7']['title'] = 'Frame'
        fig['layout']['xaxis8']['title'] = 'Frame'
        fig['layout']['xaxis11']['title'] = 'Frame'
        fig['layout']['xaxis12']['title'] = 'Frame'
        fig['layout']['yaxis']['title'] = 'C0[m]'
        fig['layout']['yaxis3']['title'] = 'C1[rad]'
        fig['layout']['yaxis5']['title'] = 'C2[1/m]'
        fig['layout']['yaxis7']['title'] = 'C3[1/m^2]'
        fig['layout']['yaxis9']['title'] = 'View range(Start/End)[m]'
        fig['layout']['yaxis2']['title'] = 'C0[m]'
        fig['layout']['yaxis4']['title'] = 'C1[rad]'
        fig['layout']['yaxis6']['title'] = 'C2[1/m]'
        fig['layout']['yaxis8']['title'] = 'C3[1/m^2]'
        fig['layout']['yaxis10']['title'] = 'View range(Start/End)[m]'
        # fig['layout']['yaxis11']['title'] = 'Ego speed[kph]'
        fig['layout']['yaxis11']['title'] = 'Confidence(Quality)'
        fig['layout']['yaxis12']['title'] = 'Confidence(Quality)'
        fig['layout']['title'] = self.recording_name + ' ' + str(self.scenario)

        fig.show()
        if 'SV.Host.LH.C0' in Datas[0].columns:
            fig.write_html(os.path.join(self.output_path, self.recording_name + '_' + str(self.scenario))
                           + '_' + legend_list[0] + '_' + legend_list[1] + '_graph.html')

    def calculate_statistics(self, lane_pos, me_data, *Datas):
        index = [' avail rate(%)',
              ' c0 median', ' c1 median', ' c2 median', ' c3 median',
              ' c0 std', ' c1 std', ' c2 std', ' c3 std',
              ' c0 min', ' c1 min', ' c2 min', ' c3 min',
              ' c0 max', ' c1 max', ' c2 max', ' c3 max', ]

        mobileye_statistics_data = np.full((len(index)), 0)
        round_num = 8

        sv_statistics_data = []

        for i in range(len(Datas)):
            if ('SV.{}.LH.Confidence'.format(lane_pos) in Datas[i].columns):

                df_l = Datas[i].loc[:,
                       ['SV.{}.LH.C0'.format(lane_pos), 'SV.{}.LH.C1'.format(lane_pos), 'SV.{}.LH.C2'.format(lane_pos), 'SV.{}.LH.C3'.format(lane_pos)]].copy()

                df_r = Datas[i].loc[:,
                       ['SV.{}.RH.C0'.format(lane_pos), 'SV.{}.RH.C1'.format(lane_pos), 'SV.{}.RH.C2'.format(lane_pos),'SV.{}.RH.C3'.format(lane_pos)]].copy()

                condition_l = Datas[i]['SV.{}.LH.Quality'.format(lane_pos)] == 3
                condition_r = Datas[i]['SV.{}.RH.Quality'.format(lane_pos)] == 3

                sv_availability_lh = np.array([len(Datas[i].loc[condition_l]) / len(Datas[i]) * 100])
                sv_availability_rh = np.array([len(Datas[i].loc[condition_r]) / len(Datas[i]) * 100])

                sv_lh_median = df_l.loc[condition_l].median(skipna=True).round(round_num).to_numpy()
                sv_rh_median = df_r.loc[condition_r].median(skipna=True).round(round_num).to_numpy()

                sv_lh_std = df_l.loc[condition_l].std(skipna=True).round(round_num).to_numpy()
                sv_rh_std = df_r.loc[condition_r].std(skipna=True).round(round_num).to_numpy()

                sv_lh_min = df_l.loc[condition_l].max(axis=0).round(round_num).to_numpy()
                sv_rh_min = df_r.loc[condition_r].max(axis=0).round(round_num).to_numpy()

                sv_lh_max = df_l.loc[condition_l].min(axis=0).round(round_num).to_numpy()
                sv_rh_max = df_r.loc[condition_r].min(axis=0).round(round_num).to_numpy()

                sv_availability = (sv_availability_lh+sv_availability_rh)/2
                sv_median = (sv_lh_median+sv_rh_median)/2
                sv_std = (sv_lh_std+sv_rh_std)/2
                sv_min = (sv_lh_min+sv_rh_min)/2
                sv_max = (sv_lh_max+sv_rh_max)/2

                sv_statistics_data.append(np.concatenate(
                    (sv_availability, sv_median, sv_std, sv_min, sv_max)))

        if ('ME.Host.LH.C0' in me_data.columns):
            condition_l = (me_data['ME.Host.LH.Quality'] >= 2)
            condition_r = (me_data['ME.Host.RH.Quality'] >= 2)

            me_availability_lh = np.array([len(me_data.loc[condition_l]) / len(me_data) * 100])
            me_availability_rh = np.array([len(me_data.loc[condition_r]) / len(me_data) * 100])

            df_l = me_data.loc[:,
                   ['ME.{}.LH.C0'.format(lane_pos), 'ME.{}.LH.C1'.format(lane_pos), 'ME.{}.LH.C2'.format(lane_pos),
                    'ME.{}.LH.C3'.format(lane_pos)]].copy()

            df_r = me_data.loc[:,
                   ['ME.{}.RH.C0'.format(lane_pos), 'ME.{}.RH.C1'.format(lane_pos), 'ME.{}.RH.C2'.format(lane_pos),
                    'ME.{}.RH.C3'.format(lane_pos)]].copy()



            me_lh_median = df_l.loc[condition_l].median(skipna=True).round(round_num).to_numpy()
            me_rh_median = df_r.loc[condition_r].median(skipna=True).round(round_num).to_numpy()

            me_lh_std = df_l.loc[condition_l].std(skipna=True).round(round_num).to_numpy()
            me_rh_std = df_r.loc[condition_r].std(skipna=True).round(round_num).to_numpy()
            me_lh_min = df_l.loc[condition_l].max(axis=0).round(round_num).to_numpy()
            me_rh_min = df_r.loc[condition_r].max(axis=0).round(round_num).to_numpy()
            me_lh_max = df_l.loc[condition_l].min(axis=0).round(round_num).to_numpy()
            me_rh_max = df_r.loc[condition_r].min(axis=0).round(round_num).to_numpy()
            me_availability = (me_availability_lh + me_availability_rh) / 2
            me_median = (me_lh_median + me_rh_median) / 2
            me_std = (me_lh_std + me_rh_std) / 2
            me_min = (me_lh_min + me_rh_min) / 2
            me_max = (me_lh_max + me_rh_max) / 2
            mobileye_statistics_data = np.concatenate(
                (me_availability, me_median, me_std, me_min, me_max))

        kpi_dict = {'index': index, 'ME': mobileye_statistics_data}
        for i in range(len(sv_statistics_data)):
            kpi_dict['value_{}'.format(i)] = sv_statistics_data[i]

        kpi_df = pd.DataFrame(kpi_dict)
        self.kpi_data = kpi_df

        return self.kpi_data



