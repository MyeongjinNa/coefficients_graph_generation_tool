
class Playlist:
    def __init__(self, my_playlist_data, i):
        # self.initialize(my_playlist_data)
        self.get_playlist_info(my_playlist_data, i)

    def initialize(self, my_playlist_data):
        if my_playlist_data.isna().sum():
            my_playlist_data['scenario1'].fillna('general', inplace=True)
            my_playlist_data['timestamp.start'].fillna(0, inplace=True)
            my_playlist_data['timestamp.end'].fillna(0, inplace=True)

    def get_playlist_info(self, my_playlist_data, i):
        self.recording_name = str(my_playlist_data.loc[i, 'recording_name']).strip('_log')
        self.scenario = str(my_playlist_data.loc[i, 'scenario1'])+'_'+str(my_playlist_data.loc[i, 'scenario2'])
        self.start_timestamp = int(my_playlist_data.loc[i, 'timestamp.start'])
        self.end_timestamp = int(my_playlist_data.loc[i, 'timestamp.end'])
        self.index = i
