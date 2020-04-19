from wwo_hist import retrieve_hist_data
import pandas as pd

class weather:
    def __init__(self, config):
        self.config = config

    def get_weather_data(self):
        """CSV will be written to the current directory."""
        retrieve_hist_data(self.config['api_key'],
                           self.config['location_list'],
                           self.config['start_date'],
                           self.config['end_date'],
                           self.config['frequency'],
                           location_label = self.config['location_label'],
                           export_csv = True,
                           store_df = False)
        return None
    


if __name__ == '__main__':
    user_config = {'frequency' : 1,
                   'start_date' : '01-Jan-2010',
                   'end_date' : '31-Jan-2010',
                   'api_key' : '2c9e967a17ba475087893244201503',
                   'location_list' : ['singapore'],
                   'location_label' : False}
    
    weather_tool = weather(user_config)
    weather_tool.get_weather_data()
    data = pd.read_csv('singapore.csv')
    data = weather_tool.proprogate_half_hourly(data)
    print(len(data))
