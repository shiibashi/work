import cv2
import numpy
import pandas
import os

class Env(object):
    def __init__(self, df):
        self.df = df
        
        self.columns = ["Open_MA_5_N", "Open_MA_25_N", "Open_MA_75_N", "Open_R_1"]
        self.target = "Profit"
        
        self.action_size = 3
        self.csv_size = (5, )
        self.img_size = (180, 180, 3)
        
        self.state = None
        
        self.time = None
        self.capability = None
        self.cut_line = None
        
    def step(self, action):
        reward = self.reward(action, self.df, self.time)
        self.time += 1
        self.cut_line += 0.01
        done = self.done_flag(self.df, self.time)
        next_obs = self.observation(self.df, self.time)
        return next_obs, reward, done, {}
        
    def reset(self):
        self.time = 0
        self.capability = 1
        self.cut_line = 1
        return self.observation(self.df, self.time)

    def observation(self, df, time):
        feature = numpy.array([df[col][time] for col in self.columns])
        status = numpy.array([self.capability - self.cut_line])
        s = len(feature) + len(status)
        csv_arr = numpy.concatenate([feature, status]).reshape(1, s)
        img_arr = self._load_chart_img(self.df, time)
        obs = (csv_arr, img_arr)
        return obs
        
    def reward(self, action, df, time):
        profit = df[self.target][time]
        if action == 2: # short
            action = -1
        
        self.capability += profit * action
        return max(0, self.capability - self.cut_line)
        
    def done_flag(self, df, time):
        if time < len(df) - 1 and self.capability >= self.cut_line - 0.2:
            return False
        else:
            return True
        
    def _load_chart_img(self, df, time):
        #timestamp = df["Timestamp"][time]
        timestamp = 1333041840
        img_arr = cv2.imread("dataset/img/{}.png".format(timestamp))
        assert img_arr is not None
        return numpy.array([img_arr]) # 1, 180, 180, 3