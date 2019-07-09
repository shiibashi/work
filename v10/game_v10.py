import gym
import numpy
import random
import os

class GameV10(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, mode="dev"):
        self.data_dir = "histday_sample" if mode=="dev" else "histday"
        self.code_list = os.listdir(self.data_dir)
        self.columns = ["close", "high", "low", "open", "volume"]
        
        lb = numpy.array([1, 1, 1])
        ub = numpy.array([0, 0, 0])
        self.observation_space = gym.spaces.Box(lb, ub, dtype=numpy.float32)
        self.action_space = gym.spaces.Discrete(2) # 0: stay, 1: long
        
        self.seed()
        self.state = None
        
        self.time = None
        self.code = None
        self.code_data = None
        self.fund = None
                
    def seed(self, seed=None):
        self.numpy_random, seed = gym.utils.seeding.numpy_random(seed)
        return [seed]
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        reward = self.reward(action, self.time, self.code_data)
        done = self.done_flag(self.time, self.code_data)
        self.time += 1
        next_obj = self.observation(self.time, self.code_data)
        return next_obj, reward, done, {}
        
    def reset(self):
        code = random.sample(self.code_list, 1)[0]
        code_data = self._load_code_data(code)
        self.time = 0
        self.code = code
        self.code_data = code_data
        self.fund = 1
        return self.observation(self.time, self.code_data)

    def observation(self, time, df):
        return numpy.array([df[col][time] for col in self.columns])
        
    def reward(self, action, time, df):
        return 1
        
    def done_flag(self, time, df):
        if time < len(df) or self.fund >= 0.7:
            return False
        else:
            return True
        
    def render(self, mode="human"):
        pass
        
    def close(self):
        pass
        
    def _load_code_data(self, code):
        dtype = {
            "code": object,
            "date": object,
            "close": float,
            "high": float,
            "low": float,
            "open": float,
            "volume": float
        }
        code_data = pandas.read_csv("{}/{}.csv".format(self.data_dir, code), dtype=dtype)
        return code_data