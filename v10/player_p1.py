import numpy
#from keras.layers import Dense
#from keras.models import Sequential
#from keras.optimizers import SGD

import keras
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout
from keras.layers.normalization import BatchNormalization
from keras import Input, Sequential
from keras.optimizers import SGD



class PlayerP1(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        
        actor, critic = self.build()
        self.actor = actor
        self.critic = critic
        
        #self.actor = self.build_actor()
        #self.critic = self.build_critic()

        #if self.load_model:
        #    self.actor.load_weights("./save_model/cartpole_actor.h5")
        #    self.critic.load_weights("./save_model/cartpole_critic.h5")
    
    """
    def build(self):
        network_input = Input(name="input", shape=(self.state_size, ))
        network = network_input
        #network = Dense(4, activation="relu", name="dense_1")(network_input)
        #network = Dense(1, activation="relu", name="dense_2")(network)
        #network = Dropout(0.5)(network)
        #network = Dense(1, activation="relu", name="dense_2")(network)
        
        network_output1 = Dense(self.action_size, activation="softmax", name="output_action")(network)
        network_output2 = Dense(self.value_size, activation="relu", name="output_value")(network)

        actor = keras.Model(inputs=network_input, outputs=network_output1)
        actor.compile(SGD(), loss="categorical_crossentropy")
        
        critic = keras.Model(inputs=network_input, outputs=network_output2)
        critic.compile(SGD(), loss="mse")
        return actor, critic
    """
    
    def build(self):
        actor = Sequential()
        actor.add(Dense(8, input_dim=self.state_size, activation='relu'))
        actor.add(Dense(self.action_size, activation='softmax'))
        actor.summary()
        actor.compile(SGD(), loss='categorical_crossentropy')

        critic = Sequential()
        critic.add(Dense(8, input_dim=self.state_size, activation='relu'))
        critic.add(Dense(self.value_size, activation='linear'))
        critic.summary()
        critic.compile(SGD(), loss="mse")
        return actor, critic
    

    def get_action(self, state):
        if numpy.random.rand() <= 0.1:
            return numpy.random.choice(self.action_size, 1)[0]        
        policy = self.actor.predict(state, batch_size=1).flatten()
        return numpy.random.choice(self.action_size, 1, p=policy)[0]

    def get_action_prob(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return policy

        
    def train_model(self, state, action, reward, next_state, done):
        target = numpy.zeros((1, self.value_size))
        advantages = numpy.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            r = 0.99
            advantages[0][action] = reward + r * (next_value) - value
            target[0][0] = reward + r * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)
