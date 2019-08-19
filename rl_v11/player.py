import numpy


import keras
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout
from keras.layers.normalization import BatchNormalization
from keras import Input, Sequential
from keras.optimizers import SGD



class Player(object):
    def __init__(self, action_size, csv_size, img_size):
        self.csv_shape = csv_size
        self.img_shape = img_size
        self.action_size = action_size
        self.value_size = 1

        self.actor = self.build_actor()
        self.critic = self.build_critic()
    
    def build_actor(self):
        input_csv = Input(name="input_csv", shape=self.csv_shape)
        csv = Dense(2, activation="sigmoid", name="csv_dense_1")(input_csv)
        csv = Dense(2, activation="sigmoid", name="csv_dense_2")(input_csv)
        
        input_img = Input(name="input_img", shape=self.img_shape)
        img = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
                     activation="relu", name="img_conv2d_1")(input_img)
        img = MaxPooling2D(pool_size=(2, 2), name="img_maxpool_1")(img)
        img = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
                     activation="relu", name="img_conv2d_2")(img)
        img = BatchNormalization(name="img_bn_1")(img)
        img = MaxPooling2D(pool_size=(2, 2), name="img_maxpool_2")(img)
        img = Flatten(name="img_flatten_1")(img)
        
        img = Dense(2, activation="relu", name="img_dense_1")(img)
        
        net = Concatenate()([csv, img])
        net = Dense(self.action_size, activation="softmax", name="output")(net)
        #net.summary()
        model = keras.Model(inputs=[input_csv, input_img], outputs=net)
        model.compile(SGD(), loss='categorical_crossentropy')
        return model
    
    
    def build_critic(self):
        input_csv = Input(name="input_csv", shape=self.csv_shape)
        csv = Dense(2, activation="sigmoid", name="csv_dense_1")(input_csv)
        csv = Dense(2, activation="sigmoid", name="csv_dense_2")(input_csv)
        
        input_img = Input(name="input_img", shape=self.img_shape)
        img = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
                     activation="relu", name="img_conv2d_1")(input_img)
        img = MaxPooling2D(pool_size=(2, 2), name="img_maxpool_1")(img)
        img = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
                     activation="relu", name="img_conv2d_2")(img)
        img = BatchNormalization(name="img_bn_1")(img)
        img = MaxPooling2D(pool_size=(2, 2), name="img_maxpool_2")(img)
        img = Flatten(name="img_flatten_1")(img)
        
        img = Dense(2, activation="relu", name="img_dense_1")(img)
        
        net = Concatenate()([csv, img])
        net = Dense(self.value_size, activation="linear", name="output")(net)
        #net.summary()
        model = keras.Model(inputs=[input_csv, input_img], outputs=net)
        model.compile(SGD(), loss='mse')
        return model
    

    def get_action(self, state):
        csv, img = state
        #print(csv.shape)
        #print(img.shape)
        policy = self.actor.predict([csv, img], batch_size=1).flatten()
        noise = numpy.random.uniform(0, 1, self.action_size)
        p = (policy + noise) / (policy + noise).sum()
        a = numpy.random.choice(self.action_size, 1, p=p)[0]
        return a

    def get_action_prob(self, state):
        csv, img = state
        policy = self.actor.predict([csv, img], batch_size=1).flatten()
        return policy

        
    def train_model(self, states, actions, rewards, next_states, dones):
        csv, img = states
        next_csv, next_img = next_states
    
        length = len(csv)
        target = numpy.zeros((length, self.value_size))
        advantages = numpy.zeros((length, self.action_size))
        #print(states)
        values = self.critic.predict([csv, img])
        next_values = self.critic.predict([next_csv, next_img])

        for i, done in enumerate(dones):
            action = actions[i]
            reward = rewards[i]
            value = values[i]
            next_value = next_values[i]
            if done:
                advantages[i][action] = reward - value
                target[i][0] = reward
            else:
                r = 0.9
                advantages[i][action] = reward + r * (next_value) - value
                target[i][0] = reward + r * next_value

        self.actor.fit([csv, img], advantages, epochs=1, verbose=0)
        self.critic.fit([csv, img], target, epochs=1, verbose=0)
