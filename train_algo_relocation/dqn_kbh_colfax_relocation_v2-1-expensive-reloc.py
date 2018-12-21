"""
DQNAgent based on work by RLCode team - Copyright (c) 2017 RLCode (MIT Licence)
https://github.com/rlcode/reinforcement-learning

Tailored to the TUSP by Evertjan Peer
KBH example

30000_instances: we generated 30000 instances of a specific size for the Binckhorst. 


Changes to V2: 
    - training on instances of different sizes. 

"""
import gc
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K
from kbh_yard_b2b_relocation_expensive_reloc import KBH_Env  # This is the environment of the shunting yard
# import convert_event_list as convert
import datetime
import pandas as pd
import time

# import instance generator
from data_retrieval_30000_14151617 import INSTANCEProvider

#visualize learning
visualization = False
if visualization:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

EPISODES = 250000  # 60000# 50000 #used to be 500000 (note that this should be long enough to at least cover the exploration_steps)

class DQNAgent:
    def __init__(self, yard, load):
        self.load_model = load
        # environment settings
        self.state_size = yard.get_state().shape  # this is specific to the state representation you choose for the problem.
        self.action_size = yard.nr_tracks  # this as well

        # set epsilon
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1  # start fully random, end with 10% random action selection.
        self.exploration_steps = 150000  # 1000000.            #this determines how long one explores (nr. steps from 1 to 0.1 epsilon)
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # train params
        self.batch_size = 32  # batch size used to update the neural nets from experience replay
        self.train_start = 10000  # first collect some experiences using random move selection
        self.update_target_rate = 10000  # after every 10000 actions update the target model
        self.discount_factor = 0.99
        self.memory = deque(maxlen=125000)  # this is the memory the DQN samples experiences from for Experience Replay

        # build model
        self.model = self.build_model()  # two models are maintained: a evaluation and target model.
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()  # here a special type of optimizer is used: why which optimizer works in what situation?

        # stuff for tensorboard.
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()

        time_stamp_dir = int(time.time())
        print(time_stamp_dir)
        self.summary_dir_name = 'summary/dqn_dummy/' + str(time_stamp_dir)
        self.summary_writer = tf.summary.FileWriter(
            self.summary_dir_name, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save/dummy_problem_weights.h5")
            print('model loaded')

    # This is a custom optimizer.
    # Need to test how this one performs w.r.t. to standard mse Adam?
    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)  # was lr = 0.00025
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # build model to approx the Q-table with a CNN
    # in: state, out: Qvalue of each state-action pair.
    def build_model(self):
        # This is the original DQN NN that is used by the Atari paper.
        # I use a simpler one for this model since I've a simpler state space here.
        #
        #        model = Sequential()
        #        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
        #                         input_shape=self.state_size))
        #        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        #        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        #        model.add(Flatten())
        #        model.add(Dense(512, activation='relu'))
        #        model.add(Dense(self.action_size))
        #        model.summary()
        #        return model

        model = Sequential()
        model.add(Conv2D(32, (4, 4), padding='same', activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # Update the target model to match the eval. net every some steps.
    def update_target_model(self):
        print('target_model_updated')
        self.target_model.set_weights(self.model.get_weights())

    # epsilon-greedy policy used to select next action.
    # with probability eps we pick a random action
    # with probability 1-eps we pick the best action according to the CNN for that state.
    def get_action(self, history):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def get_random_action(self):
        return random.randrange(self.action_size)

    # We save a <state, action, reward, next state> sample to the replay memory
    # this replay memory is later used to random sample experiences from to update the NN.
    def replay_memory(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    # Pick #batch_size samples randomly from the memory
    # note that standard DQN does this randomly
    # In next versions 'Prioritized Replay' could be implemented here to
    # prioritize some experiences over other experiences based on their 'surpriziness'
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return  # do nothing when we are still in the 'warm up' period.
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step  # here is where we decay epsilon.

        mini_batch = random.sample(self.memory, self.batch_size)  # sample randomly from the memory

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            history[i] = mini_batch[i][0]
            next_history[i] = mini_batch[i][3]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            done.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        # this is very much like Q learning
        # we get approx the state-action value pair with the direct reward + the max Q value at next state
        # we compute this using the target model, which is fixed for several training steps to keep targets fixed.
        for i in range(self.batch_size):
            if done[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                            np.amax(target_value[i])

        # train the model based on this new data of states, actions and new targets.
        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def save_model(self, name):
        self.model.save_weights(name)

    # This is for tensorboard.
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)
        episode_start_espilon = tf.Variable(0.)

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average_Max_Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average_Loss/Episode', episode_avg_loss)
        tf.summary.scalar('End_Epsilon/Episode', episode_start_espilon)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss, episode_start_espilon]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

# this function returns random colors for visualisation of learning.
def rand_cmap(nlabels, type='soft', first_color_black=True, last_color_black=False):
    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
    return random_colormap

# This is a new main loop
if __name__ == "__main__":
    instance_prov = INSTANCEProvider()
    start_time = datetime.datetime.now()
    print('start time: ', start_time)
    printcounter = 0
    scores, episodes, global_step = [], [], 0

    yrd = KBH_Env()  # Create yard
    agent = DQNAgent(yrd, False)  # Create agent
    
    if visualization:
        #visualize learning
        new_cmap = rand_cmap(200, type='soft', first_color_black=True, last_color_black=False)

    for episode in range(EPISODES):
        # this event list contains arrival, relocation_opportunity and departure events.
        event_list = instance_prov.get_random_instance()
        steps, t, total_t, score= len(event_list), 0, 0, 0
        printcounter += 1

        state = yrd.reset(event_list)  # get first observation
        history = np.float32(np.reshape(state, (1, yrd.shape[0], yrd.shape[1], yrd.shape[2])))  # reshape state.
        
        done, busy_relocating = False, False
        
        if visualization:
            yrd.show_state(history, new_cmap)

        while not done:
            attempt = 1
            backup_yard_layout, backup_arr_lookahead, backup_dep_lookahead, backup_arr_back, backup_dep_back, \
                backup_tracks, backup_tracks_used_length = yrd.backup_state_elements()

            backup_event_list = event_list.copy()

            while attempt <= 3:
                if attempt == 1:
                    action = agent.get_action(history)
                else:
                    yrd.set_state(backup_yard_layout, backup_arr_lookahead, backup_dep_lookahead, backup_arr_back,
                                  backup_dep_back, backup_tracks, backup_tracks_used_length)  # roll back state
                    event_list = backup_event_list.copy()  # roll back event list
                    action = agent.get_random_action()

                # NN outpus [0,8], yard takes [1,9]
                
                # based on that action now let environment go to new state
                event = event_list.iloc[0]
                # check if after this we are done... 
                done_ = True if len(event_list) == 1 else False  # then there is no next event
                if done_:
                    print("Reached the end of a problem!")
                if busy_relocating:
                    # here we do not drop an event from the event list.
                    coming_arrivals = event_list.loc[event_list['event_type'] == 'arrival'].reset_index(drop=True)
                    coming_departures = event_list.loc[event_list['event_type'] == 'departure'].reset_index(drop=True)

                    next_state, reward, done = yrd.reloc_destination_step(event, event_list, action+1, coming_arrivals, coming_departures, done_)
                    busy_relocating = False
                else: 
                    # These operations below are expensive: maybe just use indexing.
                    event_list.drop(event_list.index[:1], inplace=True)
                    coming_arrivals = event_list.loc[event_list['event_type'] == 'arrival'].reset_index(drop=True)
                    coming_departures = event_list.loc[event_list['event_type'] == 'departure'].reset_index(drop=True)
                    
                    # do step
                    next_state, reward, done = yrd.step(action+1, coming_arrivals, coming_departures, event, event_list, done_)
        
                    busy_relocating = True if reward == -0.99 else False
                    
                history_ = np.float32(np.reshape(next_state, (1, yrd.shape[0], yrd.shape[1], yrd.shape[2])))
                
                if visualization:
                    yrd.show_state(history_, new_cmap)
                    print('action = ', action+1)
                    print('reward = ', reward)

                
                agent.avg_q_max += np.amax(agent.model.predict(history)[0])  # log q_max for tensorboard.
                score += reward  # log direct reward of action

                # We save a <state, action, reward, next state> sample to the replay memory
                agent.replay_memory(history, action, reward, history_, done)
                agent.train_replay()  # train model (every step) on random batch.

                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()  # every some steps, we update the target model with model
                    
                # if we made a mistake we can do a second attempt. if we did the right thing just continue
                # at final step this means we do try 3 times
                # so we collect unnecessary data at that point since 'done' at the end is good (not bad)
                if done:
                    attempt += 1  #
                else:
                    break

            # now max three attempts have been done, go to the next state.
            history = history_  # next state now becomes the current state.
            t += 1  # next step in this episode
            total_t += attempt
            global_step += 1

            if done:  # based on what the environment returns.
                if global_step > agent.train_start:  # log about this episode.
                    stats = [score, agent.avg_q_max / float(total_t), t,
                             agent.avg_loss / float(total_t), agent.epsilon]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, episode + 1)

                if printcounter == 1000:  # print every 1000 episodes
                    print(datetime.datetime.now() - start_time)
                    print(datetime.datetime.now())
                    printcounter = 0
                    print("episode:", episode, "  score:", score, "  memory length:",
                          len(agent.memory), "  epsilon:", agent.epsilon,
                          "  global_step:", global_step, "  average_q:",
                          agent.avg_q_max / float(t), "  average loss:",
                          agent.avg_loss / float(t))
                    agent.model.save_weights(agent.summary_dir_name + "/weights.h5")

                agent.avg_q_max, agent.avg_loss = 0, 0  # reset for next episode.
                gc.collect()
                break  # break the while loop to end the episode when t <= len(train_arrivals)

    agent.model.save_weights(agent.summary_dir_name + "/weights.h5")
