"""
DQNAgent based on work by RLCode team - Copyright (c) 2017 RLCode (MIT Licence)
https://github.com/rlcode/reinforcement-learning

Tailored to the TUSP by Evertjan Peer
KBH example

b2b_instances: in this version we add the changing instances to the training.

"""
import gc
import math
import json
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K
from kbh_yard_b2b import KBH_Env                               #This is the environment of the shunting yard
# import convert_event_list as convert
import datetime
import itertools
import pandas as pd
import time
import copy

#import instance generator
from data_retrieval_b2b import INSTANCEProvider

#visualize learning
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
#import colorsys


EPISODES = 60000# 60000# 50000 #used to be 500000 (note that this should be long enough to at least cover the exploration_steps)

class DQNAgent:
    def __init__(self, yard, load, model_nr):
        self.load_model = load
        # environment settings
        self.state_size = yard.get_state().shape            #this is specific to the state representation you choose for the problem.
        self.action_size = yard.nr_tracks                   #this as well

        # set epsilon
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1     #start fully random, end with 10% random action selection.
        self.exploration_steps = 150000#1000000.            #this determines how long one explores (nr. steps from 1 to 0.1 epsilon)
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # train params
        self.batch_size = 32                                #batch size used to update the neural nets from experience replay
        self.train_start = 10000                            #first collect some experiences using random move selection
        self.update_target_rate = 10000                     #after every 10000 actions update the target model
        self.discount_factor = 0.99
        self.memory = deque(maxlen=125000)                  #this is the memory the DQN samples experiences from for Experience Replay

        # build model
        self.model = self.build_model()                     #two models are maintained: a evaluation and target model.
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()                   #here a special type of optimizer is used: why which optimizer works in what situation?

        # stuff for tensorboard.
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()

        # time_stamp_dir = int(time.time())
        # print(time_stamp_dir)
        # self.summary_dir_name = 'summary/dqn_dummy/' + str(time_stamp_dir)
        # self.summary_writer = tf.summary.FileWriter(
        #     self.summary_dir_name, self.sess.graph)
        # self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("../summary/dqn_dummy/"+model_nr+"/weights.h5")
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

        optimizer = RMSprop(lr=0.00025, epsilon=0.01) #was lr = 0.00025
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
    #
#        model = Sequential()
#        model.add(Conv2D(32, (5, 5), padding='same', activation='relu',
#                         input_shape=self.state_size))
#        model.add(Conv2D(64, (3, 3), activation='relu'))
#        model.add(Flatten())
#        model.add(Dense(256, activation='relu'))
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

    def get_random_action(self, history):
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
            return                                      #do nothing when we are still in the 'warm up' period.
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step     #here is where we decay epsilon.

        mini_batch = random.sample(self.memory, self.batch_size)            #sample randomly from the memory

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

# # Creates a sequence of decisions to manage a sequence of trains based on the trained CNN
# # returns 'win' if the trains are successfully parked
# # returns 'lose' if not.
# def _testcase(test_trains):
#     t=1
#     raw_observation = yrd.reset(t, test_trains)
#     score = 0
#     state = conv.convert_one_state(raw_observation)
#     history = state.reshape((1, 4, 6, 1))
#
#     while t<=len(test_trains):
#             # RL choose action based on observation
#             action = agent.get_action(history)
#             action +=1 #correct for neural network output (0-5 ipv 1-6)
#
#             # RL take action and get next observation and reward
#             raw_observation_, reward, done = yrd.step(action, t, len(test_trains), test_trains, capacity)
#             state_ = conv.convert_one_state(raw_observation_)               #convert to NN input.
#
#             state_ = np.reshape([state_], (1, 4, 6, 1))                     #reshape to (1,4,6,1) tensor
#             history_ = state_
#
#             #Rewards are clipped in the Atari paper to [-1, 1]. However in my example succes gets a 1, and lose gets a -1 so not really needed.
#             reward = np.clip(reward, -1., 1.)
#             score += reward                                                 # log direct reward of action
#
#             # swap histories for the next step.
#             history = history_
#
#             t+=1
#
#             if done:
#                 #print(raw_observation_)
#                 if raw_observation_[0]=='lose':
#                     #print('lost')
#                     return 'lose'
#                 else:
#                     return 'win'
#                 break

###
# Helper functions to convert input to event_list
###
def encode_names(name):
    if (name == 'SLT4'):
        code = '14'
    elif (name == 'SLT6'):
        code = '16'
    elif (name == 'VIRM4'):
        code = '24'
    elif (name == 'VIRM6'):
        code = '26'
    elif (name == 'DDZ6'):
        code = '36'
    else:
        code = '99'
    return code

def convert_to_event_list(in_list, out_list):
    schermnaam = []
    spoor = []
    tijd = []
    treinstellen = []
    event_type = []
    length = []

    for arrival in in_list:
        schermnaam.append(arrival['Schermnaam'])
        spoor.append(arrival['Spoor'])
        tijd.append(arrival['Tijd'])
        event_type.append('arrival')

        stellen = []
        stellenids = []
        totallen = 0
        for trainstel in arrival['Treinstellen']:
            stellen.append(encode_names(trainstel['Materieel']))
            stellenids.append(trainstel['Id'])
            totallen += int(str(trainstel['Materieel'][-1:]))
        treinstellen.append(stellen)
        length.append(totallen)

    in_df = pd.DataFrame(
        {'name': schermnaam, 'track': spoor, 'time': tijd, 'composition': treinstellen, 'length': length, 'event_type': event_type})

    schermnaam = []
    spoor = []
    tijd = []
    treinstellen = []
    event_type = []
    length = []

    # the departures we treat as if they are all separate train units.
    # the departures that happen at the same time we consider as being combined on the departure track
    for arrival in out_list:
        for trainstel in arrival['Treinstellen']:
            schermnaam.append(arrival['Schermnaam'])
            spoor.append(arrival['Spoor'])
            tijd.append(arrival['Tijd'])
            event_type.append('departure')
            treinstellen.append(encode_names(trainstel['Materieel']))
            length.append(int(str(trainstel['Materieel'][-1:])))
            # stellen = []
            # stellenids = []
            # for trainstel in arrival['Treinstellen']:
            #     stellen.append(trainstel['Materieel'])
            #     stellenids.append(trainstel['Id'])
            # treinstellen.append(str(stellen))

    out_df = pd.DataFrame(
        {'name': schermnaam, 'track': spoor, 'time': tijd, 'composition': treinstellen, 'length': length, 'event_type': event_type})

    event_list = pd.concat([in_df, out_df]).sort_values('time').reset_index(drop=True)
    return(event_list)

# this function returns random colors for visualisation of learning.
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

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

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


# This is the main RL loop.
if __name__ == "__main__":
    #Initialize problem instances generator
    instance_prov = INSTANCEProvider()

    start_time = datetime.datetime.now()
    print('start time: ', start_time)

    #Create environment KBH
    yrd = KBH_Env()

    #init counter for progress in console
    printcounter = 0

    #Create the DQNAgent with the CNN approximation of the Q-function and its experience replay and training functions.
    agent = DQNAgent(yrd, False)

    #Init containers for results.
    scores, episodes, global_step = [], [], 0

    #visualize learning
    #new_cmap = rand_cmap(200, type='soft', first_color_black=True, last_color_black=False, verbose=True)

    #We will execute #EPISODES episodes of the problem.
    for episode in range(EPISODES):
        #get random instance
        event_list = instance_prov.get_random_instance()
        steps = len(event_list)
        t = 0                                                          #stepcounter within each episode.
        printcounter +=1

        score = 0                                                       #Init score variable for this particular episode.

        state = yrd.reset(event_list)                        #Get first observation based on the first train arrival.
        history = np.float32(np.reshape(state, (1, yrd.shape[0], yrd.shape[1], yrd.shape[2])))                     #reshape state into tensor, which we call history.

        total_t = 0
        while t <= steps:
            # this is to try again when you make a mistake in a state.
            attempt = 1

            done = False
            backup_yard_layout = yrd.yard_layout.copy()
            backup_arr_lookahead = yrd.arrival_lookahead.copy()
            backup_dep_lookahead = yrd.departure_lookahead.copy()
            backup_arr_back = yrd.arrival_backlog.copy()
            backup_dep_back = yrd.departure_backlog.copy()
            backup_tracks = copy.deepcopy(yrd.tracks)
            backup_tracks_used_length = copy.deepcopy(yrd.tracks_used_length)

            backup_event_list = event_list.copy()

            while attempt <= 3:
                #reset state
                yrd.yard_layout = backup_yard_layout.copy()
                yrd.arrival_lookahead = backup_arr_lookahead.copy()
                yrd.departure_lookahead = backup_dep_lookahead.copy()
                yrd.arrival_backlog = backup_arr_back.copy()
                yrd.departure_backlog = backup_dep_back.copy()
                yrd.tracks = copy.deepcopy(backup_tracks)
                yrd.tracks_used_length = copy.deepcopy(backup_tracks_used_length)
                # yrd.set_state(backup_yard_layout, backup_arr_lookahead, backup_dep_lookahead, backup_arr_back,
                #               backup_dep_back, backup_tracks, backup_tracks_used_length)
                event_list = backup_event_list.copy()

                if attempt == 1:
                    action = agent.get_action(history)                          # RL choose action based on observation
                else:
                    action = agent.get_random_action(history)
                # RL take action and get next observation and reward
                # note the +1 at action

                raw_observation_, reward, done = yrd.step(action+1, event_list)
                history_ = np.float32(np.reshape(raw_observation_, (1, yrd.shape[0], yrd.shape[1], yrd.shape[2])))                 #reshape to tensor.

                agent.avg_q_max += np.amax(                                 #log q_max for tensorboard.
                    agent.model.predict(history)[0])

#                reward = np.clip(reward, -1., 1.)                           #clip rewards (not really needed
                score += reward                                             #log direct reward of action

                # show action
                #plt.imshow(np.float32(history_[0][0]), cmap=new_cmap, interpolation='nearest')
                #plt.show()
                #time.sleep(0.5)
                #print(action+1)
                #plt.close()
                #print(reward)

                # We save a <state, action, reward, next state> sample to the replay memory
                agent.replay_memory(history, action, reward, history_, done)
#                print(history)
#                print(action)
#                print(reward)
#                print(history_)
#                print(done)
                # train model (every step) on rfandom batch.
                agent.train_replay()
                # every some steps, we update the target model with model
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()
                #print(done)
                if done == False:
#                    print('ok, action is ok')
                    break; # we no not need to try another action
                else:
                    # try again
#                    print('try again')
                    attempt += 1

            history = history_                                          # next state now becomes the current state.
            t += 1                                                        # next step in this episode
            total_t += attempt
            global_step += 1

            if done:                                                    # based on what the environment returns.
                if global_step > agent.train_start:                     # log about this episode.
                    stats = [score, agent.avg_q_max / float(total_t), t,
                             agent.avg_loss / float(total_t), agent.epsilon]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, episode + 1)

                if printcounter == 1000:                                # print every 1000 episodes
                    print(datetime.datetime.now()-start_time)
                    print(datetime.datetime.now())
                    printcounter=0
                    print("episode:", episode, "  score:", score, "  memory length:",
                          len(agent.memory), "  epsilon:", agent.epsilon,
                          "  global_step:", global_step, "  average_q:",
                          agent.avg_q_max / float(t), "  average loss:",
                          agent.avg_loss / float(t))
                    agent.model.save_weights(agent.summary_dir_name+"/weights.h5")

                agent.avg_q_max, agent.avg_loss = 0, 0                  # reset for next episode.
                gc.collect()
                break
                                                   # break the while loop to end the episode when t <= len(train_arrivals)

    # #
    # # After training test the trained RL Agent on how well
    #å # it is able to handle all 3150 unique arrival sequences.
    # #
    # agent.epsilon = 0                                                   # set epsilon to 0 to remove random behavior.
    # failures = []
    # for perm in list(set(itertools.permutations(problem_instance))):
    #     result = _testcase(perm)
    #     if result=='lose':
    #         failures.append(perm)
    #
    # print("success: ", len(list(set(itertools.permutations(problem_instance))))-len(failures))
    # print("failures: ", len(failures))
    # print(failures)
    #

    agent.model.save_weights(agent.summary_dir_name+"/weights.h5")
