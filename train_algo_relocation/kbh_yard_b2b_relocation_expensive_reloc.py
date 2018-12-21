#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code inspired by https://morvanzhou.github.io/tutorials/
altered to the TUSP by Evertjan Peer
"""
import numpy as np
import time
import sys
import math
import copy
import convert_event_list_relocation as convert
import json
import pandas as pd
import random
import time

#visualize learning
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys

# import instance generator
from data_retrieval_30000_14151617 import INSTANCEProvider

class KBH_Env(object):
    def __init__(self):
        self.state_dims = (33, 33)
        self.track_lengths = [10, 7, 7, 9, 14, 14, 15, 19, 12]  # last one is relocation track
        self.nr_tracks = len(self.track_lengths)
        self.max_window_size = 5  # number of events we can look in the future.

        # Action 1 - 8 always correspond to a track. 9 depends on the type of state.
        # We have: Arrival/Departure/RelocationDecision/RelocationDestination states
        self.n_actions = self.nr_tracks
        self.title = 'KBHRelocation'

        # We use two lists to keep track of current occupation of tracks
        self.tracks = [[] for k in range(self.nr_tracks)]  # to keep track of trains on tracks
        self.tracks_used_length = [0 for k in range(self.nr_tracks)]

        # A state consists of 5 parts
        # yard layout (incl relocation track)
        # arrival lookahead state
        # departure lookahead state
        # arrival backlog state
        # departure backlog state

        # Init yard components:
        self.yard_layout = np.zeros((self.nr_tracks + 2, self.state_dims[0]), dtype=np.int)
        self.arrival_lookahead = np.zeros((5, self.state_dims[0]), dtype=np.int)
        self.departure_lookahead = np.zeros((5, self.state_dims[0]), dtype=np.int)
        self.arrival_backlog = np.zeros((4, self.state_dims[0]), dtype=np.int)
        self.departure_backlog = np.zeros((4, self.state_dims[0]), dtype=np.int)

        # Self.state contains our visual state representation
        self.state = self.get_state()
        self.shape = self.state.shape
        
        # Init final states
        self.win_state = np.full(self.shape, 1, dtype=int)
        self.lose_state = np.full(self.shape, -1, dtype=int)

    def get_state(self):
        """Returns 'visual' state by concatenating current state components."""
        divider = np.zeros((1, self.state_dims[0]), dtype=np.int)
        return np.reshape((np.concatenate((self.arrival_backlog, divider, self.arrival_lookahead, divider,
                                           self.yard_layout, divider, self.departure_lookahead, divider,
                                           self.departure_backlog), axis=0)),
                          (1, self.state_dims[0], self.state_dims[0]))

    def set_state(self, backup_yard_layout, backup_arr_lookahead, backup_dep_lookahead, backup_arr_back,
                  backup_dep_back, backup_tracks, backup_tracks_used_length):
        """Takes all backup components used to reset the current state and reverts the state."""
        self.yard_layout = backup_yard_layout.copy()
        self.arrival_lookahead = backup_arr_lookahead.copy()
        self.departure_lookahead = backup_dep_lookahead.copy()
        self.arrival_backlog = backup_arr_back.copy()
        self.departure_backlog = backup_dep_back.copy()
        self.tracks = copy.deepcopy(backup_tracks)
        self.tracks_used_length = copy.deepcopy(backup_tracks_used_length)
        self.state = self.get_state()

    def update_arrival_lookahead(self, problem_arrivals):
        """Takes arriving events and returns 'visual' representation of this lookahead"""
        arrival_lookahead = np.zeros((self.max_window_size, self.state_dims[0]), dtype=np.int)
        window = min(len(problem_arrivals), self.max_window_size)
        for i in range(0, window):
            look_arr = []
            arr = problem_arrivals['composition'][i]
            for part in arr:
                part_length = int(part[-1:])
                look_arr.extend([part]*part_length)
            look_arr.extend([0]*(self.state_dims[0]-len(look_arr)))
            arrival_lookahead[self.max_window_size - 1 - i] = look_arr
        return arrival_lookahead

    def update_departure_lookahead(self, problem_departures):
        """Takes departure events and returns 'visual' representation of this lookahead"""
        departure_lookahead = np.zeros((self.max_window_size, self.state_dims[0]), dtype=np.int)
        window = min(len(problem_departures), self.max_window_size)
        for i in range(0, window):
            look_dep = []
            dep = [problem_departures['composition'][i]]
            for part in dep:
                part_length = int(part[-1:])
                look_dep.extend([part]*part_length)
            look_dep.extend([0]*(self.state_dims[0]-len(look_dep)))
            departure_lookahead[i] = look_dep
        return departure_lookahead

    def update_arrival_backlog(self, problem_arrivals):
        """Takes the problem arrivals, and returns 'visual' representation of arrival backlog"""
        # the first 5 are already in the lookahead, so we work from there
        if len(problem_arrivals['length'] > 5):
            arr_backlog_sum = sum(problem_arrivals['length'][5:])
        else:
            arr_backlog_sum = 0
        ones = np.ones(arr_backlog_sum, dtype=np.int)
        zeros = np.zeros((4 * self.state_dims[0] - len(ones)), dtype=np.int)
        arr_backlog = np.concatenate((zeros, ones), axis=0).reshape((4, self.state_dims[0]))
        return arr_backlog

    def update_departure_backlog(self, problem_departures):
        """Takes the problem departures, and returns 'visual' representation of departure backlog"""
        # the first 5 are already in the lookahead, so we work rom there
        if len(problem_departures['length'] > 5):
            dep_backlog_sum = sum(problem_departures['length'][5:])
        else:
            dep_backlog_sum = 0
        ones = np.ones(dep_backlog_sum, dtype=np.int)
        zeros = np.zeros((4 * self.state_dims[0] - len(ones)), dtype=np.int)
        dep_backlog = np.concatenate((ones, zeros), axis=0).reshape((4, self.state_dims[0]))
        return dep_backlog

    def sub_method_tracks_to_visual(self):
        """Converts track occupation lists to visual representation"""
        kbh_tracks = np.zeros((self.nr_tracks, self.state_dims[0]), dtype=np.int)
        track_iterator = 0
        for track in self.tracks:
            specific_track = []
            for train in track:
                part_length = int(train[-1:])
                specific_track.extend([train]*part_length)
            # fill up with zeros until track length
            specific_track = [0]*((self.track_lengths[track_iterator] - len(specific_track))) + specific_track
            # fill up with 9's to make it a grid.
            if track_iterator == self.nr_tracks - 1:
                specific_track = [-99]*(self.state_dims[0] - len(specific_track))+specific_track
            else:
                specific_track.extend([-99]*(self.state_dims[0] - len(specific_track)))
            kbh_tracks[track_iterator] = specific_track
            track_iterator += 1
        return kbh_tracks

    def sub_method_arrival_departure_track_to_visual(self, next_event, done_):
        """Returns visual components for departure track and arrival track"""
        departure_tr = np.zeros((1, self.state_dims[0]), dtype=np.int)
        arriving_tr = np.zeros((1, self.state_dims[0]), dtype=np.int)
        if not done_:
            if next_event['event_type'] == 'arrival':
                arriving_tr = []
                arrival = next_event['composition']
                for part in arrival:
                    part_length = int(part[-1:])
                    arriving_tr.extend([part]*part_length)
                arriving_tr.extend([0]*(self.state_dims[0] - len(arriving_tr)))
                # departure track remains emtpy cuz next event = arrival

            if next_event['event_type'] == 'departure':
                departure_tr = []
                departure = [next_event['composition']]
                for part in departure:
                    part_length = int(part[-1:])
                    departure_tr.extend([part]*part_length)
                departure_tr.extend([0]*(self.state_dims[0] - len(departure_tr)))
                # arrival track remains emtpy cuz next event = departure
        return arriving_tr, departure_tr
    
    def update_yard_layout_no_next(self):
        """Returns visual representation of yard layout - no arriving and no departing train"""
        kbh_tracks = self.sub_method_tracks_to_visual()
        arriving_tr, departure_tr = np.zeros((1, self.state_dims[0]), dtype=np.int), np.zeros((1, self.state_dims[0]), dtype=np.int)
        yard_layout = np.vstack((arriving_tr, kbh_tracks, departure_tr))
        return yard_layout

    def update_yard_layout(self, next_event, done_):
        """Takes next event and returns visual representation of yard layout"""
        kbh_tracks = self.sub_method_tracks_to_visual()
        arriving_tr, departure_tr = self.sub_method_arrival_departure_track_to_visual(next_event, done_)
        yard_layout = np.vstack((arriving_tr, kbh_tracks, departure_tr))
        return yard_layout

    def reset(self, problem_instance):
        """Takes a (new) problem_instance and initializes the yard components and visual state. Returns state"""
        problem_arrivals = problem_instance.loc[problem_instance['event_type'] == 'arrival'].reset_index(drop=True)
        problem_departures = problem_instance.loc[problem_instance['event_type'] == 'departure'].reset_index(drop=True)

        # reset tracks
        self.tracks = [[] for k in range(self.nr_tracks)]  # to keep track of trains on tracks
        self.tracks_used_length = [0 for k in range(self.nr_tracks)]

        # obtain first event
        next_event_ = problem_instance.iloc[0]
        # reset class attributes
        self.yard_layout = self.update_yard_layout(next_event_, False)
        # load other parts of state space
        self.arrival_lookahead = self.update_arrival_lookahead(problem_arrivals)
        self.departure_lookahead = self.update_departure_lookahead(problem_departures)
        self.arrival_backlog = self.update_arrival_backlog(problem_arrivals)
        self.departure_backlog = self.update_departure_backlog(problem_departures)

        state = self.get_state()
        return state

    def sub_method_step_arrival(self, event, event_list, action, coming_arrivals, coming_departures, done_):
        """The chosen action will cause the arriving train to be placed on that track"""
        if action == 9:  # action 9 is not valid here.
            next_state_, reward, done_ = self.lose_state, -1, True
        else: 
            arriving_train = event['composition']
            arriving_train_length = event['length']
    
            insert_index = 0
            for train in arriving_train:
                self.tracks[action - 1].insert(insert_index, train)
                insert_index += 1
            self.tracks_used_length[action - 1] += arriving_train_length
    
            overflow = True if self.tracks_used_length[action - 1] > self.track_lengths[action - 1] else False
    
            if overflow:  # if we go wrong somewhere, done.
                next_state_, reward, done_ = self.lose_state, -1, True
            else:  # we are still good.
                if done_:  # no new event, we won!
                    next_state_, reward = self.win_state, 1
                else:  # we are not done, another event happening.
                    next_event_ = event_list.iloc[0]
                    self.yard_layout = self.update_yard_layout(next_event_, done_)
                    self.arrival_lookahead = self.update_arrival_lookahead(coming_arrivals)
                    self.departure_lookahead = self.update_departure_lookahead(coming_departures)
                    self.arrival_backlog = self.update_arrival_backlog(coming_arrivals)
                    self.departure_backlog = self.update_departure_backlog(coming_departures)
    
                    next_state_, reward = self.get_state(), 0.5
        return next_state_, reward, done_

    def sub_method_step_departure(self, event, event_list, action, coming_arrivals, coming_departures, done_):
        """The chosen action will cause the leftmost train on that track to be removed (if any)"""
        if self.tracks_used_length[action - 1] == 0: # fail, you have chosen an empty track to provide a train from.
            next_state_, reward, done_ = self.lose_state, -1, True
        else: # we have delivered a train..
            # did we do good?
            removed_train = self.tracks[action - 1].pop(0)  #remove from list
            self.tracks_used_length[action - 1] -= int(removed_train[-1:])  #deduct from used length

            if removed_train != event['composition']: # only if removed_train == asked train, then ok
                next_state_, reward, done_ = self.lose_state, -1, True
            else:  # still good! we provided the right train.
                if done_: # no new event, we won!
                    next_state_, reward = self.win_state, 1
                else:  # we are not done, another event happening.
                    next_event_ = event_list.iloc[0]
                    self.yard_layout = self.update_yard_layout(next_event_, done_)
                    self.arrival_lookahead = self.update_arrival_lookahead(coming_arrivals)
                    self.departure_lookahead = self.update_departure_lookahead(coming_departures)
                    self.arrival_backlog = self.update_arrival_backlog(coming_arrivals)
                    self.departure_backlog = self.update_departure_backlog(coming_departures)

                    next_state_, reward = self.get_state(), 1

        return next_state_, reward, done_
    
    def sub_method_step_relocation_opp(self, event, event_list, action, coming_arrivals, coming_departures, done_):
        if action == 9:
            # do not relocate, move to departure event. 
            next_event_ = event_list.iloc[0]
            self.yard_layout = self.update_yard_layout(next_event_, done_)
            next_state_, reward = self.get_state(), 0
        else:           
            # try to relocate the train to the relocation track.
            if len(self.tracks[action-1])>0:
                reloc_train = self.tracks[action-1].pop()
                self.tracks_used_length[action-1] -= int(reloc_train[-1:])
                self.tracks[-1].append(reloc_train)
                
                self.yard_layout = self.update_yard_layout_no_next()
                
                next_state_, reward = self.get_state(), -0.99
            else: 
                next_state_, reward, done_ = self.lose_state, -1, True
        return next_state_, reward, done_
    
    def reloc_destination_step(self, event, event_list, action, coming_arrivals, coming_departures, done_):
        if action == 9:  # action 9 is not valid here.
            next_state_, reward, done_ = self.lose_state, -1, True
        else: 
            reloc_train = self.tracks[-1].pop()
    
            # put train at the end of track
            self.tracks[action - 1].append(reloc_train)
            self.tracks_used_length[action - 1] += int(reloc_train[-1:])
    
            overflow = True if self.tracks_used_length[action - 1] > self.track_lengths[action - 1] else False
    
            if overflow:  # if we go wrong somewhere, done.
                next_state_, reward, done_ = self.lose_state, -1, True
            else:  # we are still good.
                if done_:  # no new event, we won!
                    next_state_, reward = self.win_state, 1
                else:  # we are not done, another event happening.
                    next_event_ = event_list.iloc[0]
                    self.yard_layout = self.update_yard_layout(next_event_, done_)
                    self.arrival_lookahead = self.update_arrival_lookahead(coming_arrivals)
                    self.departure_lookahead = self.update_departure_lookahead(coming_departures)
                    self.arrival_backlog = self.update_arrival_backlog(coming_arrivals)
                    self.departure_backlog = self.update_departure_backlog(coming_departures)
    
                    next_state_, reward = self.get_state(), 0
        return next_state_, reward, done_

            

    # This emulates the environment that responds to the action taken. 
    # It returns the next state, the reward and whether the episode is done. 
    def step(self, action, coming_arrivals, coming_departures, event, event_list, done_):
        if event['event_type'] == 'arrival':
            next_state_, reward, done_ = self.sub_method_step_arrival(event, event_list, action, coming_arrivals,
                                                                      coming_departures, done_)

        if event['event_type'] == 'departure':
            next_state_, reward, done_ = self.sub_method_step_departure(event, event_list, action, coming_arrivals,
                                                                        coming_departures, done_)
            
        if event['event_type'] == 'relocation_opp':
            next_state_, reward, done_ = self.sub_method_step_relocation_opp(event, event_list, action, coming_arrivals,
                                                                        coming_departures, done_)
            


        return next_state_, reward, done_

    def backup_state_elements(self):
        return self.yard_layout.copy(), self.arrival_lookahead.copy(), self.departure_lookahead.copy(), self.arrival_backlog.copy(), self.departure_backlog.copy(), copy.deepcopy(
            self.tracks), copy.deepcopy(self.tracks_used_length)
        
    def show_state(self, state, new_cmap):
        plt.imshow(np.float32(state[0][0]), cmap=new_cmap, interpolation='nearest')
        plt.show()
        time.sleep(0.5)


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



if __name__ == '__main__':
    #visualize learning
    new_cmap = rand_cmap(200, type='soft', first_color_black=True, last_color_black=False, verbose=True)

    #instance_prov = INSTANCEProvider()    
    #event_list = instance_prov.get_random_instance()
    #event_list.to_pickle('temp_event_list')
    event_list = pd.read_pickle('temp_event_list')
    
    # later we will go through a full episode every time, so try that out by the step functions.
    # steps equal to length of event list.
    steps = len(event_list)
    t = 0
    start_time = time.time()

    yard = KBH_Env()
    current_state = yard.reset(event_list)

    done = False
    busy_relocating = False
    while not done:
        yard.show_current_state()
        event = event_list.iloc[0]
        # check if after this we are done... 
        done_ = True if len(event_list) == 1 else False  # then there is no next event

        if busy_relocating: 
            # pick action that relocates back
            print('do the second step of relocating')
            #action = int(input("Pick an action [1,9]: "))
            action = random.randint(1,9)
            next_state, reward, done = yard.reloc_destination_step(action, event, event_list, done_)
            history_ = np.float32(np.reshape(next_state, (1, yard.shape[0], yard.shape[1], yard.shape[2])))
            busy_relocating = False
        else:
            # These operations below are expensive: maybe just use indexing.
            event_list.drop(event_list.index[:1], inplace=True)
            coming_arrivals = event_list.loc[event_list['event_type'] == 'arrival'].reset_index(drop=True)
            coming_departures = event_list.loc[event_list['event_type'] == 'departure'].reset_index(drop=True)
            
            action = random.randint(1,9)
            # do step
            next_state, reward, done = yard.step(action, coming_arrivals, coming_departures, event, event_list, done_)
            history_ = np.float32(np.reshape(next_state, (1, yard.shape[0], yard.shape[1], yard.shape[2])))

            busy_relocating = True if reward == -0.99 else False
        
        print('action = ', action)
        print('reward = ', reward)

        if done:
            plt.imshow(np.float32(history_[0][0]), cmap=new_cmap, interpolation='nearest')
            plt.show()
            time.sleep(0.5)

            print('done')
            print("--- %s seconds ---" % (time.time() - start_time))

            break
        t = t + 1