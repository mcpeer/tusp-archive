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
import convert_event_list as convert
import json
import pandas as pd
import random

class KBH_Env(object):
    def __init__(self):
        self.state_dims = (33,33)
        
        self.arrival_track = 1
        self.track_lengths = [10, 7, 7, 9, 14, 14, 15, 19, 21] #hardcoded binckhorst layout
        self.nr_tracks = len(self.track_lengths)
        self.departure_track = 1

        # if a train is arriving, an action will decide on which track the train to put
        # if a train is asked to depart, the action will trigger the first train of that track to leave.
        self.n_actions = self.nr_tracks

        self.title = 'KBHFifo'

        # A state consists of 5 parts
        # yard layout
        # arrival lookahead state
        # departure lookahead state
        # arrival backlog state
        # departure backlog state
        
        self.yard_layout = np.zeros((len(self.track_lengths)+2, self.state_dims[0]), dtype=np.int)
        self.arrival_lookahead = np.zeros((5, self.state_dims[0]), dtype=np.int)
        self.departure_lookahead = np.zeros((5, self.state_dims[0]), dtype=np.int)
        self.arrival_backlog = np.zeros((4, self.state_dims[0]), dtype=np.int)
        self.departure_backlog = np.zeros((4, self.state_dims[0]), dtype=np.int)
        
        self.state = self.get_state()
        self.shape = self.state.shape
        
        self.tracks = [[] for k in range(self.nr_tracks)]                   # to keep track of trains on tracks
        self.tracks_used_length = [0 for k in range(self.nr_tracks)]
                
    def get_state(self):
        divider = np.zeros((1,self.state_dims[0]), dtype=np.int)
        return np.reshape((np.concatenate((self.arrival_backlog, divider, self.arrival_lookahead, divider, self.yard_layout, divider, self.departure_lookahead, divider, self.departure_backlog), axis=0)), (1,self.state_dims[0], self.state_dims[0]))
    
    def set_state(self, backup_yard_layout, backup_arr_lookahead, backup_dep_lookahead, backup_arr_back, backup_dep_back, backup_tracks, backup_tracks_used_length):
        self.yard_layout = backup_yard_layout
        self.arrival_lookahead = backup_arr_lookahead
        self.departure_lookahead = backup_dep_lookahead
        self.arrival_backlog = backup_arr_back
        self.departure_backlog = backup_dep_back
        self.tracks = backup_tracks
        self.tracks_used_length = backup_tracks_used_length
        self.state = self.get_state()

    def update_arrival_lookahead(self, problem_arrivals):
        max_window_size = 5
        arrival_lookahead = np.zeros((max_window_size, self.state_dims[0]), dtype=np.int)
        #lookahead max 5
        window = min(len(problem_arrivals), max_window_size)
        for i in range(0, window):
            look_arr = []
            arr = problem_arrivals['composition'][i]
            for part in arr:
                part_length = int(part[-1:])
                for j in range(0, part_length):
                    look_arr.append(part)
            for j in range(0, self.state_dims[0] - len(look_arr)):
                look_arr.append(0)
            arrival_lookahead[max_window_size-1-i] = look_arr
        return arrival_lookahead

    def update_departure_lookahead(self, problem_departures):
        max_window_size = 5
        departure_lookahead = np.zeros((max_window_size,self.state_dims[0]), dtype=np.int)
        #lookahead max 5
        window = min(len(problem_departures), 5)
        for i in range(0, window):
            look_dep = []
            dep = [problem_departures['composition'][i]]
            for part in dep:
                part_length = int(part[-1:])
                for j in range(0, part_length):
                    look_dep.append(part)
            for j in range(0, self.state_dims[0] - len(look_dep)):
                look_dep.append(0)
            departure_lookahead[max_window_size-1-i] = look_dep
        return departure_lookahead

    def update_arrival_backlog(self, problem_arrivals):
        # the first 5 are already in the lookahead, so we work from there
        if len(problem_arrivals['length'] > 5):
            arr_backlog_sum = sum(problem_arrivals['length'][5:])
        else:
            arr_backlog_sum = 0
        ones = np.ones(arr_backlog_sum,  dtype=np.int)
        zeros = np.zeros((4*self.state_dims[0]-len(ones)),  dtype=np.int)
        arr_backlog = np.concatenate((zeros, ones), axis=0).reshape((4,self.state_dims[0]))
        return arr_backlog

    def update_departure_backlog(self, problem_departures):
        # the first 5 are already in the lookahead, so we work rom there
        if len(problem_departures['length'] > 5):
            dep_backlog_sum = sum(problem_departures['length'][5:])
        else:
            dep_backlog_sum = 0
        ones = np.ones(dep_backlog_sum, dtype=np.int)
        zeros = np.zeros((4*self.state_dims[0]-len(ones)), dtype=np.int)
        dep_backlog = np.concatenate((ones, zeros), axis=0).reshape((4,self.state_dims[0]))
        return dep_backlog

    def update_yard_layout(self, next_event, done_):
        kbh_tracks = []
        track_iterator = 0
        for track in self.tracks:
            specific_track = []
            for train in track:
                part_length = int(train[-1:])
                for i in range(0, part_length):
                    specific_track.append(train)
            #fill up with zeros untill track length 
            for i in range(0, self.track_lengths[track_iterator] - len(specific_track)):
                specific_track.insert(0,0)
            #fill up with 9's to make it a grid.  
            for i in range(0, self.state_dims[0] - len(specific_track)):
                specific_track.append(-99)
            kbh_tracks.append(specific_track)
            track_iterator+=1
        if done_:
            # no new event, so arriving and departure track is empty
            departure_tr = np.zeros((1, self.state_dims[0]), dtype=np.int)
            arriving_tr = np.zeros((1, self.state_dims[0]), dtype=np.int)
        else:
            if next_event['event_type'] == 'arrival':
                arriving_tr = []
                arrival = next_event['composition']
                for part in arrival:
                    part_length = int(part[-1:])
                    for i in range(0, part_length):
                        arriving_tr.append(part)
                for i in range(0, self.state_dims[0] - len(arriving_tr)):
                    arriving_tr.append(0)

                # departure track remains emtpy cuz next event = arrival
                departure_tr = np.zeros((1, self.state_dims[0]), dtype=np.int)

            if next_event['event_type'] == 'departure':
                departure_tr = []
                departure = [next_event['composition']]
                for part in departure:
                    part_length = int(part[-1:])
                    for i in range(0, part_length):
                        departure_tr.append(part)
                for i in range(0, self.state_dims[0] - len(departure_tr)):
                    departure_tr.append(0)

                # arrival track remains emtpy cuz next event = departure
                arriving_tr = np.zeros((1, self.state_dims[0]), dtype=np.int)

        # combine them to yard_layout
        yard_layout = np.vstack((arriving_tr, kbh_tracks, departure_tr))

        return yard_layout

    # Resetting the yard means starting completely empty and
    # subsequently loading the first arriving train onto the arrival track.
    def reset(self, problem_instance):
        problem_arrivals = problem_instance.loc[problem_instance['event_type'] == 'arrival'].reset_index(drop=True)
        problem_departures = problem_instance.loc[problem_instance['event_type'] == 'departure'].reset_index(drop=True)

        # reset tracks
        self.tracks = [[] for k in range(self.nr_tracks)]                   # to keep track of trains on tracks
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
        return(state)
    
    # This emulates the environment that responds to the action taken. 
    # It returns the next state, the reward and whether the episode is done. 
    def step(self, action, event_list):
        done_ = False

        # obtain first event
        event = event_list.iloc[0]

        event_list.drop(event_list.index[:1], inplace=True)
        coming_arrivals = event_list.loc[event_list['event_type'] == 'arrival'].reset_index(drop=True)
        coming_departures = event_list.loc[event_list['event_type'] == 'departure'].reset_index(drop=True)

        if len(event_list) == 0:
            # then there is no next event
            done_ = True
        else:
            next_event_ = event_list.iloc[0]
        #print(event['event_type'])
        if event['event_type'] == 'arrival':
            # then choosing an action will cause the arriving train to be placed on that track
            arriving_train = event['composition']
            arriving_train_length = event['length']

            insert_index = 0
            for train in arriving_train:
                self.tracks[action - 1].insert(insert_index, train)
                insert_index += 1
            self.tracks_used_length[action-1] += arriving_train_length

            # check for overflow
            overflow = False
            for used_length, capacity in zip(self.tracks_used_length, self.track_lengths):
                if used_length > capacity:                          # if this is the case, one of the tracks is too full
                    overflow = True
                    # print('overflow')
            if overflow:                                            # if we go wrong somewhere, done.
                reward = -1
                next_state_ = np.full(self.get_state().shape, -1, dtype=int)
                done_ = True
            else:                                                   # we are still good.
                if done_:                                           # no new event, we won!
                    next_state_ = np.full(self.get_state().shape, 1, dtype=int)
                    reward = 1
                else:                                               # we are not done, another event happening.
                    reward = 0.5
                    self.yard_layout = self.update_yard_layout(next_event_, done_)
                    self.arrival_lookahead = self.update_arrival_lookahead(coming_arrivals)
                    self.departure_lookahead = self.update_departure_lookahead(coming_departures)
                    self.arrival_backlog = self.update_arrival_backlog(coming_arrivals)
                    self.departure_backlog = self.update_departure_backlog(coming_departures)

                    next_state_ = self.get_state()

        if event['event_type'] == 'departure':
            # then choosing an action remove the leftmost train on that track
            if self.tracks_used_length[action-1] == 0:
                # fail, you have chosen an empty track to provide a train from.
                reward = -1
                next_state_ = np.full(self.get_state().shape, -1, dtype=int)
                done_ = True
            else:
                # at least we have delivered a train..
                # did we do good?
                removed_train = self.tracks[action-1][0]
                self.tracks[action-1].pop(0)
                self.tracks_used_length[action-1] -= int(removed_train[-1:])

                # only if removed_train == asked train, then ok
                # print(removed_train)
                # print(event['composition'])
                if removed_train != event['composition']:
                    # fail, you have chosen an empty track to provide a train from.
                    reward = -1
                    next_state_ = np.full(self.get_state().shape, -1, dtype=int)
                    done_ = True
                else:
                    # still good! we provided the right train.
                    if done_:  # no new event, we won!
                        next_state_ = np.full(self.get_state().shape, 1, dtype=int)
                        reward = 1
                    else:  # we are not done, another event happening.
                        reward = 1
                        self.yard_layout = self.update_yard_layout(next_event_, done_)
                        self.arrival_lookahead = self.update_arrival_lookahead(coming_arrivals)
                        self.departure_lookahead = self.update_departure_lookahead(coming_departures)
                        self.arrival_backlog = self.update_arrival_backlog(coming_arrivals)
                        self.departure_backlog = self.update_departure_backlog(coming_departures)

                        next_state_ = self.get_state()

        return next_state_, reward, done_
        
        
if __name__ == '__main__':
    in_list = json.load(open('in.json'))
    out_list = json.load(open('out.json'))
    event_list = convert.convert_to_event_list(in_list, out_list)
    event_list_original = convert.convert_to_event_list(in_list, out_list)
    #later we will go through a full episode every time, so try that out by the step functions.
    #steps equal to length of event list.
    steps = len(event_list)
    t = 0
    yard = KBH_Env()
    current_state = yard.reset(event_list)

    #solution
    actions = [5,8,6,2,3,2,8,5,1,
               7,5,4,7,9,9,
               5,6,6,3,4,4,9,7,9,9,9,8,
               5,5,7,7,1,8,8]
    while t < len(actions):
        # get action from memory or greedy random things.
        action = actions[t]
        # do step
        next_state, reward, done = yard.step(action, event_list)
        if done:
            print('done')
            break
        t = t + 1

        current_state = next_state
