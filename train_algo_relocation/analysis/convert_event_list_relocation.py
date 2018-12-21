#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:36:02 2017

@author: evertjanpeer
"""
import pandas as pd
import json

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
    for departure in out_list:
        for trainstel in departure['Treinstellen']:
            schermnaam.append(departure['Schermnaam'])
            spoor.append(departure['Spoor'])
            tijd.append(departure['Tijd'])
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


if __name__ == '__main__':
    in_list = json.load(open('in.json'))
    out_list = json.load(open('out.json'))
    event_list = convert_to_event_list(in_list, out_list)

