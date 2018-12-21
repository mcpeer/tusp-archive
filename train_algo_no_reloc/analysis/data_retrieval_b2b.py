# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import dateutil.parser as p
from datetime import timedelta
#import matplotlib.pyplot as plt

class INSTANCEProvider:
    def __init__(self):
        self.data = pd.read_json('data_prep/instances.json')
        self.instances = self.data.instance_id.unique()
        self.data = self.preprocess_dates(self.data)
        self.nr_instances = len(self.instances)
        #self.instance_sizes = pd.value_counts(self.data.instance_id.values, sort=True)
        #plt.hist(instance_sizes)

        ## Check the train types in this dataset.
        ## Check the train lengths in this dataset.
        #pd.value_counts(data.material_type, sort=True)
        #pd.value_counts(data.carriages, sort=True)
        # Only train types of VIRM, SLT length 4 and 6.

    def get_random_instance(self):
        good = False
        while good == False:
            instance = self.create_correct_event_list(self.get_events(self.data, self.instances[np.random.randint(self.nr_instances)]))
            if len(instance) < 35:
                good = True
        return instance

    def get_random_instance_and_id(self):
        good = False
        while good == False:
            random_instance_id = self.instances[np.random.randint(self.nr_instances)]
            instance = self.create_correct_event_list(self.get_events(self.data, random_instance_id))
            if len(instance) < 35:
                good = True
        return instance, random_instance_id


    def get_instance(self, instance_id):
        instance = self.create_correct_event_list(self.get_events(self.data, instance_id))
        return instance
    ###
    # Helper functions to convert input to event_list
    ###
    def encode_names(self, name):
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

    def preprocess_dates(self, data):
        data['time'] = data.apply(lambda row: "01/01/1970 "+row.arrival_time[12:20]+row.arrival_time[-2:], axis=1)
        return data

    def get_events(self, data, instance_id):
        event_list = data.loc[data.instance_id == instance_id].copy()
        event_list['time2'] = event_list.apply(lambda row: (p.parse(row.time)+timedelta(hours=24)) if (p.parse(row.time) < p.parse("01/01/1970 08:00:00AM")) else p.parse(row.time), axis=1)
        event_list['unix'] = event_list.apply(lambda row: int(row.time2.timestamp()), axis = 1)
    #    filtered_event_list = event_list[['arrival_departure', 'carriages', 'instance_id', 'material_type', 'train_id', 'trainunit_id', 'trainunit_position', 'unix']]

        return event_list.sort_values(by=['unix', 'trainunit_position']).reset_index(drop=True)

    def create_correct_event_list(self, event_list):
        schermnaam = []
        spoor = []
        tijd = []
        treinstellen = []
        event_type = []
        length = []

        skip = 0
        for row in event_list.itertuples():
            if skip == 0:
                stellen = []
                stellen.append(self.encode_names((row.material_type+str(row.carriages))))
                if row.arrival_departure == 'A':
                    event_type.append('arrival')
                else:
                    event_type.append('departure')
                total_length = int(row.carriages)
                schermnaam.append(row.train_id)
                tijd.append(row.unix)
                spoor.append(row.arrival_track_id)

                lookahead = row.Index+1

                if row.arrival_departure == 'A':
                    if lookahead < len(event_list):
                        while event_list.unix[lookahead] == tijd[-1]:
                            stellen.append(self.encode_names(event_list.material_type[lookahead] + str(event_list.carriages[lookahead])))
                            total_length = total_length+int(event_list.carriages[lookahead])
                            lookahead+=1
                            skip+=1
                    treinstellen.append(stellen)
                else: #departures are only 1, one string, no list.
                    treinstellen.append(stellen[0])
                length.append(total_length)

                # the departing train units are now treated separately
                #, for instance, a a SLT6 - SLT4 leave at timestep
            else:
                skip-=1


            df = pd.DataFrame(
            {'name': schermnaam, 'track': spoor, 'time': tijd, 'composition': treinstellen, 'length': length, 'event_type': event_type})
        return df

if __name__ == "__main__":
    ig = INSTANCEProvider()
    instances = ig.instances

    # for i in range(0,100):
    #     ig.get_random_instance()
    #nr_carriages = []
    #for inst in instances:
    #    nr_carriages.append(len(ig.get_instance(inst)))
    #plt.hist(nr_carriages)
    #we see that this set of problem instances is very homogeneous.
    #for now OK but later maybe not.




