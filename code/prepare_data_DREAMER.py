# This is the processing script of DREAMER dataset

import _pickle as cPickle

from train_model import *

import torch
from utils import generate_TS_channel_order
from networks import TSception
import os

class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.original_order = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6','F4', 'F8', 'AF4']

        self.TS_order = generate_TS_channel_order(self.original_order)   # generate proper channel orders for the asymmetric spatial layer in TSception

        self.graph_type = args.graph_type

    def run(self, sub_clip_list , split=False, expand=True, feature=False):
        """
        Parameters
        ----------
        sub_clip_list: the subjects and clips ID needed to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN
        feature: (bool) whether to extract features or not

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in sub_clip_list[0]:
            for clip in sub_clip_list[1]:
                data_, label_ = self.load_data_per_subject(sub,clip)
                # select label type here
                label_ = self.label_selection(label_)

                if expand:
                    # expand one dimension for deep learning(CNNs)
                    data_ = np.expand_dims(data_, axis=-3)

                if split:
                    data_, label_ = self.split(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate)

                


                print('Data and label prepared for sub{}!'.format(sub))
                print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
                print(f"The label: {label_}")
                print('----------------------')
                self.save(data_, label_, sub)

    def load_data_per_subject(self, sub,clip):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load
        clip: which clip to load

        Returns
        -------
        data: (1, 14, Signal_Length, 7680) label: (1, 3)
        """
        # Create subject folder path
        person_folder = f"person{sub+1}"
        person_path = os.path.join(self.data_path, person_folder)
        
        clip_name = f"stimuli_eeg{clip+1}.txt"
        clip_path  = os.path.join(person_path, clip_name)

        # Read Stimuli EEG data of the person{sub+1}
        with open(clip_path, 'r') as f:
            eeg_data = [list(map(float, row.strip().split('\t')[1:])) for row in f.readlines()] # exclude the electrode names
        
        # Read emotion scores
        with open(os.path.join(person_path, 'valence.txt'), 'r') as f:
            valence = float(f.readline().strip().split('\t')[clip])
        with open(os.path.join(person_path, 'arousal.txt'), 'r') as f:
            arousal = float(f.readline().strip().split('\t')[clip])
        with open(os.path.join(person_path, 'dominance.txt'), 'r') as f:
            dominance = float(f.readline().strip().split('\t')[clip])

        label = np.array([valence, arousal, dominance])
        eeg_data = np.array(eeg_data)
        
        print(f"Pesron{sub+1}, Clip{clip+1}",end = '\t')
        print('data:' + str(eeg_data.shape) + ' label:' + str(label.shape) ,end = '\t')

        #   data: (1, 14, Signal_Length, 7680) 
        #   label: (1, 3)
        # reorder the EEG channel to build the local-global graphs

        data = self.reorder_channel(data=eeg_data, graph=self.graph_type)
        print('data2:' + str(data.shape) + ' label2:' + str(label.shape))

        return data, label

    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'TS':
            graph_idx = self.TS_order
        elif graph == 'O':
            graph_idx = self.original_order

        # Convert input tensor/array to float
        data = data.astype(np.float64)  # or np.float32

        idx = []
        for chan in graph_idx:
            idx.append(self.original_order.index(chan))
        #data = data[:, :, idx, :]    # (batch_size=1, cnn_channel=1, EEG_channel=14, data_points=25472) Some channels are not selected, hence EEG channel becomes 28.
        #return data[:, idx, :]

        return data[idx,:]


    def label_selection(self, label):
        # V A D
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.label_type == 'V':
            label = label[0]
        elif self.label_type == 'A':
            label = label[1]
        elif self.label_type == 'D':
            label = label[2]
            
        # if self.args.num_class == 2:
        #     label = np.where(label <= 5, 0, label)
        #     label = np.where(label > 5, 1, label)
        #     print('Binary label generated!')
        return label

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[2] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label
