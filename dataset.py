import logging
import os
import glob
import numpy as np
import h5py
import time
import datetime
from multiprocessing import Pool

from sklearn.cluster import KMeans

from torch.utils.data import Dataset, DataLoader


class S3DISSequenceDataset(Dataset):
    """S3DIS: predict next point in a z-ordered sequence of points in a point cloud"""

    def __init__(self, root_dir, phase, chunk_size=128, seq_len=100, random_sequence=False,
                 shuffle=True, cluster=None, ratio=0.4, seq_type='normal'):
        """
        Dataset class for S3DIS self-supervision feature learning.
        Data is points and sequence indices for training a feature extractor

        :param root_dir: root folder containing the phase data folder
        :param phase: either train, valid, or test
        :param chunk_size: internal batch size
        :param seq_len: length of each sequence (max 100)
        :param random_sequence: if True then we randomly shuffle sequences, instead of z-order
        :param shuffle: if True we shuffle the sequence rows
        :param cluster: either 'random', 'kmeans', or None. If None then no clustering
        :param ratio: ratio of data to retrieve from clusters
        :param seq_type: normal, diff, orient: normal sequence, sequence difference with first point, or orientations
        """
        self.cluster = cluster
        self.ratio = ratio
        self.phase = phase
        self.chunk_size = chunk_size
        self.seq_len = seq_len
        self.random_sequence = random_sequence
        self.shuffle = shuffle
        self.seq_type = seq_type
        self.keys_to_retrieve = ['data', 'indices']

        paths = np.array(sorted(glob.glob(os.path.join(root_dir, '{}/*.hdf5'.format(phase)))))

        logging.info('Creating S3DISSequenceDataset for phase {}. Number of cloud points {}. Clustering using {}'
                     .format(phase, len(paths), self.cluster))

        # if we need to cluster with kmeans then we
        # run multi-processing else it's faster to do a for loop
        if self.cluster == 'kmeans':
            with Pool() as p:
                data = p.map(self._get_data, paths)
        else:
            data = [self._get_data(paths[i]) for i in range(paths.shape[0])]

        # if shuffle is true then we shuffle
        # the index of the point cloud files
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(data)

        self.data = data

    def _get_data(self, file):
        """
        Get the points, labels, and sequences. The file
        provided is for points, we can get the other paths
        from the points path. We can either extract all data
        if keys is None, or just data from specific keys.
        We also allow for clustering if asked. Cluster options
        are 'none' for no clustering, 'random' for random
        selection of sequences, or 'kmeans' for K-means clustering

        :param file: full path for file h5 file containing data
        :return: dictionary indexed by keys, containing data
        """
        hdf5_file = h5py.File(file, 'r')
        out = {
            'file': file
        }

        for key in self.keys_to_retrieve:
            out[key] = hdf5_file[key][...]

        hdf5_file.close()

        # check if we have to cluster
        if self.cluster:
            out['indices'] = cluster_sequences(out, self.cluster, ratio=self.ratio)

        # if random sequence => shuffle indices (columns), always keep last column
        if self.random_sequence:
            np.random.shuffle(out['indices'][:, :-1].T)

        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]['data'][:, :3]  # points
        indices = self.data[idx]['indices']  # sequence indices
        file = self.data[idx]['file']  # file name

        # if shuffle, then we shuffle the order of the sequence
        if self.shuffle:
            np.random.shuffle(indices)

        # if we ask for sequences smaller than what we have
        # then we select the last seq_len points in the sequence
        if self.seq_len < indices.shape[1]:
            indices = indices[:, -self.seq_len:]

        # we now chunk the data given the chunk size
        # we also store the length of the last chunk
        # since this will be smaller than the rest
        num_chunks = indices.shape[0] // self.chunk_size + 1
        # last dimension of sequences depends on sequence type
        sequences = np.zeros((num_chunks, self.chunk_size, indices.shape[1], data.shape[1]))
        if self.seq_type == 'orient':
            sequences = np.zeros((num_chunks, self.chunk_size, indices.shape[1] - 1, data.shape[1] + 1))
        original = np.zeros((num_chunks, self.chunk_size, indices.shape[1], data.shape[1]))
        last_length = self.chunk_size
        for k, i in enumerate(range(0, indices.shape[0], self.chunk_size)):
            chunk = indices[i:i + self.chunk_size, :]
            last_length = chunk.shape[0]
            temp = data[chunk, :]
            original[k, :last_length, :] = temp
            if self.seq_type == 'diff':
                temp = temp - temp[:, 0, np.newaxis, :]
            if self.seq_type == 'orient':
                temp = get_sequence_orientations(temp)
            sequences[k, :last_length, :] = temp

        return sequences, original, last_length, file


def get_sequence_orientations(sequence):
    """
    Compute the orientation difference of each vector
    in the sequence with respect to the first vector.
    Orientation is defined as the difference between
    unit vectors and difference between magnitudes

    :param sequence: sequence of 3D points (BxNx3)

    :return: sequence of differences (BxNx4)
    """
    vectors = np.diff(sequence, axis=1)
    mag = np.sqrt(np.sum(vectors ** 2, axis=-1))
    vectors /= mag[:, :, np.newaxis]
    v0 = vectors[:, 0, np.newaxis, :]
    unit = vectors - v0
    d = mag - mag[:, 0, np.newaxis]

    return np.concatenate((unit, d[:, :, np.newaxis]), axis=2)


def cluster_sequences(out, cluster='random', ratio=0.4):
    """
    Select indices from the sequences provided based on
    kmeans clustering or random

    :param out: dictionary with all data
    :param cluster: 'random' or 'kmeans'
    :param ratio: percentage of data to choose from
    :return: new indices array
    """
    # check type of clustering
    if cluster == 'kmeans':
        n_clusters = 8
        data, indices = out['data'], out['indices']
        sequences = data[indices, :]
        feats = np.reshape(np.diff(sequences, axis=1), (sequences.shape[0], -1))
        kmeans = KMeans(n_clusters=n_clusters).fit(feats)
        labels = kmeans.labels_
        all_indices = []
        for i in range(n_clusters):
            np.random.seed(i)
            idx = np.argwhere(labels == i).squeeze()
            choice = np.random.choice(idx, int(ratio * len(idx)), replace=False)
            all_indices.extend(choice)

        return indices[all_indices, :]

    if cluster == 'random':
        np.random.seed(0)
        indices = out['indices']
        idx = np.random.choice(np.arange(indices.shape[0]), int(ratio * indices.shape[0]), replace=False)

        return indices[idx, :]

    # if nothing then return all indices
    return out['indices']


def get_data_loaders(root_dir, phases=['train', 'valid', 'test'], shuffle=False, cluster=None, chunk_size=512,
                     batch_size=1, seq_len=100, random_sequence=False, ratio=0.4, seq_type='normal'):
    """
    Function to get train and val dataloaders for S3DIS.
    We get the points and sequence indices to train our self-
    supervised network to create point features.

    :param root_dir: root folder containing all the data
    :param phases: list of phase types ('train', 'valid, 'test')
    :param shuffle: whether we shuffle data order
    :param cluster: either None, 'random', or 'kmeans'. If none then no clustering
    :param chunk_size: size of data chunks
    :param batch_size: batch size
    :param seq_len: length of each sequence (max 100)
    :param random_sequence: if True then we randomly shuffle sequences, instead of z-order
    :param ratio: ratio of data to retrieve from clusters
    :param seq_type: normal, diff, orient: normal sequence, sequence difference with first point, or orientations

    :return: dataloaders and datasets for phases requested
    """

    datasets = {x: S3DISSequenceDataset(root_dir=root_dir,
                                        phase=x,
                                        chunk_size=chunk_size,
                                        seq_len=seq_len,
                                        random_sequence=random_sequence,
                                        shuffle=(x == 'train') and shuffle,
                                        cluster=cluster if x == 'train' else None,
                                        ratio=ratio,
                                        seq_type=seq_type)
                for x in phases}

    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                                 shuffle=False, num_workers=4)
                   for x in phases}

    return dataloaders, datasets


if __name__ == '__main__':
    numeric_level = getattr(logging, 'INFO', None)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)

    root_dir = '~/S3DIS'

    dataloaders, datasets = get_data_loaders(root_dir=root_dir,
                                             phases=['train', 'valid'],
                                             cluster='random',
                                             batch_size=1)

    since = time.time()
    for sequences, original, last_length, file in dataloaders['train']:
        logging.info('Loaded sequences {} original {} last length {} and file {}'
                     .format(sequences.size(), original.size(), last_length, file))
        break
    ellapsed_time = str(datetime.timedelta(seconds=time.time() - since)).split('.')[0]
    logging.info('Loading all data took {}'.format(ellapsed_time))
