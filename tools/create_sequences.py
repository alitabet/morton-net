from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import pathlib
import time
import numpy as np
import glob
import h5py
from multiprocessing import Pool
from sklearn.neighbors import KDTree
from z_order import round_to_int_32, get_z_order


def get_z_values(data):
    """
    Computes the z values for a point array
    :param data: Nx3 array of x, y, and z location

    :return: Nx1 array of z values
    """
    points_round = round_to_int_32(data)  # convert to int
    z = get_z_order(points_round[:, 0], points_round[:, 1], points_round[:, 2])

    return z


def get_sequences(points):
    """
    Given the points in a point cloud, we compute 5 sequences
    of length 100 for each point. At each point, we look at a
    local neighborhood, and sample points whose z order either
    starts or ends at the current point. We do this sampling 5
    times for each point and end up with 5 sequences of 100
    points, for each point in the original data. Each sequence
    ends with the point of interest, meaning that each 5
    consecutive sequences end in the same point.

    :param points: Nx3 array of x, y, and z locations
    :return: 5Nx100 array of indices for each sequences
    """
    # first we get the z values for all the points
    z = get_z_values(points)

    # we construct a KD-Tree to sample local neighbors
    tree = KDTree(points, leaf_size=40)

    num_sequences = 5
    sequence_length = 100
    # support is the extent of the neighborhood
    support = 4 * sequence_length

    # we get the closest points
    dist, ind = tree.query(points, k=support)

    all_indices = []
    for i in range(points.shape[0]):
        cpi = ind[i, :support]  # closest points for point i
        zp = z[cpi]  # corresponding z-values for closest points
        less = np.where(zp < zp[0])[0]  # points wit lower z
        more = np.where(zp > zp[0])[0]  # points with higher z

        # we sample 2 sequences for lower z values and
        # 5 with higher
        k = 0
        o = 2
        if more.shape[0] <= sequence_length:
            # if we don't have enough points then we just sample lower ones
            o = num_sequences
        idxs = np.zeros((num_sequences, sequence_length), dtype=np.int)
        # first sample lower z value sequences
        if less.shape[0] > sequence_length:
            for _ in range(o):
                temp = np.random.choice(less, sequence_length - 1, replace=False)
                z_ind = np.argsort(zp[temp])
                idxs[k, :-1] = temp[z_ind]
                idxs[k, -1] = 0
                k += 1
        # now sample higher z value ones
        for l in range(k, num_sequences):
            temp = np.random.choice(more, sequence_length - 1, replace=False)
            z_ind = np.argsort(zp[temp])
            idxs[l, 1:] = temp[z_ind]
            idxs[l, 0] = 0
            idxs[l, :] = np.flip(idxs[l, :], axis=0)

        all_indices.append(cpi[idxs])

    # return array with shape 5Nx100
    all_seq_idx = np.reshape(np.array(all_indices), (-1, 100))

    return all_seq_idx


def process_one_file(file):
    """
    Process one file of point cloud data and return indices for sequences.
    Input is an npy file containing 3D points, colors, and labels.
    The outputs are saved in HDF5 format. Data saved are points,
    labels, and the indices for the sequences.

    :param file: full path to npy file containing Nx7 (points, colors, label)
    """
    logging.info('Processing {}'.format(file))

    # output format is HDF5, using same name as npy file
    out_file = os.path.join(args.out_dir, '{}'.format(os.path.basename(file).replace('npy', 'h5')))
    if os.path.isfile(out_file):
        logging.info('File {} already exists'.format(file))
        return

    # load all data with points and labels
    data = np.load(file)

    # points includes both 3D locations plus RGB data
    points = data[:, :-1]
    labels = data[:, -1]

    # get indices of morton ordered sequences for all points
    indices = get_sequences(points[:, :3])

    # save file
    h_file = h5py.File(out_file, 'w')

    h_file.create_dataset('data', data=points, shape=points.shape, dtype=points.dtype,
                          compression='gzip', compression_opts=4)

    h_file.create_dataset('labels', data=labels, shape=labels.shape, dtype=labels.dtype,
                          compression='gzip', compression_opts=1)

    h_file.create_dataset('indices', data=indices, shape=indices.shape, dtype=indices.dtype,
                          compression='gzip', compression_opts=4)

    h_file.close()


def create_ss_dataset():
    """
    Load the data from given folder and compute sequences. If we provide
    a data index then we process that file only, otherwise we process
    all the available files. Processing all files is done using a
    multiprocessing Pool to gain speed.
    """
    start_time = time.time()

    logging.info('Computing sequences')

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_files = glob.glob(os.path.join(args.root_dir, '*.npy'))

    if args.data_idx == -1:
        with Pool() as p:
            p.map(process_one_file, all_files)
    else:
        logging.info('No multiprocessing')
        if args.data_idx >= len(all_files):
            logging.info('Index {} larger that total number of files {}'.format(args.data_idx + 1, len(all_files)))
        else:
            process_one_file(all_files[args.data_idx])

    logging.info('Time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))


if __name__ == '__main__':
    """
    Script takes npy data of point clouds and creates Morton sequences for each file.
    The data should all be in one folder, with each npy file containing an array
    of shape Nx7, where N is number of points, and 7 represents XYZRG + label.
    The output is stored in HDF5 files with the data, labels, and indices
    for the Morton sequences. The sequences can be used to train MortonNet or to
    compute Morton features from a pre-trained model.
    """
    parser = ArgumentParser(description='Create HDF5 files for the self-supervised sequence dataset',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # required arguments.
    parser.add_argument('--root_dir', required=True, type=str,
                        help='Path to the root directory with npy files')
    parser.add_argument('--out_dir', required=True, type=str,
                        help='Path to the output directory to save sequences')

    # optional arguments
    parser.add_argument('--loglevel', default='INFO', type=str,
                        help='Logging level (CRITICAL, ERROR, WARNING, INFO, DEBUG). Default: INFO')
    parser.add_argument('--data_idx', default=-1, type=int,
                        help='Index to process from file list, -1 to process all files. Default: -1')

    args = parser.parse_args()

    # we first set logging config
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                    level=numeric_level)
    delattr(args, 'loglevel')

    root_dir = args.root_dir
    create_ss_dataset()
