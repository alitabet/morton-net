from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import time
import pathlib
import glob

import torch
import numpy as np
import h5py
from tqdm import trange

from mortonnet import MortonNet
from tools.config import backward_compatible_config


def process_one_file(file):
    """
    Compute self supervised features using a trained MortonNet
    for data available in provided file.

    :param file: Full path to h5 file with data, labels, and indices
    """
    if not file.endswith('.h5'):
        return

    save_file = os.path.join(args.out_dir, os.path.basename(file))
    if os.path.isfile(save_file):
        return

    logging.info('Processing file {}'.format(file))

    hdf5_file = h5py.File(file, 'r')

    data = hdf5_file['data'][...]
    labels = hdf5_file['labels'][...]
    indices = hdf5_file['indices'][...]

    hdf5_file.close()

    out_file = h5py.File(save_file, 'w')

    # create datasets for data, labels, and features
    out_file.create_dataset('data', data=data, shape=data.shape, dtype=data.dtype,
                            compression='gzip', compression_opts=4)
    out_file.create_dataset('labels', data=labels, shape=labels.shape, dtype=data.dtype,
                            compression='gzip', compression_opts=1)
    out_file.create_dataset('features', shape=(indices.shape[0] // 5, 5, config['hidden_size']), dtype=data.dtype,
                            compression='gzip', compression_opts=4)

    chunk_size = 15000
    count = 0
    chunks = np.arange(0, indices.shape[0], chunk_size)
    for c in trange(chunks.shape[0], desc='Computing Features...'):
        j = chunks[c]
        end = j + chunk_size
        sequences = data[indices[j:end, :], :3]
        sequences = sequences - sequences[:, 0, np.newaxis, :]
        sequences = torch.from_numpy(sequences[:, :-1, :]).to(device, dtype=torch.float)
        out, h_n = model(sequences.unsqueeze(0))
        h_n = np.reshape(h_n[-1].detach().cpu().numpy(), (-1, 5, config['hidden_size']))
        out_file['features'][count:count + h_n.shape[0], :] = h_n
        count += h_n.shape[0]

    out_file.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Compute self-supervised features from a given pre-trained model.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # required arguments.
    parser.add_argument('--root_dir', required=True, type=str,
                        help='Path to directory with data and sequence h5 files')
    parser.add_argument('--best_path', required=True, type=str,
                        help='Path to the pre-trained model checkpoint file')
    parser.add_argument('--out_dir', required=True, type=str,
                        help='directory for saving')

    # optional arguments
    parser.add_argument('--loglevel', default='INFO', type=str,
                        help='logging level')
    parser.add_argument('--data_idx', default=-1, type=int,
                        help='index to process - used for parallel processing')
    parser.add_argument('--multi_gpu', default=False, action='store_true',
                        help='use multiple GPUs (all available)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='index of GPU to use (0-indexed); if multi_gpu then value is ignored')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                        level=numeric_level)
    delattr(args, 'loglevel')

    start_time = time.time()
    # we load the best model
    logging.info('Loading model...')
    best_state = torch.load(args.best_path)
    config = best_state['config']

    config = backward_compatible_config(config)

    use_multi_gpu = args.multi_gpu
    gpu_index = args.gpu
    # if the given index is not available then we use index 0
    # also when using multi gpu we should specify index 0
    if gpu_index + 1 > torch.cuda.device_count() or use_multi_gpu:
        gpu_index = 0

    logging.info('using gpu cuda:{}, script PID {}'.format(gpu_index, os.getpid()))
    device = torch.device('cuda:{}'.format(gpu_index))

    input_size = 3
    if config['seq_type'] == 'orient':
        input_size = 4
    model = MortonNet(input_size=input_size, conv_layers=config['conv_layers'],
                      rnn_layers=config['rnn_layers'], hidden_size=config['hidden_size']).to(device)
    model.load_state_dict(best_state['model'])
    model.eval()

    # create directory to save feature files
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logging.info('Features will be saved in {}'.format(args.out_dir))

    # save the best state
    torch.save(best_state, os.path.join(args.out_dir, 'best_state.pth'))

    all_files = glob.glob(os.path.join(args.root_dir, '*.h5'))

    # process one or all files
    if args.data_idx == -1:
        for file in all_files:
            process_one_file(file)
    else:
        process_one_file(all_files[args.data_idx])

    logging.info('Time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))
