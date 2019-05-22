import numpy as np
import logging

merge_to_int = {'cat': 0, 'max': 1, 'mean': 2, 'sum': 3}
cluster_to_int = {'none': 0, None: 0, 'random': 1, 'kmeans': 2}
seq_type_to_int = {'normal': 0, 'diff': 1, 'orient': 2}


class Config(object):
    def __init__(self, config_type='random'):
        """
        The Config object create configuration files to
        train MortonNet. We define here a few configurations,
        including a random version and a few baselines used
        during training. To create a custom config, create a
        new static config function and add the config to the
        switcher dictionary. When creating a Config instance,
        we can select the type of config based on the keys
        in switcher. The config parameters are created using
        the create_config method.

        :param config_type: Config type as string
        """
        switcher = {
            'random': self._create_random_config,
            'generic_class': self._create_generic_class_config,
            'generic_class_diff': self._create_generic_class_diff_config,
            'generic_class_orient': self._create_generic_class_orient_config,
            'random_sequence_baseline': self._create_random_sequence_baseline_config,
            'playground': self._create_playground_config,
        }
        self.create_config = switcher.get(config_type, None)

        logging.info('Setting config type to {}'.format(config_type))
        if self.create_config is None:
            logging.warning('Invalid config type {}. Setting config type to the default random config.'
                            .format(config_type))
            self.create_config = self._create_random_config

    @staticmethod
    def _create_random_config():
        logging.info('Generating a random config')
        config = {
            'lr': 0.01,
            'chunk_size': 2048,
            'train_max_pcs': 150,
            'val_max_pcs': 42,
            'seq_len': 100,
            'random_sequence': False,
            'conv_layers': 3,
            'rnn_layers': 4,
            'hidden_size': 400,
            'bidirectional': False,
            'dropout': 0.0,
            'batch_size': 1,
            'input_size': 3,
            'max_epochs': 50,
            'loss_extent': 1,
            'merge': 'cat',
            'cluster': 'random',
            'ratio': 0.4,
            'seq_type': 'normal',
            'lr_decay': 0.9,
            'lr_patience': 1,
            'early_stopping': 5,
            'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'merge', 'cluster', 'seq_type'],
        }
        config['_merge'] = merge_to_int[config['merge']]
        config['_cluster'] = cluster_to_int[config['cluster']]
        config['_seq_type'] = seq_type_to_int[config['seq_type']]

        return config

    @staticmethod
    def _create_generic_class_config():
        logging.info('Generating a generic class config')
        config = {
            'lr': 0.001,
            'chunk_size': 2 ** rnd_choice(11, 12, 1, output_type=int),
            'train_max_pcs': 75,
            'val_max_pcs': 21,
            'seq_len': rnd_choice(60, 100, 20, output_type=int),
            'random_sequence': False,
            'conv_layers': rnd_choice(2, 3, 1, output_type=int),
            'rnn_layers': rnd_choice(2, 4, 1, output_type=int),
            'hidden_size': rnd_choice(100, 400, 100, output_type=int),
            'bidirectional': False,
            'dropout': 0.0,
            'batch_size': 1,
            'input_size': 3,
            'max_epochs': 50,
            'loss_extent': 1,
            'merge': 'cat',
            'cluster': 'random',
            'ratio': 0.4,
            'seq_type': 'normal',
            'lr_decay': 0.9,
            'lr_patience': 1,
            'early_stopping': 5,
            'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'merge', 'cluster', 'seq_type'],
        }
        config['_merge'] = merge_to_int[config['merge']]
        config['_cluster'] = cluster_to_int[config['cluster']]
        config['_seq_type'] = seq_type_to_int[config['seq_type']]

        return config

    @staticmethod
    def _create_generic_class_diff_config():
        logging.info('Generating a generic class (seq_type = diff) config')
        config = {
            'lr': 0.001,
            'chunk_size': 2 ** rnd_choice(11, 12, 1, output_type=int),
            'train_max_pcs': 36,
            'val_max_pcs': 10,
            'seq_len': rnd_choice(60, 100, 20, output_type=int),
            'random_sequence': False,
            'conv_layers': rnd_choice(2, 3, 1, output_type=int),
            'rnn_layers': rnd_choice(2, 4, 1, output_type=int),
            'hidden_size': rnd_choice(100, 400, 100, output_type=int),
            'bidirectional': False,
            'dropout': 0.0,
            'batch_size': 1,
            'input_size': 3,
            'max_epochs': 50,
            'loss_extent': 1,
            'merge': 'cat',
            'cluster': 'random',
            'ratio': 0.4,
            'seq_type': 'diff',
            'lr_decay': 0.9,
            'lr_patience': 1,
            'early_stopping': 5,
            'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'merge', 'cluster', 'seq_type'],
        }
        config['_merge'] = merge_to_int[config['merge']]
        config['_cluster'] = cluster_to_int[config['cluster']]
        config['_seq_type'] = seq_type_to_int[config['seq_type']]

        return config

    @staticmethod
    def _create_generic_class_orient_config():
        logging.info('Generating a generic class (seq_type = orient) config')
        config = {
            'lr': 0.001,
            'chunk_size': 2 ** rnd_choice(11, 12, 1, output_type=int),
            'train_max_pcs': 36,
            'val_max_pcs': 10,
            'seq_len': rnd_choice(60, 100, 20, output_type=int),
            'random_sequence': False,
            'conv_layers': rnd_choice(2, 3, 1, output_type=int),
            'rnn_layers': rnd_choice(2, 4, 1, output_type=int),
            'hidden_size': rnd_choice(100, 400, 100, output_type=int),
            'bidirectional': False,
            'dropout': 0.0,
            'batch_size': 1,
            'input_size': 3,
            'max_epochs': 50,
            'loss_extent': 1,
            'merge': 'cat',
            'cluster': 'random',
            'ratio': 0.4,
            'seq_type': 'orient',
            'lr_decay': 0.9,
            'lr_patience': 1,
            'early_stopping': 5,
            'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'merge', 'cluster', 'seq_type'],
        }
        config['_merge'] = merge_to_int[config['merge']]
        config['_cluster'] = cluster_to_int[config['cluster']]
        config['_seq_type'] = seq_type_to_int[config['seq_type']]

        return config

    @staticmethod
    def _create_random_sequence_baseline_config():
        logging.info('Generating a random sequence baseline config')
        config = {
            'lr': 0.01,
            'chunk_size': 2 ** rnd_choice(11, 12, 1, output_type=int),
            'train_max_pcs': 75,
            'val_max_pcs': 21,
            'seq_len': rnd_choice(60, 100, 20, output_type=int),
            'random_sequence': True,
            'conv_layers': rnd_choice(2, 3, 1, output_type=int),
            'rnn_layers': rnd_choice(2, 4, 1, output_type=int),
            'hidden_size': rnd_choice(100, 400, 100, output_type=int),
            'bidirectional': False,
            'dropout': 0.0,
            'batch_size': 1,
            'input_size': 3,
            'max_epochs': 50,
            'loss_extent': 1,
            'merge': 'cat',
            'cluster': 'random',
            'ratio': 0.4,
            'seq_type': 'normal',
            'lr_decay': 0.9,
            'lr_patience': 1,
            'early_stopping': 5,
            'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'merge', 'cluster', 'seq_type'],
        }
        config['_merge'] = merge_to_int[config['merge']]
        config['_cluster'] = cluster_to_int[config['cluster']]
        config['_seq_type'] = seq_type_to_int[config['seq_type']]

        return config

    @staticmethod
    def _create_playground_config():
        logging.info('Generating playground config')
        config = {
            'lr': 0.001,
            'chunk_size': 2 ** 12,
            'train_max_pcs': 75,
            'val_max_pcs': 21,
            'seq_len': 40,
            'random_sequence': False,
            'conv_layers': 3,
            'rnn_layers': 3,
            'hidden_size': 200,
            'bidirectional': False,
            'dropout': 0.0,
            'batch_size': 1,
            'input_size': 3,
            'max_epochs': 50,
            'loss_extent': 1,
            'merge': 'cat',
            'cluster': 'random',
            'ratio': 0.1,
            'seq_type': 'normal',
            'lr_decay': 0.9,
            'lr_patience': 1,
            'early_stopping': 5,
            'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'merge', 'cluster', 'seq_type'],
        }
        config['_merge'] = merge_to_int[config['merge']]
        config['_cluster'] = cluster_to_int[config['cluster']]
        config['_seq_type'] = seq_type_to_int[config['seq_type']]

        return config


def rnd_choice(start, end, step, output_type=float):
    """
    generates a random number in [start, end] with spacing
    size equal to step. The value of end is included.
    """
    nums = np.append(np.arange(start, end, step), end)
    return output_type(np.random.choice(nums))


def backward_compatible_config(config):
    if 'seq_len' not in config:
        config['seq_len'] = 100
    if 'train_max_pcs' not in config:
        config['train_max_pcs'] = 150
    if 'val_max_pcs' not in config:
        config['val_max_pcs'] = 42
    if 'merge' not in config:
        config['merge'] = 'cat'
    if 'cluster' not in config:
        config['cluster'] = None
    if 'ratio' not in config:
        config['ratio'] = 0.4
    if 'random_sequence' not in config:
        config['random_sequence'] = False
    if 'loss_extent' not in config:
        config['loss_extent'] = 1
    if 'seq_type' not in config:
        config['seq_type'] = 'normal'
    config['do_not_dump_in_tensorboard'] = ['do_not_dump_in_tensorboard', 'merge', 'cluster', 'seq_type']

    config['_merge'] = merge_to_int[config['merge']]
    config['_cluster'] = cluster_to_int[config['cluster']]
    config['_seq_type'] = seq_type_to_int[config['seq_type']]

    return config
