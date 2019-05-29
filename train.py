from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import time
import uuid
import os
from tqdm import tqdm

from tools.config import Config, backward_compatible_config
from tools.utils import Timer, MetricTracker
from dataset import get_data_loaders
from mortonnet import MortonNet, run_one_batch

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter


def generate_experiment_dir(model_dir, config):
    # we save model checkpoints using the provided model directory
    # but we add a subfolder with format:
    #   Y-m-d_H:M_lr_chunkSize_seqLen_convLayers_rnnLayers_hiddenSize_dropout_bidirectional_class_phase__UUID
    timestamp = time.strftime('%Y-%m-%d-%H:%M:%S')
    experiment_string = '{}_{}_{}_{}_{}_{}_{:.1f}_{}_{}_{}_{}_{}_{}__{}'\
        .format(timestamp, config['lr'], config['chunk_size'], config['seq_len'], config['conv_layers'],
                config['rnn_layers'], config['hidden_size'], config['dropout'],
                config['bidirectional'], 'feat',
                config['merge'] if config['merge'] else 'none',
                config['cluster'] if config['cluster'] else 'none',
                config['ratio'] if config['cluster'] else 'none', uuid.uuid4())

    experiment_dir = os.path.join(model_dir, experiment_string)

    try:
        os.mkdir(experiment_dir)
        return experiment_dir
    except Exception as e:
        logging.warning('error making the experiment directory. {}'.format(e))
        raise Exception('error making the experiment directory')


def dump_config_details_to_tensorboard(writer, config):
    for k, v in config.items():
        if k not in config['do_not_dump_in_tensorboard']:
            writer.add_scalar('config/{}'.format(k), v, 0)


def dump_best_model_metrics_to_tensorboard(writer, phases, best_state):
    for phase in phases:
        writer.add_scalar('best_state/{}_loss'.format(phase), best_state['{}_loss'.format(phase)], 0)
        writer.add_scalar('best_state/{}_acc'.format(phase), best_state['{}_acc'.format(phase)], 0)
    writer.add_scalar('best_state/convergence_epoch', best_state['convergence_epoch'], 0)


def create_trackers(phases):
    phase_trackers = {}
    batch_trackers = {}

    # timers
    phase_trackers['timer'] = {phase: Timer() for phase in phases}
    batch_trackers['timer'] = {phase: Timer() for phase in phases}

    # loss trackers
    batch_trackers['loss'] = {phase: MetricTracker() for phase in phases}

    # accuracy trackers
    batch_trackers['acc'] = {phase: MetricTracker() for phase in phases}

    return phase_trackers, batch_trackers


def add_to_trackers(trackers, phase, loss, acc):
    trackers['loss'][phase].add(loss, 1)
    trackers['acc'][phase].add(acc, 1)


def log_plot_and_save(model, writer, model_dir, batch_trackers, phase,
                      epoch, step_number, num_steps,
                      print_every, plot_every, save_every):
    local_loss = batch_trackers['loss'][phase].aver()
    local_acc = batch_trackers['acc'][phase].aver()
    time_elapsed = batch_trackers['timer'][phase].aver(last_only=True)
    # log metrics
    if (step_number + 1) % print_every == 0:
        logging.info('{} Step [{}/{}]'.format(phase, step_number + 1, num_steps))
        logging.info('Batch time {}'.format(time_elapsed))
        logging.info('Loss: {:.2e}, Accuracy: {:.4f}'.format(local_loss, local_acc))

    # plotting
    if (step_number + 1) % plot_every == 0:
        writer.add_scalar('loss/{}'.format(phase), local_loss, epoch * num_steps + step_number)
        writer.add_scalar('acc/{}'.format(phase), local_acc, epoch * num_steps + step_number)

    # saving
    if (step_number + 1) % save_every == 0:
        file_name = os.path.join(model_dir, 'checkpoint_{}_{}_{:.2e}.pth'
                                 .format(epoch + 1, step_number + 1, local_loss))
        torch.save(model.state_dict(), file_name)


def train(config, model, criterion, optimizer, dataloaders, device, model_dir, phases,
          scheduler=None, writer=None, print_every=100, plot_every=100, save_every=100):
    """
    This function trains an RNN model to predict
    the next point in an ordered point cloud sequence

    :param config: current configuration
    :param model: PCRNN model
    :param criterion: loss function
    :param optimizer: optimizer
    :param dataloaders: dictionary with train and val dataloaders
    :param device: torch device
    :param phases: list of phase types ('train', 'valid')
    :param model_dir: directory to save checkpoints
    :param scheduler: (optional) LR scheduler
    :param writer: (optional) TensorboardX writer
    :param print_every: print frequency (default 100)
    :param plot_every: Tensorboard record frequency (default 100)
    :param save_every: Checkpoint save frequency (default 100)

    :return: Trained model
    """
    dump_config_details_to_tensorboard(writer, config)

    # best_state stores the best configuration so far
    best_state = {
        'config': config,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'train_loss': float('inf'),
        'valid_loss': float('inf'),
        'train_acc': 0.0,
        'valid_acc': 0.0,
        'convergence_epoch': 0,
        'num_epochs_since_best_valid_loss': 0
    }

    # keep track of time for whole train process
    # each epoch, and each batch
    full_timer = Timer()
    phase_trackers, batch_trackers = create_trackers(phases)

    for epoch in range(config['max_epochs']):

        for phase in phases:
            batch_trackers['loss'][phase].reset()
            batch_trackers['acc'][phase].reset()
            batch_trackers['timer'][phase].start()
            phase_trackers['timer'][phase].start()

            if phase == 'train':
                model.train()
            else:
                model.eval()

            num_steps = len(dataloaders[phase])
            for step_number, (sequences, original, last_length, _) in \
                    enumerate(tqdm(dataloaders[phase], desc='Epoch {}/{}'.format(epoch + 1, config['max_epochs']))):
                last_length = last_length[0].to(device)
                # point clouds have a lot of sequences, we process them by chunks
                for c in range(sequences.size(1)):
                    end = sequences.size(2)
                    if (c + 1) == sequences.size(1):
                        end = last_length

                    seq = sequences[:, c, :end, :].to(device, dtype=torch.float)
                    orig = original[:, c, :end, :].to(device, dtype=torch.float)

                    loss, acc = run_one_batch(model=model, data=(seq, orig),
                                              criterion=criterion, optimizer=optimizer, phase=phase,
                                              loss_extent=config['loss_extent'])

                    add_to_trackers(batch_trackers, phase, loss, acc)

                batch_trackers['timer'][phase].tick()

                # if we are in train then we check for printing, plotting, and saving
                if phase == 'train':
                    # we log, plot, and save during training steps
                    log_plot_and_save(model=model, writer=writer, model_dir=model_dir,
                                      batch_trackers=batch_trackers, phase=phase,
                                      epoch=epoch, step_number=step_number, num_steps=num_steps,
                                      print_every=print_every, plot_every=plot_every, save_every=save_every)

            phase_trackers['timer'][phase].tick()

            # we use the validation loss to update our learning rate scheduler
            if phase == 'valid' and scheduler:
                scheduler.step(batch_trackers['loss'][phase].aver())
                writer.add_scalar('params/lr', optimizer.param_groups[0]['lr'], epoch + 1)

            time_elapsed = phase_trackers['timer'][phase].aver(last_only=True)
            epoch_loss = batch_trackers['loss'][phase].aver()
            epoch_acc = batch_trackers['acc'][phase].aver()
            logging.info('{} time elapsed: {}'.format(phase, time_elapsed))
            logging.info('{} Loss: {:.2e}. Accuracy: {:.4f}. Final LR: {:.2e}'
                         .format(phase, epoch_loss, epoch_acc, optimizer.param_groups[0]['lr']))

            writer.add_scalar('loss/epoch_{}'.format(phase), epoch_loss, epoch + 1)
            writer.add_scalar('acc/epoch_{}'.format(phase), epoch_acc, epoch + 1)

        # save the model to disk if it has improved
        if best_state['valid_loss'] < batch_trackers['loss']['valid'].aver():
            best_state['num_epochs_since_best_valid_loss'] += 1
        else:
            logging.info('Got a new best model with valid loss {:.2e}'.format(batch_trackers['loss']['valid'].aver()))
            best_state = {
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'train_loss': batch_trackers['loss']['train'].aver(),
                'valid_loss': batch_trackers['loss']['valid'].aver(),
                'train_acc': batch_trackers['acc']['train'].aver(),
                'valid_acc': batch_trackers['acc']['valid'].aver(),
                'convergence_epoch': epoch + 1,
                'num_epochs_since_best_valid_loss': 0
            }

            file_name = os.path.join(model_dir, 'best_state.pth')
            torch.save(best_state, file_name)
            logging.info('saved checkpoint in {}'.format(file_name))

        # early stopping
        if best_state['num_epochs_since_best_valid_loss'] >= config['early_stopping']:
            logging.info('Validation loss did not improve for {} iterations!'.format(config['early_stopping']))
            logging.info('[Early stopping]')
            break

    full_timer.tick()

    # save the best model to Tensorboard
    dump_best_model_metrics_to_tensorboard(writer, phases, best_state)

    time_elapsed = full_timer.aver(last_only=True)
    logging.info('DONE! Training took {}'.format(time_elapsed))


def main(args):
    # make sure that model dir exists
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    use_multi_gpu = args.multi_gpu
    gpu_index = args.gpu
    # if the given index is not available then we use index 0
    # also when using multi gpu we should specify index 0
    if gpu_index + 1 > torch.cuda.device_count() or use_multi_gpu:
        gpu_index = 0

    logging.info('using gpu cuda:{}, script PID {}'.format(gpu_index, os.getpid()))
    device = torch.device('cuda:{}'.format(gpu_index))

    # get the configuration file
    config = Config(args.config_type).create_config()
    if args.state:
        # if we provide a saved state then load config from there
        logging.info('loading config from {}'.format(args.state))
        best_state = torch.load(args.state)
        config = best_state['config']

    # sanity check to make sure old configs still work with new format
    config = backward_compatible_config(config)
    # size of input depends on sequence types, either difference or orientation
    input_size = 3
    if config['seq_type'] == 'orient':
        input_size = 4
    model = MortonNet(input_size=input_size, conv_layers=config['conv_layers'],
                      rnn_layers=config['rnn_layers'], hidden_size=config['hidden_size'])
    # we use MSE loss
    criterion = nn.MSELoss()

    model.to(device)
    # if use multi_gou then convert the model to DataParallel
    if use_multi_gpu:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # we reduce learning rate when validation doesn't improve after some patience
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=config['lr_decay'],
                                  patience=config['lr_patience'],
                                  verbose=True)

    logging.info('Config {}'.format(config))

    phases = ['train', 'valid']
    dataloaders, datasets = get_data_loaders(root_dir=args.root_dir,
                                             phases=phases,
                                             shuffle=True,
                                             cluster=config['cluster'],
                                             batch_size=args.bs,
                                             chunk_size=config['chunk_size'],
                                             seq_len=config['seq_len'],
                                             random_sequence=config['random_sequence'],
                                             ratio=config['ratio'],
                                             seq_type=config['seq_type'])

    model_dir = generate_experiment_dir(args.model_dir, config)
    logging.info('TB logs and checkpoint will be saved in {}'.format(model_dir))

    # get TensorboardX writer
    writer = SummaryWriter(log_dir=model_dir)

    train(config=config, model=model, criterion=criterion, optimizer=optimizer, dataloaders=dataloaders,
          device=device, model_dir=model_dir, phases=phases, scheduler=scheduler, writer=writer,
          print_every=args.print, plot_every=args.plot, save_every=args.save)

    writer.close()


if __name__ == '__main__':
    """
    Run experiments using the parameters given in a configuration file. 
    The main parameters are learning rate, number of conv layers, number 
    of RNN layers, size of hidden state, and chunk size. We measure the 
    success of each experiment by measuring validation loss/accuracy.
    """
    parser = ArgumentParser(description='Train MortonNet to predict the next point in z-ordered sequences.',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root_dir', required=True, type=str,
                        help='root directory containing train and val folders with hdf5 files')
    parser.add_argument('--model_dir', required=True, type=str,
                        help='directory to save checkpoints')

    # optional arguments
    parser.add_argument('--config_type', default='random', type=str,
                        help='type configuration of hyperparameters')
    parser.add_argument('--multi_gpu', default=False, action='store_true',
                        help='use multiple GPUs (all available)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='index of GPU to use (0-indexed); if multi_gpu then value is ignored')
    parser.add_argument('--print', default=100, type=int,
                        help='print frequency')
    parser.add_argument('--plot', default=100, type=int,
                        help='Tensorboard record frequency')
    parser.add_argument('--save', default=100, type=int,
                        help='checkpoint frequency')
    parser.add_argument('--state', default=None, type=str,
                        help='path for best state to load')
    parser.add_argument('--bs', default=1, type=int,
                        help='batch size')
    parser.add_argument('--loglevel', default='INFO', type=str,
                        help='logging level')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                    level=numeric_level)
    delattr(args, 'loglevel')

    main(args)
