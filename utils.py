import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from pprint import pprint

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir

# The configuration below is from the paper.
default_configure = {
    'lr': 0.001,             # Learning rate
    'hidden_units': 256,
    'dropout': 0.4,
    'weight_decay': 0.0001,
    'num_epochs': 3000,
    'patience': 30,
    'num_layers':2  #GCN layers
}


sampling_configure = {
    'batch_size': 128
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = './data/GSE118389.pkl'
    args['device'] = 'cuda:0'
    args['model'] = 'scMGCN'      #if Att,run scMGCN ,else run mogonet
    args['log_dir'] = setup_log_dir(args)
    return args

def load_data(dataset):
    pkl_file = open(dataset, 'rb')

    pkldata = pickle.load(pkl_file)

    '''
    g_list = pkldata['gs']
    gs = []
    # choose different graph
    #gs.append(g_list[0])
    #gs.append(g_list[1])
    #gs.append(g_list[2])
    #gs.append(g_list[3])#single/cross-species
    #gs.append(g_list[4])#crossplatform
    #gs.append(g_list[5])#single/cross-species
    #gs.append(g_list[6])#crossplatform
    #gs.append(g_list[7])
    '''

    # all graphs
    gs = pkldata['gs']

    features = pkldata['features']
    labels = pkldata['labels']
    num_classes = pkldata['num_classes']
    train_idx = pkldata['train_idx']
    val_idx = pkldata['val_idx']
    test_idx = pkldata['test_idx']
    train_mask = pkldata['train_mask']
    val_mask = pkldata['val_mask']
    test_mask = pkldata['test_mask']

    print('dataset loaded')
    num_nodes = gs[0].number_of_nodes()
    pprint({
        'dataset': dataset,
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss): #and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss):# and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))