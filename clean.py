
# # setup
import torch
import torch.nn as nn

from models.baseline import ResnetBaseline
from hparams import BATCH_SIZE, NUM_WORKERS

model_label = 'code15clean'
epochs = 10

from dataloaders.code15 import CODE as DS
from dataloaders.code15 import CODEsplit as DSsplit
from runners.multiclass import Runner
model = ResnetBaseline(n_classes = 6)

from dataloaders.codetest import CODEtest
from utils import json_dump

# # init
database = DS(metadata_path = '/home/josegfer/datasets/code/output/metadata_clean.csv')
# model = torch.load('output/{}/partial.pt'.format(model_label))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

runner = Runner(device = device, model = model, database = database, split = DSsplit, model_label = model_label)

# # run
runner.train(epochs)
runner.eval(partial = False)

# eval codetest
tst_ds = CODEtest()
val_dl = torch.utils.data.DataLoader(runner.val_ds, batch_size = BATCH_SIZE, 
                                     shuffle = False, num_workers = NUM_WORKERS)
tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size = BATCH_SIZE, 
                                     shuffle = False, num_workers = NUM_WORKERS)

best_f1s, best_thresholds = runner._synthesis(val_dl, best_thresholds = None)
all_binary_results, all_true_labels, metrics_dict = runner._synthesis(tst_dl, best_thresholds)
json_dump(metrics_dict, model_label, test = True)
