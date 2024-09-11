
# # setup
import torch
import torch.nn as nn

from models.baseline import ResnetBaseline

# model_label = 'code15'
# epochs = 15
# from runners.multiclass import Runner
# from dataloaders.code15 import CODE as DS
# from dataloaders.code15 import CODEsplit as DSsplit
# model = ResnetBaseline(n_classes = 6)

# model_label = 'code15normal'
# epochs = 15
# from dataloaders.code15 import CODE as DS
# from dataloaders.code15 import CODEsplit as DSsplit
# from runners.normal import Runner
# model = ResnetBaseline(n_classes = 1)

# model_label = 'code15interferencia'
# epochs = 15
# from dataloaders.code15 import CODE as DS
# from dataloaders.code15 import CODEsplit as DSsplit
# from runners.interferencia import Runner
# model = ResnetBaseline(n_classes = 1)

# model_label = 'code151davb'
# epochs = 10
# from dataloaders.code15 import CODE as DS
# from dataloaders.code15 import CODEsplit as DSsplit
# from runners.solo import Runner
# model = ResnetBaseline(n_classes = 1)

# model_label = 'code151davbflag'
# epochs = 10
# from dataloaders.code15 import CODE as DS
# from dataloaders.code15 import CODEsplit as DSsplit
# from runners.solo import Runner
# model = ResnetBaseline(n_classes = 1)

# model_label = 'code15flag1davb'
# epochs = 10
# from runners.multiclass import Runner
# from dataloaders.code15 import CODE as DS
# from dataloaders.code15 import CODEsplit as DSsplit
# model = ResnetBaseline(n_classes = 6)

model_label = 'code15flagnormal'
epochs = 10
from runners.normal import Runner
from dataloaders.code15 import CODE as DS
from dataloaders.code15 import CODEsplit as DSsplit
model = ResnetBaseline(n_classes = 1)

# # init
# database = DS(metadata_path = '/home/josegfer/datasets/code/output/metadata_1davb.csv')
database = DS(metadata_path = '/home/josegfer/datasets/code/output/metadata_normal.csv')
model = torch.load('output/{}/partial.pt'.format(model_label))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# runner = Runner(device = device, model = model, database = database, split = DSsplit, model_label = model_label, output_col = ['flag_1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'])
runner = Runner(device = device, model = model, database = database, split = DSsplit, model_label = model_label)

# # run
runner.train(epochs)
runner.eval()
