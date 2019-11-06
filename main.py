import torch.utils.data
import torch
from torch.backends import cudnn
from torch.autograd import Variable

import numpy as np
import tqdm

from pytorch_metric_learning import trainers
from pytorch_metric_learning import losses

from lib.model import Generator, Embedder
import torchvision.models as models

from dataset.cars196_dataset import Cars196Dataset
from dataset.random_fixed_size_crop_mod import RandomFixedSizeCrop

dim = 512
data = "car"
lr = 7e-5
batch_size = 120
crop_size = 227

googlenet = models.googlenet(pretrained=True)
googlenet.fc = nn.Liear(1024, dim)
gen_model = Generator()
embed_model = Embedder()

trunk_optimizer = torch.optim.Adam(googlenet.parameters(), lr=lr)
gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=lr)
embed_optimizer = torch.optim.Adam(embed_model.parameters(), lr=lr)

dataset_train = dataset_class(['train'], load_in_memory=load_in_memory)
dataset_test = dataset_class(['test'], load_in_memory=load_in_memory)

stream_train = RandomFixedSizeCrop(DataStream(
    dataset_train, iteration_scheme=SequentialScheme(
        dataset_train.num_examples, batch_size)),
    which_sources=('images',), center_crop=True, window_shape=crop_size)
stream_test = RandomFixedSizeCrop(DataStream(
    dataset_test, iteration_scheme=SequentialScheme(
        dataset_test.num_examples, batch_size)),
    which_sources=('images',), center_crop=True, window_shape=crop_size)

x_batches = []
c_batches = []
for batch in tqdm(copy.copy(stream_test.get_epoch_iterator())):
    x_batch_data, c_batch_data = batch
    x_batches.append(x_batch_data)
    c_batches.append(c_batch_data)
x_data = np.concatenate(x_batches)
c_data = np.concatenate(c_batches)
x_data = torch.tensor(x_data)
c_data = torch.tensor(c_data)
dataset = torch.utils.data.TensorDataset(x_data, c_data)

# Set up your models, optimizers, loss functions etc.
models = {"trunk": googlenet,
          "embedder": embed_model,
          "G_neg_model": gen_model}

optimizers = {"trunk_optimizer": trunk_optimizer,
              "embedder_optimizer": embed_optimizer,
              "G_neg_model_optimizer": gen_optimizer}

loss_funcs = {"metric_loss": losses.AngularNPairs(alpha=35),
              "synth_loss": losses.Angular(alpha=35),
              "G_neg_adv": losses.Angular(alpha=35)}

mining_funcs = {}

loss_weights = {"metric_loss": 1,
                "classifier_loss": 0,
                "synth_loss": 0.1,
                "G_neg_adv": 0.1,
                "G_neg_hard": 0.1,
                "G_neg_reg": 0.1}

# Create trainer object
trainer = trainers.DeepAdversarialMetricLearning(
    models=models,
    optimizers=optimizers,
    batch_size=120,
    loss_funcs=loss_funcs,
    mining_funcs=mining_funcs,
    num_epochs=100,
    iterations_per_epoch=100,
    dataset=dataset,
    loss_weights=loss_weights
)

# Train!
trainer.train()
