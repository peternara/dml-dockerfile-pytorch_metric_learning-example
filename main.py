import torch.utils.data
import torch
from torch.backends import cudnn
from torch.autograd import Variable

from pytorch_metric_learning import trainers

from lib.model import Generator, Embedder
import torchvision.models as models

dim = 512
data = "car"
lr = 7e-5

googlenet = models.googlenet(pretrained=True)
googlenet.fc = nn.Liear(1024, dim)
gen_model = Generator()
embed_model = Embedder()

trunk_optimizer = torch.optim.Adam(googlenet.parameters(), lr=lr)
gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=lr)
embed_optimizer = torch.optim.Adam(embed_model.parameters(), lr=lr)

dataset = DataSet.create(data, ratio=ratio, width=width, origin_width=args.origin_width, root=args.data_root)

# Set up your models, optimizers, loss functions etc.
models = {"trunk": googlenet, 
          "embedder": embedder,
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