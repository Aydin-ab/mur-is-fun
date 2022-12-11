## Standard libraries
import os
import numpy as np
#from PIL import Image
from typing import Any
from collections import defaultdict
import time
import tree  # EDIT
import random as random_py  # EDIT

## Imports for plotting
import matplotlib.pyplot as plt
# %matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## Progress bar
from tqdm.auto import tqdm

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
from jax import random

## Flax (NN in JAX)
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

## Optax (Optimizers in JAX)
import optax

import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

tfd = tfp.distributions

## PyTorch
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST

"""We will use the same path variables `DATASET_PATH` and `CHECKPOINT_PATH` as in the previous tutorials. Adjust the paths if necessary."""

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "saved_models/tutorial5_jax"

# Seeding for random operations
seed = 0
main_rng = random.PRNGKey(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
random_py.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.random.manual_seed(seed)

print("Device:", jax.devices()[0])

"""We also have pretrained models and TensorBoards (more on this later) for this tutorial, and download them below."""

import urllib.request
from urllib.error import HTTPError
# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial5/"
# Files to download
pretrained_files = ["GoogleNet.ckpt", "ResNet.ckpt", "PreActResNet.ckpt", "DenseNet.ckpt",
                    "tensorboards/GoogleNet/events.out.tfevents.googlenet",
                    "tensorboards/ResNet/events.out.tfevents.resnet",
                    "tensorboards/PreActResNet/events.out.tfevents.preactresnet",
                    "tensorboards/DenseNet/events.out.tfevents.densenet"]
# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/",1)[0], exist_ok=True)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)

"""We will use this information to normalize our data accordingly. 
Additionally, we will use transformations from the package `torchvision` to implement data augmentations during training. 
This reduces the risk of overfitting and helps CNNs to generalize better. 
Specifically, we will apply two random augmentations. 

First, we will flip each image horizontally by a chance of 50% (`transforms.RandomHorizontalFlip`). 
The object class usually does not change when flipping an image, and we don't expect any image information to be dependent 
on the horizontal orientation. 
This would be however different if we would try to detect digits or letters in an image, as those have a certain orientation.

The second augmentation we use is called `transforms.RandomResizedCrop`. 
This transformation scales the image in a small range, while eventually changing the aspect ratio, 
and crops it afterward in the previous size. 
Therefore, the actual pixel values change while the content or overall semantics of the image stays the same. 

We will randomly split the training dataset into a training and a validation set. 
The validation set will be used for determining early stopping. 
After finishing the training, we test the models on the CIFAR test set.
"""

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

"""Throughout this tutorial, we will train and evaluate the models on the CIFAR10 dataset. 
This allows you to compare the results obtained here with the model you have implemented in the first assignment.
As we have learned from the previous tutorial about initialization, 
it is important to have the data preprocessed with a zero mean. 
Therefore, as a first step, we will calculate the mean and standard deviation of the CIFAR dataset:"""

dataset = 'CIFAR10'
# dataset = 'FMNIST'

if dataset == 'CIFAR10':
    # train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    train_dataset = CIFAR10(root="./data/CIFAR10", train=True, download=False)
    image_dim = 32
    num_epochs = 140

    # DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
    # DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
    DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    DATA_STD = np.array([0.2023, 0.1994, 0.2010])
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    # Transformations applied on each image => bring them into a numpy array
    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD  # EDIT
        return img

    test_transform = transforms.Compose([
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
        image_to_numpy
    ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
        image_to_numpy
    ])
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    # train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    train_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=train_transform, download=False)
    # val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    val_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=test_transform, download=False)

    train_set, _ = torch.utils.data.random_split(train_dataset, [49900, 100], generator=torch.Generator().manual_seed(seed))
    _, val_set = torch.utils.data.random_split(val_dataset, [49900, 100], generator=torch.Generator().manual_seed(seed))

    # Loading the test set
    # test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
    test_set = CIFAR10(root="./data/CIFAR10", train=False, transform=test_transform, download=False)
elif dataset == 'FMNIST':
    train_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, download=False)
    image_dim = 28
    num_epochs = 30

    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2)).numpy()
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2)).numpy()
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    # Transformations applied on each image => bring them into a numpy array
    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)[:,:,None]
        img = (img / 255. - DATA_MEANS) / DATA_STD  # EDIT
        return img

    test_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=train_transform, download=False)
    val_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=test_transform, download=False)

    train_set, _ = torch.utils.data.random_split(train_dataset, [59900, 100], generator=torch.Generator().manual_seed(seed))
    _, val_set = torch.utils.data.random_split(val_dataset, [59900, 100], generator=torch.Generator().manual_seed(seed))

    # Loading the test set
    test_set = FashionMNIST(root="./data/fashionMNIST", train=False, transform=test_transform, download=False)





# We define a set of data loaders that we can use for training and validation
train_loader = data.DataLoader(train_set,
                               batch_size=128,
                               shuffle=True,
                               drop_last=True,
                               collate_fn=numpy_collate)
val_loader   = data.DataLoader(val_set,
                               batch_size=128,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate)
# noinspection PyArgumentList
test_loader  = data.DataLoader(test_set,
                               batch_size=128,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate)



"""To verify that our normalization works, we can print out the mean and standard deviation of the single batch. 
The mean should be close to 0 and the standard deviation close to 1 for each channel:"""

# imgs, _ = next(iter(train_loader))
# # print("Batch mean", imgs.mean(axis=(0,1,2)))
# # print("Batch std", imgs.std(axis=(0,1,2)))
#
# """Finally, let's visualize a few images from the training set, and how they look like after random data augmentation: """
#
# NUM_IMAGES = 4
# images = [train_dataset[idx][0] for idx in range(NUM_IMAGES)]
# orig_images = [Image.fromarray(train_dataset.data[idx]) for idx in range(NUM_IMAGES)]
# orig_images = [test_transform(img) for img in orig_images]
#
# imgs = np.stack(images + orig_images, axis=0)
# imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
# img_grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, pad_value=0.5)
# img_grid = img_grid.permute(1, 2, 0)

# plt.figure(figsize=(8,8))
# plt.title("Augmentation examples on CIFAR10 (Top: augmented, bottom: original)")
# plt.imshow(img_grid)
# plt.axis('off')
# plt.show()
# plt.close()

#%%

"""## Trainer Module

In the [PyTorch version]
(https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html) 
of this tutorial, we would now introduce the framework [PyTorch Lightning](https://www.pytorchlightning.ai/) 
which simplifies the overall training of a model. So far (June 2022), 
there is no clear equivalent of it for JAX. 
Some basic training functionalities are implemented in `flax.training` 
([documentation](https://flax.readthedocs.io/en/latest/flax.training.html)),
 and predefined training modules are implemented in `trax` 
 ([documentation](https://trax-ml.readthedocs.io/en/latest/)), but neither provide a complete, 
 flexible training package yet like PyTorch Lightning.
   Hence, we need to write our own small training loop. 

   
For this, we take inspiration from PyTorch Lightning and build a trainer module/object with the following main functionalities:

1. *Storing model and parameters*: 
    In order to train multiple models with different hyperparameters,
    the trainer module creates an instance of the model class, and keeps the parameters in the same class. 
    This way, we can easily apply a model with its parameters on new inputs.
2. *Initialization of model and training state*: 
    During initialization of the trainer, we initialize the model parameters and a new train state, 
    which includes the optimizer and possible learning rate schedulers.
3. *Training, validation and test loops*:
     Similar to PyTorch Lightning, we implement simple training, validation and test loops,
     where subclasses of this trainer could overwrite the respective training, validation or test steps. 
     Since in this tutorial, all models will have the same objective, i.e. classification on CIFAR10, 
     we will pre-specify them in the trainer module below.
4. *Logging, saving and loading of models*: 
    To keep track of the training, we implement functionalities to log the training progress and save the best model 
    on the validation set. 
    Afterwards, this model can be loaded from the disk. 

Before starting to implement a trainer module with these functionalities, we need to take one prior step.
 The networks we will implement in this tutorial use BatchNormalization, which carries an exponential average 
 of the prior batch statistics (mean and std) to apply during evaluation. 
 In PyTorch, this is simply tracked by an object attribute of an object of the class `nn.BatchNorm2d`,
 but in JAX, we only work with functions. Hence, we need to take care of the batch statistics ourselves, 
 similar to the parameters, and enter them during every forward pass. 
 To simplify this a little, we overwrite the `train_state.TrainState` class of Flax by adding a field for the batch statistics:
"""

class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics and logvar parameters  # EDIT
    batch_stats: Any
    params_logvar: Any  # EDIT

"""With this, the training state contains both the training parameters and the batch statistics, 
which makes it easier to keep everything in one place.

Now that the batch statistics are sorted out, we can implement our full training module:
"""

class TrainerModule:

    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 model_hparams : dict,
                 optimizer_name : str,
                 optimizer_hparams : dict,
                 objective_hparams : dict,
                 exmp_imgs : Any,
                 seed=0):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.objective_hparams = objective_hparams
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        # Prepare logging
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)
        print(self.model.tabulate(random.PRNGKey(0), x=exmp_imgs[0]))

    def create_functions(self):
        def merge_params(params_1, params_2):  # EDIT
            flat_params_1 = flax.traverse_util.flatten_dict(params_1)
            flat_params_2 = flax.traverse_util.flatten_dict(params_2)
            flat_params = flat_params_1 | flat_params_2
            unflat_params = flax.traverse_util.unflatten_dict(flat_params)
            return unflat_params
        def split_params(params):  # EDIT
            flat_params_fixed = flax.traverse_util.flatten_dict(params)
            flat_params_jac = flax.traverse_util.flatten_dict(params)
            keys = flat_params_fixed.keys()
            for key in list(keys):
                if "Dense" in str(key):
                    flat_params_fixed.pop(key)
                else:
                    flat_params_jac.pop(key)
            unflat_params_fixed = flax.traverse_util.unflatten_dict(flat_params_fixed)
            unflat_params_fixed = unflat_params_fixed
            unflat_params_jac = flax.traverse_util.unflatten_dict(flat_params_jac)
            unflat_params_jac = unflat_params_jac
            return unflat_params_fixed, unflat_params_jac
        def calculate_cov(jac, sigma):  # EDIT
            # jac has shape (batch_dim, output_dim, params_dims...)
            # jac_2D has shape (batch_dim * output_dim, nb_params)
            batch_dim, output_dim = jac.shape[:2]
            jac_2D = jnp.reshape(jac, (batch_dim * output_dim, -1))
            # sigma_flatten has shape (nb_params,) and will be broadcasted to the same shape as jac_2D
            sigma_flatten = jnp.reshape(sigma, (-1,))
            # jac_sigma_product has the same shape as jac_2D
            jac_sigma_product = jnp.multiply(jac_2D, sigma_flatten)
            cov = jnp.matmul(jac_sigma_product, jac_2D.T)
            cov = jnp.reshape(cov, (batch_dim, output_dim, batch_dim, output_dim))
            return cov

        def calculate_entropy(probs) :
            p_logp = jnp.multiply(probs, jnp.log(probs))
            h = p_logp.sum(axis= 1) # We sum along the output dim
            return h

        def approximate_xstar(inputs, params_fixed, params_jac, batch_stats, radius= 1) :
            # input_point shape 3x32x32 = 1 image
            entropy_fun = lambda inputs : calculate_entropy(
                self.model.apply({'params': merge_params(params_fixed, params_jac), 'batch_stats': batch_stats}, inputs, train=False)
            )
            jacobian = jax.jacobian(entropy_fun)(inputs).mean(axis= 1)
            xstar = inputs + radius * jacobian / jnp.linalg.norm(jacobian)
            return xstar



        def calculate_moments(params_mean, params_logvar, inputs, batch_stats, params_tree, prior):  # EDIT
            if prior:
                params_mean = jax.tree_map(lambda x: params_mean * jnp.ones_like(x), params_tree)
                params_var = params_logvar
                params_logvar = jax.tree_map(lambda x: jnp.log(params_var) * jnp.ones_like(x), params_tree)

            ### last-layer Jacobian
            params_fixed, params_jac = split_params(params_mean)
            _params = merge_params(params_fixed, params_jac)
            params_logvar_fixed, params_logvar_jac = split_params(params_logvar)
            params_var_jac = jax.tree_map(lambda x: jnp.exp(x), params_logvar_jac)
            pred_fn = lambda params_jac: self.model.apply({'params': merge_params(params_fixed, params_jac), 'batch_stats': batch_stats}, inputs, train=True, mutable=['batch_stats'])
            
            ### full Jacobian:
            # params_jac = params_mean
            # params_var_jac = jax.tree_map(lambda x: jnp.exp(x), params_logvar)
            # pred_fn = lambda params_jac: self.model.apply({'params': params_jac, 'batch_stats': batch_stats}, inputs, train=True, mutable=['batch_stats'])
            mean = pred_fn(params_jac)[0]
            jacobian = jax.jacobian(pred_fn)(params_jac)[0]

            cov = tree.map_structure(calculate_cov, jacobian, params_var_jac)
            cov = jnp.stack(tree.flatten(cov), axis=0).sum(axis=0)
            return mean, cov
        def calculate_function_kl(params_mean, params_logvar, inputs, batch_stats):  # EDIT
            params_prior_mean = self.objective_hparams["prior_mean"]
            params_prior_var = self.objective_hparams["prior_var"]

            mean, cov = calculate_moments(params_mean, params_logvar, inputs, batch_stats, jax.lax.stop_gradient(params_mean), False)
            # mean_prior, cov_prior = calculate_moments(params_prior_mean, params_prior_var, inputs, batch_stats, jax.lax.stop_gradient(params_mean), True)
            
            # Compute mean prior following the MUR behavior
            params_fixed, params_jac = split_params(params_mean)
            xstar = approximate_xstar(inputs, params_fixed, params_jac, batch_stats)
            mean_prior = self.model.apply({'params': merge_params(params_fixed, params_jac), 'batch_stats': batch_stats}, xstar, train=False)

            kl = 0
            n_samples = mean.shape[0]
            n_output_dims = mean.shape[-1]
            cov_jitter = 10  # TODO: remove hardcoding
            for j in range(n_output_dims):
                _mean = mean[:, j].transpose()
                _cov = cov[:, j, :, j] + jnp.eye(n_samples) * cov_jitter

                # _mean_prior = mean_prior[:, j].transpose()
                # _cov_prior = cov_prior[:, j, :, j] + jnp.eye(n_samples) * cov_jitter
                #_mean_prior = jnp.ones([n_samples]) * params_prior_mean
                _mean_prior = mean_prior[:, j].transpose()
                _cov_prior = jnp.eye(n_samples) * params_prior_var


                q = tfd.MultivariateNormalFullCovariance(
                    loc=_mean,
                    covariance_matrix=_cov,
                    validate_args=False,
                    allow_nan_stats=True,
                )
                p = tfd.MultivariateNormalFullCovariance(
                    loc=_mean_prior,
                    covariance_matrix=_cov_prior,
                    validate_args=False,
                    allow_nan_stats=True,
                )
                _kl = tfd.kl_divergence(q, p, allow_nan_stats=False)

                kl += _kl

            return kl
        def sample_parameters(params, params_logvar, rng_key):  # EDIT
          eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), params_logvar)
          params_std_sample = jax.tree_map(lambda x, y: x * jax.numpy.exp(y) ** 0.5, eps, params_logvar)
          params_sample = jax.tree_map(lambda x, y: x + y, params, params_std_sample)
          return params_sample
        def calculate_params_kl(mean, logvar):  # EDIT
          var = jax.tree_map(lambda x: jax.numpy.exp(x), logvar)
          kl_fn = lambda x, y: 0.5 * (-jax.numpy.log(y ** 0.5) + y + x ** 2 - 1)
          kl = jax.numpy.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: kl_fn(x, y), mean, var))[0])
          return kl
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(params, params_logvar, rng_key, batch_stats, batch, train):
            imgs, labels = batch

            if self.objective_hparams["stochastic"]:  # EDIT
                params = sample_parameters(params, params_logvar, rng_key)  # EDIT

            # Run model. During training, we need to update the BatchNorm statistics.
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                    imgs,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)
            logits, new_model_state = outs if train else (outs, None)

            if self.objective_hparams["reg_type"] == "parameter_kl":  # EDIT
                assert self.objective_hparams["stochastic"] == True
                reg = calculate_params_kl(params, params_logvar)  # EDIT
            elif self.objective_hparams["reg_type"] == "function_kl":  # EDIT
                reg = calculate_function_kl(params, params_logvar, imgs, batch_stats)  # EDIT
            elif self.objective_hparams["reg_type"] == "function_norm":  # EDIT
                reg = jax.numpy.sum(jax.numpy.square(logits))  # EDIT
            else:
                reg = 0

            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            loss += self.objective_hparams["reg_scale"] * reg  # EDIT

            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, new_model_state)
        # Training function
        def train_step(state, batch, rng_key):
            loss_fn = lambda params, params_logvar: calculate_loss(params, params_logvar, rng_key, state.batch_stats, batch, train=True)  # EDIT
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, argnums=(0,1,), has_aux=True)(state.params, state.params_logvar)  # EDIT
            grads, grads_logvar = grads[0], grads[1]  # EDIT
            loss, acc, new_model_state = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
            return state, loss, acc
        # Eval function
        def eval_step(state, batch, rng_key):
            # Return the accuracy for a single batch
            _, (acc, _) = calculate_loss(state.params, state.params_logvar, rng_key, state.batch_stats, batch, train=False)  # EDIT
            return acc
        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = jax.random.PRNGKey(self.seed)
        init_rng_logvar, _ = random.split(init_rng)  # EDIT

        variables = self.model.init(init_rng, exmp_imgs, train=True)
        variables_logvar = self.model.init(init_rng_logvar, exmp_imgs, train=True)  # EDIT

        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']
        self.init_params_logvar = jax.tree_map(lambda x: x - 20.0, variables_logvar['params'])  # EDIT

        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'
        # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        # lr_schedule = optax.piecewise_constant_schedule(  # original
        #     init_value=self.optimizer_hparams.pop('lr'),
        #     boundaries_and_scales=
        #         {int(num_steps_per_epoch*num_epochs*0.6): 0.1,
        #          int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        # )
        lr_schedule = optax.cosine_decay_schedule(  # EDIT
            init_value=self.optimizer_hparams.pop('lr'),
            decay_steps=num_steps_per_epoch*num_epochs,
            alpha=1e-3,
        )
        # Clip gradients at max value, and evt. apply weight decay
        transf = []
        # transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))
        optimizer = optax.chain(
            # *transf,  # EDIT
            opt_class(lr_schedule, **self.optimizer_hparams)
        )
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       params_logvar=self.init_params_logvar, # if self.state is None else self.state.params,  # EDIT
                                       batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                       tx=optimizer)

    def train_model(self, train_loader, val_loader, rng_key, num_epochs=200):  # EDIT
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Track best eval accuracy
        best_eval = 0.0
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            self.train_epoch(train_loader, epoch=epoch_idx, rng_key=rng_key)  # EDIT
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(test_loader, rng_key)  # EDIT
                # eval_acc = self.eval_model(val_loader, rng_key)  # EDIT
                self.logger.add_scalar('val/acc', eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()
                print(f"\nEvaluation Accuracy: {eval_acc*100:.2f}")  # EDIT

    def train_epoch(self, train_loader, epoch, rng_key):  # EDIT
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        # for batch in train_loader:
        for batch in train_loader:
        # for batch in tqdm(train_loader, desc='Training', leave=False):
            self.state, loss, acc = self.train_step(self.state, batch, rng_key)  # EDIT
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar('train/'+key, avg_val, global_step=epoch)

    def eval_model(self, data_loader, rng_key):  # EDIT
        # Test model on all images of a data loader and return avg loss
        correct_class, count = 0, 0
        for batch in data_loader:
            acc = self.eval_step(self.state, batch, rng_key)  # EDIT
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'params_logvar': self.state.params_logvar,  # EDIT
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                   overwrite=True)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
            params_logvar = jax.tree_map(lambda x: x - 10.0, state_dict['params'])  # EDIT

        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       params_logvar=params_logvar,  # EDIT
                                       batch_stats=state_dict['batch_stats'],
                                       tx=self.state.tx if self.state else optax.sgd(0.1)   # Default optimizer
                                      )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))

"""Next, we can use this trainer module to create a compact training function:"""

# def train_classifier(*args, num_epochs=200, rng_key, **kwargs):  # EDIT
#     # Create a trainer module with specified hyperparameters
#     trainer = TrainerModule(*args, **kwargs)
#     if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
#         trainer.train_model(train_loader, val_loader, rng_key, num_epochs=num_epochs)
#         trainer.load_model()
#     else:
#         trainer.load_model(pretrained=True)
#         trainer.train_model(train_loader, val_loader, rng_key, num_epochs=num_epochs)  # EDIT
#     # Test trained model
#     val_acc = trainer.eval_model(val_loader, rng_key)  # EDIT
#     test_acc = trainer.eval_model(test_loader, rng_key)  # EDIT
#     return trainer, {'val': val_acc, 'test': test_acc}

# Conv initialized with kaiming int, but uses fan-out instead of fan-in mode
# Fan-out focuses on the gradient distribution, and is commonly used in ResNets
resnet_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')

class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(2, 2), kernel_init=resnet_kernel_init)(x)

        x_out = self.act_fn(z + x)
        return x_out

"""The overall ResNet architecture consists of stacking multiple ResNet blocks, of which some are downsampling the input. 
When talking about ResNet blocks in the whole network, we usually group them by the same output shape. 
Hence, if we say the ResNet has `[3,3,3]` blocks, it means that we have 3 times a group of 3 ResNet blocks, 
where a subsampling is taking place in the fourth and seventh block. 
The ResNet with `[3,3,3]` blocks on CIFAR10 is visualized below.

<center width="100%"><img src="https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial5/resnet_notation.svg?raw=1" width="500px"></center>

The three groups operate on the resolutions $32\times32$, $16\times16$ and $8\times8$ respectively. 
The blocks in orange denote ResNet blocks with downsampling.
 The same notation is used by many other implementations such as in the [torchvision library](https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18) 
 from PyTorch or [flaxmodels](https://github.com/matthias-wright/flaxmodels) 
 (pretrained ResNets and more for JAX). Thus, our code looks as follows:
"""

class ResNet(nn.Module):
    num_classes : int
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], (3, 3), (1, 1), padding=[(1, 1), (1, 1)], kernel_init=resnet_kernel_init, use_bias=False)(x)
        # x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=resnet_kernel_init, use_bias=False)(x)  # original
        if self.block_class == ResNetBlock:  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)

        # Mapping to classification output
        x = x.mean(axis=(1, 2)) # x was shape batch x 4 x 4 x 512 and now it's shape batch x 512 bcs we average on the pixels
        x = nn.Dense(self.num_classes)(x)
        return x

def train_classifier(*args, num_epochs=200, rng_key, **kwargs):  # EDIT
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(*args, **kwargs)
    if True: #not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(train_loader, val_loader, rng_key, num_epochs=num_epochs)
        # trainer.load_model()
    else:
        trainer.load_model(pretrained=True)
        trainer.train_model(train_loader, val_loader, rng_key, num_epochs=num_epochs)  # EDIT
        # trainer.train_model(train_loader, val_loader, rng_key, num_epochs=num_epochs)  # EDIT
    # Test trained model
    val_acc = trainer.eval_model(val_loader, rng_key)  # EDIT
    test_acc = trainer.eval_model(test_loader, rng_key)  # EDIT
    print(f"\nValidation Accuracy: {val_acc*100:.2f}")
    print(f"Test Accuracy: {test_acc*100:.2f}")
    return trainer, {'val': val_acc, 'test': test_acc}

rng_key = main_rng

# from jax import config
# config.update('jax_disable_jit', True)

resnet_trainer, resnet_results = train_classifier(model_name="ResNet",
                                                  model_class=ResNet,
                                                  model_hparams={"num_classes": 10,
                                                                 # "c_hidden": (16, 32, 64),
                                                                 # "num_blocks": (3, 3, 3),
                                                                 "c_hidden": (64, 128, 256, 512),  # EDIT
                                                                 "num_blocks": (2, 2, 2, 2),  # EDIT
                                                                 "act_fn": nn.relu,
                                                                 "block_class": ResNetBlock},
                                                  optimizer_name="SGD",
                                                  optimizer_hparams={"lr": 0.1,
                                                                     "momentum": 0.9,
                                                                     # "weight_decay": 0.0  # EDIT
                                                                     },
                                                  objective_hparams={"stochastic": True,  # EDIT
                                                                     "reg_scale": 1.,  # EDIT
                                                                     "prior_mean": 0.,  # EDIT
                                                                     "prior_var": 10000,  # EDIT
                                                                     "reg_type": "function_kl"},  # EDIT
                                                  exmp_imgs=jax.device_put(
                                                      next(iter(train_loader))[0]),
                                                  num_epochs=num_epochs,  # EDIT
                                                  rng_key=rng_key)  # EDIT
