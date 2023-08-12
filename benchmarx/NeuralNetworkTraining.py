### Imports
import jax
import jax.numpy as jnp               # JAX NumPy

from flax import linen as nn          # The Linen API
from flax.training import train_state
import optax                          # The Optax gradient processing and 
                                      # optimization library

import numpy as np                    # Ordinary NumPy
import tensorflow_datasets as tfds    # TFDS for MNIST
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import time
import json
import logging
import sys
import os
from typing import List, Dict

from benchmarx.plotter import Plotter

#sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), "..")))

### Model
class CNN(nn.Module):
    @nn.compact
    # Provide a constructor to register a new parameter 
    # and return its initial value
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)    # There are 10 classes in FashionMNIST
        return x


model = CNN()

class NeuralNetwokTraining:
    train_ds = None
    test_ds = None
    config: Dict = None
    aval_methods = ['sgd', 'adam', 'novograd', 'adagrad']
    method: str = None
    def __init__(self, config = None, method: str = 'sgd') -> None:
        global model
        model = CNN()
        if method not in self.aval_methods:
            logging.critical(f'Wrong method {method}. Available methods: {self.aval_methods}.')
            exit(1)
        self.method = method
        if config is not None:
            self.config = config
        else:
            self.config = {
                        "N_EPOCHS": 20,
                        "BATCH_SIZE": 1128,
                        "Dataset": "FashionMNIST",
                        "seed": 282,
                        "LEARNING_RATE": 1e-1
                        }
        self.train_ds, self.test_ds = self._get_datasets()

    def _get_datasets(self):
        ds_builder = tfds.builder('fashion_mnist')
        ds_builder.download_and_prepare()
        # Split into training/test sets
        train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', 
                                                    batch_size=-1))
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', 
                                                    batch_size=-1))
        # Convert to floating-points
        train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
        test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
        return train_ds, test_ds
    
    ### Utilities
    def compute_metrics(self, logits, labels):
        loss = jnp.mean(optax.softmax_cross_entropy(logits, 
                                            jax.nn.one_hot(labels, num_classes=10)))
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {
            "loss": loss,
            "accuracy": accuracy
        }
        return metrics

    #@jax.jit
    def train_step(self, state, batch):
        def loss(params):
            logits = model.apply({'params': params}, batch['image'])
            loss = jnp.mean(optax.softmax_cross_entropy(
                logits=logits, 
                labels=jax.nn.one_hot(batch['label'], num_classes=10)))
            return loss, logits
        grad_fn = jax.value_and_grad(loss, has_aux=True)
        (_, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = self.compute_metrics(logits, batch['label'])
        return state, metrics

    def train_epoch(self, state, train_ds, batch_size, epoch, rng):
        train_ds_size = len(train_ds['image'])
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, len(train_ds['image']))
        perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))

        batch_metrics = []
        total_time = 0
        iters = 0
        for perm in perms:
            batch = {k: v[perm, ...] for k, v in train_ds.items()}
            start = time.time()
            state, metrics = self.train_step(state, batch)
            end = time.time()
            total_time += (end - start)
            iters += 1
            batch_metrics.append(metrics)

        training_batch_metrics = jax.device_get(batch_metrics)
        training_epoch_metrics = {
            k: np.mean([metrics[k] for metrics in training_batch_metrics])
            for k in training_batch_metrics[0]}
        print(f"🤖 Epoch: {epoch}, "
            f"Train loss: {training_epoch_metrics['loss']:.3f}, "
            f"Train accuracy: {100*training_epoch_metrics['accuracy']:.2f}")

        return state, training_epoch_metrics, [total_time, iters]
    
    
    #@jax.jit
    def eval_step(self, params, batch):
        logits = model.apply({'params': params}, batch['image'])
        return self.compute_metrics(logits, batch['label'])

    def eval_model(self, model, test_ds):
        metrics = self.eval_step(model, test_ds)
        metrics = jax.device_get(metrics)
        eval_summary = jax.tree_map(lambda x: x.item(), metrics)
        print(f"🤖 Test loss: {eval_summary['loss']:.2f},"
            f" Test accuracy: {100*eval_summary['accuracy']:.2f}")
        return eval_summary
    
    def train_model_on_fashionmnist(self):
        RNG = jax.random.PRNGKey(self.config["seed"])
        RNG, init_RNG = jax.random.split(RNG)
        train_ds, test_ds = self.train_ds, self.test_ds

        input_shape = [1, 28, 28, 1] # Batch_size x Width x Height x Channels
        params = model.init(init_RNG, jnp.ones(input_shape))['params']

        # This creates transformation of the parameters w.r.t. Gradient Descent method
        # Note, that stochasticity come from the batches later
        optimizer = optax.sgd(learning_rate=self.config["LEARNING_RATE"])
        if self.method == 'adam':
            optimizer = optax.adam(learning_rate=self.config["LEARNING_RATE"])
        if self.method == 'adagrad':
            optimizer = optax.adagrad(learning_rate=self.config["LEARNING_RATE"])
        if self.method == 'novograd':
            optimizer = optax.novograd(learning_rate=self.config["LEARNING_RATE"])

        state = train_state.TrainState.create(apply_fn=model.apply, 
                                            params=params, 
                                            tx=optimizer)

        # Evaluate on the test set before training epoch
        test_metrics = self.eval_model(state.params, self.test_ds)
        metrics_to_return = dict()
        metrics_to_return['train_accuracy_history'] = list()
        metrics_to_return['test_accuracy_history'] = list()
        metrics_to_return['test_loss_history'] = list()
        metrics_to_return['train_loss_history'] = list()

        for epoch in tqdm(range(1, self.config["N_EPOCHS"])):
            # Use a separate PRNG key to permute image data during shuffling
            RNG, input_RNG = jax.random.split(RNG)
            # Run an optimization step over a training batch
            state, train_metrics, stats = self.train_epoch(state, train_ds, self.config["BATCH_SIZE"], 
                                            epoch, input_RNG)
            # # Evaluate on the test set after each training epoch
            #with open('test.txt', 'w') as file:
            #  print(train_metrics, file=file)
            #self.model = state.params
            test_metrics = self.eval_model(state.params, test_ds)
            
            metrics_to_return['test_loss_history'].append(float(test_metrics['loss']))
            metrics_to_return['train_loss_history'].append(float(train_metrics['loss']))
            metrics_to_return['train_accuracy_history'].append(float(train_metrics['accuracy']))
            metrics_to_return['test_accuracy_history'].append(float(test_metrics['accuracy']))
        #print(metrics_to_return)
        return metrics_to_return
    
    def __del__(self):
        model = CNN()
    


class NNBenchmark:
    # methods: {method: config}. LABEL key in config
    methods: Dict = None
    aval_methods = ['sgd', 'adam', 'novograd', 'adagrad']
    save: bool
    path: str
    def __init__(self, methods: Dict) -> None:
        for label, data in methods.items():
            if data['method'] not in self.aval_methods:
                logging.critical(f'Bad method {data["method"]}. Available methods: {self.aval_methods}')
                exit(1)
        self.methods = methods
    
    def run(self, save: bool = False, path: str = ''):
        if len(path) > 0:
            self.path = path
        result_metrics = dict()

        for label, config in self.methods.items():
            tmp_model = NeuralNetwokTraining(config=config, method=config['method'])
            tmp_metrics = tmp_model.train_model_on_fashionmnist()
            method = config['method']
            config.pop('method')
            result_metrics[label] = tmp_metrics

        if save and len(path) > 0:
            with open(path, 'w') as fp:
                json.dump(result_metrics, fp, indent=4)
        else:
            print(result_metrics)

    def plot(self,
             metrics_to_plot = ['train_acc', 'test_acc', 'train_loss', 'test_loss'],
             data_path='',
             dir_path = '.',
             save: bool = True,
             show: bool = False,
             log: bool = True):
        if len(data_path) == 0:
            data_path = self.path
        
        plotter_ = Plotter(metrics=metrics_to_plot, 
                           data_path=data_path,
                           dir_path=dir_path)
        plotter_._plot_nn_data(save=save, show=show, log=log)


def run_experiment():
    bencmark = NNBenchmark({
        'adam' : {
            "N_EPOCHS": 40,
            "BATCH_SIZE": 25000,
            "Dataset": "FashionMNIST",
            "seed": 282,
            "LEARNING_RATE": 1e-2,
            'method': 'adam'
        },
        'sgd_1' : {
            "N_EPOCHS": 40,
            "BATCH_SIZE": 25000,
            "Dataset": "FashionMNIST",
            "seed": 282,
            "LEARNING_RATE": 1e-2,
            'method': 'sgd'
        },
        'sgd_2' : {
            "N_EPOCHS": 40,
            "BATCH_SIZE": 25000,
            "Dataset": "FashionMNIST",
            "seed": 282,
            "LEARNING_RATE": 0.05,
            'method': 'sgd'
        }
    })

    path_to_save = 'sgd_model_data.json'
    bencmark.run(save=True, path=path_to_save)


def draw():
    from plotter import Plotter

    plot = Plotter(['train_acc', 'test_acc', 'train_loss', 'test_loss'], 
                   data_path='sgd_model_data.json',
                   dir_path='nn_plots')
    plot._plot_nn_data()

def test_local():
    run_experiment()
    draw()


if __name__ == '__main__':
    test_local()