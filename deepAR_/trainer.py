import click
from conf.conf import Conf
import torch
from data.ts_dataset import TrainDataset, TestDataset, WeightedSampler
from models.net import Net
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from models.utils import G_loss_fun, N_loss_fun, init_metrics
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler
from progress_bar import ProgressBar
from time import time
import numpy as np

@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--conf_file_path', type=str, default="./conf/elect.yaml")
@click.option("--seed", type = int, default = None)
def main(exp_name, conf_file_path, seed):

    if exp_name is None:
        exp_name = click.prompt('▶ experiment name', default='default')

    log_each_step = True
    if '!' in exp_name:
        exp_name = exp_name.replace('!', '')
        log_each_step = False

    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name, log=log_each_step)
    print(f'\n{cnf}')
    print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

    trainer = Trainer(cnf=cnf)
    trainer.run()
    #trainer.run()


class Trainer(object):
    def __init__(self, cnf):
        self.cnf = cnf

        torch.set_num_threads(3)

        train_loader = TrainDataset
        test_loader  = TestDataset
        weight_loader = WeightedSampler

        # init dataset
        dataset_train   = train_loader(cnf, "elect")
        dataset_test   =  test_loader(cnf, "elect")
        dataset_weight = weight_loader(cnf, "elect")

        # init models
        model_choice = cnf.all_params["model"]
        if model_choice == "deepAR":
            self.model = Net(cnf.all_params)
        self.model = self.model.to(cnf.device)

        # init train - test loader
        self.train_loader = DataLoader(dataset    = dataset_train,
                                       batch_size = cnf.batch_size,
                                       num_workers= cnf.n_workers,
                                       sampler = dataset_weight)

        self.test_loader  = DataLoader(dataset = dataset_test,
                                      batch_size= cnf.predict_batch,
                                      sampler= RandomSampler(dataset_test),
                                      num_workers= cnf.n_workers)

        # init optimizer
        # we dont init loss function because it just used finding distribution parameters.
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = cnf.lr)

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)
        self.train_losses = []

        self.test_loss = []
        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_smape = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch= cnf.epochs)

        # possibly load checkpoint
        self.load_ck()

        print("Finished preparing datasets.")

    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.best_test_loss = self.best_test_loss

    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_test_loss': self.best_test_loss
        }
        torch.save(ck, self.log_path / 'training.ck')

        #print(dataset_train)

    def train(self):
        """
        train model for one epoch on training-set
        """
        start_time = time()
        self.model.train()

        print(self.model.train())

        times= []

        for i, (train_batch, idx, labels_batch) in enumerate(self.train_loader):
            t = time()
            self.optimizer.zero_grad()

            # Feed input to the model (deepAR)
            train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(self.cnf.device)  # not scaled
            labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(self.cnf.device)  # not scaled
            idx = idx.unsqueeze(0).to(self.cnf.device)


            # Compute loss
            loss = torch.zeros(1, device= self.cnf.device)
            hidden = self.model.init_hidden(self.cnf.batch_size)
            cell = self.model.init_cell(self.cnf.batch_size)

            for t in range(self.cnf.train_window):
                zero_index = (train_batch[t, :, 0] == 0)

                if t > 0 and torch.sum(zero_index) > 0:
                    train_batch[t, zero_index, 0] = mu[zero_index]


                mu, sigma, hidden, cell = self.model(train_batch[t].unsqueeze_(0).clone(),
                                                     idx,
                                                     hidden,
                                                     cell)
                if self.cnf.distribution == "g":
                    loss += G_loss_fun(mu, sigma, labels_batch[t])
                elif self.cnf.distribution == "negbin":
                    loss += N_loss_fun(mu, sigma, labels_batch[t])

            loss.backward()
            self.optimizer.step()
            loss = loss.item() / self.cnf.train_window
            self.train_losses.append(loss)

            # print an incredible progress bar
            times.append(time() - t)
            if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(self.train_losses):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
                self.progress_bar.inc()

        # log average loss of this epoch
        # evaluate
        mean_epoch_loss = np.mean(self.train_losses)

        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
        self.train_losses = []

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')

    def test(self, sample = True):
        """
        test model on the test set
        """
        self.model.eval()
        with torch.no_grad():
            plot_batch = np.random.randint(len(self.test_loader) - 1)

        summary_metric = {}
        raw_metrics = init_metrics(sample=sample)

        t = time()
        for i, (test_batch, id_batch, v, labels) in enumerate(self.test_loader):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(self.cnf.device)
            id_batch = id_batch.unsqueeze(0).to(self.cnf.device)
            v_batch = v.to(torch.float32).to(self.cnf.device)
            labels = labels.to(torch.float32).to(self.cnf.device)
            batch_size = test_batch.shape[1]
            input_mu = torch.zeros(batch_size, params.test_predict_start, device=self.cnf.device)  # scaled
            input_sigma = torch.zeros(batch_size, params.test_predict_start, device=self.cnf.device)  # scaled
            hidden = self.model.init_hidden(batch_size)
            cell = self.model.init_cell(batch_size)

    def run(self):
        """
        start model traning procedure (train -> test -> checkpoint -> repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()



if __name__ == '__main__':
    main()