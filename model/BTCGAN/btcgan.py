import warnings, os

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, Sigmoid, Conv2d, ModuleList, functional, Tanh, PReLU, Softmax

from model.BTCGAN.data_transformer import DataTransformer
from model.BTCGAN.base import BaseSynthesizer


class Discriminator(Module):
    def __init__(self, conv_dim, extra_dim, discriminator_dim, device, pac=10):
        super(Discriminator, self).__init__()
        self.pac = pac
        self.conv_pacdim = conv_dim * pac
        self.extra_pacdim = extra_dim * pac
        
        self.seqs = []
        dim = (conv_dim+extra_dim) * pac
        for item in list(discriminator_dim):
#             seq += [Linear(dim, item), BatchNorm1d(item), LeakyReLU(0.2), Dropout(0.5)]
            self.seqs.append(Sequential(
                Linear(dim, item), BatchNorm1d(item), LeakyReLU(0.2), Dropout(0.5),
            ).to(device))
            dim = item + self.extra_pacdim
        
        self.act = Sequential(Linear(dim, 1), Sigmoid())
#         self.seq = Sequential(*seq)
        
    def forward(self, conv, extra):
        assert conv.shape[0] % self.pac == 0
        conv = conv.view((-1, self.conv_pacdim))
        extra = extra.view((-1, self.extra_pacdim))
        
        output = self.seqs[0](torch.cat([conv, extra], dim=1))
        output = torch.cat([output, extra], dim=1)
        for i in range(1, len(self.seqs)):
            output = self.seqs[i](output)
            output = torch.cat([output, extra], dim=1)
        return self.act(output)
    
    
class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        seq.append(Tanh())
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


class BTCGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.
    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), 
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1, use_gm=True,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, cuda_index=0,
                 max_clusters=10, bayes_edges=None, label=None, mean_weight=1, std_weight=1, mse_weight=1):

        assert batch_size % pac == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self._use_gm = use_gm
        self._max_clusters = max_clusters
        self._bayes_edges = bayes_edges
        self._mean_weight = mean_weight
        self._std_weight = std_weight
        self._mse_weight = mse_weight
        
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = f'cuda:{cuda_index}'

        self._device = torch.device(device)

        self._transformer = None
        self._generator = None
        self._continuous_pos_in_cond = None

        
    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError('Invalid columns found: {}'.format(invalid_columns))

    def fit(self, train_data, discrete_columns=tuple(), epochs=None, bayes_edges=None, label=None, structure_algorithm='PC', bayes_train_samples=100000, gm_epochs=5):
        """Fit the CTGAN Synthesizer models to the training data.
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._column_order = list(train_data.columns)
        self._validate_discrete_columns(train_data, discrete_columns)

        if label is not None:
            if bayes_edges is None:
                bayes_edges = []
            for col in self._column_order:
                if col != label:
                    bayes_edges.append((label, col))

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer(use_bayes=True, max_clusters=self._max_clusters, gm_epochs=gm_epochs, use_gm=self._use_gm)
        self._transformer.fit(train_data, discrete_columns, bayes_edges=bayes_edges, no_parents=label, structure_algorithm=structure_algorithm, bayes_train_samples=bayes_train_samples)
        
        train_data = self._transformer.transform(train_data)
        data_dim = self._transformer.output_dimensions
        conv_dim = self._transformer.cond_pos.sum().item()
        extra_dim = data_dim - conv_dim
        if extra_dim == 0:
            return
        self._continuous_pos_in_cond = []
        cur_pos = 0
        for col_info in self._transformer.output_info_list:
            if len(col_info) == 1: # 离散列
                cur_pos += col_info[0].dim
            else:
                self._continuous_pos_in_cond.append(cur_pos)
                cur_pos += col_info[1].dim

        self._generator = Generator(
            self._embedding_dim + conv_dim,
            self._generator_dim,
            len(self._column_order) - len(discrete_columns)
        ).to(self._device)

        discriminator = Discriminator(
            conv_dim, extra_dim,
            self._discriminator_dim,
            self._device,
            pac = self.pac,
        ).to(self._device)
        
        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay = self._generator_decay
        )
        
        optimizerD = optim.Adam(
            discriminator.parameters(), lr = self._discriminator_lr,
            betas = (0.5, 0.9), weight_decay = self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        
        train_discriminator = True
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            idx = np.arange(len(train_data))
            np.random.shuffle(idx)
            gen_idx = np.arange(len(train_data))
            np.random.shuffle(gen_idx)
            for id_ in range(steps_per_epoch):
                if train_discriminator: # 原来的鉴别器
                    fakez = torch.normal(mean=mean, std=std)

                    real = train_data[idx[id_*self._batch_size:(id_+1)*self._batch_size]]
                    fake_cond = real[:, self._transformer.cond_pos]
                    np.random.shuffle(real)
                    real_cond = real[:, self._transformer.cond_pos]
                    real_extra = real[:, ~self._transformer.cond_pos]
                    fake_cond = torch.from_numpy(fake_cond).contiguous().to(self._device)
                    fakez = torch.cat([fakez, fake_cond], dim=1)

                    fakeact = self._generator(fakez)
                    real_cond = torch.from_numpy(real_cond).contiguous().to(self._device)
                    real_extra = torch.from_numpy(real_extra).contiguous().to(self._device)

#                     torch_cat = []
#                     last_pos = 0
#                     for _, pos in enumerate(self._continuous_pos_in_cond):
#                         if pos != 0:
#                             if pos-last_pos > 1:
#                                 torch_cat.append(real_cond[:, last_pos:pos])
#                             else:
#                                 torch_cat.append(real_cond[:, last_pos:pos].reshape((-1, 1)))
#                         torch_cat.append(fakeact[:, _].reshape((-1, 1)))
#                         last_pos = pos
#                     if last_pos < real_cond.shape[1]:
#                         if real_cond.shape[1]-last_pos > 1:
#                             torch_cat.append(real_cond[:, last_pos:])
#                         else:
#                             torch_cat.append(real_cond[:, last_pos:].reshape((-1, 1)))
#                     fake_cat = torch.cat(torch_cat, dim=1)

                    y_fake = discriminator(fake_cond, fakeact)
                    y_real = discriminator(real_cond, real_extra)

                    loss_d = -(torch.mean(torch.log(y_real))) - (torch.mean(torch.log(1-y_fake)))
                    optimizerD.zero_grad()
                    loss_d.backward()
                    optimizerD.step()
                    yy_fake = y_fake

                fakez = torch.normal(mean=mean, std=std)
                
                condvec = train_data[gen_idx[id_*self._batch_size:(id_+1)*self._batch_size]]
                extra_data = condvec[:, ~self._transformer.cond_pos]
                extra_data = torch.from_numpy(extra_data).to(self._device)
                condvec = condvec[:, self._transformer.cond_pos]
                condvec = torch.from_numpy(condvec).contiguous().to(self._device)
                fakez = torch.cat([fakez, condvec], dim=1)

                fakeact = self._generator(fakez)
                
#                 torch_cat = []
#                 last_pos = 0
#                 for _, pos in enumerate(self._continuous_pos_in_cond):
#                     if pos != 0:
#                         if pos-last_pos > 1:
#                             torch_cat.append(condvec[:, last_pos:pos])
#                         else:
#                             torch_cat.append(condvec[:, last_pos:pos].reshape((-1, 1)))
#                     torch_cat.append(fakeact[:, _].reshape((-1, 1)))
#                     last_pos = pos
#                 if last_pos < condvec.shape[1]:
#                     if condvec.shape[1]-last_pos > 1:
#                         torch_cat.append(condvec[:, last_pos:])
#                     else:
#                         torch_cat.append(condvec[:, last_pos:].reshape((-1, 1)))
#                 fake_cat = torch.cat(torch_cat, dim=1)

                y_fake = discriminator(condvec, fakeact)
                
                mean_loss = ((fakeact.mean(dim=0) - extra_data.mean(dim=0))**2).sum() / extra_dim
                std_loss = ((fakeact.std(dim=0) - extra_data.std(dim=0))**2).sum() / extra_dim
                mse_loss = functional.mse_loss(fakeact, extra_data)
                loss_g = torch.mean(torch.log(1-y_fake)) + self._mean_weight*mean_loss + self._std_weight*std_loss + self._mse_weight*mse_loss
                
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            
            loss_d_real = torch.mean(y_real).detach().cpu()
            loss_d_fake = torch.mean(yy_fake).detach().cpu()
            if train_discriminator and loss_d_real - loss_d_fake > 0.05:
                train_discriminator = False
#                     discriminator.eval()
#                 print(i, 'stop')
            elif not train_discriminator and torch.mean(y_fake) >= 0.485:
                train_discriminator = True
#                     discriminator.train()
#                 print(i, 'start')
            if self._verbose and (i % 50 == 0 or i == epochs-1):
                loss_d = loss_d.detach().cpu()
#                 print('Epoch %d: LossG(%.4f -fake+l2): fake: %.4f, mean: %.4f, std: %.4f, mse:%.4f; LossD(-(real-fake) %.4f): real: %.4f, fake: %.4f' % 
#                      (i+1, loss_g.detach().cpu(), torch.mean(y_fake).detach().cpu(), mean_loss.detach().cpu(), std_loss.detach().cpu(), mse_loss.detach().cpu(), loss_d, loss_d_real, loss_d_fake))


    def sample(self, n):
#         self._generator.eval()
        steps = (n+self._batch_size-1) // self._batch_size
        data = np.zeros((n, self._transformer.output_dimensions), dtype='float32')
        cross_entropy = {}
#         for i in tqdm(range(steps), desc='Sampling...'):
        for i in range(steps):
            if i == 0:
                condvec = self._transformer.get_bayes_convec(self._batch_size, use_buff=False)
            else:
                condvec = self._transformer.get_bayes_convec(self._batch_size, use_buff=True)
                
            if condvec.shape[1] == data.shape[1]: # 没有连续列
                if (i+1)*self._batch_size <= n:
                    data[i*self._batch_size:(i+1)*self._batch_size] = condvec
                else:
                    data[i*self._batch_size:n] = condvec[:n-(i*self._batch_size)]
                continue
                
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            c1 = torch.from_numpy(condvec).to(self._device)
            fakez = torch.cat([fakez, c1], dim=1)

            fakeact = self._generator(fakez)

            torch_cat = []
            last_pos = 0
            for _, pos in enumerate(self._continuous_pos_in_cond):
                if pos != 0:
                    if pos-last_pos > 1:
                        torch_cat.append(c1[:, last_pos:pos])
                    else:
                        torch_cat.append(c1[:, last_pos:pos].reshape((-1, 1)))
                torch_cat.append(fakeact[:, _].reshape((-1, 1)))
                last_pos = pos
            if last_pos < c1.shape[1]:
                if condvec.shape[1]-last_pos > 1:
                    torch_cat.append(c1[:, last_pos:])
                else:
                    torch_cat.append(c1[:, last_pos:].reshape((-1, 1)))
            fakeact = torch.cat(torch_cat, dim=1)
            
            if (i+1)*self._batch_size <= n:
                data[i*self._batch_size:(i+1)*self._batch_size] = fakeact.detach().cpu().numpy().astype('float32')
            else:
                data[i*self._batch_size:n] = fakeact.detach().cpu().numpy().astype('float32')[:n-(i*self._batch_size)]
        
        buff_path = os.path.join(os.getcwd(), 'temp', f'sample_info_{self._transformer._random_seed}.pkl')
        if os.path.exists(buff_path):
            os.system(f'rm -rf {buff_path}')

        return self._transformer.inverse_transform(data)#, cross_entropy

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)