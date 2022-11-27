import warnings

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
    def __init__(self, data_dim, discriminator_dim, device, pac=10):
        super(Discriminator, self).__init__()
        self.pac = pac
        self.data_pacdim = data_dim * pac
        
        self.seqs = []
        dim = self.data_pacdim
        for item in list(discriminator_dim):
#             seq += [Linear(dim, item), BatchNorm1d(item), LeakyReLU(0.2), Dropout(0.5)]
            self.seqs.append(Sequential(
                Linear(dim, item), BatchNorm1d(item), LeakyReLU(0.2), Dropout(0.5),
            ).to(device))
            dim = item + self.data_pacdim
        
        self.act = Sequential(Linear(dim, 1), Sigmoid())
#         self.seq = Sequential(*seq)
        
    def calc_gradient_penalty(self, real_conv, fake_data, device='cpu', pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, data):
        assert data.shape[0] % self.pac == 0
        data = data.view((-1, self.data_pacdim))
        
        output = self.seqs[0](data)
        output = torch.cat([output, data], dim=1)
        for i in range(1, len(self.seqs)):
            output = self.seqs[i](output)
            output = torch.cat([output, data], dim=1)
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
    def __init__(self, embedding_dim, generator_dim, data_dim, one_hot_in_cond):
        super(Generator, self).__init__()
        self._one_hot_in_cond = one_hot_in_cond
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
#         self.act_one_hot = Softmax()
#         self.act_extra = Tanh()

    def forward(self, input):
        data = self.seq(input)
        return data
#         outputs = []
#         for pos_start, pos_end in self._one_hot_in_cond:
#             outputs.append(self.act_one_hot(data[:, pos_start:pos_end]))
#         extra_start_pos = self._one_hot_in_cond[-1][1]
#         if extra_start_pos < data.shape[1]:
#             outputs.append(self.act_extra(data[:, extra_start_pos:]))
#         return torch.cat(outputs, dim=1)


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
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
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
        self._max_clusters = max_clusters
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
        
    def _softmax(data, one_hot_in_cond, tau=0.2, hard=False, eps=1e-10, dim=-1):
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = functional.gumbel_softmax(data[:, st:ed], tau=tau, hard=hard, eps=eps, dim=-1)
                    data_t.append(transformed)
                    st = ed
                else:
                    assert 0

        return torch.cat(data_t, dim=1)
        
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

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer(use_bayes=True, max_clusters=self._max_clusters, gm_epochs=gm_epochs)
        self._transformer.fit(train_data, discrete_columns, bayes_edges=bayes_edges, no_parents=label, structure_algorithm='prior_bayes', bayes_train_samples=bayes_train_samples)
        
        train_data = self._transformer.transform(train_data)
        data_dim = self._transformer.output_dimensions
        
        one_hot_in_cond = []
        cur_pos = 0
        for col_info in self._transformer.output_info_list:
            if len(col_info) == 1: # 离散列
                cur_dim = col_info[0].dim
            else:
                cur_dim = col_info[1].dim
            one_hot_in_cond.append((cur_pos, cur_pos+cur_dim))
            cur_pos += cur_dim

        self._generator = Generator(
            self._embedding_dim,
            self._generator_dim,
            data_dim,
            one_hot_in_cond,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim,
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
        lr = self._discriminator_lr
        for i in range(epochs):
            idx = np.arange(len(train_data))
            np.random.shuffle(idx)
            gen_idx = np.arange(len(train_data))
            np.random.shuffle(gen_idx)
            for id_ in range(steps_per_epoch):
                if train_discriminator: # 原来的鉴别器
                    fakez = torch.normal(mean=mean, std=std)

                    real = train_data[idx[id_*self._batch_size:(id_+1)*self._batch_size]]
                    real = torch.from_numpy(real).to(self._device)
                    fakeact = self._generator(fakez)
                    
                    y_fake = discriminator(fakeact)
                    y_real = discriminator(real)

                    loss_d = -(torch.mean(torch.log(y_real))) - (torch.mean(torch.log(1-y_fake)))
                    optimizerD.zero_grad()
                    loss_d.backward()
                    optimizerD.step()
                    yy_fake = y_fake

                real_data = train_data[gen_idx[id_*self._batch_size:(id_+1)*self._batch_size]]
                real_data = torch.from_numpy(real_data).to(self._device)
                fakez = torch.normal(mean=mean, std=std)
                fakeact = self._generator(fakez)
                y_fake = discriminator(fakeact)
                
                mean_loss = ((fakeact.mean(dim=0) - real.mean(dim=0))**2).sum() / data_dim
                std_loss = ((fakeact.std(dim=0) - real.std(dim=0))**2).sum() / data_dim
                mse_loss = functional.mse_loss(fakeact, real)
                loss_g = torch.mean(torch.log(1-y_fake)) + self._mean_weight*mean_loss + self._std_weight*std_loss + self._mse_weight*mse_loss
                
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            
            loss_d_real = torch.mean(y_real).detach().cpu()
            loss_d_fake = torch.mean(yy_fake).detach().cpu()
            if loss_d_real - loss_d_fake > 0.1 and lr > 1e-12:
                lr /= 5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = lr
                print('Set discriminator lr: %f' % lr)
            if self._verbose and (i % 50 == 0 or i == epochs-1):
                loss_d = loss_d.detach().cpu()
                print('Epoch %d: LossG(%.4f -fake+l2): fake: %.4f, mean: %.4f, std: %.4f, mse:%.4f; LossD(-(real-fake) %.4f): real: %.4f, fake: %.4f' % 
                     (i+1, loss_g.detach().cpu(), torch.mean(y_fake).detach().cpu(), mean_loss.detach().cpu(), std_loss.detach().cpu(), mse_loss.detach().cpu(), loss_d, loss_d_real, loss_d_fake))


    def sample(self, n):
#         self._generator.eval()
        steps = (n+self._batch_size-1) // self._batch_size
        data = np.zeros((n, self._transformer.output_dimensions), dtype='float32')
        for i in tqdm(range(steps), desc='Sampling...'):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)
            fakeact = self._generator(fakez)

            if (i+1)*self._batch_size <= n:
                data[i*self._batch_size:(i+1)*self._batch_size] = fakeact.detach().cpu().numpy().astype('float32')
            else:
                data[i*self._batch_size:n] = fakeact.detach().cpu().numpy().astype('float32')[:n-(i*self._batch_size)]

        return self._transformer.inverse_transform(data)#, cross_entropy

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)