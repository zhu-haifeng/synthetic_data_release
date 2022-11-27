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
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, cuda_index=0,
                 max_clusters=10, bayes_edges=None, label=None, mean_weight=0.3, std_weight=0.3):

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
        self._bayes_edges = bayes_edges
        self._mean_weight = mean_weight
        self._std_weight = std_weight
        
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

        self._transformer = DataTransformer(use_bayes=True, max_clusters=self._max_clusters, gm_epochs=gm_epochs)
        self._transformer.fit(train_data, discrete_columns, bayes_edges=bayes_edges, no_parents=label, structure_algorithm=structure_algorithm, bayes_train_samples=bayes_train_samples)
        
        data_dim = self._transformer.output_dimensions
        conv_dim = self._transformer.cond_pos.sum().item()
        self._extra_dim = data_dim - conv_dim
        if self._extra_dim == 0:
            return
        self._continuous_pos_in_cond = []
        self._continuous_column_onehot_dim = []
        self._continuous_column_info = []
        cur_pos = 0
        for col_info in self._transformer.output_info_list:
            if len(col_info) == 1: # 离散列
                cur_pos += col_info[0].dim
            else:
                self._continuous_pos_in_cond.append(cur_pos)
                self._continuous_column_onehot_dim.append(col_info[1].dim)
                cur_pos += col_info[1].dim
                
        for column_transform_info in self._transformer._column_transform_info_list:
            if column_transform_info.column_type == 'continuous':
                self._continuous_column_info.append(column_transform_info)

    def sample(self, n):
#         self._generator.eval()
        steps = (n+self._batch_size-1) // self._batch_size
        data = np.zeros((n, self._transformer.output_dimensions), dtype='float32')
        cross_entropy = {}
        for i in tqdm(range(steps), desc='Sampling...'):
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
            
            numpy_cat = []
            last_pos = 0
            for _, pos in enumerate(self._continuous_pos_in_cond):
                if pos != 0:
                    if pos-last_pos > 1:
                        numpy_cat.append(condvec[:, last_pos:pos])
                    else:
                        numpy_cat.append(condvec[:, last_pos:pos].reshape((-1, 1)))
                
                dim, column_info = self._continuous_column_onehot_dim[_], self._continuous_column_info[_]
                split, gm_info_list = column_info.transform_aux
                gm = column_info.transform
                
                selected_component = np.argmax(condvec[:, pos:pos+dim], axis=1)
                column = np.zeros(self._batch_size, dtype='float32')
                for index, gm_info in enumerate(gm_info_list):
                    this_gm_index = (selected_component == index)
                    if not this_gm_index.any(): continue # 如果这个区间没有数据
                    elif gm_info[0] == gm_info[1]: # 只有一个元素
                        this_gm_value = 0
                        # this_gm_value = gm_info[1]
                    else:
                        this_gm_value = np.random.normal(size=this_gm_index.sum())
#                         gm_index = np.argmax(gm.predict_proba(np.array([[gm_info[2],]])), axis=1).item()
#                         mean, var = gm.means_[gm_index][0], gm.covariances_[gm_index][0][0]
#                         need_dim = this_gm_index.sum()
#                         this_gm_value = np.random.normal(mean, var**0.5, self._batch_size)
#                         this_gm_value = this_gm_value[(this_gm_value >= gm_info[0]) & (this_gm_value < gm_info[1])]
#                         while this_gm_value.shape[0] < need_dim:
#                             temp = np.random.normal(mean, var**0.5, self._batch_size)
#                             temp = temp[(temp >= gm_info[0]) & (temp < gm_info[1])]
#                             this_gm_value = np.concatenate([this_gm_value, temp])
#                         this_gm_value = this_gm_value[:need_dim]
                        
#                         scaler = max(abs(gm_info[0]-gm_info[2]), abs(gm_info[1]-gm_info[2])) / 0.9999
#                         this_gm_value = (this_gm_value-gm_info[2]) / scaler
                    column[this_gm_index] = this_gm_value
                
                numpy_cat.append(column.reshape((-1, 1)))
                last_pos = pos
            if last_pos < condvec.shape[1]:
                if condvec.shape[1]-last_pos > 1:
                    numpy_cat.append(condvec[:, last_pos:])
                else:
                    numpy_cat.append(condvec[:, last_pos:].reshape((-1, 1)))
            fakeact = np.concatenate(numpy_cat, axis=1)
            
            if (i+1)*self._batch_size <= n:
                data[i*self._batch_size:(i+1)*self._batch_size] = fakeact.astype('float32')
            else:
                data[i*self._batch_size:n] = fakeact.astype('float32')[:n-(i*self._batch_size)]
        
        buff_path = os.path.join(os.getcwd(), 'temp', f'sample_info_{self._transformer._random_seed}.pkl')
        if os.path.exists(buff_path):
            os.system(f'rm -rf {buff_path}')

        return self._transformer.inverse_transform(data)#, cross_entropy

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)