import warnings
from model.BTCGAN.btcgan import BTCGAN as _btcgan

from utils.logging import LOGGER
from pandas import DataFrame

from utils.logging import LOGGER

from generative_models.generative_model import GenerativeModel

class BTCGAN(GenerativeModel):
    
    def __init__(self,metadata):
        self.synthesiser = _btcgan()

        self.datatype = DataFrame
        cols = metadata['columns']
        self.discrete_columns = [col['name'] for col in cols if col['type'] == 'Categorical' or col['type'] == 'Ordinal']
        self.multiprocess = False
        self.infer_ranges = True
        self.trained = False

        self.__name__ = 'BTCGAN'
    def fit(self, data):
        
        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') # 忽略所有警告
            self.synthesiser.fit(data, discrete_columns = self.discrete_columns)

        LOGGER.debug(f'Finished fitting')
        self.trained = True
        
    def generate_samples(self, nsamples):
        """Generate random samples from the fitted Gaussian distribution"""
        assert self.trained, "Model must first be fitted to some data."

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        synthetic_data = self.synthesiser.sample(nsamples)

        return synthetic_data
        