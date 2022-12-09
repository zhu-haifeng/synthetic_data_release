from collections import namedtuple

import os, time, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.BTCGAN.RDT import OneHotEncodingTransformer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
from model.BTCGAN import bayes_structure_learning
import subprocess
root_dir = os.getcwd()
temp_dir = os.path.join(root_dir, 'temp')

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo", ["column_name", "column_type",
                            "transform", "transform_aux",
                            "output_info", "output_dimensions"])


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=20.0, use_bayes=False, gm_epochs=5, use_gm=True, random_seed=None):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = 1.0 / max_clusters / weight_threshold
        self._use_bayes = use_bayes
        self._col_start_dim = None
        self._discrete_label = None
        self._child_parents = None
        self.condition_prob = None
        self._child_order = None
        self.cond_pos = None
        self.gen_cond_pos = None
        self._col_unique_values = None
        self._bayes_df = None
        self._use_gm = use_gm
        self._random_seed = np.random.randint(100000) if random_seed is None else random_seed
        self._gm_epochs = gm_epochs
        
    def _fit_continuous_quantile(self, column_name, raw_column_data):
        kbd = KBinsDiscretizer(
            n_bins = self._max_clusters,
            encode = 'ordinal',
            strategy = 'quantile',
        )
        kbd.fit(raw_column_data.reshape(-1, 1))
        
        unique_data = np.unique(raw_column_data)
        split = [(i-1, border) for i, border in enumerate(kbd.bin_edges_[0])]
        split.pop(0)

        num_components = len(split)

        temp = [[] for i in range(num_components)]
        kbd_info = []
        left_border = raw_column_data.min() - 1
        for index, right_border in split:
            cond_row = (raw_column_data >= left_border) & (raw_column_data < right_border)
            if not cond_row.any(): continue
            temp[index].append(raw_column_data[cond_row])
            left_border = right_border
        for values in temp:
            this_value = np.concatenate(values)
#                 z_score = (this_value - means[index]) / stds[index]
            z_score = this_value
            kbd_info.append((z_score.min().item(), z_score.max().item(), z_score.mean().item(), z_score.var().item()))

        return ColumnTransformInfo(
            column_name=column_name, column_type="continuous", transform=kbd,
            transform_aux=(split, kbd_info),
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)
        

    def _fit_continuous(self, column_name, raw_column_data):
        """Train Bayesian GMM for continuous column."""
        gm = BayesianGaussianMixture(
            n_components = self._max_clusters,
            weight_concentration_prior_type = 'dirichlet_process',
            weight_concentration_prior = 0.001,
            n_init = 1 if not self._use_bayes else self._gm_epochs,
        )

        gm.fit(raw_column_data.reshape(-1, 1))
        
        if self._use_bayes:
            unique_data = np.unique(raw_column_data)
            gm_index = np.argmax(gm.predict_proba(unique_data.reshape((-1, 1))), axis=1)
            split = []
            left_index = gm_index[0]
            real_index = 0
            for i in range(1, unique_data.shape[0]):
                if gm_index[i] != left_index:
                    split.append((real_index, (unique_data[i-1]+unique_data[i])/2))
                    real_index += 1
                    left_index = gm_index[i]
            split.append([real_index, unique_data[-1]+(unique_data[-2]+unique_data[-1])/2])
            
            num_components = len(split)
            
            temp = [[] for i in range(num_components)]
            gm_info = []
            left_border = raw_column_data.min() - 1
            for index, right_border in split:
                cond_row = (raw_column_data >= left_border) & (raw_column_data < right_border)
                if not cond_row.any(): continue
                temp[index].append(raw_column_data[cond_row])
                left_border = right_border
            for values in temp:
                this_value = np.concatenate(values)
#                 z_score = (this_value - means[index]) / stds[index]
                z_score = this_value
                gm_info.append((z_score.min().item(), z_score.max().item(), z_score.mean().item(), z_score.var().item()))
        
            return ColumnTransformInfo(
                column_name=column_name, column_type="continuous", transform=gm,
                transform_aux=(split, gm_info),
                output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
                output_dimensions=1 + num_components)
        
        else:
            valid_component_indicator = gm.weights_ > self._weight_threshold
            num_components = valid_component_indicator.sum()

            return ColumnTransformInfo(
                column_name=column_name, column_type="continuous", transform=gm,
                transform_aux=valid_component_indicator,
                output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
                output_dimensions=1 + num_components)

    def _fit_discrete(self, column_name, raw_column_data):
        """Fit one hot encoder for discrete column."""
        ohe = OneHotEncodingTransformer()
        ohe.fit(raw_column_data)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type="discrete", transform=ohe,
            transform_aux=None,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)
            
    def fit(self, raw_data, discrete_columns, bayes_edges=None, no_parents=None, structure_algorithm='PC', bayes_train_samples=100000):
        """Fit GMM for continuous columns and One hot encoder for discrete columns.

        This step also counts the #columns in matrix data, and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        discrete_columns = set(discrete_columns)

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            raw_data = pd.DataFrame(raw_data)
        else:
            self.dataframe = True

        self._column_raw_dtypes = raw_data.infer_objects().dtypes

        self._column_transform_info_list = []
        self._column_index = {col: i+1 for i, col in enumerate(raw_data.columns)}
        
        cur_dim = 0
        self._col_start_dim = {}
        self.cond_pos = []
        self.gen_cond_pos = []
#         for column_name in tqdm(raw_data.columns, desc='Fit data...'):
        for column_name in raw_data.columns:
            raw_column_data = raw_data[column_name].values
            if column_name in discrete_columns: # 离散列
                column_transform_info = self._fit_discrete(
                    column_name, raw_column_data)
                self._col_start_dim[column_name] = cur_dim
                self.cond_pos.extend([True, ] * column_transform_info.output_dimensions)
                self.gen_cond_pos.extend([True, ] * column_transform_info.output_dimensions)
            else:
                if self._use_gm:
                    column_transform_info = self._fit_continuous(
                        column_name, raw_column_data)
                else:
                    column_transform_info = self._fit_continuous_quantile(
                        column_name, raw_column_data)
                self._col_start_dim[column_name] = cur_dim + 1
                self.cond_pos.append(False)
                self.cond_pos.extend([True, ] * (column_transform_info.output_dimensions-1))
                self.gen_cond_pos.extend([False, True, True])

            cur_dim += column_transform_info.output_dimensions
            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)
            
        self.cond_pos = np.array(self.cond_pos, dtype=np.bool)
        self.gen_cond_pos = np.array(self.gen_cond_pos, dtype=np.bool)
            
        if self._use_bayes: # 如果使用贝叶斯网络
            eps = 0.0001 # 控制高斯分量边界精度
            self._discrete_label = {}
            random_state = np.random.RandomState(self._random_seed)
            select_row = np.arange(len(raw_data))
            if bayes_train_samples < len(raw_data):
                np.random.shuffle(select_row)
                select_row = select_row[:bayes_train_samples]
            bayes_value = np.zeros((min(bayes_train_samples, len(raw_data)), len(raw_data.columns)), dtype=np.int32)
            for col_index, column_transform_info in enumerate(self._column_transform_info_list):
                column_name, column_type = column_transform_info.column_name, column_transform_info.column_type
                if column_type == "continuous":
#                     valid_component_indicator = column_transform_info.transform_aux
                    column_value = raw_data[column_name].values
                    min_value, max_value = column_value.min(), column_value.max()
#                     column_value_unique = np.unique(column_value)
#                     means = gm.means_[valid_component_indicator].ravel()
#                     stds = np.sqrt(gm.covariances_[valid_component_indicator]).ravel()
                    
#                     column_gm_prob = gm.predict_proba(column_value.reshape((-1, 1)))[:, valid_component_indicator]
#                     column_gm_index = np.zeros(column_gm_prob.shape[0], dtype='int32')
#                     for i in range(column_gm_prob.shape[0]):
#                         p = column_gm_prob[i] + (1e-6)
#                         p /= p.sum()
#                         column_gm_index[i] = random_state.choice(column_gm_prob.shape[1], p=p)
                    
#                     column_gm_index = np.argmax(column_gm_index, axis=1)
#                     left_border, left_index = column_value_unique[0], column_gm_index[0]
#                     for i in range(1, column_value_unique.shape[0]):
#                         if column_gm_index[i] != left_index:
#                             split.append((left_index, (column_value_unique[i-1]+column_value_unique[i])/2))
#                             left_border, left_index = column_value_unique[i], column_gm_index[i]
#                     split.append((left_index, column_value_unique[-1]+(column_value_unique[-2]+column_value_unique[-1])/2))
                    
                    split, gm_info = column_transform_info.transform_aux
                    left_border = min_value - 1
                    for index, right_border in split:
                        cond_row = (column_value >= left_border) & (column_value < right_border)
                        this_value = column_value[cond_row]
                        
                        cond_row = cond_row[select_row] # 只要部分数据求贝叶斯
                        bayes_value[cond_row, col_index] = index
                        left_border = right_border
                else:
                    ohe = column_transform_info.transform
                    discrete_values = list(raw_data[column_name].unique())
                    labels = np.argmax(ohe.transform(discrete_values), axis=1)
                    value_to_index = {value: label for value, label in zip(discrete_values, labels)}
                    
                    column_value = raw_data[column_name].values#[select_row] # 只要部分数据求贝叶斯
                    column_value_bayes = column_value[select_row]
                    for value, index in value_to_index.items():
                        bayes_value[(column_value_bayes == value), col_index] = index
                    self._discrete_label[column_name] = value_to_index
            
            self._bayes_df = pd.DataFrame()
            
            self._col_unique_values = {}
            for col_index, column_transform_info in enumerate(self._column_transform_info_list):
                column_name = column_transform_info.column_name
                col_values = bayes_value[:, col_index]
                self._bayes_df[column_name] = col_values
                self._col_unique_values[column_name] = np.unique(col_values).tolist()
            
            if structure_algorithm == 'PC':
                algorithm = bayes_structure_learning.PC
            elif structure_algorithm == 'prior_bayes':
                algorithm = bayes_structure_learning.prior_bayes
            elif structure_algorithm == 'ChowLiu':
                algorithm = bayes_structure_learning.ChowLiu
            else:
                algorithm = bayes_structure_learning.PC
            start = time.perf_counter()
#             self._child_parents = algorithm(bayes_df.iloc[select_idx], prior_edges=bayes_edges, no_parents=no_parents)
#             print(self._bayes_df.info())
            self._child_parents = algorithm(self._bayes_df, prior_edges=bayes_edges, no_parents=no_parents)
            end = time.perf_counter()
#             print('Learning structure use %.4fs'%(end-start))
#             print(self._child_parents)
            self._set_bayes_condition_prob(raw_data)
            
            return self._bayes_df
            
            
    def _makeup_values(self, nodes=None):
        if nodes is None:
            nodes = list(self._col_unique_values.keys())
        status_index = [0, ] * len(nodes)
        is_finish = False
        while not is_finish:
            yield {
                node: self._col_unique_values[node][status_index[i]] for i, node in enumerate(nodes)
            }

            is_change = False
            for node_i in range(len(nodes)):
                node = nodes[node_i]
                if status_index[node_i] == len(self._col_unique_values[node])-1:
                    if node_i == len(status_index)-1:
                        is_finish = True
                        break
                    else:
                        status_index[node_i] = 0
                else:
                    status_index[node_i] += 1
                    break        
    
    
    def _set_bayes_condition_prob(self, raw_data):
        self.condition_prob = {}
        
#         for child, parents in tqdm(self._child_parents.items(), desc='Learning condition prob...'):
        for child, parents in self._child_parents.items():
#             bayes_title = 'bayes: '+','.join(parents)+' -> '+child #
#             raw_data[bayes_title] = 0 #
            prob_table = pd.DataFrame()
            prob_table_dict = {child: [], 'probability': []}
            for node in parents:
                prob_table_dict[node] = []
                
            for condition in self._makeup_values(nodes=parents):
                condition_df = self._bayes_df.query(
                    ' and '.join(f'(`{node}` == {value})' for node, value in condition.items())
                )
                    
                for parent, parent_value in condition.items():
                    prob_table_dict[parent].extend([parent_value, ] * len(self._col_unique_values[child]))
                for child_value in self._col_unique_values[child]:
                    if len(condition_df) == 0:
                        child_condition_prob = 1e-6
                    else:
                        child_condition_prob = len(condition_df[condition_df[child] == child_value]) / len(condition_df)
                        if child_condition_prob == 0:
                            child_condition_prob = 1e-6
#                     raw_data.iloc[condition_df[condition_df[child] == child_value].index, -1] = child_condition_prob #
                    prob_table_dict[child].append(child_value)
                    prob_table_dict['probability'].append(child_condition_prob)
        
            for key, value in prob_table_dict.items():
                if key == 'probability':
                    p = np.array(value, dtype='float64')
                    child_values_num = len(self._col_unique_values[child])
                    for j in range(0, len(prob_table_dict['probability'])//child_values_num):
                        temp = p[j*child_values_num:(j+1)*child_values_num]
                        p[j*child_values_num:(j+1)*child_values_num] = temp / temp.sum()
                    prob_table[key] = p
                else:
                    prob_table[key] = value
                    prob_table[key] = prob_table[key].astype('category')
            self.condition_prob[child] = prob_table
            
        no_parent_nodes = set(
            column_info.column_name
            for column_info in self._column_transform_info_list if column_info.column_name not in self._child_parents.keys()
        )
        for node in no_parent_nodes:
            node_df = pd.DataFrame()
            value_counts = self._bayes_df[node].value_counts()
            node_df[node], p = value_counts.index, (value_counts.values / len(self._bayes_df)).astype('float64')
            node_df[node] = node_df[node].astype('category')
            p /= p.sum()
            node_df['probability'] = p
            self.condition_prob[node] = node_df
            
    def _transform_continuous(self, column_transform_info, raw_column_data):
        gm = column_transform_info.transform

        if self._use_bayes: # 把软边界换成硬边界
            random_state = np.random.RandomState(self._random_seed)
#             normalized_values = (raw_column_data - means) / (stds)
            normalized_values = raw_column_data.ravel()
            data_len = len(raw_column_data)
            split, gm_info_list = column_transform_info.transform_aux
            num_components = len(split)
            
            selected_component = np.zeros(data_len, dtype='int32')
            selected_normalized_value = np.zeros(data_len, dtype='float32')
            left_border = normalized_values.min() - 1
            for index, right_border in split:
                this_gm_index = (normalized_values >= left_border) & (normalized_values < right_border)
                if not this_gm_index.any(): # 如果这个区间里没数据
                    left_border = right_border
                    continue
                selected_component[this_gm_index] = index
                gm_info = gm_info_list[index]
                if gm_info[0] == gm_info[1]: # 只有一个元素
                    this_gm_value = 0
                else:
                    this_gm_value = normalized_values[this_gm_index]
                    scaler = max(abs(gm_info[0]-gm_info[2]), abs(gm_info[1]-gm_info[2])) / 0.9999
                    this_gm_value = (this_gm_value-gm_info[2]) / scaler
                selected_normalized_value[this_gm_index] = this_gm_value
                left_border = right_border
            selected_normalized_value = selected_normalized_value.reshape((-1, 1))

#             selected_normalized_value = normalized_values[np.arange(data_len), selected_component].reshape((-1, 1))
            selected_component_onehot = np.zeros((data_len, num_components), dtype='int32')
        else:
            valid_component_indicator = column_transform_info.transform_aux
            num_components = valid_component_indicator.sum()

            means = gm.means_[valid_component_indicator].reshape((1, num_components))
            stds = np.sqrt(gm.covariances_[valid_component_indicator]).reshape((1, num_components))
            
            normalized_values = ((raw_column_data - means) / (4 * stds))
            component_probs = gm.predict_proba(raw_column_data)[:, valid_component_indicator]
            selected_component = np.zeros(len(raw_column_data), dtype='int')
            for i in range(len(raw_column_data)):
                component_porb_t = component_probs[i] + 1e-6
                component_porb_t = component_porb_t / component_porb_t.sum()
                selected_component[i] = np.random.choice(
                    np.arange(num_components), p=component_porb_t)
            selected_normalized_value = normalized_values[
                np.arange(len(raw_column_data)), selected_component].reshape([-1, 1])
            selected_normalized_value = np.clip(selected_normalized_value, -0.99, 0.99)
            selected_component_onehot = np.zeros_like(component_probs)
            
        selected_component_onehot[np.arange(len(raw_column_data)), selected_component] = 1
        return [selected_normalized_value, selected_component_onehot]

    def _transform_discrete(self, column_transform_info, raw_column_data):
        ohe = column_transform_info.transform
        return [ohe.transform(raw_column_data)]

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)
            
        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_data = raw_data[[column_transform_info.column_name]].values
            if column_transform_info.column_type == "continuous":
                column_data_list += self._transform_continuous(
                    column_transform_info, column_data)
            else:
                assert column_transform_info.column_type == "discrete"
                column_data_list += self._transform_discrete(
                    column_transform_info, column_data)

        return np.concatenate(column_data_list, axis=1).astype(np.float32)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform

        selected_normalized_value = column_data[:, 0]
        selected_component_probs = column_data[:, 1:]

        if sigmas is not None:
            sig = sigmas[st]
            selected_normalized_value = np.random.normal(selected_normalized_value, sig)
        
        if self._use_bayes:
            num_components = selected_component_probs.shape[1]
            
            split, gm_info_list = column_transform_info.transform_aux
            selected_component = np.argmax(selected_component_probs, axis=1)
            column = np.zeros(selected_normalized_value.shape[0], dtype='float32')
            for index, gm_info in enumerate(gm_info_list):
                this_gm_index = (selected_component == index)
                if not this_gm_index.any(): continue # 如果这个区间没有数据
                    
                if gm_info[0] == gm_info[1]: # 只有一个元素
                    this_gm_value = gm_info[1]
                else:
                    this_gm_value = selected_normalized_value[this_gm_index]
                    scaler = max(abs(gm_info[0]-gm_info[2]), abs(gm_info[1]-gm_info[2])) / 0.9999
                    this_gm_value = this_gm_value * scaler + gm_info[2]
#                 this_gm_value = this_gm_value * stds[index] + means[index]
                column[this_gm_index] = this_gm_value
            if str(self._column_raw_dtypes[column_transform_info.column_name])[:3] == 'int':
                column = np.round(column)
        else:
            valid_component_indicator = column_transform_info.transform_aux
            selected_normalized_value = np.clip(selected_normalized_value, -1, 1)
            component_probs = np.ones((len(column_data), self._max_clusters)) * -100
            component_probs[:, valid_component_indicator] = selected_component_probs

            means = gm.means_.reshape([-1])
            stds = np.sqrt(gm.covariances_).reshape([-1])
            selected_component = np.argmax(component_probs, axis=1)

            std_t = stds[selected_component]
            mean_t = means[selected_component]
            column = selected_normalized_value * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        return ohe.reverse_transform(column_data)

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]

            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            else:
                assert column_transform_info.column_type == 'discrete'
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.values

        return recovered_data

    def data2genconv(self, data):
        self.gen_conv = 0
        st = 0
        new_conv = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions

            if column_transform_info.column_type == 'continuous':
                split, gm_info_list = column_transform_info.transform_aux
                cur_conv = data[:, st+1:st+dim]
                selected_component = np.argmax(cur_conv, axis=1)
                trans_conv = np.zeros((selected_component.shape[0], 3), dtype='float32')
                for i, gm_info in enumerate(gm_info_list):
                    select_rows = (selected_component == i)
                    if not select_rows.any(): continue
                    trans_conv[select_rows, 0] = data[selected_component, st]
                    trans_conv[select_rows, 1] = gm_info[2]
                    trans_conv[select_rows, 2] = gm_info[3]
                new_conv.append(trans_conv)
                self.gen_conv += 2
            else:
                cur_conv = data[:, st:st + dim]
                new_conv.append(cur_conv)
                self.gen_conv += dim

            st += dim
        new_conv = np.concatenate(new_conv, axis=1).astype(np.float32)
        return new_conv
    
    def conv2genconv(self, conv):
        st = 0
        new_conv = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions

            if column_transform_info.column_type == 'continuous':
                split, gm_info_list = column_transform_info.transform_aux
                cur_conv = conv[:, st:st+dim-1]
                selected_component = np.argmax(cur_conv, axis=1)
                trans_conv = np.zeros((selected_component.shape[0], 2), dtype='float32')
                for i, gm_info in enumerate(gm_info_list):
                    select_rows = (selected_component == i)
                    trans_conv[select_rows, 0] = gm_info[2]
                    trans_conv[select_rows, 1] = gm_info[3]
                new_conv.append(trans_conv)
            else:
                cur_conv = conv[:, st:st + dim]
                new_conv.append(cur_conv)

            st += dim
        new_conv = np.concatenate(new_conv, axis=1).astype(np.float32)
        return new_conv
    
    def get_bayes_convec(self, batch_size, use_buff=False):
        if not use_buff:
            convec_dim = 0
            child_parentsset_dict = {child: set(parents) for child, parents in self._child_parents.items()}
            self._child_order = []
            no_parent_nodes = set()
            col_start_dim_without_continuous = {}
            for column_info in self._column_transform_info_list:
                column_name = column_info.column_name
                col_start_dim_without_continuous[column_name] = convec_dim
                if len(column_info.output_info) == 1: # 离散列
                    convec_dim += column_info.output_dimensions
                else: # 连序列
                    convec_dim += column_info.output_dimensions-1
                if column_name not in self._child_parents.keys():
                    no_parent_nodes.add(column_name)

            if len(self._child_order) == 0:
                have_visited = no_parent_nodes.copy()
                while len(have_visited) < len(self._column_transform_info_list):
                    for child, parents in child_parentsset_dict.items():
                        if child in have_visited:
                            continue
                        all_in = True
                        for parent in parents:
                            if parent not in have_visited:
                                all_in = False
                                break
                        if all_in:
                            have_visited.add(child)
                            self._child_order.append(child)

            with open(os.path.join(temp_dir, f'sample_info_{self._random_seed}.pkl'), 'wb') as file:
                pickle.dump((batch_size, convec_dim, no_parent_nodes, col_start_dim_without_continuous, child_parentsset_dict, self.condition_prob, self._child_order), file)

        subprocess.run((
            'python3',
            os.path.join(root_dir, 'model', 'BTCGAN', 'generate_convec.py'),
            os.path.join(temp_dir, f'sample_info_{self._random_seed}.pkl'),
            str(self._random_seed),
        ))
        with open(os.path.join(temp_dir, f'convec_{self._random_seed}.pkl'), 'rb') as file:
            convec = pickle.load(file)
        if os.path.exists(os.path.join(temp_dir, f'convec_{self._random_seed}.pkl')):
            os.system('rm -rf %s' % os.path.join(temp_dir, f'convec_{self._random_seed}.pkl'))
        return convec
    
# def _sample_bayes_condvec(index, generate_num, no_parent_nodes, col_start_dim_without_continuous, child_parentsset_dict, condition_prob, child_order, random_state):
#     result = []
#     for i in range(generate_num):
#         condition = {}
#         pos = []
#         for node in no_parent_nodes:
#             df = condition_prob[node]
#             value = random_state.choice(df[node].values, p=df['probability'].values)
#             condition[node] = value
#             pos.append(col_start_dim_without_continuous[node]+int(value))

#         for node in child_order:
#             df = condition_prob[node]
#             condition_df = df.query(
#                 ' and '.join('(`%s` == %s)' % (parent, value)
#                              for parent, value in condition.items() if parent in child_parentsset_dict[node])
#             )

#             value = random_state.choice(condition_df[node].values, p=condition_df['probability'].values)
#             condition[node] = value
#             pos.append(col_start_dim_without_continuous[node]+int(value))
#         result.append(np.array(pos))
#     return index, result
