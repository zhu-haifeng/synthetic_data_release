import numpy as np


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data, output_info, log_frequency, use_bayes=False):
        self._data = data
        self._use_bayes = use_bayes

        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == "softmax")

        if self._use_bayes:
            n_discrete_columns = len(output_info)
        else:
            n_discrete_columns = sum([1 for column_info in output_info if is_discrete_column(column_info)])
        

        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype="int32")

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        # _rid_by_cat_cols[a][b]存的是第a个离散列的取值为b的行数有多少 改成所有类型
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info): # 离散列
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else: # 连序列
                if self._use_bayes:
                    st += 1
                    span_info = column_info[1]
                    ed = st + span_info.dim
                    rid_by_cat = []
                    for j in range(span_info.dim):
                        rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                    self._rid_by_cat_cols.append(rid_by_cat)
                    st = ed
                else:
                    st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        if self._use_bayes:
            max_category = max((
                column_info[0].dim if is_discrete_column(column_info) else column_info[1].dim
                for column_info in output_info
            ), default=0)
        else:
            max_category = max((
                column_info[0].dim for column_info in output_info
                if is_discrete_column(column_info)
            ), default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns # 离散列的个数
    
        if self._use_bayes:
            self._n_categories = sum([
                column_info[0].dim if is_discrete_column(column_info) else column_info[1].dim
                for column_info in output_info
            ]) # 所有离散变量和连续变量的个数之和
        else:
            self._n_categories = sum(
                [column_info[0].dim for column_info in output_info
                 if is_discrete_column(column_info)]) # 所有离散变量的个数之和

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
#                 self._discrete_column_category_origin_prob[current_id, :span_info.dim] = (category_freq / np.sum(category_freq))
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = (category_prob)
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                if self._use_bayes:
                    span_info = column_info[1]
                    st += column_info[0].dim
                    ed = st + span_info.dim
                    category_freq = np.sum(data[:, st:ed], axis=0)
                    if log_frequency:
                        category_freq = np.log(category_freq + 1)
                    category_prob = category_freq / np.sum(category_freq)
                    self._discrete_column_category_prob[current_id, :span_info.dim] = (category_prob)
                    self._discrete_column_cond_st[current_id] = current_cond_st
                    self._discrete_column_n_category[current_id] = span_info.dim
                    current_cond_st += span_info.dim
                    current_id += 1
                    st = ed
                else:
                    st += sum([span_info.dim for span_info in column_info])

    def _random_choice_prob_index(self, discrete_column_id):
#         if relation:
#             probs = self._discrete_column_category_origin_prob[discrete_column_id]
#         else:
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype='float32') # 所有离散变量的个数
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32') # 离散列的个数
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col # 条件向量，选择的离散列onehot，每一行选择的label编码，选择的离散值label编码

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None
        
        cond = np.zeros((batch, self._n_categories), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    def sample_data(self, n, col=None, opt=None):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return self._data[idx]

    def dim_cond_vec(self):
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id = self._discrete_column_matrix_st[condition_info["discrete_column_id"]] + condition_info["value_id"]
        vec[:, id] = 1
        return vec
