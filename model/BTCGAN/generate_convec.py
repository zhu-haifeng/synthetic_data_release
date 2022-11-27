import os, pickle, sys, random
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, as_completed

def sample_bayes_condvec(index, generate_num, no_parent_nodes, col_start_dim_without_continuous, child_parentsset_dict, condition_prob, child_order, random_state):
    result = []
    for i in range(generate_num):
        condition = {}
        pos = []
        for node in no_parent_nodes:
            df = condition_prob[node]
            value = random_state.choice(df[node].values, p=df['probability'].values)
            condition[node] = value
            pos.append(col_start_dim_without_continuous[node]+int(value))

        for node in child_order:
            df = condition_prob[node]
            condition_df = df.query(
                ' and '.join('(`%s` == %s)' % (parent, value)
                             for parent, value in condition.items() if parent in child_parentsset_dict[node])
            )

            value = random_state.choice(condition_df[node].values, p=condition_df['probability'].values)
            condition[node] = value
            pos.append(col_start_dim_without_continuous[node]+int(value))
        result.append(np.array(pos))
    return index, result

if __name__ == '__main__':
    info_data_path = sys.argv[1]
    file_seed = sys.argv[2]
    with open(info_data_path, 'rb') as file:
        batch_size, convec_dim, no_parent_nodes, col_start_dim_without_continuous, child_parentsset_dict, condition_prob, child_order = pickle.load(file)
    
        
    max_cores = multiprocessing.cpu_count()
    process_batch = (batch_size+max_cores-1) // max_cores
    random_seed = np.random.randint(100000)
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        fs = [executor.submit(
            sample_bayes_condvec,
            i, process_batch,
            no_parent_nodes, col_start_dim_without_continuous, child_parentsset_dict, condition_prob, child_order, np.random.RandomState(random_seed+i)
        ) for i in range(max_cores)]
        
        convec = np.zeros((batch_size, convec_dim), dtype='float32')
        
        for future in as_completed(fs):
            index, poses = future.result()
            for i, pos in zip(range(index*process_batch, min((index+1)*process_batch, batch_size)), poses):
                convec[i, pos] = 1.0
    with open(os.path.join(info_data_path[:info_data_path.rindex('/')], f'convec_{file_seed}.pkl'), 'wb') as file:
        pickle.dump(convec, file)
    exit(0)