import os
import numpy as np
import pandas as pd
import pickle

root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'data') # 弃用
output_dir = os.path.join(root_dir, 'data')

def read_adult():
    adult_list = [[] for i in range(15)]
    for adult_path in (os.path.join(data_dir, 'Adult', 'adult.data'),
                       os.path.join(data_dir, 'Adult', 'adult.test')):
        with open(adult_path, 'r', encoding='UTF-8') as file:
            for line in file:
                features = line[:-1].split(', ')
                if len(features) != 15:
                    continue

                if '?' in features:
                    continue

                if features[-1][-1] == '.':
                    features[-1] = features[-1][:-1]

                for i, feature in enumerate(adult_list):
                    feature.append(features[i])

    df = pd.DataFrame()
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                 'salary']
    col_types = ['int32', 'category', 'int64', 'category', 'int32',
                 'category', 'category', 'category', 'category', 'category',
                 'int32', 'int32', 'int32', 'category', 'category']
    data_meta = {}
    for i in range(len(adult_list)):
        col_name, col_type = col_names[i], col_types[i]
        values = adult_list[i]
        df[col_name] = values
        # if col_type == 'category':
        #     df[col_name] = df[col_name].astype('object')
        # else:
        df[col_name] = df[col_name].astype(col_type)
        data_meta[col_name] = col_type
    return df, data_meta

def read_loan():
    loan_path = os.path.join(data_dir, 'Loan', 'Bank_Personal_Loan_Modelling.csv')
    data_meta = {
        'Age': 'int32',
        'Experience': 'int32',
        'Income': 'int32',
        'Family': 'category',
        'CCAvg': 'float32',
        'Education': 'category',
        'Mortgage': 'int32',
        'Personal Loan': 'category',
        'Securities Account': 'category',
        'CD Account': 'category',
        'Online': 'category',
        'CreditCard': 'category',
    }
    df = pd.read_csv(
        loan_path,
        usecols = ['Age', 'Experience', 'Income',
                   'Family', 'CCAvg', 'Education', 'Mortgage',
                   'Personal Loan', 'Securities Account',
                   'CD Account', 'Online', 'CreditCard'],
        dtype = data_meta,
    )
    return df, data_meta

def read_credit():
    credit_path = os.path.join(data_dir, 'Credit', 'creditcard.csv')
    data_meta = {
        'Amount': 'float32',
        'Class': 'category',
    }
    for i in range(1, 29):
        data_meta['V%d'%i] = 'float32'
    df = pd.read_csv(
        credit_path,
        usecols = ['V%d' % i for i in range(1, 29)] + ['Amount', 'Class'],
        dtype = data_meta,
    )
    return df, data_meta

def read_covertype():
    covertype_path = os.path.join(data_dir, 'Covertype', 'covtype.data')
    covertype_list = [[] for i in range(55)]
    with open(covertype_path, 'r', encoding='UTF-8') as file:
        for line in file:
            features = line.split(',')
            if len(features) != 55: continue
            for i, _ in enumerate(covertype_list):
                _.append(features[i])

    col_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'] + \
                ['Wilderness_Area_%02d'%i for i in range(1, 5)] + \
                ['Soil_Type_%d'%i for i in range(1, 41)] + \
                ['Cover_Type']
    col_types = ['int32']*10 + ['category']*(4+40+1)
    
    df, data_meta = pd.DataFrame(), {col: dtype for col, dtype in zip(col_names, col_types)}
    for i in range(len(covertype_list)):
        col = col_names[i]
        df[col] = covertype_list[i]
        df[col] = df[col].astype(col_types[i])
    return df, data_meta

def read_instrusion():
    instrusion_path = os.path.join(data_dir, 'Instrusion', 'corrected')
    instrusion_list = [[] for i in range(42)]
    with open(instrusion_path, 'r', encoding='UTF-8') as file:
        for line in file:
            features = line.split(',')
            if len(features) != 42: continue
            features[-1] = features[-1][:-2]
            for i, _ in enumerate(instrusion_list):
                _.append(features[i])
            
    col_names = ['duration', 'protocol_type', 'service', 'flag',
                 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                 'num_compromised', 'root_shell', 'su_attempted',
                 'num_root', 'num_file_creations', 'num_shells',
                 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'type']
    col_types = ['int32', 'category', 'category', 'category',
                 'int32', 'int32', 'category', 'category',
                 'category', 'category', 'category', 'category',
                 'category', 'category', 'category', 'category',
                 'category', 'category', 'category', 'category',
                 'category', 'category', 'int32', 'int32',
                 'float32', 'float32', 'float32', 'float32',
                 'float32', 'float32', 'float32', 'int32',
                 'int32', 'float32', 'float32', 'float32',
                 'float32', 'float32', 'float32', 'float32', 'float32', 'category']
    df, data_meta = pd.DataFrame(), {col: dtype for col, dtype in zip(col_names, col_types)}
    for i in range(len(instrusion_list)):
        col = col_names[i]
        df[col] = instrusion_list[i]
        df[col] = df[col].astype(col_types[i])
    return df, data_meta

def read_who():
    WHO_path = os.path.join(data_dir, 'WHO','Life.csv')
    data_meta = {
        'Country': 'category',
        'Year': 'category',
        'Status': 'category',
        'Life expectancy': 'float32',
        'Adult Mortality': 'int32',
        'infant deaths': 'int32',
        'Alcohol': 'float32',
        'percentage expenditure': 'float32',
        'Hepatitis B': 'int32',
        'Measles': 'int32',
        'BMI': 'float32',
        'under-five deaths': 'int32',
        'Polio': 'int32',
        'Total expenditure': 'float32',
        'Diphtheria': 'int32',
        'HIV/AIDS': 'float32',
        'GDP': 'float32',
        'Population': 'float32',
        'thinness  1-19 years': 'float32',
        'thinness 5-9 years': 'float32',
        'Income composition of resources': 'float32',
        'Schooling': 'float32',
    }
    df = pd.read_csv(
        WHO_path,
        usecols = [
        'Country',
        'Year',
        'Status',
        'Life expectancy',
        'Adult Mortality',
        'infant deaths',
        'Alcohol',
        'percentage expenditure',
        'Hepatitis B',
        'Measles',
        'BMI',
        'under-five deaths',
        'Polio',
        'Total expenditure',
        'Diphtheria',
        'HIV/AIDS',
        'GDP',
        'Population',
        'thinness  1-19 years',
        'thinness 5-9 years',
        'Income composition of resources',
        'Schooling'
    ],
        dtype = data_meta,
    )
    return df, data_meta


def read_realestate():
    RealEstate_path = os.path.join(data_dir, 'RealEstate','Real estate.csv')
    data_meta = {
        'No': 'int32',
        'X1 transaction date': 'category',
        'X2 house age': 'float32',
        'X3 distance to the nearest MRT station': 'float32',
        'X4 number of convenience stores': 'int32',
        'X5 latitude': 'float32',
        'X6 longitude': 'float32',
        'Y house price of unit area': 'float32',
    }
    df = pd.read_csv(
        RealEstate_path,
        usecols = [
        'No',
        'X1 transaction date',
        'X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores',
        'X5 latitude',
        'X6 longitude',
        'Y house price of unit area',
    ],
        dtype = data_meta,
    )
    return df, data_meta


def read_ols():
    OLS_path = os.path.join(data_dir, 'OLS', 'cancer.csv')
    data_meta = {
        'avgAnnCount': 'float32',
        'avgDeathsPerYear': 'int32',
        'TARGET_deathRate': 'float32',
        'incidenceRate': 'float32',
        'medIncome': 'int32',
        'popEst2015': 'int32',
        'povertyPercent': 'float32',
        'studyPerCap': 'float32',
        'binnedInc': 'category',
        'MedianAge': 'float32',
        'MedianAgeMale': 'float32',
        'MedianAgeFemale': 'float32',
        'Geography': 'category',
        'AvgHouseholdSize': 'float32',
        'PercentMarried': 'float32',
        'PctNoHS18_24': 'float32',
        'PctHS18_24': 'float32',
        'PctBachDeg18_24': 'float32',
        'PctHS25_Over': 'float32',
        'PctBachDeg25_Over': 'float32',
        'PctEmployed16_Over': 'float32',
        'PctUnemployed16_Over': 'float32',
        'PctPrivateCoverage': 'float32',
        'PctPrivateCoverageAlone': 'float32',
        'PctEmpPrivCoverage': 'float32',
        'PctPublicCoverage': 'float32',
        'PctPublicCoverageAlone': 'float32',
        'PctWhite': 'float32',
        'PctBlack': 'float32',
        'PctAsian': 'float32',
        'PctOtherRace': 'float32',
        'PctMarriedHouseholds': 'float32',
        'BirthRate': 'float32',
    }
    df = pd.read_csv(
        OLS_path,
        usecols = [
        'avgAnnCount',
        'avgDeathsPerYear',
        'TARGET_deathRate',
        'incidenceRate',
        'medIncome',
        'popEst2015',
        'povertyPercent',
        'studyPerCap',
        'binnedInc',
        'MedianAge',
        'MedianAgeMale',
        'MedianAgeFemale',
        'Geography',
        'AvgHouseholdSize',
        'PercentMarried',
        'PctNoHS18_24',
        'PctHS18_24',
        'PctBachDeg18_24',
        'PctHS25_Over',
        'PctBachDeg25_Over',
        'PctEmployed16_Over',
        'PctUnemployed16_Over',
        'PctPrivateCoverage',
        'PctPrivateCoverageAlone',
        'PctEmpPrivCoverage',
        'PctPublicCoverage',
        'PctPublicCoverageAlone',
        'PctWhite',
        'PctBlack',
        'PctAsian',
        'PctOtherRace',
        'PctMarriedHouseholds',
        'BirthRate'
    ],
        dtype = data_meta,
    )
    return df, data_meta


def read_medical():
    Medical_path = os.path.join(data_dir, 'Medical', 'insurance.csv')
    data_meta = {
        'age': 'int32',
        'sex': 'category',
        'bmi': 'float32',
        'children': 'int32',
        'smoker': 'category',
        'region': 'category',
        'charges': 'float32',
    }
    df = pd.read_csv(
        Medical_path,
        usecols = [
        'age',
        'sex',
        'bmi',
        'children',
        'smoker',
        'region',
        'charges'
    ],
        dtype = data_meta,
    )
    return df, data_meta


def read_fish():
    Fish_path = os.path.join(data_dir, 'Fish', 'Fish.csv')
    data_meta = {
        'Species': 'category',
        'Weight': 'float32',
        'Length1': 'float32',
        'Length2': 'float32',
        'Length3': 'float32',
        'Height': 'float32',
        'Width': 'float32',
    }
    df = pd.read_csv(
        Fish_path,
        usecols = [
        'Species',
        'Weight',
        'Length1',
        'Length2',
        'Length3',
        'Height',
        'Width'
    ],
        dtype = data_meta,
    )
    return df, data_meta


def read_cdc():
    CDC_path = os.path.join(data_dir, 'CDC', 'Nutrition_Physical_Activity_Obesity.csv')
    data_meta = {
        'YearStart': 'category',
        'YearEnd': 'category',
        'LocationAbbr': 'category',
        'LocationDesc': 'category',
        'Datasource': 'category',
        'Class': 'category',
        'Topic': 'category',
        'Question': 'category',
        'Data_Value_Type': 'category',
        'Data_Value': 'float32',
        'Data_Value_Alt': 'float32',
        'Low_Confidence_Limit': 'float32',
        'High_Confidence_Limit': 'float32',
        'Sample_Size': 'int32',
        'GeoLocation': 'category',
        'ClassID': 'category',
        'TopicID': 'category',
        'QuestionID': 'category',
        'DataValueTypeID': 'category',
        'LocationID': 'int32',
        'StratificationCategory1': 'category',
        'Stratification1': 'category',
        'StratificationCategoryId1': 'category',
        'StratificationID1': 'category',
    }
    df = pd.read_csv(
        CDC_path,
        usecols = [
        'YearStart',
        'YearEnd',
        'LocationAbbr',
        'LocationDesc',
        'Datasource',
        'Class',
        'Topic',
        'Question',
        'Data_Value_Type',
        'Data_Value',
        'Data_Value_Alt',
        'Low_Confidence_Limit',
        'High_Confidence_Limit',
        'Sample_Size',
        'GeoLocation',
        'ClassID',
        'TopicID',
        'QuestionID',
        'DataValueTypeID',
        'LocationID',
        'StratificationCategory1',
        'Stratification1',
        'StratificationCategoryId1',
        'StratificationID1'],
        dtype = data_meta,
    )
    return df, data_meta

def read_car():
    car_path = os.path.join(data_dir, 'Car', 'car.data')
    data_meta = {
        'buying': 'category',
        'maint': 'category',
        'doors': 'category',
        'persons': 'category',
        'lug_boot': 'category',
        'safety': 'category',
        'class': 'category',
    }
    df = pd.read_csv(
        car_path,
        header = None,
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'],
        dtype = data_meta,
    )
    return df, data_meta

def read_magic():
    magic_path = os.path.join(data_dir, 'Magic', 'magic04.data')
    data_meta = {
        'fLength': 'float32',
        'fWidth': 'float32',
        'fSize': 'float32',
        'fConc': 'float32',
        'fConc1': 'float32',
        'fAsym': 'float32',
        'fM3Long': 'float32',
        'fM3Trans': 'float32',
        'fAlpha': 'float32',
        'class': 'category',
    }
    df = pd.read_csv(
        magic_path,
        header = None,
        names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
                 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class'],
        dtype = data_meta,
    )
    return df, data_meta

def read_nursery():
    nursery_path = os.path.join(data_dir, 'Nursery', 'nursery.data')
    data_meta = {
        'parents': 'category',
        'has_nurs': 'category',
        'form': 'category',
        'children': 'category',
        'housing': 'category',
        'finance': 'category',
        'social': 'category',
        'health': 'category',
        'class': 'category',
    }
    df = pd.read_csv(
        nursery_path,
        header = None,
        names = ['parents', 'has_nurs', 'form', 'children',
                 'housing', 'finance', 'social', 'health', 'class'],
        dtype = data_meta,
    )
    return df, data_meta

def read_satellite():
    data_meta = {
        'attr_%d'%i: 'int32' for i in range(1, 37)
    }
    data_meta['class'] = 'category'
    columns = ['attr_%d'%i for i in range(1, 37)] + ['class']
    df = pd.concat(
        [
            pd.read_csv(
                os.path.join(data_dir, 'Satellite', 'sat.trn'),
                sep = ' ',
                names = columns,
                dtype = data_meta,
            ),
            pd.read_csv(
                os.path.join(data_dir, 'Satellite', 'sat.tst'),
                sep = ' ',
                names = columns,
                dtype = data_meta,
            ),
        ],
        axis = 'index',
        ignore_index = True,
        copy = False,
    )
    return df, data_meta

def read_letter_recog():
    letter_rocog_list = [[] for i in range(17)]
    letter_rocog_path = os.path.join(data_dir, 'Letter_recog', 'letter-recognition.data')
    with open(letter_rocog_path, 'r', encoding='UTF-8') as file:
        for line in file:
            features = line[:-1].split(',')
            if len(features) != 17:
                continue

            for i, feature in enumerate(letter_rocog_list):
                feature.append(features[i])

    df = pd.DataFrame()
    col_names = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix',
                 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr',
                 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
    col_types = ['category', 'int32', 'int32', 'int32', 'int32',
                 'int32', 'int32', 'int32', 'int32',
                 'int32', 'int32', 'int32', 'int32',
                 'int32', 'int32', 'int32', 'int32',]
    data_meta = {}
    for i in range(len(letter_rocog_list)):
        col_name, col_type = col_names[i], col_types[i]
        values = letter_rocog_list[i]
        df[col_name] = values
        df[col_name] = df[col_name].astype(col_type)
        data_meta[col_name] = col_type
    return df, data_meta

def read_chess():
    chess_path = os.path.join(data_dir, 'Chess', 'krkopt.data')
    data_meta = {
        'White King file': 'category',
        'White King rank': 'category',
        'White Rook file': 'category',
        'White Rook rank': 'category',
        'Black King file': 'category',
        'Black King rank': 'category',
        'optimal depth-of-win for White': 'category',
    }
    df = pd.read_csv(
        chess_path,
        header = None,
        names = ['White King file', 'White King rank', 'White Rook file',
                 'White Rook rank', 'Black King file', 'Black King rank', 'optimal depth-of-win for White'],
        dtype = data_meta,
    )
    return df, data_meta

def read_connect4():
    connect4_list = [[] for i in range(43)]
    connect4_path= os.path.join(data_dir, 'Connect-4', 'connect-4.data')
    with open(connect4_path, 'r', encoding='UTF-8') as file:
        for line in file:
            features = line[:-1].split(',')
            if len(features) != 43:
                continue

            for i, feature in enumerate(connect4_list):
                feature.append(features[i])

    df = pd.DataFrame()
    col_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6',
                 'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
                 'c1', 'c2', 'c3', 'c4', 'c5', 'c6',
                 'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
                 'e1', 'e2', 'e3', 'e4', 'e5', 'e6',
                 'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
                 'g1', 'g2', 'g3', 'g4', 'g5', 'g6',
                 'class']
    col_types = ['category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                 'category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                 'category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                 'category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                 'category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                 'category', 'category', 'category']
    data_meta = {}
    for i in range(len(connect4_list)):
        col_name, col_type = col_names[i], col_types[i]
        values = connect4_list[i]
        df[col_name] = values
        df[col_name] = df[col_name].astype(col_type)
        data_meta[col_name] = col_type
    return df, data_meta

def read_shuttle():
    shuttle_list = [[] for i in range(10)]
    for shuttle_path in (os.path.join(data_dir, 'Shuttle', 'shuttle.trn'),
                         os.path.join(data_dir, 'Shuttle', 'shuttle.tst')):
        with open(shuttle_path, 'r', encoding='UTF-8') as file:
            for line in file:
                features = line[:-1].split(' ')
                if len(features) != 10:
                    continue

                for i, feature in enumerate(shuttle_list):
                    feature.append(features[i])

    df = pd.DataFrame()
    col_names = ['time', 'C1', 'C2', 'C3', 'C4',
                 'C5', 'C6', 'C7', 'C8', 'CLASS']
    col_types = ['category', 'int32', 'int32', 'int32', 'int32',
                 'int32', 'int32', 'int32', 'int32', 'category']
    data_meta = {}
    for i in range(len(shuttle_list)):
        col_name, col_type = col_names[i], col_types[i]
        values = shuttle_list[i]
        df[col_name] = values
        df[col_name] = df[col_name].astype(col_type)
        data_meta[col_name] = col_type

    return df, data_meta

def read_pokerhand():
    pokerhand_list = [[] for i in range(11)]
    for pokerhand_path in (os.path.join(data_dir, 'Pokerhand', 'poker-hand-testing.data'),
                           os.path.join(data_dir, 'Pokerhand', 'poker-hand-training-true.data')):
        with open(pokerhand_path, 'r', encoding='UTF-8') as file:
            for line in file:
                features = line[:-1].split(',')
                if len(features) != 11:
                    continue

                for i, feature in enumerate(pokerhand_list):
                    feature.append(features[i])

    df = pd.DataFrame()
    col_names = ['S1', 'C1', 'S2', 'C2', 'S3',
                 'C3', 'S4', 'C4', 'S5', 'C5',
                 'CLASS']
    col_types = ['category', 'category', 'category', 'category', 'category',
                 'category', 'category', 'category', 'category', 'category',
                 'category']
    data_meta = {}
    for i in range(len(pokerhand_list)):
        col_name, col_type = col_names[i], col_types[i]
        values = pokerhand_list[i]
        df[col_name] = values
        df[col_name] = df[col_name].astype(col_type)
        data_meta[col_name] = col_type
    return df, data_meta

dataset = {
    'Adult': read_adult,
    'Loan': read_loan,
    'Credit': read_credit,
    'Covertype': read_covertype,
    'Instrusion': read_instrusion,
    'WHO': read_who,
    'RealEstate': read_realestate,
    'OLS': read_ols,
    'Medical': read_medical,
    'Fish': read_fish,
    'CDC': read_cdc,
    'Car': read_car,
    'Magic': read_magic,
    'Nursery': read_nursery,
    'Satellite': read_satellite,
    'Letter': read_letter_recog,
    'Chess': read_chess,
    'Connect-4': read_connect4,
    'Shuttle': read_shuttle,
    'Pokerhand': read_pokerhand,
}

def get_dataset_names():
    return dataset.keys()

def load_dataset(data_name, write_dir=None):
    df, meta = dataset[data_name]()
    if write_dir is not None:
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        df.to_csv(os.path.join(output_dir, 'origin.csv'), index=False)
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as file:
        pickle.dump(meta, file)
    return df, meta

def read_dataset(data_name, model_name_list, change_float=False, change_object=True):
    dataset_dir = os.path.join(output_dir, data_name)
    with open(os.path.join(dataset_dir, 'meta.pkl'), 'rb') as file:
        meta = pickle.load(file)
    if change_object:
        for col, col_type in meta.items():
            if col_type == 'category':
                meta[col] = 'object'

    # df_train_list, df_test_list = [], []
    if change_float:
        float_meta = {col: (dtype if dtype == 'object' or dtype == 'category' else 'float64') for col, dtype in meta.items()}
    else:
        float_meta = meta
    i = 0
    while True:
        if not os.path.exists(os.path.join(dataset_dir, '%s_train_%d.csv' % (model_name_list[0].lower(), i))):
            break
        this_df_train_list, this_df_test_list = [], []
        for model_name in model_name_list:
#             if not os.path.exists(os.path.join(dataset_dir, '%s_train_%d.csv' % (model_name.lower(), i))):
#                 continue
            this_df_train_list.append(pd.read_csv(
                os.path.join(dataset_dir, '%s_train_%d.csv' % (model_name.lower(), i)),
                dtype = meta if model_name == 'origin' else float_meta,
                engine = 'c',
            ))
            this_df_test_list.append(pd.read_csv(
                os.path.join(dataset_dir, '%s_test_%d.csv' % (model_name.lower(), i)),
                dtype = meta if model_name == 'origin' else float_meta,
                engine = 'c',
            ))
        yield this_df_train_list, this_df_test_list
        # df_train_list.append(this_df_train_list)
        # df_test_list.append(this_df_test_list)
        i += 1
    # return df_names, df_train_list, df_test_list