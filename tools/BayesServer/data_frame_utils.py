# __author__ = 'Bayes Server'
# __version__= '0.3'
# __copyright__ = "Copyright 2021, Bayes Server"

import numpy as np
import pandas as pd
import jpype    # pip install jpype1    (version 1.2.1 or later)
import jpype.imports
from jpype.types import *
from jpype import java

from com.bayesserver.data import *

def _to_java_class(data_type):
    """
    Converts numpy data type to equivalent Java class
    :param data_type: the numpy data type
    :return: The Java Class
    """
    if data_type == np.int32:
        return java.lang.Integer(0).getClass()  # .class not currently supported by jpype
    elif data_type == np.int64:
        return java.lang.Long(0).getClass()  # .class not currently supported by jpype
    elif data_type == np.float32:
        return java.lang.Float(0).getClass()  # .class not currently supported by jpype
    elif data_type == np.float64:
        return java.lang.Double(0.0).getClass()  # .class not currently supported by jpype
    elif data_type == np.bool:
        return java.lang.Boolean(False).getClass()  # .class not currently supported by jpype
    elif data_type == np.object:
        return java.lang.String().getClass()  # .class not currently supported by jpype

    raise ValueError('dtype [{}] not currently supported'.format(data_type))


def to_data_table(df):
    """
    将DataFrame转成DataTable
    """
    data_table = DataTable()
    cols = data_table.getColumns()

    for name, data_type in df.dtypes.iteritems():
        java_class = _to_java_class(data_type)
        data_column = DataColumn(name, java_class)
        cols.add(data_column)

    for index, row in df.iterrows():

        xs = [None if pd.isnull(x) else x for x in row]

        data_table.getRows().add(xs)

    return data_table