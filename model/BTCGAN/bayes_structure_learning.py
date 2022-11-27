import os
import numpy as np
import pandas as pd

import jpype
import jpype.imports
from jpype.types import *

# Launch the JVM
root_dir = os.getcwd()
classpath = os.path.join(root_dir, 'tools', 'BayesServer', 'BayesServer-9.5.jar')
jpype.startJVM(*['-Xms2048M', '-Xmx8192M'], classpath=[classpath], convertStrings=True)

# import the Java modules
from tools.BayesServer import data_frame_utils as dfu
from com.bayesserver import *
from com.bayesserver.inference import *
from com.bayesserver.data import *
from com.bayesserver.data.discovery import *
from com.bayesserver.learning.structure import *
from jpype import java

def prior_bayes(bayes_df, prior_edges, no_parents=None):
    child_parents = {}
    for parent, child in prior_edges:
        if no_parents is not None and child == no_parents:
            continue
        if child not in child_parents.keys():
            child_parents[child] = []
        elif parent in child_parents[child]:
            continue
        child_parents[child].append(parent)
    return child_parents

def PC(bayes_df, prior_edges=None, no_parents=None):
    """
    This example uses a Pandas DataFrame as the data source for learning the structure of a Bayesian network
    You can also connect to databases using DatabaseDataReaderCommand
    """
    child_parents = {}
    dt = dfu.to_data_table(bayes_df)
    network = Network()
    data_reader_command = DataTableDataReaderCommand(dt)
    options = VariableGeneratorOptions()
    variable_defs = []
    for col in bayes_df.columns:
        variable_defs.append(VariableDefinition(col, col, VariableValueType.DISCRETE))

    variable_infos = VariableGenerator.generate(
        data_reader_command,
        java.util.Arrays.asList(variable_defs),
        options
    )

    for i, vi in enumerate(variable_infos):
        variable = vi.getVariable()
        network.getNodes().add(Node(variable))

    learning = PCStructuralLearning()
    data_reader_command = DataTableDataReaderCommand(dt)
    variable_references = []
    for v in network.getVariables():
        variable_references.append(VariableReference(v, ColumnValueType.NAME, v.getName()))

    reader_options = ReaderOptions()  # we do not have a case column in this example
    evidence_reader_command = DefaultEvidenceReaderCommand(
        data_reader_command,
        java.util.Arrays.asList(variable_references),
        reader_options
    )
    options = PCStructuralLearningOptions()
#     options.setMaximumConditional(3) # 设置最大的条件数量
    output = learning.learn(evidence_reader_command, network.getNodes(), options)

    for linkOutput in output.getLinkOutputs():
        parent, child = linkOutput.getLink().getFrom().getName(), linkOutput.getLink().getTo().getName()
        if no_parents is not None and child == no_parents:
            continue
        if child not in child_parents.keys():
            child_parents[child] = []
        child_parents[child].append(parent)
    if prior_edges is not None:
        for parent, child in prior_edges:
            if no_parents is not None and child == no_parents:
                continue
            if child not in child_parents.keys():
                child_parents[child] = []
            elif parent in child_parents[child]:
                continue
            child_parents[child].append(parent)
    return child_parents

def ChowLiu(bayes_df, prior_edges=None, no_parents=None):
    child_parents = {}
    dt = dfu.to_data_table(bayes_df)
    network = Network()
    data_reader_command = DataTableDataReaderCommand(dt)
    options = VariableGeneratorOptions()
    variable_defs = []
    for col in bayes_df.columns:
        variable_defs.append(VariableDefinition(col, col, VariableValueType.DISCRETE))

    variable_infos = VariableGenerator.generate(
        data_reader_command,
        java.util.Arrays.asList(variable_defs),
        options
    )
    
    root_node = None
    for i, vi in enumerate(variable_infos):
        variable = vi.getVariable()
        node = Node(variable)
        if variable.getName() == no_parents:
            root_node = node
        network.getNodes().add(node)

    learning = ChowLiuStructuralLearning()
    data_reader_command = DataTableDataReaderCommand(dt)
    variable_references = []
    for v in network.getVariables():
        variable_references.append(VariableReference(v, ColumnValueType.NAME, v.getName()))

    reader_options = ReaderOptions()  # we do not have a case column in this example
    evidence_reader_command = DefaultEvidenceReaderCommand(
        data_reader_command,
        java.util.Arrays.asList(variable_references),
        reader_options
    )
    options = ChowLiuStructuralLearningOptions()
    options.setRoot(root_node)
    output = learning.learn(evidence_reader_command, network.getNodes(), options)

    for linkOutput in output.getLinkOutputs():
        parent, child = linkOutput.getLink().getFrom().getName(), linkOutput.getLink().getTo().getName()
        if no_parents is not None and child == no_parents:
            continue
        if child not in child_parents.keys():
            child_parents[child] = []
        child_parents[child].append(parent)
    if prior_edges is not None:
        for parent, child in prior_edges:
            if no_parents is not None and child == no_parents:
                continue
            if child not in child_parents.keys():
                child_parents[child] = []
            elif parent in child_parents[child]:
                continue
            child_parents[child].append(parent)
    return child_parents