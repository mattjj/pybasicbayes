from __future__ import division
import numpy as np

model = None
labels_list = None

def _get_sampled_labels(idx):
    model.add_data(model.labels_list[idx].data,initialize_from_prior=False)
    l = model.labels_list.pop()
    return l.z, l._normalizer

def _get_sampled_component_params(idx):
    model.components[idx].resample([l.data[l.z == idx] for l in labels_list])
    return model.components[idx].parameters

