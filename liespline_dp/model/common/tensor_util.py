"""
A collection of utilities for working with nested tensor structures consisting
of numpy arrays and torch tensors.
"""

import collections

import numpy as np
import torch


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert list not in type_func_dict
    assert tuple not in type_func_dict
    assert dict not in type_func_dict

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError("Cannot handle data type %s" % str(type(x)))


def map_tensor(x, func):
    """
    Apply function @func to torch.Tensor objects in a nested dictionary or
    list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        func (function): function to apply to each tensor

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: func,
            type(None): lambda x: x,
        },
    )


def unsqueeze(x, dim):
    """
    Adds dimension of size 1 at dimension @dim in all torch tensors and numpy arrays
    in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        dim (int): dimension

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.unsqueeze(dim=dim),
            np.ndarray: lambda x: np.expand_dims(x, axis=dim),
            type(None): lambda x: x,
        },
    )


def flatten_single(x, begin_axis=1):
    """
    Flatten a tensor in all dimensions from @begin_axis onwards.

    Args:
        x (torch.Tensor): tensor to flatten
        begin_axis (int): which axis to flatten from

    Returns:
        y (torch.Tensor): flattened tensor
    """
    fixed_size = x.size()[:begin_axis]
    _s = list(fixed_size) + [-1]
    return x.reshape(*_s)


def flatten(x, begin_axis=1):
    """
    Flatten all tensors in nested dictionary or list or tuple, from @begin_axis onwards.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): which axis to flatten from

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis: flatten_single(x, begin_axis=b),
        },
    )


def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions in a tensor to a target dimension.

    Args:
        x (torch.Tensor): tensor to reshape
        begin_axis (int): begin dimension
        end_axis (int): end dimension
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (torch.Tensor): reshaped tensor
    """
    assert begin_axis <= end_axis
    assert begin_axis >= 0
    assert end_axis < len(x.shape)
    assert isinstance(target_dims, (tuple, list))
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions for all tensors in nested dictionary or list or tuple
    to a target dimension.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis, t=target_dims: (
                reshape_dimensions_single(x, begin_axis=b, end_axis=e, target_dims=t)
            ),
            np.ndarray: lambda x, b=begin_axis, e=end_axis, t=target_dims: (
                reshape_dimensions_single(x, begin_axis=b, end_axis=e, target_dims=t)
            ),
            type(None): lambda x: x,
        },
    )


def join_dimensions(x, begin_axis, end_axis):
    """
    Joins all dimensions between dimensions (@begin_axis, @end_axis) into a flat dimension, for
    all tensors in nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]
            ),
            np.ndarray: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]
            ),
            type(None): lambda x: x,
        },
    )


def expand_at_single(x, size, dim):
    """
    Expand a tensor at a single dimension @dim by @size

    Args:
        x (torch.Tensor): input tensor
        size (int): size to expand
        dim (int): dimension to expand

    Returns:
        y (torch.Tensor): expanded tensor
    """
    assert dim < x.ndimension()
    assert x.shape[dim] == 1
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)


def expand_at(x, size, dim):
    """
    Expand all tensors in nested dictionary or list or tuple at a single
    dimension @dim by @size.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        size (int): size to expand
        dim (int): dimension to expand

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(x, lambda t, s=size, d=dim: expand_at_single(t, s, d))


def unsqueeze_expand_at(x, size, dim):
    """
    Unsqueeze and expand a tensor at a dimension @dim by @size.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        size (int): size to expand
        dim (int): dimension to unsqueeze and expand

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    x = unsqueeze(x, dim)
    return expand_at(x, size, dim)
