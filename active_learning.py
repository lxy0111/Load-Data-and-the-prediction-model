import numpy as np


def factorial_design(domain):
    """
    convert the parameter domain into a long sample space list
    :param domain: a dict in form {'name': [lower_bound, upper_bound, levels]}
    :return: a numpy array of shape(number of sample points, dimension of sample space)
    """
    Xs = np.meshgrid(*[np.linspace(l, u, levels) for name, (l, u, levels) in domain.items()])
    raveled = []
    for axis in Xs:
        raveled.append(axis.ravel())
    return np.vstack(raveled).T


def space_constraint(space):
    """
    apply constraint that the overall charging state should be less than 1
    :param space: a numpy array of shape(x, 6)
    :return: a numpy array of shape(x, 6)
    """
    tp = space.T
    constrain = (tp[0] * tp[1] + tp[2] * tp[3] + tp[4] * tp[5] < 1)
    return space[constrain]


parameter_domain = {
    'C1': [0.01, 0.3, 30],
    't1': [0, 2, 13],
    'C2': [0.01, 0.3, 30],
    't2': [0, 2, 13],
    'C3': [0.01, 0.3, 30],
    't3': [0, 2, 13]
}

# build sample space with parameter domain
sample_space = factorial_design(parameter_domain)

# apply constraint on the sample space
sample_space = space_constraint(sample_space)
