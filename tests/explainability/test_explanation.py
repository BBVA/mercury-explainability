import unittest
import typing as TP

from mercury.explainability.explanations import CounterfactualBasicExplanation

import pytest
import numpy as np


class ExplanationTest(unittest.TestCase):
    """ A collection of simple 2D cases. """

    # Simple 1D cases
    def test_invalid_shape(self):
        """ Case where from_.shape != to_.shape. """
        with self.assertRaises(AssertionError):
            CounterfactualBasicExplanation(
                np.array([1.0]),        # from
                np.array([]),           # to
                0.,                     # p
                np.array([]),           # path
                np.array([]),           # path_ps
                np.array([[-1., 1.]]),  # bounds
                np.array([]),           # explored
                np.array([]))           # explored_ps

    def test_invalid_invalid_probability(self):
        """ Case where probability is invalid. """
        with self.assertRaises(AssertionError):
            CounterfactualBasicExplanation(
                np.array([1.0]),        # from
                np.array([2.0]),        # to
                -1.,                    # p
                np.array([]),           # path
                np.array([]),           # path_ps
                np.array([[-1., 1.]]),  # bounds
                np.array([]),           # explored
                np.array([]))           # explored_ps

    def test_invalid_path_shape(self):
        """ Case where path.shape != path_ps.shape. """
        with self.assertRaises(AssertionError):
            CounterfactualBasicExplanation(
                np.array([1.0]),                # from
                np.array([2.0]),                # to
                0.,                             # p
                np.array([1.2, 1.8, 2.0]),      # path
                np.array([0.8, 0.2]),           # path_ps
                np.array([[-1., 1.]]),          # bounds
                np.array([]),                   # explored
                np.array([]))                   # explored_ps

    def test_invalid_bounds(self):
        """ Case where bounds.shape[0] != from_.shape[0] """
        with self.assertRaises(AssertionError):
            CounterfactualBasicExplanation(
                np.array([1.0]),                # from
                np.array([2.0]),                # to
                0.,                             # p
                np.array([1.2, 1.8, 2.0]),      # path
                np.array([0.8, 0.2, 0.]),       # path_ps
                np.array([]),                   # bounds
                np.array([]),                   # explored
                np.array([]))                   # explored_ps

    def test_invalid_path2(self):
        """ Case where explored.shape[0] != explored_ps.shape[0] """
        with self.assertRaises(AssertionError):
            CounterfactualBasicExplanation(
                np.array([1.0]),                            # from
                np.array([2.0]),                            # to
                0.,                                         # p
                np.array([1.2, 1.8, 2.]),                   # path
                np.array([0.8, 0.2, 0.01]),                 # path_ps
                np.array([[-1., 1.]]),                      # bounds
                np.array([1.1, 1.2, 1.5, 1.8, 2.]),         # explored
                np.array([]))                               # explored_ps

    def test_invalid_labels(self):
        """ Case where len(labels) != bounds.shape[0]. """
        with self.assertRaises(AssertionError):
            CounterfactualBasicExplanation(
                np.array([1.0]),                            # from
                np.array([2.0]),                            # to
                0.,                                         # p
                np.array([1.2, 1.8, 2.]),                   # path
                np.array([0.8, 0.2, 0.01]),                 # path_ps
                np.array([[-1., 1.]]),                      # bounds
                np.array([1.1, 1.2, 1.5, 1.8, 2.]),         # explored
                np.array([0.6, 0.3, 0.45, 0.1, 0.]),        # explored_ps
                labels=['a', 'b'])                          # labels
