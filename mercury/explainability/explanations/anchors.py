import typing as TP
import numpy as np
import pandas as pd
import heapq
import itertools

class AnchorsWithImportanceExplanation(object):
    """
    Extended Anchors Explanations

    Args:
        explain_data:
            A pandas DataFrame containing the observations for which an explanation has to be found.
        explanations:
            A list containing the results of computing the explanations for explain_data.
        categorical:
            A dictionary containing as key the features that are categorical and as value, the possible
            categorical values.
    """

    def __init__(
            self,
            explain_data: pd.DataFrame,
            explanations: TP.List,
            categorical: dict = {}
        ) -> None:
        self.explain_data = explain_data
        self.explanations = explanations
        self.categorical = categorical
    
    def interpret_explanations(self, n_important_features: int) -> str:
        """
        This method prints a report of the important features obtaiend.

        Args:
            n_important_features:
                The number of imporant features that will appear in the report.
                Defaults to 3.
        """
        names = []
        explanations_found = [explan for explan in self.explanations if not isinstance(explan, str)]
        for expl in explanations_found:
            for name in expl.data['anchor']:
                # split without an argument splits by spaces, and in every item in expl['names']
                # the first word refers to the feature name.
                if (
                    (' = ' in name) or 
                    ((len(self.categorical) > 0) and (name in [item for sublist in list(self.categorical.values()) for item in sublist]))
                    ):
                    names.append(name)
                else:
                    names.append(' '.join(name[::-1].split('.', 1)[1][::-1].split()[:-1]))

        unique_names, count_names = np.unique(names, return_counts=True)
        top_feats = heapq.nlargest(n_important_features, count_names)
        print_values = ['The ', str(n_important_features), ' most common features are: ']
        unique_names_ordered = sorted(unique_names.tolist(), key=lambda x: count_names[unique_names.tolist().index(x)], reverse=True)
        count_names_ordered = sorted(count_names.tolist(), reverse=True)
        n_explanations = 0
        for unique_name, count_name in zip(unique_names_ordered[:n_important_features], count_names_ordered[:n_important_features]):
            if n_explanations == 0:
                print_values.append([unique_name, ' with a frequency of ', 
                    str(count_name), ' (', str(100 * count_name / len(explanations_found)), '%) '])
            elif n_explanations == n_important_features - 1:
                print_values.append([' and ', unique_name, ' with a frequency of ', 
                    str(count_name), ' (', str(100 * count_name / len(explanations_found)), '%) '])
            else:
                print_values.append([', ',unique_name, ' with a frequency of ', 
                    str(count_name), ' (', str(100 * count_name / len(explanations_found)), '%) '])
        n_explanations += 1
        interptretation = ''.join(list(itertools.chain(*print_values)))
        print(interptretation)
        return interptretation
