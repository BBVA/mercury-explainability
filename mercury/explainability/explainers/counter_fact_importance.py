import typing as TP
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from abc import ABC
from .explainer import MercuryExplainer

from mercury.explainability.explanations.counter_factual import (
    CounterfactualWithImportanceExplanation
)

from alibi.api.interfaces import Explanation
from alibi.explainers import CounterFactual, CounterFactualProto


class CounterFactualExplainerBase(ABC):
    """
    Abstract base class for CounterFactual explainers which hold common
    functionalities. You should extend this if you are implementing a
    custom counter factual explainer.

    Args:
        feature_names:
            List of names of the features used in the model.
        drop_features:
            List with the names, as specified in feature_names, or with the
            indexes of the variables to leave out while looking for
            explanations.
    """

    def __init__(self, feature_names: TP.List[str],
                 drop_features: TP.List[TP.Any] = []):
        self.feature_names = feature_names
        self.drop_features = drop_features
        # Each subclass will use its internal Alibi member (CounterFactual or
        # CounterFactualProto)
        self.cf_super = None

    def _regularize_counterfactual(
        self,
        predict_fn: TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], TP.Union[float, 'np.ndarray']],
        explain_data: 'np.ndarray',
        counterfactual_original: 'np.ndarray',
        min_prob: float,
        threshold_incr: float = 0.01,
        init_threshold: float = 0.1,
        max_iterations: int = 10000) -> TP.Tuple[TP.Any]:
        """
        Internal function to regularize the encountered counterfactuals.

        Args:
            predict_fn:
                Model prediction function. This function should return the probabilities of belonging to each class.
            explain_data:
                Numpy array with containing the values for all features for the original instance.
            counterfactual_original:
                Numpy array containing the resulting counterfactual computed by alibi.
            min_prob:
                Target minimum probability to be reached while regularizing.
            threshold_incr:
                Increments to be applied to the threshold by which each feature is reduced to zero. Defaults to 0.01.
            init_threshold:
                Initial threshold that determines whether the feature has to be reduced to zero or not. Defaults 0.1.
            max_iterations:
                Maximum number of iterations for which the regularization process runs. Defaults to 10000.

            Returns:
                A tuple containing the new counterfactual, the difference between the new counterfactual and the original
                instance and finally, the achieved probability.
        """
        try:
            if len(explain_data[0]) != len(counterfactual_original[0]):
                raise AttributeError("""
                The dimensions of the original instance ({}) and that
                of the counterfactual instance ({}) do not match
                """.format(len(explain_data[0]), len(counterfactual_original[0])))
        except AttributeError:
            raise

        differences = (explain_data - counterfactual_original)[0]
        prediction = predict_fn(counterfactual_original)[0]
        counter_class = np.argmax(prediction)
        counter_proba = prediction[counter_class]

        try:
            if min_prob >= counter_proba:
                raise AttributeError("""
                The objective class probability is larger than that of the counterfactual passed to this method.
                This means that either the target probability is greater than that stablished when calling the
                counterfactual method or that the counterfactual being passed to this method is not well formed.
                The inputed target_probability was {:.3f} while the obtained probability for this counterfactual is {:.3f}
                """.format(min_prob, counter_proba))
        except AttributeError:
            raise

        new_proba = counter_proba
        new_cfs = []
        threshold = init_threshold
        n_iter = 0

        while new_proba >= min_prob:
            new_cf = []
            for i, diff in enumerate(differences):
                if np.abs(diff) > threshold:
                    new_cf.append(counterfactual_original[0][i])
                else:
                    new_cf.append(explain_data[0][i])
            new_probas_ = predict_fn(np.array(new_cf).reshape(1, -1))[0]
            new_proba_ = new_probas_[counter_class]
            if new_proba_ > min_prob:
                new_cfs.append(new_cf)
            else:
                if len(new_cfs) == 0:
                    new_cfs.append(counterfactual_original[0])

            differences = (explain_data - new_cfs[-1])[0]
            threshold += threshold_incr
            new_proba = new_proba_
            n_iter += 1

            if n_iter == max_iterations:
                print('No solution...')
                print('Final threshold: ', threshold)
                break

        regularized_counterfactual = np.array(new_cfs[-1])
        diff = explain_data - regularized_counterfactual
        new_probability = predict_fn(regularized_counterfactual.reshape(1, -1))[0][counter_class]

        return regularized_counterfactual, diff, new_probability

    def _compute_feature_range(
        self,
        feature_range: TP.Union[TP.Tuple, str],
        drop_features: TP.List[int],
        explain_data: 'np.ndarray'
    ) -> 'np.ndarray':
        """
        This function computes a feature range adjusted to a given observation
        such that the range for the features established on argument drop_features
        do not allow for change.

        Args:
            feature_range:
                Tuple with min and max ranges to allow for perturbed instances. Min and max
                ranges have to be numpy arrays with dimension (1 x nb of features)
                for feature-wise ranges when dropping features.
            drop_features:
                List with the indexes of the variables to leave out while looking
                for explanations.
            explain_data:
                Instance to be explained

        Returns:
            Tuple containing the new feature range.
        """
        explain_data = explain_data.reshape(1, -1)
        feat_range = feature_range
        feat_range[0][drop_features] = explain_data[0][drop_features]
        feat_range[1][drop_features] = explain_data[0][drop_features]
        return feat_range

    def get_feature_importance(
        self,
        explain_data: pd.DataFrame,
        n_important_features: int = 3,
        print_every: int = 5,
        threshold_probability=0.5,
        threshold_incr: float = 0.01,
        init_threshold: float = 0.1,
        max_iterations: int = 10000,
        get_report: bool = True
    ) -> CounterfactualWithImportanceExplanation:
        """
        This method computes the feature importance for a set of observations.

        Args:
            explain_data:
                Pandas DataFrame containing all the instances for which to find an anchor and therefore
                obtain feature importances.
            n_important_features:
                Number of top features that will be printed. Defaults to 3.
            print_every:
                Logging information (default=5).
            threshold_incr:
                The increment of the threshold to be used in the regularizer so as to bring the differences
                between a counterfactual instance and the original one to zero.
            init_threshold:
                The initial and minimum value to be used in the regularizer to bring the differences
                between a counterfactual instance and the original one to zero.
            max_iterations:
                Maximum number of iterations for the regularizer. This parameter gets pass
                down to the regularize_counterfactual method.
            get_report:
                Boolean determining whether to print the explanation or not.

        Returns:
            A CounterfactualExtendedExplanation object that contains all explanations as well as
            their interpretation.
        """
        failed_explanations = 0
        counterfactuals = []
        attempted_explanations = 0
        for item_idx, item in explain_data.iterrows():
            try:
                attempted_explanations += 1
                cf_explanation = self.explain(
                    item.values.reshape(1, -1)
                )
                cf_regularized, diff, new_proba = self._regularize_counterfactual(
                    self.predict_fn,
                    item.values.reshape(1, -1),
                    cf_explanation.data['cf']['X'],
                    min_prob=threshold_probability,
                    threshold_incr=threshold_incr,
                    max_iterations=max_iterations,
                    init_threshold=init_threshold
                )
                cf_explanation = {
                    'cf_original': cf_explanation.data['cf']['X'],
                    'cf_class': cf_explanation.data['cf']['class'],
                    'cf_proba': cf_explanation.data['cf']['proba'][0][
                        cf_explanation.data['cf']['class']
                    ],
                    'cf_regularized': cf_regularized,
                    'cf_reg_proba': new_proba,
                    'diff_orig_reg': diff
                }
            except Exception:
                failed_explanations += 1
                print('''
                There's been a problem while computing the explanation for observation: {}.
                If this observation is critical to you try with some other hyperparameters.
                '''.format(item_idx))
                cf_explanation = {
                    'cf_original': None,
                    'cf_class': None,
                    'cf_proba': None,
                    'cf_regularized': None,
                    'cf_reg_proba': None,
                    'diff_orig_reg': None
                }
            if attempted_explanations % print_every == 0:
                print('{} counterfactuals already computed. Found a solution for {} of them'.format(
                    attempted_explanations, attempted_explanations - failed_explanations
                ))
            counterfactuals.append(cf_explanation)
        print('All counterfactuals have been computed and regularized')

        count_differences = dict(
            (feat_name, 0.0) for feat_name in self.feature_names
        )
        total_diffs = np.zeros(len(self.feature_names))

        if len(counterfactuals) > 0:
            for counterfactual_i in counterfactuals:
                if counterfactual_i['cf_original'] is not None:
                    total_diffs += counterfactual_i['diff_orig_reg'][0]
                    diffs_dict = dict(zip(self.feature_names, counterfactual_i['diff_orig_reg'][0]))
                    for k, v in diffs_dict.items():
                        if np.abs(v) > 0.0:
                            count_differences[k] += 1

        max_diff = np.abs(total_diffs).max(axis=0)
        total_diffs /= max_diff

        importances = [
            (name, value, direction)
            for value, direction, name
            in sorted(
                zip(np.abs(total_diffs), np.sign(total_diffs), self.feature_names),
                reverse=True
            )
        ]

        count_differences_norm = dict(
            (key, value / (explain_data.shape[0] - failed_explanations))
            for key, value
            in count_differences.items()
        )
        count_differences_norm = {
            key: value
            for key, value
            in sorted(
                count_differences_norm.items(),
                key=lambda item: item[1],
                reverse=True
            )
        }

        n_important_features = n_important_features if n_important_features < len(self.feature_names) else len(self.feature_names)

        counterfactualExtendedExplanation = CounterfactualWithImportanceExplanation(
            explain_data,
            counterfactuals,
            importances,
            count_differences,
            count_differences_norm
        )

        if get_report:
            if n_important_features == len(self.feature_names):
                print('The total number of important features was too large and therefore all will be shown')
            if failed_explanations > 0:
                print('There were a total of {:d} fails'.format(failed_explanations))
            counterfactualExtendedExplanation.interpret_explanations(
                n_important_features
            )
        return counterfactualExtendedExplanation

    def __new__(cls, *args, **kwargs):
        if cls is CounterFactualExplainerBase:
            raise TypeError("""You shouldn't be instantiating this. """ +
                            """Use the proper subclass""")
        return object.__new__(cls)


class CounterfactualExplainer(CounterFactualExplainerBase):
    """
    Backed by Alibi's CounterFactual, this class extends its functionality to allow for the computation of Feature
    Importances by means of the computation of several Counterfactuals.

    Args:
        predict_fn:
            Model prediction function. This function should return the probabilities of belonging to each class.
        feature_names:
            List of names of the features used in the model.
        shape:
            Shape of input data starting with batch size. By default it is inferred from feature_names.
        drop_features:
            List with the names, as specified in feature_names, or with the indexes of the variables to leave
            out while looking for explanations (in get_feature_importance method).
        distance_fn:
            Distance function to use in the loss term
        target_proba:
            Target probability for the counterfactual to reach
        target_class:
            Target class for the counterfactual to reach, one of 'other', 'same' or an integer denoting
            desired class membership for the counterfactual instance
        max_iter:
            Maximum number of interations to run the gradient descent for (inner loop)
        early_stop:
            Number of steps after which to terminate gradient descent if all or none of found instances are solutions
        lam_init:
            Initial regularization constant for the prediction part of the Wachter loss
        max_lam_steps:
            Maximum number of times to adjust the regularization constant (outer loop) before terminating the search
        tol:
            Tolerance for the counterfactual target probability
        learning_rate_init:
            Initial learning rate for each outer loop of lambda
        feature_range:
            Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be floats or
            numpy arrays with dimension (1 x nb of features) for feature-wise ranges
        eps:
            Gradient step sizes used in calculating numerical gradients, defaults to a single value for all
            features, but can be passed an array for feature-wise step sizes
        init:
            Initialization method for the search of counterfactuals, currently must be 'identity'
        decay:
            Flag to decay learning rate to zero for each outer loop over lambda
        write_dir:
            Directory to write Tensorboard files to
        debug:
            Flag to write Tensorboard summaries for debugging
        sess:
            Optional Tensorflow session that will be used if passed instead of creating or inferring one internally
    """
    def __init__(
        self,
        predict_fn: TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], TP.Union[float, 'np.ndarray']],
        feature_names: TP.List[str],
        shape: TP.Tuple[TP.Any] = None,
        drop_features: TP.List[TP.Any] = [],
        distance_fn: str = 'l1',
        target_proba: float = 1.0,
        target_class: TP.Union[str, int] = 'other',
        max_iter: int = 1000,
        early_stop: int = 50,
        lam_init: float = 1e-1,
        max_lam_steps: int = 10,
        tol: float = 0.05,
        learning_rate_init: float = 0.1,
        feature_range: TP.Union[TP.Tuple[TP.Any], str] = (-1e10, 1e10),
        eps: TP.Union[float, 'np.ndarray'] = 0.01,
        init: str = 'identity',
        decay: bool = True,
        write_dir: str = None,
        debug: bool = False,
        sess: tf.compat.v1.Session = None
    ) -> None:
        super(CounterfactualExplainer,
              self).__init__(feature_names, drop_features)

        self.predict_fn = predict_fn
        self.shape = shape if shape else (1, len(feature_names))
        self.distance_fn = distance_fn
        self.target_proba = target_proba
        self.target_class = target_class
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.lam_init = lam_init
        self.max_lam_steps = max_lam_steps
        self.tol = tol
        self.learning_rate_init = learning_rate_init
        self.feature_range = feature_range
        self.eps = eps
        self.init = init
        self.decay = decay
        self.write_dir = write_dir
        self.debug = debug
        self.sess = sess
        self.cat_vars = None

        # Instantiate counterfactual
        self.cf_super = CounterFactual(
            predict_fn=self.predict_fn, shape=self.shape,
            distance_fn=self.distance_fn, target_proba=self.target_proba,
            target_class=self.target_class, max_iter=self.max_iter,
            early_stop=self.early_stop, lam_init=self.lam_init,
            max_lam_steps=self.max_lam_steps, tol=self.tol,
            learning_rate_init=self.learning_rate_init,
            feature_range=self.feature_range, eps=self.eps,
            init=self.init, decay=self.decay, write_dir=self.write_dir,
            debug=self.debug, sess=self.sess)

    def explain(self, explain_data: 'np.ndarray') -> Explanation:
        """
        This method serves two purposes. Firstly as an interface between the user and the CounterFactual class
        explain method and secondly, if there are features that shouldn't be taken into account, the method
        recomputes feature_range to account for this and reinstantiates the parent class (CounterFactual).
        Explain an instance and return the counterfactual with metadata.

        Args:
            explain_data (np.ndarray): Instance to be explained

        Returns:
            Explanation object as specified in Alibi's original explain method.
        """
        return self.cf_super.explain(
            explain_data
        )


class CounterfactualProtoExplainer(CounterFactualExplainerBase):
    """
    Backed by Alibi's CounterFactualProto, this class extends its functionality to allow for the computation of Feature
    Importances by means of the computation of several Counterfactuals.

    Args:
        predict_fn:
            Model prediction function. This function should return the probabilities of belonging to
            each class.
        train_data:
            Numpy array or Pandas dataframe with the training examples
        feature_names:
            List of names of the features used in the model. Will be inferred by default from train_data
        shape:
            Shape of input data starting with batch size. Will be inferred by default from train_data
        drop_features:
            List with the names, as specified in feature_names, or with the indexes of the variables to leave
            out while looking for explanations.
        beta:
            Regularization constant for L1 loss term
        kappa:
            Confidence parameter for the attack loss term
        feature_range:
            Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be floats or
            numpy arrays with dimension (1x nb of features) for feature-wise ranges.
            Will be inferred by default from train_data
        gamma:
            Regularization constant for optional auto-encoder loss term
        ae_model:
            Optional auto-encoder model used for loss regularization
        enc_model:
            Optional encoder model used to guide instance perturbations towards a class prototype
        theta:
            Constant for the prototype search loss term. Default is 5. Set it to zero to disable it.
        cat_vars:
            Dict with as keys the categorical columns and as values
            the number of categories per categorical variable.
        ohe:
            Whether the categorical variables are one-hot encoded (OHE) or not. If not OHE, they are
            assumed to have ordinal encodings.
        use_kdtree:
            Whether to use k-d trees for the prototype loss term if no encoder is available
        learning_rate_init:
            Initial learning rate of optimizer
        max_iterations:
            Maximum number of iterations for finding a counterfactual
        c_init:
            Initial value to scale the attack loss term. If the computation shall be fastened up, this
            parameter should take a small value (0 or 1).
        c_steps:
            Number of iterations to adjust the constant scaling the attack loss term. If the computation
            shall be fastened up this parameter should take a somewhat small value (between 1 and 5).
        eps:
            If numerical gradients are used to compute dL/dx = (dL/dp) * (dp/dx), then eps[0] is used to
            calculate dL/dp and eps[1] is used for dp/dx. eps[0] and eps[1] can be a combination of float values and
            numpy arrays. For eps[0], the array dimension should be (1x nb of prediction categories) and for
            eps[1] it should be (1x nb of features)
        clip:
            Tuple with min and max clip ranges for both the numerical gradients and the gradients
            obtained from the TensorFlow graph
        update_num_grad:
            If numerical gradients are used, they will be updated every update_num_grad iterations. If
            the computation shall be fastened up this parameter should take a somewhat large value
            (between 100 and 250).
        update_num_grad:
            If numerical gradients are used, they will be updated every update_num_grad iterations
        write_dir:
            Directory to write tensorboard files to
        sess:
            Optional Tensorflow session that will be used if passed instead of creating or inferring one internally
        trustscore_kwargs:
            keyword arguments for the trust score object used to define the k-d trees for each class.
            (See original alibi) counterfactual guided by prototypes docs.
        d_type:
            Distance metric. Options are "abdm", "mvdm" or "abdm-mvdm".
        w:
            If the combined metric "abdm-mvdm" is used, w is the weight (between 0 and 1) given to abdm.
        disc_perc:
            List with percentiles (int) used for discretization
        standardize_cat_vars:
            Whether to return the standardized values for the numerical distances of each categorical feature.
        smooth:
            if the difference in the distances between the categorical variables is too large, then a lower
            value of the smooth argument (0, 1) can smoothen out this difference. This would only be relevant
            if one categorical variable has significantly larger differences between its categories than others.
            As a result, the counterfactual search process will likely leave that categorical variable unchanged.
        center:
            Whether to center the numerical distances of the categorical variables between the min and max feature ranges.
        update_feature_range:
            whether to update the feature_range parameter for the categorical variables based on the min and max values
            it computed in the constructor.
    """

    def __init__(self,
        predict_fn: TP.Callable[[TP.Union['np.ndarray', pd.DataFrame]], TP.Union[float, 'np.ndarray']],
        train_data: TP.Union[np.ndarray, pd.DataFrame],
        feature_names: TP.List[str] = None,
        shape: TP.Tuple[TP.Any] = None,
        drop_features: TP.List[TP.Any] = [],
        kappa: float = 0.,
        beta: float = .1,
        feature_range: TP.Union[TP.Tuple[TP.Any], str] = None,
        gamma: float = 0.,
        ae_model: tf.keras.Model = None,
        enc_model: tf.keras.Model = None,
        theta: float = 5.0,
        cat_vars: dict = None,
        ohe: bool = False,
        use_kdtree: bool = False,
        learning_rate_init: float = 1e-2,
        max_iterations: int = 1000,
        c_init: float = 10.,
        c_steps: int = 10,
        eps: TP.Union[float,'np.ndarray'] = (1e-3, 1e-3),
        clip: TP.Tuple[float] = (-1000, 1000),
        update_num_grad: int = 1,
        write_dir: str = None,
        sess: tf.compat.v1.Session = None,
        trustscore_kwargs: dict = None,
        d_type: str = 'abdm',
        w: float = None,
        disc_perc: TP.Union[TP.List[TP.Union[int, float]], TP.Tuple[TP.Union[int, float]]] = [25, 50, 75],
        standardize_cat_vars: bool = False,
        smooth: float = 1.,
        center: bool = True,
        update_feature_range: bool = True
    ) -> None:
        if feature_names is None and type(train_data) is not pd.DataFrame:
            raise ValueError("If feature_names is not present, train_data" +
                             "should be a Pandas DataFrame. ")

        super(CounterfactualProtoExplainer,
              self).__init__(feature_names, drop_features)

        self.fitted = False
        self.feature_range = feature_range
        self.predict_fn = predict_fn
        self.shape = (1, train_data.shape[-1])
        self.feature_names = feature_names
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.ae_model = ae_model
        self.enc_model = enc_model
        self.theta = theta
        self.cat_vars = cat_vars
        self.ohe = ohe
        self.use_kdtree = True if not enc_model and not ae_model else False
        self.learning_rate_init = learning_rate_init
        self.max_iterations = max_iterations
        self.c_init = c_init
        self.c_steps = c_steps
        self.eps = eps
        self.clip = clip
        self.update_num_grad = update_num_grad
        self.write_dir = write_dir
        self.sess = sess

        self.train_data = train_data

        if isinstance(train_data, pd.DataFrame):
            self.feature_names = list(train_data.columns)
            self.feature_range = \
                (train_data.min().values, train_data.max().values) \
                    if not feature_range else (-2e5, 2e5)
            self.train_data=train_data.values

        self.cf_super = CounterFactualProto(
            predict=predict_fn, shape=self.shape, kappa=self.kappa, beta=self.beta,
            feature_range=self.feature_range, gamma=self.gamma, ae_model=self.ae_model,
            enc_model=self.enc_model, theta=self.theta, cat_vars=self.cat_vars, ohe=self.ohe,
            use_kdtree=self.use_kdtree, learning_rate_init=self.learning_rate_init,
            max_iterations=self.max_iterations, c_init=self.c_init, c_steps=self.c_steps,
            eps=self.eps, clip=self.clip, update_num_grad=self.update_num_grad,
            write_dir=self.write_dir, sess=self.sess
        )

        self.cf_super = self.cf_super.fit(self.train_data, trustscore_kwargs=trustscore_kwargs, d_type=d_type,
            w=w, disc_perc=disc_perc, standardize_cat_vars=standardize_cat_vars,
            smooth=smooth, center=center, update_feature_range=update_feature_range)

    def explain(
        self,
        explain_data: 'np.ndarray',
        Y: 'np.ndarray' = None,
        target_class: list = None,
        k: int = None,
        k_type: str = 'mean',
        threshold: float = 0.,
        verbose: bool = False,
        print_every: int = 100,
        log_every: int = 100) -> Explanation:
        """
        This method serves three purposes. Firstly as an interface between the user and the CounterFactualProto
        class explain method, secondly, if there are features that shouldn't be taken into account, the method
        recomputes feature_range to account for this and reinstantiates the parent class (CounterFactualProto)
        and thridly, if a new CounterFactualProto instance has been created it's fitted.
        Explain instance and return counterfactual with metadata.

        Args:
            X:
                Instances to explain
            Y:
                Labels for X as one-hot-encoding
            target_class:
                List with target classes used to find closest prototype. If None, the nearest prototype
                except for the predict class on the instance is used.
            k:
                Number of nearest instances used to define the prototype for a class. Defaults to using all
                instances belonging to the class if an encoder is used and to 1 for k-d trees.
            k_type:
                Use either the average encoding of the k nearest instances in a class (k_type='mean') or
                the k-nearest encoding in the class (k_type='point') to define the prototype of that class.
                Only relevant if an encoder is used to define the prototypes.
            threshold:
                Threshold level for the ratio between the distance of the counterfactual to the prototype of the
                predicted class for the original instance over the distance to the prototype of the predicted class
                for the counterfactual. If the trust score is below the threshold, the proposed counterfactual does
                not meet the requirements.
            verbose:
                Print intermediate results of optimization if True
            print_every:
                Print frequency if verbose is True
            log_every:
                Tensorboard log frequency if write directory is specified

        Returns:
            explanation, Dictionary containing the counterfactual with additional metadata
        """

        return self.cf_super.explain(
            explain_data,
            Y,
            target_class,
            k,
            k_type,
            threshold,
            verbose,
            print_every,
            log_every
        )
