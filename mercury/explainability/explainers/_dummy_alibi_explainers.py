# This module contains dummy versions of Explainers with Alibi dependencies.
# This classes are imported in the __init__.py if the import of the original class fails
# If the user tries to import any of the classes, and error is raised indicating that alibi
# must be installed first

class _DummyAlibiExplainer:
    """
    Class which raises and error if instantiated
    """
    def __init__(self):
        raise ModuleNotFoundError("You need to install alibi library to use this explainer.")

class ALEExplainer(_DummyAlibiExplainer):

    def __init__(self, predictor, target_names):
        super().__init__()

class AnchorsWithImportanceExplainer(_DummyAlibiExplainer):

    def __init__(self, predict_fn=None, train_data=None, categorical_names=None, disc_perc=None):
        super().__init__()

class CounterfactualExplainer(_DummyAlibiExplainer):

    def __init__(
        self,
        predict_fn=None,
        feature_names=None,
        shape=None,
        drop_features=None,
        distance_fn=None,
        target_proba=None,
        target_class=None,
        max_iter=None,
        early_stop=None,
        lam_init=None,
        max_lam_steps=None,
        tol=None,
        learning_rate_init=None,
        feature_range=None,
        eps=None,
        init=None,
        decay=None,
        write_dir=None,
        debug=None,
        sess=None
    ):
        super().__init__()

class CounterfactualProtoExplainer(_DummyAlibiExplainer):

    def __init__(
        self,
        predict_fn=None,
        train_data=None,
        feature_names=None,
        shape=None,
        drop_features=None,
        kappa=None,
        beta=None,
        feature_range=None,
        gamma=None,
        ae_model=None,
        enc_model=None,
        theta=None,
        cat_vars=None,
        ohe=None,
        use_kdtree=None,
        learning_rate_init=None,
        max_iterations=None,
        c_init=None,
        c_steps=None,
        eps=None,
        clip=None,
        update_num_grad=None,
        write_dir=None,
        sess=None,
        trustscore_kwargs=None,
        d_type=None,
        w=None,
        disc_perc=None,
        standardize_cat_vars=None,
        smooth=None,
        center=None,
        update_feature_range=None
    ):
        super().__init__()
