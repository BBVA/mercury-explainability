import mercury.explainability.explainers._tree_splitters as tree


def test_public_api_surface():
    # the names should be re-exported from the Cython module
    assert callable(tree.get_min_mistakes_cut)
    assert callable(tree.get_min_surrogate_cut)

    # Some compiled modules may not expose __all__; validate it only when present.
    if hasattr(tree, "__all__"):
        assert sorted(tree.__all__) == [
            "get_min_mistakes_cut",
            "get_min_surrogate_cut",
        ]


if __name__ == "__main__":
	test_public_api_surface()
