import os, shutil

from mercury.explainability.create_tutorials import create_tutorials


def test_create_tutorials():
    # Silently remove the full tree './explainability_tutorials/'
    shutil.rmtree('./explainability_tutorials', ignore_errors = True)

    create_tutorials('./')

    assert os.path.isfile('./explainability_tutorials/BasicTutorial.ipynb')

    # Clean up
    shutil.rmtree('./explainability_tutorials', ignore_errors = True)


if __name__ == "__main__":
    test_create_tutorials()
