import os, pkg_resources, shutil


def create_tutorials(destination, silent = False):
    """
    Copies mercury.explainability tutorial notebooks to `destination`. A folder will be created inside
    destination, named 'explainability_tutorials'. The folder `destination` must exist.

    Args:
        destination (str): The destination directory
        silent (bool): If True, suppresses output on success.

    Raises:
        ValueError: If `destination` is equal to source path.

    Examples:
        >>> # copy tutorials to /tmp/explainability_tutorials
        >>> from mercury.explainability import create_tutorials
        >>> create_tutorials('/tmp')

    """
    src = pkg_resources.resource_filename(__package__, 'tutorials')
    dst = os.path.abspath(destination)

    assert src != dst, 'Destination (%s) cannot be the same as source.' % src

    assert os.path.isdir(dst), 'Destination (%s) must be a directory.' % dst

    dst = os.path.join(dst, 'explainability_tutorials')

    assert not os.path.exists(dst), 'Destination (%s) already exists' % dst

    shutil.copytree(src, dst)

    if not silent:
        print('Tutorials copied to: %s' % dst)
