import subprocess
import tempfile
from glob import glob


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)


def test():
    for f in glob('docs/examples/*.ipynb'):
        print('running',f)
        _exec_notebook(f)

if __name__ == '__main__':
    test()
