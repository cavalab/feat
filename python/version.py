# Source: https://github.com/Changaco/version.py

from os.path import dirname, isdir, join
import re
from subprocess import CalledProcessError, check_output


PREFIX = ''

tag_re = re.compile(r'\btag: %s([0-9][^,]*)\b' % PREFIX)
version_re = re.compile('^Version: (.+)$', re.M)


def get_version(write=False):
    # Return the version if it has been injected into the file by git-archive
    try:
        version = tag_re.search('$Format:%D$')
        if version:
            return version.group(1)

        d = dirname(__file__) + '/../'

        if isdir(join(d, '.git')):
            # Get the version using "git describe".
            cmd = 'git describe --tags --match %s[0-9]* --dirty' % PREFIX
            try:
                version = check_output(cmd.split()).decode().strip()[len(PREFIX):]
            except CalledProcessError:
                raise RuntimeError('Unable to get version number from git tags')

            # PEP 440 compatibility
            if '-' in version:
                if version.endswith('-dirty'):
                    raise RuntimeError('The working tree is dirty')
                version = '.post'.join(version.split('-')[:2])

        else:
            # Extract the version from the PKG-INFO file.
            with open(join(d, 'PKG-INFO')) as f:
                version = version_re.search(f.read()).group(1)

        # write version
        if write:
            with open('versionstr.py','w') as wf:
                wf.write('__version__="{}"'.format(version))

        return version
    except Exception as e:
        print('Version error:',e)
        print(
        'Unable to get version number from git tags. Not updating version.')
        from versionstr import __version__
        return __version__



if __name__ == '__main__':
    print(get_version())
