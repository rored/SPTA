import glob
import os
import subprocess


# convert .ptu files to .dat
def parse(set_path=True, path=''):
    """
    .ptu file parser. Having a wineconsole and ptu_export.exe is necessary. You may download wineconsole here:
    http://manpages.ubuntu.com/manpages/trusty/man1/wineconsole.1.html
    You may download ptu_export.exe file here:

    Unfortunately you need to press any key after finished wine process as it is what wine requires.
    :param set_path: - set to True or False if you respectively do or do not want set the path to the catalogue
    containing .ptu files yourself
    :param path: string - a path to catalogue containing .ptu files
    :return:
    """

    # sets path
    if not set_path:
        #tochange
        work_path = os.getcwd()
    else:
        work_path = path

    for file in glob.glob(work_path):
        output = os.path.basename(file)
        output = output.replace('ptu', 'dat')
        command = ['wineconsole', 'ptu_export.exe', file, output]
        subprocess.call(command)