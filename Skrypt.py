import gc
import os

from Parser import parse
from FileExtractor import Record, extract_groups, extract_file
from Graphs import create_group_graphs, create_single_cumulative_graph, create_single_histogram


def filter_func(record: Record) -> bool:
    """
    Function for filtering necessary data
    :param record:  Record - structure containing file parameters
    :return: filtering criteria
    """
    return record.type == 'CHN' and record.channel == '2'


def start(path: str, save_path: str, background: str, group_key: str, min_prob: float=0.1) -> None:
    """
    Function for generating plots. Find out more on how to create regular expressions here:
    https://docs.python.org/2/library/re.html
    :param path: string - a path to catalogue where derived files are stored
    :param save_path: string - a path to catalogue where generated files and plots will be saved
    :param background: string - a regex (regular expression) based on which a background file complementary files is
    identified
    :param group_key: string - a regex (regular expression) based on which complementary files are identified
    :param min_prob: float - defines minimum probability score based on which probability weighted cumulative plot is
    created
    :return:
    """
    print("Start")
    parse(path=path)
    result = extract_groups(path=path, group=group_key, group_background=background)

    for i in result.groups:
        group = result.groups[i]
        background = extract_file([x for x in group if x.is_background][0], filter_func)
        rest = [x for x in group if not x.is_background]

        for j in range(len(rest)):
            file = extract_file(rest[j], filter_func)
            print(file.file_name)
            # SINGLE PLOTS
            create_single_histogram(file, save_path=save_path)
            create_single_cumulative_graph(file, save_path=save_path)

        # MULTIPLE PLOTS
        create_group_graphs(background, rest, min_prob, save_path=save_path)

        # free memory
        result.groups[i] = []
        gc.collect()

if __name__ == '__main__':
    cwd = os.getcwd()
    path = cwd
    save_path = cwd

    start(path, save_path, 'background', '.*(dat)')