import glob
import os
import re
from decimal import Decimal
from typing import Callable, List

TimeConvertFunc = Callable[[float], float]


# Structures proving an optimal data storage
class Record:
    def __init__(self, time_convert_func: TimeConvertFunc, r_id: str, r_type: str, channel: str,
                 n_sync: str=None, true_time: str=None, d_time: str=None) -> None:
        self.id = int(r_id) if r_id else None
        self.type = r_type
        self.channel = channel
        self.n_sync = float(n_sync) if n_sync else None
        self.true_time = float(true_time) if true_time else None
        self.d_time = time_convert_func(float(d_time)) if d_time else None

    def __str__(self) -> str:
        return '{} {} {} {} {} {}'.format(self.id, self.type, self.channel, self.n_sync, self.true_time, self.d_time)

    def __repr__(self) -> str:
        return str(self)

FilterFunc = Callable[[Record], bool]


class FileResult:
    def __init__(self, path: str, is_background: bool, headers: List[str]=None, records: List[Record]=None) -> None:
        self.file_path = path
        self.file_name = os.path.basename(path)
        self.is_background = is_background
        self.headers = headers if headers else []
        self.records = records if records else []

    def __str__(self) -> str:
        return 'Name: {}\n is_background: {}\n\n {}'.format(self.file_name, self.is_background, str(self.records))

    def __repr__(self) -> str:
        return str(self)


class Result:
    def __init__(self) -> None:
        self.groups = {}

    def add_group(self, name: str, file: FileResult):
        if name not in self.groups:
            self.groups[name] = []
        self.groups[name].append(file)

    def __str__(self) -> str:
        return str(self.groups)

    def __repr__(self) -> str:
        return str(self)


def bit_converter(value: float, bin_size: float=3125, laser_pulse_time: float=50) -> float:
    """
    This function converts values from column dtime from bit format to floats corresponding to ns time unit.
    :param value: float - bit value of time from dtime column
    :param bin_size: float - parameter od conversion given in ptu file specification
    :param laser_pulse_time: float - time between subsequent laser pulses
    :return: float - converted value of time corresponding to ns time units
    """
    return float(Decimal(value) / (Decimal(bin_size) / Decimal(laser_pulse_time)))


def extract_groups(path: str=os.getcwd(), group: str='.*(dat)', group_background: str='background') -> Result:
    """
    Function providing aggregation of data from files. Find out more on how to create regular expressions here:
    https://docs.python.org/2/library/re.html
    :param path: string - a path to catalogue containing files
    :param group: string - a regex (regular expression) based on which complementary files are identified
    :param group_background: string - a regex (regular expression) based on which a background file complementary files
    is identified
    :return: Result - an object containing grouped data from the files
    """
    group_regex = re.compile(group)

    result = Result()
    for file in glob.glob(path):
        group_name = group_regex.match(os.path.basename(file))
        if not group_name or (group_name and not group_name.group(1)):
            raise FileNotFoundError('group in file not found or group_name does not contain group regex reference')

        result.add_group(group_name.group(1), FileResult(file, file.find(group_background) != -1))

    return result


def extract_file(file: FileResult, filter_func: FilterFunc=None,
                 time_convert_func: TimeConvertFunc=bit_converter) -> FileResult:
    """
    Saves necessary data obtained from file.
    :param file: FileResult - object containing basic information about given file
    :param filter_func: FilterFunction - function determining which information are necessary
    :param time_convert_func: TimeConvertFunc - function providing units conversion if needed
    :return: FileResult - object containing extracted necessary data from given file
    """
    with open(file.file_path, "r") as f:
        counter = 0
        for line in f:
            line = line.split()
            if counter < 152:
                file.headers.append(line)
                counter += 1
            else:
                record = Record(time_convert_func, *line)
                if not filter_func or (filter_func and filter_func(record)):
                    file.records.append(record)
                else:
                    del record
    return file
