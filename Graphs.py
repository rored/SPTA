from operator import attrgetter
from os.path import splitext, dirname, sep
from typing import List, Callable, Any, Tuple

from matplotlib.pyplot import plot, bar, xlabel, ylabel, xticks, title, yscale, clf, savefig, step, legend, gca
from numpy import histogram, array, exp, inf, sort, concatenate, arange, argmax, subtract, cumsum, multiply, ndarray
from scipy.optimize import curve_fit

from FileExtractor import FileResult

FitFunc = Callable[[Any], Any]
mult = 1000000000

# Fitting functions
def exponential_function(x: ndarray, a: float, b: float, c: float) -> float:
    """
    An exponential function that may be fitted to histogram data derived from TCSPC method in order to determine
    the fluorescence decay function of obtained FCS data.
    :param x: ndarray - numpy array of histogram values derived from TCSPC method
    :param a: float - a parameter in exponential function formula
    :param b: float - b parameter in exponential function formula, becomes a fluorescence lifetime in fluorescence decay
    function fitting
    :param c: float - c parameter in exponential function formula
    :return: an exponential function formula
    """
    return (a * exp(-x / b)) + c


def linear_function(x: ndarray, a: float, b: float) -> float:
    """
    A linear function that may be fitted to cumulative data derived from TCSPC method in order to determine
    countrate function
    :param x: ndarray - numpy array of cumulative values derived from TCSPC method
    :param a: float - a parameter in linear function formula
    :param b: float - b parameter in linear function formula
    :return: an linear function formula
    """
    return a * x + b


def _set_info(title_label: str, x_label: str="Time [ns]", y_label: str="Photon counts", show_legend: bool=True) -> None:
    """
    Function providing a description of generated plots.
    :param title_label: string - label of generated plot
    :param x_label: string - label of x axis
    :param y_label: string - label of y axis
    :param show_legend: boolean - set to True or False determines weather to show or not plot legend respectively
    :return:
    """
    xlabel(x_label)
    ylabel(y_label)
    title(title_label)
    if show_legend:
        legend()


# Help functions
def _save(file: FileResult, save_path: str, new_ext: str, histogram_plot: bool=False, art: List[Any]=None) -> None:
    """
    Function used to save generated plots.
    :param file: FileResult - contains data from file based on which plot was generated
    :param save_path: string - a path to catalogue where generated plot will be saved
    :param new_ext: string - provides a different file extension
    :param histogram_plot: bool - set to True or False determines respectively weather to save or not additional plot
    legend
    :param art: List - contains additional plot legend information
    :return:
    """
    art = art if art else []
    ext = splitext(file.file_name)[1]
    save_path = save_path if save_path else dirname(file.file_path)
    path = save_path + sep + file.file_name.replace(ext, new_ext)
    if histogram_plot:
        savefig(path, additional_artists=art, bbox_inches="tight")
    else:
        savefig(path)
    clf()


def _histogram(file: FileResult, bins: int) -> [ndarray, ndarray]:
    """
    Function creating basic histogram.
    :param file: FileResult - contains data based on which histogram is created
    :param bins: integer - number of bins that data is segregated to
    :return: [ndarray, ndarray] - a list containing two numpy arrays with x and y histogram values
    """
    count, edges = histogram([x.d_time for x in file.records], bins=bins)
    return array(edges[:len(count)]), array(count)


def create_single_histogram(file: FileResult, fit: bool=True, fit_func: FitFunc=exponential_function,
                            log_scale: bool=False, bins: int=100, save_path: str=None, bar_type: bool=True) -> None:
    """
    Function creating single histogram.
    :param bar_type: enum - defines histogram style - dots, bar or both (if both 2 separate histograms will be created)
    :param file: FileResult - contains data based on which histogram is created
    :param fit: bool - set to True or False determines respectively weather to fit or not given function to obtained
    data
    :param fit_func: FitFunc - function that will be fitted to obtained data if necessary
    :param log_scale: bool - set to True or False determines respectively weather to create a graph in logarythic scale
     or not
    :param bins: integer - number of bins that data is segregated to
    :param save_path: string - a path to catalogue where generated plot will be saved
    :return:
    """
    x, y = _histogram(file, bins)
    res_name = file.file_name
    res_name = res_name.replace('.dat', '')

    _set_info("Histogram of {}".format(res_name), show_legend=False)

    if bar_type:
        hist_line = bar(x, y, color=None, edgecolor='b', label='Experimental data')
        gca().add_artist(legend(handles=[hist_line]))

        if log_scale:
            yscale('log', nonposy='clip')

        leg = []
        if fit:
            max_idx = y.argmax()
            popt, _ = curve_fit(fit_func, x[max_idx:], y[max_idx:], p0=None, sigma=None, bounds=(0, inf), maxfev=1000000)
            fit_line, = plot(x[max_idx:], fit_func(x[max_idx:], *popt), 'r-', label='fit = %.2f * exp(-t / %.2f) + %.2f' %
                                                                                    tuple(popt))
            leg = [legend(handles=[fit_line], bbox_to_anchor=(-0.07, -0.3, 0., .102),
                          loc='center', ncol=1, mode="expand", borderaxespad=0.)]

        if log_scale:
            yscale('log', nonposy='clip')

        _save(file, save_path, '_histogram.png', histogram_plot=True, art=leg)

    else:
        hist_line, = plot(x, y, 'bo', label='Experimental data')
        gca().add_artist(legend(handles=[hist_line]))

        leg = []
        if fit:
            max_idx = y.argmax()
            popt, _ = curve_fit(fit_func, x[max_idx:], y[max_idx:], p0=None, sigma=None, bounds=(0, inf), maxfev=1000000)
            fit_line, = plot(x[max_idx:], fit_func(x[max_idx:], *popt), 'r-', label='fit = %.2f * exp(-t / %.2f) + %.2f' %
                                                                                    tuple(popt))
            leg = [legend(handles=[fit_line], bbox_to_anchor=(-0.07, -0.3, 0., .102),
                          loc='center', ncol=1, mode="expand", borderaxespad=0.)]

        if log_scale:
            yscale('log', nonposy='clip')

        _save(file, save_path, '_histogram.png', histogram_plot=True, art=leg)

def create_multiple_histograms(file: FileResult, background: FileResult, fit: bool = True,
                               fit_func: FitFunc = exponential_function, log_scale: bool = False, bins: int = 100,
                               save_path: str = None, bar_type: bool = False) -> None:

    """
    Function creating multiple histogram.
    :param bar_type:  enum - defines histogram style - dots, bar or both (if both 2 separate histograms will be created)
    :param file: FileResult - contains data file based on which histogram is created
    :param background: FileResult - contains data with background corresponding to given file based on which histogram
    is created
    :param fit: bool - set to True or False determines respectively weather to fit or not given function to obtained
    data
    :param fit_func:  FitFunc - function that will be fitted to obtained data if necessary
    :param log_scale: bool - set to True or False determines respectively weather to create a graph in logarythic scale
     or not
    :param bins: integer - number of bins that data is segregated to
    :param save_path: string - a path to catalogue where generated plot will be saved
    :return:
    """

    bg_name = background.file_name
    bg_name = bg_name.replace('.dat', '')

    x_bg, y_bg = _histogram(background, bins)
    x_res, y_res = _histogram(file, bins)

    res_name = file.file_name
    res_name = res_name.replace('.dat', '')

    _set_info("Histogram of {}".format(res_name), show_legend=False)

    if bar_type:
        ind = arange(len(x_bg))

        bg_max, res_max = max(y_bg), max(y_res)
        prop_y_res, prop_y_bg = y_res, y_bg
        if res_max > bg_max:
            sub = subtract(y_res, y_bg)
            add = [prop_y_res[i] if sub[i] < 0 else 0 for i in range(len(sub))]
        else:
            sub = subtract(y_bg, y_res)
            add = [prop_y_bg[i] if sub[i] < 0 else 0 for i in range(len(sub))]

        hist_line_res = bar(ind, prop_y_res, color=None, edgecolor='b', label='Experimental data')
        hist_line_bg = bar(ind, prop_y_bg, color=None, edgecolor='r', label='Background')
        bar(ind, add, color=('b' if res_max > bg_max else 'r'))
        gca().add_artist(legend(handles=[hist_line_res, hist_line_bg]))

        if fit:
            max_idx = y_res.argmax()
            popt, _ = curve_fit(fit_func, x_res[max_idx:], y_res[max_idx:], p0=None, sigma=None, bounds=(0, inf),
                                maxfev=1000000)
            fit_line, = plot(ind[max_idx:], fit_func(x_res[max_idx:], *popt), 'g-',
                             label='fit = %.2f * exp(-t / %.2f) + %.2f' %
                                   tuple(popt))
            leg = [legend(handles=[fit_line], bbox_to_anchor=(-0.07, -0.3, 0., .102),
                          loc='center', ncol=1, mode="expand", borderaxespad=0.)]

            if log_scale:
                yscale('log', nonposy='clip')

            breaks = int(bins / 20)
            labels = [x * 10 for x in range(breaks + 1)]
            locations = [x * 2 for x in labels]
            xticks(locations, labels)
            _save(file, save_path, ' and ' + bg_name + '_histograms_bar.png', histogram_plot=True, art=leg)

        else:
            if log_scale:
                yscale('log', nonposy='clip')

            breaks = int(bins / 20)
            labels = [x * 10 for x in range(breaks + 1)]
            locations = [x * 2 for x in labels]
            xticks(locations, labels)
            _save(file, save_path, ' and ' + bg_name + '_histograms_bar.png', histogram_plot=True,
                  art=[legend(handles=[hist_line_res, hist_line_bg])])

    else:
        hist_line_bg, = plot(x_bg, y_bg, 'ro', label='Photons count of ' + bg_name)
        hist_line_res, = plot(x_res, y_res, 'bo', label='Photons count of ' + res_name)
        gca().add_artist(legend(handles=[hist_line_res, hist_line_bg]))

        if fit:
            max_idx = y_res.argmax()
            popt, _ = curve_fit(fit_func, x_res[max_idx:], y_res[max_idx:], p0=None, sigma=None, bounds=(0, inf),
                                maxfev=1000000)
            fit_line, = plot(x_res[max_idx:], fit_func(x_res[max_idx:], *popt), 'g-',
                             label='fit = %.2f * exp(-t / %.2f) + %.2f' %
                                   tuple(popt))
            leg = [legend(handles=[fit_line], bbox_to_anchor=(-0.07, -0.3, 0., .102),
                          loc='center', ncol=1, mode="expand", borderaxespad=0.)]

            if log_scale:
                yscale('log', nonposy='clip')

            _save(file, save_path, ' and ' + bg_name + '_histograms_dots.png', histogram_plot=True, art=leg)

        else:
            if log_scale:
                yscale('log', nonposy='clip')

            _save(file, save_path, ' and ' + bg_name + '_histograms_dots.png', histogram_plot=True,
                  art=[legend(handles=[hist_line_res, hist_line_bg])])


def create_single_cumulative_graph(file: FileResult, log_scale: bool=False, save_path: str=None) -> None:
    """
    Function creating cumulative plot of analyzed fluorophore.
    :param file: FileResult - contains data based on which cumulative plot is created
    :param log_scale: bool - set to True or False determines respectively weather to create a graph in logarithmic scale
     or not
    :param save_path: string - a path to catalogue where generated plot will be saved
    :return:
    """
    data = sort([x.true_time for x in file.records])
    step(concatenate([data, data[[-1]]]), arange(data.size + 1), 'g-', label='Experimental data')

    res_name = file.file_name
    res_name = res_name.replace('.dat', '')
    _set_info('Cumulative plot of {}'.format(res_name))

    if log_scale:
        yscale('log', nonposy='clip')

    _save(file, save_path, '_cumulative.png')


def _cumulative_histogram(fit_func: FitFunc, t_time: List[float], bins: int, style: str,
                          label: str) -> [Any, Any, Any, Any]:
    """
    Function creating histogram based cumulative graph.
    :param fit_func: FitFunc - function that will be fitted to obtained data if necessary
    :param t_time: List[float] - list containing data values based on which plot will be created
    :param bins: integer - number of bins that data is segregated to
    :param style: string - determines plot style
    :param label: string - label of generated plot
    :return:
    """
    values, base = histogram(t_time, bins)
    cum = cumsum(values)
    line, = plot(base[:-1], cum, style, label=label)
    popt, s = curve_fit(fit_func, base[:-1], cum, p0=None, sigma=None, bounds=(0, inf), maxfev=10000)
    fit, = plot(base[:-1], fit_func(base[:-1], *popt), style[:-1], label='fit = %.2f * t + %.2f' % tuple(popt))
    return base, line, popt, fit


def _cumulative(file: FileResult, result_tt: List[float], bg_tt: List[float], prob_tt: List[float],
                min_prob: float=0.1, fit_func: FitFunc=linear_function, log_scale: bool=False, bins: int=1000000,
                save_path: str=None) -> None:
    """
    Function that creates multiple cumulative plots.
    :param file: FileResult - contains data from file based on which plot was generated
    :param result_tt: List[float] - list containing result ttime data values based on which plot will be created
    :param bg_tt: List[float] - list containing corresponding background ttime data values based on which plot will be
    created
    :param prob_tt: List[float] - list containing result ttime data values matching minimal probability criterion
    :param min_prob: float - defines minimum probability score based on which probability weighted cumulative plot is
    created
    :param fit_func: FitFunc - function that will be fitted to obtained data if necessary
    :param log_scale: log_scale: bool - set to True or False determines respectively weather to create a graph in
    logarithmic scale or not
    :param bins: integer - number of bins that data is segregated to
    :param save_path: string - a path to catalogue where generated plot will be saved
    :return:
    """

    # CUMULATIVE PLOT OF RESULT
    res_name = file.file_name
    res_name = res_name.replace('.dat', '')
    r_base, r_line, r_popt, r_fit = _cumulative_histogram(fit_func, result_tt, bins, 'g--',
                                                          '{} cumulative'.format(res_name))
    # CUMULATIVE PLOT OF BACKGROUND
    b_base, b_line, b_popt, b_fit = _cumulative_histogram(fit_func, bg_tt, bins, 'b--', 'background cumulative')

    # CUMULATIVE PLOT OF BACKGROUND REDUCES RESULT
    s_sub = subtract(r_popt, b_popt)
    s_fit, = plot(r_base[:-1], fit_func(r_base[:-1], *s_sub), 'r-', label='substraction fit: a={}, b={}'.format(*s_sub))

    # CUMULATIVE PLOT OF PROBABILITY WEIGHTED RESULT
    c_base, c_line, c_popt, c_fit = _cumulative_histogram(fit_func, prob_tt, bins, 'y--',
                                                          'probability cumulative of {}'.format(res_name))

    # CUMULATIVE PLOT OF RESULT
    res_name = file.file_name
    res_name = res_name.replace('.dat', '')
    r_base, r_line, r_popt, r_fit = _cumulative_histogram(fit_func, result_tt, bins, 'g--', 'Experimental data')
    r_lab, = plot(r_base[:-1], fit_func(r_base[:-1], *r_popt), 'g-',
                  label='fit = %.2f * t + %.2f' % (r_popt[0] * mult, r_popt[1]))

    # CUMULATIVE PLOT OF BACKGROUND
    b_base, b_line, b_popt, b_fit = _cumulative_histogram(fit_func, bg_tt, bins, 'b--', 'Background')
    b_lab, = plot(r_base[:-1], fit_func(r_base[:-1], *b_popt), 'b-',
                  label='fit = %.2f * t + %.2f' % (b_popt[0] * mult, b_popt[1]))

    # CUMULATIVE PLOT OF BACKGROUND REDUCES RESULT
    s_sub = subtract(r_popt, b_popt)
    s_fit, = plot(r_base[:-1], fit_func(r_base[:-1], *s_sub), 'r-', label='Substraction (Experimental - Background)')
    s_lab, = plot(r_base[:-1], fit_func(r_base[:-1], *s_sub), 'r-',
                  label='fit = %.2f * t + %.2f' % (s_sub[0] * mult, s_sub[1]))

    # CUMULATIVE PLOT OF PROBABILITY WEIGHTED RESULT
    c_base, c_line, c_popt, c_fit = _cumulative_histogram(fit_func, prob_tt, bins, 'y--', 'Probability (min_prob = '
                                                          + str(min_prob) + ')')
    c_lab, = plot(r_base[:-1], fit_func(c_base[:-1], *c_popt), 'y-',
                  label='fit = %.2f * t + %.2f' % (c_popt[0] * mult, c_popt[1]))

    gca().add_artist(legend(handles=[r_line, b_line, s_fit, c_line]))
    _set_info('Real countrates of {}'.format(file.file_name), show_legend=False)

    if log_scale:
        yscale('log', nonposy='clip')

    _save(file, save_path, '.countrate_prob_' + str(min_prob) + '.png', histogram_plot=True,
          art=[legend(handles=[r_lab, b_lab, s_lab, c_lab], bbox_to_anchor=(-0.07, -0.3, 0., .102),
                      loc='center', ncol=1, mode="expand", borderaxespad=0.)])


def normalize(y: ndarray, max_value: float) -> ndarray:
    """
    Performs normalization of given values
    :param y: ndarray - numpy array of histogram values derived from TCSPC method
    :param max_value: float - maximal value in given data
    :return: ndarray - numpy array of normalized histogram values derived from TCSPC method
    """
    return array([i/max_value for i in y])


def _plot_probability(file: FileResult, save_path: str, x: ndarray, y: ndarray, prob: ndarray, prob_type: str = '',
                      y_start: int = 0, bar_type=False, fit=False, fit_func: FitFunc = exponential_function) -> None:
    """
    Function creates probability distribution plot and weighted by probabilities histogram plot.
    :param file: FileResult - contains data from file based on which plot was generated
    :param save_path: string - a path to catalogue where generated plots will be saved
    :param x: ndarray - numpy array of x values
    :param y: ndarray - numpy array of y values
    :param y_start: int - starting index
    :param prob: ndarray - numpy array of probability distribution values
    :return:
    """

    res_name = file.file_name
    res_name = res_name.replace('.dat', '')
    # PROBABILITY DISTRIBUTION
    plot(x[y_start:], prob, '--', label='{}'.format(res_name))
    _set_info('Probability distribution of {}'.format(res_name), x_label='Time [ns]', y_label='Probability')
    _save(file, save_path, '_prob{}.png'.format(prob_type))

    # PROBABILITY WEIGHTED HISTOGRAM
    if bar_type:
        w_data_y = multiply(y[y_start:], prob)

        weight_hist_line = bar(x[y_start:], w_data_y, color=None, edgecolor='b',
                               label='{}'.format(res_name))
        gca().add_artist(legend(handles=[weight_hist_line]))

        leg = []
        if fit:
            max_idx = w_data_y.argmax()
            popt, _ = curve_fit(fit_func, x[max_idx:], w_data_y, p0=None, sigma=None, bounds=(0, inf),
                                maxfev=1000000)
            fit_line, = plot(x[max_idx:], fit_func(x[max_idx:], *popt), 'r-',
                             label='fit = %.2f * exp(-t / %.2f) + %.2f' %
                                   tuple(popt))
            leg = [legend(handles=[fit_line], bbox_to_anchor=(-0.07, -0.3, 0., .102),
                          loc='center', ncol=1, mode="expand", borderaxespad=0.)]

        _set_info('Weighted countrate of {}'.format(res_name))
        _save(file, save_path, '_weighted{}.png'.format(prob_type), histogram_plot=True, art=leg)

    else:
        w_data_y = multiply(y[y_start:], prob)
        plot(x[y_start:], w_data_y, 'bo', label='{}'.format(res_name))
        if fit:
            max_idx = w_data_y.argmax()
            popt, _ = curve_fit(fit_func, x[max_idx:], w_data_y, p0=None, sigma=None, bounds=(0, inf),
                                maxfev=1000000)
            fit_line, = plot(x[max_idx:], fit_func(x[max_idx:], *popt), 'r-',
                             label='fit = %.2f * exp(-t / %.2f) + %.2f' %
                                   tuple(popt))
            leg = [legend(handles=[fit_line], bbox_to_anchor=(-0.07, -0.3, 0., .102),
                          loc='center', ncol=1, mode="expand", borderaxespad=0.)]
            _set_info('Weighted countrate of {}'.format(res_name))
            _save(file, save_path, '_weighted{}.png'.format(prob_type), histogram_plot=True, art=leg)
        else:
            _set_info('Weighted countrate of {}'.format(res_name))
            _save(file, save_path, '_weighted{}.png'.format(prob_type), histogram_plot=True)


def filter_by_probability(records: List[Tuple[float, float]], min_prob: float, x: ndarray, y: ndarray):
    """
    Function filters result ttime data by given minimal probability criterion.
    :param records: List[Tuple[float, float]]
    :param min_prob: float - defines minimum probability score based on which probability weighted cumulative plot is
    created
    :param x: ndarray - numpy array of x values (result ttimes)
    :param y: ndarray - numpy array of y values (probability distribution)
    :return: List - list of result ttimes matching given minimal probability criterion
    """
    sets, to_add = [], ()
    for i in range(len(y)):
        val = y[i]
        if val >= min_prob and len(to_add) == 0:
            to_add = (x[i],)
        elif val < min_prob and len(to_add) > 0:
            sets.append(to_add + (x[i - 1],))
            to_add = ()

    if len(to_add) > 0:
        sets.append(to_add + (x[len(y) - 1],))

    result, last_index = [], 0
    sort_result = sorted(records, key=attrgetter('d_time'))
    for tup in sets:
        for i in range(last_index, len(sort_result)-1):
            time = sort_result[i].d_time
            if tup[0] <= time <= tup[1]:
                result.append(sort_result[i].true_time)
            elif time > tup[1]:
                last_index = i
                break

    return result


def create_group_graphs(background: FileResult, files: List[FileResult], bins: int=100, save_path: str=None,
                        min_prob: float=0.1) -> None:
    """
    Function creating graphs with multiple substances.
    :param min_prob: float - defines minimum probability score based on which probability weighted cumulative plot is
    created
    :param background: FileResult - contains data from background file
    :param files: List[FileResult] - contains all files corresponding to background file
    :param bins: integer - number of bins that data is segregated to
    :param save_path:  string - a path to catalogue where generated plots will be saved
    :return:
    """
    min_probs = [0.05, 0.14, 0.36]
    bg_tt = [x.true_time for x in background.records]
    x, y_bg = _histogram(background, bins)
    idx_bg_max = argmax(y_bg)

    for file in files:
        # MULTIPLE HISTOGRAMS
        create_multiple_histograms(file, background, save_path=save_path)

        for mp in min_probs:
            # PROBABILITY DISTRIBUTION
            x, y_result = _histogram(file, bins)
            y_result_bg_reduced = y_result

            idx_result_max = argmax(y_result)
            y_max_value = y_result[idx_result_max]
            idx_result_max = 0
            idx_bg_max = 0

            if (idx_result_max - idx_bg_max) >= 0:
                bg_avg = y_bg[0]
                for i in range(0, (idx_result_max - idx_bg_max)):
                    y_result_bg_reduced[i] = y_result_bg_reduced[i] - bg_avg

                for i in range((idx_result_max - idx_bg_max), len(y_result)):
                    y_result_bg_reduced[i] = y_result_bg_reduced[i] - y_bg[i - (idx_result_max - idx_bg_max)]
            else:
                bg_avg = y_bg[len(y_bg) - 1]
                for i in range(0, len(y_result) + (idx_result_max - idx_bg_max)):
                    y_result_bg_reduced[i] = y_result_bg_reduced[i] - y_bg[i - (idx_result_max - idx_bg_max)]
                for i in range(len(y_result) + (idx_result_max - idx_bg_max), len(y_result)):
                    y_result_bg_reduced[i] = y_result_bg_reduced[i] - bg_avg

            prob = normalize(y_result_bg_reduced, y_max_value)
            prob_tt = filter_by_probability(file.records, mp, x, prob)
            _plot_probability(file, save_path, x, y_result, prob)

            # MULTIPLE CUMULATIVE PLOTS
            res_tt = [x.true_time for x in file.records]
            _cumulative(file, res_tt, bg_tt, prob_tt, mp, save_path=save_path)

