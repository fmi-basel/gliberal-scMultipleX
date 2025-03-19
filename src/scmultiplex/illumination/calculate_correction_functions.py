# Copyright (C) 2025 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from lmfit.models import PolynomialModel
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, peak_widths, savgol_filter

logger = logging.getLogger(__name__)
matplotlib.use("Agg")


def normalize_array(np_array):
    # normalize to highest value, so that max value is 1
    return np_array / np.amax(np_array)


def compute_norm_percentile_over_z(img, percentile):
    """
    img is 3D numpy array. typically it has been multiplied by object mask to select object of interest.
    percentile is integer value 0-100, e.g. 80
    z_np is a numpy array of the z indeces in array (starts with 0, ends with shape(0)-1).
    q_np is a numpy array of the calculated percentile in the z-slice. Same length as z_np.
    """
    # loop over z-slices
    q = []
    z = []
    for i, plane in enumerate(img):
        plane_f = plane.flatten()
        # delete all 0 values in array (i.e. pixels that have been segmented away)
        plane_thresh = np.delete(plane_f, np.where(plane == 0))

        q.append(np.percentile(plane_thresh, q=percentile))
        z.append(i)

    # check that length of list matches number of z-slices
    if len(q) != img.shape[0]:
        raise ValueError(
            "Number of extracted values does not match image z-slice count. Check input image."
        )

    # check that z and q have the same length
    if len(q) != len(z):
        raise ValueError(
            "Number of percentile values extracted does not match number of z-slices."
        )

    z_np = np.array(z)
    # normalize array so that max intensity value is 1
    q_np = normalize_array(np.array(q))

    return z_np, q_np


def compute_smoothed_derivative(x, y):
    dy_dx_smooth = savgol_filter(
        y, window_length=12, polyorder=3, deriv=1, delta=x[1] - x[0]
    )
    return dy_dx_smooth


def find_curve_start(dy_dx):
    """
    Use first derivative curve of percentile vs. z-slice graph to identify z-index of where organoid starts.
    """
    # add pad to start of derivative curve so that peak is detected even if does not have decreasing values on either side
    pad_length = 1
    dy_dx_padded = np.pad(dy_dx, (pad_length, 0), mode="constant", constant_values=0)
    peaks, _ = find_peaks(dy_dx_padded)  # find index of peak (in derivative curve)

    # if no peaks found, use first z-slice index
    if len(peaks) == 0:
        start_i = 0
        logger.warning(
            "No starting peaks detected. Defaulting to first z-slice of array."
        )
        return int(start_i)

    # in case when multiple peaks are detected in derivate curve, identify index of highest peak
    index_max = np.argmax(dy_dx_padded[peaks])
    # determine width of selected peak
    # wlen = 40 limits search of peak to a 40-zslice window
    width = np.array(
        peak_widths(dy_dx_padded, [peaks[index_max]], rel_height=0.8, wlen=40)
    )
    height = dy_dx_padded[peaks[index_max]]

    # remove pad from peak indeces
    peaks = peaks - pad_length
    # remove pad from width left ips, right ips
    width[2] = width[2] - pad_length
    width[3] = width[3] - pad_length

    # if value of derivative curve at the peak position is greater than threshold value of 0.05, the peak is large
    # enough to qualify as a peak, and the starting index is selected at the right edge of the peak, determined
    # using the peak width
    if height > 0.05:
        start_i = int(
            np.ceil(width[3])
        )  # select right ips, round up and convert to integer
    else:
        start_i = 0
        logger.warning(
            "No large starting peaks detected. Defaulting to first z-slice of array"
        )

    return int(start_i), height, width, peaks, index_max


def find_curve_end(dy_dx, start_i=0):
    """
    Use first derivative curve of percentile vs. z-slice graph to identify z-index of where organoid ends.
    Best to remove the first identified peak (with find_curve_start) from input
    """

    # crop start to remove first peak
    dy_dx = dy_dx[start_i:]
    # add pad to end of derivative curve so that peak is detected even if does not have decreasing values on either side
    pad_length = 1
    dy_dx_padded = np.pad(dy_dx, (0, pad_length), mode="constant", constant_values=0)
    last_index = len(dy_dx) - 1 + pad_length

    # negate result to reflect across x-axis
    dy_dx_padded = -dy_dx_padded

    peaks, _ = find_peaks(dy_dx_padded)  # find index of peak (in derivative curve)

    # if no peaks found, use last z-slice index
    if len(peaks) == 0:
        end_i = len(dy_dx) - 1
        logger.warning("No end peaks detected. Defaulting to last z-slice of array.")
        return int(end_i)

    # if a peak has been detected within of the pad, remove it
    if any(p >= last_index for p in peaks):
        peaks = [
            p for p in peaks if p >= last_index
        ]  # Remove all occurrences of peaks within pad
        logger.warning("Peak detected within pad.")

    # in case when multiple peaks are detected in derivate curve, identify index of highest peak
    index_max = np.argmax(dy_dx_padded[peaks])
    # determine width of selected peak
    # wlen = 40 limits search of peak to a 40-zslice window
    width = np.array(
        peak_widths(dy_dx_padded, [peaks[index_max]], rel_height=0.6, wlen=40)
    )
    height = dy_dx_padded[peaks[index_max]]

    if height > 0.05:
        end_i = int(
            np.floor(width[2])
        )  # select left ips, round down and convert to integer to index
    else:
        end_i = len(dy_dx) - 1
        logger.warning(
            "No large end peaks detected. Defaulting to last z-slice of array."
        )

    # compensate for start_i crop
    peaks = peaks + start_i
    end_i = end_i + start_i
    # compensate for start_i crop in left ips, right ips
    width[2] = width[2] + start_i
    width[3] = width[3] + start_i

    return int(end_i), height, width, peaks, index_max


def ExponentialDecay(x, amplitude, decay, shift):
    """Model an exponential decay."""
    return (amplitude * np.exp(-x / decay)) + shift


def run_fits(model_list, x, y):
    """
    Run Polynomial, ExponentialDecayConcave, and ExponentialDecayConvex on input data
    Typically x is 1D numpy array of zslice number, y is the intensity per slices (e.g.
    computed with compute_percentile_over_z)
    Returns: (1) dictionary of models, where key is the the model name (str) and value is lmfit.Model object
    (2) dict of fit results, where key is the model name (str), and value is MinimizerResult from lmfit
    (3) dict of chi-squared values for each fit, where key is the model name (str), and value is chi-sqr (float)
    """
    models = {}
    results = {}
    chisqr = {}

    for m in model_list:
        if m == "Polynomial":

            mod = PolynomialModel(degree=4, prefix="poly_")
            pars = mod.guess(y, x=x)
            out = mod.fit(y, pars, x=x)

            models["Polynomial"] = mod
            results["Polynomial"] = out
            chisqr["Polynomial"] = out.chisqr

        elif m == "ExponentialDecayConcave":

            mod = Model(ExponentialDecay)

            # set concave (bulge)-shaped parameters
            mod.set_param_hint("decay", value=-50, max=0)
            mod.set_param_hint("amplitude", value=-0.05, max=0)
            mod.set_param_hint("shift", value=0.5)

            pars = mod.make_params()
            try:
                out = mod.fit(y, pars, x=x)
            except ValueError as e:
                logger.warning(
                    f"ValueError encountered for ExponentialDecayConcave fit: {e}"
                )
            else:
                models["ExponentialDecayConcave"] = mod
                results["ExponentialDecayConcave"] = out
                chisqr["ExponentialDecayConcave"] = out.chisqr

        elif m == "ExponentialDecayConvex":

            mod = Model(ExponentialDecay)

            # set convex (saddle)-shaped parameters
            mod.set_param_hint("decay", value=50, min=0)
            mod.set_param_hint("amplitude", value=1, min=0)
            mod.set_param_hint("shift", value=0.5)

            pars = mod.make_params()
            try:
                out = mod.fit(y, pars, x=x)
            except ValueError as e:
                logger.warning(
                    f"ValueError encountered for ExponentialDecayConvex fit: {e}"
                )
            else:
                models["ExponentialDecayConvex"] = mod
                results["ExponentialDecayConvex"] = out
                chisqr["ExponentialDecayConvex"] = out.chisqr

        else:
            raise ValueError(f"Model {m} does not exist")

    return models, results, chisqr


def select_best_fit(chisqr):
    """
    Select best fit (Polynomial, ExponentialDecayConcave, and ExponentialDecayConvex) based on lowest chisqr value
    Return the name of the fit as string (fit_name)
    """
    # Get the key with the lowest chi-squared value, which selects the best fit
    fit_name = min(chisqr, key=chisqr.get)
    return fit_name


def evaluate_model(model, result, x):
    y_pred = model.eval(params=result.params, x=x)
    return y_pred


def extend_eval(y_pred, x_pred, roi_start_z, full_z_count):
    y_pred_norm = normalize_array(y_pred)

    # find first and last values of correction curve
    start_intensity = y_pred_norm[0]
    end_intensity = y_pred_norm[-1]

    # add values to start of y_pred array
    start_repeat_count = x_pred[0] + roi_start_z
    new_start_values = np.full(start_repeat_count, start_intensity)

    # add values to start of y_pred array
    end_repeat_count = full_z_count - (x_pred[-1] + roi_start_z + 1)
    new_end_values = np.full(end_repeat_count, end_intensity)

    # generate extended y_pred_norm array
    y_full = np.concatenate((new_start_values, y_pred_norm, new_end_values))

    # generate x values (i.e. z-slice counts) for full array
    x_full = np.arange(0, full_z_count, 1)  # will have length == full_z_count

    if len(y_full) != len(x_full):
        raise ValueError("Array lengths do not match!")

    return x_full, y_full


def save_zillum_plots(
    x,
    y,
    dy_dx,
    dy_dx_smooth,
    start_i,
    start_peaks,
    start_width,
    start_index_max,
    start_height,
    end_i,
    end_peaks,
    end_height,
    end_width,
    end_index_max,
    results,
    sel_res,
    roi_start_z,
    x_fit,
    z_corr,
    i_corr,
    label_str,
    filepath,
):

    filename = f"{label_str}.pdf"
    filepath = os.path.join(filepath, filename)

    # Create a PdfPages object to save the plots
    with PdfPages(filepath) as pdf:

        # Plot 1
        plt.figure(figsize=(15, 8))  # Set figure size
        plt.plot(x, y, label="x_y")
        plt.plot(x, dy_dx, label="dy_dx")
        plt.plot(x, dy_dx_smooth, label="dy_dx_smooth")
        plt.plot(x[start_peaks], dy_dx_smooth[start_peaks], "x")
        plt.plot(x[start_i], y[start_i], "X", label="curve_start")
        plt.hlines(
            y=start_width[1], xmin=start_width[2], xmax=start_width[3], color="black"
        )
        plt.vlines(
            x=x[start_peaks[start_index_max]],
            ymin=start_width[1],
            ymax=start_height,
            color="black",
        )

        plt.title("curve start")
        plt.xlabel("z")
        plt.ylabel("normalized intensity")
        plt.legend()
        plt.grid(True)

        pdf.savefig()  # Save the current plot to the PDF
        plt.close()  # Close the plot

        # Plot 2
        plt.figure(figsize=(15, 8))  # Set figure size
        plt.plot(x, y, label="x_y")
        plt.plot(x, dy_dx, label="dy_dx")
        plt.plot(x, dy_dx_smooth, label="dy_dx_smooth")
        plt.plot(x[end_peaks], dy_dx_smooth[end_peaks], "x")
        plt.plot(x[end_i], y[end_i], "X", label="curve_end")
        plt.hlines(y=-end_width[1], xmin=end_width[2], xmax=end_width[3], color="black")
        plt.vlines(
            x=x[end_peaks[end_index_max]],
            ymin=-end_width[1],
            ymax=-end_height,
            color="black",
        )

        plt.title("curve end")
        plt.xlabel("z")
        plt.ylabel("normalized intensity")
        plt.legend()
        plt.grid(True)

        pdf.savefig()  # Save the current plot to the PDF
        plt.close()  # Close the plot

        # Plot 3
        plt.figure(figsize=(15, 8))  # Set figure size
        plt.plot(x + roi_start_z, y, ".", markersize=12, color="black", alpha=1)

        colors = [
            "darkorange",
            "olivedrab",
            "darkviolet",
            "blue",
            "brown",
        ]  # Define colors

        for i, (m, out) in enumerate(results.items()):
            cc = colors[i % len(colors)]

            plt.plot(
                x_fit + roi_start_z,
                out.init_fit,
                "--",
                linewidth=2,
                color=cc,
                label=f"{m}_init",
                alpha=0.4,
            )
            plt.plot(
                x_fit + roi_start_z,
                out.best_fit,
                "-",
                linewidth=2.5,
                color=cc,
                label=f"{m}_best",
                alpha=0.7,
            )

        plt.vlines(x=start_i + roi_start_z, ymin=0, ymax=1, color="red")
        plt.vlines(x=end_i + roi_start_z - 1, ymin=0, ymax=1, color="red")
        plt.plot(
            x_fit + roi_start_z,
            sel_res.best_fit,
            ".",
            markersize=2.5,
            color="red",
            label="selected",
            alpha=1,
        )
        plt.plot(z_corr, i_corr, label="final_correction")

        plt.title("model fits")
        plt.xlabel("z")
        plt.ylabel("normalized intensity")
        plt.legend()
        plt.grid(True)

        pdf.savefig()  # Save the current plot to the PDF
        plt.close()  # Close the plot

    return


def convert_xy_to_dict(x, y):
    """
    x, y are 1D numpy arrays. The elements of x (str) are set as keys in dictionary (d), and y are values
    """
    d = {}
    for xi, yi in zip(x, y):
        d[str(xi)] = yi
    return d


def calculate_correction(
    img, roi_start_z, full_z_count, label_str, filepath, percentile=80
):
    x, y = compute_norm_percentile_over_z(img, percentile=percentile)
    dy_dx = np.gradient(y, x)  # numerical derivative
    dy_dx_smooth = compute_smoothed_derivative(x, y)

    start_i, start_height, start_width, start_peaks, start_index_max = find_curve_start(
        dy_dx_smooth
    )
    end_i, end_height, end_width, end_peaks, end_index_max = find_curve_end(
        dy_dx_smooth, start_i
    )

    logger.info(f"Computed object start at index {start_i} and end at index {end_i}")

    if start_i > end_i:
        raise ValueError("End index must be higher than starting index.")

    if end_i - start_i < len(x) / 2:
        logger.warning(
            "Detected object spans less than half of z-extent of region of interest"
        )

    x_fit = x[start_i:end_i]
    y_fit = y[start_i:end_i]

    model_list = ["Polynomial", "ExponentialDecayConcave", "ExponentialDecayConvex"]

    models, results, chisqr = run_fits(model_list, x_fit, y_fit)

    fit_name = select_best_fit(chisqr)

    logger.info(f"Selecting {fit_name} model for z-correction.")

    sel_model = models[fit_name]
    sel_res = results[fit_name]

    y_pred = evaluate_model(sel_model, sel_res, x_fit)

    x_full, y_full = extend_eval(y_pred, x_fit, roi_start_z, full_z_count)

    row = convert_xy_to_dict(x_full, y_full)
    row.update({"label": label_str})

    save_zillum_plots(
        x,
        y,
        dy_dx,
        dy_dx_smooth,
        start_i,
        start_peaks,
        start_width,
        start_index_max,
        start_height,
        end_i,
        end_peaks,
        end_height,
        end_width,
        end_index_max,
        results,
        sel_res,
        roi_start_z,
        x_fit,
        x_full,
        y_full,
        label_str,
        filepath,
    )

    return row


def check_zillum_correction_table(adata, low_threshold, high_threshold):
    """
    Raise warnings if any values of anndata X matrix are less than or equal to a low_threshold value, or
    greater than a high_threshold value.
    This checks whether z-illumination matrix normalization and fits have run as expected.
    """

    low_indices = np.where(np.any(adata.X <= low_threshold, axis=1))[0]
    hi_indices = np.where(np.any(adata.X > high_threshold, axis=1))[0]

    # if any rows have values below threshold, raise a warning.
    if low_indices.size > 0:
        low_objects = adata.obs["label"].iloc[low_indices].tolist()

        logger.warning(
            f"Values lower than {low_threshold} detected in correction table, which may lead to over-correction "
            f"of objects. Check fits for objects {low_objects}."
        )

    if hi_indices.size > 0:
        raise ValueError(
            f"Value greater than {high_threshold} detected in matrix. Check normalization."
        )

    return
