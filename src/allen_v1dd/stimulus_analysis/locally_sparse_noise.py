from os import path

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np
import scipy.optimize as opt


from .stimulus_analysis import StimulusAnalysis


def get_imshow_extent(azimuths, altitudes):
    return [azimuths[0], azimuths[-1], altitudes[0], altitudes[-1]]


class LocallySparseNoise(StimulusAnalysis):
    """Used to analyze the locally sparse noise stimulus.

    1. Does a cell have receptive field?
    2. What is a cell's receptive field?
    """

    def __init__(self, session, plane, trace_type="dff"):
        super().__init__("locally_sparse_noise", "lsn", session, plane, trace_type)

        self.authors = "Chase King, Naomi Donovan"

        if trace_type == "dff":
            self.baseline_time_window = (
                -1,
                0,
            )  # Baseline is the 1 sec window before (note this spans 3 different stimulus frames)
            self.response_time_window = (
                0,
                4 * self.time_per_frame,
            )  # Response is 0.67 sec window after (note this spans the desired stimulus along with the next one)
            # self.response_time_window = (0, 1) # Response is 0.67 sec window after (note this spans the desired stimulus along with the next one)
        else:
            self.baseline_time_window = None
            # self.response_time_window = (0, 2*self.time_per_frame) # 0.33 sec stimulus duration
            self.response_time_window = (
                0,
                2 * self.time_per_frame,
            )  # Span to the next stimulus

        self.pixel_on = 255
        self.pixel_off = 0
        self.pixel_gray = 127

        self.n_shuffles = 10000
        self.frac_sig_trials_thresh = 0.25
        self.response_thresh_alpha = (
            0.01  # Significance level for determining RFs based on p-values
        )

        self._frame_images = None
        self._sweep_responses = None
        self._design_matrix = None
        self._trial_template = None
        self._receptive_fields = None
        self._rf_centers = None
        self._imshow_extent = None
        self._stas = None
        self._shuffled_stas = None
        self._r2s = None
        self._shuffled_r2s = None
        # self._pvals = None
        # self._receptive_fields_sta = None

        self.whiten = True

    def save_to_h5(self, group):
        super().save_to_h5(group)

        group.attrs["trace_type"] = self.trace_type
        group.attrs["frac_sig_trials_thresh"] = self.frac_sig_trials_thresh
        group.attrs["image_shape"] = self.image_shape
        group.attrs["azimuths"] = self.azimuths
        group.attrs["altitudes"] = self.altitudes
        group.attrs["grid_size"] = self.grid_size

        # Is responsive
        is_responsive = np.zeros((self.n_rois, 3), dtype=bool)
        for roi in range(self.n_rois):
            if self.is_roi_valid[roi]:
                resp_on = self.has_receptive_field(roi, rf_type="on")
                resp_off = self.has_receptive_field(roi, rf_type="off")
                is_responsive[roi, :] = [resp_on, resp_off, resp_on or resp_off]
        ds = group.create_dataset("is_responsive", data=is_responsive)
        ds.attrs["columns"] = ["has_rf_on", "has_rf_off", "has_rf_on_or_off"]

        # Receptive fields
        ds = group.create_dataset("receptive_fields", data=self.receptive_fields)
        ds.attrs["dimensions"] = ["roi", "on_off", "row", "column"]

        # RF centers
        ds = group.create_dataset("rf_centers", data=self.rf_centers)
        ds.attrs["dimensions"] = [
            "roi",
            "on (0) and off (1)",
            "azimuth (0) and altitude (1) (deg)",
        ]

        # RF centers (computed by argmax)
        rf_centers_argmax = np.full((self.n_rois, 2, 2), np.nan)
        rois, onoffs, alts, azis = np.where(self.receptive_fields)
        for roi in np.unique(rois):
            for onoff in np.unique(onoffs[rois == roi]):
                rf = self.receptive_fields[roi, onoff]
                azi, alt = np.unravel_index(rf.argmax(), rf.shape)  # note the ordering
                alt, azi = self.point_to_alt_azi(
                    alt_ctr=alt + 0.5, azi_ctr=azi + 0.5
                )  # Add 0.5 to center in pixel
                rf_centers_argmax[roi, onoff, :] = (azi, alt)
        ds = group.create_dataset("rf_centers_argmax", data=rf_centers_argmax)
        ds.attrs["dimensions"] = [
            "roi",
            "on (0) and off (1)",
            "azimuth (0) and altitude (1) (deg)",
        ]

    @property
    def duration(self):
        return self.stim_meta["duration"]

    @property
    def grid_size(self):
        return self.stim_meta["grid_size"]

    @property
    def frame_images(self):
        if self._frame_images is None:
            self._load_frames()
        return self._frame_images

    @property
    def azimuths(self):
        if self._azimuths is None:
            self._load_frames()
        return self._azimuths

    @property
    def altitudes(self):
        if self._altitudes is None:
            self._load_frames()
        return self._altitudes

    @property
    def imshow_extent(self):
        if self._imshow_extent is None:
            self._imshow_extent = get_imshow_extent(
                azimuths=self.azimuths, altitudes=self.altitudes
            )
        return self._imshow_extent

    @property
    def trial_template(self):
        if self._trial_template is None:
            self._trial_template = self.frame_images[self.stim_table.frame.values]
        return self._trial_template

    @property
    def n_sweeps(self):
        return len(self.stim_table)

    @property
    def n_pixels(self):
        return self.frame_images.shape[1] * self.frame_images.shape[2]

    @property
    def image_shape(self):
        return self.frame_images.shape[1], self.frame_images.shape[2]

    @property
    def design_matrix(self):
        """
        The design matrix is a matrix of shape (2*n_pixels, n_sweeps) that contains
        information about what pixels are ON or OFF in different stimulus showings.
        A stimulus sweep is a particular showing of a stimulus (that need not be unique).

        Specifically, the first half of rows are indexed by (i, j) where i is 1 iff pixel i is ON in stimulus condition j.
        The second half of rows are indexed by (n_pixels+i, j) where i is 1 iff pixel i is OFF in stimulus condition j.

        Returns:
            np.ndarray: Of shape (2*n_pixels, n_sweeps)
        """
        if self._design_matrix is None:
            stim_pixels = self.frame_images[self.stim_table.frame.values].reshape(
                (-1, self.n_pixels)
            )
            design_matrix_on = np.where(stim_pixels == self.pixel_on, True, False)
            design_matrix_off = np.where(stim_pixels == self.pixel_off, True, False)
            self._design_matrix = np.concatenate(
                (design_matrix_on, design_matrix_off), axis=1
            )
            self._design_matrix = self._design_matrix.T  # shape (2*n_pixels, n_sweeps)

        return self._design_matrix

    # def convolve(self, img, sigma=4):
    #     """
    #     2D Gaussian convolution.

    #     Copied from https://github.com/AllenInstitute/AllenSDK/blob/9ef5214dcb04a61fe4c04bf19a5cb13c9e1b03f1/allensdk/brain_observatory/receptive_field_analysis/utilities.py#L56
    #     """
    #     from scipy.interpolate import interp2d
    #     from scipy.ndimage import gaussian_filter
    #     from skimage.measure import block_reduce

    #     if img.sum() == 0:
    #         return img

    #     img_pad = np.zeros((3 * img.shape[0], 3 * img.shape[1]))
    #     img_pad[img.shape[0]:2 * img.shape[0], img.shape[1]:2 * img.shape[1]] = img

    #     x = np.arange(3 * img.shape[0])
    #     y = np.arange(3 * img.shape[1])
    #     g = interp2d(y, x, img_pad, kind='linear')

    #     if img.shape[0] == 16:
    #         upsample = 4
    #         offset = -(1 - .625)
    #     elif img.shape[0] == 8:
    #         upsample = 8
    #         offset = -(1 - .5625)
    #     else:
    #         raise NotImplementedError

    #     ZZ_on = g(offset + np.arange(0, img.shape[1] * 3, 1. / upsample), offset + np.arange(0, img.shape[0] * 3, 1. / upsample))
    #     ZZ_on_f = gaussian_filter(ZZ_on, float(sigma), mode='constant')

    #     z_on_new = block_reduce(ZZ_on_f, (upsample, upsample))
    #     z_on_new = z_on_new / z_on_new.sum() * img.sum()
    #     z_on_new = z_on_new[img.shape[0]:2 * img.shape[0], img.shape[1]:2 * img.shape[1]]

    #     return z_on_new

    @property
    def design_matrix_blur(self):
        # TODO
        # for stim_condition_index in range(design_matrix.shape[1]):
        #     design_matrix[:lsn.n_pixels, stim_condition_index] = convolve(design_matrix[:lsn.n_pixels, stim_condition_index].reshape(lsn.image_shape)).flatten()
        #     design_matrix[lsn.n_pixels:, stim_condition_index] = convolve(design_matrix[lsn.n_pixels:, stim_condition_index].reshape(lsn.image_shape)).flatten()
        return None

    @property
    def sweep_responses(self):
        """
        Sweep responses is a np.ndarray of shape (n_stim_showings, n_rois) where the value at
        position (i, j) is the jth ROI's response to the ith stimulus shown.
        """

        if self._sweep_responses is None:
            self._sweep_responses = np.zeros(
                (len(self.stim_table), self.n_rois), dtype=float
            )
            for i in self.stim_table.index:
                start = self.stim_table.at[i, "start"]
                self._sweep_responses[i] = self.get_responses(
                    start,
                    self.baseline_time_window,
                    self.response_time_window,
                    self.trace_type,
                )

        return self._sweep_responses

    # @property
    # def receptive_fields(self):
    #     """
    #     Array of shape (n_rois, 2, n_image_rows, n_image_columns) where each entry is the fraction of significant responses
    #     at each pixel. Dimension 1 corresponds to ON (0) and OFF (1). Values less than self.frac_sig_trials are set to zero.
    #     """
    #     if self._receptive_fields is None:
    #         design_matrix_int = self.design_matrix.astype(
    #             int
    #         )  # shape (2*n_pixels, n_sweeps)
    #         n_pixel_trials = self.design_matrix.sum(axis=1)  # shape (2*n_pixels,)
    #         roi_boot_95 = np.quantile(
    #             self.get_spont_null_dist(
    #                 self.baseline_time_window,
    #                 self.response_time_window,
    #                 n_boot=self.n_shuffles,
    #                 trace_type=self.trace_type,
    #                 cache=False,
    #             ),
    #             0.95,
    #             axis=1,
    #         )  # shape (n_rois,)
    #         sig_sweep_responses = (
    #             self.sweep_responses > roi_boot_95
    #         )  # shape (n_sweeps, n_rois)
    #         frac_sig_pixel_responses = (
    #             design_matrix_int.dot(sig_sweep_responses).T / n_pixel_trials
    #         )  # shape (n_rois, 2*n_pixels)
    #         frac_sig_pixel_responses[
    #             frac_sig_pixel_responses < self.frac_sig_trials_thresh
    #         ] = 0  # Zero out pixels below significance threshold
    #         frac_sig_pixel_responses[~self.is_roi_valid] = 0  # Zero out invalid ROIs
    #         self._receptive_fields = frac_sig_pixel_responses.reshape(
    #             self.n_rois, 2, *self.image_shape
    #         )

    #     return self._receptive_fields

    @property
    def receptive_fields(self):
        """
        Array of shape (n_rois, 2, n_image_rows, n_image_columns) where each entry is the value of the fitted 2D Gaussian kernel at
        each pixel. Dimension 1 corresponds to ON (0) and OFF (1). Gaussian fits that don't converge are set to zero. ROIs that aren't
        valid are set to zero.
        """

        if self._receptive_fields is None:

            self._receptive_fields = np.zeros(
                (self.n_rois, 2, *self.image_shape), dtype=float
            )
            self._r2s = np.zeros((self.n_rois, 2), dtype=float)
            self._shuffled_r2s = np.zeros((self.n_rois, 2), dtype=float)

            stas = self.get_spike_triggered_averages(shuffle=False)
            shuffled_stas = self.get_spike_triggered_averages(shuffle=True)

            for roi in range(self.n_rois):

                on_sta = stas[roi, 0, :, :]
                off_sta = stas[roi, 1, :, :]

                popt_on, pcov_on, fitted_on, r2_on = self.fit_gaussian(
                    on_sta, type="on"
                )
                popt_off, pcov_off, fitted_off, r2_off = self.fit_gaussian(
                    off_sta, type="off"
                )

                self._receptive_fields[roi, 0, :, :] = (
                    fitted_on if r2_on is not None else np.zeros_like(on_sta)
                )
                self._receptive_fields[roi, 1, :, :] = (
                    fitted_off if r2_off is not None else np.zeros_like(off_sta)
                )
                self._r2s[roi, 0] = r2_on if r2_on is not None else 0
                self._r2s[roi, 1] = r2_off if r2_off is not None else 0

                # for shuffled distribution significance
                on_sta_shuff = shuffled_stas[roi, 0, :, :]
                off_sta_shuff = shuffled_stas[roi, 1, :, :]

                popt_on_shuff, pcov_on_shuff, fitted_on_shuff, r2_on_shuff = (
                    self.fit_gaussian(on_sta_shuff, type="on")
                )
                popt_off_shuff, pcov_off_shuff, fitted_off_shuff, r2_off_shuff = (
                    self.fit_gaussian(off_sta_shuff, type="off")
                )

                self._shuffled_r2s[roi, 0] = (
                    r2_on_shuff if r2_on_shuff is not None else 0
                )
                self._shuffled_r2s[roi, 1] = (
                    r2_off_shuff if r2_off_shuff is not None else 0
                )

            ## Zero out invalid ROIs
            self._receptive_fields[~self.is_roi_valid, :, :, :] = 0

            ## Zero out ROIs where fit is poor based on shuffled distribution
            r2_on_threshold = np.percentile(self._shuffled_r2s[:, 0], 95)
            r2_off_threshold = np.percentile(self._shuffled_r2s[:, 1], 95)
            self._receptive_fields[self._r2s[:, 0] < r2_on_threshold, 0, :, :] = 0
            self._receptive_fields[self._r2s[:, 1] < r2_off_threshold, 1, :, :] = 0

        return self._receptive_fields

    def get_spike_triggered_averages(self, shuffle=False):
        """
        Array of shape (n_rois, 2, n_image_rows, n_image_columns) where dimension 1 corresponds to ON (0) and OFF (1).
        Returns the spike-triggered averages for each ROI and for ON and OFF pixels. Can optionally shuffle the
        design matrix to get a null distribution. Also, note that the STAs are whitened by default to reduce stimulus
        correlations (can change with self.whiten).

        **Can probably rewrite this to be a bit more efficient (less repeated code)!**
        """

        if shuffle:
            if self._shuffled_stas is None:
                self._shuffled_stas = np.zeros(
                    (self.n_rois, 2, *self.image_shape), dtype=float
                )

                design_matrix_int = self.design_matrix.astype(
                    int
                )  # shape (2*n_pixels, n_sweeps)

                np.random.seed(0)
                np.random.shuffle(design_matrix_int.T)  # shuffle along sweeps
                assert (
                    design_matrix_int.shape == self.design_matrix.shape
                ), "shuffling changed design matrix shape!"

                pixels_on = design_matrix_int[
                    : design_matrix_int.shape[0] // 2, :
                ].reshape(
                    *self.image_shape, design_matrix_int.shape[1]
                )  # on pixels
                pixels_on = np.select(
                    [pixels_on == 1, pixels_on == 0],
                    [self.pixel_on, self.pixel_on / 2],
                    pixels_on,
                )

                pixels_off = design_matrix_int[
                    design_matrix_int.shape[0] // 2 :, :
                ].reshape(
                    *self.image_shape, design_matrix_int.shape[1]
                )  # off pixels
                pixels_off = np.select(
                    [pixels_off == 1, pixels_off == 0],
                    [self.pixel_off, self.pixel_on / 2],
                    pixels_off,
                )

                if self.whiten:
                    mu, cov_matrix = self.get_whitening_params(pixels_on, pixels_off)

                for roi in range(self.n_rois):
                    on_sta = (pixels_on * self.sweep_responses[:, roi]).sum(
                        axis=2
                    ) / self.sweep_responses[:, roi].sum()
                    off_sta = (pixels_off * self.sweep_responses[:, roi]).sum(
                        axis=2
                    ) / self.sweep_responses[:, roi].sum()

                    if self.whiten:
                        on_sta, off_sta = self.whiten_stas(
                            mu, cov_matrix, on_sta, off_sta, roi
                        )

                    self._shuffled_stas[roi, 0, :, :] = on_sta
                    self._shuffled_stas[roi, 1, :, :] = off_sta

            return self._shuffled_stas

        if not shuffle:
            if self._stas is None:
                self._stas = np.zeros((self.n_rois, 2, *self.image_shape), dtype=float)
                design_matrix_int = self.design_matrix.astype(
                    int
                )  # shape (2*n_pixels, n_sweeps)
                pixels_on = design_matrix_int[
                    : design_matrix_int.shape[0] // 2, :
                ].reshape(
                    *self.image_shape, design_matrix_int.shape[1]
                )  # on pixels
                pixels_on = np.select(
                    [pixels_on == 1, pixels_on == 0],
                    [self.pixel_on, self.pixel_on / 2],
                    pixels_on,
                )

                pixels_off = design_matrix_int[
                    design_matrix_int.shape[0] // 2 :, :
                ].reshape(
                    *self.image_shape, design_matrix_int.shape[1]
                )  # off pixels
                pixels_off = np.select(
                    [pixels_off == 1, pixels_off == 0],
                    [self.pixel_off, self.pixel_on / 2],
                    pixels_off,
                )

                if self.whiten:
                    mu, cov_matrix = self.get_whitening_params(pixels_on, pixels_off)

                for roi in range(self.n_rois):
                    on_sta = (pixels_on * self.sweep_responses[:, roi]).sum(
                        axis=2
                    ) / self.sweep_responses[:, roi].sum()
                    off_sta = (pixels_off * self.sweep_responses[:, roi]).sum(
                        axis=2
                    ) / self.sweep_responses[:, roi].sum()

                    if self.whiten:
                        on_sta, off_sta = self.whiten_stas(
                            mu, cov_matrix, on_sta, off_sta, roi
                        )

                    self._stas[roi, 0, :, :] = on_sta
                    self._stas[roi, 1, :, :] = off_sta

            return self._stas

    def get_whitening_params(self, pixels_on, pixels_off):
        pixels = pixels_on + pixels_off
        pixels = pixels - (self.pixel_on / 2)  # center pixels around 127.5
        flattened_pixels = pixels.reshape(-1, pixels.shape[2])  # (n_pixels, n_sweeps)

        mu = np.mean(flattened_pixels, axis=1, keepdims=True)
        centered_pixels = flattened_pixels - mu  # shape (n_pixels, n_sweeps
        cov_matrix = np.cov(centered_pixels)  # shape (n_pixels, n_pixels)

        return mu, cov_matrix

    def whiten_stas(self, mu, cov_matrix, on_sta, off_sta, lam=1e-3):

        centered_on_sta = on_sta.flatten() - mu.flatten()
        centered_off_sta = off_sta.flatten() - mu.flatten()

        A = cov_matrix + lam * np.eye(cov_matrix.shape[0])
        k_on = np.linalg.solve(A, centered_on_sta)  # whitened STA
        k_off = np.linalg.solve(A, centered_off_sta)  # whitened STA

        return k_on.reshape(self.image_shape), k_off.reshape(self.image_shape)

    def twoD_Gaussian(self, xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = xy
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
            2 * sigma_y**2
        )
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
            4 * sigma_y**2
        )
        c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
            2 * sigma_y**2
        )
        g = offset + amplitude * np.exp(
            -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
        )
        return g.ravel()

    def fit_gaussian(self, data, type=None):
        """
        data: 2D numpy array (lsn.image_shape) of average response to ON or OFF pixels
        """

        # Figure out initial guess parameters
        if type is None:
            idx = np.argmax(np.abs(data))
        elif type == "on":
            idx = np.argmax(data)
        elif type == "off":
            idx = np.argmin(data)

        x0_0, y0_0 = idx % data.shape[1], idx // data.shape[1]
        A0 = data[y0_0, x0_0] - np.median(data)
        B0 = np.median(data)
        initial_guess = (A0, x0_0, y0_0, 1, 1, 0, B0)

        # Create x and y coordinate arrays
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        x, y = np.meshgrid(x, y)

        # Try fitting the Gaussian model to the data
        try:
            popt, pcov = opt.curve_fit(
                self.twoD_Gaussian,
                (x, y),
                data.reshape(-1),
                p0=initial_guess,
                maxfev=100000,
            )

            fitted_data = self.twoD_Gaussian((x, y), *popt).reshape(data.shape)

            resid = data - fitted_data
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((data - data.mean()) ** 2) + 1e-12
            r2 = 1 - ss_res / ss_tot

            return popt, pcov, fitted_data, r2
        except:
            return None, None, np.zeros_like(data), None

    @property
    def rf_centers(self):
        """
        Array of shape (n_rois, 2, 2). Dimension 1 corresponds to ON (0) and OFF (1). Dimension 2 corresponds to
        azimuth (0) and altitude (1). Values of np.nan mean the ROI does not have a given RF.
        """
        if self._rf_centers is None:
            self._rf_centers = np.full((self.n_rois, 2, 2), np.nan)

            # Only iterate over ROIs with an RF
            rois, onoffs, alts, azis = np.where(self.receptive_fields)

            for roi in np.unique(rois):
                roi_mask = rois == roi
                for onoff in np.unique(onoffs[roi_mask]):
                    mask = roi_mask & (onoffs == onoff)
                    alt, azi = self.point_to_alt_azi(
                        alt_ctr=np.mean(alts[mask]) + 0.5,
                        azi_ctr=np.mean(azis[mask]) + 0.5,
                    )  # Add 0.5 to center in pixel
                    self._rf_centers[roi, onoff, :] = (azi, alt)

        return self._rf_centers

    def has_receptive_field(self, roi, rf_type=None):
        if rf_type is None:
            rf = self.receptive_fields[roi]
        else:
            rf = self.receptive_fields[roi, self._rf_type_idx(rf_type)]
        return bool(
            rf.max() >= self.frac_sig_trials_thresh
        )  # otherwise it is a numpy type

    def _rf_type_idx(self, rf_type):
        if type(rf_type) is int:
            return rf_type

        if rf_type == "on":
            return 0
        elif rf_type == "off":
            return 1
        else:
            raise ValueError(f"Bad rf_type: {rf_type}")

    def _load_frames(self):
        lsn_frames_file = path.join(
            self.session.v1dd_client.database_path,
            "stim_movies",
            "lsn_9deg_28degExclusion_jun_256.npy",
        )
        all_frame_images = np.load(lsn_frames_file)

        # Incorrect stimulus:
        # from tifffile import tifffile
        # lsn_frames_file = path.join(self.session.v1dd_client.database_path, "stim_movies", "stim_locally_sparse_nois_16x28.tif")
        # all_frame_images = tifffile.imread(lsn_frames_file)

        self._frame_images = all_frame_images[: self.stim_table["frame"].max() + 1]

        _, nrows, ncols = all_frame_images.shape
        self._azimuths = (np.arange(ncols) - ncols // 2 + 0.5) * self.grid_size
        self._altitudes = (np.arange(nrows) - nrows // 2 + 0.5) * self.grid_size

    def get_stim_indices_from_frames(self, frames: list):
        return self.stim_table.index[self.stim_table["frame"].isin(frames)]

    def point_to_alt_azi(self, alt_ctr, azi_ctr):
        # Assumes lsn.altitudes and lsn.azimuths are equally-spaced (which is true in our case; 9.3 deg spacing)
        alt = (
            alt_ctr * (self.altitudes[-1] - self.altitudes[0]) / len(self.altitudes)
            + self.altitudes[0]
        )
        azi = (
            azi_ctr * (self.azimuths[-1] - self.azimuths[0]) / len(self.azimuths)
            + self.azimuths[0]
        )
        return alt, azi

    def plot_rf(self, rf, rf_type, desc=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        is_on = self._rf_type_idx(rf_type) == 0
        ax.imshow(
            rf,
            cmap=("Reds" if is_on else "Blues"),
            interpolation="none",
            origin="lower",
            vmin=0,
            vmax=0.5,
            extent=self.imshow_extent,
        )
        # ax.set_xticks(ticks=self.azimuths, labels=[f"{azi:.0f}" for azi in self.azimuths])
        # ax.set_yticks(ticks=self.altitudes, labels=[f"{alt:.0f}" for alt in self.altitudes])
        ax.set_xlabel("Azimuth (°)", fontsize=12)
        ax.set_ylabel("Altitude (°)", fontsize=12)
        ax.set_title(
            f"{'ON' if is_on else 'OFF'} receptive field{'' if desc is None else f' ({desc})'}",
            color=("red" if is_on else "blue"),
        )
        ax.axvline(x=0, color="lightgray", linewidth=0.5, zorder=0)
        ax.axhline(y=0, color="lightgray", linewidth=0.5, zorder=0)
        return ax
