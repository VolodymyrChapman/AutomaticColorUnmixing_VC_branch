import math
import os

# original line:
# import digitalpathology.image.io.imagereader as dptimagereader

# replace departmental io with skimage.io.imread
from skimage import io

import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as km
from PIL import Image
from recordclass import dataobject
from skimage.color import label2rgb
from tqdm import tqdm


class StainInfo(dataobject):
    tile_coords: list
    all_dist: list
    all_clusters: list


class DetermineStains(object):

    def __init__(self, config):
        self.filename = None
        self.figure_path = config['figure_path']
        self.spacing = config['spacing']
        self.threshold = config['threshold']
        self.percentile = config['percentile']
        self.OD_threshold = config['OD_threshold']
        self.patch_size = config['patch_size']
        self.verbose = config['verbose']
        self.multi_processing = config['multi_processing']
        self.img_obj = None
        self.mask_obj = None
        self.apply_color_aug = config['apply_color_aug']
        self.color_aug_sigma = config['sigma']
        self.color_aug_bias = config['bias']

        self.dim_y = None
        self.dim_x = None

    def get_peaks_distance(self, cxcy):
        # Determine peaks in clusterusing k-means
        if np.shape(cxcy)[0] > 4:
            est = km.KMeans(n_clusters=2)
            est.fit(cxcy)
            k_center = np.sort(est.cluster_centers_, axis=0)
            # measure distance between two peaks
            dist = np.linalg.norm(k_center[0, :] - k_center[1, :])

        else:
            dist = 0
            k_center = np.array([[np.nan, np.nan], [np.nan, np.nan]])

        return dist, k_center

    def rgb_2_hsd(self, tile, rgb0=[255, 255, 255]):
        #   Converts RGB pixel to cx, cy
        d_rgb = np.empty(np.shape(tile))
        d = np.zeros(np.shape(tile)[:2])

        for channel in range(3):
            a = tile[:, :, channel] / rgb0[channel]
            d_rgb[:, :, channel] = -np.log(np.clip(a, 1e-8, 1.0))
            d += d_rgb[:, :, channel]

        # add small value so fraction can not become 0/3
        d = (d + 1e-4) / 3
        thresh = d > self.OD_threshold
        # calculate c_x for every pixel
        c_x = (d_rgb[thresh, 0] / d[thresh]) - 1
        # calculate c_y for every pixel
        c_y = (d_rgb[thresh, 1] - d_rgb[thresh, 2]) / (np.sqrt(3) * d[thresh])

        c_x_s = np.array([c_x]).reshape(-1, 1)
        c_y_s = np.array([c_y]).reshape(-1, 1)
        coords = np.concatenate((c_x_s, c_y_s), axis=1)

        return coords

    def plot_stains(self, stain1, stain2, roi_cluster, affix_stain):
        """
    
        """

        fig = plt.figure()

        plt.plot(roi_cluster[:, 0, 0], roi_cluster[:, 0, 1], 'r.')
        plt.plot(roi_cluster[:, 1, 0], roi_cluster[:, 1, 1], 'b.')

        plt.plot(stain1[0], stain1[1], 'go')
        plt.plot(stain2[0], stain2[1], 'go')

        plt.ylim(-2, 2)
        plt.xlim(-1, 2)

        fig.savefig(os.path.join(self.figure_path.format(image=self.filename), affix_stain))
        plt.close()

    # def get_ratio(self, spacing_a, spacing_b):
    #     # down_sample_index_a = np.where(np.isclose(self.img_obj.size[0], spacing_a, atol=spacing_a * 0.25))[0][0]
    #     # down_sample_index_b = np.where(np.isclose(self.img_obj.size[1], spacing_b, atol=spacing_b * 0.25))[0][0]

    #     # ratio = self.img_obj.downsamplings[down_sample_index_a] / self.img_obj.downsamplings[down_sample_index_b]

    #     return ratio

    def plot_thumbnail(self, stain_info, patch_numbers, affix_thumbnail):
        if self.verbose:
            print('Plotting thumbnail...')

        # dimension_y, dimension_x = self.img_obj.shapes[np.where(np.isclose(self.img_obj.spacings,
        #                                                                    projection_level,
        #                                                                    atol=projection_level * 0.25))[0][0]]
        dimension_y, dimension_x, _ = self.img_obj.shape

        full_image = self.img_obj
        rgb_mask = np.zeros([dimension_y, dimension_x])
        tile_spots = np.array([stain_info.tile_coords[i] for i in patch_numbers])
        
        # If WSIs used with certain downsampling:
        # projection_level = 8.0
        # ratio = self.get_ratio(self.spacing, projection_level)

        # if non-downsampled (1x) tiles used as input instead
        ratio = dimension_y / self.patch_size
        
        # draw stuff
        # with tqdm(total=len(tile_spots),
        #           desc='Progress', unit='tile', disable=self.use_tqdm) as pbar:

        for i in tile_spots:
            tile_coords = np.asanyarray(i * ratio, dtype=int)
            rgb_mask[tile_coords[1]:tile_coords[3], tile_coords[0]:tile_coords[2]] = 1
            # pbar.update(1)

        plt.figure(1)
        image_label_overlay = label2rgb(rgb_mask, alpha=0.3, image=full_image.astype(dtype=np.uint8), bg_label=0)
        plt.imshow(image_label_overlay)
        im = Image.fromarray(np.array(image_label_overlay * 255, dtype=np.uint8))
        im.save(os.path.join(self.figure_path.format(image=self.filename), affix_thumbnail))
        plt.close()

    def determine_cxcy(self, tile, stain_info, tile_coords):
        # if self.verbose:
        #     print('Determining cxcy...')
        if np.mean(tile) < self.threshold:
            cxcy = self.rgb_2_hsd(tile)

            if np.any(cxcy):  # check if cxcy is not empty, due to tresholding of D

                dist, k_center = self.get_peaks_distance(cxcy)

                stain_info.all_dist.append(dist)
                stain_info.all_clusters.append(k_center)
                stain_info.tile_coords.append(tile_coords)
            else:
                stain_info.all_dist.append(np.nan)
                stain_info.all_clusters.append(np.nan)
                stain_info.tile_coords.append(np.nan)
        else:
            stain_info.all_dist.append(np.nan)
            stain_info.all_clusters.append(np.nan)
            stain_info.tile_coords.append(np.nan)
    

    #### Need to modify this to use np arrays as image inputs instead of dep. read functions
    def search_stain(self,
                    #  color_augmenter=None,
                     affix_stain="cxcy.png",
                     affix_thumbnail='overlay.png'):

        tile_coords = []
        all_dist = []
        all_clusters = []

        stain_info = StainInfo(tile_coords, all_dist, all_clusters)
        if self.verbose:
            print('\nSearching stains...')

        with tqdm(total=int(math.floor(self.dim_y / self.patch_size) * math.floor(self.dim_x / self.patch_size)),
                  desc='Progress', unit='tile', disable=1 - self.verbose + self.multi_processing) as pbar:
            
            # Split into patches acrosss y
            for y_index in range(0, self.dim_y, self.patch_size):
                # Split into patches acrosss x
                for x_index in range(0, self.dim_x, self.patch_size):

                    # Sanity check that all tiles being processed
                    # print(f'x_index = {x_index}; y_index = {y_index}')
                    
                    # Stop once patch extends past dimensions of image
                    if x_index + self.patch_size > self.dim_x or y_index + self.patch_size > self.dim_y:
                        # pbar.update(1)
                        continue
                    
                    # Read relevant region of mask object or initialise mask tile
                    if self.mask_obj:
                        # mask_tile = self.mask_obj.read(self.spacing, y_index, x_index, self.patch_size, self.patch_size)
                        
                        # As above but np implementation
                        mask_tile = self.mask_obj[y_index:y_index + self.patch_size, x_index: x_index + self.patch_size, :]

                    else:
                        mask_tile = np.ones((self.patch_size, self.patch_size, 1))

                    if np.any(mask_tile):
                        # Slice relevant region of image tile
                        # tile = self.img_obj.read(self.spacing,
                        #                          y_index,
                        #                          x_index,
                        #                          self.patch_size,
                        #                          self.patch_size)
                       
                        # As above but np implementation
                        tile = self.img_obj[y_index:y_index + self.patch_size, x_index: x_index + self.patch_size, :]

                        tile[mask_tile.squeeze() == 0] = [254, 254, 254]

                        self.determine_cxcy(tile, stain_info, [x_index,
                                                               y_index,
                                                               x_index + self.patch_size,
                                                               y_index + self.patch_size])

                    pbar.update(1)
        # Select ROI
        #
        if np.any(~np.isnan(stain_info.all_dist)):
            perc_n = np.percentile(np.array(stain_info.all_dist)[~np.isnan(stain_info.all_dist)], [self.percentile])
            patch_numbers = [i for i, x in enumerate(stain_info.all_dist) if ~np.isnan(x) and x > perc_n]
            roi_cluster = np.array([stain_info.all_clusters[i] for i in patch_numbers])

            # Select median values of all region of interest values as color
            stain1, stain2 = np.median(roi_cluster, axis=0)
            np.savetxt(os.path.join(self.figure_path.format(image=self.filename), 'stain_clusters.txt'),
                       [stain1, stain2])

            if self.figure_path:
                self.plot_stains(stain1, stain2, roi_cluster, affix_stain)
                self.plot_thumbnail(stain_info, patch_numbers, affix_thumbnail)

            return stain1, stain2
        else:
            print("Something went wrong with file: {}".format(self.filename))
            pass

    @staticmethod
    def calculate_stain_matrix(stain1, stain2):
        # Calculate stain matrix  with cx cy coordinates
        D = 1
        matrix = []
        for stain in [stain1, stain2]:
            cx = stain[0]
            cy = stain[1]
            Dr = D * (cx + 1)
            Dg = 0.5 * D * (2 - cx + np.sqrt(3) * cy)
            Db = 0.5 * D * (2 - cx - np.sqrt(3) * cy)
            vector = np.linalg.norm([[Dr], [Dg], [Db]])
            line = np.array([Dr / vector, Dg / vector, Db / vector])
            matrix.append(line)
        matrix.append([0, 0, 0])
        matrix = np.array(np.reshape(matrix, [3, 3]))
        matrix[2, :] = np.cross(matrix[0, :], matrix[1, :])
        return matrix

    def process_slide(self, input):

        input_image_path, mask_image_path, stain_matrix_file = input
        self.filename = input_image_path.stem
        if self.verbose:
            tqdm.write("\nProcessing: {}".format(self.filename))

        self.img_obj = io.imread(input_image_path.__str__())
        self.mask_obj = io.imread(mask_image_path.__str__()) if mask_image_path else None

        # self.dim_y, self.dim_x = self.img_obj.shapes[np.where(np.isclose(self.img_obj.spacings, self.spacing, atol=4.0 * 0.25))[0][0]]
        # self.dim_y, self.dim_x,_ = self.img_obj.shapes[
        #     np.where(np.isclose(self.img_obj.spacings, self.spacing, atol=self.spacing * 0.25))[0][0]]

        self.dim_y, self.dim_x,_ = self.img_obj.shape

        # Find org stain without color aug.
        #
        stain1, stain2 = self.search_stain(affix_stain="cxcy.png", affix_thumbnail='overlay.png')

        # Calculate stainmatrix used for color deconvolution
        #
        stain_matrix = self.calculate_stain_matrix(stain1, stain2)
        np.savetxt(os.path.join(stain_matrix_file, 'stain_matrix.txt'), stain_matrix)
