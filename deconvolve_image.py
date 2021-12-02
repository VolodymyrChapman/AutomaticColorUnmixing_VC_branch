import os
from shutil import copyfile
from pathlib import Path
import time 
import numpy as np
from skimage.util.dtype import img_as_float
import cv2

# original lines:
# import multiresolutionimageinterface as mir
# import digitalpathology.image.io.imagereader as dptimagereader

import multiresolutionimageinterface as mir
from skimage import io as dptimagereader # remember to replace with io throughout file

#----------------------------------------------------------------------------------------------------

class DeconvolveImage(object):
    
    def __init__(self, config):

        self.spacing = config['spacing']
        self.number_of_stains = config['number_of_stains']
        self.stepsize = config['step_size']
        self.workdir = config['work_dir']
        self.max_method = config['max_method']
                                 
        # TODO: We have to change the compression type to LZW if this is not 3, otherwise we get an tile artifact
        #
        assert self.number_of_stains == 3
        assert self.max_method in ['gaussian', 'tile_percentile', 'max']

        if self.workdir: 
            os.makedirs(self.workdir, exist_ok=True)
        
        self.file_name = None
        self.local_output_file = None
        self.target_output_file = None

        self.img_obj = None
        self.mask_obj = None
        self.stain_matrix = None
    
        self.dim_y = None
        self.dim_x = None
        
        self.inversed_od = None
        self.result_array = None 
        
        self.deconv_max_list = list()
        self.deconv_max = list()     

    def write_output_tif(self, output_path: str):
        """[summary]

        Args:
            output_path (str): [description]
        """

        tile_size = 512 #fixed this value, because it's the fasted tile size for loading 
        output_spacing = self.img_obj.spacings[np.where(np.isclose(self.img_obj.spacings, 
                                                                   self.spacing, 
                                                                   atol=self.spacing*0.25))[0][0]]

        pixel_size_vec = mir.vector_double()
        pixel_size_vec.push_back(output_spacing)
        pixel_size_vec.push_back(output_spacing)

        writer = mir.MultiResolutionImageWriter()
        if self.workdir:
            self.target_output_file = output_path
            Path(output_path).touch()

            self.local_output_file = os.path.join(self.workdir, self.filename + '.tif')
            writer.openFile(self.local_output_file)

        else: 
            writer.openFile(output_path)
                                 
        writer.setTileSize(tile_size) 
        writer.setCompression(mir.JPEG)
        writer.setJPEGQuality(75)
        writer.setDataType(mir.UChar)
        writer.setColorType(mir.Indexed)
        writer.setNumberOfIndexedColors(self.number_of_stains)
        writer.writeImageInformation(self.dim_x, self.dim_y)
        writer.setSpacing(pixel_size_vec)

        for y_indx in range(0, self.dim_y, tile_size):
            for x_indx in range(0, self.dim_x, tile_size):

                tmp_patch = self.result_array[y_indx:y_indx + tile_size,x_indx:x_indx + tile_size]
                if np.any(tmp_patch):
                    writer.writeBaseImagePartToLocation(tmp_patch.flatten().astype(np.uint8), x_indx, y_indx)

        writer.finishImage()                                 
                                 
        if self.local_output_file: 
            copyfile(self.local_output_file, self.target_output_file)              

    @staticmethod
    def color_deconvolution(patch: np.ndarray, inversed_od: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            patch (np.ndarray): [description]
            inversed_od (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
    
        # Remove zeros
        # 
        patch[patch == 0] = 1
        od = -np.log(img_as_float(patch, force_copy=True))

        # reshape image on row per pixel
        # 
        reshaped_od = np.reshape(od, (-1, 3))
        
        # do the deconvolution
        # 
        deconv = np.dot(reshaped_od, inversed_od)
    
        return np.reshape(deconv, od.shape)
    
    def load_stain_matrix(self, stain_matrix_file: str):
        """[summary]

        Args:
            stain_matrix_file (str): [description]
        """
        self.stain_matrix = np.loadtxt(stain_matrix_file)
        
        normalized_od = np.asarray([r / np.linalg.norm(r) for r in self.stain_matrix])
        self.inversed_od = np.linalg.inv(normalized_od)   
        
    def load_images(self, input_image_path: str, mask_image_path: str):
        """[summary]

        Args:
            input_image_path (str): [description]
            mask_image_path (str): [description]
        """
        self.img_obj = dptimagereader.ImageReader(input_image_path)
        self.mask_obj = dptimagereader.ImageReader(mask_image_path) if mask_image_path else None        
        self.dim_y, self.dim_x = self.img_obj.shapes[np.where(np.isclose(self.img_obj.spacings, 
                                                                         self.spacing, 
                                                                         atol=self.spacing * 0.25))[0][0]]
        
        self.result_array = np.zeros((self.dim_y, self.dim_x, 3), dtype=np.uint8)    

    def get_processing_patch(self, y_indx: int, x_indx: int) -> list:
        """[summary]

        Args:
            y_indx (int): [description]
            x_indx (int): [description]

        Returns:
            list: [description]
        """
        patch_y_start = np.max((y_indx, 0))
        patch_y_end = np.min((y_indx + self.stepsize, self.dim_y))
        
        patch_x_start = np.max((x_indx, 0))
        patch_x_end = np.min((x_indx + self.stepsize, self.dim_x))

        return [patch_y_start, patch_y_end, patch_x_start, patch_x_end]                                
                                 
    def find_decov_max(self):
        """
            Determine maximum optical density per stain
        """
        if self.max_method == "tile_percentile":
                                 
            for y_indx in range(0, self.dim_y, self.stepsize):
                for x_indx in range(0, self.dim_x, self.stepsize):
                    
                    patch_coords = self.get_processing_patch(y_indx, x_indx)

                    patch_height = patch_coords[1] - patch_coords[0]
                    patch_width = patch_coords[3] - patch_coords[2]

                    mask_tile = self.get_mask_patch([patch_coords[0], patch_coords[2], patch_height, patch_width])

                    if np.any(mask_tile):
                        tile = self.img_obj.read(self.spacing,
                                                 y_indx,
                                                 x_indx,
                                                 self.stepsize,
                                                 self.stepsize)

                        deconv = self.color_deconvolution(tile, self.inversed_od)
                        self.deconv_max_list.append(deconv.max())
            self.deconv_max = np.percentile(self.deconv_max_list, 95) + 1e-5
                                 
        else:
            dim_y, dim_x = self.img_obj.shapes[np.where(np.isclose(self.img_obj.spacings, 
                                                                   4.0, 
                                                                   atol=4.0 * 0.25))[0][0]]

            patch = self.img_obj.read(4.0, 
                                      0, 
                                      0, 
                                      dim_y, 
                                      dim_x)

            deconv = self.color_deconvolution(patch, self.inversed_od)
            if self.max_method == "gaussian":
                self.deconv_max = [np.max(cv2.GaussianBlur(deconv[:,:,i], (5, 5), 0)) + 1e-5 for i in range(deconv.shape[-1])]
            elif self.max_method == "max": 
                self.deconv_max = [np.max(deconv[:,:,i]) for i in range(deconv.shape[-1]) + 1e-5]
                                 
    def get_mask_patch(self, coords: list) -> np.ndarray:
        """[summary]

        Args:
            coords (list): [description]

        Returns:
            np.ndarray: [description]
        """
        y_min, x_min, y_max, x_max = coords
        
        if self.mask_obj:
            mask_tile = self.mask_obj.read(self.spacing, 
                                           y_min, 
                                           x_min, 
                                           y_max, 
                                           x_max)

        else:
            mask_tile = np.ones((self.stepsize, self.stepsize, 1))

        return mask_tile 
                                 
    def apply_deconv(self):
        """[summary]
        """
        for y_indx in range(0, self.dim_y, self.stepsize):
            for x_indx in range(0, self.dim_x, self.stepsize):
                patch_coords = self.get_processing_patch(y_indx, x_indx)

                patch_height = patch_coords[1] - patch_coords[0]
                patch_width = patch_coords[3] - patch_coords[2]
                                 
                mask_tile = self.get_mask_patch([patch_coords[0], patch_coords[2], patch_height, patch_width])
    
                if np.any(mask_tile):
                    tile = self.img_obj.read(self.spacing, 
                                             patch_coords[0], 
                                             patch_coords[2], 
                                             patch_height, 
                                             patch_width)
                    
                    deconv = self.color_deconvolution(tile, self.inversed_od)
                    deconv = deconv * mask_tile
                    deconv_write_tile = np.clip(deconv * 255.0 / np.asarray(self.deconv_max), 0, 255)
                                        
                    self.result_array[patch_coords[0]:patch_coords[1], patch_coords[2]:patch_coords[3]] = deconv_write_tile

    def process_slide(self, input: list):
        """[summary]

        Args:
            input (list): [description]
        """
        
        input_image_path, stain_matrix_file, mask_image_path, output_file = input

        self.filename = os.path.splitext(os.path.basename(input_image_path))[0]
        print("Processing: {}".format(self.filename))

        start_time = time.time()
        self.load_images(input_image_path.__str__(), mask_image_path.__str__())
        self.load_stain_matrix(stain_matrix_file.__str__())
        self.find_decov_max()
        self.apply_deconv()
        self.write_output_tif(output_file.__str__())
        print("Processing took: {}s".format(time.time() - start_time))

