import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import deconvolve_image


def assemble_jobs(image_path, stain_matrix_path, mask_path, output_path):
    """
    Assemble (source image path, source mask path, target output path) job triplets for network application.
    Args:
        image_path (str): Path of the image to classify.
        stain_matrix_path (str): Path of stain matrices to use.
        mask_path (str): Path of the mask image to use.
        output_path (str): Path of the result image.
    Returns:
        list: List of job tuples.

    """

    # Find out operation mode. If all paths are file paths the mode is 'file'.
    #
    file_mode = os.path.isfile(image_path)
    print('Mode: {mode}'.format(mode=('file' if file_mode else 'folder')))

    result_job_list = []
    if file_mode:
        # Return a single triplet if the paths were existing files.
        #
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        target_output_path = output_path.format(image=image_base)
        mask_path = mask_path.format(image=image_base)
        stain_matrix_path = stain_matrix_path.format(image=image_base)
        if os.path.isfile(mask_path) and mask_path.endswith('.tif'):
            mask_file = Path(mask_path)
        elif not os.path.isfile(mask_path):
            mask_file = ""
        else:
            raise Exception("Mask extension ({extension}) should be of file format .tif".format(
                extension=os.path.splitext(mask_path)[1]))

        # mask_file = mask_path if os.path.isfile(mask_path) and mask_path.endswith('.tif'): else ""

        if not os.path.isfile(stain_matrix_path):
            raise Exception("Stain matrix file not found")

        # Add job item to the list.
        #
        job_item = [Path(image_path), Path(stain_matrix_path), mask_file, Path(target_output_path)]
        result_job_list.append(job_item)

    else:

        image_file_path_map = glob.glob(image_path)

        # Assemble list.
        #
        for image_key in image_file_path_map:
            image_base = os.path.splitext(os.path.basename(image_key))[0]

            mask_file_path = mask_path.format(image=image_base)
            stain_matrix_file_path = stain_matrix_path.format(image=image_base)
            target_output_path = output_path.format(image=image_base)

            if os.path.isfile(stain_matrix_file_path):
                if os.path.isfile(mask_file_path) and mask_file_path.endswith('.tif'):
                    mask_file = Path(mask_file_path)
                elif not os.path.isfile(mask_file_path):
                    mask_file = ""
                else:
                    raise Exception("Mask extension ({extension}) should be of file format .tif".format(
                        extension=os.path.splitext(mask_path)[1]))

                job_item = [Path(image_key), Path(stain_matrix_file_path), mask_file, Path(target_output_path)]
                result_job_list.append(job_item)

    # Return the result list.
    #
    return result_job_list


def collect_arguments():
    """
    Collect command line arguments.

    Returns:
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Apply stain matrix on given image.')

    argument_parser.add_argument('-i', '--input_path', required=True, type=str)
    argument_parser.add_argument('-c', '--config_path', required=True, type=str)
    argument_parser.add_argument('-s', '--stain_matrix_path', required=True, type=str)
    argument_parser.add_argument('-p', '--mask_path', required=False, type=str, default='')
    argument_parser.add_argument('-d', '--output_path', required=True, type=str)
    argument_parser.add_argument('-n', '--num_processes', required=False, type=int, default=1)
    argument_parser.add_argument('-w', '--work_dir', required=False, type=str, default="")
    argument_parser.add_argument('-o', '--overwrite', required=False, default=False,  action='store_true')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_file_base_path = arguments['input_path']
    parsed_config_base_path = arguments['config_path']
    parsed_stain_matrix_path = arguments['stain_matrix_path']
    parsed_mask_base_path = arguments['mask_path']
    parsed_output_path = arguments['output_path']
    parsed_number_processes = arguments['num_processes']
    parsed_work_dir = arguments['work_dir']
    parsed_overwrite = arguments['overwrite']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input path: {parsed_file_base_path}'.format(parsed_file_base_path=parsed_file_base_path))
    print('Input path: {parsed_config_base_path}'.format(parsed_config_base_path=parsed_config_base_path))
    print('Stain matrix path: {parsed_stain_matrix_path}'.format(parsed_stain_matrix_path=parsed_stain_matrix_path))
    print('Mask path: {parsed_mask_base_path}'.format(parsed_mask_base_path=parsed_mask_base_path))
    print('Output path: {parsed_output_path}'.format(parsed_output_path=parsed_output_path))
    print('Workdir: {parsed_work_dir}'.format(parsed_work_dir=parsed_work_dir))
    print('Number of processes: {parsed_number_processes}'.format(parsed_number_processes=parsed_number_processes))
    print('Overwrite: {parsed_overwrite}'.format(parsed_overwrite=parsed_overwrite))

    # Return parsed values.
    #
    return (parsed_file_base_path,
            parsed_config_base_path,
            parsed_stain_matrix_path,
            parsed_mask_base_path,
            parsed_output_path,
            parsed_work_dir,
            parsed_number_processes,
            parsed_overwrite)


if __name__ == '__main__':

    file_base_path, config_path, stain_matrix_path, mask_base_path, output_path, work_dir, num_processes, overwrite = collect_arguments()

    job_list = assemble_jobs(file_base_path, stain_matrix_path, mask_base_path, output_path)

    with open(config_path) as config_json:
        config = json.load(config_json)['deconvolve_image']
        config['work_dir'] = work_dir
        config['multi_processing'] = (num_processes > 1)
    deconvolve_image = deconvolve_image.DeconvolveImage(config)

    process_list = []
    processing_pool = Pool(processes=num_processes)

    for job in job_list:
        if not job[-1].is_file() or overwrite:
            process_list.append(job)
            job[-1].parent.mkdir(exist_ok=True)
        else:
            print(f"Folder {job[0].stem} already exists, skipping...")

    processing_pool.map(deconvolve_image.process_slide, process_list)
