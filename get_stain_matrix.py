import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import search_stain


def assemble_jobs(image_path, mask_path, output_path):
    """
    Assemble (source image path, source mask path, target output path) job triplets for network application.
    Args:
        image_path (str): Path of the image to classify.
        mask_path (str): Path of the mask image to use.
        output_path (str): Path of the result image.
    Returns:
        list: List of job tuples.
    Raises:
        InvalidDataFileExtensionError: The format cannot be derived from the file extension.
        InvalidDataSourceTypeError: Invalid JSON or YAML file.
        PurposeListAndRatioMismatchError: The configured purpose distribution does not match the available purposes.
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
        # target_output_path = os.path.join(output_path, (image_base.format(image=image_base)))
        target_output_path = output_path.format(image=image_base)
        mask_path = mask_path.format(image=image_base)
        print(target_output_path)

        if os.path.isfile(mask_path) and mask_path.endswith('.tif'):
            mask_file = Path(mask_path)
        elif not os.path.isfile(mask_path):
            mask_file = ""
        else:
            raise Exception("Mask extension ({extension}) should be of file format .tif".format(
                extension=os.path.splitext(mask_path)[1]))

        # Add job item to the list.
        #
        job_item = [Path(image_path), mask_file, Path(target_output_path)]
        result_job_list.append(job_item)

    else:

        image_file_path_map = glob.glob(image_path)
        # Assemble list.
        #
        for image_key in image_file_path_map:
            image_base = os.path.splitext(os.path.basename(image_key))[0]

            mask_file_path = mask_path.format(image=image_base)
            # print(mask_file_path)
            target_output_path = output_path.format(image=image_base)
            # target_output_path = os.path.join(output_path, (image_base.format(image=image_base)))

            if os.path.isfile(mask_file_path) and mask_file_path.endswith('.tif'):
                mask_file = Path(mask_file_path)
            elif not os.path.isfile(mask_path):
                mask_file = ""
            else:
                raise Exception("Mask extension ({extension}) should be of file format .tif".format(
                    extension=os.path.splitext(mask_path)[1]))
            # Add job item to the list.
            #
            job_item = [Path(image_key), mask_file, Path(target_output_path)]
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
    argument_parser = argparse.ArgumentParser(description='Get stain matrix for given image.')

    argument_parser.add_argument('-i', '--input_path', required=True, type=str)
    argument_parser.add_argument('-c', '--config_path', required=True, type=str)
    argument_parser.add_argument('-p', '--mask_path', required=False, type=str, default='')
    argument_parser.add_argument('-d', '--output_path', required=True, type=str)
    argument_parser.add_argument('-n', '--num_processes', required=False, type=int, default=1)
    argument_parser.add_argument('-o', '--overwrite', required=False, default=False,  action='store_true')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_file_base_path = arguments['input_path']
    parsed_config_base_path = os.path.abspath(arguments['config_path'])
    parsed_mask_base_path = arguments['mask_path']
    parsed_output_path = arguments['output_path']
    parsed_number_processes = arguments['num_processes']
    parsed_overwrite = arguments['overwrite']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input path: {parsed_file_base_path}'.format(parsed_file_base_path=parsed_file_base_path))
    print('Config path: {parsed_config_base_path}'.format(parsed_config_base_path=parsed_config_base_path))
    print('Mask path: {parsed_mask_base_path}'.format(parsed_mask_base_path=parsed_mask_base_path))
    print('Output path: {parsed_output_path}'.format(parsed_output_path=parsed_output_path))
    print('Number of processes: {parsed_number_processes}'.format(parsed_number_processes=parsed_number_processes))
    print('Overwrite: {parsed_overwrite}'.format(parsed_overwrite=parsed_overwrite))

    # Return parsed values.
    #
    return (parsed_file_base_path,
            parsed_config_base_path,
            parsed_mask_base_path,
            parsed_output_path,
            parsed_number_processes,
            parsed_overwrite)


if __name__ == '__main__':

    file_base_path, config_path, mask_base_path, output_path, num_processes, overwrite = collect_arguments()

    job_list = assemble_jobs(file_base_path, mask_base_path, output_path)

    with open(config_path) as config_json:
        config = json.load(config_json)['search_stain']
        config['figure_path'] = output_path
        config['multi_processing'] = (num_processes > 1)

    determine_stain = search_stain.DetermineStains(config)

    process_list = []
    processing_pool = Pool(processes=num_processes)

    for job in job_list:
        if not job[2].joinpath('stain_matrix.txt').is_file() or overwrite:
            process_list.append(job)
            job[2].mkdir(exist_ok=True)
        else:
            print(f"Folder {job[0].stem} already exists, skipping...")

    processing_pool.map(determine_stain.process_slide, process_list)
