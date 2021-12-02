# Automatic Color Unmixing #

## Background:
This repository is a public clone of computationalpathologygroup/AutomaticColorUnmixing by Daan Geijs at the University of Radboud Medical Centre, Computational Pathology Group.
The purpose of this fork is for conversion of all dependencies to openly available / easy to install via conda libraries. 

Files converted to date:
- search_stain.py (all dependencies on digitalpathology library converted to numpy)
- deconvolve_image.py (conversion away from use of multiresolutioninterface / ASAP in progress)

If using this repository in work, please cite Daan's work:

D. J. Geijs, M. Intezar, J. A. W. M. van der Laak, and G. J. S. Litjens "Automatic color unmixing of IHC stained whole slide images", Proc. SPIE 10581, Medical Imaging 2018: Digital Pathology, 105810L (6 March 2018); https://doi.org/10.1117/12.2293734 

## 1. Run 'get_stain_matrix.py' to retrieve the clusters and the stain_matrix for the color deconvolution. 
**Single file mode**
```
python get_stain_matrix.py [ARGUMENTS]

ARGUMENTS:

    --input_path, -i        the path (e.g. .mrxs) of the input image.
    --config_path, -c       the path (.json) of config file. 
    --mask_path, -p         the path (e.g. .tif) of a mask which excludes certain areas. If no mask, then this argument 
                                should be the empty string ("").
    --output_path, -d       the output folder of results with placeholder "{image}"
    --num_processes, -n     number of processes
    --overwrite, -o         overwrite previous results

$ python get_stain_matrix.py -e /mnt/netcache/pathology/archives/lung/immunotherapy_pilot/all_images/LI_S01_P000001_C0041_L01_A01.tif -c config.json -p /mnt/netcache/pathology/archives/lung/immunotherapy_pilot/all_images/LI_S01_P000001_C0001_L01_A01_ACU.tif -d results/{image} -n 1
```
**Folder mode**
```
python get_stain_matrix.py [ARGUMENTS]

ARGUMENTS:

    --input_path, -i        the regular expression path (e.g. *.mrxs) of the input images.
    --config_path, -c       the path (.json) of config file. 
    --mask_path, -p         the path with a placeholder "{image}" (.tif) of a masks which excludes certain areas. 
                                The placeholder makes sure that the input image and the mask are connected. If no masks, 
                                then this argument should be the empty string ("").
    --output_path, -d       the output folder of results 
    --num_processes, -n     number of processes
    --overwrite, -o         overwrite previous results

$ python get_stain_matrix.py -e /mnt/netcache/pathology/archives/lung/immunotherapy_pilot/all_images/LI_S01_P000001_C004*_L01_A01.tif -c config.json -p /mnt/netcache/pathology/archives/lung/immunotherapy_pilot/all_images/{image}_ACU.tif -d results/{image} -n 1
```
## 2. Run 'deconvolve_image.py' to retrieve .tif of color deconvolution.
**Single file mode**
 ```
python deconvolve_image.py [ARGUMENTS]

ARGUMENTS:

    --input_path, -i            the path (e.g. .mrxs) of the input image.
    --config_path, -c           the path (.json) of config file. 
    --stain_matrix_path, -s     the path (.txt) of stain_matrix file. Placeholder "{image}" can be used (e.g. \results\{image}\stain_matrix.csv).  
    --mask_path, -p             the path (.tif) of a mask which excludes certain areas. If no mask, then this argument 
                                    should be the empty string ("").
    --output_path, -d           the output folder of results with placeholder "{image}"
    --num_processes, -n         number of processes (optional, default=1)
    --work_dir, -w              work_dir (optional, default="")
    --overwrite, -o             overwrite previous results
    

$ python deconvolve_image.py 
    -e /mnt/netcache/pathology/archives/lung/immunotherapy_pilot/all_images/LI_S01_P000001_C0001_L01_A01.tif -c config.json -p /mnt/netcache/pathology/archives/lung/immunotherapy_pilot/all_images/LI_S01_P000001_C0001_L01_A01_ACU.tif -d results/{image} -n 1
```  
**Folder mode**
 ```
python deconvolve_image.py  [ARGUMENTS]

ARGUMENTS:

    --input_path, -i            the regular expression path (e.g. *.mrxs) of the input images.
    --config_path, -c           the path (.json) of config file. 
    --stain_matrix_path, -s     the path (.txt) of stain_matrix file. Placeholder "{image}" can be used (e.g. \results\{image}\stain_matrix.csv).
    --mask_path, -p             the path with a placeholder "{image}" (.tif) of a masks which excludes certain areas. 
                                    The placeholder makes sure that the input image and the mask are connected. If no 
                                    masks, then this argument should be the empty string ("").
    --output_path, -d           the output folder of results with placeholder "{image}"
    --num_processes, -n         number of processes (optional, default=1)
    --work_dir, -w              work_dir (optional, default="")
    --overwrite, -o              overwrite previous results
    

$ python apply_color_deconv.py -i /mnt/netcache/pathology/archives/lung/immunotherapy_pilot/all_images/LI_S01_P000001_C004*_L01_A01.tif -c config.json -s results/{image}/stain_matrix.txt -p /mnt/netcache/pathology/archives/lung/immunotherapy_pilot/all_images/{image}_ACU.tif  -d results/{image}/deconvolution.tif --overwrite
```  

