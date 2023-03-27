# Vignette Correction #

This project performs vignette correction for images.

### How to use ###

* Calibration: using repo (https://github.com/gaowenliang/vignetting_calib)
* Change image and vignetting parameters in "config.py"
* For the first time, change "build_table" to "True" to build the correction lookup table (may takes a long time)
* Next time you use, change "build_table" to "False" and it will try to load the table used last time
* Change "source_path" and "result_path" of the images
* Run "python3 correction.py" and it will rectify all the ".png" images in the "source_path" and save the results in "result_path"