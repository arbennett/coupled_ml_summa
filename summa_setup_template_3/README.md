# SUMMA Setup Template
This repo provides a bare-bones sample of a [SUMMA](https://github.com/NCAR/summa) setup.

## forcings directory
This directory contains forcing files in NetCDF format.
Additionally it contains `forcing_file_list.txt`, which is a simple listing of all of the files to be used as input.
These files must be enclosed in single quotes, as the example file shows.

## output directory
This is a dummy directory which SUMMA output will be written to.

## settings
The settings directory contains various configuration and settings files.
Users will primarily be interested in modifying the `decisions.txt` and
`output_control.txt` files. They control the model structure/parameterization
and what data gets written out to output, respectively.

## Other files
The `template_file_manager.txt` contains a template of the master configuration.
It is not usable in the default form, and should be "installed" by running the
`install_local_setup.sh` script. This script localizes all of the directories by
setting absolute paths.
