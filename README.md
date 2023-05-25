# hv_image_processing
Repository to process images from the KIT HV electrode test setup.
For more information, see the [wiki page](https://ikp-katrin-wiki.ikp.kit.edu/darwin/index.php/XENONnT:High_Voltage_Electrode_Test)

# Basic usage
### Download
 * Make sure you have access to this repository: your github account needs to be a member of [KIT-Dark-Matter](https://github.com/KIT-Dark-Matter) and you need an associated [ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
 * Open your favorite command line interface in your workspace
 * Type `git clone git@github.com:KIT-Dark-Matter/hv_image_processing.git`
 * `cd hv_image_processing`
 * Open the files `run_handler.py` and `image_handler.py` in your favorite python IDE and change the two global path variables to make them point to the directories `cache` and `run_data` of this module (Complaints about having to do this and suggestions on how to do this better will not be accepted.)
 * Now every one of the notebooks in the `notebooks` directory should work out-of-the-box. If not, you might not have access to the data directories. See [the data structure section](#data-structure)

### Naming conventions
The process of ramping up the voltage, observing some glow or sparks or breakdowns and then ramping the voltage back down is called a `run`. One `run` ideally has the same conditions in terms of lighting and Ar environment. Multiple runs from the same day can be collected in a test measurement, or measurement `campaign`. The runs in one campaign dont have to have the same conditions, but in general they will at least share a wiki page.

During one `run` there will be several pictures taken. The contour finding procedure in this module tries to detect areas of increased brightness, in the ideal case areas where the mesh glows from electron emission. These areas are called `cluster`s. A `cluster` has an `area` and a `perimeter`. These are the number of pixels contained in the `cluster`, and the `cluster` edge respectively. A `cluster` also has a `mass`, that is the sum of all pixels within the `cluster`.

### Config files
The config files are meant to enable loading and processing an entire measurement campaign without any additional input. They are JSON files and examples can be found in the `configs` directory. They currently have these keys:
 * `id`: A unique identifier for this campaign. If it is not unique, you risk overwriting other data, or loading wrong data from cache.
 * `date`: The date of this campaign in the format `YYYY-MM-DD`
 * `image_path`: The absolute path to the images. If the images of individual runs are in separate subdirectories, this should be the parent of those subdirs.
 * `image_subdirs`: In case runs are separate, add the *relative* path from `image_path` as a list (same length as number of runs)
 * `bright_images`: Sometimes we want to plot features on a bright image with the same positional cofiguration. These are the images that will be used. List of length 1 or same length as number of runs.
 * `hv_path`: Absolute path to the corresponding HV output.

# Data structure
### Where to find the data?
The data is (at time of writing this) stored in `Y:\Cathode\LocalHVTest`. If you dont have access to this directory, ask someone for help (I forgot how to get access for this, ask Vera).

### I made a new test measurement, how should I store the data?
Somehow similar to the previous campaigns:
Historically grown, the folder structure like this
 * Main directory for this campaign. Examples are `SpareMesh` or `SpareMesh_2nd_230502`.
   * `hv_output_data`: the folder where we put the HV data
     * Individual HV files. Some of the HV files will have headers and others will not. There is no pattern to this and I have not yet made the run handler be able to load both. If a HV csv file still has its headers, delete them or adjust the run handler to ignore headers where present.
   * Directory for NIKON camera images (the jpg ones)
     * Misc diretories for calibration, for edited files, for everything you want
     * One directory per run. You can call them what you want. The config file for the campaign just has to reflect what you chose
   * Directory for CANON camera images (the cr2 raw ones)
     * Same as the NIKON directory



# Module overview
## processing.py
Contains all the functions to process images, like brightness and contrast adjustment or scaling.

## image_handler.py
Contains the class and functions to work with individual images.

## run_handler.py
Contains the classes and functions to work with runs and campaigns.
