from faim_hcs.hcs.Experiment import Experiment
from faim_hcs.records.PlateRecord import PlateRecord
from faim_hcs.records.WellRecord import WellRecord
from faim_hcs.records.OrganoidRecord import OrganoidRecord

from glob import glob

from os.path import join

from os.path import isdir, dirname, split, basename, splitext, exists
from tqdm.notebook import tqdm

import re

import pandas as pd

from skimage.measure import regionprops, label
from skimage.morphology import binary_erosion
import numpy as np
import math
import copy
import scipy

from matching import matching


def quartiles(regionmask, intensity):
    return np.percentile(intensity[regionmask], q=(25, 50, 75, 90, 95, 99))

def skewness(regionmask, intensity):
    return scipy.stats.skew(intensity[regionmask])

def kurtos(regionmask, intensity):
    return scipy.stats.kurtosis(intensity[regionmask])

def stdv(regionmask, intensity):
    # ddof=1 for sample var
    return np.std(intensity[regionmask], ddof=1)


pd.set_option("display.max_rows", 200)


# Create a new HCS experiment
# name: Name of the experiment
# root_dir: Root directory of the image data
# save_dir: Directory where the experiment is saved. Should be different from the root_dir. This
#           is where the measurements are stored.

exp = Experiment(name='20220507GCPLEX_R3',
                 root_dir='/tungstenfs/scratch/gliberal/Users/repinico/Yokogawa/20220507GCPLEX_R3',
                 save_dir='/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo')

# List directories in root_dir. In this dataset all directories correspond to a plate.
# Add a new PlateRecord for each found directory.
# experiment: The HCS Experiment we created above.
# plate_id: The name of the plate e.g. the directory name.
# save_dir: The directory in which this plate is saved to disk. The measurements will be in this direcotry.

plates = glob(exp.root_dir + "/*")

for p in plates:
    if isdir(p):
        plate = PlateRecord(experiment=exp,
                            plate_id=split(p)[1], save_dir=exp.get_experiment_dir())


# Parsing the directory and find all wells, well-overviews, well-segmentations, objects, and object-segmentations.

# NOTE: This parsing is specifically for the example dir from Nicole. If you want to parse another experiment you might
# have to adapt this.

well_regex = re.compile("_[A-Z]{1}[0-9]{2}_")

raw_ch_regex = re.compile("C[0-9]{2}O.*_TIF-OVR.tif")
#raw_ch_regex = re.compile("C[0-9]{2}O.*_TIF-OVR_bin.tif")
mask_ending = "MASK"
#mask_ending = "MASK_bin"
nuc_ending = "NUC_SEG3D_220523" #load segmentation images
mem_ending = "MEM_SEG3D_220523" #load segmentation images

#nuc_seg_regex = re.compile("_NUC_SEG.*.tif") #add here the neural network name
mask_regex = re.compile(mask_ending + ".tif")
nuc_seg_regex = re.compile(nuc_ending + ".tif")
cell_seg_regex = re.compile(mem_ending + ".tif")

for plate in tqdm(exp.plates.values()):
    ovr_mips = glob(join(exp.root_dir, plate.plate_id, 'TIF_OVR_MIP', '*.tif'))

    well_ids = [well_regex.findall(basename(om))[0][1:-1] for om in ovr_mips]

    for well_id in well_ids:
        # Add a well.
        w = WellRecord(plate=plate,
                       well_id=well_id,
                       save_dir=plate.plate_dir)

        # Add the raw files corresponding to this well. These are the MIP overview images.
        raw_files = list(filter(lambda p: well_regex.findall(basename(p))[0][1:-1] == well_id, ovr_mips))
        for raw_file in raw_files:
            raw_name = splitext(raw_file)[0][-3:]
            w.add_raw_data(raw_name, raw_file, spacing=(0.216, 0.216))

        # Add segmentation files of the MIP overview images (organoid seg).
        seg_mips = glob(join(exp.root_dir, plate.plate_id, 'TIF_OVR_MIP_SEG', 'obj_v0.3', '*.tif'))
        seg_files = list(filter(lambda p: basename(p).split('_')[3] == well_id, seg_mips))  # corrected :)
        # seg_files = list(filter(lambda p: basename(p).split('_')[2] == well_id, seg_mips)) #corrected :)
        for seg_file in seg_files:
            seg_name = splitext(seg_file)[0][-3:]
            w.add_segmentation(seg_name, seg_file)

        # Search for region-extracted organoids and add them.
        # This adds all raw files and segmentation files.
        organoid_parent_dir = join(exp.root_dir, plate.plate_id, "obj_v0.3_ROI")
        if exists(organoid_parent_dir):
            organoids = glob(join(organoid_parent_dir, well_id, "*"))
            for organoid in tqdm(organoids, leave=False):
                org = OrganoidRecord(w, basename(organoid), save_dir=w.well_dir)
                organoid_files = glob(join(organoid, '*.tif'))
                for f in organoid_files:
                    raw_channel = raw_ch_regex.findall(f)
                    mask = mask_regex.findall(f)
                    nuc_seg = nuc_seg_regex.findall(f)
                    cell_seg = cell_seg_regex.findall(f)
                    # cell seg would go here
                    if len(raw_channel) > 0:
                        org.add_raw_data(raw_channel[0][:3], f, spacing=(0.6, 0.216, 0.216))
                    elif len(mask) > 0:
                        org.add_segmentation(mask[0][:-4], f)
                        # cell seg would go here
                    elif len(nuc_seg) > 0:
                        org.add_segmentation(nuc_seg[0][:-4], f)
                    elif len(cell_seg) > 0:
                        org.add_segmentation(cell_seg[0][:-4], f)
                    else:
                        print("Unknown file: {}".format(f))

# Save the whole datastructure to disk.
exp.save()

# Get an overview dataframe.
# The columns correspond to the different names of added raw-data and segmentation data.
exp.build_overview()

# First iterate over wells and extract OVR-level features
# Iterate over wells only
exp.only_iterate_over_wells(True)
exp.reset_iterator()
ovr_channel = "C01"  # almost always DAPI is C01 -- if not, change this!

for well in exp:
    # Load the segmentation file.
    ovr_seg = well.get_segmentation(ovr_channel)  # this is the seg image

    if well.plate.plate_id in ["day4p5"]:
        continue  # skip these timepoints

    if ovr_seg is not None:
        # calculate global coordinates
        # flag org touching tile borders
        ovr_seg_org = ovr_seg[0]  # first image is label map
        ovr_seg_tiles = ovr_seg[1]  # second image is drogon tiling
        ovr_seg_tiles[ovr_seg_tiles > 0] = 1  # binarize tiles image

        tile_borders = (ovr_seg_tiles - binary_erosion(ovr_seg_tiles)).astype(bool)  # generate tile borders

        touching_labels = np.unique(ovr_seg_org[tile_borders])  # includes the 0 background label

        lst = ['object_' + str(x) for x in
               touching_labels[touching_labels > 0]]  # create list of labels and remove 0 background label

        # feature extraction
        df_ovr = pd.DataFrame()
        ovr_features = regionprops(ovr_seg_org)

        for obj in ovr_features:
            organoid_id = 'object_' + str(obj['label'])
            row = {
                'hcs_experiment': well.plate.experiment.name,
                'plate_id': well.plate.plate_id,
                'well_id': well.well_id,
                'organoid_id': organoid_id,
                'segmentation_ovr': well.segmentations[ovr_channel],
                'flag_tile_border': organoid_id in lst,  # TRUE (1) means organoid is touching a tile border
                'x_pos_pix_global': obj['centroid'][1],
                'y_pos_pix_global': obj['centroid'][0],
                'area_pix_global': obj['area']
            }

            df_ovr = pd.concat([df_ovr, pd.DataFrame.from_records([row])], ignore_index=True)
            # df_ovr = df_ovr.append(row,ignore_index=True)

        # Save measurement into the well directory.
        name = "regionprops_ovr_" + str(ovr_channel)
        path = join(well.well_dir, name + ".csv")
        df_ovr.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        well.add_measurement(name, path)
        well.save()  # updates json file

# Set the experiment to iterate over all organoids. This is the default.
# After iterating over it the iterator has to be reset.
exp.only_iterate_over_wells(False)
exp.reset_iterator()

for organoid in exp:
    # Load the segmentation file.
    nuc_seg = organoid.get_segmentation(nuc_ending)  # load segmentation images
    mem_seg = organoid.get_segmentation(mem_ending)  # load segmentation images
    org_seg = organoid.get_segmentation(mask_ending)
    # Only continue data loading if the segmentation exists.
    if org_seg is None:
        continue  # skip organoids that don't have a mask (this will never happen)

    if organoid.well.plate.plate_id in ["day4p5"]:
        continue  # skip these timepoints

    # for each raw image, extract organoid-level features
    for channel in organoid.raw_files:
        # Load raw data.
        raw = organoid.get_raw_data(channel)  # this is the raw image

        # create organoid MIP
        raw_mip = np.zeros(org_seg.shape)  # initialize array to dimensions of mask image
        for plane in raw:
            raw_mip = np.maximum(raw_mip, plane)

        # organoid feature extraction
        df_org = pd.DataFrame()
        org_features = regionprops(org_seg, raw_mip, extra_properties=(quartiles, skewness, kurtos, stdv))
        abs_min_intensity = np.amin(raw_mip)
        # voxel_area = organoid.spacings[channel][1] * organoid.spacings[channel][2] #calculate voxel area in um2 (x*y)

        for obj in org_features:
            box_area = obj['area'] / (org_seg.shape[0] * org_seg.shape[1])  # for filtering of wrong segmentations
            circularity = (4 * math.pi * (obj['area'] / (math.pow(obj['perimeter'], 2))))

            row = {
                'hcs_experiment': organoid.well.plate.experiment.name,
                'root_dir': organoid.well.plate.experiment.root_dir,
                'plate_id': organoid.well.plate.plate_id,
                'well_id': organoid.well.well_id,
                'channel_id': channel,
                'object_type': 'organoid',
                'organoid_id': organoid.organoid_id,
                'org_label': organoid.organoid_id.rpartition("_")[2],
                'segmentation_org': organoid.segmentations[mask_ending],
                'intensity_img': organoid.raw_files[channel],
                'x_pos_pix': obj['centroid'][1],
                'y_pos_pix': obj['centroid'][0],
                'x_pos_weighted_pix': obj['weighted_centroid'][1],
                'y_pos_weighted_pix': obj['weighted_centroid'][0],
                'x_massDisp_pix': obj['weighted_centroid'][1] - obj['centroid'][1],
                'y_massDisp_pix': obj['weighted_centroid'][0] - obj['centroid'][0],
                'mean_intensityMIP': obj['mean_intensity'],
                'max_intensity': obj['max_intensity'],
                'min_intensity': obj['min_intensity'],
                'abs_min': abs_min_intensity,
                'area_pix': obj['area'],
                #                 'area_um2':obj['area'] * voxel_area,
                'eccentricity': obj['eccentricity'],
                'majorAxisLength': obj['major_axis_length'],
                'minorAxisLength': obj['minor_axis_length'],
                'axisRatio': obj['minor_axis_length'] / obj['major_axis_length'],
                'eulerNumber': obj['euler_number'],  # for filtering wrong segmentations
                'objectBoxRatio': box_area,  # for filtering wrong segmentations
                'perimeter': obj['perimeter'],
                'circularity': circularity,
                'quartile25': obj['quartiles'][0],
                'quartile50': obj['quartiles'][1],
                'quartile75': obj['quartiles'][2],
                'quartile90': obj['quartiles'][3],
                'quartile95': obj['quartiles'][4],
                'quartile99': obj['quartiles'][5],
                'stdev': obj['stdv'],
                'skew': obj['skewness'],
                'kurtosis': obj['kurtos']
            }

            df_org = pd.concat([df_org, pd.DataFrame.from_records([row])], ignore_index=True)
            # df_org = df_org.append(row,ignore_index=True)

        # Save measurement into the organoid directory.
        name = "regionprops_org_" + str(channel)
        path = join(organoid.organoid_dir, name + ".csv")
        df_org.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file

        # NUCLEAR feature extraction
        if nuc_seg is None:
            continue  # skip organoids that don't have a nuclear segmentation

        # organoid feature extraction
        df_nuc = pd.DataFrame()

        # make binary organoid mask and crop nuclear labels to this mask
        # extract nuclei features that only belong to organoid of interest and exclude pieces of neighboring organoids
        org_seg_binary = copy.deepcopy(org_seg)
        org_seg_binary[org_seg_binary > 0] = 1
        nuc_seg = nuc_seg * org_seg_binary

        nuc_features = regionprops(nuc_seg, raw, extra_properties=(quartiles, skewness, kurtos, stdv),
                                   spacing=(3, 1, 1))
        # voxel_volume = organoid.spacings[channel][0] * organoid.spacings[channel][1] * organoid.spacings[channel][2] #calculate voxel area in um2 (x*y)
        # https://www.analyticsvidhya.com/blog/2022/01/moments-a-must-known-statistical-concept-for-data-science/
        # mean is first raw moment
        # variance is the second central moment
        # skewness is the third normalized moment
        # kurtosis is the fourth standardize moment

        for nuc in nuc_features:
            row = {
                'hcs_experiment': organoid.well.plate.experiment.name,
                'root_dir': organoid.well.plate.experiment.root_dir,
                'plate_id': organoid.well.plate.plate_id,
                'well_id': organoid.well.well_id,
                'channel_id': channel,
                'object_type': 'nucleus',
                'organoid_id': organoid.organoid_id,
                'org_label': organoid.organoid_id.rpartition("_")[2],
                'nuc_id': int(nuc['label']),
                'segmentation_nuc': organoid.segmentations[nuc_ending],
                'intensity_img': organoid.raw_files[channel],
                'x_pos_vox': nuc['centroid'][2],
                'y_pos_vox': nuc['centroid'][1],
                'z_pos_vox': nuc['centroid'][0],
                #                 'x_pos_um':nuc['centroid'][2]*organoid.spacings[channel][2],
                #                 'y_pos_um':nuc['centroid'][1]*organoid.spacings[channel][1],
                #                 'z_pos_um':nuc['centroid'][0]*organoid.spacings[channel][0],
                'volume_pix': nuc['area'],
                #                 'volume_um3':nuc['area'] * voxel_volume,
                'mean_intensity': nuc['mean_intensity'],
                'max_intensity': nuc['max_intensity'],
                'min_intensity': nuc['min_intensity'],
                'quartile25': nuc['quartiles'][0],
                'quartile50': nuc['quartiles'][1],
                'quartile75': nuc['quartiles'][2],
                'quartile90': nuc['quartiles'][3],
                'quartile95': nuc['quartiles'][4],
                'quartile99': nuc['quartiles'][5],
                'stdev': nuc['stdv'],
                'skew': nuc['skewness'],
                'kurtosis': nuc['kurtos']
            }

            df_nuc = pd.concat([df_nuc, pd.DataFrame.from_records([row])], ignore_index=True)
            # df_nuc = df_nuc.append(row,ignore_index=True)

        # Save measurement into the organoid directory.
        name = "regionprops_nuc_" + str(channel)
        path = join(organoid.organoid_dir, name + ".csv")
        df_nuc.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file

        # MEMBRANE feature extraction
        if mem_seg is None:
            continue  # skip organoids that don't have a cell segmentation

        # organoid feature extraction
        df_mem = pd.DataFrame()

        # make binary organoid mask and crop cell labels to this mask
        # MAYBE EXPAND BINARY MASK??
        mem_seg = mem_seg * org_seg_binary

        mem_features = regionprops(mem_seg, raw, extra_properties=(quartiles, skewness, kurtos, stdv),
                                   spacing=(3, 1, 1))

        for mem in mem_features:
            row = {
                'hcs_experiment': organoid.well.plate.experiment.name,
                'root_dir': organoid.well.plate.experiment.root_dir,
                'plate_id': organoid.well.plate.plate_id,
                'well_id': organoid.well.well_id,
                'channel_id': channel,
                'object_type': 'membrane',
                'organoid_id': organoid.organoid_id,
                'org_label': organoid.organoid_id.rpartition("_")[2],
                'mem_id': int(mem['label']),
                'segmentation_mem': organoid.segmentations[mem_ending],
                'intensity_img': organoid.raw_files[channel],
                'x_pos_vox': mem['centroid'][2],
                'y_pos_vox': mem['centroid'][1],
                'z_pos_vox': mem['centroid'][0],
                #                 'x_pos_um':mem['centroid'][2]*organoid.spacings[channel][2],
                #                 'y_pos_um':mem['centroid'][1]*organoid.spacings[channel][1],
                #                 'z_pos_um':mem['centroid'][0]*organoid.spacings[channel][0],
                'volume_pix': mem['area'],
                #                 'volume_um3':mem['area'] * voxel_volume,
                'mean_intensity': mem['mean_intensity'],
                'max_intensity': mem['max_intensity'],
                'min_intensity': mem['min_intensity'],
                'quartile25': mem['quartiles'][0],
                'quartile50': mem['quartiles'][1],
                'quartile75': mem['quartiles'][2],
                'quartile90': mem['quartiles'][3],
                'quartile95': mem['quartiles'][4],
                'quartile99': mem['quartiles'][5],
                'stdev': mem['stdv'],
                'skew': mem['skewness'],
                'kurtosis': mem['kurtos']
            }

            df_mem = pd.concat([df_mem, pd.DataFrame.from_records([row])], ignore_index=True)
            # df_mem = df_mem.append(row,ignore_index=True)

        # Save measurement into the organoid directory.
        name = "regionprops_mem_" + str(channel)
        path = join(organoid.organoid_dir, name + ".csv")
        df_mem.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file


#LINKING
# Load an existing faim-hcs Experiment from disk.
exp = Experiment()
exp.load('/tungstenfs/scratch/gliberal/Users/repinico/Microscopy/Analysis/20220528_GCPLEX_redo/20220507GCPLEX_R0/summary.csv')

iop_cutoff = 0.6

exp.only_iterate_over_wells(False)
exp.reset_iterator()

ovr_channel = "C01"

for organoid in exp:
    # nuclear feature extraction
    nuc_seg = organoid.get_segmentation(nuc_ending)  # load segmentation images
    mem_seg = organoid.get_segmentation(mem_ending)  # load segmentation images
    # org_seg = organoid.get_segmentation("MASK")
    org_seg = organoid.get_segmentation(mask_ending)

    if nuc_seg is None:
        continue  # skip organoids that don't have a nuclear segmentation
    if mem_seg is None:
        continue  # skip organoids that don't have a membrane segmentation

    org_seg_binary = copy.deepcopy(org_seg)
    org_seg_binary[org_seg_binary > 0] = 1

    nuc_seg = nuc_seg * org_seg_binary
    mem_seg = mem_seg * org_seg_binary

    # match each nuclear label to a cell label
    stat = matching(mem_seg, nuc_seg, criterion='iop', thresh=iop_cutoff, report_matches=True)

    #     print(stat[2], 'out of', stat[10], 'nuclei are not surrounded by a cell')
    #     print(stat[4], 'out of', stat[9], 'cells do not contain a nucleus')

    match = pd.DataFrame(list(zip([x[0] for x in stat[14]], [x[1] for x in stat[14]], stat[15])),
                         columns=['mem_id', 'nuc_id', 'iop'])
    match_filt = match.loc[(match["iop"] > iop_cutoff)].copy(deep=True)  # this is the linking df

    # update all organoid measurements with numbers of nuclei and membrane detected and linked
    for meas_name in [k for k, v in organoid.measurements.items() if k.startswith('regionprops_org')]:
        meas = organoid.get_measurement(meas_name)

        # add columns to dataframe
        meas['nuc_without_mem'] = stat[2]
        meas['nuc_total'] = stat[10]
        meas['mem_without_nuc'] = stat[4]
        meas['mem_total'] = stat[9]

        name = str(meas_name)
        path = join(organoid.organoid_dir, name + ".csv")
        meas.to_csv(path, index=False)  # saves csv

        # Add the measurement to the faim-hcs datastructure and save.
        organoid.add_measurement(name, path)
        organoid.save()  # updates json file

    # add metadata and save linking
    match_filt['hcs_experiment'] = organoid.well.plate.experiment.name
    match_filt['root_dir'] = organoid.well.plate.experiment.root_dir
    match_filt['plate_id'] = organoid.well.plate.plate_id
    match_filt['well_id'] = organoid.well.well_id
    match_filt['channel_id'] = ovr_channel
    match_filt['organoid_id'] = organoid.organoid_id
    match_filt['org_label'] = organoid.organoid_id.rpartition("_")[2]

    # Save measurement into the organoid directory.
    name = "linking_nuc_to_mem"
    path = join(organoid.organoid_dir, name + ".csv")
    match_filt.to_csv(path, index=False)  # saves csv

    # Add the measurement to the faim-hcs datastructure and save.
    organoid.add_measurement(name, path)
    organoid.save()  # updates json file
