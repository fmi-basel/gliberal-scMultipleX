import os
import pandas as pd
import anndata as ad
import numpy as np
import warnings
from pathlib import Path

from .functions_roi_loading import read_table, get_metadata, convert_ROI_table_to_indices, load_intensity_roi

# Functions: Zarr reading / Anndata writing utils


def listdir(path, only_dirs=False):
    file_list = os.listdir(path)
    if only_dirs:
        file_list = [file_name for file_name in file_list if os.path.isdir(os.path.join(path, file_name))]
    return file_list


def zarr_wellpaths(exp_path, select_mip = True, make_zarr_url = True):
# loop over plate folders
# plate name is taken as plate id
    well_paths = {}
    plate_ids = {}
    well_ids = {}
    row_ids = {}
    col_ids = {}
    
    for i, plate_id in enumerate(listdir(exp_path, only_dirs=True)):
        path = os.path.join(exp_path, plate_id)
        zarrs = listdir(path, only_dirs=True)
        
        if select_mip:
            zarrf = [match for match in zarrs if "_mip.zarr" in match]
        else: 
            zarrf = [match for match in zarrs if "_mip.zarr" not in match]

        if len(zarrf) > 1:
            raise ValueError('expecting one mip and one full zarr in plate %s' %plate_id)

        if len(zarrf) < 1:
            warnings.warn('no zarr detected in plate %s' %plate_id)

        # select zarr folder
        if len(zarrf) == 1: 
            path = os.path.join(path, zarrf[0])

            # loop over row folders (A, B, C...)
            for j, row_id in enumerate(listdir(path, only_dirs=True)):
                # loop over column folders (01, 02, 03...)
                for k, col_id in enumerate(listdir(os.path.join(path, row_id), only_dirs=True)):
                    # well_id in format B02, C02, etc
                    well_id = row_id + col_id
                    if make_zarr_url:
                        well_path = os.path.join(path, row_id, col_id, '0')
                    else:
                        well_path = os.path.join(path, row_id, col_id)
                    well_key = (str(plate_id), str(well_id))
                    well_paths[well_key] = well_path
                    plate_ids[well_key] = plate_id
                    well_ids[well_key] = well_id
                    row_ids[well_key] = row_id
                    col_ids[well_key] = col_id
                    
    return well_paths, plate_ids, well_ids, row_ids, col_ids


def append_table_to_zarr_url(zarr_url_dict, table_name):
    zarr_url_tables_dict = {}
    for key in zarr_url_dict:
        path = zarr_url_dict[key]
        zarr_url_tables_dict[key] = os.path.join(path, "tables", table_name)
    return zarr_url_tables_dict
   
                
def load_features_for_well(path):
# function modified from Clara & Joel's Fractal_Feature_loading notebook 
    adata = ad.read_zarr(path)
    if adata:
        df = adata.to_df()
        df_labels = adata.obs
        df_labels["index"] = df_labels.index
        df["index"] = df.index
        df = pd.merge(df_labels, df, on="index")
    else:
        df = None
        warnings.warn('empty feature table in %s' %path)
    return df


def make_anndata(df, org_numerics_list, org_obs_list):
    X_tidy_regex = '|'.join(org_numerics_list)
    obs_tidy_regex = '|'.join(org_obs_list)

    X=df.filter(regex=X_tidy_regex)
    obs = df.filter(regex=obs_tidy_regex)

    # parse vars
    X_feats = X.columns

    for x in X_feats:
        if "." not in x:
            X_feats = X_feats.str.replace(x, 'C00.' + x)

    channel_id = pd.Series(X_feats.str.split('.').str[0]).to_frame(name = 'channel_id').T
    feat_name = pd.Series(X_feats.str.split('.').str[1]).to_frame(name = 'feat_name').T

    var = pd.DataFrame(columns = X_feats)

    var = pd.concat([channel_id, feat_name], axis = 0, ignore_index=False)
    var.columns = X_feats
    var = var.T

    adata = ad.AnnData(X = np.array(X),
                       obs = obs,
                       var = var,
                       dtype = np.float32)
    return adata


### Functions: Prep Zarr data for plotting
# invert dictionary of conditions so that each condition is key, set of wells is value
def invert_conditions_dict(conditions):
    inv_cond = {}
    for k, v in conditions.items():
        inv_cond.setdefault(v, set()).add(k)
    return inv_cond
    

# generate dict where keys are conditions 
# and values are all potential plates, wells, and organoids as list of tuples (plate, well_id, org_id as in labelmap)  
def make_object_dict(inv_cond, zarr_url_dict, roi_name): 

    # generate dictionary with file paths to object FOV tables
    zarr_url_roi_dict = append_table_to_zarr_url(zarr_url_dict, roi_name)

    objects_to_randomize = {}

    for cond in sorted(set(inv_cond.keys())): #for each unique condition...

        roi_set = list()

        # for each well load FOV table...
        for well in inv_cond[cond]:
            sample_path = zarr_url_roi_dict[well]
            roi_an = ad.read_zarr(sample_path)
            
            # determine objects from names of rows in anndata (obs_names)
            # note here FOV is NOT org_id from label map; it is the row index of FOV table (usually org_id - 1)
            # assume that org_id = FOV_id + 1; CAUTION may not always be the case?
            
            roi_set_well = list((well + (str(int(FOV)+1),)) for FOV in roi_an.obs_names)
            # generate tuple in form (plate, well_id, org_id)
            roi_set.extend(roi_set_well)

        # add to dict
        objects_to_randomize[cond] = roi_set
    
    return objects_to_randomize
        
    
# choose random set of objects to visualize
# input objects_to_randomize is dict where keys are conditions 
# and values are all potential plates, wells, and organoids as list of tuples (plate, well_id, org_id as in labelmap) 
def randomize_object_dict(objects_to_randomize, n_obj = 6, seed = 1): 
    #n_obj is integer number of objects desired 

    objects_randomized = {}

    rng = np.random.default_rng(seed = seed)
    for cond in sorted(set(objects_to_randomize.keys())):
        roi_set = objects_to_randomize[cond]

        if len(roi_set) > n_obj:
            inds = rng.choice(len(roi_set), n_obj, replace=False)
            roi_set_rand = [roi_set[ind] for ind in inds]
        else: 
            warnings.warn('number of objects is less than the desired random set. selecting all objects in condition %s' %cond)
            roi_set_rand = roi_set

        objects_randomized[cond] = roi_set_rand

    return objects_randomized


# create nested dictionary were key1 is condition, key2 is random object tuple, and value is numpy image of raw MIP (uint16)
# load images for a given channel index
def load_imgs_from_object_dict(objects_randomized, 
                               zarr_url_dict, 
                               channel_index = 0, 
                               level=0,
                               roi_name = "org_ROI_table", 
                               reset_origin=False):

    roi_npimg_dict = {}
    
    # for each condition...
    for cond in sorted(set(objects_randomized.keys())): 
        roi_npimg_dict[cond] = {}

        # for each object in condition...
        for obj in objects_randomized[cond]:
            # extract path (by plate id, well id)
            zarr_url = Path(zarr_url_dict[obj[0:2]])
            # roi is third tuple element and convert to FOV_id
            roi_of_interest = str(int(obj[2])-1)

            npimg, scaleimg = load_intensity_roi(
                zarr_url,
                roi_of_interest,
                channel_index,
                level,
                roi_name,
                reset_origin)

            roi_npimg_dict[cond][obj] = npimg
            
    return roi_npimg_dict


# during organoid filtering or selecting pos/neg orgs for graphing, filter the img dictionary. 
# can randomize output of this function
def make_filtered_dict(all_objects, my_object_list, omit_my_list = False):
    
    filtered_objects = {}

    
    for cond in sorted(set(all_objects.keys())):
        roi_set = all_objects[cond]
        
        if omit_my_list:
            # select images that are NOT in my_object_list
            roi_set_filt=list(filter(lambda i: i not in list(my_object_list), roi_set))
        else:
            # select images that are in my_object_list
            roi_set_filt=list(filter(lambda i: i in list(my_object_list), roi_set))


        filtered_objects[cond] = roi_set_filt
    
    return filtered_objects
