import pandas as pd
import numpy as np
import math
from skimage.exposure import rescale_intensity
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt



### Functions: heatmap plotting of plate data

# mask missing values; if input data has Nan values will be displayed black
def plot_heatmap(df_hm, cmap, vmin, vmax, annot = True):
    plt.figure(figsize=(10,10))
    f1 = sns.heatmap(data= df_hm,
            linewidth = 0.5,
            linecolor='white',
            square = True,
            cmap=cmap,
            robust = False,
            annot=annot,
            clip_on=False,
            mask=df_hm.isnull(),
            vmin = vmin,
            vmax = vmax)

    f1.xaxis.set_ticks_position("top")
    f1.tick_params(left=False, top=False, bottom=False, right=False)
    return f1

# Function modified from Maurice
def build_heatmap_df(plate_size):
    
    if plate_size == 384:
        index_lst = ["A", "B" , "C", "D", "E", "F", "G", "H", "I", "J", "K" , "L" , "M" , "N" , "O", "P"]
        col_lst = [str(x) for x in range(1,25)]
        for i,el in enumerate(col_lst):
            # make all col numbers two-digit (i.e. add 0 to single-digit columns)
            if len(el) == 1:
                col_lst[i] = str(0)+el
                
    if plate_size == 96:
        index_lst = ["A", "B" , "C", "D", "E", "F", "G", "H"]
        col_lst = [str(x) for x in range(1,13)]
        for i,el in enumerate(col_lst):
            if len(el) == 1:
                col_lst[i] = str(0)+el
        
    df_plot_HM = pd.DataFrame(np.nan, index = index_lst, columns = col_lst)

    if (plate_size != 96) and (plate_size != 384):
        print("plate size not configured.")

    return df_plot_HM


# TODO: set colorscale limits to max across all plates so that scaling is the same between plates
def plot_heatmap_means(df, feature = 'C03.mean_intensity', plate_size = 96, vmax_multiplier = 1):
    vmin = df[feature].min()
    vmax = df[feature].max()
    vmax = vmax * vmax_multiplier
    for plate in np.unique(df['plate_id']):
        df_hm = build_heatmap_df(plate_size = plate_size)
        df_plt = df[(df.plate_id == plate)].copy(deep = True)
        grouped = df_plt.groupby(["well_id"])[feature].mean().to_frame()

        for well in grouped.index:
            df_hm.loc[well[0], well[1:]] = grouped.loc[well][feature]

        hm = plot_heatmap(df_hm, 'viridis', vmin, vmax, annot = False)

        hm.set_title(plate + "\n", loc = 'left')
        plt.subplots_adjust(top = 0.6)
        
        
 ### Functions: gridded image plotting across conditions (RGB and single-channel)


# find lowest value in all arrays for min_quantile
# find highest value in all arrays for max_quantile
def quantile_value_across_arrays(array_list,min_quantile,max_quantile):
    
    max_value = float('-inf') # Initialize
    min_value = float('inf') # Initialize

    for array in array_list:
        quantile_in_array = np.quantile(array, [min_quantile,max_quantile])
        max_value = max(max_value, quantile_in_array[1])
        min_value = min(min_value, quantile_in_array[0])
    
    return (min_value, max_value)


def make_rgb_imagescale(r,g,b, min_quantile = 0, max_quantile = 0.999):
    
    # normalize so range is 0-1
    r = r / 65535.0
    g = g / 65535.0
    b = b / 65535.0

    # rescale to intensity
    if np.any(r):
        r = rescale_intensity(r, in_range=tuple(np.quantile(r, [min_quantile,max_quantile])), out_range = np.float64)
    if np.any(g):
        g = rescale_intensity(g, in_range=tuple(np.quantile(g, [min_quantile,max_quantile])), out_range = np.float64)
    if np.any(b):
        b = rescale_intensity(b, in_range=tuple(np.quantile(b, [min_quantile,max_quantile])), out_range = np.float64)

    # make RGB stack
    rgb = (np.dstack((r,g,b)))

    # since float values can be quite low, first normalize to max value of array and then convert to 255
    rgb_uint8 = ((rgb / np.amax(rgb)) * 255.0).astype('uint8')
    
    return rgb_uint8


#r_range is a tuple of (min_value, max_value) passed to rescale_intensity
def make_rgb_globalscale(r,g,b, r_range, g_range, b_range):
    # normalize so range is 0-1
    r = r / 65535.0
    g = g / 65535.0
    b = b / 65535.0

    # rescale to intensity
    r_range = tuple(t/65535.0 for t in r_range)
    g_range = tuple(t/65535.0 for t in g_range)
    b_range = tuple(t/65535.0 for t in b_range)
        
    if np.any(r):
        r = rescale_intensity(r, in_range=r_range, out_range = np.float64)
    if np.any(g):
        g = rescale_intensity(g, in_range=g_range, out_range = np.float64)
    if np.any(b):
        b = rescale_intensity(b, in_range=b_range, out_range = np.float64)

    # make RGB stack
    rgb = (np.dstack((r,g,b)))

    # since float values can be quite low, first normalize to max value of array and then convert to 255
    rgb_uint8 = ((rgb / np.amax(rgb)) * 255.0).astype('uint8')
    
    return rgb_uint8


# plot objects, one channel
def plot_image_grid(roi_npimg_dict, brighten = 0.5, ncols=None, cmap='gray'):
    '''Plot a grid of images'''
    
    imgs = [roi_npimg_dict[cond][obj][0] for cond in sorted(set(roi_npimg_dict.keys())) for obj in sorted(set(roi_npimg_dict[cond].keys()))]
    plate_id = [obj[0] for cond in sorted(set(roi_npimg_dict.keys())) for obj in sorted(set(roi_npimg_dict[cond].keys()))]
    well_id = [obj[1] for cond in sorted(set(roi_npimg_dict.keys())) for obj in sorted(set(roi_npimg_dict[cond].keys()))]
    org_id = [obj[2] for cond in sorted(set(roi_npimg_dict.keys())) for obj in sorted(set(roi_npimg_dict[cond].keys()))] #note this value corresponds to organoid labelmap value (NOT feature index or FOV value)
    # note that if there are 0 organoids in a condition, it is skipped
    cond_id = [cond for cond in sorted(set(roi_npimg_dict.keys())) for obj in sorted(set(roi_npimg_dict[cond].keys()))]
    
    conditions = sorted(np.unique(cond_id))
    nrows = len(conditions) # nrows is the number of conditions
    
    ncols = float('-inf') # Initialize
    for array in roi_npimg_dict.values():
        array_length = len(array)
        ncols = max(ncols, array_length)
    
    
    f, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True, tight_layout=True)
    
    xmax = np.amax([img.shape[1] for img in imgs])
    ymax = np.amax([img.shape[0] for img in imgs])
    umax = np.amax([xmax, ymax]) # set box shape to square, and max of all organoid dims
    
    img_count = 0 # count number of images total in figure (including blanks)
    plotted_count = 0 # count number of images already plotted
    
    for row in range(nrows):
        for col in range(ncols):
            plot_cond = conditions[row]
            ax=axes[row][col] # axes for next image slot
            # if the image to be plotted is the correct condition for this row, plot it. if not, plot empty img
            if cond_id[plotted_count] == plot_cond:
                img=imgs[plotted_count]
                p=plate_id[plotted_count]
                w=well_id[plotted_count]
                o=org_id[plotted_count]
                c=cond_id[plotted_count]
                ax.imshow(img, cmap=cmap, vmax = brighten * np.amax(img), origin = 'upper', extent=[(umax/2)-img.shape[1]/2., (umax/2)+img.shape[1]/2., (umax/2)-img.shape[0]/2., (umax/2)+img.shape[0]/2. ])
                ax.set_xlim(0, umax)
                ax.set_ylim(umax, 0)
                plotted_count += 1 # increment plotted count to mark image as "plotted"
            else:
                img = 0*imgs[0]
                p='None'
                w='None'
                o='None'
                c=plot_cond
                ax.imshow(img, cmap=cmap, vmax = brighten * np.amax(img), origin = 'upper', extent=[(umax/2)-img.shape[1]/2., (umax/2)+img.shape[1]/2., (umax/2)-img.shape[0]/2., (umax/2)+img.shape[0]/2. ])
                ax.set_xlim(0, umax)
                ax.set_ylim(umax, 0)
            
            if img_count % ncols == 0:
            # for first image of row, set cond in title name
                ax.set_title(c + "\n" + p+' '+w+' '+o)
            else:
                ax.set_title(p+' '+w+' '+o)
        
            img_count += 1 # to count images for setting condition title



def plot_rgb_grid(r_dict, g_dict, b_dict, brighten = 0.5, ncols=None, min_quantile = 0, max_quantile = 0.999,
                 global_norm = False, auto_range = True, ranges = ()):
    '''Plot a grid of images'''
    # ranges is tuple of ranges in order r,g,b ex. ((100,200), (120,3000), (120, 4567))
    
    if np.any(r_dict):
        metadict = r_dict
    elif np.any(g_dict):
        metadict = g_dict
    else:
        metadict = b_dict


    # load images from dict
    if np.any(r_dict):
        r_imgs = [r_dict[cond][obj][0] for cond in sorted(set(metadict.keys())) for obj in sorted(set(metadict[cond].keys()))]
    if np.any(g_dict):
        g_imgs = [g_dict[cond][obj][0] for cond in sorted(set(metadict.keys())) for obj in sorted(set(metadict[cond].keys()))]
    if np.any(b_dict):
        b_imgs = [b_dict[cond][obj][0] for cond in sorted(set(metadict.keys())) for obj in sorted(set(metadict[cond].keys()))]
    # add else statements for what happens for empty zero arrays 
    
    if global_norm:
        if auto_range:
            r_range = quantile_value_across_arrays(r_imgs,min_quantile,max_quantile)
            g_range = quantile_value_across_arrays(g_imgs,min_quantile,max_quantile)
            b_range = quantile_value_across_arrays(b_imgs,min_quantile,max_quantile)
        else:
            r_range = ranges[0]
            g_range = ranges[1]
            b_range = ranges[2]
        print('r_range: ', r_range, 'g_range: ', g_range, 'b_range: ', b_range)

    
    plate_id = [obj[0] for cond in sorted(set(metadict.keys())) for obj in sorted(set(metadict[cond].keys()))]
    well_id = [obj[1] for cond in sorted(set(metadict.keys())) for obj in sorted(set(metadict[cond].keys()))]
    org_id = [str(int(obj[2])+1) for cond in sorted(set(metadict.keys())) for obj in sorted(set(metadict[cond].keys()))] #note this value corresponds to organoid labelmap value (NOT feature index or FOV value)
    cond_id = [cond for cond in sorted(set(metadict.keys())) for obj in sorted(set(metadict[cond].keys()))]
    
    conditions = sorted(np.unique(cond_id))
    nrows = len(conditions) # nrows is the number of conditions
    ncols = math.ceil(sum(len(v) for v in metadict.values()) / len(metadict)) # ncols is number of samples
    
    
    f, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True, tight_layout=True)

    
    #would break if r_dict is a zero array! FIX
    xmax = np.amax([img.shape[1] for img in r_imgs])
    ymax = np.amax([img.shape[0] for img in r_imgs])
    umax = np.amax([xmax, ymax]) # set box shape to square, and max of all organoid dims
    
    img_count = 0 # count number of images total in figure (including blanks)
    plotted_count = 0 # count number of images already plotted

    
    for row in range(nrows):
        for col in range(ncols):
            plot_cond = conditions[row]
            ax=axes[row][col] # axes for next image slot
            # if the image to be plotted is the correct condition for this row, plot it. if not, plot empty img
            if cond_id[plotted_count] == plot_cond:
                r = r_imgs[plotted_count]
                g = g_imgs[plotted_count]
                b = b_imgs[plotted_count]

                p=plate_id[plotted_count]
                w=well_id[plotted_count]
                o=org_id[plotted_count]
                c=cond_id[plotted_count]

                if global_norm:
                    rgb_uint8 = make_rgb_globalscale(r,g,b, r_range, g_range, b_range)
                else:
                    rgb_uint8 = make_rgb_imagescale(r,g,b, min_quantile, max_quantile)

                ax.imshow(rgb_uint8, origin = 'upper',
                         extent=[(umax/2)-rgb_uint8.shape[1]/2., (umax/2)+rgb_uint8.shape[1]/2., (umax/2)-rgb_uint8.shape[0]/2., (umax/2)+rgb_uint8.shape[0]/2. ])
                ax.set_xlim(0, umax)
                ax.set_ylim(umax, 0)

                plotted_count += 1 # increment plotted count to mark image as "plotted"
            #TODO would not work if first condition is empty and rgb_uint8 does not exist
            else:
                r = r*0
                g = g*0
                b = b*0
                p='None'
                w='None'
                o='None'
                c=plot_cond
                ax.imshow(rgb_uint8*0, origin = 'upper',
                         extent=[(umax/2)-rgb_uint8.shape[1]/2., (umax/2)+rgb_uint8.shape[1]/2., (umax/2)-rgb_uint8.shape[0]/2., (umax/2)+rgb_uint8.shape[0]/2. ])
                ax.set_xlim(0, umax)
                ax.set_ylim(umax, 0)

            if img_count % ncols == 0:
            # for first image of row, set cond in title name
                ax.set_title(c + "\n" + p+' '+w+' '+o)
            else:
                ax.set_title(p+' '+w+' '+o)

            img_count += 1 # to count images for setting condition title
            

# specify which condition and organoid to load from image dictionary (output of load_imgs_from_object_dict)
def plot_single_image(c01_npimg_dict, cond = 'd3-P1.04', plate_id = '20230712-d3-P1', well_id = 'B04', org_id = '12'):
    im = c01_npimg_dict[cond][(plate_id, well_id, org_id)][0]
    plt.figure(figsize=(5,5))
    plt.imshow(im, cmap='gray')
    plt.colorbar(shrink=0.8)
    
def plot_single_rgb(r_dict, g_dict, b_dict, cond = 'd3-P1.04', plate_id = '20230712-d3-P1', well_id = 'B04', org_id = '12'):
    b = b_dict[cond][(plate_id, well_id, org_id)][0]
    g = g_dict[cond][(plate_id, well_id, org_id)][0]
    r = r_dict[cond][(plate_id, well_id, org_id)][0]
    rgb_uint8 = make_rgb_imagescale(r,g,b,min_quantile = 0, max_quantile = 0.99)
    plt.figure(figsize=(5,5))
    plt.imshow(rgb_uint8)
    
#extract np array images from nested dictionary, into list of np arrays
def extract_imgs(img_dict):
    imgs = [img_dict[cond][obj][0] for cond in sorted(set(img_dict.keys())) for obj in sorted(set(img_dict[cond].keys()))]
    return imgs

# grouped is output of count_positive_fraction
#r/g/b_ch_idx are channel indeces (usually 0,1,2,3) to be used for r,g,b, channels resp.
def plot_pos_and_neg_sets(df_filtered, grouped, inv_cond, 
                          zarr_url_dict, roi_name, n_obj, seed, level, 
                          min_quantile, max_quantile,
                          r_ch_idx =2, g_ch_idx=1, b_ch_idx=0):
    
    positive = df_filtered.loc[(df_filtered['pos.count'] == 1)]['oUID_tuple'].tolist()
    negative = df_filtered.loc[(df_filtered['pos.count'] == 0)]['oUID_tuple'].tolist()

    all_objects = make_object_dict(inv_cond, zarr_url_dict, roi_name)

    neg_objects = make_filtered_dict(all_objects, negative, omit_my_list = False)
    pos_objects = make_filtered_dict(all_objects, positive, omit_my_list = False)
    

    neg_objects_randomized = randomize_object_dict(neg_objects, n_obj = n_obj, seed = seed)
    pos_objects_randomized = randomize_object_dict(pos_objects, n_obj = n_obj, seed = seed)

    b_neg_dict = load_imgs_from_object_dict(neg_objects_randomized,
                                                zarr_url_dict,
                                                channel_index = b_ch_idx,
                                                level=level,
                                                roi_name = roi_name, 
                                                reset_origin=False)

    g_neg_dict = load_imgs_from_object_dict(neg_objects_randomized,
                                                zarr_url_dict,
                                                channel_index = g_ch_idx,
                                                level=level,
                                                roi_name = roi_name, 
                                                reset_origin=False)

    r_neg_dict = load_imgs_from_object_dict(neg_objects_randomized,
                                                zarr_url_dict,
                                                channel_index = r_ch_idx,
                                                level=level,
                                                roi_name = roi_name, 
                                                reset_origin=False)

    b_pos_dict = load_imgs_from_object_dict(pos_objects_randomized,
                                                zarr_url_dict,
                                                channel_index = b_ch_idx,
                                                level=level,
                                                roi_name = roi_name, 
                                                reset_origin=False)

    g_pos_dict = load_imgs_from_object_dict(pos_objects_randomized,
                                                zarr_url_dict,
                                                channel_index = g_ch_idx,
                                                level=level,
                                                roi_name = roi_name, 
                                                reset_origin=False)

    r_pos_dict = load_imgs_from_object_dict(pos_objects_randomized,
                                                zarr_url_dict,
                                                channel_index = r_ch_idx,
                                                level=level,
                                                roi_name = roi_name, 
                                                reset_origin=False)
    # calculate global quantile ranges
    r_neg_imgs = extract_imgs(r_neg_dict)
    r_pos_imgs = extract_imgs(r_pos_dict)
    g_neg_imgs = extract_imgs(g_neg_dict)
    g_pos_imgs = extract_imgs(g_pos_dict)
    b_neg_imgs = extract_imgs(b_neg_dict)
    b_pos_imgs = extract_imgs(b_pos_dict)
    
    r_imgs = np.concatenate((r_neg_imgs, r_pos_imgs))
    g_imgs = np.concatenate((g_neg_imgs, g_pos_imgs))
    b_imgs = np.concatenate((b_neg_imgs, b_pos_imgs))
    
    r_range = quantile_value_across_arrays(r_imgs,min_quantile,max_quantile)
    g_range = quantile_value_across_arrays(g_imgs,min_quantile,max_quantile)
    b_range = quantile_value_across_arrays(b_imgs,min_quantile,max_quantile)
    
    ranges = (r_range, g_range, b_range)
        
    print('negative')
    plot_rgb_grid(r_neg_dict, g_neg_dict, b_neg_dict, ncols=None, min_quantile = min_quantile, max_quantile = max_quantile,
                          global_norm = True, auto_range = False, ranges = ranges)

    print('positive')
    plot_rgb_grid(r_pos_dict, g_pos_dict, b_pos_dict, ncols=None, min_quantile = min_quantile, max_quantile = max_quantile,
                          global_norm = True, auto_range = False, ranges = ranges)
        
        
        
### Functions: seaborn plots of intensities and organoid proportions (violin, barplots)


def count_positive_fraction(df, colname, thresh):
    df['count'] = 1
    df['pos.count'] = df[colname].gt(thresh).astype(int)
    grouped = df.groupby(["condition", "plate_id", "well_id", "timepoint"])[['count','pos.count']].sum().reset_index()
    grouped['pos_fraction'] = grouped['pos.count']/ grouped['count']
    
    return df, grouped


def plot_positive_fraction(grouped):
    for tp in grouped["timepoint"].unique():
        df_tp = grouped[(grouped.timepoint == tp)].copy(deep = True)

        ncolors = len(df_tp["condition"].unique()) 
        cond_order = sorted(df_tp["condition"].unique()) # sort alphabetically

        husl = sns.color_palette("husl", ncolors).as_hex()

        plt.figure(figsize= (14,8))
        #cx = sns.violinplot(x="condition", y="pos_fraction", data=df_tp, color="0.15", linewidth = 1.3, order=cond_order)
        # sns.boxenplot(x="condition", y="pos_fraction", data=df_tp, color="grey", width=0.1, order=cond_order)
        cx = sns.barplot(x="condition", y="pos_fraction", data=df_tp, palette = husl, errcolor=".5", order=cond_order)
        cx = sns.stripplot(x="condition", y="pos_fraction", data=df_tp, size=7, color = "white", order=cond_order)

        plt.title(tp, fontsize=12)
        cx.set_ylim(0, 1.1)


        plt.show()

    
# must have timepoint and condition columns
# df is feature extraction df
def plot_feature_violin(df, colname):
    for tp in df["timepoint"].unique():
        df_tp = df[(df.timepoint == tp)].copy(deep = True)

        ncolors = len(df_tp["condition"].unique()) 
        cond_order = sorted(df_tp["condition"].unique()) # sort alphabetically
        # set color palette to the number of conditions being plotted
        husl = sns.color_palette("husl", ncolors).as_hex()

        plt.figure(figsize= (14,8))
        cx = sns.violinplot(x="condition", y=colname, data=df_tp, color="0.15", linewidth = 1.3, order=cond_order)
        sns.stripplot(x="condition", y=colname, hue = "condition", data=df_tp, size=5, palette = husl, hue_order=cond_order, order=cond_order, zorder =1)
        
        cx.legend_.remove()
        
        plt.title(tp, fontsize=12)
        # set edge color and transparency
        for i in range(len(cx.collections)):
            cx.collections[i].set_edgecolor('0.7')
            #cx.collections[i].set_facecolor('red')
            cx.collections[i].set_alpha(1)
        plt.show()