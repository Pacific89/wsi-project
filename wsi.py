import openslide
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage import color
import matplotlib.patches as patches
import os
import pandas as pd

class WSI():

    def __init__(self, params):
        self.params = params
        self.path = params["path"]
        self.base_path = params["base_path"] # base path for saving results
        self.save = params["save"]
        self.save_intermediate = False
        self.save_objects = False

        if self.save == "all":
            self.save_intermediate = True
            self.save_objects = True
        elif self.save == "objects":
            self.save_objects = True
        elif self.save == "intermediate":
            self.save_intermediate = True
        else:
            raise NotImplementedError

        self.slide_handler = openslide.OpenSlide(params["path"])
        self.ext = params["path"].split(".")[-1]
        self.level_dimensions = self.slide_handler.level_dimensions
        self.dimensions = self.slide_handler.dimensions
        self.downsample = params["downsample"]
        self.bbox_ext = params["bbox_ext"]
        # self.best_level = self.slide_handler.get_best_level_for_downsample(self.downsample)
        self.best_level = 2
            
        self._set_mpp()
        # self.best_level = self.slide_handler.get_best_level_for_downsample()
        self.median_thresh = params["median_thresh"]
        self.closing = params["closing"]
        self.opening = params["opening"]

        print("Processing: ", self.path)
        print("Dimensions: ", self.dimensions)
        print("File Extension of WSI: ", self.ext)
        # print("Resolution [Microns per Pixel]: X/Y : {0}/{1}".format(self.mpp_x, self.mpp_y))
        print("Best Level for Downsampling by {0} : {1}".format(self.downsample, self.best_level))
        # print("MPPX: ", self.mpp_x)
        # print("MPPY: ", self.mpp_y)

    def _set_mpp(self):
        if self.ext == "svs":
            self.mpp_x = float(self.slide_handler.properties["openslide.mpp-x"])
            self.mpp_y = float(self.slide_handler.properties["openslide.mpp-y"])
        if self.ext == "tif":
            spacing = self.slide_handler.properties["philips.DICOM_PIXEL_SPACING"].split("\"")
            self.mpp_x = float(spacing[1])/(10**-3)
            self.mpp_y = float(spacing[3])/(10**-3)

    def print_stats(self):
        print(self.slide_handler.get_best_level_for_downsample(30))

    def get_segmentation(self):

        level = self.best_level
        self.img = np.array(self.slide_handler.read_region((0,0), level, self.level_dimensions[level]))

        # self.img = np.array(self.slide_handler.get_thumbnail((2000,2000)))
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        # self.img_med = cv2.medianBlur(img_hsv[:,:,1], self.median_thresh)
        # self.mask = cv2.adaptiveThreshold(self.img_med, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        blur = cv2.GaussianBlur(img_hsv[:,:,1],(5,5), 0)
        _, self.mask_otsu = cv2.threshold(blur,5,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Morphologygical Operations:

        # Closing
        kernel_close = np.ones((self.closing,self.closing),np.uint8)
        self.mask_closing = cv2.morphologyEx(self.mask_otsu, cv2.MORPH_CLOSE, kernel_close)

        # Opening
        # kernel_open = np.ones((self.opening,self.opening),np.uint8)
        # self.mask_opening = cv2.morphologyEx(self.mask_closing, cv2.MORPH_OPEN, kernel_open)

    def label_regions(self):

        self.regions = measure.label(self.mask_closing, background = 0)
        region_props = measure.regionprops(self.regions)
        region_prop_keys = [x for x in region_props[0]]
        region_props = sorted(region_props, key=lambda x: x.area, reverse=True)
        region_props_table = pd.DataFrame(measure.regionprops_table(self.regions, properties=region_prop_keys))
        # region_props_table = sorted(region_props_table, key=lambda x: x.area, reverse=True)

        areas = [x.area for x in region_props]
        major_areas = self._major_areas_by_diff(areas)

        # print("Regions found: ", len(region_props))
        # print("Region Areas:", areas)
        # print("Major Region Area(s): ", areas[:major_areas])

        self.keep_regions = region_props[:major_areas]
        self.keep_regions_table = region_props_table[region_props_table["area"] > areas[major_areas]]
        self.major_bbox = [x.bbox for x in self.keep_regions]
        # print("Major Region(s) BBOX:", self.major_bbox)

    def _major_areas_by_diff(self, areas):
        """ find largest regions by differentiating sorted area sizes

        Parameters
        ----------
        areas : [type] array
            array of sorted area sizes

        Returns
        -------
        [type] int
            maximum of absolute differentiated array of sorted areas plus one (since start is at zero)
        """
        return np.argmax(abs(np.diff(areas)))+1

    def _get_bbox(self, bbox):

            top = bbox[0]
            left = bbox[1]
            bot = bbox[2]
            right = bbox[3]
            width = right - left
            height = bot - top

            return top, left, bot, right, width, height

    def _extend_bbox(self, top, left, bot, right, width, height):

            new_width = int(width + width*self.bbox_ext)
            new_height = int(height + height*self.bbox_ext)

            top_shift = int((new_height-height)/2)
            left_shift = int((new_width-width)/2)

            new_left = max(0, int(left-left_shift))
            new_top = max(0, int(top-top_shift))

            return new_top, new_left, new_width, new_height

    def generate_overlay(self):

        # plt.figure()
        # plt.imshow(self.img)
        
        # plt.figure()
        # plt.imshow(self.mask)

        # plt.figure()
        # plt.imshow(self.mask_opening)

        # plt.figure()
        # plt.imshow(self.regions)

        fig, ax = plt.subplots()
        ax.imshow(self.img)
        reg = color.label2rgb(self.regions, bg_label=0, bg_color=(255,255,255))
        # print(.shape)
        ax.imshow(reg, alpha=0.5)


        for bbox in self.major_bbox:
            top, left, bot, right, width, height = self._get_bbox(bbox)

            if self.bbox_ext > 0:
                top, left, width, height = self._extend_bbox(top, left, bot, right, width, height)

            rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='g', facecolor='none')

            ax.add_patch(rect)
        
        # if self.save_results:
        save_extension = ".png"
        file_name = self.path.split("/")[-1].split(".")[0] + "_overlay" + save_extension
        file_path = os.path.join(self.base_path, file_name)
        plt.savefig(file_path)

        # plt.show()

    def extract_objects(self):

        object_count = 1
        self.extracted_objects = []
        for region in self.keep_regions:
            bbox = region.bbox
            top, left, bot, right, width, height = self._get_bbox(bbox)

            if self.bbox_ext > 0:
                top, left, width, height = self._extend_bbox(top, left, bot, right, width, height)

            # print(top, width, height)

            bot = min(top+height, self.img.shape[0])
            right = min(left+width, self.img.shape[1])

            print(self.img.shape)
            print(top, left, bot, right)
            object_img = self.img[top:bot, left:right]

            save_extension = ".png"
            file_name = self.path.split("/")[-1].split(".")[0] + "_object_{0}".format(region.label) + save_extension
            file_path = os.path.join(self.base_path, file_name)
            self.extracted_objects.append(file_name)
            
            cv2.imwrite(file_path, object_img)
            object_count += 1


    def run(self):
        self.get_segmentation()
        self.label_regions()

        if self.save_intermediate:
            self.generate_overlay()

        if self.save_objects:
            self.extract_objects()

def get_wsi_paths(directory, ext=".svs"):

    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                print(directory, file, root)
                file_list.append(os.path.join(root, file))

    return file_list


def extract_regionprops(regionprop_list, extracted_objects, base_path):

    print(regionprop_list)

    df = pd.concat(regionprop_list)
    print(df)
    
    # for c, prop in enumerate(regionprop_list):
    #     name = extracted_objects[c]
    #     values = [prop[key] for key in keys[1:]]
    #     values.insert(0, name)

    #     df = df.append(dict(zip(keys, values)), ignore_index=True)
        
    # # print(df)

    csv_path = os.path.join(base_path, "regionprops.csv")
    df.to_csv(csv_path)

    

if __name__ == "__main__":
    path_one = "/home/user/Documents/Master/data/one/one.svs"
    path_normal = "/home/user/Documents/Master/data/normal_one/normal_042.tif"
    path_two = "/home/user/Documents/Master/data/SS7534_1_2_1062344/SS7534_1_2_1062344.svs"
    path_three = "/home/user/Documents/Master/data/SS7534_1_5_1062347/SS7534_1_5_1062347.svs"

    # paths = [path_one, path_two, path_three]

    paths = get_wsi_paths("/home/user/Documents/Master/data")

    params = {  "path" : "",
                "base_path" : "/home/user/Documents/Projects+/results",
                "stain_normalize" : True,
                "save" : "all",
                "median_thresh" : 5,
                "closing" : 50,
                "opening" : 5,
                "downsample" : 40,
                "bbox_ext" : 0
                }

    regionprop_list = []
    extracted_objects = []

    for path in paths:
        params.update({"path" : path})

        wsi = WSI(params)
        wsi.run()
        major_regionsprops = wsi.keep_regions_table
        obj_names = wsi.extracted_objects
        regionprop_list.append(major_regionsprops)

    extract_regionprops(regionprop_list, extracted_objects, params["base_path"])