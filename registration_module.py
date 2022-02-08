import openslide
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage import color
import matplotlib.patches as patches
import glob, os
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import scipy.signal
from PIL import Image
from sklearn.feature_extraction import image as sk_image
import staintools

class Registrator():

    def __init__(self, params):

        self.params = params
        self.base_path = params["base_path"]
        self.csv_path = self._get_csv_path()
        self.region_props = pd.read_csv(self.csv_path)
        self.ext_object_paths = self._get_object_paths()

        # if self.stain_normalize:
        #     self._get_stain_normalizer()


    def _get_stain_normalizer(self, method="macenko"):
        target = staintools.read_image(self.ext_object_paths[0])
        self.normalizer = staintools.StainNormalizer(method=method)
        self.normalizer.fit(target)

    def _get_object_paths(self, ext=".png"):
        file_list = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(ext):
                    file_list.append(os.path.join(root, file))

        print("Object Files Found: ", len(file_list))
        print("Paths:")
        [print(x) for x in file_list]

        return file_list[:4]

    def _get_csv_path(self):

        csv_files = []
        for file in glob.glob(os.path.join(self.base_path, "*.csv")):
            csv_files.append(file)

        return csv_files[0]

    def _get_distances(self, dataframe):
        distances = pdist(dataframe.values, metric='euclidean')
        dist_matrix = squareform(distances)

        return dist_matrix

    def check_regionprop_similarities(self, measure="area", percent=0.05):
        # print(self.region_props.drop("slice", axis=1))
        drop_list = [x for x in self.region_props.columns if self.region_props[x].dtype == "object"]
        # print(self.region_props.drop(drop_list, axis=1))

        # remove list values (polygon, image_list etc)
        dropped = self.region_props.drop(drop_list, axis=1).fillna(0)
        area_frame = dropped.iloc[:,1:2]
        
        print(dropped["label"])

    def check_opencv_similarities(self):
        print("OpenCV Checks...")

        for count, object_path in enumerate(self.ext_object_paths):
            for reference_path in self.ext_object_paths:

                if object_path == reference_path:
                    continue
                else:
                    print("Computing: {0} vs {1}".format(object_path, reference_path))
                    self._compute_sim(object_path, reference_path, grid_size=1)

    def _resize_image(self, img, scale_percent=60):
        print('Original Dimensions : ',img.shape)
 
        # scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ',resized.shape)

        return resized

    def _compute_sim(self, object_path, reference_path, grid_size=1, equalize=False, color_normalize=True):
        template = cv2.imread(object_path)
        # template = Image.open(object_path).convert("L")
        # template = self.normalizer.transform(template)
        template = self._resize_image(template)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        print("type: ", type(template))
        reference = cv2.imread(reference_path)
        # img = Image.open(reference_path).convert("L")
        # reference = self.normalizer.transform(reference)
        reference = self._resize_image(reference)
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        if equalize:
            template = cv2.equalizeHist(template)
            reference = cv2.equalizeHist(reference)

        if grid_size == 0:
            score = self._compute_sim_sift(template, reference)
            return score

        else:
            # patches = np.split(template, [grid_size, grid_size])
            patch_size_x = int(template.shape[0]/grid_size)
            patch_size_y = int(template.shape[1]/grid_size)
            print(template.shape)
            print(patch_size_x, patch_size_y)
            # all_patches = sk_image.extract_patches_2d(template, (patch_size_y, patch_size_x))
            M = patch_size_x
            N = patch_size_y
            patches = [template[x:x+M,y:y+N] for x in range(0,template.shape[0],M) for y in range(0,template.shape[1],N) if template[x:x+M,y:y+N].size == M*N]

            for patch in patches:
                score = self._compute_sim_sift(patch, reference)
                # conv = self._convolve(patch, reference)

    def _convolve(self, A, B):

        print("CONV")
        A = 255 - A
        B = 255 - B
        conv = scipy.signal.fftconvolve(B, A, mode = 'same')

        max_ind = np.argwhere(conv == np.max(conv))[0]
        print(max_ind)

        plt.figure()
        plt.imshow(conv)
        plt.plot(max_ind[1], max_ind[0], marker="x", markersize=3)
        plt.figure()
        plt.imshow(B)
        plt.figure()
        plt.imshow(A)
        plt.show()

        return conv

    def _compute_sim_sift(self, template, reference, register=True, plot_matching=True):


        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        try:
            kp1, des1 = sift.detectAndCompute(template,None)
        except Exception as e:
            print(e)
            return 0
        kp2, des2 = sift.detectAndCompute(reference,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(template,kp1,reference,kp2,good,None,flags=2)

        if plot_matching:
            plt.imshow(img3)
            plt.show()

        score = 0
        if len(matches) > 0:
            score = len(good)/len(matches)

        if register:
            if score < 0.005:
                print("Not registering due to unsimilar images; score: ", score)
                # logging.warning(f"{s['filename']} - \tcheck Similarity")
            else:
                print("Score: {0} \n Try registering...".format(score))
                try:
                    template_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good])
                    img_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good])

                    H, status = cv2.findHomography(img_matched_kpts, template_matched_kpts, cv2.RANSAC,5.0)
                    
                    warped_image = cv2.warpPerspective(reference, H, (reference.shape[1], reference.shape[0]))

                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                    ax1.imshow(template)
                    ax1.set_title("Reference")
                    ax2.imshow(warped_image)
                    ax2.set_title("Warped")
                    ax3.imshow(reference)
                    ax3.set_title("Before Warping")
                    fig.tight_layout()
                    plt.show()
                except Exception as e:
                    print("Registration Error: ", e)
                    print("Continuing...")

                # pil_im = Image.fromarray(warped_image)
                # pil_im.save("warped.png")

                # pil_im_temp = Image.fromarray(template)
                # pil_im_temp.save("template.png")

        return score

if __name__ == "__main__":

    params = { 
                "base_path" : "/home/user/Documents/Projects+/results",
                }

    reg = Registrator(params)
    # reg.check_regionprop_similarities()
    reg.check_opencv_similarities()