#!/usr/bin/env python

# Source: https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html

import numpy as np
import cv2 as cv

import argparse
import sys
 
modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

def main():
    base_name = "/workspaces/python_docker/opencv_extra/testdata/stitching/newspaper"
    num_images = 4  # Number of images you have
    input_imgs = [f"{base_name}{i}.jpg" for i in range(1, num_images + 1)]
    output_img = "result.png"

    # read input images
    imgs = []
    for img_name in input_imgs:
        img = cv.imread(img_name)
        if img is None:
            print("can't read image " + img_name)
            sys.exit(-1)
        imgs.append(img)
 
    #![stitching]
    stitcher = cv.Stitcher.create(cv.Stitcher_SCANS)
    # status, scan = stitcher.stitch(imgs)

    stitchedImg = imgs[0]
    i = 1
    for img in imgs[1:]:
        status, stitchedImg = stitcher.stitch([stitchedImg, img])
        if status != cv.Stitcher_OK:
            print(f"Failed on img: {i}")
            print(f"Stitching failed with status code {status}")
            break
        i+=1
 
    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    #![stitching]
 
    cv.imwrite(output_img, scan)
    print("stitching completed successfully. %s saved!" % output_img)
 
    print('Done')

if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
