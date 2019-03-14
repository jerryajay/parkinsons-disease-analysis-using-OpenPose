import matplotlib
matplotlib.use('Agg')
import glob
import traceback

import os

import cv2 as cv
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import pylab as plt
import re
import sys
import skvideo.io
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

def tryint(s):
    try: return int(s)
    except: return s

def alphanum_key(s):
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

# Read input
images_read = glob.glob('/home/jerryant/Desktop/skeleton/image1/*.jpg')			# <----------- Change Your Input Folder Path here!
images_read.sort(key = alphanum_key)
param, model = config_reader()

caffe.set_mode_gpu()
caffe.set_device(2)  # set to your device!
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

# Model from OpenCV to run a bounding box
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# plt.ion()
for ik in range(len(images_read)):
    try:
	test_image = images_read[ik]
        oriImg = cv.imread(test_image)  # B,G,R order
	oriImg = cv.resize(oriImg, (480, 320)) 
	orig = oriImg.copy()

        multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))  # 19
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))  # 38

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
#            print imageToTest_padded.shape 

            net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            # net.forward() # dry run
            net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                                       (3, 2, 0, 1)) / 256 - 0.5;
            start_time = time.time()
            output_blobs = net.forward()
#            print('At scale %d, The CNN took %.2f ms.' % (m, 1000 * (time.time() - start_time)))

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data),
                                   (1, 2, 0))  # output 1 is heatmaps
            heatmap = cv.resize(heatmap, (0, 0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

            paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
            paf = cv.resize(paf, (0, 0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)

        from scipy.ndimage.filters import gaussian_filter

        all_peaks = []
        peak_counter = 0

        for part in range(19 - 1):
            x_list = []
            y_list = []
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > param['thre1']))
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
			if norm == 0: break                        
			vec = np.divide(vec, norm)

                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
#                        print "found = 2"
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:  ####???
                        row = -1 * np.ones(20)  ####???
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        deleteIdx = [];
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

    except:
        traceback.print_exc()
    else:
	### Note:
	# Joint's index in candidate is as the following:
	#  nose: 0, neck: 1, Rsho: 2, Relb: 3, Rwri: 4, Lsho: 5, Lelb: 6, Lwri: 7, Rhip: 8, 
        #  Rkne: 9, Rank: 10, Lhip: 11, Lkne: 12, Lank: 13, Leye: 14, Reye: 15, Lear: 16, Rear: 17, pt19: 18
	
	#####################################
	######### Draw Bounding Box #########
	#####################################
	# Detect people in the image
	(rects, weights) = hog.detectMultiScale(oriImg, winStride=(4, 4), padding=(8, 8), scale=1.05)

	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.5)

	# Draw the bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv.rectangle(oriImg, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# CONTINUE #
	# Example to obtain Neck's (x, y) coordinate
	# neckX, neckY = candidate[subset[z][1].astype(int), 1], candidate[subset[z][1].astype(int), 0]
	for z in range(len(subset)):
        	for q in range(18):
                	index = subset[z][q]
                    	Y = candidate[index.astype(int), 0]
                    	X = candidate[index.astype(int), 1]

	# Output image into a folder "results"
        save_img_name = test_image.split('/')[-1]
        cv.imwrite('result/' + save_img_name, oriImg)

     
	# Command line to make a video



