# from GPUtils.startup_guyga import *
import sys, os, copy, math
from os.path import join as pjoin, exists as pexists, basename, isfile
import numpy as np
from numpy import array, unique
import torch
import torch.utils.data as data
import cv2
import json
from termcolor import cprint
cprintm = lambda s: cprint(s, 'magenta')
identity = lambda x: x
from tqdm import tqdm
import itertools

def chained(l):
    return list(itertools.chain(*l))

class_id_map = {}
with open('/data/guyga/datasets/xview/xview_class_labels.txt', 'r') as f:
    for x in f.readlines():
        splitted = x.split(':')
        class_id_map[int(splitted[0])] = splitted[1].strip('\n')
    class_id_map[75] = 'Undefined1'
    class_id_map[82] = 'Undefined2'

'''
coarse_class_map = [
    [11, 12, 13, 74, 84],                               # group 0 aircraft
    [17, 18, 19],                                           # group 1 cars
    [71, 72, 73, 76, 77, 79, 83, 86, 89, 93, 94],           # group 2 buildings
    [20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 60, 91],       # group 3 trucks
    [33, 34, 35, 36, 37, 38],                               # group 4 trains
    [40, 41, 42, 44, 45, 47, 49, 50, 51, 52],               # group 5 boats
    [53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 65, 66]        # group 6 docks
    ]      
coarse_class_names = ['aircraft', 'cars', 'buildings', 'trucks', 'trains', 'boats', 'docks']
'''
coarse_class_map = [
    [11, 12, 13],  # Airplane
    # [15],  # Helicopter
    [17, 18],  # Car/small trucks
    [20, 21],  # Small trucks (pickup truck etc.)
    [23, 24, 25, 26, 27, 28, 29, 32, 60, 61],  # Big trucks
    [53, 62, 63, 64, 65, 66, 56, 57],  # Engineering Vehicle
    [19],  # Bus
    [73, 74],  # Buildings
    # [94],  # Tower
    # [71],  # Hut/Tent
    [72],  # Shed (elongated)
    [76, 79],  # Construction Site / Damaged Building
    # [77],  # Facility (soccer court, water park etc.)
    # [84],  # Helipad
    [93],  # Pylon
    [83],  # Vehicle Lot
    [89, 91],  # Shipping containers
    # [86],  # Storage Tank (round)
    [86, 94],  # Storage Tank/Tower (round)
    [33, 34, 35, 36, 37, 38],  # Train element
    # [41, 42, 44, ]  # Small boats
    # [40, 45, 47, 51, 52],  # Big ships
    [40, 41, 42, 44, 45, 47, 49, 50, 51, 52],  # Boat/Ship
    [54, 55, 59],  # Cranes
    ]     
coarse_class_names = ['Airplane', 'Car/small trucks', 'Small trucks', 'Big trucks', 'Engineering Vehicle', \
    'Bus', 'Buildings', 'Shed', 'Construction Site/Damaged Building', 'Pylon', 'Vehicle Lot', \
        'Containers', 'Storage Tank/Tower', 'Train element', 'Boat/Ship', 'Cranes']
 
# '''

def annot_valid(image_name, annot):
    ''' Screening rules for BB validity '''
    return True
# def [coords, valid] = clean_coords(chip_number, coords, classes, image_h, image_w)
#     % j = chip_number == 2294;
#     % coords = coords(j,:);
#     % classes = classes(j);
#     % image_h = image_h(j,:);
#     % image_w = image_w(j,:);

#     x1 = coords(:,1);
#     y1 = coords(:,2);
#     x2 = coords(:,3);
#     y2 = coords(:,4);
#     w = x2-x1;
#     h = y2-y1;
#     area = w.*h;

#     % crop
#     x1 = min( max(x1,0), image_w);
#     y1 = min( max(y1,0), image_h);
#     x2 = min( max(x2,0), image_w);
#     y2 = min( max(y2,0), image_h);
#     w = x2-x1;
#     h = y2-y1;
#     new_area = w.*h;
#     new_ar = max(w./h, h./w);
#     coords = [x1 y1 x2 y2];

#     % no nans or infs in bounding boxes
#     i0 = ~any(isnan(coords) | isinf(coords), 2);

#     % % sigma rejections on dimensions (entire dataset)
#     % [~, i1] = fcnsigmarejection(new_area,21, 3);
#     % [~, i2] = fcnsigmarejection(w,21, 3);
#     % [~, i3] = fcnsigmarejection(h,21, 3);
#     i1 = true(size(w));
#     i2 = true(size(w));
#     i3 = true(size(w));

#     % sigma rejections on dimensions (per class)
#     uc=unique(classes(:));
#     for i = 1:numel(uc)
#         j = find(classes==uc(i));
#         [~,v] = fcnsigmarejection(new_area(j),12,3);    i1(j) = i1(j) & v;
#         [~,v] = fcnsigmarejection(w(j),12,3);           i2(j) = i2(j) & v;
#         [~,v] = fcnsigmarejection(h(j),12,3);           i3(j) = i3(j) & v;
#     end

#     % manual dimension requirements
#     i4 = new_area >= 20 & w > 4 & h > 4 & new_ar<15;  

#     % extreme edges (i.e. don't start an x1 10 pixels from the right side)
#     i5 = x1 < (image_w-10) & y1 < (image_h-10) & x2 > 10 & y2 > 10;  % border = 5

#     % cut objects that lost >90% of their area during crop
#     new_area_ratio = new_area./ area;
#     i6 = new_area_ratio > 0.25;

#     % no image dimension nans or infs, or smaller than 32 pix
#     hw = [image_h image_w];
#     i7 = ~any(isnan(hw) | isinf(hw) | hw < 32, 2);

#     % remove invalid classes 75 and 82 ('None' class, wtf?)
#     i8 = ~any(classes(:) == [75, 82],2);

#     % remove 18 and 73 (small cars and buildings) as an experiment
#     %i9 = any(classes(:) == [11, 12, 13, 15, 74, 84],2);  % group 0 aircraft
#     %i9 = any(classes(:) == [17, 18, 19],2);  % group 1 cars
#     %i9 = any(classes(:) == [71, 72, 76, 77, 79, 83, 86, 89, 93, 94],2);  % group 2 buildings
#     %i9 = any(classes(:) == [20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 60, 91],2);  % group 3 trucks
#     %i9 = any(classes(:) == [33, 34, 35, 36, 37, 38],2);  % group 4 trains
#     %i9 = any(classes(:) == [40, 41, 42, 44, 45, 47, 49, 50, 51, 52],2);  % group 5 boats
#     %i9 = any(classes(:) == [53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 65, 66],2);  % group 6 docks

#     valid = i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7 & i8;
#     coords = coords(valid,:)

class XView(data.Dataset):
    """xview Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, transform=identity, coarse_class=True):
        self.root = root
        self.transform = transform
        self.coarse_class = coarse_class
        '''
        # Work with preprocessed annotations (cleaned etc. by xviewyolov3)
        targets_c60 = sio.loadmat('/data/guyga/PycharmProjects/xviewyolov3/utils/targets_c60.mat')
        self.annotations = {'{}.tif'.format(x):[] for x in targets_c60['image_numbers'].flatten().astype('int')}
        for image_id, xview_annot in zip(targets_c60['id'].flatten().astype('int'), targets_c60['targets'].astype('int')):
            # self.annotations['{}.tif'.format(image_id)].append(list(xview_annot[1:]) + [xview_annot[0]])
            class_orig = list(class_id_map.keys())[xview_annot[0]]
            # print(xview_annot[0], class_orig)
            if class_orig in chained(coarse_class_map):
                found = [class_orig in inner_classes for inner_classes in coarse_class_map]
                assert sum(found) == 1, (found, class_orig)
                class_coarse = found.index(True)    
                # print(class_orig, class_coarse)
                self.annotations['{}.tif'.format(image_id)].append(list(xview_annot[1:]) + \
                    # [class_coarse])
                    [class_orig])
        ''' 
        # Work with original geojson
        with open('/data/guyga/datasets/xview/xView_train.geojson') as f:
            jsondata = json.load(f)
        # self.annotations = {x['properties']['image_id']:[] for x in jsondata['features']}
        self.annotations = {}
        cprintm('   >> %s' % self.root)
        for xview_annot in tqdm(jsondata['features']):
            image_name = xview_annot['properties']['image_id']
            if pexists(pjoin(self.root, image_name)):
                class_orig = int(xview_annot['properties']['type_id'])
                if class_orig in chained(coarse_class_map):
                    annot = [int(x) for x in xview_annot['properties']['bounds_imcoords'].split(',')]
                    if coarse_class:
                        found = [class_orig in inner_classes for inner_classes in coarse_class_map]
                        assert sum(found) == 1, (found, class_orig)
                        class_coarse = found.index(True)    
                        annot.append(class_coarse)
                    else:
                        annot.append(class_orig)
                    #  + \
                        # [class_coarse]
                        # [class_orig]
                        # [list(class_id_map.keys()).index(class_orig)]
                    if annot_valid(image_name, annot):
                        if image_name not in self.annotations:
                            self.annotations[image_name] = []
                        self.annotations[image_name].append(annot)
                    
        self.data = list(self.annotations.items())
        # '''
        # self.images_paths = natsorted(glob.glob(root + '/*'))

    def __getitem__(self, index):
        image_name, target = self.data[index]
        img = cv2.imread(pjoin(self.root, image_name))
        # image_name = basename(self.images_paths[index])
        # target = array(self.annotations[image_name])
        # img = cv2.imread(self.images_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        height, width, channels = img.shape

        target = array(target, dtype='float32')
        sample = {'img': img, 'annot': target}
        sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)
        # return len(self.images_paths)

    def num_classes(self):
        return len(unique(np.concatenate([unique(array(x)[:,-1]) for x in self.annotations.values()])))

    def label_to_name(self, label):
        if self.coarse_class:
            return coarse_class_names[label]
        else:
            return class_id_map[label]

    def load_annotations(self, index):
        _, target = self.data[index]
        target = array(target, dtype='float32')
        return target

if __name__ == "__main__":
    train_path = '/data/guyga/datasets/xview/train_images'
    dset = XView(train_path)
    print(dset.num_classes())