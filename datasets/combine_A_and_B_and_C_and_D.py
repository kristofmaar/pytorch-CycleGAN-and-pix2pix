import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def image_write(path_A, path_B, path_C, path_D, path_ABCD):
    im_A = cv2.imread(path_A, 1)
    im_B = cv2.imread(path_B, 1)
    im_C = cv2.imread(path_C, 1)
    im_D = cv2.imread(path_D, 1)

    # Concatenate images horizontally
    im_ABCD = np.concatenate([im_A, im_B, im_C, im_D], 1)

    cv2.imwrite(path_ABCD, im_ABCD)

parser = argparse.ArgumentParser('create image quads')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image C', type=str, default='../dataset/fold_C')
parser.add_argument('--fold_D', dest='fold_D', help='input directory for image D', type=str, default='../dataset/fold_D')
parser.add_argument('--fold_ABCD', dest='fold_ABCD', help='output directory', type=str, default='../dataset/test_ABCD')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_ABCD', dest='use_ABCD', help='if true: (0001_A, 0001_B, 0001_C, 0001_D) to (0001_ABCD)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A)

if not args.no_multiprocessing:
    pool = Pool()

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_fold_C = os.path.join(args.fold_C, sp)
    img_fold_D = os.path.join(args.fold_D, sp)
    img_list = os.listdir(img_fold_A)
    if args.use_ABCD:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_ABCD = os.path.join(args.fold_ABCD, sp)
    if not os.path.isdir(img_fold_ABCD):
        os.makedirs(img_fold_ABCD)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        name_B = name_A.replace('_A.', '_B.') if args.use_ABCD else name_A
        name_C = name_A.replace('_A.', '_C.') if args.use_ABCD else name_A
        name_D = name_A.replace('_A.', '_D.') if args.use_ABCD else name_A
        path_B = os.path.join(img_fold_B, name_B)
        path_C = os.path.join(img_fold_C, name_C)
        path_D = os.path.join(img_fold_D, name_D)
        if os.path.isfile(path_A) and os.path.isfile(path_B) and os.path.isfile(path_C) and os.path.isfile(path_D):
            name_ABCD = name_A.replace('_A.', '_ABCD.') if args.use_ABCD else name_A.replace('.', '_ABCD.')
            path_ABCD = os.path.join(img_fold_ABCD, name_ABCD)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_C, path_D, path_ABCD))
            else:
                image_write(path_A, path_B, path_C, path_D, path_ABCD)

if not args.no_multiprocessing:
    pool.close()
    pool.join()
