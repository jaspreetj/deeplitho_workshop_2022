from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import phidl.geometry as pg
from copy import deepcopy
from shapely.ops import unary_union
import rasterio.features
import matplotlib.pyplot as plt
import os, cv2
from shapely.geometry.polygon import Polygon
import gdspy

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array(
            [
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (111, 74, 0),
                (81, 0, 81),
                (128, 64, 128),
                (244, 35, 232),
                (250, 170, 160),
                (230, 150, 140),
                (70, 70, 70),
                (102, 102, 156),
                (190, 153, 153),
                (180, 165, 180),
                (150, 100, 100),
                (150, 120, 90),
                (153, 153, 153),
                (153, 153, 153),
                (250, 170, 30),
                (220, 220, 0),
                (107, 142, 35),
                (152, 251, 152),
                (70, 130, 180),
                (220, 20, 60),
                (255, 0, 0),
                (0, 0, 142),
                (0, 0, 70),
                (0, 60, 100),
                (0, 0, 90),
                (0, 0, 110),
                (0, 80, 100),
                (0, 0, 230),
                (119, 11, 32),
                (0, 0, 142),
            ],
            dtype=np.uint8,
        )
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def gds_to_img(filename, dirpath, poly_layers=[0, 1], true_design_size=6.144):
    try:
        D = pg.import_gds(filename=dirpath / filename, cellname=None, flatten=True)
    except ValueError as e:
        e = str(e)
        cell_name = e[e.find("named ") + len("named ") : e.find(" already")]
        pg.gdspy.current_library.remove(cell_name)
        D = pg.import_gds(filename=dirpath / filename, cellname=None, flatten=True)

    zx = deepcopy(D)

    # check design size
    permitted_size = true_design_size
    design_size = [round(each, 3) for each in D.size.tolist()]
    if design_size[0] > permitted_size or design_size[1] > permitted_size:
        raise ValueError(
            "Incorrect GDS size. Design size is %s" % (design_size),
            f"must be less than or equal to {permitted_size}um x {permitted_size}um",
        )

    del D

    # remove unncessary layers but the one on Layer 0 and 1
    zx.remove_layers(
        layers=[
            int(each) for each in list(set(poly_layers).symmetric_difference(zx.layers))
        ]
    )

    dss = true_design_size  # design space size
    tr = 1024  # target res
    sm = tr / dss
    all_poly_objs = []
    from shapely.ops import unary_union

    for each_polygon in zx.get_polygons():
        # center the design surrounded by empty space if smaller than design space size
        poly_pts = [
            tuple(
                (
                    np.array([round(each, 3) for each in each_vertex])
                    - np.array([zx.xmin, zx.ymin])
                    + np.array([(dss - zx.xsize) / 2, (dss - zx.ysize) / 2])
                )
                * sm
            )
            for each_vertex in each_polygon.tolist()
        ]
        poly_inst = Polygon(poly_pts)
        all_poly_objs.append(poly_inst)

    poly_inst = unary_union(all_poly_objs)

    img = rasterio.features.rasterize([poly_inst], out_shape=(tr, tr))
    img_pil = Image.fromarray(np.flipud(img) * 255)

    plot = True
    if plot:
        plt.imshow(img_pil)
        plt.show()

    save = True
    if save:
        if not os.path.exists(dirpath / "test_A"):
            os.mkdir(dirpath / "test_A")
        img_pil.save(dirpath / "test_A" / str(filename.split(".")[0] + ".jpg"))

    return tr, tr, dirpath, ""


def img_to_poly(img_path, fs=45, plot=True, proc_flag=0, delta_px=0):

    src = cv2.imread(img_path, 0)
    src = cv2.flip(src, 0)
    # upscale img to 6144x6144
    dim = [each * 6 for each in src.shape]
    src = cv2.resize(src, dim, cv2.INTER_AREA)
    # remove high freq noise
    blur = cv2.medianBlur(src, fs)
    thresh = cv2.threshold(
        blur, blur.min(), blur.max(), cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # kernel for erosion and dilation operations
    kernel = np.ones((3, 3), np.uint8) * thresh.max()

    if proc_flag == 1:
        thresh = cv2.erode(thresh, kernel, iterations=delta_px)
        # erode
    elif proc_flag == 2:
        # dilate
        thresh = cv2.dilate(thresh, kernel, iterations=delta_px)

    if plot:
        plt.figure(
            figsize=(7, 7), dpi=300,
        )
        plt.subplot(1, 2, 1)
        plt.imshow(src, "gray", interpolation="none")
        plt.subplot(1, 2, 2)
        plt.imshow(src, "gray", interpolation="none")
        plt.imshow(thresh, "jet", interpolation="none", alpha=0.4)
        plt.show()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    poly_pts = []

    for c in contours:
        poly_pts.append(np.squeeze(c).tolist())

    return poly_pts


def write_gds(
    polygon_v, filepath="litho_example.gds", cellname="litho", output_layers=[0]
):
    # clear gds cell library for conflicts
    [
        gdspy.current_library.remove(each_cell)
        for each_cell in list(gdspy.current_library.cells)
    ]
    lib = gdspy.GdsLibrary()
    cell = lib.new_cell(cellname)

    for idx, val in enumerate(output_layers):
        ld_litho = {"layer": val, "datatype": 0}

        # declare empty lithography polygon instance
        litho_polyinst = gdspy.Polygon([], **ld_litho)

        for i in range(len(polygon_v[idx])):
            # scale points back to microns
            temp_pts = np.array(polygon_v[idx][i]) * 1e-3
            temp_pts = temp_pts.tolist()
            if len(polygon_v[idx][i]) > 2:
                litho_polyinst = gdspy.boolean(
                    litho_polyinst,
                    gdspy.Polygon(temp_pts, layer=val, datatype=0),
                    operation="xor",
                    layer=val,
                    datatype=0,
                )

        cell.add(litho_polyinst)

    lib.write_gds(filepath)

