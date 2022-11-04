def gds_to_jpg(filename, file_dir, poly_layers = [0, 1]):
    file_dir = os.getcwd() + file_dir

    try:
        D = pg.import_gds(filename =file_dir+'/'+filename, cellname = None, flatten=True)
    except ValueError as e:
        e = str(e)
        cell_name = e[e.find('named ')+len('named '):e.find(' already')]
        pg.gdspy.current_library.remove(cell_name)
        D = pg.import_gds(filename = file_dir + '/' + filename, cellname = None, flatten=True)


    zx = deepcopy(D)

    #check design size
    permitted_size = 5
    design_size = D.size.tolist()
    if(design_size[0] > permitted_size or design_size[1] > permitted_size):
        print("Incorrect GDS size. Design size is %s"%(design_size), 'must be less than or equal to 5um x 5um')
        

    del(D)

    #remove unncessary layers but the one on Layer 0 and 1
    zx.remove_layers(layers=[int(each) for each in list(set(poly_layers).symmetric_difference(zx.layers))])

    #find image resolution
    xrange = []
    yrange = []

    for each_polygon in zx.get_polygons():
        poly_pts = [tuple(each_vertex) for each_vertex in each_polygon.tolist()]
        poly_inst = Polygon(poly_pts)

        if(len(xrange)<2):
            xrange.append(poly_inst.bounds[0])
        else:
            if(poly_inst.bounds[0]<xrange[0]):
                xrange[0] = poly_inst.bounds[0]

        if(len(xrange)<2):
            xrange.append(poly_inst.bounds[2])
        else:
            if(poly_inst.bounds[2]>xrange[1]):
                xrange[1] = poly_inst.bounds[2]

        if(len(yrange)<2):
            yrange.append(poly_inst.bounds[1])
        else:
            if(poly_inst.bounds[1]<yrange[0]):
                yrange[0] = poly_inst.bounds[1]

        if(len(yrange)<2):
            yrange.append(poly_inst.bounds[3])
        else:
            if(poly_inst.bounds[3]>yrange[1]):
                yrange[1] = poly_inst.bounds[3]


    true_size = 5 #microns
    img_block_size = 1024 #pixels

    print(design_size)

    #resize based on the training image micron to pixels
    res_w = np.ceil(design_size[0]*img_block_size/true_size)
    res_h = np.ceil(design_size[1]*img_block_size/true_size)
    print(res_w, res_h)



    #for each polygon, convert to image
    fig = plt.figure(figsize=(res_w, res_h), dpi=1)
    ax = fig.add_subplot(111)

    xrange = []
    yrange = []

    #convert to jpg
    for each_polygon in zx.get_polygons():
        poly_pts = [tuple(each_vertex) for each_vertex in each_polygon.tolist()]
        #convert to polygon
        poly_inst = Polygon(poly_pts)
        poly_patch = PolygonPatch(poly_inst,facecolor='k', alpha=1, edgecolor='none', rasterized=True)
        ax.add_patch(poly_patch)

        if(len(xrange)<2):
            xrange.append(poly_inst.bounds[0])
        else:
            if(poly_inst.bounds[0]<xrange[0]):
                xrange[0] = poly_inst.bounds[0]

        if(len(xrange)<2):
            xrange.append(poly_inst.bounds[2])
        else:
            if(poly_inst.bounds[2]>xrange[1]):
                xrange[1] = poly_inst.bounds[2]

        if(len(yrange)<2):
            yrange.append(poly_inst.bounds[1])
        else:
            if(poly_inst.bounds[1]<yrange[0]):
                yrange[0] = poly_inst.bounds[1]

        if(len(yrange)<2):
            yrange.append(poly_inst.bounds[3])
        else:
            if(poly_inst.bounds[3]>yrange[1]):
                yrange[1] = poly_inst.bounds[3]



    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_aspect(1)
    plt.axis('off')
    plt.tight_layout()

    save = True
    if(save):
      os.mkdir(file_dir+'/test_A/')
      plt.savefig(file_dir+'/test_A/'+filename.split('.')[0]+'.jpg')

    #remove variables
    del(zx, poly_pts, poly_inst, poly_patch)
    return 1024, 1024, file_dir, ''




#convert to GDS
def img_to_poly(img_path, fs=11, plot=True, proc_flag = 0, proc_itr = 1):
      src = cv2.imread(img_path, 0)
      src = cv2.flip(src, 0)
      blur = cv2.medianBlur(src,fs)
      thresh = cv2.threshold(blur,blur.min(),blur.max(),cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      kernel = np.ones((3,3),np.uint8) * thresh.max()

      if(proc_flag == 1):
        thresh = cv2.erode(thresh,kernel,iterations = proc_itr)
        #erode
      elif(proc_flag == 2):
        #dilate
        thresh = cv2.dilate(thresh,kernel,iterations = proc_itr)



      if(plot):
        plt.figure(figsize=(7,7), dpi=300,)
        plt.subplot(1,2,1)
        plt.imshow(src, 'gray', interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(src, 'gray', interpolation='none')
        plt.imshow(thresh, 'jet', interpolation='none', alpha=0.4)
        plt.show()

      contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      plot_pts = []
      poly_pts = []
      scale = 5.000e9/1024 #pm to pixels ratio
      for c in contours:
          poly_temp = cv2.approxPolyDP(c, 0.001, True)
          temp_shape = poly_temp.shape
          poly_temp = poly_temp.reshape(temp_shape[0], temp_shape[-1])
          plot_pts.append(poly_temp)
          poly_temp = poly_temp*scale #scale to nanometers
          poly_temp = np.round(poly_temp,0)
          #can use current polygon format with gdspy
          poly_pts.append(poly_temp.tolist())

          #further processing to format polygons to phidl format
          #poly_temp = np.hsplit(poly_temp,2)
          #poly_pts.append([poly_temp[0].flatten().tolist(), poly_temp[1].flatten().tolist()])
      
      if(plot):
          fig, ax = plt.subplots()
          patches = []

          for pts in plot_pts:
              polygon = matpoly.Polygon(np.array(pts).dot([[1,0],[0,-1]]), True)
              patches.append(polygon)

          p = PatchCollection(patches, cmap=cm.jet, alpha=0.4)

          colors = len(patches)*np.random.rand(len(patches))
          p.set_array(np.array(colors))

          ax.add_collection(p)
          ax.autoscale_view()

          plt.show()
      return poly_pts
