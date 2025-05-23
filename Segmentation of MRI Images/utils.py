# This program is intended to help urologists in planning fusion biopsies using MRI images of the prostate. Segmentation
# of the prostate, accuracy evaluation of the segmentation mask, and extracting tissue information is performed.
# Date created: July 16, 2023,          Last modified: July 21, 2023, 20:00

# ----------------------------------------------------------------------------------------------------------------------
import SimpleITK as sitk
import matplotlib.pyplot as plt


def initial_visualization(mri, segment):
    '''
    This function visualizes the data that was read in.
    :param mri: The MRI volume.
    :param segment: The standard segment that was provided.
    :return: Visualization by plotting.
    '''
    # Applying Label Overlay (provided segment volume) to the MRI volume.
    res_mask = sitk.Resample(segment, mri, interpolator=sitk.sitkNearestNeighbor)
    mri_and_segment = sitk.LabelOverlay(mri, labelImage=res_mask)

    # Visualizing slice 13 of the original MRI volume and volume with provided mask.
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayFromImage(mri[:, :, 13]), cmap='gray')
    plt.title('Prostate MRI: Slice 13')
    plt.colorbar()
    plt.axis()
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayFromImage(mri_and_segment[:, :, 13]), cmap='gray')
    plt.title('Original MRI with Given Segment Mask')
    plt.axis()
    plt.colorbar()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# IMAGE PRE-PROCESSING

def enhancement_trials(mri):
     '''
    This function executes variations of the Median and Mean filters.
    :param mri: The MRI volume.
    :return: Visualizes the filters under varying parameters.
    '''
    # Option 1: Median filter
    median_filter = sitk.MedianImageFilter()

    # Option 2: Mean filter
    mean = sitk.MeanImageFilter()

    j = 0
    for j in range(4):
        if j % 2 == 0:
            median_filter.SetRadius(j)
            med = median_filter.Execute(mri)

            mean.SetRadius(j)
            mimg = mean.Execute(mri)
        j + 1

        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(sitk.GetArrayFromImage(med[:, :, 13]), cmap='gray')
        plt.title(f'Median Filter Radius {j}')
        plt.axis()
        plt.subplot(1, 2, 2)
        plt.imshow(sitk.GetArrayFromImage(mimg[:, :, 13]), cmap='gray')
        plt.title(f'Mean Filter Radius {j}')
        plt.axis()
        # plt.subplot(1, 3, 3)
        # plt.imshow(sitk.GetArrayFromImage(img_filt[:, :, 13]), cmap='gray')
        # plt.title(f'Intensity Window Output: {i}')
        # plt.axis()
        plt.show()


def intensity_trial(mri):
     '''
    This function iterates through varying values to ouput different results of the paramters in the
    IntensityWindowImageFilter.
    :param mri: The MRI volume.
    :return: Visualizes the filter under varying parameters.
    '''
    # Option 3: Intensity Window Filter
    filt = sitk.IntensityWindowingImageFilter()

    i = 0
    for i in range(100):
        if i % 25 == 0:
            filt.SetOutputMaximum(i)
            filt.SetOutputMinimum(80)
            filt.SetWindowMaximum(90)
            filt.SetWindowMinimum(10)

            img_filt = filt.Execute(mri)
            plt.imshow(sitk.GetArrayFromImage(img_filt[:, :, 13]), cmap='gray')
            plt.title(f'Intensity Window Output: {i}')
            plt.axis()
            plt.show()
            print(f'Output Maximum: {i}')
        i + 25


def preprocess_img(mri):
    '''
    This function rescales the MRI volume to be within the range of 0-255. This is a preprocessing step to aid in the
    segmentation process downstream.
    :param mri: The MRI volume.
    :return: The rescaled volume and a version with the Median filter applied.
    '''
    # Rescaling the intensity
    rescale = sitk.RescaleIntensity(mri, 0, 255)

    # Creation of the Median Filter
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius(4)
    med = median_filter.Execute(rescale)

    # Visualizing applied filters on rescaled MRI volume.
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayFromImage(mri[:, :, 13]), cmap='gray')
    plt.title('Original Volume')
    plt.axis()
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayFromImage(med[:, :, 13]), cmap='gray')
    plt.title('Rescaled Image Median Filter, Radius =4')
    plt.axis()
    plt.show()

    return rescale, med


# ----------------------------------------------------------------------------------------------------------------------
# PART A: Segmentation of the prostate in T2W MR volume
def test_segmenters(mri, segment, rescale, res_mask):
    rescale = sitk.RescaleIntensity(mri, 0, 255)
    # Making a binary image from original MRI volume.
    btif = sitk.BinaryThresholdImageFilter()
    btif.SetLowerThreshold(80)
    btif.SetUpperThreshold(98)
    btif.SetInsideValue(1)
    btif.SetOutsideValue(0)
    btif_mri = btif.Execute(mri)

    # Overlay
    mri_bt_lo = sitk.LabelOverlay(mri, labelImage=btif_mri)

    # Making a binary image from rescaled MRI volume.
    btif1 = sitk.BinaryThresholdImageFilter()
    btif1.SetLowerThreshold(30)
    btif1.SetUpperThreshold(45)
    btif1.SetInsideValue(1)
    btif1.SetOutsideValue(0)
    rescaled_mri = btif1.Execute(rescale)

    # Overlay
    med_laymri = sitk.LabelOverlay(rescale, labelImage=rescaled_mri)

    # Visualization of the Binary filters on both original and rescaled volumes.
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 3, 1)
    # Visualization of segmentation volume with Binary filter.
    plt.imshow(sitk.GetArrayFromImage(segment[:, :, 13]), cmap='gray')
    plt.title('Segment Mask')
    plt.axis()
    plt.subplot(1, 3, 2)
    # Visualization of segmentation volume with Binary filter.
    plt.imshow(sitk.GetArrayFromImage(btif_mri[:, :, 13]), cmap='gray')
    plt.title('Binary Mask')
    plt.axis()
    plt.subplot(1, 3, 3)
    # Visualization of segmentation volume with Median Filter.
    plt.imshow(sitk.GetArrayFromImage(rescaled_mri[:, :, 13]), cmap='gray')
    plt.title('Median Mask')
    plt.axis()
    # plt.colorbar()
    plt.show()

    # Resampled (provided) segment mask overlay on MRI volume.
    mri_and_segment = sitk.LabelOverlay(mri, labelImage=res_mask)

    # Visually comparing the provided segment against the overlay of the masks onto the volume.
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(sitk.GetArrayFromImage(mri_and_segment[:, :, 13]), cmap='gray')
    plt.title('Provided Segment Overlay')
    plt.axis()
    plt.subplot(1, 3, 2)
    plt.imshow(sitk.GetArrayFromImage(mri_bt_lo[:, :, 13]), cmap='gray')
    plt.title('Binary w/ Overlay: Slice 13')
    plt.axis()
    plt.subplot(1, 3, 3)
    plt.imshow(sitk.GetArrayFromImage(med_laymri[:, :, 13]), cmap='gray')
    plt.title('Median Filter')
    plt.axis()
    plt.show()


def confidence_test(mri):
    '''
    This function iterated through a loop, testing different results given a change in the "SetMultiplier" parameter for
    the ConfidenceConnectedImageFilter.
    :param mri: The MRI volume.
    :return: Visualizations of all combinations of the tested parameter.
    '''
    i = 0
    for i in range(3):
        c_test = sitk.ConfidenceConnectedImageFilter()
        c_test.SetNumberOfIterations(6)
        c_test.SetMultiplier(i)  # 1.50, 1.52 same,1.55 too much, 1.45 good!, 1.35,1.30 is the winner!!!
        c_test.SetSeedList([(533, 543, 13)])
        c_test = c_test.Execute(mri)

        plt.imshow(sitk.GetArrayFromImage(c_test[:, :, 13]), cmap='gray')
        plt.title('CC Mask Alone')
        plt.axis()
        plt.show()
        print(f'Multiplier= {i}')
    i + 0.5


def prostate_segmenter(mri, mri_and_segment):
    '''
    The function produces the results of the segmentation using the Confidence Connected filter. It cleans the segment,
    and then compares the newly created segmentation mask to the standard one.
    :param mri: The MRI volume.
    :param mri_and_segment: The original MRI volume overlaid with the standard segment.
    :return: The segment created from the Confidence Connected filter, as well as a cleaned version  of it.
    '''
    # Using the Confidence Connected filter on enhanced image.
    cc_filter = sitk.ConfidenceConnectedImageFilter()
    cc_filter.SetNumberOfIterations(6)
    cc_filter.SetMultiplier(1.30)
    cc_filter.SetSeedList([(533, 543, 13)])

    cc_filter = cc_filter.Execute(mri)

    # Applying Label Overlay.
    confidence_image = sitk.LabelOverlay(mri, labelImage=cc_filter)

    # Visualizing the Confidence Connected filter alone and on the MRI slice.
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayFromImage(cc_filter[:, :, 13]), cmap='gray')
    plt.title('CC Mask Alone')
    plt.axis()
    plt.subplot(1, 2, 2)
    # Visualization of segmented volume with Median Filter.
    plt.imshow(sitk.GetArrayFromImage(confidence_image[:, :, 13]), cmap='gray')
    plt.title('CC Mask on MRI')
    plt.axis()
    plt.show()

    # Cleaning the mask.
    vectorRadius = (10, 10, 10)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(cc_filter, vectorRadius, kernel)

    # Applying cleaned Label Overlay.
    confidence_image1 = sitk.LabelOverlay(mri, labelImage=seg_clean)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayFromImage(seg_clean[:, :, 13]), cmap='gray')
    plt.title('Cleaned CC Mask Alone')
    plt.axis()
    plt.subplot(1, 2, 2)
    # Visualization of segmented volume with Median Filter.
    plt.imshow(sitk.GetArrayFromImage(confidence_image1[:, :, 13]), cmap='gray')
    plt.title('Cleaned CC Mask on MRI')
    plt.axis()
    # plt.colorbar()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayFromImage(mri_and_segment[:, :, 13]), cmap='gray')
    plt.title('Overlaid MRI: Slice 13')
    plt.colorbar()
    plt.axis()
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayFromImage(confidence_image1[:, :, 13]), cmap='gray')
    plt.title('CC Segment Mask Overlay')
    plt.axis()
    plt.colorbar()
    plt.show()

    return cc_filter, seg_clean


def export_segment(seg_clean):
     '''
    This function exports the cleaned segmentation mask in 2 ways: by writing a new image, and by exporting it to 3D
    Slicer.
    :param seg_clean: The cleaned, and final, segmentation mask.
    '''
    # Exports selected filtered image as '.nrrd' file
    sitk.WriteImage(seg_clean, 'my_segmentation.nrrd')

    # Export image with overlay to 3D Slicer.
    external_viewer = sitk.ImageViewer()
    slicer_app_location = r"C:\Users\augustina\AppData\Local\NA-MIC\Slicer 5.2.2\Slicer.exe"
    external_viewer.SetApplication(slicer_app_location)

    external_viewer.Execute(seg_clean)


# ----------------------------------------------------------------------------------------------------------------------
# PART B: Quantitative evaluation of segmentation
def seg_eval_dice(res_mask, cc_filter, btif_mri, seg_clean):
    '''
    This function evaluates the accuracy of the newly created filter against the standard mask that was provided.
    :param res_mask: The resampled standard segment mask.
    :param cc_filter: The ConfidenceConnected mask.
    :param btif_mri: The mask created from the BinaryThreshold filter.
    :param seg_clean: The cleaned and final segmentation mask.
    :return: The DSC values of each mask.
    '''
    # Calculate the label overlap measures and get the Dice Similarity Coefficient.
    calc0 = sitk.LabelOverlapMeasuresImageFilter()
    calc0.Execute(cc_filter, res_mask)
    dsc0 = calc0.GetDiceCoefficient()
    print(f'The Dice Coefficient for cc_filter mask is {dsc0:.3f}.')

    calc1 = sitk.LabelOverlapMeasuresImageFilter()
    calc1.Execute(btif_mri, res_mask)
    dsc1 = calc1.GetDiceCoefficient()
    print(f'The Dice Coefficient for the Binary Mask is {dsc1:.3f}.')

    calc2 = sitk.LabelOverlapMeasuresImageFilter()
    calc2.Execute(seg_clean, res_mask)
    dsc2 = calc2.GetDiceCoefficient()
    print(f'The Dice Coefficient for seg_clean mask is {dsc2:.3f}.')

    return dsc0, dsc1, dsc2


# ----------------------------------------------------------------------------------------------------------------------
# PART C: Extract tissue properties for post biopsy analysis

def pixel_extract(mri, point, width, space, ori):
    '''
    This function extracts information from the target point(-20, 20, 30) of the prostate and aims to get the intensity
    values in a 6x6x6 cubic region surrounding the target point.
    :param mri: The MRI volume.
    :param point: The target point (-20, 20, 30).
    :param width: The width of the region is 6.
    :param space: The spacing between the pixels.
    :param ori: The origin of the MRI volume.
    :return: The intensities within the cubic region.
    '''
    # Convert LPS coordinate to index coordinate
    index = mri.TransformPhysicalPointToIndex(point)

    # Get the intensity value of a pixel at index (x,y,z)
    intensity = mri.GetPixel(index)
    print(f'Intensity at point is {intensity}.')

    # Get indices of points relative to the origin.
    x_place = point[0] - ori[0]
    y_place = point[1] - ori[1]
    z_place = point[2] - ori[2]

    x1_div = int((x_place - (width / 2)) / space[0])
    x2_div = int((x_place + (width / 2)) / space[0])
    y1_div = int((y_place - (width / 2)) / space[1])
    y2_div = int((y_place + (width / 2)) / space[1])
    z1_div = int((z_place - (width / 2)) / space[2])
    z2_div = int((z_place + (width / 2)) / space[2])

    # Convert to array.
    array = sitk.GetArrayFromImage(mri)
    cube = array[z1_div:z2_div, x1_div:x2_div, y1_div:y2_div]

    return cube


def intensity_boxplot(cube, point):
    '''
    This function produces a boxplot using the values that were curated in the previous function, the cubic values.
    :param cube: Holds the intensities of the pixels in the target region.
    :param point: The point is the target (-20, 20, 30).
    :return: A boxplot.
    '''
    # Flatten the array.
    flat_cube = cube.flatten()

    # Visualize as boxplot.
    plt.boxplot(flat_cube)
    plt.xlabel(f'{point}')
    plt.title('Intensity Distribution in Target Region')
    plt.ylabel('Intensity')
    plt.show()

# END OF SCRIPT
# ----------------------------------------------------------------------------------------------------------------------
