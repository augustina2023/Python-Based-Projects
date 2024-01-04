import utils
import SimpleITK as sitk

def main():
    '''
    This is the main function that calls all the other functions.
    '''
    # READING IMAGE FILES
    # Reading the '10522_1000532_t2wMRI.mha' file.
    reader = sitk.ImageFileReader()
    reader.SetImageIO('MetaImageIO')
    reader.SetFileName(
        r'C:\Users\augustina\Documents\Program\804\Mini-Project\10522_1000532_t2wMRI'
        r'.mha')
    mri = reader.Execute()
    # Reading the '10522_1000532_segmentation.nii.gz' file.
    reader = sitk.ImageFileReader()
    reader.SetImageIO('NiftiImageIO')
    reader.SetFileName(r'C:\Users\augustina\Documents\Program\804\Mini-Project\10522_1000532_'
                       r'segmentation.nii.gz')
    segment = reader.Execute()

    # Meta-data for MRI volume
    print('MRI Size:', mri.GetSize())
    print('MRI Origin:', mri.GetOrigin())
    print('MRI Spacing:', mri.GetSpacing())
    print('MRI Direction:', mri.GetDirection())
    print('MRI Pixel Type:', mri.GetPixelIDTypeAsString())

    # Meta-data for segment volume
    print('Segment Size:', segment.GetSize())
    print('Segment Size:', segment.GetOrigin())
    print('Segment Spacing:', segment.GetSpacing())
    print('Segment Direction:', segment.GetDirection())
    print('Segment Pixel Type:', segment.GetPixelIDTypeAsString())

    # Initial visualization and pre-processing.
    utils.initial_visualization(mri, segment)
    utils.enhancement_trials(mri)
    utils.intensity_trial(mri)
    utils.preprocess_img(mri)

    # Testing different filters and making enhancements.
    rescale = sitk.RescaleIntensity(mri, 0, 255)
    res_mask = sitk.Resample(segment, mri, interpolator=sitk.sitkNearestNeighbor)
    utils.test_segmenters(mri, segment, rescale, res_mask)
    utils.confidence_test(mri)

    # Selecting the best filter/mask and cleaning it.
    mri_and_segment = sitk.LabelOverlay(mri, labelImage=res_mask)
    utils.prostate_segmenter(mri, mri_and_segment)
    cc_filter = sitk.ConfidenceConnectedImageFilter()
    cc_filter = cc_filter.Execute(mri)
    vectorRadius = (10, 10, 10)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(cc_filter, vectorRadius, kernel)
    utils.export_segment(seg_clean)

    # Evaluation of the masks using the DCS
    btif = sitk.BinaryThresholdImageFilter()
    btif_mri = btif.Execute(mri)
    utils.seg_eval_dice(res_mask, cc_filter, btif_mri, seg_clean)

    # Extracting information from the prostate and plotting using boxplot.
    space = mri.GetSpacing()
    ori = mri.GetOrigin()
    width = 6
    point = (-20, 20, 30)
    utils.pixel_extract(mri, point, width, space, ori)

    # Plotting boxplot.
    x1_div = int(((point[0] - ori[0]) - (width / 2)) / space[0])
    x2_div = int(((point[0] - ori[0]) + (width / 2)) / space[0])
    y1_div = int(((point[1] - ori[1]) - (width / 2)) / space[1])
    y2_div = int(((point[1] - ori[1]) + (width / 2)) / space[1])
    z1_div = int(((point[2] - ori[2]) - (width / 2)) / space[2])
    z2_div = int(((point[2] - ori[2]) + (width / 2)) / space[2])
    array = sitk.GetArrayFromImage(mri)
    cube = array[z1_div:z2_div, x1_div:x2_div, y1_div:y2_div]
    utils.intensity_boxplot(cube, point)

main()
