package nifi;

import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import static org.bytedeco.javacpp.opencv_core.LINE_AA;
import static org.bytedeco.javacpp.opencv_core.cvClearMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.commons.io.IOUtils;
import org.apache.nifi.annotation.behavior.InputRequirement;
import org.apache.nifi.annotation.behavior.InputRequirement.Requirement;
import org.apache.nifi.annotation.documentation.CapabilityDescription;
import org.apache.nifi.annotation.documentation.Tags;
import org.apache.nifi.components.PropertyDescriptor;
import org.apache.nifi.flowfile.FlowFile;
import org.apache.nifi.logging.ComponentLog;
import org.apache.nifi.processor.AbstractProcessor;
import org.apache.nifi.processor.ProcessContext;
import org.apache.nifi.processor.ProcessSession;
import org.apache.nifi.processor.ProcessorInitializationContext;
import org.apache.nifi.processor.Relationship;
import org.apache.nifi.processor.exception.ProcessException;
import org.apache.nifi.processor.io.InputStreamCallback;
import org.apache.nifi.processor.io.OutputStreamCallback;
import org.apache.nifi.processor.util.StandardValidators;
import org.apache.nifi.stream.io.ByteArrayInputStream;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.helper.opencv_core.AbstractCvMemStorage;
import org.bytedeco.javacpp.helper.opencv_core.AbstractCvScalar;
import org.bytedeco.javacpp.helper.opencv_core.AbstractIplImage;
import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacpp.presets.opencv_objdetect;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

/**
 * A NiFi processor, which takes as input video frames coming from a video camera,
 * detects human faces in each frames and crops these detected faces.
 */
@InputRequirement(Requirement.INPUT_REQUIRED)
@Tags({"ekstream", "video", "stream", "face", "detection", "crop"})
@CapabilityDescription("This processors takes as input video frames coming from a video camera, "
        + "detects human faces in each frames and crops these detected faces.")
public class FaceDetector extends AbstractProcessor {

    /** Scale factor for face detection. */
    static final double SCALE_FACTOR = 1.5;

    /** Neighbors for face detection. */
    static final int MIN_NEIGHBOURS = 3;

    /** Relationship "Success". */
    public static final Relationship REL_SUCCESS = new Relationship.Builder()
            .name("success")
            .description("Video frames have been properly captured.")
            .build();

    /** Relationship "Failure".*/
    public static final Relationship REL_FAILURE = new Relationship.Builder()
            .name("failure")
            .description("If no files have been detected, then original input files "
                    + "are transferred to this relationship")
            .build();

    /** Processor property. */
    public static final PropertyDescriptor SAVE_IMAGES = new PropertyDescriptor.Builder()
            .name("Save images")
            .description("Specifies whether interim results should be saved.")
            .allowableValues(new HashSet<String>(Arrays.asList("true", "false")))
            .defaultValue("true")
            .required(true)
            .addValidator(StandardValidators.BOOLEAN_VALIDATOR)
            .build();

    /** Processor property. */
    public static final PropertyDescriptor IMAGE_WIDTH = new PropertyDescriptor.Builder()
            .name("Image width")
            .description("Specifies the width of images with detected faces")
            .defaultValue("92")
            .required(true)
            .addValidator(StandardValidators.INTEGER_VALIDATOR)
            .build();

    /** Processor property. */
    public static final PropertyDescriptor IMAGE_HEIGHT = new PropertyDescriptor.Builder()
            .name("Image height")
            .description("Specifies the height of images with detected faces")
            .defaultValue("112")
            .required(true)
            .addValidator(StandardValidators.INTEGER_VALIDATOR)
            .build();

    /** Converter for Frames and IplImages. */
    private static OpenCVFrameConverter.ToIplImage converter;

    /** Converter for byte arrays and images. */
    private static Java2DFrameConverter flatConverter;

    /** List of processor properties. */
    private List<PropertyDescriptor> properties;

    /** List of processor relationships. */
    private Set<Relationship> relationships;

    /** Logger. */
    private ComponentLog logger;

    /**
     * {@inheritDoc}
     */
    @Override
    protected void init(final ProcessorInitializationContext context) {

        Loader.load(opencv_objdetect.class);

        logger = getLogger();

        final Set<Relationship> procRels = new HashSet<Relationship>();
        procRels.add(REL_SUCCESS);
        procRels.add(REL_FAILURE);
        relationships = Collections.unmodifiableSet(procRels);

        final List<PropertyDescriptor> supDescriptors = new ArrayList<>();
        supDescriptors.add(IMAGE_WIDTH);
        supDescriptors.add(IMAGE_HEIGHT);
        supDescriptors.add(SAVE_IMAGES);
        properties = Collections.unmodifiableList(supDescriptors);

        converter = new OpenCVFrameConverter.ToIplImage();
        flatConverter = new Java2DFrameConverter();

        logger.info("Initialision complete!");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Set<Relationship> getRelationships() {
        return relationships;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected List<PropertyDescriptor> getSupportedPropertyDescriptors() {
        return properties;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrigger(final ProcessContext aContext, final ProcessSession aSession)
            throws ProcessException {

        FlowFile flowFile = aSession.get();
        if (flowFile == null) {
            return;
        }

        aSession.read(flowFile, new InputStreamCallback() {

            @Override
            public void process(final InputStream aStream) throws IOException {

                ByteArrayInputStream inputStream = new ByteArrayInputStream(
                        IOUtils.toByteArray(aStream));

                BufferedImage bufferedImage = ImageIO.read(inputStream);
                IplImage image = toIplImage(bufferedImage);

                opencv_imgcodecs.cvSaveImage(System.currentTimeMillis() + "-received.png", image);

                ArrayList<IplImage> faces = detect(image);
                if (!faces.isEmpty()) {
                    ArrayList<IplImage> resizedFaces = resizeImages(faces,
                            Integer.parseInt(aContext.getProperty(IMAGE_WIDTH).getValue()),
                            Integer.parseInt(aContext.getProperty(IMAGE_HEIGHT).getValue()));

                    //now transfer the cropped images forward
                    for (IplImage face : resizedFaces) {

                        if (aContext.getProperty(SAVE_IMAGES).asBoolean()) {
                            opencv_imgcodecs.cvSaveImage(System.currentTimeMillis()
                                    + "-face.png", face);
                        }

                        FlowFile result = aSession.create(flowFile);
                        result = aSession.write(result, new OutputStreamCallback() {

                            @Override
                            public void process(final OutputStream aStream) throws IOException {
                                aStream.write(toByteArray(face));
                            }
                        });
                        aSession.transfer(result, REL_SUCCESS);
                        //aSession.commit();
                    }
                }
            }
        });

        //TODO how to destroy the original flowfile?
        aSession.transfer(flowFile, REL_FAILURE);
        //aSession.commit();

    }

    /**
     * Detects faces in an input image.
     *
     * @param aImage input image
     * @return an array of detected faces as images
     */
    public static ArrayList<IplImage> detect(final IplImage aImage) {

        ArrayList<IplImage> result = new ArrayList<IplImage>();

        CvHaarClassifierCascade cascade =
                new CvHaarClassifierCascade(cvLoad("haarcascade_frontalface_default.xml"));
        CvMemStorage storage = AbstractCvMemStorage.create();
        CvSeq sign = cvHaarDetectObjects(aImage,
                cascade, storage, SCALE_FACTOR, MIN_NEIGHBOURS, CV_HAAR_DO_CANNY_PRUNING);

        for (int i = 0; i < sign.total(); i++) {
            CvRect r = new CvRect(cvGetSeqElem(sign, i));
            //opencv_imgproc.cvRectangle(aImage, cvPoint(r.x(), r.y()),
            //        cvPoint(r.width() + r.x(), r.height() + r.y()),
            //        AbstractCvScalar.RED, 2, LINE_AA, 0);

            IplImage image = cropImage(aImage, r);
            result.add(image);
        }

        cvClearMemStorage(storage);
        return result;
    }

    /**
     * Crops an image to a given rectangle.
     *
     * @param aImage input image
     * @param aX x coordinate
     * @param aY y coordinate
     * @param aW width
     * @param aH height
     * @return cropped image
     */
    public static IplImage cropImage(final IplImage aImage,
            final int aX, final int aY, final int aW, final int aH) {

        // IplImage orig = cvLoadImage("orig.png");
        // Creating rectangle by which bounds image will be cropped
        CvRect r = new CvRect(aX, aY, aW, aH);
        // After setting ROI (Region-Of-Interest) all processing will only be
        // done on the ROI
        opencv_core.cvSetImageROI(aImage, r);
        IplImage cropped = opencv_core.cvCreateImage(opencv_core.cvGetSize(aImage),
                aImage.depth(), aImage.nChannels());
        // Copy original image (only ROI) to the cropped image
        opencv_core.cvCopy(aImage, cropped);

        //opencv_imgcodecs.cvSaveImage("data/" + System.currentTimeMillis() + "-crop.png", cropped);

        return cropped;

    }

    /**
     * Crops an image to a given rectangle.
     *
     * @param aImage original image
     * @param aRectangle rectangle
     * @return cropped image
     */
    public static IplImage cropImage(final IplImage aImage, final CvRect aRectangle) {

        // After setting ROI (Region-Of-Interest) all processing will only be
        // done on the ROI
        opencv_core.cvSetImageROI(aImage, aRectangle);
        IplImage cropped = opencv_core.cvCreateImage(opencv_core.cvGetSize(aImage),
                aImage.depth(), aImage.nChannels());
        // Copy original image (only ROI) to the cropped image
        opencv_core.cvCopy(aImage, cropped);

        //opencv_imgcodecs.cvSaveImage("data/" + System.currentTimeMillis() + "-crop.png", cropped);

        return cropped;

    }

    /**
     * Converts an IplImage into a byte array.
     *
     * @param aImage input image
     * @return byte array
     * @throws IOException exception
     */
    public static byte[] toByteArray(final IplImage aImage) throws IOException {

        BufferedImage result = flatConverter.convert(converter.convert(aImage));
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(result, "png", baos);
        baos.flush();
        byte[] byteImage = baos.toByteArray();
        baos.close();

        return byteImage;
    }

    /**
     * Converts a frame into a byte array.
     *
     * @param aFrame input frame
     * @return byte array
     * @throws IOException exception
     */
    public static byte[] toByteArray(final Frame aFrame) throws IOException {

        BufferedImage result = flatConverter.convert(aFrame);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(result, "png", baos);
        baos.flush();
        byte[] byteImage = baos.toByteArray();
        baos.close();

        return byteImage;
    }

    /**
     * Converts a buffered image into a JavaCV frame.
     *
     * @param aImage buffered image
     * @return frame
     * @throws IOException exception
     */
    public static Frame toFrame(final BufferedImage aImage) throws IOException {

        Frame result = flatConverter.convert(aImage);
        return result;
    }

    /**
     * Converts a buffered image into a JavaCV image.
     *
     * @param aImage buffered image
     * @return image
     * @throws IOException exception
     */
    public static IplImage toIplImage(final BufferedImage aImage) throws IOException {

        IplImage result = converter.convertToIplImage(flatConverter.convert(aImage));
        return result;
    }

    /**
     * Resizes and returns a single image.
     *
     * @param aImage original image
     * @param aWidth image width
     * @param aHeight image height
     * @return resized image
     */
    public static IplImage resizeImage(final IplImage aImage, final int aWidth, final int aHeight) {

        IplImage result = AbstractIplImage.create(aWidth, aHeight,
                aImage.depth(), aImage.nChannels());

        //======================
        opencv_imgcodecs.cvSaveImage(System.currentTimeMillis()
                + "-notresizedface.png", aImage);
        //======================

        opencv_imgproc.cvResize(aImage, result);

        return result;
    }

    /**
     * Resizes and returns resized images.
     *
     * @param aImages original images
     * @param aWidth image width
     * @param aHeight image height
     * @return an array of resized images
     */
    public static ArrayList<IplImage> resizeImages(final ArrayList<IplImage> aImages,
            final int aWidth, final int aHeight) {

        ArrayList<IplImage> result = new ArrayList<IplImage>();

        for (IplImage image : aImages) {
            result.add(resizeImage(image, aWidth, aHeight));
        }

        return result;
    }



}
