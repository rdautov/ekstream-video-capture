package nifi;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.nifi.annotation.behavior.InputRequirement;
import org.apache.nifi.annotation.behavior.TriggerWhenEmpty;
import org.apache.nifi.annotation.behavior.InputRequirement.Requirement;
import org.apache.nifi.annotation.documentation.CapabilityDescription;
import org.apache.nifi.annotation.documentation.Tags;
import org.apache.nifi.components.PropertyDescriptor;
import org.apache.nifi.flowfile.FlowFile;
import org.apache.nifi.processor.AbstractProcessor;
import org.apache.nifi.processor.ProcessContext;
import org.apache.nifi.processor.ProcessSession;
import org.apache.nifi.processor.ProcessorInitializationContext;
import org.apache.nifi.processor.Relationship;
import org.apache.nifi.processor.exception.ProcessException;
import org.apache.nifi.processor.io.OutputStreamCallback;
import org.apache.nifi.processor.util.StandardValidators;
import org.apache.nifi.stream.io.ByteArrayOutputStream;
import org.apache.nifi.logging.ComponentLog;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.presets.opencv_objdetect;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;

/**
 * A NiFi processor which accesses the default video camera, captures the video stream,
 * samples it into separate frames, and transfers forward for face recognition.
 */
@TriggerWhenEmpty
@InputRequirement(Requirement.INPUT_FORBIDDEN)
@Tags({"ekstream", "video", "stream", "capturing", "sampling"})
@CapabilityDescription("Testing JavaCV api")
public class VideoCapturer extends AbstractProcessor {

    /** Relationship "Success". */
    public static final Relationship REL_SUCCESS = new Relationship.Builder().name("success")
            .description("Video frames have been properly captured.").build();

    /** Processor property. */
    public static final PropertyDescriptor FRAME_INTERVAL = new PropertyDescriptor.Builder()
            .name("Time interval between frames")
            .description("Specified the time interval between two captured video frames, in ms")
            .defaultValue("1000")
            .required(true)
            .addValidator(StandardValidators.INTEGER_VALIDATOR)
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

    /** List of processor properties. */
    private List<PropertyDescriptor> properties;

    /** List of processor relationships. */
    private Set<Relationship> relationships;

    /** JavaCV frame grabber. */
    private static FrameGrabber grabber;

    /** Converter for Frames and IplImages. */
    private static OpenCVFrameConverter.ToIplImage converter;

    /** Converter for byte arrays and images. */
    private static Java2DFrameConverter flatConverter;

    /** Logger. */
    private static ComponentLog logger;

    /**
     * {@inheritDoc}
     */
    @Override
    protected void init(final ProcessorInitializationContext context) {

        Loader.load(opencv_objdetect.class);

        logger = getLogger();

        final Set<Relationship> procRels = new HashSet<>();
        procRels.add(REL_SUCCESS);
        relationships = Collections.unmodifiableSet(procRels);

        final List<PropertyDescriptor> supDescriptors = new ArrayList<>();
        supDescriptors.add(FRAME_INTERVAL);
        supDescriptors.add(SAVE_IMAGES);
        properties = Collections.unmodifiableList(supDescriptors);

        try {
            grabber = FrameGrabber.createDefault(0);
        } catch (Exception e) {
            logger.error("Something went wrong with the video capture!", e);
        }
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

        try {

            grabber.start();

            Frame frame = grabber.grab();
            byte[] result = toByteArray(frame);

            if (aContext.getProperty(SAVE_IMAGES).asBoolean()) {
                opencv_imgcodecs.cvSaveImage(System.currentTimeMillis() + "-captured.png",
                        converter.convert(frame));
            }

            //transfer the image
            FlowFile flowFile = aSession.create();
            flowFile = aSession.write(flowFile, new OutputStreamCallback() {

                @Override
                public void process(final OutputStream aStream) throws IOException {

                    aStream.write(result);
                }
            });
            aSession.transfer(flowFile, REL_SUCCESS);
            aSession.commit();

            Thread.currentThread();
            Thread.sleep(Integer.parseInt(aContext.getProperty(FRAME_INTERVAL).getValue()));

            grabber.stop();

        } catch (Exception e) {
            logger.error("Something went wrong with the video capture!", e);
        } catch (InterruptedException e) {
            logger.error("Something went wrong with the threads!", e);
        } catch (IOException e) {
            logger.error("Something went wrong with saving the file!", e);
        }

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
}
