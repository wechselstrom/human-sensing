/**
 * by: Stefan Weber (stefan.weber@iit.it)
 **/
// reference for YARP
#include <yarp/sig/Image.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Module.h>
#include <yarp/os/Network.h>
#include <yarp/sig/Vector.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include "opencv2/objdetect.hpp"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;



class MyModule:public RFModule
{
	
    BufferedPort<ImageOf<PixelRgb> > imageIn;  // make a port for reading images
    BufferedPort<ImageOf<PixelRgb> > imageOut; // make a port for passing the result to
	
	IplImage* cvImage;
    IplImage* display;
	
    dlib::shape_predictor sp;
    cv::CascadeClassifier face_cascade;


    dlib::image_window win, win_faces;

    int counter=0;
    std::vector<cv::Rect> dets;
	
	
public:



    MyModule(){}; 							// constructor 
    ~MyModule(){};							// deconstructor


    double getPeriod()						// the period of the loop.
    {
        return 0.0; 						// 0 is equal to the real-time
    }


    bool configure(yarp::os::ResourceFinder &rf)
    {
        
        bool ok2;
        ok2 = this->imageIn.open("/test/image/in");                           // open the ports
        ok2 = ok2 && this->imageOut.open("/test/image/out");                  //
		if (!ok2) {
			fprintf(stderr, "Error. Failed to open image ports. \n");
			return false;
		}	

        dlib::deserialize(rf.findFileByName("shape_predictor_68_face_landmarks.dat")) >> sp;
        if( !face_cascade.load((string)rf.findFileByName("haarcascade_frontalface_alt.xml")) ){ 
            std::cerr << "--(!)Error loading face cascade\n" << std::endl;
            return -1;
        }

		
		ImageOf<PixelRgb> *imgTmp = this->imageIn.read();  				// read an image
		if (imgTmp != NULL) 
		{ 
			IplImage *cvImage = (IplImage*)imgTmp->getIplImage();        
			ImageOf<PixelRgb> &outImage = this->imageOut.prepare(); 		//get an output image
			outImage.resize(imgTmp->width(), imgTmp->height());		
			outImage = *imgTmp;
			display = (IplImage*) outImage.getIplImage();
            //Mat captured_image = display;                         // opencv < 3.0
            cv::Mat captured_image = cv::cvarrToMat(display);               // opencv 3.0
		}

	
        return true;
    }




	bool updateModule()
	{


		//cout << __FILE__ << ": " << __LINE__ << endl;


		ImageOf<PixelRgb> *imgTmp = this->imageIn.read();  // read an image
		if (imgTmp == NULL) 
		{
			return true;
		}
		

		IplImage *cvImage = (IplImage*)imgTmp->getIplImage();        

		ImageOf<PixelRgb> &outImage = this->imageOut.prepare(); //get an output image
		outImage.resize(imgTmp->width(), imgTmp->height());		
		outImage = *imgTmp;
		display = (IplImage*) outImage.getIplImage();
        //Mat captured_image = display;
        cv::Mat captured_image = cv::cvarrToMat(display);

        dlib::cv_image<dlib::bgr_pixel> cimg(cvImage);


        cv::Mat frame_gray;
        cv::cvtColor( captured_image, frame_gray, cv::COLOR_BGR2GRAY );
        equalizeHist( frame_gray, frame_gray );
        if (!((counter++)%3)) {
            face_cascade.detectMultiScale( frame_gray, dets, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) );
        }

        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.
        std::vector<dlib::full_object_detection> shapes;
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            dlib::rectangle r = dlib::rectangle(dets[j].x, dets[j].y, dets[j].x+dets[j].width, dets[j].y+dets[j].height);
            dlib::full_object_detection shape = sp(cimg, r);
            if (shape.num_parts() == 68) {
                cv::Point2d mean = cv::Point2d(0, 0);
                for (long i=48; i < 60;i++) {
                    auto p = shape.part(i);
                    cv::circle(captured_image, cv::Point2d(p.x(), p.y()), 3, cv::Scalar( 255, 0, 0 ));
                    mean.x += p.x();
                    mean.y += p.y();
                }
                mean.x /= (60-48);
                mean.y /= (60-48);
                cv::circle(captured_image, cv::Point2d(mean.x, mean.y), 3, cv::Scalar( 0, 255, 0 ));
            }
        }

		this->imageOut.write();
		
		return true;
	}
	
    bool interruptModule()
    {
        cout<<"Interrupting your module, for port cleanup"<<endl;
        this->imageIn.interrupt();
        return true;
    }

    bool close()
    {
		cout<<"Calling close function\n";
		this->imageIn.close();
		this->imageOut.close();
		
		
        return true;
    }	
    
protected:

private:

};

// ----------------
// ----- Main -----
// ----------------

int main (int argc, char **argv)
{

	MyModule thisModule;
	ResourceFinder rf;
	cout<<"Object initiated!"<<endl;
	thisModule.configure(rf);
	thisModule.runModule();
	return 0;
}

