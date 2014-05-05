// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-
/* 
 * Copyright (C) 2012 Department of Robotics Brain and Cognitive Sciences - Istituto Italiano di Tecnologia
 * Author: Ali Paikan
 * email:  ali.paikan@iit.it
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * http://www.robotcub.org/icub/license/gpl.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
*/
#include <string>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/sig/Image.h>
#include <opencv/cvaux.h>
#include <yarp/dev/Drivers.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/sig/Vector.h>

#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace yarp::sig;
using namespace std;

typedef struct __circle_t {
    float x;
    float y;
    float r;
}circle_t;


YARP_DECLARE_DEVICES(icubmod)

class Detector
{
public:

    Detector()
    {
        cvImage = NULL;
        display = NULL;
        counter = 0;
        face.x = 0;
        face.y = 0;
        face.r = 0;
        foundFace.x = 0;
        foundFace.y = 0;
        foundFace.r = 0;
        prev_x = prev_y = prev_z = 0.0; 
        certainty=3.0;
        strCascade="";
        // constructor
    }

    bool open(yarp::os::ResourceFinder &rf);
    bool open(int _c, string _sC);

    bool close();

    void loop(ImageOf<PixelRgb> *In, ImageOf<PixelBgr> &Out);

    bool interrupt();

public: 
    string strCascade;

    IplImage *cvImage;
    IplImage* display;
    unsigned int counter;
    circle_t face;
    circle_t foundFace;
    cv::CascadeClassifier cascade;

    yarp::sig::Vector getCenterOfFace();

private:
    bool isIn(circle_t& c1, circle_t& c2);
    void detectAndDraw( cv::Mat& img, double scale, circle_t &c);
	int certainty;
    double prev_x;
    double prev_y;
    double prev_z;
};

   
   



   
