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

#include "Detector.h"
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

using namespace yarp::sig;
using namespace yarp::os;
using namespace std;
using namespace cv;


void Detector::loop(ImageOf<PixelRgb> *In, ImageOf<PixelBgr> &Out)
{        
    IplImage *cvImage = (IplImage*)In->getIplImage();
                      
    CvSize size = cvSize(cvImage->width, cvImage->height);
    if(display)
        cvReleaseImage(&display);
    display = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvCvtColor(cvImage, display, CV_BGR2RGB);

    Mat imgMat = display;
    circle_t c;
    detectAndDraw(imgMat, 1.0, c);
    // check whether a right size face-bounded circle found
    if(c.r > 0)
    {
        if(isIn(c, face))
        {
            // we found an stable face
            if(++counter > certainty)
            {
		        foundFace.x = face.x;
		        foundFace.y = face.y;
		        foundFace.r = face.r;
                cvCircle(display, cvPoint(cvRound(face.x), cvRound(face.y)), 2, CV_RGB(0,255,0), 3, 8, 0 );
                cvCircle(display, cvPoint(cvRound(face.x), cvRound(face.y)), cvRound(face.r), CV_RGB(0,255,0), 2, 8, 0 );
            }
        }
        else
        {
            face = c;

            counter = 0;
            cvCircle(display, cvPoint(cvRound(c.x), cvRound(c.y)), 2, CV_RGB(255,0,0), 2, 8, 0 );
            cvCircle(display, cvPoint(cvRound(c.x), cvRound(c.y)), cvRound(c.r), CV_RGB(255,0,0), 1, 8, 0 );
        }
    }
    else
    {
        foundFace.x = 0;
        foundFace.y = 0;
        foundFace.r = 0;
        counter = 0;
    }

    // ImageOf<PixelRgb> &outImage = outPort.prepare(); //get an output image
    // outImage.resize(cvImage->width, cvImage->height);
    Out.resize(cvImage->width, cvImage->height);
    Out.wrapIplImage((IplImage*)display);
}


bool Detector::isIn(circle_t& c1, circle_t& c2)
{
    float margin = c2.r/2.0;
    if(c1.x < c2.x-margin)
        return false;
    if(c1.x > c2.x+margin)
        return false;
    if(c1.y < c2.y-margin)
        return false;
    if(c1.y > c2.y+margin)
        return false;
    return true;
}

yarp::sig::Vector Detector::getCenterOfFace()
{
    yarp::sig::Vector result(2,0.0);
    result(0)=foundFace.x;
    result(1)=foundFace.y;
    return result;
}

void Detector::detectAndDraw(cv::Mat& img, double scale, circle_t &c)
{
    int i = 0;
    double t = 0;
    vector<Rect> faces;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray;

    cvtColor( img, gray, CV_BGR2GRAY );
    equalizeHist( gray, gray );

    cascade.detectMultiScale( gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    //printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    c.r = 0;
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Point center;
        Scalar color = colors[i%8];
        int radius;
        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        radius = cvRound((r->width + r->height)*0.25*scale);
        // circle( img, center, radius, color, 3, 8, 0 );
        //  cvCircle(img, cvPoint(cvRound(center.x), cvRound(center.y)), cvRound(radius), CV_RGB(100,10,130), 1, 8, 0 );
        if(c.r < radius)		// take the biggest face
        {
           c.r = radius;
           c.x = center.x;
           c.y = center.y; 
        }
    }
}


bool Detector::open(int _c, string _sC)
{
    certainty=_c;
    strCascade=_sC;

    bool ret=true;
    ret = ret && cascade.load( strCascade.c_str());

    printf("DETECTOR:\n\t certainty: %i \n\tstrCascade: %s \n",
            certainty,strCascade.c_str());
    return ret;
}

bool Detector::open(yarp::os::ResourceFinder &rf)
{
	certainty  = rf.check("certainty", Value(3.0)).asInt();
    strCascade = rf.check("cascade", Value("haarcascade_frontalface_alt.xml")).asString().c_str();
 
    bool ret=true;
    ret = ret && cascade.load( strCascade.c_str());

    return ret;
}

bool Detector::close()
{
    return true;
}

bool Detector::interrupt()
{
    return true;
}


