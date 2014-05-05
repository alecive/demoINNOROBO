/* 
 * Copyright (C) 2013 iCub Facility - Istituto Italiano di Tecnologia
 * Author: Alessandro Roncone
 * email:  alessandro.roncone@iit.it
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

/**
\defgroup icub_demoINNOROBO demoINNOROBO
@ingroup icub_contrib_modules

A module for implementing the DEMO INNOROBO on the iCub.
It tests the positions of the chessboard's corners on both the image planes.

Date first release: 13/11/2013

CopyPolicy: Released under the terms of the GNU GPL v2.0.

\section intro_sec Description
This is a module for implementing the DEMO INNOROBO on the iCub.
It tests the positions of the chessboard's corners on both the image planes.

\section lib_sec Libraries 
YARP and OpenCV

\section parameters_sec Parameters

--context    \e path
- Where to find the called resource.

--from       \e from
- The name of the .ini file with the configuration parameters.

--name       \e name
- The name of the module (default demoINNOROBO).

--robot      \e rob
- The name of the robot (either "icub" or "icub"). Default "icub".
  If you are guessing: Yes, the test HAS to be performed on the real robot!

--rate       \e rate
- The period used by the thread. Default 100ms.

--verbosity  \e verb
- Verbosity level (default 1). The higher is the verbosity, the more
  information is printed out.

--type		 \e type
- The type of task (default '0').
  It's deprecated, but kept for retrocompatibility and for eventual future use.

\section portsc_sec Ports Created
- <i> /<name>/imgL:i </i> port that receives input from the left  camera
- <i> /<name>/imgR:i </i> port that receives input from the right camera
- <i> /<name>/imgL:o </i> port that sends the output of the left  camera
- <i> /<name>/imgR:o </i> port that sends the output of the right camera
- <i> /<name>/disp:i </i> port that receives the stereoDisparity
- <i> /<name>/disp:o </i> port that sends the stereoDisparity output

\section in_files_sec Input Data Files
None.

\section out_data_sec Output Data Files
- <i> left_START.png </i>  left  image acquired at the beginning of the test 
- <i> right_START.png </i> right image acquired at the beginning of the test
- <i> results.log </i>     log file. It stores the shift (on the x and y axis)
		in both the left and right images for any of the 48 corners detected, the
		configuration file used, some statistics on the errors (min, max, avg), and
		the values of the encoders at the beginning and the end of the test (they 
		should usually be equal).
- <i> left_END.png </i>    left  image acquired at the end of the test
- <i> right_END.png </i>   right image acquired at the end of the test
 
\section tested_os_sec Tested OS
Linux (Ubuntu 12.04, Debian Squeeze).

\author Alessandro Roncone
*/ 

#include <cv.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>

#include <iostream>
#include <string.h> 
#include <ctime>
#include <sstream>

#include "Detector.h"

YARP_DECLARE_DEVICES(icubmod)

using namespace yarp;
using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::sig;
using namespace cv;

using namespace std;

/**
* \ingroup demoINNOROBOModule
*
* The module that achieves the demoINNOROBO task.
*  
*/
class demoINNOROBO: public RFModule
{
private:
    string name;
    string robot;
    string arm;
    int verbosity;
    double rate;

    RpcClient             rpcClnt;
    RpcServer             rpcSrvr;

    ResourceFinder    *rf;
    PolyDriver         drvArm;
    PolyDriver         drvCart;
    PolyDriver         drvGaze;
    IEncoders         *iencs;
    IPositionControl  *iposs;
    IControlMode      *ictrl;
    IImpedanceControl *iimp;
    ICartesianControl *iarm;
    IGazeControl      *igaze;
    int contextGaze;
    int contextCart;
    int    nEncs;

    BufferedPort<ImageOf<PixelRgb> > *imagePortInR;  // port for reading images
    BufferedPort<ImageOf<PixelRgb> > *imagePortInL;  // port for reading images
    Port imagePortOutR; // port for streaming images
    Port imagePortOutL; // port for streaming images
    ImageOf<PixelRgb> *imageInR;
    ImageOf<PixelRgb> *imageInL;

    Port portOutInfo;      // port for communicating with iSpeak

    int    certainty;
    string strCascade;
    Detector* detectorL;
    Detector* detectorR;

public:
    demoINNOROBO()
    {
        imagePortInR  = new BufferedPort<ImageOf<PixelRgb> >;
        imagePortInL  = new BufferedPort<ImageOf<PixelRgb> >;   
    }

    bool configure(ResourceFinder &rf)
    {
        printf(" Starting Configure\n");
        this->rf=&rf;

        //****************** PARAMETERS ******************
        robot     = rf.check("robot",Value("icub")).asString().c_str();
        name      = rf.check("name",Value("demoINNOROBO")).asString().c_str();
        arm       = rf.check("arm",Value("right_arm")).asString().c_str();
        verbosity = rf.check("verbosity",Value(0)).asInt();    
        rate      = rf.check("rate",Value(0.5)).asDouble();

        if (arm!="right_arm" && arm!="left_arm")
        {
            printMessage(0,"ERROR: arm was set to %s, putting it to right_arm\n",arm.c_str());
            arm="right_arm";
        }

        printMessage(1,"Parameters correctly acquired\n");
        printMessage(1,"Robot: %s \tName: %s \tArm: %s\n",robot.c_str(),name.c_str(),arm.c_str());
        printMessage(1,"Verbosity: %i \t Rate: %g \n",verbosity,rate);

        //****************** DETECTOR ******************
        certainty     = rf.check("certainty", Value(10.0)).asInt();
        strCascade = rf.check("cascade", Value("haarcascade_frontalface_alt.xml")).asString().c_str();

        detectorL=new Detector();
        detectorL->open(certainty,strCascade);
        detectorR=new Detector();
        detectorR->open(certainty,strCascade);

        printMessage(1,"Detectors opened\n");

        //****************** DRIVERS ******************
        Property optionArm("(device remote_controlboard)");
        optionArm.put("remote",("/"+robot+"/"+arm).c_str());
        optionArm.put("local",("/"+name+"/"+arm).c_str());
        if (!drvArm.open(optionArm))
        {
            printf("Position controller not available!\n");
            return false;
        }

        Property optionCart("(device cartesiancontrollerclient)");
        optionCart.put("remote",("/"+robot+"/cartesianController/"+arm).c_str());
        optionCart.put("local",("/"+name+"/cart/").c_str());
        if (!drvCart.open(optionCart))
        {
            printf("Cartesian controller not available!\n");
            // return false;
        }

        Property optionGaze("(device gazecontrollerclient)");
        optionGaze.put("remote","/iKinGazeCtrl");
        optionGaze.put("local",("/"+name+"/gaze").c_str());
        if (!drvGaze.open(optionGaze))
        {
            printf("Gaze controller not available!\n");
            // return false;
        }

        // quitting conditions
        bool andArm=drvArm.isValid()==drvCart.isValid()==drvGaze.isValid();
        if (!andArm)
        {
            printMessage(0,"Something wrong occured while configuring drivers... quitting!\n");
            return false;
        }

        drvArm.view(iencs);
        drvArm.view(iposs);
        drvArm.view(ictrl);
        drvArm.view(iimp);
        drvCart.view(iarm);
        drvGaze.view(igaze);
        iencs->getAxes(&nEncs);

        iimp->setImpedance(0,  0.4, 0.03);
        iimp->setImpedance(1, 0.35, 0.03);
        iimp->setImpedance(2, 0.35, 0.03);
        iimp->setImpedance(3,  0.2, 0.02);
        iimp->setImpedance(4,  0.2, 0.00);

        ictrl -> setImpedancePositionMode(0);
        ictrl -> setImpedancePositionMode(1);
        ictrl -> setImpedancePositionMode(2);
        ictrl -> setImpedancePositionMode(3);
        ictrl -> setImpedancePositionMode(2);
        ictrl -> setImpedancePositionMode(4);

        igaze -> storeContext(&contextGaze);
        igaze -> setSaccadesStatus(false);
        igaze -> setNeckTrajTime(0.75);
        igaze -> setEyesTrajTime(0.5);

        iarm -> storeContext(&contextCart);

        printMessage(1,"Drivers opened\n");

        //****************** PORTS ******************
        imagePortInR        -> open(("/"+name+"/imageR:i").c_str());
        imagePortInL        -> open(("/"+name+"/imageL:i").c_str());
        imagePortOutR.open(("/"+name+"/imageR:o").c_str());
        imagePortOutL.open(("/"+name+"/imageL:o").c_str());
        portOutInfo.open(("/"+name+"/info:o").c_str());

        if (robot=="icub")
        {
            Network::connect("/icub/camcalib/left/out",("/"+name+"/imageL:i").c_str());
            Network::connect("/icub/camcalib/right/out",("/"+name+"/imageR:i").c_str());
        }
        else
        {
            Network::connect("/icubSim/cam/left",("/"+name+"/imageL:i").c_str());
            Network::connect("/icubSim/cam/right",("/"+name+"/imageR:i").c_str());
        }

        Network::connect(("/"+name+"/imageL:o").c_str(),"/demoINN/left");
        Network::connect(("/"+name+"/imageR:o").c_str(),"/demoINN/right");
        Network::connect(("/"+name+"/info:o").c_str(),"/iSpeak");

        rpcClnt.open(("/"+name+"/rpc:o").c_str());
        rpcSrvr.open(("/"+name+"/rpc:i").c_str());
        attach(rpcSrvr);
        
        printMessage(0,"Configure Finished!\n");
        return true;
    }

    double getPeriod()  { return rate; }
    bool updateModule()
    { 
        return true;
    }

    bool respond(const Bottle &command, Bottle &reply)
    {
        int ack =Vocab::encode("ack");
        int nack=Vocab::encode("nack");

        if (command.size()>0)
        {
            switch (command.get(0).asVocab())
            {
                //-----------------
                case VOCAB4('l','o','o','k'):
                {
                    if (lookForFaces())
                    {
                        iarm -> waitMotionDone();
                        sendMessage(0,"Hello, would you like to take one of our samples?\n");
                        reply.addVocab(ack);
                        return true;
                    }
                    reply.addVocab(nack);
                    return true;
                }
                case VOCAB4('s','o','r','r'):
                {
                    sendMessage(0,"Oh, sorry.\n");
                    reply.addVocab(ack);
                    return true;
                }
                case VOCAB4('t','h','a','n'):
                {
                    sendMessage(0,"Thank you!\n");
                    reply.addVocab(ack);
                    return true;
                }
                case VOCAB4('h','e','r','e'):
                {
                    sendMessage(0,"Here you go!\n");
                    reply.addVocab(ack);
                    return true;
                }
                case VOCAB4('h','o','m','e'):
                {
                    if (goHome())
                    {
                        reply.addVocab(ack);
                    }
                    else
                    {
                        reply.addVocab(nack);
                    }
                    return true;
                }
                case VOCAB4('a','s','k','n'):
                {

                    sendMessage(0,"Can you please give me another sample?\n");
                    reply.addVocab(ack);
                    return true;
                }
                //-----------------
                default:
                    return RFModule::respond(command,reply);
            }
        }

        reply.addVocab(nack);
        return true;
    }

    bool goHome()
    {
        printMessage(0,"Going home...\n");
        yarp::sig::Vector poss(7,0.0);
        yarp::sig::Vector vels(7,0.0);
        printMessage(1,"Configuring arm...\n");
        poss[0]=-30.0;  vels[0]=10.0;                     poss[1]=30.0;  vels[1]=10.0;
        poss[2]= 00.0;  vels[2]=10.0;                     poss[3]=45.0;  vels[3]=10.0;
        poss[4]= 00.0;  vels[4]=10.0;                     poss[5]=00.0;  vels[5]=10.0;
        poss[6]= 00.0;  vels[6]=10.0;

        for (int i=0; i<7; i++)
        {
            iposs->setRefSpeed(i,vels[i]);
            iposs->positionMove(i,poss[i]);
        }

        printMessage(1,"Configuring hand...\n");
        poss.resize(9,0.0);
        vels.resize(9,0.0);

        poss[0]=40.0;  vels[0]=60.0;                     poss[1]=10.0;  vels[1]=60.0;
        poss[2]=60.0;  vels[2]=60.0;                     poss[3]=70.0;  vels[3]=60.0;
        poss[4]=00.0;  vels[4]=60.0;                     poss[5]=00.0;  vels[5]=60.0;
        poss[6]=70.0;  vels[6]=60.0;                     poss[7]=100.0; vels[7]=60.0;
        poss[8]=240.0; vels[8]=120.0; 
        for (int i=7; i<nEncs; i++)
        {
            iposs->setRefAcceleration(i,1e9);
            iposs->setRefSpeed(i,vels[i-7]);
            iposs->positionMove(i,poss[i-7]);
        }
        return true;
    }

    bool lookForFaces()
    {
        yarp::sig::Vector pxl(2,0.0);
        yarp::sig::Vector pxr(2,0.0);

        while (true)
        {
            imageInR = imagePortInR -> read(false);
            imageInL = imagePortInL -> read(false);

            if (imageInL)
            {
                ImageOf<PixelBgr> imageOutL;
                detectorL->loop(imageInL,imageOutL);
                imagePortOutL.write(imageOutL);
            }

            if (imageInR)
            {
               ImageOf<PixelBgr> imageOutR;
               detectorR->loop(imageInR,imageOutR);
               imagePortOutR.write(imageOutR);
            }
            printMessage(2,"counterL: %i counterR: %i\n",detectorL->counter,detectorR->counter);
            if (detectorL->counter > certainty &&
                detectorR->counter > certainty)
            {
                pxl=detectorL->getCenterOfFace();
                pxr=detectorR->getCenterOfFace();
                printMessage(1,"I have found a face! pxl: %s pxr: %s\n",pxl.toString().c_str(),pxr.toString().c_str());
                if (pxl(0)!=0.0 && pxr(0)!=0.0)
                {
                    igaze->lookAtStereoPixels(pxl,pxr); 
                }
                return true;
            }
            Time::delay(0.1);
        }

        return false;
    }

    int printMessage(const int l, const char *f, ...) const
    {
        if (verbosity>=l)
        {
            fprintf(stdout,"*** %s: ",name.c_str());

            va_list ap;
            va_start(ap,f);
            int ret=vfprintf(stdout,f,ap);
            va_end(ap);

            return ret;
        }
        else
            return -1;
    }

    int sendMessage(const int l, const char *f, ...) const
    {
        string msg = f;
        Bottle b;
        b.clear();
        b.addString(msg.c_str());
        portOutInfo.write(b);

        if (verbosity>=l)
        {
            fprintf(stdout,"*** %s - ISPEAK: ",name.c_str());

            va_list ap;
            va_start(ap,f);
            int ret=vfprintf(stdout,f,ap);
            va_end(ap);

            return ret;
        }
        else
            return -1;
    }

    bool close()
    {
        printMessage(0,"Closing stuff..\n");
        igaze -> restoreContext(contextGaze);
        igaze -> stopControl();
        drvGaze.close();

        ictrl -> setPositionMode(0);
        ictrl -> setPositionMode(1);
        ictrl -> setPositionMode(2);
        ictrl -> setPositionMode(3);
        ictrl -> setPositionMode(2);
        ictrl -> setPositionMode(4);
        drvArm.close();

        iarm -> restoreContext(contextCart);
        iarm -> stopControl();
        drvCart.close();

        delete detectorL;
        detectorL = 0;

        return true;
    }
};

/**
* Main function.
*/
int main(int argc, char * argv[])
{
    YARP_REGISTER_DEVICES(icubmod)

    ResourceFinder rf;
    rf.setVerbose(true);
    rf.setDefaultContext("demoINNOROBO");
    rf.setDefaultConfigFile("demoINNOROBO.ini");
    rf.configure(argc,argv);

    if (rf.check("help"))
    {    
        cout << endl << "Options:" << endl;
        cout << "   --context    path:  where to find the called resource" << endl;
        cout << "   --from       from:  the name of the .ini file." << endl;
        cout << "   --name       name:  the name of the module (default demoINNOROBO)." << endl;
        cout << "   --robot      robot: the name of the robot. Default icub." << endl;
        cout << "   --rate       rate:  the period used by the thread. Default 100ms." << endl;
        cout << "   --verbosity  int:   verbosity level (default 1)." << endl;
        cout << endl;
        return 0;
    }

    Network yarp;
    if (!yarp.checkNetwork())
    {
        printf("No Network!!!\n");
        return -1;
    }

    demoINNOROBO chEyeTest;
    return chEyeTest.runModule(rf);
}
// empty line to make gcc happy
