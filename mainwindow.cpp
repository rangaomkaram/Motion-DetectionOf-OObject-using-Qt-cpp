#include "mainwindow.h"
#include "ui_mainwindow.h"

//opencv
//#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

//c++
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <iomanip>

using namespace std;
using namespace cv;

/** Global variables */

int cam_number = 2; //from which camera to read
cv::Size CamFrameSize (640, 480);
cv::Size WinSize (800, 600);
//cv::Size WinSize (640, 480);
int houghparam1_slider = 50;
double hough_param1 = 50;
int houghparam2_slider = 50;
double hough_param2 = 50;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //process();
    // destroyAllWindows();
    return;

}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::webcam()
{
    namedWindow("Frame1");
    Mat frame1 = Mat::zeros( CamFrameSize, CV_8UC3 );
    VideoCapture capture;
    //if source is camera:
    capture.open(cam_number); //builtin on laptop
    //capture.open(2); //>1 = external webcam
    if(!capture.isOpened()){
        cout<<"ERROR ACQUIRING VIDEO FEED\n" << endl;
        return;
    }
    while( waitKey(30) != 27)
    {
        capture.read(frame1);
        //        int width = tmp.cols;
        //        int height = tmp.rows;
        //        cout<<"width: " << width << " height:" << height << endl;

        //-- Show what we got
        imshow( "Frame1", frame1 );
    }
    //destroy GUI windows
    destroyWindow("Frame1");
    return;

}

void MainWindow::numeric()
{
    //create GUI windows
    //namedWindow("Frame1",WINDOW_AUTOSIZE);
    namedWindow("Frame_numeric",WINDOW_AUTOSIZE);
    VideoCapture capture;
    //Mat tmp;
    Mat frame1 = Mat::zeros( CamFrameSize, CV_8UC3 );
    int intensity_region_cols = 66;
    int intensity_region_rows = 90;

    cout<<"intensity_region_width: " << intensity_region_cols << "x" << intensity_region_rows << endl;

    int center_screen_x = CamFrameSize.width/2;
    int center_screen_y = CamFrameSize.height/2;
    cout<<"center_screen: " << center_screen_x << "x" << center_screen_y << endl;

    int intensity_region_cols_start = center_screen_x -  intensity_region_cols/2;
    int intensity_region_cols_end = intensity_region_cols_start + intensity_region_cols;
    int intensity_region_rows_start = center_screen_y -  intensity_region_rows/2;
    int intensity_region_rows_end = intensity_region_rows_start + intensity_region_rows;
//file:///home/sys3d/a_svn_sys3d/fhnb/Neujahrskoll2019/Forschung19.odp
    cout<<"intensity_region_x_from: " << intensity_region_cols_start << "to " << intensity_region_cols_end << endl;
    cout<<"intensity_region_y_from: " << intensity_region_rows_start << "to " << intensity_region_rows_end << endl;

    //Mat frame_gray = Mat::zeros( CamFrameSize, CV_8U );

    capture.open(cam_number); //>1 = external webcam

    if(!capture.isOpened()){
        cout<<"ERROR ACQUIRING VIDEO FEED\n" << endl;
        return;
    }

    //check for keyboard input
    while( waitKey(30) != 27)
    {
        //Mat frame_num = Mat::zeros( cv::Size(1024,768), CV_8U );
        Mat frame_gray = Mat::zeros( CamFrameSize, CV_8U );
        Mat frame_num = Mat::zeros( cv::Size(1024,768), CV_8UC3 );


        //read first frame
        capture.read(frame1);
        cvtColor(frame1,frame_gray,COLOR_RGB2GRAY);

        //NOTE: in OpenCV pixels are accessed in (row,col) format
        for( int y = intensity_region_rows_start; y < intensity_region_rows_end; y++ ) {
            for( int x = intensity_region_cols_start; x < intensity_region_cols_end; x++ ) {
//                            for( int y = 0; y < frame_gray.rows; y++ ) {
//                                for( int x = 0; x < frame_gray.cols; x++ ) {
                //                    //for( int c = 0; c < image.channels(); c++ ) {
               // cout << y << x << endl;
                int intensity = static_cast<int>(frame_gray.at<uchar>(y,x));
                //int intensity = (int)frame_gray.at<uchar>(y,x);
                //write the position of the object to the screen
                int x_print = (x-intensity_region_cols_start) *24;
                int y_print = (y-intensity_region_rows_start)*8;

                //putText(frame_num, to_string(intensity),Point(x_print,y_print),1,0.7,255,1);
                putText(frame_num, to_string(intensity),Point(x_print,y_print),1,0.7,Scalar(255,45,0),1);

            }
        }


        //-- Show what we got
        //imshow( "Frame1", frame_gray );
        imshow( "Frame_numeric", frame_num );
        //imshow( "Frame1", frame1 );
    }

    //destroy GUI windows
    destroyAllWindows();
    return;
}

void MainWindow::edge()
{
    namedWindow("Edges");
    Mat tmp, frame1;
    VideoCapture capture;
    //if source is camera:
    capture.open(cam_number); //builtin on laptop
    //capture.open(2); //>1 = external webcam
    if(!capture.isOpened()){
        cout<<"ERROR ACQUIRING VIDEO FEED\n" << endl;
        printf( "error video opening");
        getchar();
        return;
    }
    while( waitKey(30) != 27)
    {
        capture.read(tmp);
        cv::resize(tmp, frame1, WinSize);
        Mat frame_gray = Mat::zeros( frame1.size(), CV_8U );
        Mat frame_blurr = Mat::zeros( frame1.size(), CV_8U );

        // convert to grayscale
        cvtColor(frame1,frame_gray,COLOR_RGB2GRAY);

        // Blur Effect to reduce noise
        GaussianBlur(frame_gray,frame_blurr,cv::Size(5, 5),1.8);

        // sobel/Scharr
        Mat grad_x, grad_y, imgSobel, imgScharr, imgCanny;
        Mat abs_grad_x, abs_grad_y;
        int scale = 1;    //parameters for Sobel and Scharr operator
        int delta = 0;    //parameters for Sobel and Scharr operator
        int ddepth = CV_16S;  // The depth of the output image. We set it to CV_16S to avoid overflow.

        int lowTh = 45;        //low threshold for canny edge detector
        int highTh = 90;

        //Sobel
        Sobel( frame_blurr, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        Sobel( frame_blurr, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        convertScaleAbs( grad_y, abs_grad_y );
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgSobel );

        //Scharr
        Scharr( frame_blurr, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        Scharr( frame_blurr, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        convertScaleAbs( grad_y, abs_grad_y );
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgScharr );

        // Canny (= GaussianBlur5x5, gradient, Nonmaximum suppression - upper threshold, Thresholding with hysterysis - upper/lower threshold)
        cv::Canny(frame_blurr,imgCanny,lowTh,highTh);       // Canny Edge Image

        //-- Show what we got
        imshow( "Edges", imgCanny );
    }
    //destroy GUI windows
    destroyWindow("Edges");
    return;
}

void MainWindow::on_hough_param1_trackbar( int, void* ){
    hough_param1 = static_cast<double> (houghparam1_slider);

}
void MainWindow::on_hough_param2_trackbar( int, void* ){
    hough_param2 = static_cast<double> (houghparam2_slider);

}

void MainWindow::hough()
{
    namedWindow("Frame1");
    Mat tmp, frame1;
    float old_x = 0.0, old_y = 0.0;
    int keyboard = 0;
    bool paused = false;

//    char Trackbar_hough_param1_Name[50];
//    sprintf( Trackbar_hough_param1_Name, "hough_param1 x %d", 100 );
    createTrackbar( "hough_param1", "Frame1", &houghparam1_slider, 100, on_hough_param1_trackbar );
    createTrackbar( "hough_param2", "Frame1", &houghparam2_slider, 100, on_hough_param2_trackbar );



    VideoCapture capture;
    //if source is camera:
    capture.open(cam_number); //builtin on laptop
    //capture.open(2); //>1 = external webcam
    if(!capture.isOpened()){
        cout<<"ERROR ACQUIRING VIDEO FEED\n" << endl;
        printf( "error video opening");
        getchar();
        return;
    }
    while( keyboard != 27)
    {
        //capture.read(tmp);
        capture.read(frame1);
        //cv::resize(tmp, frame1, WinSize);

        Mat frame_gray = Mat::zeros( frame1.size(), CV_8U );

        Mat frame_blurr = Mat::zeros( frame1.size(), CV_8U );

        // convert to grayscale
        cvtColor(frame1,frame_gray,COLOR_RGB2GRAY);

        // Blur Effect to reduce noise
        GaussianBlur(frame_gray,frame_blurr,cv::Size(5, 5),1.8);
        equalizeHist( frame_gray, frame_gray );

        vector<Vec3f> circles;
        // Apply the Hough Transform to find the circles
        HoughCircles( frame_blurr, circles, HOUGH_GRADIENT, 1, frame_gray.rows/8,  hough_param1, hough_param2, 10, 60 );

        //sort circles from x_min to x_max
        // vector<Vec3f> circles_sorted;
        Vec3f temp;
        //std::sort(circles.begin(), circles.end);
        // std::sort
        for( size_t i = 0; i < circles.size(); i++ )
        {
            for( size_t j = 0; j < circles.size(); j++ )
            {
                if(circles[i][0] > circles[j][0])
                {
                    temp = circles.at(i);
                    circles.at(i) = circles.at(j);
                    circles.at(j) = temp;
                }
            }
        }



        if (!paused)
//        if (circles.size() > 0 && ((abs(old_x - circles[0][0]) > 5.0) || (abs(old_y - circles[0][1]) > 5.0)))
       {
            // Draw the circles detected
            for( size_t i = 0; i < circles.size(); i++ ){
                Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));

                int radius = cvRound(circles[i][2]);

                //draw circles
                // circle center
                circle( frame1, center, 3, Scalar(0,255,0), -1, 8, 0 );
                // circle outline
                circle( frame1, center, radius, Scalar(255,150,255), 1, 8, 0 );

                //draw measurement lines
                int lastcircle_nr = static_cast<int>(circles.size()) - 1;

                int end_vert_line = cvRound(circles[lastcircle_nr][1])+100;
                int start_hor_line_x = cvRound(circles[i][0]);
                int end_hor_line_x = cvRound(circles[lastcircle_nr][0]);
                Point endofline(cvRound(circles[i][0]), end_vert_line);
                //Point endofline(cvRound(circles[i][0]), cvRound(circles[lastcircle_nr][1])+200);
                Point dimensionlineStart(start_hor_line_x, end_vert_line);
                Point dimensionlineEnd(end_hor_line_x, end_vert_line);

                line(frame1, center, endofline, Scalar(0,255,0),1,8,0);
                line(frame1, dimensionlineStart, dimensionlineEnd, Scalar(0,255,0),2,8,0);
                putText(frame1,intToString(cvRound(circles[i][0])),endofline,1,1.5,Scalar(255,45,45),2);

                //remind user of keyboard options
                putText(frame1,"p",Point(20,20),1,1,Scalar(255,45,45),2);



                //-- Show what we got
                imshow( "Frame1", frame1 );
                }
        }
        keyboard = waitKey(100);
        if (keyboard == 'p') {paused = !paused;}

    }
    //destroy GUI windows
    destroyWindow("Frame1");
    return;

}

//int to string helper function
string MainWindow::intToString(int number){

    //this function has a number input and string output
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void MainWindow::motiondetect()
{
    //our sensitivity value to be used in the absdiff() function
    const static int SENSITIVITY_VALUE = 20;
    //size of blur used to smooth the intensity image output from absdiff() function
    const static int BLUR_SIZE = 10;
    //some boolean variables for added functionality
    //these two can be toggled by pressing 'd' or 't'
    bool debug = false;
    bool trackingEnabled = false;
    //pause and resume code
    bool pause = false;
    //set up the matrices that we will need
    //the two frames we will be comparing
    Mat frame1,frame2;
    //their grayscale images (needed for absdiff() function)
    Mat grayImage1,grayImage2;
    //resulting difference image
    Mat differenceImage;
    //thresholded difference image (for use in findContours() function)
    Mat thresholdImage;

    //create GUI windows
    namedWindow("Frame1");

    //video capture object
    VideoCapture capture;

    while(1){
        //we can loop the video by re-opening the capture every time the video 	  reaches its last frame
        //capture.open("Steel_Bearings_06___4K_res.mp4");
        capture.open("bouncingBall.avi");
        if(!capture.isOpened()){
            cout<<"ERROR ACQUIRING VIDEO FEED\n";
            return;
        }
        //check if the video has reached its last frame.
        //we add '-1' because we are reading two frames from the video at a 	     time.
        //if this is not included, we get a memory error!
        while(capture.get(CAP_PROP_POS_FRAMES)<capture.get(CAP_PROP_FRAME_COUNT)-1){
            //read first frame
            capture.read(frame1);
            //convert frame1 to gray scale for frame differencing
            cv::cvtColor(frame1,grayImage1,COLOR_BGR2GRAY);
            //copy second frame
            capture.read(frame2);
            //convert frame2 to gray scale for frame differencing
            cv::cvtColor(frame2,grayImage2,COLOR_BGR2GRAY);
            //perform frame differencing with the sequential images. This will output an "intensity image"
            //do not confuse this with a threshold image, we will need to perform thresholding afterwards.
            cv::absdiff(grayImage1,grayImage2,differenceImage);
            //threshold intensity image at a given sensitivity value
            cv::threshold(differenceImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);

            //show the difference image and threshold image
            //cv::imshow("Difference Image",differenceImage);
            //cv::imshow("Threshold Image", thresholdImage);
            //blur the image to get rid of the noise. This will output an intensity image
            cv::blur(thresholdImage,thresholdImage, cv::Size(BLUR_SIZE,BLUR_SIZE));

            //threshold again to obtain binary image from blur output
            cv::threshold(thresholdImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);

            //show the threshold image after it's been "blurred"
            if (debug) {imshow("Final Threshold Image",thresholdImage);}

            MainWindow::searchForMovement(thresholdImage,frame1);

            //flip(frame1, frame1, 0);    //turn image upside down
            //show our captured frame
            imshow("Frame1",frame1);

            //check to see if a button has been pressed.
            switch(waitKey(10)){
            case 27: //'esc' key has been pressed, exit program.
            {//destroy GUI windows
                destroyAllWindows();
                return;
            }
            case 116: //'t' has been pressed. this will toggle tracking
                trackingEnabled = !trackingEnabled;
                if(trackingEnabled == false) cout<<"Tracking disabled."<<endl;
                else cout<<"Tracking enabled."<<endl;
                break;
            case 100: //'d' has been pressed. this will debug mode
                debug = !debug;
                if(debug == false) cout<<"Debug mode disabled."<<endl;
                else cout<<"Debug mode enabled."<<endl;
                break;
            case 112: //'p' has been pressed. this will pause/resume the code.
                pause = !pause;
                if(pause == true){ cout<<"Code paused, press 'p' again to 							resume"<<endl;
                    while (pause == true){
                        //stay in this loop until
                        switch (waitKey()){
                        //a switch statement inside a switch statement? Mind blown.
                        case 112:
                            //change pause back to false
                            pause = false;
                            cout<<"Code resumed."<<endl;
                            break;
                        }
                    }
                }
            }
        }
        //release the capture before re-opening and looping again.
        capture.release();

    }

}

void MainWindow::searchForMovement(Mat thresholdImage, Mat &cameraFeed){
    //notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
    //to take the values passed into the function and manipulate them, rather than just working with a copy.
    //eg. we draw to the cameraFeed to be displayed in the main() function.

    int theObject[2] = {0,0};
    bool objectDetected = false;

    //bounding rectangle of the object, we will use the center of this as its position.
    Rect objectBoundingRectangle = Rect(0,0,0,0);


    Mat temp;
    thresholdImage.copyTo(temp);
    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    findContours(temp,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE );// retrieves external contours

    //if contours vector is not empty, we have found some objects
    if(contours.size()>0)objectDetected=true;
    else objectDetected = false;

    if(objectDetected){
        //		 printf( "huhu");

        //the largest contour is found at the end of the contours vector
        //we will simply assume that the biggest contour is the object we are looking for.
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size()-1));
        //make a bounding rectangle around the largest contour then find its centroid
        //this will be the object's final estimated position.
        objectBoundingRectangle = boundingRect(largestContourVec.at(0));
        int x_mid = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
        int y_mid = objectBoundingRectangle.y+objectBoundingRectangle.height/2;
        //upper left corner
        int x_ul = objectBoundingRectangle.x;
        int y_ul = objectBoundingRectangle.y;
        //lower right corner
        int x_lr = objectBoundingRectangle.x + objectBoundingRectangle.width;
        int y_lr = objectBoundingRectangle.y + objectBoundingRectangle.height;

        //update the objects positions by changing the 'theObject' array values
        theObject[0] = x_mid; theObject[1] = y_mid;

        //draw some crosshairs around the object
        rectangle(cameraFeed,objectBoundingRectangle, Scalar(0,255,0),2);
        //	rectangle(cameraFeed,Point(x_lr,y_lr), Point(x_ul,y_ul), Scalar(0,255,0),2);

        //line(cameraFeed,Point(x_mid,10),Point(x_mid,400),Scalar(0,255,0),2);

        //write the position of the object to the screen
        putText(cameraFeed,"Position", Point(30,30),1,2,Scalar(255,255,145),2);
        putText(cameraFeed,intToString(x_mid),Point(200,30),1,2,Scalar(255,255,145),2);
        putText(cameraFeed,intToString(y_mid),Point(270,30),1,2,Scalar(255,255,145),2);

    }

}

void MainWindow::streifenlicht()
{
    //create GUI windows
    namedWindow("Frame1", WINDOW_AUTOSIZE);
    Mat frame1;
    //pause and resume code
    bool pause = false;

    //video capture object
    VideoCapture capture;

    while(1){
        //we can loop the video by re-opening the capture every time the video 	  reaches its last frame
        capture.open("picasion.gif");
        if(!capture.isOpened()){
            cout<<"ERROR ACQUIRING VIDEO FEED\n" <<endl;
            //getchar();
            return;
        }
        while(capture.get(CAP_PROP_POS_FRAMES)<10) //this gif has 10 frames
        {
            capture.read(frame1);
            //  cout<<"did read frame" <<endl;
            imshow("Frame1",frame1);

            switch(waitKey(1000)){
            case 27: //'esc' key has been pressed, exit program.
            {//destroy GUI windows
                destroyAllWindows();
                return;
            }
            case 112: //'p' has been pressed. this will pause/resume the code.
                pause = !pause;
                if(pause == true){ cout<<"Code paused, press 'p' again to resume"<<endl;
                    while (pause == true){
                        //stay in this loop until
                        switch (waitKey()){
                        //a switch statement inside a switch statement? Mind blown.
                        case 112:
                            //change pause back to false
                            pause = false;
                            cout<<"Code resumed."<<endl;
                            break;
                        }
                    }
                }
            }
        }
        //release the capture before re-opening and looping again.
        capture.release();
    }

}

void MainWindow::cam_calib()
{
    Mat frame;
    Mat drawToFrame;
    Mat cameraMatrix = Mat::eye(3,3,CV_64F);  //creates a Matlab-style identity matrix
    Mat distanceCoefficients;
    const Size chessboardDimensions=Size(9,6); //this is the crosses between the fields



    vector<Mat> savedImages;

    //vector<vector<Point2f>> makrkerCorners, rejectedCandidates;

    VideoCapture vid(cam_number);

    if(!vid.isOpened())
    {
        cout<<"ERROR ACQUIRING VIDEO FEED\n";
        exit(EXIT_FAILURE);
    }
    else{
        cout<<"Juhu, found the VIDEO FEED\n";
    }

    namedWindow("Calibration", WINDOW_AUTOSIZE);

    while(true)
    {
        if(!vid.read(frame))
            break;
        vector<Vec2f> foundPoints;
        bool found = false;
        found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(drawToFrame);
        drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);
        if(found)
            imshow("Calibration", drawToFrame);
        else
            imshow("Calibration", frame);

        switch(waitKey(10))
        {
        case 27:
            //exit by pressing ESC
            destroyWindow("Calibration");
            cout<<"Programm beendet."<<endl;
            return;
        }
    }
}

void MainWindow::featuredetection()
{
    Mat imgOriginal;       // image to be searched in to locate keypoints
    Mat imgOrigGray;       // grayscale image
    Mat imgTemplate;       // image of object to find
    Mat imgTemplateGray;   // grayscale image

    //load the first image, usually they are too big and colored --> so resize 	and convert to gray
    Mat Temp; //neded for resize
    //imgOriginal = imread("U://opencvProj//UE_6//Pictures//pic3.jpg", 	IMREAD_COLOR);
    imgOriginal = imread("feature_pic1.jpg",  	IMREAD_COLOR);
    cv::resize(imgOriginal, Temp,  Size(600,480),0,0, INTER_LINEAR); //make 	smaller
    cvtColor(Temp, imgOrigGray, COLOR_BGR2GRAY);  // convert to grayscale
    //load the second image, resize and convert to gray
    //imgTemplate = imread("U://opencvProj//UE_6//Pictures//pic4.jpg", 	IMREAD_COLOR);
    imgTemplate = imread("feature_pic2.jpg", 	IMREAD_COLOR);
    cv::resize(imgTemplate, Temp,  Size(600,480),0,0, INTER_LINEAR); //make 	smaller
    cvtColor(Temp, imgTemplateGray, COLOR_BGR2GRAY);  // convert to grayscale
    std::cout << "images loaded\n"; //just for debugging


    //next, apply the SURF-Operator to detect keypoints in both images:

    // Define keypoints vector
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    // Define feature detector
    cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SURF::create(2000.0);
    //cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(2000.0);
    // Keypoint detection
    ptrFeature2D->detect(imgOrigGray,keypoints1);
    ptrFeature2D->detect(imgTemplateGray,keypoints2);
    // Extract the descriptor
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    ptrFeature2D->compute(imgOrigGray,keypoints1,descriptors1);
    ptrFeature2D->compute(imgTemplateGray,keypoints2,descriptors2);


    // Match the two image descriptors
    std::vector<cv::DMatch> matches;
    // Construction of the matcher
    //cv::BFMatcher matcher(cv::NORM_L2);
    //matcher.match(descriptors1,descriptors2, matches);
    //alterantive matcher2: repeat the same procedure a second time, but this time,
    //each keypoint of the second image is compared with all the
    // keypoints of the first image. A match is considered valid only if we obtain the same pair of
    //keypoints in both directions
    cv::BFMatcher matcher2(cv::NORM_L2, true);
    matcher2.match(descriptors1,descriptors2, matches);

    // next, take only the best matches, delete the rest and draw them in a new	image:
    // extract the 80 best matches, they are at the beginning
    std::nth_element(matches.begin(), matches.begin()+40, matches.end());
    //delete the rest
    matches.erase(matches.begin()+40, matches.end());
    //Draw the matching results
    cv::Mat imgMatch;   //new output image, combines the 2 orig images and draws lines between corr. keypoints
    cv::drawMatches(imgOrigGray,keypoints1, imgTemplateGray,keypoints2, matches, imgMatch, cv::Scalar(255,0,255), cv::Scalar(255,0,0));
    // explanation:   cv::drawMatches((imgOrigGray,keypoints1, // first image and its keypoints
    //                imgTemplateGray,keypoints2, // second image and its keypoints
    //                matches, // vector of matches
    //                imgMatch, //this is the output image
    //                cv::Scalar(255,255,255), // color of lines
    //                cv::Scalar(255,255,255)); // color of points

    //next, declare some windows and show the result:

    cv::namedWindow("imgMatches", WINDOW_AUTOSIZE);
    // show in these windows our images

    cv::imshow("imgMatches", imgMatch);

    //check for keyboard input
    while(1)
    {
        if(waitKey(30) == 27)
        {//destroy GUI window
            destroyWindow("imgMatches");
            return;
        }
    }
}

bool MainWindow::loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
    ifstream inStream(name);
    if (inStream)
    {
        uint16_t rows;
        uint16_t columns;

        inStream>>rows;
        inStream>>columns;

        cameraMatrix=Mat(Size(columns, rows), CV_64F);

        for(int r=0; r<rows; r++)
        {
            for(int c=0; c < columns; c++)
            {
                double read=0.0f;
                inStream>>read;
                cameraMatrix.at<double>(r,c)=read;
                cout<<cameraMatrix.at<double>(r,c)<<"\n";
            }
        }
        //Distance Coefficients
        inStream>>rows;
        inStream>>columns;

        distanceCoefficients=Mat::zeros(rows, columns, CV_64F);

        for(int r=0; r<rows; r++)
        {
            for(int c=0; c < columns; c++)
            {
                double read=0.0f;
                inStream>>read;
                distanceCoefficients.at<double>(r,c)=read;
                cout<<distanceCoefficients.at<double>(r,c)<< "\n";
            }
        }
        inStream.close();
        return true;
    }
    cout<<"ERROR reading Cameracalibration\n" << endl;
    return false;
}

int MainWindow::startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension)

{
    Mat frame;
    std::ostringstream vector_to_marker;
    vector<int>markerIds;
    vector<vector<Point2f>>markerCorners, rejectedCandidates;
    aruco::DetectorParameters parameters;

    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

    VideoCapture vid(cam_number);

    if(!vid.isOpened())
    {
        return -1;
    }

    namedWindow("Position Monitoring", WINDOW_AUTOSIZE);
    vector<Vec3d>rotationVectors, translationVectors;
    //      Vec3f &theta;
    //        //double theta = 45;
    //           // Calculate rotation about x axis
    //           Mat R_x = (Mat_<double>(3,3) <<
    //                      1,       0,              0,
    //                      0,       cos(theta[0]),   -sin(theta[0]),
    //                      0,       sin(theta[0]),   cos(theta[0])
    //                      );

    //           // Calculate rotation about y axis
    //           Mat R_y = (Mat_<double>(3,3) <<
    //                      cos(theta[1]),    0,      sin(theta[1]),
    //                      0,               1,      0,
    //                      -sin(theta[1]),   0,      cos(theta[1])
    //                      );

    //           // Calculate rotation about z axis
    //           Mat R_z = (Mat_<double>(3,3) <<
    //                      cos(theta[2]),    -sin(theta[2]),      0,
    //                      sin(theta[2]),    cos(theta[2]),       0,
    //                      0,               0,                  1);


    //           // Combined rotation matrix
    //           Mat R = R_z * R_y * R_x;



    while(true)
    {
        if(!vid.read(frame))
            break;
        //cout<<"columns: "<< frame.cols <<endl;
        //cout<<"rows: "<< frame.rows <<endl;
        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);



        for(int i=0; i < markerIds.size(); i++)
        {
//            double half_side = 0;
//            cv::Mat rot_mat;
//            Rodrigues(rotationVectors[i], rot_mat);

//            // transpose of rot_mat for easy columns extraction
//            Mat rot_mat_t = rot_mat.t();
//            // transform along z axis
//            double * rz = rot_mat_t.ptr<double>(2); // x=0, y=1, z=2
//            translationVectors[i][0] +=  rz[0]*half_side;
//            translationVectors[i][1] +=  rz[1]*half_side;
//            translationVectors[i][2] +=  rz[2]*half_side;

            vector_to_marker << std::fixed;
            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(2)
                             << "x: " << std::setw(0)<<  translationVectors[0][0]
                             << "meters";
            cv::putText(frame, vector_to_marker.str(),
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 252, 124), 2);

            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(2)
                             << "y: " << std::setw(0) << translationVectors[0](1)
                             << "meters";
            cv::putText(frame, vector_to_marker.str(),
                        Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 252, 124), 2);

            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(2)
                             << "z: " << std::setw(0) << translationVectors[0](2)
                             << "meters";
            cv::putText(frame, vector_to_marker.str(),
                        Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 252, 124), 2);

            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(0)
                             << "pitch: " << std::setw(0)<<  (rotationVectors[i][0] * 57.295) << "degrees";
            cv::putText(frame, vector_to_marker.str(),
                        Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 252, 124), 2);
            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(0)
                             << "roll: " << std::setw(0)<<  (rotationVectors[i][1] * 57.295) << "degrees";
            cv::putText(frame, vector_to_marker.str(),
                        Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 252, 124), 2);
            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(0)
                             << "yaw: " << std::setw(0)<<   (rotationVectors[i][2] * 57.295) << "degrees";
            cv::putText(frame, vector_to_marker.str(),
                        Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(255, 252, 124), 2);
            aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i],
                            arucoSquareDimension * 1.5f);



        }
        imshow("Position Monitoring", frame);
        if(waitKey(30)>=0) break;
    }
    destroyAllWindows();
    return 1;
}


//int MainWindow::startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension)
//{
//    Mat frame;
//    vector<int>markerIds;
//    vector<vector<Point2f>>markerCorners, rejectedCandidates;
//    aruco::DetectorParameters parameters;

//    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

//    VideoCapture vid(cam_number);

//    if(!vid.isOpened())
//    {
//        return -1;
//    }

//    namedWindow("navigation", WINDOW_AUTOSIZE);

//    vector<Vec3d>rotationVectors, translationVectors;

//    while(true)
//    {
//        if(!vid.read(frame))
//            break;

//        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
//        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);


//        for(int i=0; i < markerIds.size(); i++)
//        {
//            aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f);

//        }
//        imshow("navigation", frame);
//        if(waitKey(30) == 27)
//        {
//            destroyWindow("navigation");
//            break;
//        }
//    }
//    return 1;
//}

void MainWindow::aruco()
{
    Mat cameraMatrix = Mat::eye(3,3,CV_64F);  //creates a Matlab-style identity matrix
    Mat distanceCoefficients;
    const float arucoSquareDimension=0.2f; //size of our future aruco markers

    loadCameraCalibration("Cameracalibration", cameraMatrix, distanceCoefficients);
    startWebcamMonitoring(cameraMatrix, distanceCoefficients, arucoSquareDimension);
}

void MainWindow::facedetect() {

    //-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
    String face_cascade_name = "haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    //string window_name = "Capture - Face detection";
    std::vector<Rect> faces;
    namedWindow("Frame1",1);   //create GUI window

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return; };
    //cout << "cascades loaded" << endl;
    //printf( "cascades loaded");

    VideoCapture capture;

    //if source is camera:
    capture.open(cam_number); //>1 = external webcam

    if(!capture.isOpened()){
        cout<<"ERROR ACQUIRING VIDEO FEED\n" << endl;
        printf( "error video opening");
        return;
    }
    Mat tmp, frame1;

    //check for keyboard input
    while( waitKey(10) != 27)
    {

        //read first frame
        capture.read(tmp);

        cv::resize(tmp, frame1, WinSize);
        Mat frame_gray = Mat::zeros( frame1.size(), CV_8U );
        cvtColor( frame1, frame_gray, COLOR_BGR2GRAY );
        equalizeHist( frame_gray, frame_gray );
        //-- Detect faces
        face_cascade.detectMultiScale( frame_gray, faces );
        //   face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        for( size_t i = 0; i < faces.size(); i++ )
        {
            Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
            ellipse( frame1, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );

            Mat faceROI = frame_gray( faces[i] );
            std::vector<Rect> eyes;

            //-- In each face, detect eyes
            //        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAL_CMP_GE, Size(30, 30) );
            //            eyes_cascade.detectMultiScale( faceROI, eyes );

            //            for( size_t j = 0; j < eyes.size(); j++ )
            //            {
            //                Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            //                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            //                circle( frame1, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
            //            }
        }
        //-- Show what we got
        imshow( "Frame1", frame1 );

    }

    //destroy GUI windows
    destroyAllWindows();
    return;
}



void MainWindow::on_pBwebcam_clicked()
{
    ui->lineEdit->setText("webcam");
    MainWindow::webcam();
}

void MainWindow::on_pBfacedetect_clicked()
{
    ui->lineEdit->setText("facedetect");
    MainWindow::facedetect();
}

void MainWindow::on_pBExit_clicked()
{
    QApplication::quit();
}

void MainWindow::on_pBnumeric_clicked()
{
    ui->lineEdit->setText("numeric");
    MainWindow::numeric();
}

void MainWindow::on_pBedge_clicked()
{
    ui->lineEdit->setText("edgedetect");
    MainWindow::edge();
}

void MainWindow::on_pBhough_clicked()
{
    ui->lineEdit->setText("hough");
    MainWindow::hough();
}

void MainWindow::on_pBmotiondetect_clicked()
{
    ui->lineEdit->setText("motiondetect");
    MainWindow::motiondetect();
}

void MainWindow::on_pBstreifenlicht_clicked()
{
    ui->lineEdit->setText("streifenlicht");
    MainWindow::streifenlicht();
}

void MainWindow::on_pBfeature_clicked()
{
    ui->lineEdit->setText("featuredetection");
    MainWindow::featuredetection();
}

void MainWindow::on_pBcamcalib_clicked()
{
    ui->lineEdit->setText("cam_calib");
    MainWindow::cam_calib();
}

void MainWindow::on_pBaruco_clicked()
{
    ui->lineEdit->setText("aruco");
    MainWindow::aruco();
}
