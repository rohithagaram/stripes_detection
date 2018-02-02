/**
 * @file detectStripes.cpp
 * detectStripes.cpp detects white stripes, using hybrid approach, so has 
 * less detection distance in rough, invariant-lighted road scenes
 *
 *

   USAGE GUIDELINE
 * Syntax: ./detectStripes <path to video> 
 * <path to video> : argument one
 * <method no> : argument two is only required to explicitly invoke detectStripeBars.cpp 
 * Example: ./detectStripes /home/rrc/Desktop/Nalco.mp4 


 * ALGORITHM OVERVIEW:
 *      > read a video frame 
 *      > resize frame to 320x240
 *      > get homography matrix using 4 fixed points, set heuristically
 *      > histogram equalization
 *      > binary global thresholding
 *      > erode and dilate with vertical structuring element 
 *      > blob detection 
 *      > applying stripe detection using y-location, blob diameter, inter-blob distance in x and y, inter-blob diameter
 *      > reverse y-location using inverse homography matrix
 *      > draw yellow rectangle, write black "Zebra", and draw red circles around blobs at y-location with +-10 precision
 *
 */


# include <stdio.h>
# include <math.h>
# include <vector>
# include <ctime>
# include "opencv2/core/core.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/imgproc/imgproc.hpp"
# include "opencv2/calib3d/calib3d.hpp"
# include "opencv2/features2d/features2d.hpp"
# include <iostream>



using namespace cv;
using namespace std;

/// global variables
Mat im_with_keypoints, org_im_with_keypoints; /// < stores original processed and output frames
int z_found = 0, z_yloc; /// < temporal traking variables :: OPTIONAL


int drawZebra(vector<KeyPoint> keypoints, Mat drawImg, Mat drawOrgImg, int xpos, int ypos) {


    Point xyLoc;
    rectangle(im_with_keypoints, Point(xyLoc.x=0, xyLoc.y=ypos), Point(xyLoc.x+drawImg.cols, xyLoc.y=ypos+15), Scalar(0,255,255));
    putText(im_with_keypoints,"Zebra", Point(xyLoc.x=0, xyLoc.y=ypos+15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1, LINE_AA);
    drawKeypoints( im_with_keypoints, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


    rectangle(org_im_with_keypoints, Point(xyLoc.x=0+20, xyLoc.y=ypos-10), Point(xyLoc.x+drawImg.cols+20, xyLoc.y=ypos+20-10), Scalar(0,255,255));
    putText(org_im_with_keypoints,"Zebra", Point(xyLoc.x=0+20, xyLoc.y=ypos+15-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1, LINE_AA);
    drawKeypoints( org_im_with_keypoints, keypoints, org_im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    return 1;
}


int zebraDetect(vector<KeyPoint> keypoints, Mat inImg1, Mat outImg1, int y_init=0) {

    //drawing zebra lines
    Point xyLoc;
    float angl, diam, diam1;
    int len, x=0, x1=0, y=0, y1=0, y_sum, y_max, y_max1, y_max2, angl_sum, k_max=0, k_max1=0, k_max2=0, y_beg, y_end;

    len = (int) keypoints.size(); 


    if (!keypoints.empty() && len >=3) {

                   
            for (int i=0; i<len; i++) {
                x = (int) keypoints[i].pt.x;
                y = (int) keypoints[i].pt.y;
                diam = (float) keypoints[i].size;
                int k=0;

                vector<KeyPoint> select_keypoint;
                select_keypoint.push_back(keypoints[i]);

                if (y<90) continue;
                if (x>300 || x<50 && y<120) continue;


                for (int j=i+1; j<len; j++) {
                    y1 = (int) keypoints[j].pt.y;
                    x1 = (int) keypoints[j].pt.x;
                    diam1 = (float) keypoints[j].size;
                    
                    if (x1>200 || x1<100 || y1<90) continue;
                    //(abs(y1-y)<5 && abs(x1-x)<100 && abs(diam1-diam)<10) // ::Best NALCO
                    //(abs(y1-y)<10 && abs(x1-x)<100 && abs(diam1-diam)<10) || (abs(y1-y)<10) && abs(diam1-diam)<15) //:: Best yet
                    if ( (abs(y1-y)<10 && abs(x1-x)<100 && abs(diam1-diam)<10) ) {
                        
                        k++;
                        select_keypoint.push_back(keypoints[j]);

                    }


                }

                if ( k>=3 && k>k_max ) {

                    k_max=k; //multiple same patch detections together :: when commented
                    y_max=y; 
                }

              //SECOND PASS
                if (y>120) {

                    int k1 = 1;
                    for (int j=i+1; j<len; j++) {
                        y1 = (int) keypoints[j].pt.y;
                        x1 = (int) keypoints[j].pt.x;
                        diam1 = (float) keypoints[j].size;
                        //if ( (y-5 <= y1 <= y+5) && (x-30 <= x1 <= x+30) && (diam-10 <= diam1 <= diam+10)) {
                        if ( (abs(y1-y)<20) && abs(diam1-diam)<10 && diam1>30 ) { k1++; }
                    } 

                    if (k1>=2) { k_max1 = k1; y_max1 = y;}

                                   
                }

             //THIRD PASS
                int k2=1;
                for (int j=i+1; j<len; j++) {
                    y1 = (int) keypoints[j].pt.y;
                    x1 = (int) keypoints[j].pt.x;
                    diam1 = (float) keypoints[j].size;
                    //if ( (y-5 <= y1 <= y+5) && (x-30 <= x1 <= x+30) && (diam-10 <= diam1 <= diam+10)) {
                    if ( (abs(y1-y)<20) && abs(diam1-diam)<10 ) { k2++; }
                } 

                if (k2>=2) { k_max2 = k2; y_max2 = y;}
                  
            } //outer loop
            
            
            if (k_max>=3) { 
                drawZebra(keypoints, outImg1, inImg1, 0, y_max+y_init);
            }
                 //}
            if (k_max1>=2) { 
                drawZebra(keypoints, outImg1, inImg1, 0, y_max1+y_init);
            }
                //delete select_keypoint;
/*
            if (k_max2>=2) { 
                drawZebra(keypoints, outImg1, inImg1, 0, y_max2+y_init);
            }
*/      
            z_found=1; z_yloc = y_max;
            
       

    } 
    else {
        org_im_with_keypoints = inImg1;
        z_found=0;
    }

return 1;

}




int findBlobs(Mat inImg1, Mat inImg, int width, int height, int blobchoice) {

    
    // REMOVE THE BACKGROUND
    Mat inImgCopy = inImg1;
    Mat outImg;
    
    copyMakeBorder( inImg, inImg, 20, 20, 20, 20, BORDER_CONSTANT, 0 ); //top, bottom, left, right :: order
    copyMakeBorder( inImg1, inImg1, 20, 20, 20, 20, BORDER_CONSTANT, 0 ); //top, bottom, left, right :: order

    //thresholding and enhancing image
    int left = 20;
    int top = 20;

    //bilateralFilter(inImg, outImg, 4, 3.0, 0.1, BORDER_DEFAULT );
    equalizeHist(inImg, outImg);
    threshold(outImg, outImg, 0.90*255, 255, CV_THRESH_BINARY_INV);

    Mat inImg1Copy;
    inImg1.copyTo(inImg1Copy);
    //cout << inImg1Copy.rows << inImg1Copy.cols;
    equalizeHist(inImg1Copy(Range(top+80, top+260), Range(left+0, left+340)), inImg1Copy(Range(top+80, top+260), Range(left+0, left+340)));
    threshold(inImg1Copy, inImg1Copy, 0.90*255, 255, CV_THRESH_BINARY_INV);


/*
 // LARA 
    int size = 2;
    Point anchor = Point(size, size);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * size + 1, 2 * size + 1), anchor);
    //remove background
    morphologyEx(outImg, outImg, MORPH_OPEN, element, anchor);
*/


// NALCO 
    int size = 1;
    Point anchor = Point(size, size);
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 10));

    
    dilate(inImg1Copy, inImg1Copy, element, Point(-1,-1));
    erode(inImg1Copy, inImg1Copy, element, Point(-1,-1));

    element = getStructuringElement(MORPH_RECT, Size(2 * size + 1, 2 * size + 1), anchor);
    erode(inImg1Copy, inImg1Copy, element);
    blur(inImg1Copy, inImg1Copy, Size(2,2));

// ***********************************************

    //blob detector 
    SimpleBlobDetector::Params params;

    // Set blob color
    //int blobchoice;
    params.filterByColor = 1;
    params.blobColor = blobchoice;

    // Change thresholds
    params.minThreshold = 235;
    params.maxThreshold = 255;
     
    // Filter by Area.
    params.filterByArea = true; //true;
    params.minArea = 40;
    //params.maxArea = 1600;
     
    // Filter by Circularity
    params.filterByCircularity = true; //false;
    params.minCircularity = 0.1;
     
    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.87;
     
    // Filter by Inertiafi
    params.filterByInertia = true;
    params.minInertiaRatio = 0.1; // 0.01

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    //SimpleBlobDetector detector;
    vector<KeyPoint> keypoints, keypoints1;
    detector->detect( outImg, keypoints);
    detector->detect(inImg1Copy, keypoints1);
    


    //drawKeypoints( inImg1Copy, keypoints1, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    //imshow("Blobs", im_with_keypoints);

    
    inImg1Copy.copyTo(im_with_keypoints);
    cvtColor(im_with_keypoints, im_with_keypoints, CV_GRAY2BGR);
    //outImg.copyTo(im_with_keypoints);

    inImg1.copyTo(org_im_with_keypoints);
    cvtColor(org_im_with_keypoints, org_im_with_keypoints, CV_GRAY2BGR);


    zebraDetect(keypoints, inImg1, inImg1Copy); // USE THIS :: Best
    


    zebraDetect(keypoints1, inImg1, inImg1Copy);

                

    //imshow("Detect", im_with_keypoints);

    resize(org_im_with_keypoints, org_im_with_keypoints, Size (width, height), 0, 0, INTER_LINEAR );                   

    imshow("Org Detect", org_im_with_keypoints);
    
 
    return 1;
}

/**
 * calls blob detection method
 * which calls, zebra detect method
 * which calls, draw zebra method
 * writes video frame after detection, and quits
 */
int readWriteVideo(char** argv) {

    // Images
    Mat inputImg, inputImgGray;
    Mat outputImg, outputImgGray;

    // Video
    string videoFileName = argv[1];    
    VideoCapture video;
    if( !video.open(videoFileName) )
        return 1;

    // Show video information
    int width = 0, height = 0, fps = 0, fourcc = 0; 
    width = static_cast<int>(video.get(CV_CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(video.get(CV_CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<int>(video.get(CV_CAP_PROP_FPS));
    fourcc = static_cast<int>(video.get(CV_CAP_PROP_FOURCC));

    cout << "Input video: (" << width << "x" << height << ") at " << fps << ", fourcc = " << fourcc << endl;

    //set video range
    video.set(CV_CAP_PROP_POS_MSEC, 300);

  
    //resizing to a standard size
    width = 640; height = 480;

    //writing video frames
    Size frameSize(static_cast<int>(width), static_cast<int>(height));

    //                     *************************************************************
    VideoWriter oVideoFile("/home/rrc/Desktop/opencv_projects/DetectZebra2/ZebraVid.avi", CV_FOURCC('P','I','M','1'), 20, frameSize);
    //                                change to your local directory
    //                     *************************************************************


    if (!oVideoFile.isOpened()) {
        printf("ERROR: Failed to write video file\n");
        return -1;
    }



    // Main loop
    int frameNum = 0;
    vector<int> trak_y;
    for( ; ; )
    {
        printf("FRAME #%6d ", frameNum);
        fflush(stdout);
        frameNum++;

        // Get current image        
        video >> inputImg;
        if( inputImg.empty() )
            break;

         // Color Conversion
         if(inputImg.channels() == 3)        
             cvtColor(inputImg, inputImgGray, CV_BGR2GRAY);                      
         else    
             inputImg.copyTo(inputImgGray);                  

         //resizing to 640x480
         resize(inputImg, inputImg, Size(width, height), 0, 0, INTER_LINEAR );


         // Process
         clock_t begin = clock();


         outputImg = inputImg;
     
         clock_t end = clock();
         double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
         printf("%.2f (ms)\r", 1000*elapsed_secs);


         //detection
         if(outputImg.channels() == 3)        
             cvtColor(outputImg, outputImgGray, CV_BGR2GRAY);  
         else 
             outputImg.copyTo(outputImgGray);
                          

         resize(outputImgGray, outputImgGray, Size (320, 240), 0, 0, INTER_LINEAR );
         resize(outputImg, outputImg, Size (320, 240), 0, 0, INTER_LINEAR );

        
         findBlobs(outputImgGray, outputImgGray, width, height, 0);


         oVideoFile.write(org_im_with_keypoints);       


         waitKey(1);
    }

    return 1;   
}






//end-of-code
