/**
 * @file detectStripeBars.cpp
 * detectStripeBars.cpp detects white stripes with bars above and below them, like so
 *  ------------------------------
 *      |   |   |   |   |   |
 *      |   |   |   |   |   |
 *  ------------------------------
 * as in Nalco dataset (though is totally applicable with stripes without bars too)
 *      >This algorithm is called by explicitly passing method number = '2' as 2nd argument 
 *      >DEFAULT: Not supplying any 2nd arument invokes detectStripes.cpp (hybrid algo)



   USAGE GUIDELINE
 * Syntax: ./detectStripes <path to video> <method no>
 * <path to video> : argument one
 * <method no> : argument two
 * Example: ./detectStripes /home/rrc/Desktop/Nalco.mp4 2


 * ALGORITHM OVERVIEW:
 *      > read a video frame 
 *      > resize frame to 320x240
 *      > get homography matrix using 4 fixed points, set heuristically
 *      > warp perspective to get IPM
 *      > histogram equalization
 *      > binary global thresholding
 *      > erode and dilate with vertical structuring element
 *      > blob detection 
 *      > applying stripe detection using y-location, blob diameter, inter-blob distance in x and y, inter-blob diameter
 *      > reverse y-location using inverse homography matrix
 *      > draw yellow rectangle, write black "Zebra", and draw red circles around blobs at y-location with +-10 precision
 *      > track a true detection up to 3 previous frames before marking it, as true
 *

 * 

 */



// imports
# include <stdio.h>
# include <math.h>
# include <vector>
# include <ctime>
# include "opencv2/core/core.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/imgproc/imgproc.hpp"
# include <iostream>
# include "opencv2/calib3d/calib3d.hpp"
# include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;


/// global variables
Mat im_with_keypoints2, org_im_with_keypoints2; /// < stores original processed and output frames
int z_yloc2, d_got2; /// < temporal traking variables


/**
 * reverses the image coordinates from IPM to original mapping
 * the homoraphy matrix is inverted and multiplied to the image coordinates in homogeneous space
 * then, the coordinates are converted back to cartesian space, to pass to drawZebra()
 */
int reverseHomoMat(int x, int y, Mat h) {
    
    invert(h, h);
    
    double h00 = (double) h.at<double>(0,0);
    double h01 = (double) h.at<double>(0,1);
    double h02 = (double) h.at<double>(0,2);
    double h10 = (double) h.at<double>(1,0);
    double h11 = (double) h.at<double>(1,1);
    double h12 = (double) h.at<double>(1,2);
    double h20 = (double) h.at<double>(2,0);
    double h21 = (double) h.at<double>(2,1);
    double h22 = (double) h.at<double>(2,2); 

    double tx = ( (h00*x + h01*y + h02));
    double ty = ( (h10*x + h11*y + h12));
    double tz = ( (h20*x + h21*y + h22));

    if (tz!=0)    
        y = (int) (ty/tz);

    return y;
}


/**
 * draws rectangle around the detected y-location with +-10 precision in YELLOW, and
 * draws text "Zebra" in BLACK, and
 * draws circles around blobs responsible in the detection in RED color
 */
int drawZebra2(vector<KeyPoint> keypoints, Mat drawImg, int xpos, int ypos) {


    Point xyLoc;
    rectangle(drawImg, Point(xyLoc.x=0+20, xyLoc.y=ypos-10), Point(xyLoc.x+drawImg.cols, xyLoc.y=ypos+20-10), Scalar(0,255,255));
    putText(drawImg,"Zebra", Point(xyLoc.x=0+20, xyLoc.y=ypos+15-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1, LINE_AA);
    drawKeypoints( drawImg, keypoints, drawImg, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    return 1;
}


/**
 * detects blobs as white stripes using conditional statements or filters, e.g. diameter, inter-blob (x,y) distance 
 * applies 3 passes with various filter sizes
 * tracks a positive detection on each frame in each pass, a detection in any pass is a true detection
 * returns an integer flag as set/unset, i.e. >0 or <0 for a true/false detection, respectively
 */
int zebraDetect2(vector<KeyPoint> keypoints, Mat inImg1, Mat outImg1, Mat homoMat, Mat inp, int y_init=0) {

    //drawing zebra lines
    Point xyLoc;
    float angl, diam, diam1;
    int len, x=0, x1=0, y=0, y1=0, y_sum, y_max, y_max1, y_max2, angl_sum, k_max=0, k_max1=0, k_max2=0, y_beg, y_end;
    int z_got = -2;
    len = (int) keypoints.size(); 


    if (!keypoints.empty() && len >=3) {

                  
            for (int i=0; i<len; i++) {
                x = (int) keypoints[i].pt.x;
                y = (int) keypoints[i].pt.y;
                diam = (float) keypoints[i].size;
                int k=0;

                vector<KeyPoint> select_keypoint;
                select_keypoint.push_back(keypoints[i]);


                for (int j=i+1; j<len; j++) {
                    y1 = (int) keypoints[j].pt.y;
                    x1 = (int) keypoints[j].pt.x;
                    diam1 = (float) keypoints[j].size;
                    
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
                if (y>0) {

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
                int yy = reverseHomoMat(0, y_max+y_init-20, homoMat);
                //cout<<"y_max "<< y_max <<" "<< endl;
                drawZebra2(keypoints, org_im_with_keypoints2, 0, yy+20);
                drawZebra2(keypoints, im_with_keypoints2, 0, y_max+y_init);
                z_got = 2;

            }
                 //}
            if (k_max1>=2) { 
                int yy = reverseHomoMat(0, y_max1+y_init-20, homoMat);
                drawZebra2(keypoints, org_im_with_keypoints2, 0, yy+20);
                drawZebra2(keypoints, im_with_keypoints2, 0, y_max1+y_init);
                z_got = 2;
            }
                //delete select_keypoint;
/*
            if (k_max2>=2) { 
                int yy = reverseHomoMat(0, y_max2+y_init-20, homoMat);
                drawZebra2(keypoints, org_im_with_keypoints2, 0, yy+20);
                drawZebra2(keypoints, im_with_keypoints2, 0, y_max2+y_init);
                z_got = 2;
            }
*/          
            z_yloc2 = y_max;
            
              
    } else {
        org_im_with_keypoints2 = inp; //inImg1;

    }

return z_got;

}


int findBlobs2(Mat inImg1, Mat inImg, int width, int height, int blobchoice, Mat homoMat, Mat inputImage) {

    
    // REMOVE THE BACKGROUND
    Mat inImgCopy = inImg1;
    Mat outImg;
    
    copyMakeBorder( inImg, inImg, 20, 20, 20, 20, BORDER_CONSTANT, 0 ); //top, bottom, left, right :: order
    copyMakeBorder( inImg1, inImg1, 20, 20, 20, 20, BORDER_CONSTANT, 0 ); //top, bottom, left, right :: order
    copyMakeBorder( inputImage, inputImage, 20, 20, 20, 20, BORDER_CONSTANT, 0 ); //top, bottom, left, right :: order

    //thresholding and enhancing image
    int left = 20;
    int top = 20;

    //bilateralFilter(inImg, outImg, 4, 3.0, 0.1, BORDER_DEFAULT );
    equalizeHist(inImg, outImg);
    threshold(outImg, outImg, 0.90*255, 255, CV_THRESH_BINARY_INV);

    Mat inImg1Copy;
    inImg1.copyTo(inImg1Copy);

    equalizeHist(inImg1Copy(Range(top+0, top+260), Range(left+0, left+340)), inImg1Copy(Range(top+0, top+260), Range(left+0, left+340)));
    threshold(inImg1Copy, inImg1Copy, 0.90*255, 255, CV_THRESH_BINARY_INV);

    //imshow("Thresh", outImg);


/*
    // LARA : STRIPES WITHOUT BARS
    int size = 2;
    Point anchor = Point(size, size);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * size + 1, 2 * size + 1), anchor);
    //remove background
    morphologyEx(outImg, outImg, MORPH_OPEN, element, anchor);
*/


//********************************************
    // NALCO : STRIPES WITH BARS
    int size = 1;
    Point anchor = Point(size, size);
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 10));  
    
    dilate(inImg1Copy, inImg1Copy, element, Point(-1,-1));
    erode(inImg1Copy, inImg1Copy, element, Point(-1,-1));

    element = getStructuringElement(MORPH_RECT, Size(2 * size + 1, 2 * size + 1), anchor);
    erode(inImg1Copy, inImg1Copy, element);
    blur(inImg1Copy, inImg1Copy, Size(2,2));

//*********************************************

    //blob detector 
    SimpleBlobDetector::Params params;

    // Set blob color
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
    
    
    inImg1Copy.copyTo(im_with_keypoints2);
    cvtColor(im_with_keypoints2, im_with_keypoints2, CV_GRAY2BGR);
    //outImg.copyTo(im_with_keypoints2);

    //inImg1.copyTo(org_im_with_keypoints2);
    inputImage.copyTo(org_im_with_keypoints2);
    cvtColor(org_im_with_keypoints2, org_im_with_keypoints2, CV_GRAY2BGR);

    //drawKeypoints( outImg, keypoints, im_with_keypoints2, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


    // invoke detection algo
    d_got2 = zebraDetect2(keypoints, inImg1, inImg1Copy, homoMat, inputImage); // USE THIS :: Best (ONE PASS)
    
    d_got2 = zebraDetect2(keypoints1, inImg1, inImg1Copy, homoMat, inputImage);

    imshow("Detect", im_with_keypoints2); // OPTIONAL 
    //process viewing purpose only

    //resize(org_im_with_keypoints2, org_im_with_keypoints2, Size (width, height), 0, 0, INTER_LINEAR );   // OPTIONAL : (width, height) = (640, 480)
    //inal output enlarged viewing purpose only                

    imshow("Org Detect", org_im_with_keypoints2);
    
 
    return 1;
}

/*
 * calls blob detection method
 * which calls, zebra detect method
 * which calls, draw zebra method
 * writes video frame after detection, and quits
*/
int readWriteVideo2(char** argv) {

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
    width = 320; height = 240;

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


    // The 4-points at the input image  
    vector<Point2f> origPoints;

    origPoints.push_back( Point2f(-abs(2*width/8), abs(height)) );
    origPoints.push_back( Point2f(abs(10*width/8), abs(height)) );
    origPoints.push_back( Point2f(abs(5*width/8), abs(4*height/8)) );
    origPoints.push_back( Point2f(abs(2*width/8), abs(4*height/8)) );

    // The 4-points correspondences in the destination image
    vector<Point2f> dstPoints;
    dstPoints.push_back( Point2f(0, height) );
    dstPoints.push_back( Point2f(width, height) );
    dstPoints.push_back( Point2f(width, 0) );
    dstPoints.push_back( Point2f(0, 0) );
        

    // Main loop
    int frameNum = 0;
    vector<int> trak_y(1);

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


         Mat h = findHomography(origPoints, dstPoints);
         
         //cout<<h << endl;
         //Mat h = getPerspectiveTransform(origPoints, dstPoints); // ALTERNATIVE : h
         
         warpPerspective(inputImg, outputImg, h, Size(width, height));

         //outputImg = inputImg; // OPTIONAL : TO REMOVE IPM, uncomment it, comment above two lines

       
         clock_t end = clock();
         double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
         printf("%.2f (ms)\r", 1000*elapsed_secs);


         if(outputImg.channels() == 3)        
             cvtColor(outputImg, outputImgGray, CV_BGR2GRAY);  
         else 
             outputImg.copyTo(outputImgGray);

         
                           

         // resize input frames
         resize(outputImgGray, outputImgGray, Size (320, 240), 0, 0, INTER_LINEAR );
         resize(outputImg, outputImg, Size (320, 240), 0, 0, INTER_LINEAR );
         resize(inputImgGray, inputImgGray, Size (320, 240), 0, 0, INTER_LINEAR );



        findBlobs2(outputImgGray, outputImgGray, width, height, 0, h, inputImgGray);
         
        
        //oVideoFile.write(org_im_with_keypoints2); // OPTIONAL: uncomment to remove temporal tracking, comment the lines below



        //temporal trak
        trak_y.push_back(int(d_got2));
        cout<< "trak_y " << trak_y[frameNum] << endl;


        if ( frameNum >= 3) {
            //if ( z_yloc-10 < (trak_y[frameNum-1]+trak_y[frameNum-2]+trak_y[frameNum-3])/3 < z_yloc+10 ) 
            if ( trak_y[frameNum-1] + trak_y[frameNum-2] + trak_y[frameNum-3] > 0 ) 
                oVideoFile.write(org_im_with_keypoints2);
            else {
                copyMakeBorder( inputImgGray, inputImgGray, 20, 20, 20, 20, BORDER_CONSTANT, 0 ); //top, bottom, left, right :: order
                resize(inputImgGray, inputImgGray, Size (320, 240), 0, 0, INTER_LINEAR );
                cvtColor(inputImgGray, inputImgGray, CV_GRAY2BGR);
                oVideoFile.write(inputImgGray);
            }
        } 




         waitKey(1);
    }

    return 1;   
}






//end-of-code
