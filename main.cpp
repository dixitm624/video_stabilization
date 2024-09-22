    /*
    Copyright (c) 2014, Nghia Ho
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    Modified by Dixit Mudakavi 
    */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

const int STABILIZATION_RADIUS = 100; // Frames for smoothing; larger values leads to more stability

struct TransformationParameters
{
    TransformationParameters() {}
    TransformationParameters(double offsetX, double offsetY, double angle) 
    {
        dx = offsetX;
        dy = offsetY;
        da = angle;
    }

    double dx; // Horizontal displacement
    double dy; // Vertical displacement
    double da; // Rotation angle

    void applyTransformation(Mat &T)
    {
        // Construct transformation matrix based on new parameters
        T.at<double>(0, 0) = cos(da);
        T.at<double>(0, 1) = -sin(da);
        T.at<double>(1, 0) = sin(da);
        T.at<double>(1, 1) = cos(da);

        T.at<double>(0, 2) = dx;
        T.at<double>(1, 2) = dy;
    }
};

struct Path
{
    Path() {}
    Path(double positionX, double positionY, double angle) {
        x = positionX;
        y = positionY;
        a = angle;
    }

    double x; // X position
    double y; // Y position
    double a; // Angle
};

vector<Path> accumulateTransformations(vector<TransformationParameters> &transforms)
{
    vector<Path> trajectory; // Path at all frames
    double accumulatedX = 0;
    double accumulatedY = 0;
    double accumulatedA = 0;

    for (const auto &transform : transforms) 
    {
        accumulatedX += transform.dx;
        accumulatedY += transform.dy;
        accumulatedA += transform.da;

        trajectory.emplace_back(accumulatedX, accumulatedY, accumulatedA);
    }

    return trajectory; 
}

vector<Path> smoothPath(vector<Path>& trajectory, int radius)
{
    vector<Path> smoothedPath;  
    for (size_t i = 0; i < trajectory.size(); i++) {
        double sumX = 0;
        double sumY = 0;
        double sumA = 0;
        int count = 0;

        for (int j = -radius; j <= radius; j++) {
            if (i + j >= 0 && i + j < trajectory.size()) {
                sumX += trajectory[i + j].x;
                sumY += trajectory[i + j].y;
                sumA += trajectory[i + j].a;

                count++;
            }
        }

        smoothedPath.emplace_back(sumX / count, sumY / count, sumA / count);
    }

    return smoothedPath; 
}

void adjustBorder(Mat &stableFrame)
{
    Mat rotationMatrix = getRotationMatrix2D(Point2f(stableFrame.cols / 2, stableFrame.rows / 2), 0, 1.04); 
    warpAffine(stableFrame, stableFrame, rotationMatrix, stableFrame.size()); 
}

int main(int argc, char **argv)
{
    // Open the webcam
    VideoCapture camera(0);

    if (!camera.isOpened()) {
        cerr << "Error: Unable to access the webcam" << endl;
        return -1;
    }

    // Get dimensions and FPS of the frame
    int width = static_cast<int>(camera.get(CAP_PROP_FRAME_WIDTH)); 
    int height = static_cast<int>(camera.get(CAP_PROP_FRAME_HEIGHT));
    double framesPerSecond = camera.get(cv::CAP_PROP_FPS);
    if (framesPerSecond <= 0) framesPerSecond = 10;
    int totalFrames = static_cast<int>(framesPerSecond * 10);
  
    cout << "Setting up VideoWriter..." << endl;
    VideoWriter videoOutput("output.avi", VideoWriter::fourcc('H', '2', '6', '4'), framesPerSecond, Size(640, 480));
    if (!videoOutput.isOpened()) {
        cerr << "Error: Unable to initialize video writer." << endl;
        return -1;
    }
    cout << "VideoWriter is ready." << endl;

    // Variables for frame storage
    Mat currentFrame, currentGray;
    Mat previousFrame, previousGray;

    // Capture the first frame
    camera >> previousFrame;

    // Convert to grayscale
    cvtColor(previousFrame, previousGray, COLOR_BGR2GRAY);

    vector<TransformationParameters> transformations; 
    Mat lastTransformationMatrix;

    for (int i = 1; i < totalFrames - 1; i++)
    {
        vector<Point2f> previousPoints, currentPoints;

        // Detect features in the previous frame
        goodFeaturesToTrack(previousGray, previousPoints, 200, 0.01, 30);

        // Capture the current frame
        bool readSuccess = camera.read(currentFrame);
        if (!readSuccess || currentFrame.empty()) {
            cerr << "Error: Unable to read the current frame." << endl;
            break; 
        }

        // Convert current frame to grayscale
        cvtColor(currentFrame, currentGray, COLOR_BGR2GRAY);

        // Check for detected points
        if (previousPoints.empty()) {
            cerr << "Error: No features available to track!" << endl;
            return -1; 
        }

        // Ensure frame sizes match
        if (previousGray.size() != currentGray.size()) {
            cerr << "Error: Frame dimensions do not match!" << endl;
            return -1; 
        }
        
        // Compute optical flow
        vector<uchar> status;
        vector<float> error;
        calcOpticalFlowPyrLK(previousGray, currentGray, previousPoints, currentPoints, status, error);

        auto prevIt = previousPoints.begin(); 
        auto currIt = currentPoints.begin(); 
        for (size_t k = 0; k < status.size(); k++) 
        {
            if (status[k]) 
            {
                prevIt++; 
                currIt++; 
            }
            else 
            {
                prevIt = previousPoints.erase(prevIt);
                currIt = currentPoints.erase(currIt);
            }
        }

        // Calculate the transformation matrix
        Mat transformationMatrix = estimateRigidTransform(previousPoints, currentPoints, false); 

        // Use the last known good transformation if none is found
        if (transformationMatrix.empty()) lastTransformationMatrix.copyTo(transformationMatrix);
        transformationMatrix.copyTo(lastTransformationMatrix);

        // Extract translation and rotation
        double translationX = transformationMatrix.at<double>(0, 2);
        double translationY = transformationMatrix.at<double>(1, 2);
        double rotationAngle = atan2(transformationMatrix.at<double>(1, 0), transformationMatrix.at<double>(0, 0));

        // Store the transformation
        transformations.emplace_back(translationX, translationY, rotationAngle);

        // Move to the next frame
        currentGray.copyTo(previousGray);

        // cout << "Frame: " << i << "/" << totalFrames << " - Tracked points: " << previousPoints.size() << endl;
    }

    // Compute cumulative trajectory
    vector<Path> trajectory = accumulateTransformations(transformations);

    // Smooth the trajectory
    vector<Path> smoothedPath = smoothPath(trajectory, STABILIZATION_RADIUS); 

    vector<TransformationParameters> smoothedTransforms;
  
    for (size_t i = 0; i < transformations.size(); i++)
    {
        double deltaX = smoothedPath[i].x - trajectory[i].x;
        double deltaY = smoothedPath[i].y - trajectory[i].y;
        double deltaA = smoothedPath[i].a - trajectory[i].a;

        double newDx = transformations[i].dx + deltaX;
        double newDy = transformations[i].dy + deltaY;
        double newDa = transformations[i].da + deltaA;

        smoothedTransforms.emplace_back(newDx, newDy, newDa);
    }

    camera.set(cv::CAP_PROP_POS_FRAMES, 0);
    Mat transformMatrix(2, 3, CV_64F);
    Mat frame, stabilizedFrame, outputFrame; 

    for (int i = 0; i < totalFrames - 1; i++) 
    { 
        bool readSuccess = camera.read(frame);
        if (!readSuccess) 
        {
            cerr << "Error: Unable to read frame " << i << "!" << endl;
            break;
        }

        // Get the current transformation
        smoothedTransforms[i].applyTransformation(transformMatrix); 

        // Apply the transformation
        warpAffine(frame, stabilizedFrame, transformMatrix, frame.size());

        // Adjust the border
        adjustBorder(stabilizedFrame); 

        // Combine original and stabilized frames
        hconcat(frame, stabilizedFrame, outputFrame);

        // Resize if too large
        if (outputFrame.cols > 1920) 
        {
            resize(outputFrame, outputFrame, Size(outputFrame.cols / 2, outputFrame.rows / 2));
        }

        imshow("Before and After", outputFrame);
        videoOutput.write(outputFrame);
        waitKey(10);
    }

    // Release resources
    camera.release();
    videoOutput.release();
    destroyAllWindows();

    return 0;
}
