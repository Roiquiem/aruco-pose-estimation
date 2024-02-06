#include <iostream>
#include <opencv2/opencv.hpp>
#include <map>
#include <string>

// for cameraDatabase.json
#include <nlohmann/json.hpp>
#include <fstream>


// defining variables
const std::string cameraName  = "2K GPlus";
const int maxMarkers = 6 * 3;
const int markerRightTop = 68;
const int markerRightBottom = 60;
const int markerLeftTop = 58;
const int markerLeftBottom = 43;
const int markerCenterOuter = 93;
const int markerCenterInner = 76;

const float markerLengthOuter = 0.1895;
const float markerLengthInner = 0.0210;
const float markerLengthCorner = 0.0150;
const float xDistanceCenterCornerToCenter = 0.06354;
const float yDistanceCenterCornerToCenter = 0.06354;

// Drawing settings
const bool drawCornerAxes = true;
const bool drawCenterInnerAxes = false;
const bool drawCenterOuterAxes = false;

// defining Kalman variables
cv::KalmanFilter KF;
int nstates = 18;
int nMeasurements = 6;
int nInputs = 0;
double dt = 0.125;



// creating cameraMatrix, distCoefficients, rvecs, tvecs, objPoints
struct MarkerParameters {
    cv::Mat cameraMatrix;
    cv::Mat distCoefficients;
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
    cv::Mat objPoints;
};

// creating path for cameraDatabase
const std::string jsonFilePath = "/home/pascal/cmain/cameraCalibration/camerasDatabase.json";

// function for getting the camera parameters from the external cameraDatabase file
void initializeCameraParameters(const std::string& cameraName, MarkerParameters& markerParams) {
   // read the JSON file
   std::ifstream ifs(jsonFilePath);
   if (!ifs.is_open()) {
      std::cerr << "Error: Unable to open JSON file." << std::endl;
      return;
   }

   // parse the JSON content
   nlohmann::json jsonContent;
   ifs >> jsonContent;

   // find the camera parameters based on the cameraName
   auto cameras = jsonContent["cameras"];
   for (const auto& cameraParams : cameras) {
      if (cameraParams["name"] == cameraName) {
         // extract camera matrix and distortion coefficients
         cv::Mat newCameraMatrix = (cv::Mat_<double>(3,3) <<
            cameraParams["cameraMatrix"][0][0], cameraParams["cameraMatrix"][0][1], cameraParams["cameraMatrix"][0][2],
            cameraParams["cameraMatrix"][1][0], cameraParams["cameraMatrix"][1][1], cameraParams["cameraMatrix"][1][2],
            cameraParams["cameraMatrix"][2][0], cameraParams["cameraMatrix"][2][1], cameraParams["cameraMatrix"][2][2]);

         cv::Mat newDistCoefficients = (cv::Mat_<double>(5, 1) <<
                cameraParams["distCoefficients"][0], cameraParams["distCoefficients"][1],
                cameraParams["distCoefficients"][2], cameraParams["distCoefficients"][3],
                cameraParams["distCoefficients"][4]);  

         // assign the new camera parameters
         markerParams.cameraMatrix = newCameraMatrix;
         markerParams.distCoefficients = newDistCoefficients;
         
         std::cout << "Camera parameters loaded for: " << cameraName << std::endl;
         return;
      } 
   }
   std::cerr << "Error: Camera parameters not found for this camera: " << cameraName << std::endl;
}

// Function to initialize object points based on marker length
void initializeObjectPoints(cv::Mat& objPoints, float markerLength) {
    objPoints = (cv::Mat_<float>(4, 3) <<
        markerLength / 2.f, -markerLength / 2.f, 0,
        -markerLength / 2.f, -markerLength / 2.f, 0,
        -markerLength / 2.f, markerLength / 2.f, 0,
        markerLength / 2.f, markerLength / 2.f, 0);
}

// Function to calculate offset vector based on marker ID
cv::Vec3d calculateOffsetVector(int markerId) {
    switch (markerId) {
        case markerRightTop:
            return cv::Vec3d(xDistanceCenterCornerToCenter, yDistanceCenterCornerToCenter, 0);
        case markerRightBottom:
            return cv::Vec3d(xDistanceCenterCornerToCenter, -yDistanceCenterCornerToCenter, 0);
        case markerLeftTop:
            return cv::Vec3d(-xDistanceCenterCornerToCenter, yDistanceCenterCornerToCenter, 0);
        case markerLeftBottom:
            return cv::Vec3d(-xDistanceCenterCornerToCenter, -yDistanceCenterCornerToCenter, 0);
        default:
            return cv::Vec3d(0, 0, 0);
    }
}

// Function to calculate rotated offset vector
cv::Vec3d calculateRotatedOffset(const cv::Vec3d& offset, const cv::Matx33d& rotationMatrix) {
    return rotationMatrix.t() * offset;
}

void processMarkers(const std::vector<int>& ids, const std::vector<std::vector<cv::Point2f>>& corners, const cv::aruco::ArucoDetector& detector, cv::VideoCapture& cap, cv::Mat& imgcpy, MarkerParameters& markerParams) {
    for (size_t i = 0; i < ids.size(); ++i) {
        int currentId = ids[i];
        float markerLength = 0.0;

        if (currentId == markerRightTop || currentId == markerRightBottom || currentId == markerLeftTop || currentId == markerLeftBottom) {
            markerLength = markerLengthCorner;
            initializeObjectPoints(markerParams.objPoints, markerLength);

            if (corners[i].size() == 4) {
               cv::Vec3d offset = calculateOffsetVector(currentId);
               cv::Vec3d offset_rotated;
               
            // for debug
            //    std::cout << "objPoints size: " << objPoints.size() << std::endl;
            // std::cout << "corners[" << i << "] size: " << corners[i].size() << std::endl;

               cv::solvePnP(markerParams.objPoints, corners[i], markerParams.cameraMatrix, markerParams.distCoefficients, markerParams.rvecs[i], markerParams.tvecs[i]);

               // check if solvePnP was successful
               if (!cv::checkRange(markerParams.tvecs[i]) && !cv::checkRange(markerParams.rvecs[i])) {
                     std::cerr << "Error: solvePnP failed for ID " << currentId << std::endl;
                   continue;
               } else if (!cv::checkRange(markerParams.tvecs[i]) || !cv::checkRange(markerParams.rvecs[i])) {
                  std::cerr << "Error: Insufficient corners for ID " << currentId << std::endl;
                   continue;
               }

               // calculate offset vector in marker coordinate system
                cv::Matx33d rotationMatrix;
                cv::Rodrigues(markerParams.rvecs[i], rotationMatrix);
                offset_rotated = calculateRotatedOffset(offset, rotationMatrix);

                // Add the offset vector to the translation vector
                markerParams.tvecs[i] += offset_rotated;

                std::cout << "ID: " << currentId << " " << markerParams.tvecs[i] << std::endl;
            } else {
               std::cerr << "Error: Insufficient corners for ID " << currentId << std::endl;
            }
        } else if (currentId == markerCenterInner) {
            markerLength = markerLengthInner;
            initializeObjectPoints(markerParams.objPoints, markerLength);
            cv::solvePnP(markerParams.objPoints, corners[i], markerParams.cameraMatrix, markerParams.distCoefficients, markerParams.rvecs[i], markerParams.tvecs[i]);

            // check if solvePnP was successful
            if (!cv::checkRange(markerParams.tvecs[i]) && !cv::checkRange(markerParams.rvecs[i])) {
                    std::cerr << "Error: solvePnP failed for ID " << currentId << std::endl;
                continue;
            } else if (!cv::checkRange(markerParams.tvecs[i]) || !cv::checkRange(markerParams.rvecs[i])) {
                std::cerr << "Error: Insufficient corners for ID " << currentId << std::endl;
                continue;
            }

            std::cout << "ID: " << currentId << " " << markerParams.tvecs[i] << std::endl;
        } else if (currentId == markerCenterOuter) {
            markerLength = markerLengthOuter;
            initializeObjectPoints(markerParams.objPoints, markerLength);
            cv::solvePnP(markerParams.objPoints, corners[i], markerParams.cameraMatrix, markerParams.distCoefficients, markerParams.rvecs[i], markerParams.tvecs[i]);
            
            // check if solvePnP was successful
            if (!cv::checkRange(markerParams.tvecs[i]) && !cv::checkRange(markerParams.rvecs[i])) {
                    std::cerr << "Error: solvePnP failed for ID " << currentId << std::endl;
                continue;
            } else if (!cv::checkRange(markerParams.tvecs[i]) || !cv::checkRange(markerParams.rvecs[i])) {
                std::cerr << "Error: Insufficient corners for ID " << currentId << std::endl;
                continue;
            }

            std::cout << "ID: " << currentId << " " << markerParams.tvecs[i] << std::endl;
        } else {
           std::cerr << "Error: Unexpected marker ID " << currentId << std::endl;
        }

        // draw axes for each marker
        if (currentId == markerRightTop || currentId == markerRightBottom || currentId == markerLeftTop || currentId == markerLeftBottom && drawCornerAxes) {
            cv::drawFrameAxes(imgcpy, markerParams.cameraMatrix, markerParams.distCoefficients, markerParams.rvecs[i], markerParams.tvecs[i], 0.05);
        } else if (currentId == markerCenterInner && drawCenterInnerAxes) {
            cv::drawFrameAxes(imgcpy, markerParams.cameraMatrix, markerParams.distCoefficients, markerParams.rvecs[i], markerParams.tvecs[i], 0.070);
        } else if (currentId == markerCenterOuter && drawCenterOuterAxes) {
            cv::drawFrameAxes(imgcpy, markerParams.cameraMatrix, markerParams.distCoefficients, markerParams.rvecs[i], markerParams.tvecs[i], 0.070);
        }
    }
}


int main() {
    MarkerParameters markerParams;

    initializeCameraParameters(cameraName, markerParams);

    float markerLength = 0.0;

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_100);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    cv::VideoCapture cap;
    cap.open(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera." << std::endl;
        return -1;
    }

    // initiallize rvecs and tvecs vectors assuming a maximum of 6 markers
    markerParams.tvecs.resize(maxMarkers);
    markerParams.rvecs.resize(maxMarkers);

    while (true) {
        cv::Mat img;
        if (!cap.read(img)) {
            std::cerr << "Error reading frame." << std::endl;
            break;
        }

        if (img.empty()) {
            std::cerr << "Error: Empty frame" << std::endl;
            break;
        }

        cv::Mat imgcpy;
        img.copyTo(imgcpy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        detector.detectMarkers(img, corners, ids, rejected);

        if (!ids.empty()) {
            processMarkers(ids, corners, detector, cap, imgcpy, markerParams);
            // debug:
            // cv::aruco::drawDetectedMarkers(imgcpy, corners, ids);
        }

        cv::imshow("output", imgcpy);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
