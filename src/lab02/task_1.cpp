#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to calculate and display histogram
void calcAndShowHistogram(const Mat& image, const string& name)
{
    // Separate the image in B, G, and R planes
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    // Establish the number of bins
    int histSize = 256;

    // Set the ranges (0 to 256) for B, G, and R
    float range[] = {0, 256};
    const float* histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    // Compute the histograms for B, G, and R
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // Create an image to display the histogram
    int hist_w = 512; 
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // Normalize the result to fit the histogram image height
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX);

    // Draw the histograms for B, G, and R
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, 
             Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
             Scalar(255, 0, 0), 2);

        line(histImage, 
             Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
             Scalar(0, 255, 0), 2);

        line(histImage, 
             Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
             Scalar(0, 0, 255), 2);
    }

    // Display the histogram
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, histImage);
}

int main()
{
    // Load the image
    Mat image = imread("input.jpg");

    // Check if the image is loaded successfully
    if (image.empty())
    {
        cout << "Could not open or find the image.\n";
        return -1;
    }

    // Display the original image
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    // Calculate and display the histogram of the original image
    calcAndShowHistogram(image, "Histogram of Original Image");

    // Convert the image to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Apply histogram equalization
    Mat equalizedImage;
    equalizeHist(gray, equalizedImage);

    // Display the equalized image
    namedWindow("Equalized Image", WINDOW_AUTOSIZE);
    imshow("Equalized Image", equalizedImage);

    // Calculate and display the histogram of the equalized image
    calcAndShowHistogram(equalizedImage, "Histogram of Equalized Image");

    // Perform linear contrast stretching
    Mat stretchedImage;
    gray.convertTo(stretchedImage, -1, 1.5, 0); // Adjust the alpha value for contrast

    // Display the contrast-stretched image
    namedWindow("Contrast Stretched Image", WINDOW_AUTOSIZE);
    imshow("Contrast Stretched Image", stretchedImage);

    // Calculate and display the histogram of the stretched image
    calcAndShowHistogram(stretchedImage, "Histogram of Contrast Stretched Image");

    waitKey(0);
    return 0;
}
