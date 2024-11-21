#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void calcAndShowHistogram(const Mat& image, const string& name)
{
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    int histSize = 256;

    float range[] = {0, 256};
    const float* histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX);

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

    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, histImage);
}

int main()
{
    Mat image = imread("Moon.jpg");

    if (image.empty())
    {
        cout << "Could not open or find the image.\n";
        return -1;
    }

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    calcAndShowHistogram(image, "Histogram of Original Image");

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Mat equalizedImage;
    equalizeHist(gray, equalizedImage);

    namedWindow("Equalized Image", WINDOW_AUTOSIZE);
    imshow("Equalized Image", equalizedImage);

    calcAndShowHistogram(equalizedImage, "Histogram of Equalized Image");

    Mat stretchedImage;
    gray.convertTo(stretchedImage, -1, 1.5, 0);

    namedWindow("Contrast Stretched Image", WINDOW_AUTOSIZE);
    imshow("Contrast Stretched Image", stretchedImage);

    calcAndShowHistogram(stretchedImage, "Histogram of Contrast Stretched Image");

    waitKey(0);
    return 0;
}