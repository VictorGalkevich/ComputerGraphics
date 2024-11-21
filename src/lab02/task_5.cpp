#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void rankFilter(const Mat& src, Mat& dst, int ksize, int rank)
{
    dst = Mat::zeros(src.size(), src.type());

    int radius = ksize / 2;

    for (int i = radius; i < src.rows - radius; i++)
    {
        for (int j = radius; j < src.cols - radius; j++)
        {
            Mat window = src(Rect(j - radius, i - radius, ksize, ksize));

            Mat windowArray;
            window.copyTo(windowArray);
            windowArray = windowArray.reshape(0, 1);

            cv::sort(windowArray, windowArray, SORT_EVERY_ROW + SORT_ASCENDING);

            uchar value = windowArray.at<uchar>(0, rank);

            dst.at<uchar>(i, j) = value;
        }
    }
}

int main()
{
    Mat image = imread("Moon.jpg", IMREAD_GRAYSCALE);
    if (image.empty())
    {
        cout << "Could not open or find the image.\n";
        return -1;
    }

    Mat medianFiltered;
    medianBlur(image, medianFiltered, 3);

    Mat minFiltered;
    rankFilter(image, minFiltered, 3, 0);

    Mat maxFiltered;
    rankFilter(image, maxFiltered, 3, 8);

    Mat rankFiltered;
    rankFilter(image, rankFiltered, 3, 4);

    imshow("Internl image", image);
    imshow("Median filter", medianFiltered);
    imshow("Minimum filter", minFiltered);
    imshow("Maximum filter", maxFiltered);
    imshow("Rank filter (rank=4)", rankFiltered);

    waitKey(0);
    return 0;
}