#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void naiveLine(Mat& img, int x0, int y0, int x1, int y1, Scalar color)
{
    float dx = x1 - x0;
    float dy = y1 - y0;

    if (abs(dx) >= abs(dy))
    {
        float slope = dy / dx;
        if (x0 > x1)
        {
            swap(x0, x1);
            swap(y0, y1);
        }
        for (int x = x0; x <= x1; x++)
        {
            int y = y0 + slope * (x - x0);
            img.at<Vec3b>(y, x) = Vec3b(color[0], color[1], color[2]);
        }
    }
    else
    {
        float inv_slope = dx / dy;
        if (y0 > y1)
        {
            swap(x0, x1);
            swap(y0, y1);
        }
        for (int y = y0; y <= y1; y++)
        {
            int x = x0 + inv_slope * (y - y0);
            img.at<Vec3b>(y, x) = Vec3b(color[0], color[1], color[2]);
        }
    }
}

void ddaLine(Mat& img, int x0, int y0, int x1, int y1, Scalar color)
{
    int dx = x1 - x0;
    int dy = y1 - y0;

    int steps = max(abs(dx), abs(dy));

    float x_inc = dx / (float)steps;
    float y_inc = dy / (float)steps;

    float x = x0;
    float y = y0;

    for (int i = 0; i <= steps; i++)
    {
        img.at<Vec3b>(round(y), round(x)) = Vec3b(color[0], color[1], color[2]);
        x += x_inc;
        y += y_inc;
    }
}

void bresenhamLine(Mat& img, int x0, int y0, int x1, int y1, Scalar color)
{
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);

    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;

    int err = dx - dy;

    while (true)
    {
        img.at<Vec3b>(y0, x0) = Vec3b(color[0], color[1], color[2]);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;

        if (e2 > -dy)
        {
            err -= dy;
            x0 += sx;
        }

        if (e2 < dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

void bresenhamCircle(Mat& img, int xc, int yc, int radius, Scalar color)
{
    int x = 0;
    int y = radius;
    int d = 3 - 2 * radius;

    while (y >= x)
    {
        img.at<Vec3b>(yc + y, xc + x) = Vec3b(color[0], color[1], color[2]);
        img.at<Vec3b>(yc + y, xc - x) = Vec3b(color[0], color[1], color[2]);
        img.at<Vec3b>(yc - y, xc + x) = Vec3b(color[0], color[1], color[2]);
        img.at<Vec3b>(yc - y, xc - x) = Vec3b(color[0], color[1], color[2]);
        img.at<Vec3b>(yc + x, xc + y) = Vec3b(color[0], color[1], color[2]);
        img.at<Vec3b>(yc + x, xc - y) = Vec3b(color[0], color[1], color[2]);
        img.at<Vec3b>(yc - x, xc + y) = Vec3b(color[0], color[1], color[2]);
        img.at<Vec3b>(yc - x, xc - y) = Vec3b(color[0], color[1], color[2]);

        x++;

        if (d > 0)
        {
            y--;
            d = d + 4 * (x - y) + 10;
        }
        else
        {
            d = d + 4 * x + 6;
        }
    }
}

int main()
{
    Mat img = Mat::zeros(500, 500, CV_8UC3);

    Scalar red(0, 0, 255);
    Scalar green(0, 255, 0);
    Scalar blue(255, 0, 0);
    Scalar white(255, 255, 255);

    int x0 = 50, y0 = 50;
    int x1 = 450, y1 = 450;

    naiveLine(img, x0, y0, x1, y1, red);

    ddaLine(img, x0, y1, x1, y0, green);

    bresenhamLine(img, x0, y0, x1, y0, blue);

    int xc = 250, yc = 250, radius = 100;
    bresenhamCircle(img, xc, yc, radius, white);

    imshow("Rasterization: ", img);
    waitKey(0);

    return 0;
}