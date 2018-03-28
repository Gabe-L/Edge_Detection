//#include "stdafx.h"//header for Visual Studio
#include <opencv2/core/core.hpp>//header for OpenCV core
#include <opencv2/highgui/highgui.hpp>//header for OpenCV UI
#include <iostream>//header for c++ IO
#include <chrono>

//set opencv and c++ namespaces
using namespace cv;
using namespace std;

//import timing information
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::cout;
using std::endl;

typedef std::chrono::steady_clock the_clock;

int kernelX[3][3] = {
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

int kernelY[3][3] = {
	-1, -2, -1,
	0, 0, 0,
	1, 2, 1
};

int kernel_blur[3][3] = {
	1, 2, 1,
	2, 4, 2,
	1, 2, 1
};

void grey_filter(Mat* img) {
	for (int x = 0; x < img->rows; x++) {
		for (int y = 0; y < img->cols; y++) {

			int total = 0;
			total += img->at<Vec3b>(x, y)[0];
			total += img->at<Vec3b>(x, y)[1];
			total += img->at<Vec3b>(x, y)[2];

			img->at<Vec3b>(x, y)[0] = (int)(total / 3);
			img->at<Vec3b>(x, y)[1] = (int)(total / 3);
			img->at<Vec3b>(x, y)[2] = (int)(total / 3);
		}
	}
}

void blur(Mat* img) {
	
	Mat edges;

	edges.create(img->rows, img->cols, 0);

	for (int x = 1; x < img->rows - 1; x++) {
		for (int y = 1; y < img->cols - 1; y++) {

			//construct matrix from surrounding pixels
			float intensity = 0;

			uchar pixel_matrix[3][3] = {
				img->at<Vec3b>(x - 1,y - 1)[0],img->at<Vec3b>(x + 0,y - 1)[0],img->at<Vec3b>(x + 1,y - 1)[0],
				img->at<Vec3b>(x - 1,y + 0)[0],img->at<Vec3b>(x + 0,y + 0)[0],img->at<Vec3b>(x + 1,y + 0)[0],
				img->at<Vec3b>(x - 1,y + 1)[0],img->at<Vec3b>(x + 0,y + 1)[0],img->at<Vec3b>(x + 1,y + 1)[0]
			};

			for (int k_x = 0; k_x < 3; k_x++) {
				for (int k_y = 0; k_y < 3; k_y++) {
					intensity += pixel_matrix[k_x][k_y] * kernel_blur[k_x][k_y];
				}
			}

			intensity /= 16;

			edges.at<uchar>(x, y) = (int)intensity;

		}
	}

	*img = edges;

	return;
}

void edge_filter(Mat* img) {


	vector<vector<float>> mags;
	mags.resize(img->rows);

	for (int i = 0; i < img->rows; i++) {
		mags[i].resize(img->cols);
	}

	for (int x = 1; x < img->rows - 1; x++) {
		for (int y = 1; y < img->cols - 1; y++) {

			//construct matrix from surrounding pixels
			float x_total = 0; float y_total = 0;

			float pixel_matrix[3][3] = {
				img->at<Vec3b>(x - 1,y - 1)[0],img->at<Vec3b>(x + 0,y - 1)[0],img->at<Vec3b>(x + 1,y - 1)[0],
				img->at<Vec3b>(x - 1,y + 0)[0],img->at<Vec3b>(x + 0,y + 0)[0],img->at<Vec3b>(x + 1,y + 0)[0],
				img->at<Vec3b>(x - 1,y + 1)[0],img->at<Vec3b>(x + 0,y + 1)[0],img->at<Vec3b>(x + 1,y + 1)[0]
			};

			for (int k_x = 0; k_x < 3; k_x++) {
				for (int k_y = 0; k_y < 3; k_y++) {
					x_total += pixel_matrix[k_x][k_y] * kernelX[k_x][k_y];
					y_total += pixel_matrix[k_x][k_y] * kernelY[k_x][k_y];
				}
			}

			float temp = sqrt((x_total * x_total) + (y_total * y_total));

			temp /= 1141;

			temp *= 255;

			mags[x][y] = temp;
		}
	}

	for (int x = 1; x < img->rows - 1; x++) {
		for (int y = 1; y < img->cols - 1; y++) {

			img->at<Vec3b>(x, y)[0] = mags[x][y];
			img->at<Vec3b>(x, y)[1] = mags[x][y];
			img->at<Vec3b>(x, y)[2] = mags[x][y];
		}
	}

	return;

}

void edge_filter_v2(Mat* img) {

	Mat edges;

	edges.create(img->rows, img->cols, 0);

	for (int x = 1; x < img->rows - 1; x++) {
		for (int y = 1; y < img->cols - 1; y++) {

			//construct matrix from surrounding pixels
			float x_total = 0; float y_total = 0;

			uchar pixel_matrix[3][3] = {
				img->at<Vec3b>(x - 1,y - 1)[0],img->at<Vec3b>(x + 0,y - 1)[0],img->at<Vec3b>(x + 1,y - 1)[0],
				img->at<Vec3b>(x - 1,y + 0)[0],img->at<Vec3b>(x + 0,y + 0)[0],img->at<Vec3b>(x + 1,y + 0)[0],
				img->at<Vec3b>(x - 1,y + 1)[0],img->at<Vec3b>(x + 0,y + 1)[0],img->at<Vec3b>(x + 1,y + 1)[0]
			};

			for (int k_x = 0; k_x < 3; k_x++) {
				for (int k_y = 0; k_y < 3; k_y++) {
					x_total += pixel_matrix[k_x][k_y] * kernelX[k_x][k_y];
					y_total += pixel_matrix[k_x][k_y] * kernelY[k_x][k_y];
				}
			}

			float intensity = sqrt((x_total * x_total) + (y_total * y_total));

			intensity /= 1141; //makes temp proportional to rough max value of convolution
			intensity *= 255; //converts proportion to intensity value

			edges.at<uchar>(x, y) = (int)intensity;

		}
	}

	*img = edges;

	return;

}

int main()
{
	Mat image;//create mat variable for our image
	image = imread("test2.jpg", 1); // read image.  1 : read as rgb, 0 : read as grayscale image

	if (!image.data) // check whether the image is found or not
	{
		cout << "Image is not found. Please write the file name and path location correctly." << endl;
	}

	grey_filter(&image);

	//blur(&image);

	the_clock::time_point time_start = the_clock::now();

	edge_filter_v2(&image);

	the_clock::time_point time_end = the_clock::now();

	auto time_taken = duration_cast<milliseconds>(time_end - time_start).count();

	cout << "Edge filter time taken: " << time_taken << "ms." << endl;

	namedWindow("Edges", WINDOW_AUTOSIZE);// create window for showing our image
	imshow("Edges", image);// showing our image
	waitKey(0);// imshow will show image once the program hit this "waitKey".
	return 0;
}