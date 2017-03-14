#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

Mat float2byte(const Mat& If)
{
    double minVal, maxVal;
    minMaxLoc(If,&minVal,&maxVal);
    Mat Ib;
    If.convertTo(Ib, CV_8U, 255.0/(maxVal - minVal),-minVal * 255.0/(maxVal - minVal));
    return Ib;
}

struct Data {
 Mat Ix, Iy, G;
};

void onTrackbar(int sigma, void* p)
{
    Mat A=*(Mat*)p;
    Mat B;
    if (sigma)
    {
        GaussianBlur(A,B,Size(0,0),sigma);
        imshow("images",B);
    }
    else
        imshow("images",A);
}

void getNeighbours(Point &A, Point &B, int i, int j, int ix, int iy) {

  if (ix < 0) {
    ix = -ix;
    iy = -iy;
  }

  Vec2f gradient(ix, iy);
  Vec2f up(0, -1);
  Vec2f rightUp(1, -1);
  Vec2f right(1, 0);
  Vec2f rightdown(1, 1);
  float upFloat = fabs(up.dot(gradient));
  float rightUpFloat = fabs(rightUp.dot(gradient));
  float rightFloat = fabs(right.dot(gradient));
  float rightdownFloat = fabs(rightdown.dot(gradient));
  float floats[] = {upFloat, rightUpFloat, rightFloat, rightdownFloat};
  float maxFloat = 0;
  for (int i = 0; i < 4; i++) {
    if (floats[i] > maxFloat) {maxFloat = floats[i];}
  }
  if (maxFloat == upFloat) {
    A.x = i;
    A.y = j - 1;
    B.x = i;
    B.y = j + 1;
    return;
  }
  if (maxFloat ==rightUpFloat) {
    A.x = i + 1;
    A.y = j - 1;
    B.x = i - 1;
    B.y = j + 1;
    return;
  }
  if (maxFloat == rightFloat) {
    A.x = i + 1;
    A.y = j;
    B.x = i - 1;
    B.y = j;
    return;
  }
  if (maxFloat == rightdownFloat) {
    A.x = i + 1;
    A.y = j + 1;
    B.x = i - 1;
    B.y = j - 1;
  }
}

Data getData(Mat& I) {
   int m=I.rows,n=I.cols;
  cout << "Rows: " << m <<endl;
  cout << "Columns: " << n << endl;
  // * GETTING IX, IY, G
    Mat Ix(m,n,CV_32F),Iy(m,n,CV_32F),G(m,n,CV_32F);
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
        {
            float ix,iy;
            if (i==0 || i==m-1 || j== 0 || j == n-1) {
                iy=0;
                ix=0;
            }
            else {
                iy=(float(I.at<uchar>(i+1,j)) - float(I.at<uchar>(i-1,j))) * 2;
                iy+=(float(I.at<uchar>(i+1, j-1)) - float(I.at<uchar>(i-1, j-1)));
                iy+=(float(I.at<uchar>(i+1, j+1)) - float(I.at<uchar>(i-1, j+1)));

                ix=(float(I.at<uchar>(i,j+1)) - float(I.at<uchar>(i,j-1))) * 2;
                ix+=(float(I.at<uchar>(i-1, j+1)) - float(I.at<uchar>(i-1, j-1)));
                ix+=(float(I.at<uchar>(i+1, j+1)) - float(I.at<uchar>(i+1, j-1)));
            }
            Ix.at<float>(i,j)=ix;
            Iy.at<float>(i,j)=iy;
            G.at<float>(i,j)=sqrt(ix*ix+iy*iy);
        }
    }
    imshow("images",float2byte(Ix));
    waitKey();
    imshow("images",float2byte(Iy));
    waitKey();
    imshow("images",float2byte(G));
    waitKey();
    Data D;
    D.Ix = Ix;
    D.Iy = Iy;
    D.G = G;
    return D;
}

bool isHighestOnSlope(const Data &D, int i, int j) {
    float ix, iy;
    ix = D.Ix.at<float>(i,j);
    iy = D.Iy.at<float>(i,j);
    Point A,B;
    getNeighbours(A, B, i, j, ix, iy);
    if (D.G.at<float>(i,j) > D.G.at<float>(A.x, A.y) &&
        D.G.at<float>(i,j) > D.G.at<float>(B.x, B.y)) return true;
    else return false;
}

void getBorder(Mat& I, float p1, float p2) {
    int m = I.rows,n = I.cols;
    Data D = getData(I);
    double minVal, maxVal;
    minMaxLoc(float2byte(D.G), &minVal, &maxVal);
    cout << "Max value of gradient: " << maxVal << endl;
    cout << "Min value of gradient: " << minVal << endl;

    float s1 = (1 - p1) * minVal + p1 * maxVal;
    float s2 = (1 - p2) * minVal + p2 * maxVal;
    cout << "Threshold 1: " << s1 << endl;
    cout << "Threshold 2: " << s2 << endl;

    // Getting original grains
    std::queue<Point> grains;
    std::cout << "Number of original grains: " << grains.size() << std::endl;

    Mat treated(m, n, CV_32F);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        treated.at<float>(i,j) = 0;
      }
    }

    Mat C(m, n, CV_32F);
    for (int i = 0; i< m; i++) {
      for (int j =0 ; j < n ; j ++) {
        C.at<float>(i,j) = 0;
      }
    }

    int passingThreshold = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          if (D.G.at<float>(i,j) >= s1) {
            passingThreshold++;
            if (isHighestOnSlope(D, i, j)) {
                grains.push(Point(i, j));
                C.at<float>(i,j) = 1;
            }
          }
        }
    }

    cout << "Number of points passing the threshold: " << passingThreshold << endl;
    std::cout << "Number of original grains: " << grains.size() << std::endl;
    imshow("contour", float2byte(C));
    waitKey();

    // Propagating
    int c = 0;
    int tours = 0;
    while (!grains.empty()) {
      //cout<< "Before taking element" << endl;
      Point grain = grains.front();
      int x = grain.x;
      int y = grain.y;
      //cout << "Got element" <<endl;
      grains.pop();
      //cout << "Popped" << endl;
      if (treated.at<float>(grain.x, grain.y) == 1) {continue;}
      tours++;
      //cout << "Before checks" <<endl;
      if (D.G.at<float>(grain.x, grain.y) >= s2 && isHighestOnSlope(D, grain.x, grain.y)) {
         C.at<float>(grain.x, grain.y) = 1;
      }
      Point A, B;
      float ix = D.Ix.at<float>(x, y);
      float iy = D.Iy.at<float>(grain.x, grain.y);
      getNeighbours(A, B, grain.x, grain.y, iy, -ix);
      //grains.push(B);
      grains.push(A);
      if (D.G.at<float>(A.x, A.y) >= s2) {
        grains.push(A);
        c++;
      }
      if (D.G.at<float>(B.x, B.y) >= s2) {
        grains.push(B);
        c++;
      }
      //grains.push(B);
      treated.at<float>(grain.x, grain.y) = 1;
    }

    cout << "Tours: " << tours << endl;
    cout << "Grains sprouted: " << c << endl;
    imshow("contour", float2byte(C));
    waitKey();
}

int main()
{
    Mat A=imread("../fruits.jpg");
    namedWindow("images", CV_WINDOW_AUTOSIZE);
    Vec3b c=A.at<Vec3b>(12,12);
    cout << c << endl;
    resizeWindow("images", 500, 500);
    imshow("images",A);
    waitKey();
    Mat I;
    cvtColor(A,I,CV_BGR2GRAY);
    cout << int(I.at<uchar>(12,12)) << endl;
    imshow("images",I);
    waitKey();
    getBorder(I, 0.6, 0.1);
    //Mat C;
    //threshold(G,C,10,1,THRESH_BINARY);
    //imshow("images",float2byte(C));
    int value = 0;
    createTrackbar("sigma","images", &value, 20, onTrackbar, &A);
    cout << "created trackbar" << endl;
    imshow("images",A);
    waitKey();

    return 0;
}
