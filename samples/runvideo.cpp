#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#include <dirent.h>

#include <gflags/gflags.h>

using namespace std;
using namespace cv;

DEFINE_bool(gray, false, "If use raw gray level features or not. [default=false]");
DEFINE_bool(hog, true, "If use hog features or not. [default=true]");
DEFINE_bool(lab, false, "If use lab features or not. [default=false]");
DEFINE_bool(multiscale, true, "Performs multi-scale detection. [default=true]");
DEFINE_bool(fixedwindow, false, "Keep the window size fixed when in single-scale mode (multi-scale always used a fixed window). [default=false]");
DEFINE_string(input, "", "Input file name with path. [default=""]");
//DEFINE_string(output, "", "Output file name with path. [default=""]");

Mat frame, frameDisplay;
Rect result;
bool setGT = false;

void on_mouse(int event, int x, int y, int flags, void *ustc) {
  static Point pre_pt = Point(0, 0);
  static Point cur_pt = Point(0, 0);
  if (event == CV_EVENT_LBUTTONDOWN) {
    pre_pt = Point(x,y);
  }
  else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)) {
    frame.copyTo(frameDisplay);
    cur_pt = Point(x,y);
    rectangle(frameDisplay, pre_pt, cur_pt, Scalar(255,0,0), 1, 8, 0);
    imshow("display", frameDisplay);
  }
  else if (event == CV_EVENT_LBUTTONUP)
  {
    frame.copyTo(frameDisplay);
    cur_pt = Point(x,y);
    rectangle(frameDisplay, pre_pt, cur_pt, Scalar(255,0,0), 1, 8, 0);
    imshow("display", frameDisplay);
    result.x = pre_pt.x;
    result.y = pre_pt.y;
    result.width = cur_pt.x - pre_pt.x + 1;
    result.height = cur_pt.y - pre_pt.y + 1;
    setGT = true;
  }
}

int main(int argc, char* argv[]) {
	printf("open main");
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_gray == true) FLAGS_hog == false;
  if (FLAGS_lab == true) FLAGS_hog == true;

  KCFTracker tracker(FLAGS_hog, FLAGS_fixedwindow, FLAGS_multiscale, FLAGS_lab);

  namedWindow("display", WINDOW_AUTOSIZE);
  VideoCapture cap(FLAGS_input.c_str());
  if(!cap.isOpened()) {
    cout << "Open input video failed!!\n";
    return -1;
  }
  long frameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);

  cap.read(frame);
  setMouseCallback("display", on_mouse, 0);
  imshow("display", frame);
  waitKey(0);
  while(!setGT) {}
  tracker.init(result, frame);

  for (int idxFrame = 1; idxFrame < frameNumber; idxFrame ++) {
    cap.read(frame);
    result = tracker.update(frame);
    rectangle(frame, Point(result.x, result.y),
      Point(result.x+result.width, result.y+result.height),
      Scalar(0, 255, 255), 1, 8, 0);
    imshow("display", frame);
    if(waitKey(30) >= 0) break;
  }
  cap.release();

  waitKey(0);
  return 0;
}
