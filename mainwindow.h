#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void process();     //main loop
    void webcam();
    void numeric();
    void edge();
    void hough();
    void motiondetect();
    void searchForMovement(Mat thresholdImage, Mat &cameraFeed);
    void detectAndDisplay( Mat &frame );
    void streifenlicht();
    void cam_calib();
    void featuredetection();
    int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension);
    bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients);
    void aruco();
    void facedetect();

    std::string intToString(int number);

private slots:
    void on_pBwebcam_clicked();

    void on_pBfacedetect_clicked();

    void on_pBExit_clicked();

    void on_pBnumeric_clicked();

    void on_pBedge_clicked();

    void on_pBhough_clicked();

    void on_pBmotiondetect_clicked();

    void on_pBstreifenlicht_clicked();

    void on_pBfeature_clicked();

    void on_pBcamcalib_clicked();

    void on_pBaruco_clicked();
    static void on_hough_param1_trackbar( int, void* );
    static void on_hough_param2_trackbar( int, void* );

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
