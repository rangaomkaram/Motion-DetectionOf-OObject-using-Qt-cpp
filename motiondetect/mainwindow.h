#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
using namespace cv;
namespace Ui {
class MainWindow;
}
class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = NULL);
    ~MainWindow();
    void processVideo();
    std::string intToString(int number);
    void searchForMovement(Mat thresholdImage, Mat &cameraFeed);
private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
