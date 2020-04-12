#include <iostream>
#include <fstream>
using namespace std;

#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry> 
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> 
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/cloud_viewer.h>
#include<Eigen/StdVector>
#include <pcl/registration/ndt.h>
void readImageName(std::vector<std::string>& v_imageName, std::string imageNameFile) {
    std::fstream file(imageNameFile);
    std::string temp;
    while (getline(file, temp)) {
        v_imageName.push_back(temp);
    }
}
Eigen::Isometry3d readPose(string path)
{
    double pose_tem[12];
    ifstream fin(path);
    for(int i=0;i<12;i++)
    {
        fin>>pose_tem[i];
    }

    Eigen::Vector3d trans;
    Eigen::Matrix3d rotate;
    rotate<<pose_tem[0],pose_tem[1],pose_tem[2],
            pose_tem[4],pose_tem[5],pose_tem[6],
            pose_tem[8],pose_tem[9],pose_tem[10];
    trans<<pose_tem[3],pose_tem[7],pose_tem[11];

    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();
    T.rotate(rotate);
    T.pretranslate(trans);
    return T;
}
std::vector<double> readIntrinsic(string path)
{
    double pose_tem[8];
    ifstream fin(path);
    for(int i=0;i<8;i++)
    {
        fin>>pose_tem[i];
    }
    vector<double> intirn_vec;
    intirn_vec.push_back(pose_tem[0]);
    intirn_vec.push_back(pose_tem[2]);
    intirn_vec.push_back(pose_tem[5]);
    intirn_vec.push_back(pose_tem[6]);
    return intirn_vec;
}
// 定义点云使用的格式：这里用的是XYZRGB
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
PointCloud::Ptr image2Cloud(vector<std::string> imageNames,vector<std::string> poseNames,
                 vector<std::string> depthNames,vector<double> imageIntrin,bool print)
{

    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    vector<Eigen::Isometry3d,Eigen::aligned_allocator<Eigen::Isometry3d>> poses;         // 相机位姿

    /*ifstream fin("./data/pose.txt");
    if (!fin)
    {
        cerr<<"cannot find pose file"<<endl;
        return 1;
    }*/

    for ( int i=0; i<imageNames.size(); i++ )
    {
        //boost::format fmt( "./data/%s/%d.%s" ); //图像文件格式
        cv::Mat image=cv::imread(imageNames[i]);
        cv::Mat resize_img;
       // cv::resize(image,resize_img,cv::Size(640,480));
        colorImgs.push_back(image);
        cv::Mat depth=cv::imread(depthNames[i],-1);
        cv::Mat resize_depth;
        cv::resize(depth,resize_depth,cv::Size(image.cols,image.rows));
        depthImgs.push_back(resize_depth);
        //colorImgs.push_back( cv::imread( (fmt%"color"%(i+1)%"png").str() ));
        //depthImgs.push_back( cv::imread( (fmt%"depth"%(i+1)%"pgm").str(), -1 )); // 使用-1读取原始图像

        Eigen::Isometry3d T=readPose(poseNames[i]);
        poses.push_back(T);
    }

    // 计算点云并拼接
    // 相机内参
    double cx =imageIntrin[1];
    double cy = imageIntrin[3];
    double fx = imageIntrin[0];
    double fy = imageIntrin[2];
    double depthScale = 1000.0;

    cout<<"正在将图像转换为点云..."<<endl;

    // 新建一个点云
    int image_height=-1,image_width=-1;
    PointCloud::Ptr pointCloud( new PointCloud );
    for ( int i=0; i<imageNames.size(); i++ )
    {
        PointCloud::Ptr current( new PointCloud );
        cout<<"转换图像中: "<<i+1<<endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        //std::cout<<T.rotation()<<std::endl;
       // std::cout<<T.translation()<<std::endl;
        image_height=color.rows;
        image_width=color.cols;
        //std::cout<<"debug"<<color.rows<<" "<<color.cols<<std::endl;
        for ( int v=0; v<color.rows; v=v+16 )
            for ( int u=0; u<color.cols; u=u+16 )
            {
                unsigned int d = depth.ptr<unsigned short> ( v )[u]; // 深度值

                if ( d==0 ) continue; // 为0表示没有测量到
                if ( d >= 7000 ) continue; // 深度太大时不稳定，去掉

                Eigen::Vector3d point;
                point[2] = double(d)/depthScale;
                point[0] = (u-cx)*point[2]/fx;
                point[1] = (v-cy)*point[2]/fy;
                Eigen::Vector3d pointWorld = T*point;

                PointT p ;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[ v*color.step+u*color.channels() ];
                p.g = color.data[ v*color.step+u*color.channels()+1 ];
                p.r = color.data[ v*color.step+u*color.channels()+2 ];
                current->points.push_back( p );

            }
        // depth filter and statistical removal
        PointCloud::Ptr tmp ( new PointCloud );
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(current);
        statistical_filter.filter( *tmp );

        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false;
    cout<<"点云共有"<<pointCloud->size()<<"个点."<<endl;

    // voxel filter
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize( 0.001, 0.001, 0.001 );       // resolution
    PointCloud::Ptr tmp ( new PointCloud );
    voxel_filter.setInputCloud( pointCloud );
    voxel_filter.filter( *tmp );
    tmp->swap( *pointCloud );
    cout<<"滤波之后，点云共有"<<pointCloud->size()<<"个点."<<endl;

    return pointCloud;
}
int main( int argc, char** argv )
{
    string root_dir="/home/mameng/dataset/scannet/datadir_file/scene0000_00/";
    string imges_path=root_dir+"color_path.txt";
    string pose_path=root_dir+"pose_path.txt";
    string depth_path=root_dir+"depth_path.txt";
    string intrin_path=root_dir+"intrinsics_color.txt";

    std::vector<std::string> imageNames,poseNames,depthNames;
    vector<double> imageIntrin;
    readImageName(imageNames,imges_path);
    readImageName(poseNames,pose_path);
    readImageName(depthNames,depth_path);
    imageIntrin=readIntrinsic(intrin_path);
   // std::cout<<imageIntrin[0]<<" "<<imageIntrin[1]<<" "<<imageIntrin[2]<<" "<<imageIntrin[3]<<std::endl;
    string anathor_dir="/home/mameng/dataset/scannet/datadir_file/scene0000_01/";
    string imges1_path=anathor_dir+"color_path.txt";
    string pose1_path=anathor_dir+"pose_path.txt";
    string depth1_path=anathor_dir+"depth_path.txt";
    string intrin1_path=anathor_dir+"intrinsics_color.txt";

    std::vector<std::string> image1Names,pose1Names,depth1Names;
    vector<double> image1Intrin;
    readImageName(image1Names,imges1_path);
    readImageName(pose1Names,pose1_path);
    readImageName(depth1Names,depth1_path);
    image1Intrin=readIntrinsic(intrin1_path);

    PointCloud::Ptr scene1=image2Cloud(imageNames,poseNames,depthNames,imageIntrin,false);
    /*PointCloud::Ptr scene2=image2Cloud(image1Names,pose1Names,depth1Names,imageIntrin,true);
    std::cout<<scene1->size()<<" "<<scene2->size()<<std::endl;
    Eigen::Matrix4f trans;
    trans<<   0.99936,    0.0357549,  6.34187e-05  ,  -0.234423,
              -0.0357549 ,    0.999359,   0.00144617,     0.056108,
            -1.16718e-05,  -0.00144752  ,   0.999999 , -0.00313964,
                      0  ,          0    ,       0 ,           1;//T_21
    PointCloud::Ptr tem_cloud(new PointCloud());
    pcl::transformPointCloud(*scene2,*tem_cloud,trans);
    *scene1+=*tem_cloud;*/
    //pcl::io::savePCDFileBinary("scene23.pcd", *scene2 );

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
        viewer->addPointCloud(scene1, "bunny");
         while (!viewer->wasStopped())
         {
             viewer->spinOnce(100);
             boost::this_thread::sleep(boost::posix_time::microseconds(100000));
         }
  //  pcl::io::savePCDFileBinary("scene3.pcd", *scene2 );
    /*std::cout<<"read point end"<<std::endl;

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    //设置依赖尺度NDT参数
    //为终止条件设置最小转换差异
    ndt.setTransformationEpsilon(0.00001);
    //为More-Thuente线搜索设置最大步长
    ndt.setStepSize(0.1);
    //设置NDT网格结构的分辨率（VoxelGridCovariance）
    ndt.setResolution(0.05);
    //设置匹配迭代的最大次数
    ndt.setMaximumIterations(10000);
    // 设置要配准的点云
    ndt.setInputCloud(scene2);
    //设置点云配准目标
    ndt.setInputTarget(scene1);

    //设置使用机器人测距法得到的初始对准估计结果
     Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();


    //计算需要的刚体变换以便将输入的点云匹配到目标点云
     PointCloud::Ptr output_cloud(new PointCloud);
     ndt.align(*output_cloud, init_guess);
     std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
           << " score: " << ndt.getFitnessScore() << std::endl;
    //使用创建的变换对未过滤的输入点云进行变换
    pcl::transformPointCloud(*scene2, *output_cloud, ndt.getFinalTransformation());
    // 初始化点云可视化界面*/
    /*boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
    viewer->addPointCloud(scene1, "bunny");
    viewer->addPointCloud(output_cloud, "bunny1");
     while (!viewer->wasStopped())
     {
         viewer->spinOnce(100);
         boost::this_thread::sleep(boost::posix_time::microseconds(100000));
     }
*/
    return (0);


    /*std::vector<std::vector<int> > point2view(pointCloud->size());

     for(size_t p=0;p<pointCloud->size();p++)
     {
         std::vector<int> views;
         PointT point = pointCloud->points[p];
         Eigen::Vector3d pointWorld;
         pointWorld<<point.x,point.y,point.z;
         for (int i=0;i<imageNames.size();i++)
         {
             Eigen::Isometry3d T = poses[i];
             Eigen::Isometry3d T_inv = T.inverse();

             Eigen::Vector3d cur_p=T_inv*pointWorld;
             if(cur_p[2]>0)
             {
                 double u=(fx*cur_p[0]+cx*cur_p[2])/cur_p[2];
                 double v=(fy*cur_p[1]+cy*cur_p[2])/cur_p[2];
                 if(u>=0 && u<image_width && v>=0 && v<image_height)
                 {
                     views.push_back(i);
                 }
             }
         }
         point2view[p]=views;
     }
    ofstream fos(root_dir+"sfm_data.txt");
    fos<<imageNames.size()<<std::endl;
    for (int i=0;i<imageNames.size();i++)
    {
        Eigen::Isometry3d T = poses[i];
        Eigen::Matrix3d R=T.rotation().inverse();
        Eigen::Vector3d C=T.translation();
        Eigen::Quaterniond quad(R);
        fos<<imageNames[i]<<" "<<quad.w()<<" "<<quad.x()<<" "<<quad.y()<<" "<<quad.z()<<" "
          <<C[0]<<" "<<C[1]<<" "<<C[2]<<std::endl;
    }
    fos<<fx<<" "<<cx<<" "<<fy<<" "<<cy<<std::endl;
    fos<<point2view.size()<<std::endl;
    for(size_t i=0;i<point2view.size();i++)
    {
        std::vector<int>& views=point2view[i];
        PointT point = pointCloud->points[i];
        fos<<point.x<<" "<<point.y<<" "<<point.z<<" "<<views.size();
        for(int v=0;v<views.size();v++)
        {
            fos<<" "<<views[v];
        }
        fos<<std::endl;
    }
    fos.close();

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
    viewer->addPointCloud(pointCloud, "bunny");
     while (!viewer->wasStopped())
     {
         viewer->spinOnce(100);
         boost::this_thread::sleep(boost::posix_time::microseconds(100000));
     }

    cout<<"滤波之后，点云共有"<<pointCloud->size()<<"个点."<<endl;
    
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud );*/
    return 0;
}

