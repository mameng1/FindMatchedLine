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
#include<pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/ransac.h>
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
                 vector<std::string> depthNames,vector<double> imageIntrin,int& image_width,int image_height)
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
void readLines(std::string line_path,vector<cv::Vec4d>& lines)
{
    std::ifstream fline(line_path);
    std::string line_str;
    while(std::getline(fline,line_str))
    {
        std::stringstream ssline(line_str);
        float x1,y1,x2,y2;
        ssline>>x1>>y1>>x2>>y2;
        lines.push_back(cv::Vec4d(x1,y1,x2,y2));
    }
}
double interDepth(double x,double y,const cv::Mat& depth)
{
  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;
  int height=depth.rows;
  int width=depth.cols;
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (double)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (double)x_low;
  } else {
    x_high = x_low + 1;
  }
  double ly = y - y_low;
  double lx = x - x_low;
  double hy = 1. - ly, hx = 1. - lx;
  double w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  double pos1 = depth.ptr<unsigned short> ( y_low )[x_low];
  double pos2 = depth.ptr<unsigned short> ( y_low )[x_high];
  double pos3 = depth.ptr<unsigned short> ( y_high )[x_low];
  double pos4 = depth.ptr<unsigned short> ( y_high )[x_high];
  double output_val=w1 * pos1 +w2 * pos2 +w3 * pos3 +w4 * pos4;
  return output_val;
}
Eigen::Vector3d img2word(double u,double v,double d,double fx,double fy,double cx,double cy)
{
    double x = (u-cx)*d/fx;
    double y = (v-cy)*d/fy;
    Eigen::Vector3d p;
    p<<x,y,d;
    return p;
}
std::vector<float> fitLineRansac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,std::vector<int>& inliers)
{
    Eigen::VectorXf line_c;
    std::vector<float> line_tem;
    if(cloud->empty())
        return line_tem;
    pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr
        model_s(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
    ransac.setDistanceThreshold (3);
    ransac.setMaxIterations(10000);
    ransac.computeModel();
    ransac.getInliers(inliers);
    ransac.getModelCoefficients(line_c);
    line_tem.resize(4);
    line_tem[0]=line_c[0];
    line_tem[1]=line_c[1];
    line_tem[2]=line_c[3];
    line_tem[3]=line_c[4];
    return line_tem;
}
Eigen::Vector3d getIntersectWithLineAndPlane(Eigen::Vector3d point,Eigen::Vector3d direct,Eigen::Vector3d planeNormal,Eigen::Vector3d planePoint)
{
        float d = (planePoint - point).dot(planeNormal)/(direct.dot(planeNormal));
        return d * direct + point;
}
Eigen::Vector3d calculatePlaneCoffe(const std::vector<float>& line_c,double fx,double fy,double cx,double cy)
{
    float u1=line_c[0];
    float v1=line_c[1];
    float u2=u1+1000*line_c[2];
    float v2=v1+1000*line_c[3];

    double x1 = (u1-cx)/fx;
    double y1 = (v1-cy)/fy;
    double x2 = (u2-cx)/fx;
    double y2 = (v2-cy)/fy;

    Eigen::Vector3d p1,p2;
    p1<<x1,y1,1.0;
    p2<<x2,y2,1.0;
    Eigen::Vector3d norm_vec=p1.cross(p2);
    norm_vec.normalize();
    return norm_vec;
}
int main( int argc, char** argv )
{
    string root_dir="/home/mameng/dataset/scannet/datadir_file/scene0000_00/";
    string imges_path=root_dir+"color_path.txt";
    string pose_path=root_dir+"pose_path.txt";
    string depth_path=root_dir+"depth_path.txt";
    string line_path=root_dir+"line_path.txt";
    string intrin_path=root_dir+"intrinsics_color.txt";

    std::vector<std::string> imageNames,poseNames,depthNames,linesName;
    vector<double> imageIntrin;
    readImageName(imageNames,imges_path);
    readImageName(poseNames,pose_path);
    readImageName(depthNames,depth_path);
    readImageName(linesName,line_path);
    imageIntrin=readIntrinsic(intrin_path);

    int image_height=-1,image_width=-1;
    //PointCloud::Ptr pointCloud=image2Cloud(imageNames,poseNames,depthNames,imageIntrin,image_width,image_height);
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图
    vector<Eigen::Isometry3d,Eigen::aligned_allocator<Eigen::Isometry3d>> poses;         // 相机位姿


    for ( int i=0; i<imageNames.size(); i++ )
    {
        //boost::format fmt( "./data/%s/%d.%s" ); //图像文件格式
        cv::Mat image=cv::imread(imageNames[i]);
        image_height=image.rows;
        image_width=image.cols;
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
    std::cout<<cx<<" "<<cy<<" "<<fx<<" "<<fy<<std::endl;
    double depthScale = 1000.0;

    cout<<"正在将图像转换为点云..."<<endl;

    // 新建一个点云
    cv::Mat left_color = colorImgs[0];
    cv::Mat left_depth = depthImgs[0];
    Eigen::Isometry3d left_T = poses[0];
    std::vector<cv::Vec4d> left_lines;
    readLines(linesName[0],left_lines);

    cv::Mat right_color = colorImgs[1];
    cv::Mat right_depth = depthImgs[1];
    Eigen::Isometry3d right_T= poses[1];
    std::vector<cv::Vec4d> right_lines;
    readLines(linesName[1],right_lines);


    Eigen::Isometry3d rela_T=right_T.inverse()*left_T;//T_rl
    std::cout<<rela_T.matrix()<<std::endl;
    for(int i=0;i<left_lines.size();i++)
    {
        cv::Vec4d line=left_lines[i];
        cv::line(left_color,cv::Point(line[0],line[1]),cv::Point(line[2],line[3]),cv::Scalar(255,0,0),1);
        double x_len=std::abs(line[0]-line[2]);
        double y_len=std::abs(line[1]-line[3]);
        double x1=line[0],y1=line[1],x2=line[2],y2=line[3];
        Eigen::Vector3d line_c;
        line_c[0]=y1-y2;
        line_c[1]=-(x1-x2);
        line_c[2]=x1*y2-x2*y1;

        std::vector<cv::Vec2d> valid_p;
        if(x_len>y_len)
        {
            double max_x=std::max(line[0],line[2]);
            double min_x=std::min(line[0],line[2]);
            for(double x=min_x;x<=max_x;x++)
            {
                double y=-(line_c[0]*x+line_c[2])/line_c[1];
                valid_p.push_back(cv::Vec2d(x,y));
                cv::circle(left_color,cv::Point(x,y),1,cv::Scalar(0,255,0),1);
            }
        }
        else
        {
            double max_y=std::max(line[1],line[3]);
            double min_y=std::min(line[1],line[3]);
            for(double y=min_y;y<=max_y;y++)
            {
                double x=-(line_c[1]*y+line_c[2])/line_c[0];
                valid_p.push_back(cv::Vec2d(x,y));
                cv::circle(left_color,cv::Point(x,y),1,cv::Scalar(0,255,0),1);
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr linepoint_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for(int p=0;p<valid_p.size();p++)
        {
            int u=valid_p[p][0];
            int v=valid_p[p][1];
            double d = left_depth.ptr<unsigned short> ( v )[u]; // 深度值
            if ( d==0 ) continue; // 为0表示没有测量到
            if ( d >= 7000 ) continue; // 深度太大时不稳定，去掉

            Eigen::Vector3d point;
            point[2] = double(d)/depthScale;
            point[0] = (u-cx)*point[2]/fx;
            point[1] = (v-cy)*point[2]/fy;
            Eigen::Vector3d pointWorld = rela_T*point;
            if(pointWorld[2]>0)
            {
                u=(fx*pointWorld[0]+cx*pointWorld[2])/pointWorld[2];
                v=(fy*pointWorld[1]+cy*pointWorld[2])/pointWorld[2];

                if(u>=0 && u<image_width && v>=0 && v<image_height)
                {
                    pcl::PointXYZ proj_p;
                    proj_p.x=u;proj_p.y=v;proj_p.z=0;
                    // proj_p.x=pointWorld[0];proj_p.y=pointWorld[1];proj_p.z=pointWorld[];
                    linepoint_cloud->push_back(proj_p);
                    cv::circle(right_color,cv::Point(u,v),2,cv::Scalar(255,0,0),2);
                }
            }
        }

        std::vector<int> inliers_idx;
        std::vector<float> fitline_c;
        if(linepoint_cloud->size()<10) continue;
        fitline_c=fitLineRansac(linepoint_cloud,inliers_idx);
        double valid_ratio=(double)inliers_idx.size()/linepoint_cloud->size();
        if(valid_ratio<0.5) continue;
        std::cout<<valid_ratio<<std::endl;
        Eigen::Vector3d plan_norm=calculatePlaneCoffe(fitline_c,fx,fy,cx,cy);
        Eigen::Vector3d plan_r;
        plan_r<<plan_norm[0],plan_norm[1],plan_norm[2];
        Eigen::Vector3d plan_l=rela_T.rotation().inverse()*plan_r;
        Eigen::Vector3d point_l=rela_T.inverse().translation();
        Eigen::Vector3d word_p1=img2word(x1,y1,1,fx,fy,cx,cy);
        Eigen::Vector3d word_p2=img2word(x2,y2,1,fx,fy,cx,cy);
        word_p1.normalize();
        word_p2.normalize();
        Eigen::Vector3d line_norm1=word_p1;
        Eigen::Vector3d line_norm2=word_p2;
        Eigen::Vector3d line_endp1=getIntersectWithLineAndPlane(word_p1,line_norm1,plan_l,point_l);
        Eigen::Vector3d line_endp2=getIntersectWithLineAndPlane(word_p2,line_norm2,plan_l,point_l);
        Eigen::Vector3d rendp1=rela_T*line_endp1;
        Eigen::Vector3d rendp2=rela_T*line_endp2;

        double ue1=(fx*rendp1[0]+cx*rendp1[2])/rendp1[2];
        double ve1=(fy*rendp1[1]+cy*rendp1[2])/rendp1[2];
        if(ue1>=0 && ue1<image_width && ve1>=0 && ve1<image_height)
        {
            cv::circle(right_color,cv::Point(ue1,ve1),4,cv::Scalar(255,255,0),2);
        }
        double ue2=(fx*rendp2[0]+cx*rendp2[2])/rendp2[2];
        double ve2=(fy*rendp2[1]+cy*rendp2[2])/rendp2[2];
        if(ue2>=0 && ue2<image_width && ve2>=0 && ve2<image_height)
        {
            cv::circle(right_color,cv::Point(ue2,ve2),4,cv::Scalar(255,255,0),2);
        }

        for(size_t p=0;p<inliers_idx.size();p++)
        {
            pcl::PointXYZ pp=linepoint_cloud->points[inliers_idx[p]];
            cv::circle(right_color,cv::Point(pp.x,pp.y),1,cv::Scalar(0,0,255),1);
        }
    }
    cv::imshow("left",left_color);
    cv::imshow("right",right_color);
    cv::waitKey(0);
    /*PointCloud::Ptr pointCloud( new PointCloud );
    for ( int i=0; i<imageNames.size(); i++ )
    {
        PointCloud::Ptr current( new PointCloud );
        cout<<"转换图像中: "<<i+1<<endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        std::vector<Eigen::Vector4d> lines;
        readLines(linesName[i],lines);
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

    std::vector<std::vector<int> > point2view(pointCloud->size());

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
        fos<<imageNames[i]<<" "<<linesName[i]<<" "<<quad.w()<<" "<<quad.x()<<" "<<quad.y()<<" "<<quad.z()<<" "
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

