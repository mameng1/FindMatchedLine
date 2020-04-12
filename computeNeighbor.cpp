#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <Eigen/Geometry>
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
#include "lib/view.h"
#include<omp.h>
using namespace std;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class FindNeighbor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FindNeighbor(std::string images_path,std::string pose_path,std::string depth_path,
                 std::string intrin_path,std::string matching_file)
    {
        loadFile(images_path,pose_path,depth_path,intrin_path,matching_file);
    }
    void readNames(std::vector<std::string>& v_imageName, std::string imageNameFile);
    void readIntrinsic(string path);
    Eigen::Isometry3d readPose(string path);
    PointCloud::Ptr image2Cloud();
    void computeNeighbor(std::vector<std::string>& imageNames,
                         std::vector<std::string>& poseNames,
                         std::vector<std::string>& depthNames);
    void constructViews(std::vector<std::string>& imageNames,
                        std::vector<std::string>& poseNames,
                        std::vector<std::string>& depthNames,
                        std::vector<cv::Mat>& colorImgs,
                        std::vector<cv::Mat>& depthImgs,
                        std::vector<Eigen::Isometry3d>& poses);
    void constructPointCloud(std::vector<cv::Mat>& colorImgs,
                             std::vector<cv::Mat>& depthImgs,
                             std::vector<Eigen::Isometry3d>& poses);
    void loadFile(std::string images_path,std::string pose_path,std::string depth_path,
                           std::string intrin_path,std::string matching_file);
    void findNeighborForOneCam(int camID);

    void mergeNeighbor(std::string matching_file);

    std::map<unsigned int,std::list<unsigned int> > worldpoints2views_;
    std::map<unsigned int,std::list<unsigned int> > views2worldpoints_;
    std::map<unsigned int,std::set<unsigned int> > visual_neighbors_;
    std::map<unsigned int,unsigned int> num_worldpoints_;
    std::vector<L3DPP::View::Ptr> views_;
    int iheight_=-1;
    int iwidth_=-1;
    double fx_,fy_,cx_,cy_;
};
Eigen::Isometry3d FindNeighbor::readPose(string path)
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

void FindNeighbor::readNames(std::vector<std::string>& v_imageName, std::string imageNameFile) {
    std::fstream file(imageNameFile);
    std::string temp;
    while (getline(file, temp)) {
        v_imageName.push_back(temp);
    }
}
void FindNeighbor::readIntrinsic(string path)
{
    double pose_tem[8];
    ifstream fin(path);
    for(int i=0;i<8;i++)
    {
        fin>>pose_tem[i];
    }
    fx_=pose_tem[0];
    cx_=pose_tem[2];
    fy_=pose_tem[5];
    cy_=pose_tem[6];
}
void FindNeighbor::constructViews(std::vector<std::string>& imageNames,
                                  std::vector<std::string>& poseNames,
                                  std::vector<std::string>& depthNames,
                                  std::vector<cv::Mat>& colorImgs,
                                  std::vector<cv::Mat>& depthImgs,
                                  std::vector<Eigen::Isometry3d>& poses)
{
    views_.resize(imageNames.size());
    Eigen::Matrix3d K;
    K<<fx_,0,cx_,
       0,fy_,cy_,
       0,0,1;

    for ( int i=0; i<imageNames.size(); i++ )
    {
        cv::Mat image=cv::imread(imageNames[i]);
        iheight_=image.rows;
        iwidth_=image.cols;
        colorImgs.push_back(image);
        cv::Mat depth=cv::imread(depthNames[i],-1);
        cv::Mat resize_depth;
        cv::resize(depth,resize_depth,cv::Size(image.cols,image.rows));
        depthImgs.push_back(resize_depth);
        Eigen::Isometry3d T=readPose(poseNames[i]);
        poses.push_back(T);

        Eigen::Isometry3d T_inv=T.inverse();

        L3DPP::View::Ptr view_ptr;
        view_ptr.reset(new L3DPP::View(i,std::vector<Eigen::Vector4f>(),
                                       K,T_inv.rotation(),T_inv.translation(),iwidth_,iheight_,0,imageNames[i]));
        views_[i]=view_ptr;
    }
}
void FindNeighbor::constructPointCloud(std::vector<cv::Mat>& colorImgs,
                         std::vector<cv::Mat>& depthImgs,
                         std::vector<Eigen::Isometry3d>& poses)
{
    // 计算点云并拼接
    double depthScale = 1000.0;

    cout<<"正在将图像转换为点云..."<<endl;

    // 新建一个点云
    PointCloud::Ptr pointCloud( new PointCloud );
    for ( int i=0; i<colorImgs.size(); i++ )
    {
        PointCloud::Ptr current( new PointCloud );
        cout<<"转换图像中: "<<i+1<<endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        iheight_=color.rows;
        iwidth_=color.cols;
        for ( int v=0; v<color.rows; v=v+16 )
            for ( int u=0; u<color.cols; u=u+16 )
            {
                unsigned int d = depth.ptr<unsigned short> ( v )[u]; // 深度值
                if ( d==0 ) continue; // 为0表示没有测量到
                if ( d >= 7000 ) continue; // 深度太大时不稳定，去掉
                Eigen::Vector3d point;
                point[2] = double(d)/depthScale;
                point[0] = (u-cx_)*point[2]/fx_;
                point[1] = (v-cy_)*point[2]/fy_;
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
    voxel_filter.setLeafSize( 0.01, 0.01, 0.01 );       // resolution
    PointCloud::Ptr tmp ( new PointCloud );
    voxel_filter.setInputCloud( pointCloud );
    voxel_filter.filter( *tmp );
    tmp->swap( *pointCloud );
    cout<<"滤波之后，点云共有"<<pointCloud->size()<<"个点."<<endl;

     for(size_t p=0;p<pointCloud->size();p++)
     {
         PointT point = pointCloud->points[p];
         Eigen::Vector3d pointWorld;
         pointWorld<<point.x,point.y,point.z;
         for (int i=0;i<colorImgs.size();i++)
         {
             Eigen::Isometry3d T = poses[i];
             Eigen::Isometry3d T_inv = T.inverse();
             cv::Mat& depth = depthImgs[i];
             Eigen::Vector3d cur_p=T_inv*pointWorld;
             if(cur_p[2]>0)
             {
                 double u=(fx_*cur_p[0]+cx_*cur_p[2])/cur_p[2];
                 double v=(fy_*cur_p[1]+cy_*cur_p[2])/cur_p[2];
                 if(u>=0 && u<iwidth_ && v>=0 && v<iheight_)
                 {
                     unsigned int d = depth.ptr<unsigned short> ( int(v) )[int(u)]; // 深度值
                     double valid_d=double(d)/depthScale;
                     if(std::abs(valid_d-cur_p[2])<0.1)
                     {
                        worldpoints2views_[p].push_back(i);
                        views2worldpoints_[i].push_back(p);
                     }
                 }
             }
         }
     }
     for (int i=0;i<colorImgs.size();i++)
     {
        num_worldpoints_[i]=views2worldpoints_[i].size();
     }
}
void FindNeighbor::loadFile(std::string images_path,std::string pose_path,std::string depth_path,
                       std::string intrin_path,std::string matching_file)
{
    readIntrinsic(intrin_path);
    std::vector<std::string> imageNames,poseNames,depthNames;

    std::vector<cv::Mat> colorImgs;
    std::vector<cv::Mat> depthImgs;
    std::vector<Eigen::Isometry3d> poses;
    readNames(imageNames,images_path);
    readNames(poseNames,pose_path);
    readNames(depthNames,depth_path);
    constructViews(imageNames,poseNames,depthNames,colorImgs,depthImgs,poses);
    constructPointCloud(colorImgs,depthImgs,poses);
    for(size_t i=0;i<imageNames.size();i++)
    {
        findNeighborForOneCam(i);
    }
    mergeNeighbor(matching_file);
}
void FindNeighbor::findNeighborForOneCam(int camID)
{
    if(camID==48)
        int debug=0;
    if(visual_neighbors_.find(camID) != visual_neighbors_.end())
    {
        visual_neighbors_[camID].clear();
    }
    std::map<unsigned int,unsigned int> commonWPs;

    std::list<unsigned int>::const_iterator wp_it = views2worldpoints_[camID].begin();
    for(; wp_it!=views2worldpoints_[camID].end(); ++wp_it)
    {
        // iterate over worldpoints
        unsigned int wpID = *wp_it;

        std::list<unsigned int>::const_iterator view_it = worldpoints2views_[wpID].begin();
        for(; view_it!=worldpoints2views_[wpID].end(); ++view_it)
        {
            // all views are potential neighbors
            unsigned int vID = *view_it;
            if(vID != camID)
            {
                if(commonWPs.find(vID) == commonWPs.end())
                {
                    commonWPs[vID] = 1;
                }
                else
                {
                    ++commonWPs[vID];
                }
            }
        }
    }
    if(commonWPs.size() == 0)
        return;

    // find visual neighbors
    std::set<unsigned int> used_neighbors;
    std::list<L3DPP::VisualNeighbor> neighbors;
    L3DPP::View::Ptr v = views_[camID];
    std::map<unsigned int,unsigned int>::const_iterator c_it = commonWPs.begin();
    for(; c_it!=commonWPs.end(); ++c_it)
    {
        unsigned int vID = c_it->first;
        unsigned int num_common_wps = c_it->second;

        L3DPP::VisualNeighbor vn;
        vn.camID_ = vID;
        vn.score_ = 2.0f*float(num_common_wps)/float(num_worldpoints_[camID]+num_worldpoints_[vID]);
        vn.axisAngle_ = v->opticalAxesAngle(views_[vID]);
        vn.distance_score_ = v->distanceVisualNeighborScore(views_[vID]);

        // check baseline
        if(vn.axisAngle_ < 1.571f && num_common_wps > 4) // ~ PI/2
        {
            neighbors.push_back(vn);
        }
    }
    neighbors.sort(L3DPP::sortVisualNeighborsByScore);

    // reduce to best neighbors
    if(neighbors.size() > 1)
    {
        // copy neighbors
        std::list<L3DPP::VisualNeighbor> neighbors_tmp = neighbors;

        // get max score
        float score_t = 0.80f*neighbors.front().score_;
        unsigned int num_bigger_t = 0;

        // count the number of highly similar views
        std::list<L3DPP::VisualNeighbor>::const_iterator nit = neighbors.begin();
        while(nit!=neighbors.end() && (*nit).score_ > score_t)
        {
            ++num_bigger_t;
            ++nit;
        }

        neighbors.resize(num_bigger_t);

        // resort based on projective_score and world_point_score
        neighbors.sort(L3DPP::sortVisualNeighborsByDistScore);

        if(neighbors.size() > 1)
            neighbors.resize(1);
    }
    std::list<L3DPP::VisualNeighbor>::const_iterator nit = neighbors.begin();
    while(nit!=neighbors.end())
    {
        visual_neighbors_[camID].insert(nit->camID_);
        nit++;
    }
}
void FindNeighbor::mergeNeighbor(std::string matching_file)
{
    std::set<string> str_set;
    std::map<unsigned int,std::set<unsigned int> >::iterator it;
    for(it=visual_neighbors_.begin();it != visual_neighbors_.end();it++)
    {
        std::vector<unsigned int> all_camId;
        unsigned int camId=it->first;
        all_camId.push_back(camId);
        std::set<unsigned int> neighbor=it->second;
        std::set<unsigned int>::iterator sit=neighbor.begin();
        for(;sit!=neighbor.end();sit++)
        {
            unsigned int nId=*sit;
            all_camId.push_back(nId);
        }
        std::sort(all_camId.begin(),all_camId.end());
        stringstream ss_t;
        ss_t<<all_camId.size();
        std::string tem_str=ss_t.str()+" ";
        for(size_t i=0;i<all_camId.size();i++)
        {
            stringstream sstr;
            sstr<<all_camId[i]<<" ";
            tem_str+=sstr.str();
        }
        str_set.insert(tem_str);
    }
    ofstream fos(matching_file);
    std::set<string>::iterator sit=str_set.begin();
    for(;sit!=str_set.end();sit++)
    {
        stringstream sstr(*sit);
        int cam_num;
        sstr>>cam_num;
        std::string res_str;
        for(int i=0;i<cam_num;i++)
        {
            int id;
            sstr>>id;
            L3DPP::View::Ptr v =views_[id];
            res_str+=(v->image_name_+" ");
        }
        fos<<res_str<<std::endl;
    }
    fos.close();
}
int main( int argc, char** argv )
{
    string root_dir="/home/mameng/dataset/scannet/datadir_endline";
    boost::filesystem::path root_path(root_dir);
    std::vector<std::string> subdir_vec;
    boost::filesystem::directory_iterator beg_iter(root_path);
    boost::filesystem::directory_iterator end_iter;
    for (; beg_iter != end_iter; ++beg_iter)
    {
        if (boost::filesystem::is_directory(*beg_iter))
        {
            subdir_vec.push_back(beg_iter->path().string());
        }
    }
    std::sort(subdir_vec.begin(),subdir_vec.end());
    //omp_set_num_threads(20);
    #pragma omp parallel for
    for(size_t i=0;i<subdir_vec.size();i++)
    {
        string sroot_dir=subdir_vec[i];
        string imges_path=sroot_dir+"/color_path.txt";
        string pose_path=sroot_dir+"/pose_path.txt";
        string depth_path=sroot_dir+"/depth_path.txt";
        string intrin_path=sroot_dir+"/intrinsics_color.txt";
        string matching_file=sroot_dir+"/matching.txt";
        FindNeighbor neighbor(imges_path,pose_path,depth_path,intrin_path,matching_file);
    }
    //string root_dir="/home/mameng/dataset/scannet/datadir_file/scene0000_00/";

    return 0;
}

