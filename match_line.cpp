#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
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
#include<boost/filesystem.hpp>
#include<omp.h>
using namespace std;

class MatchingLine
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MatchingLine(std::string matching_path,std::string intrin_path)
    {
        matching_path_=matching_path;
        loadFile(matching_path,intrin_path);

    }
    void readNames(std::vector<std::string>& v_imageName, std::string imageNameFile);
    void readIntrinsic(string path);
    void readmatcing(std::string matching_path);
    void readLines(std::string line_path,std::vector<Eigen::Vector4f>& lines);
    Eigen::Isometry3d readPose(string path);

    Eigen::Vector3d computeCrossPoint(Eigen::Vector3d& lp,Eigen::Vector3d& lnorm,
                           Eigen::Vector3d& pp,Eigen::Vector3d& pnorm);

    void findMatching(std::vector<L3DPP::View::Ptr>& views_ptr,Eigen::MatrixXd& angles,
                      Eigen::MatrixXd& dists,Eigen::MatrixXd& len_ratio);
    void findUnMatchingLine(std::vector<L3DPP::View::Ptr>& views_ptr);
    std::vector<Eigen::Vector2i> findMatchingOrNOPair(Eigen::MatrixXd& angle,Eigen::MatrixXd& dist,Eigen::MatrixXd& ratio,
                                            std::vector<L3DPP::View::Ptr>& views);
    void filterPairBase3DPos(std::vector<L3DPP::View::Ptr>& views,Eigen::MatrixXd& score);

    bool constructViews(std::vector<std::string>& imageNames,
                        std::vector<std::string>& poseNames,
                        std::vector<std::string>& depthNames,
                        std::vector<std::string>& lineNames,
                        std::vector<L3DPP::View::Ptr>& views);

    void fitLineRansac(std::vector<Eigen::Vector3d>& img_points,
                       std::vector<double>& line_tem,
                       std::vector<Eigen::Vector3d>& inliers_p,double threshold);

    cv::Scalar randomColor(cv::RNG& rng);

    void loadFile(std::string matching_path,std::string intrin_path);

    std::vector<Eigen::Vector2i> computeScoreMatrix(std::vector<L3DPP::View::Ptr>& views,cv::Mat& m);

    void readmatcing(std::string matching_path,
                     std::vector<std::string>& limgs_path,std::vector<std::string>& rimgs_path,
                     std::vector<std::string>& llines_path,std::vector<std::string>& rlines_path,
                     std::vector<std::string>& ldepths_path,std::vector<std::string>& rdepths_path,
                     std::vector<std::string>& lposes_path,std::vector<std::string>& rposes_path);
    void mergeImage(cv::Mat img1,cv::Mat img2,string save_path);
    int iheight_=-1;
    int iwidth_=-1;
    double fx_,fy_,cx_,cy_;
    string matching_path_;
};
void MatchingLine::readLines(std::string line_path,std::vector<Eigen::Vector4f>& lines)
{
    std::ifstream fline(line_path);
    std::string line_str;
    while(std::getline(fline,line_str))
    {
        std::stringstream ssline(line_str);
        float x1,y1,x2,y2;
        ssline>>x1>>y1>>x2>>y2;
        Eigen::Vector4f lt;
        lt<<x1,y1,x2,y2;
        lines.push_back(lt);
    }
}
void MatchingLine::fitLineRansac(std::vector<Eigen::Vector3d>& img_points,
                                 std::vector<double>& line_tem,
                                 std::vector<Eigen::Vector3d>& inliers_p,double threshold)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for(size_t i=0;i<img_points.size();i++)
    {
        pcl::PointXYZ p;
        p.x=img_points[i][0];
        p.y=img_points[i][1];
        p.z=0;
        cloud->push_back(p);
    }
    Eigen::VectorXf line_c;
    if(cloud->empty())
        return;
    cloud->is_dense = false;
    std::vector<int> inliers;
    pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr
        model_s(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
    ransac.setDistanceThreshold (threshold);
    ransac.setMaxIterations(10000);
    ransac.computeModel();
    ransac.getInliers(inliers);
    ransac.getModelCoefficients(line_c);
    line_tem.resize(4);
    line_tem[0]=line_c[0];
    line_tem[1]=line_c[1];
    line_tem[2]=line_c[0]+1000*line_c[3];
    line_tem[3]=line_c[1]+1000*line_c[4];
    for(size_t i=0;i<inliers.size();i++)
    {
        int id=inliers[i];
        Eigen::Vector3d p;
        p<<cloud->points[id].x,cloud->points[id].y,1;
        inliers_p.push_back(p);
    }
}
Eigen::Vector3d MatchingLine::computeCrossPoint(Eigen::Vector3d& lp,Eigen::Vector3d& lnorm,
                       Eigen::Vector3d& pp,Eigen::Vector3d& pnorm)
{
    float d = (pp - lp).dot(pnorm)/(lnorm.dot(pnorm));
    return d * lnorm + lp;
}
void MatchingLine::readmatcing(std::string matching_path,
                               std::vector<std::string>& limgs_path,std::vector<std::string>& rimgs_path,
                               std::vector<std::string>& llines_path,std::vector<std::string>& rlines_path,
                               std::vector<std::string>& ldepths_path,std::vector<std::string>& rdepths_path,
                               std::vector<std::string>& lposes_path,std::vector<std::string>& rposes_path)
{
    ifstream in_s(matching_path);
    std::string cstr;
    while(std::getline(in_s,cstr))
    {
        stringstream sstr(cstr);
        string left_path,right_path;
        sstr>>left_path>>right_path;
        limgs_path.push_back(left_path);
        rimgs_path.push_back(right_path);
    }

    for(size_t i=0;i<limgs_path.size();i++)
    {
        boost::filesystem::path limg_path(limgs_path[i]);
        std::string base_name=limg_path.stem().string();
        boost::filesystem::path ppath=limg_path.parent_path().parent_path();
        std::string line_path=ppath.string()+"/endline/"+base_name+".txt";
        std::string pose_path=ppath.string()+"/pose/"+base_name+".txt";
        std::string depth_path=ppath.string()+"/depth/"+base_name+".png";
        llines_path.push_back(line_path);
        ldepths_path.push_back(depth_path);
        lposes_path.push_back(pose_path);
    }
    for(size_t i=0;i<rimgs_path.size();i++)
    {
        boost::filesystem::path rimg_path(rimgs_path[i]);
        std::string base_name=rimg_path.stem().string();
        boost::filesystem::path ppath=rimg_path.parent_path().parent_path();
        std::string line_path=ppath.string()+"/endline/"+base_name+".txt";
        std::string pose_path=ppath.string()+"/pose/"+base_name+".txt";
        std::string depth_path=ppath.string()+"/depth/"+base_name+".png";
        rlines_path.push_back(line_path);
        rdepths_path.push_back(depth_path);
        rposes_path.push_back(pose_path);
    }

}
std::vector<Eigen::Vector2i> MatchingLine::computeScoreMatrix(std::vector<L3DPP::View::Ptr>& views,cv::Mat& debug_m)
{
    boost::filesystem::path left_path(views[0]->image_name_);
    boost::filesystem::path right_path(views[1]->image_name_);
    boost::filesystem::path match_path(matching_path_);
    string save_path=match_path.parent_path().string()+"/match/";
    boost::filesystem::path ssave_path(save_path);
    if(!boost::filesystem::exists(ssave_path))
        boost::filesystem::create_directory(ssave_path);
    string leftbase_name=left_path.stem().string();
    string rightbase_name=right_path.stem().string();

    string leftsave_path=ssave_path.string()+leftbase_name+"_"+rightbase_name+"_"+leftbase_name+".txt";
    string rightsave_path=ssave_path.string()+leftbase_name+"_"+rightbase_name+"_"+rightbase_name+".txt";
    string matchimg_path=ssave_path.string()+leftbase_name+"_"+rightbase_name+"m.png";
    string umatchimg_path=ssave_path.string()+leftbase_name+"_"+rightbase_name+"um.png";
    if(boost::filesystem::exists(boost::filesystem::path(leftsave_path)) &&
       boost::filesystem::exists(boost::filesystem::path(rightsave_path)) &&
       boost::filesystem::exists(boost::filesystem::path(matchimg_path)) &&
       boost::filesystem::exists(boost::filesystem::path(umatchimg_path)))
        return std::vector<Eigen::Vector2i>();

    //if(views[0]->getLineSize()==0 || views[1]->getLineSize()==0)
    //    return std::vector<Eigen::Vector2i>();
    //std::cout<<"debug1:"<<matching_path_<<std::endl;
    //std::cout<<views[0]->getLineSize()<<" "<<views[1]->getLineSize()<<std::endl;
    Eigen::MatrixXd lrangles,rlangles;
    Eigen::MatrixXd lrdists,rldists;
    Eigen::MatrixXd lrlen_ratio,rllen_ratio;
    lrangles=Eigen::MatrixXd::Zero(views[0]->getLineSize(),views[1]->getLineSize());
    lrdists=Eigen::MatrixXd::Zero(views[0]->getLineSize(),views[1]->getLineSize());
    lrlen_ratio=Eigen::MatrixXd::Zero(views[0]->getLineSize(),views[1]->getLineSize());
    //lrangles.resize(views[0]->getLineSize(),views[1]->getLineSize(),);
    //lrdists.resize(views[0]->getLineSize(),views[1]->getLineSize());
    //lrlen_ratio.resize(views[0]->getLineSize(),views[1]->getLineSize());

    findMatching(views,lrangles,lrdists,lrlen_ratio);
    std::vector<L3DPP::View::Ptr> rev_views;
    rev_views.push_back(views[1]);
    rev_views.push_back(views[0]);

    rlangles=Eigen::MatrixXd::Zero(rev_views[0]->getLineSize(),rev_views[1]->getLineSize());
    rldists=Eigen::MatrixXd::Zero(rev_views[0]->getLineSize(),rev_views[1]->getLineSize());
    rllen_ratio=Eigen::MatrixXd::Zero(rev_views[0]->getLineSize(),rev_views[1]->getLineSize());
   // rlangles.resize(rev_views[0]->getLineSize(),rev_views[1]->getLineSize());
   // rldists.resize(rev_views[0]->getLineSize(),rev_views[1]->getLineSize());
   // rllen_ratio.resize(rev_views[0]->getLineSize(),rev_views[1]->getLineSize());
    findMatching(rev_views,rlangles,rldists,rllen_ratio);

    double angle_s1=lrangles.maxCoeff();
    double angle_s2=rlangles.maxCoeff();
    double max_angles=std::max(angle_s1,angle_s2);

    lrangles/=max_angles;
    rlangles/=max_angles;

    double dist_s1=lrdists.maxCoeff();
    double dist_s2=rldists.maxCoeff();
    double max_dist=std::max(dist_s1,dist_s2);

    lrdists/=max_dist;
    rldists/=max_dist;

    double len_s1=lrlen_ratio.maxCoeff();
    double len_s2=rllen_ratio.maxCoeff();
    double max_len=std::max(len_s1,len_s2);

    lrlen_ratio/=max_len;
    rllen_ratio/=max_len;

    Eigen::MatrixXd angle_sm=lrangles+rlangles.transpose();
    Eigen::MatrixXd dist_sm=lrdists+rldists.transpose();
    Eigen::MatrixXd len_sm=lrlen_ratio+rllen_ratio.transpose();

    std::vector<Eigen::Vector2i> corre_idx;
    corre_idx=findMatchingOrNOPair(angle_sm,dist_sm,len_sm,views);

    findUnMatchingLine(views);
    findUnMatchingLine(rev_views);

    cv::RNG rng(0xFFFFFFFF);

    ofstream flos(leftsave_path);
    ofstream fros(rightsave_path);
    views[0]->resetImage();
    views[1]->resetImage();

    for(size_t i=0;i<corre_idx.size();i++)
    {
        int left_idx=corre_idx[i][0];
        int right_idx=corre_idx[i][1];
        Eigen::Vector4f l_line=views[0]->getLineSegment2D(left_idx);
        Eigen::Vector4f r_line=views[1]->getLineSegment2D(right_idx);
        flos<<i<<" "<<l_line[0]<<" "<<l_line[1]<<" "<<l_line[2]<<" "<<l_line[3]<<std::endl;
        fros<<i<<" "<<r_line[0]<<" "<<r_line[1]<<" "<<r_line[2]<<" "<<r_line[3]<<std::endl;

        cv::Scalar color=randomColor(rng);
        //views[1]->drawCloud(inliers_p,color);
        views[0]->drawSingleLine(left_idx,color);
        views[1]->drawSingleLine(right_idx,color);
    }
    mergeImage(views[0]->getImage(),views[1]->getImage(),matchimg_path);
    views[0]->resetImage();
    views[1]->resetImage();
    int unm_label=corre_idx.size();
    int left_us=0,right_us=0;
    for(size_t i=0;i<views[0]->getLineSize();i++)
    {
        if(views[0]->unmatch_lines_[i]==1)
        {
            cv::Scalar color=randomColor(rng);
            std::vector<Eigen::Vector3d> inliers_p;
            Eigen::Vector4f line_c=views[0]->getLineSegment2D(i);
            flos<<unm_label<<" "<<line_c[0]<<" "<<line_c[1]<<" "<<line_c[2]<<" "<<line_c[3]<<std::endl;
            unm_label++;
            left_us++;

            Eigen::Vector3d e1,e2;
            double x1,y1,x2,y2;
            x1=line_c[0],y1=line_c[1],x2=line_c[2],y2=line_c[3];
            e1<<x1,y1,1;
            e2<<x2,y2,1;
            inliers_p.push_back(e1);
            inliers_p.push_back(e2);
            views[0]->drawLabel(inliers_p,color,i);
            views[0]->drawSingleLine(i,color);
        }
    }
    for(size_t i=0;i<views[1]->getLineSize();i++)
    {
        if(views[1]->unmatch_lines_[i]==1)
        {
            cv::Scalar color=randomColor(rng);
            std::vector<Eigen::Vector3d> inliers_p;
            Eigen::Vector4f line_c=views[1]->getLineSegment2D(i);
            fros<<unm_label<<" "<<line_c[0]<<" "<<line_c[1]<<" "<<line_c[2]<<" "<<line_c[3]<<std::endl;
            unm_label++;
            right_us++;

            Eigen::Vector3d e1,e2;
            double x1,y1,x2,y2;
            x1=line_c[0],y1=line_c[1],x2=line_c[2],y2=line_c[3];
            e1<<x1,y1,1;
            e2<<x2,y2,1;
            inliers_p.push_back(e1);
            inliers_p.push_back(e2);
             views[1]->drawLabel(inliers_p,color,i);
             views[1]->drawSingleLine(i,color);
        }
    }
    mergeImage(views[0]->getImage(),views[1]->getImage(),umatchimg_path);

    std::cout<<"msize:"<<corre_idx.size()<<" "<<"lms:"<<left_us<<" "<<
              "rms:"<<right_us<<" "<<views[0]->getLineSize()<<" "<<views[1]->getLineSize()<<std::endl;

    //cv::imshow("left",result);
    //cv::imshow("right",views[1]->getImage());
  //  cv::waitKey(0);
    return corre_idx;
}
void MatchingLine::mergeImage(cv::Mat left_img,cv::Mat right_img,string save_path)
{
    cv::Mat result(left_img.rows/2,left_img.cols,left_img.type());
    cv::Mat resize1,resize2;
    cv::resize(left_img,resize1,cv::Size(left_img.cols/2,left_img.rows/2));
    cv::resize(right_img,resize2,cv::Size(right_img.cols/2,right_img.rows/2));
    cv::Mat r1 = result(cv::Rect(0, 0, result.cols/2, result.rows));
    resize1.copyTo(r1);
    cv::Mat r2 = result(cv::Rect(result.cols/2, 0, result.cols/2, result.rows));
    resize2.copyTo(r2);
    cv::imwrite(save_path,result);
}
std::vector<Eigen::Vector2i> MatchingLine::findMatchingOrNOPair(Eigen::MatrixXd& angle,Eigen::MatrixXd& dist,Eigen::MatrixXd& ratio,
                                        std::vector<L3DPP::View::Ptr>& views)
{
    std::vector<int> left_flag(angle.rows(),0);
    std::vector<int> right_flag(angle.cols(),0);
    Eigen::MatrixXd score_m=angle+dist+ratio;
    filterPairBase3DPos(views,score_m);

    std::vector<L3DPP::ScoreAndID> score_vec;
    for(size_t i=0;i<score_m.rows();i++)
    {
        for(size_t j=0;j<score_m.cols();j++)
        {
            if(score_m(i,j)>0)
            {
                L3DPP::ScoreAndID tem_v;
                tem_v.score_=score_m(i,j);
                tem_v.row_=i;
                tem_v.col_=j;
                score_vec.push_back(tem_v);
            }
        }
    }
    std::sort(score_vec.begin(),score_vec.end(),[](L3DPP::ScoreAndID& i1,L3DPP::ScoreAndID& i2)
    {
        return i2.score_<i1.score_;
    });
    std::vector<Eigen::Vector2i> corre_vec;
    for(size_t i=0;i<score_vec.size();i++)
    {

        L3DPP::ScoreAndID& item=score_vec[i];
        if(left_flag[item.row_]==1 || right_flag[item.col_]==1) continue;

        Eigen::Vector2i index;
        index<<item.row_,item.col_;
        left_flag[item.row_]=1;
        right_flag[item.col_]=1;
        views[0]->valid_flag_[item.row_]=2;//标记成匹配点
        views[1]->valid_flag_[item.col_]=2;//标记成匹配点
        corre_vec.push_back(index);
    }
    return corre_vec;
}
cv::Scalar MatchingLine::randomColor(cv::RNG& rng)
{
    int icolor = (unsigned)rng;
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}
void MatchingLine::findMatching(std::vector<L3DPP::View::Ptr>& views_ptr,Eigen::MatrixXd& angles,
                                Eigen::MatrixXd& dists,Eigen::MatrixXd& len_ratio)
{
    cv::RNG rng(0xFFFFFFFF);
    L3DPP::View::Ptr left_view=views_ptr[0],right_view=views_ptr[1];
    for(size_t i=0;i<left_view->getLineSize();i++)
    {
        int debug_label=-1;
        if(i==debug_label)
            int debug=0;
        std::vector<Eigen::Vector3d> cloud,img_points;
        left_view->unprojectLine(i,cloud,false);
        right_view->projectCloud(cloud,img_points,false);
        if((double)img_points.size()/cloud.size()<0.2 || img_points.size()<10)
            continue;


        if(i==debug_label)
            int debug=0;
        std::vector<double> line_coffes;
        std::vector<Eigen::Vector3d> inliers_p;
        fitLineRansac(img_points,line_coffes,inliers_p,2);
        //cv::Scalar color=randomColor(rng);
        //right_view->drawCloud(inliers_p,color);
        //right_view->drawLabel(inliers_p,color,i);
        if(i==debug_label)
        {
            std::vector<Eigen::Vector3d> dimg_points;
            cv::Scalar color(0,255,0);
            cv::Scalar colord(0,0,255);
          //  left_view->projectCloud(cloud,dimg_points,true);
           // left_view->drawCloud(dimg_points,colord);
           // right_view->drawCloud(inliers_p,color);
          //  right_view->drawLabel(inliers_p,color,i);
        }

        double valid_ratio=(double)inliers_p.size()/img_points.size();
        if(valid_ratio<0.4) continue;
        left_view->valid_flag_[i]=1;//标记成可共视

        Eigen::Vector3d plane_coffe=right_view->computePlaneC(line_coffes);
        Eigen::Vector3d plane_wordp=right_view->C();
        std::vector<Eigen::Vector3d> line_endnorm=left_view->unprojectLineEndP(i);
        Eigen::Vector3d camC=left_view->C();
        Eigen::Vector3d wend1=computeCrossPoint(camC,line_endnorm[0],plane_wordp,plane_coffe);
        Eigen::Vector3d wend2=computeCrossPoint(camC,line_endnorm[1],plane_wordp,plane_coffe);

        L3DPP::SegmentEnd3D end3d;
        end3d.e1_=wend1;end3d.e2_=wend2;
        left_view->setSegment3DPos(i,end3d);

        Eigen::Vector3d rend1=right_view->projectWithCheck(wend1);
        Eigen::Vector3d rend2=right_view->projectWithCheck(wend2);

        right_view->computeValidEndPoint(rend1,rend2);
       // double llen=right_view->computeLineLen(rend1,rend2);
       // cv::Scalar cend(0,0,255);
        cv::Scalar cend(0,0,255);
        if(i==debug_label)
        {
            right_view->drawPoint(rend1,cend);
            right_view->drawPoint(rend2,cend);
        }
        //right_view->drawPoint(rend1,cend);
        //right_view->drawPoint(rend2,cend);

        float x1=rend1[0],y1=rend1[1];
        float x2=rend2[0],y2=rend2[1];
        Eigen::Vector4f project_line;
        project_line<<x1,y1,x2,y2;

        for(size_t r=0;r<right_view->getLineSize();r++)
        {
            if(r==8)
                int debug=0;
            double angle=right_view->getLineAngle(r,project_line);
            if(angle<std::cos(5.0f*M_PI/180.0f)) continue;
            double dist=right_view->line2lineDistance(r,project_line);
            if(dist>9) continue;
           // double rlen=right_view->computeLineLen(r);
            double ratio=right_view->computeLineCrossLen(rend1,rend2,r);//rlen>llen? llen/rlen:rlen/llen;
            if(ratio<0.2) continue;
          //  std::vector<Eigen::Vector3d> rcloud;
         //   left_view->unprojectLine(r,rcloud);
         //   std::vector<double> rline_coffes;
         //   std::vector<Eigen::Vector3d> rinliers_p;
           // fitLineRansac(rcloud,line_coffes,inliers_p,0.1);
          //  right_view->line2lineDistanceIn3D(wend1,wend2,)

            double angle_score=1-std::sqrt(1.0-angle*angle);//-std::acos(angle);
            double dist_score=std::exp(-dist/10);
            double ratio_score=ratio;
            angles(i,r)=angle_score;//std::exp(2*angle_score);
            dists(i,r)=dist_score;
            len_ratio(i,r)=ratio_score;
        }
    }
    //if(debug==39)
    {
        //Eigen::VectorXd ddddd=angles.col(8);
       // Eigen::VectorXd ddddd1=dists.col(8);
       // Eigen::VectorXd ddddd2=len_ratio.col(8);
       // int debug1=0;
    }
    cv::Scalar color(0,255,0);
   // right_view->drawLineImage(color);
}
void MatchingLine::findUnMatchingLine(std::vector<L3DPP::View::Ptr>& views_ptr)
{
    cv::RNG rng(0xFFFFFFFF);
    L3DPP::View::Ptr left_view=views_ptr[0],right_view=views_ptr[1];
    for(size_t i=0;i<left_view->getLineSize();i++)
    {
        int debug_label=-1;
        if(i==debug_label)
            int debug=0;
        if(left_view->valid_flag_[i]==2) continue;

        std::vector<Eigen::Vector3d> cloud,img_points;
        bool valid_flag=left_view->unprojectLine(i,cloud,true);
        if(valid_flag) continue;
        right_view->projectCloud(cloud,img_points,false);
       // if((double)img_points.size()/cloud.size()<0.1 || img_points.size()<5)
         if((img_points.size()<2))
        {
            left_view->unmatch_lines_[i]=1;
            continue;
        }

        std::vector<double> line_coffes;
        std::vector<Eigen::Vector3d> inliers_p;
        fitLineRansac(img_points,line_coffes,inliers_p,1.5);
        //cv::Scalar color=randomColor(rng);
        if(i==debug_label)
        {
            cv::Scalar color(0,255,0);
            right_view->drawCloud(inliers_p,color);
            right_view->drawLabel(inliers_p,color,i);
        }

        double valid_ratio=(double)inliers_p.size()/img_points.size();
        if(valid_ratio<0.75) continue;

        L3DPP::SegmentEnd3D left_seg3d=left_view->get3DLineByID(i);
        Eigen::Vector3d wend1=left_seg3d.e1_;
        Eigen::Vector3d wend2=left_seg3d.e2_;

        Eigen::Vector3d rend1=right_view->projectWithCheck(wend1);
        Eigen::Vector3d rend2=right_view->projectWithCheck(wend2);

        right_view->computeValidEndPoint(rend1,rend2);
       // double llen=right_view->computeLineLen(rend1,rend2);
        cv::Scalar cend(0,0,255);
        if(i==debug_label)
        {
            right_view->drawPoint(rend1,cend);
            right_view->drawPoint(rend2,cend);
        }

        float x1=rend1[0],y1=rend1[1];
        float x2=rend2[0],y2=rend2[1];
        Eigen::Vector4f project_line;
        project_line<<x1,y1,x2,y2;

        bool left_match=false;
        for(size_t r=0;r<right_view->getLineSize();r++)
        {
            if(r==60)
                int debug=0;
            if(right_view->valid_flag_[r]==2) continue;
            if(right_view->unmatch_lines_[r]==1) continue;
            double angle=right_view->getLineAngle(r,project_line);

            if(std::abs(angle)<std::cos(15.0f*M_PI/180.0f)) continue;
            double dist=right_view->line2lineDistance(r,project_line);
            if(dist>30) continue;

           // double rlen=right_view->computeLineLen(r);
            double ratio=right_view->computeLineCrossLen(rend1,rend2,r);//rlen>llen? llen/rlen:rlen/llen;
            if(ratio<0.05) continue;
          //  std::vector<Eigen::Vector3d> rcloud;
         //   left_view->unprojectLine(r,rcloud);
         //   std::vector<double> rline_coffes;
         //   std::vector<Eigen::Vector3d> rinliers_p;
           // fitLineRansac(rcloud,line_coffes,inliers_p,0.1);
          //  right_view->line2lineDistanceIn3D(wend1,wend2,)

          left_match=true;
          break;
        }
        if(!left_match)
            left_view->unmatch_lines_[i]=1;
    }
    cv::Scalar color(0,255,0);
   // right_view->drawLineImage(color);
}
void MatchingLine::filterPairBase3DPos(std::vector<L3DPP::View::Ptr>& views,Eigen::MatrixXd& score)
{
    L3DPP::View::Ptr left_view=views[0];
    L3DPP::View::Ptr right_view=views[1];
    for(size_t i=0;i<score.rows();i++)
    {
        L3DPP::SegmentEnd3D seg3d=left_view->get3DLineByID(i);
        for(size_t j=0;j<score.cols();j++)
        {
            if(score(i,j)>0)
            {
                double dist=right_view->line2lineDistanceIn3D(seg3d,j);
                if(dist>1)
                    score(i,j)=0;
            }
        }
    }
}
Eigen::Isometry3d MatchingLine::readPose(string path)
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

void MatchingLine::readNames(std::vector<std::string>& v_imageName, std::string imageNameFile) {
    std::fstream file(imageNameFile);
    std::string temp;
    while (getline(file, temp)) {
        v_imageName.push_back(temp);
    }
}
void MatchingLine::readIntrinsic(string path)
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
bool MatchingLine::constructViews(std::vector<std::string>& imageNames,
                                  std::vector<std::string>& poseNames,
                                  std::vector<std::string>& depthNames,
                                  std::vector<std::string>& lineNames,
                                  std::vector<L3DPP::View::Ptr>& views)
{
    views.resize(imageNames.size());
    Eigen::Matrix3d K;
    K<<fx_,0,cx_,
       0,fy_,cy_,
       0,0,1;

    for ( int i=0; i<imageNames.size(); i++ )
    {
        cv::Mat image=cv::imread(imageNames[i]);
        iheight_=image.rows;
        iwidth_=image.cols;
        //colorImgs.push_back(image);
        cv::Mat depth=cv::imread(depthNames[i],-1);
        cv::Mat resize_depth;
        cv::resize(depth,resize_depth,cv::Size(image.cols,image.rows));
        //depthImgs.push_back(resize_depth);
        Eigen::Isometry3d T=readPose(poseNames[i]);
      //  poses.push_back(T);

        Eigen::Isometry3d T_inv=T.inverse();

        std::vector<Eigen::Vector4f> lines;
        readLines(lineNames[i],lines);
        if(lines.empty()) return false;
        L3DPP::View::Ptr view_ptr;
        view_ptr.reset(new L3DPP::View(i,lines,K,T_inv.rotation(),T_inv.translation(),
                                       image.cols,image.rows,0,imageNames[i]));
        view_ptr->loadDepthAndRawImg(resize_depth,image);
        views[i]=view_ptr;

    }
    return true;
}

void MatchingLine::loadFile(std::string matching_path,std::string intrin_path)
{
    std::vector<std::string> limgs_path;std::vector<std::string> rimgs_path;
    std::vector<std::string> llines_path;std::vector<std::string> rlines_path;
    std::vector<std::string> ldepths_path;std::vector<std::string> rdepths_path;
    std::vector<std::string> lposes_path;std::vector<std::string> rposes_path;

    readmatcing(matching_path,limgs_path,rimgs_path,
                llines_path,rlines_path,
                ldepths_path,rdepths_path,
                lposes_path,rposes_path);

    readIntrinsic(intrin_path);

    for(size_t i=0;i<limgs_path.size();i++)
    {
        //std::cout<<i<<std::endl;
        //i=32;//////////////////////////////////
        std::vector<std::string> imageNames(2);
        std::vector<std::string> lineNames(2);
        std::vector<std::string> depthNames(2);
        std::vector<std::string> poseNames(2);
        imageNames[0]=limgs_path[i];imageNames[1]=rimgs_path[i];
        poseNames[0]=lposes_path[i];poseNames[1]=rposes_path[i];

        depthNames[0]=ldepths_path[i];depthNames[1]=rdepths_path[i];
        lineNames[0]=llines_path[i];lineNames[1]=rlines_path[i];

        std::vector<L3DPP::View::Ptr> views_vec;
        bool flag=constructViews(imageNames,poseNames,depthNames,lineNames,views_vec);
        if(!flag) continue;
        cv::Mat res;
       // if(i==4)
        //    int debug=0;
        computeScoreMatrix(views_vec,res);

    }
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
   /* subdir_vec.clear();

    subdir_vec.push_back("/home/mameng/dataset/scannet/datadir_file/scene0269_00");
    subdir_vec.push_back("/home/mameng/dataset/scannet/datadir_file/scene0174_00");
    subdir_vec.push_back("/home/mameng/dataset/scannet/datadir_file/scene0146_02");
    subdir_vec.push_back("/home/mameng/dataset/scannet/datadir_file/scene0387_02");
    subdir_vec.push_back("/home/mameng/dataset/scannet/datadir_file/scene0418_00");
    subdir_vec.push_back("/home/mameng/dataset/scannet/datadir_file/scene0572_01");
    */

//subdir_vec.size()
    //omp_set_num_threads(2);
    //#pragma omp parallel for
    for(size_t i=1500;i<subdir_vec.size();i++)
    //for(size_t i=0;i<subdir_vec.size();i++)
    {
        std::cout<<i<<"-------------------------------------"<<endl;
        string sroot_dir=subdir_vec[i];
        string intrin_path=sroot_dir+"/intrinsics_color.txt";
        string matching_file=sroot_dir+"/matching.txt";
        MatchingLine neighbor(matching_file,intrin_path);
        //std::cout<<"end"<<std::endl;
    }
    return 0;
}

