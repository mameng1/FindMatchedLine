#include<Eigen/Core>
#include<Eigen/Eigen>
#include<fstream>
#include <iostream>

using namespace std;
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
int main()
{
    string left="/home/mameng/dataset/scannet/LPI_evaviewpoint/label/scene0591_02/000900.txt";
    string right="/home/mameng/dataset/scannet/LPI_evaviewpoint/label/scene0591_02/001700.txt";
    Eigen::Isometry3d lT=readPose(left);
    Eigen::Isometry3d rT=readPose(right);
    Eigen::Isometry3d lr=lT.inverse()*rT;
    Eigen::AngleAxisd rotation_vector(lr.rotation());
    float angle=rotation_vector.angle()*180.0/M_PI;
    std::cout<<angle<<" "<<lr.translation().norm()<<endl;
}
