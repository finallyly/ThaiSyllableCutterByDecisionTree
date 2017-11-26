/*===============================================================
*   Copyright (C) 2017 All rights reserved.
*   
*   FileName:DecisionTree.h
*   creator:yuliu1@microsoft.com
*   Time:11/24/2017
*   Description:
*   Notice: 
*   Updates:
*
================================================================*/
 
#ifndef _DECISIONTREE_H
#define _DECISIONTREE_H
// please add your code here!
#include<iostream>
#include<string>
#include<map>
#include<list>
#include<vector>
#define NUM_CLASS 4
#define SPLIT_THRESHOLD 0.001
struct Sample
{
  int labelId;//S,B,M,E
  std::string text;// thai letter e.g. ว
  std::string grammar;//co
  std::map<int,int> FeatMap;  
};

struct Node 
{
  int featsId;
  int labelId;
  Node *left;
  Node *right;
  Node():featsId(-1),labelId(-1),left(NULL),right(NULL){};
};

class DecisionTree
{
  public:
    Node *GetRoot() {return root;}
    int Predict(Sample&sample);
    int NodeCount;
    int PredictByTraverse(Sample &sample, Node* root);
    int GetIdByLabelName(std::string label);
    int ComputeNodeLabelId(std::list<Sample>&trainingSet);
    std::pair<int,double> GetBestFeatId(std::list<Sample>&trainingSet);
    std::string GetLabelNameById(int labelId);
    void Train(std::list<Sample>& trainingSet);
    void ConstructDecisionTree(std::list<Sample>&trainingSet,Node*root);
    double CalcEntropy(std::list<Sample>&trainingSet);
    double CalcEntropy(std::vector<int> statistics);
    double CalcInfoGain(std::list<Sample>&traingSet,int featsId);
    std::vector<std::pair<int,double> > CalcInfoGainsBatch(std::list<Sample>&trainingSet);
    void SplitTrainingSet(std::list<Sample>&trainingSet, int featsId, std::list<Sample>&subSet1,std::list<Sample>&subSet2);
    std::vector<int> CountLabels(std::list<Sample>&trainingSet);
    void DeleteNode(Node*root);
    void clear();
    void LoadTrainingSet(const char *filename,std::list<Sample>& trainingSet);
    void LoadFeats(const char *filename);
    DecisionTree():root(NULL){};
    ~DecisionTree()
    {
     // clear();
    }
  private:
    Node *root;
    std::map<int,int> visitedFeats;
    std::map<int,int> allFeats;
};
#endif
