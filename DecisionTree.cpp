/*===============================================================
*   Copyright (C) 2017 All rights reserved.
*   
*   FileName:DecisionTree.cpp
*   creator:yuliu1@microsoft.com
*   Time:11/24/2017
*   Description:
*   Notice: 
*   Updates:
*
================================================================*/
#include "DecisionTree.h"
#include <math.h>
#include <fstream>
#include <stdlib.h>
bool IsMeLarger(const std::pair<int,double>&p1, const std::pair<int,double>&p2)
{
  if (p1.second>p2.second)
  {
    return true;
  }
  return false;
}
std::vector<std::string> StringSplit(std::string sstr, const char* delim)
{
  std::vector<std::string> results;
  char *src = new char [sstr.length() + 1];
  strncpy(src,sstr.c_str(),sstr.length());
  src[sstr.length()] = 0;
  char *p = strtok(src,delim);
  if ( p!= NULL)
  {
    results.push_back(p);
  }
  while ( (p=strtok(NULL,delim)) != NULL )
  {
    results.push_back(p);
  }
  if (src != NULL )
  {
    delete [] src;
    src = NULL;
  }
  return results;
}

// please add your code here!
void DecisionTree::SplitTrainingSet(std::list<Sample>&trainingSet, int featsId, std::list<Sample>&subSet1,std::list<Sample>&subSet2)
{
  for (std::list<Sample>::iterator it = trainingSet.begin(); it != trainingSet.end();it++)
  {
     if((it->FeatMap).count(featsId))
     {
       subSet1.push_back(*it);
     }
     else
     {
       subSet2.push_back(*it);
     }
  }
}

int DecisionTree:: GetIdByLabelName(std::string label)
{
  if (label=="S")
  {
    return 0;
  }
  else if (label=="B")
  {
    return 1;
  }
  else if (label=="M")
  {
    return 2;
  }
  else
  {
    return 3;
  }
}

std::string DecisionTree:: GetLabelNameById(int labelId)
{
  if (labelId==0)
  {
    return "S";
  }
  else if (labelId==1)
  {
    return "B";
  }
  else if (labelId==2)
  {
    return "M";
  }
  else
  {
    return "E";
  }
}

std::vector<int> DecisionTree::CountLabels(std::list<Sample>&trainingSet)
{
  std::vector<int> results(NUM_CLASS,0);
  for (std::list<Sample>::iterator it = trainingSet.begin(); it !=trainingSet.end();it++)
  {
    results[it->labelId]+=1;
  }
  return results;
}

double DecisionTree::CalcEntropy(std::vector<int> statistics)
{
  std::vector<double> probs(NUM_CLASS,0);
  int total  = 0;
  double entropy = 0;
  for (std::vector<int>::iterator it = statistics.begin(); it
    != statistics.end();it++)
  {
    total+=*it;
  }
  if (total==0)
  {
    return 0;
  }

  for (int i = 0; i < int(statistics.size()); i++)
  {
    probs[i] = double(statistics[i])/total;
  }

  for (int i = 0; i < int(probs.size());i++)
  {
    if (probs[i]!=0)
    {
      entropy += -probs[i]*log(probs[i])/log(2.0);
    }
  }
  return entropy;
}

double DecisionTree::CalcEntropy(std::list<Sample>&trainingSet)
{
  std::vector<int> counts = CountLabels(trainingSet);
  double entropy = CalcEntropy(counts);
  return entropy; 
}

double DecisionTree::CalcInfoGain(std::list<Sample>&trainingSet, int featsId)
{
  if (trainingSet.size()==0)
  {
    return 0;
  }
  std::list<Sample> subSet1;
  std::list<Sample> subSet2;
  SplitTrainingSet(trainingSet,featsId,subSet1,subSet2);
  double Horiginal = CalcEntropy(trainingSet);
  double H1 = CalcEntropy(subSet1);
  double H2 = CalcEntropy(subSet2);
  int totalCnt = trainingSet.size();
  int cnt1 = subSet1.size();
  int cnt2 = subSet2.size();
  double p1 = (double)cnt1/totalCnt;
  double p2 = (double)cnt2/totalCnt;
  double Hcondition = p1*H1+p2*H2;
  return Horiginal - Hcondition;
}

void DecisionTree::LoadFeats(const char *filename)
{
  std::ifstream fin(filename);
  std::string line = "";
  while(std::getline(fin,line))
  {
    if(line=="")
    {
      continue;
    }
    std::vector<std::string> lines = StringSplit(line,"\t");
    allFeats.insert(std::make_pair(atoi(lines[0].c_str()),1));
  }

}
void DecisionTree:: LoadTrainingSet(const char *filename, std::list<Sample>&trainingSet)
{
  std::ifstream fin(filename);
  std::string line = "";
  while(std::getline(fin,line))
  {
    if (line=="")
    {
      continue;
    }
    std::vector<std::string> lines = StringSplit(line,"\t");
    Sample s;
    int len = int(lines.size());
    std::vector<std::string> cols = StringSplit(lines[len-1],"#");
    s.labelId = atoi(lines[len-2].c_str());
    s.grammar = cols[1];
    s.text = cols[0];
    for (int i = 0; i <len-2; i++)
    {
      s.FeatMap.insert(std::make_pair(atoi(lines[i].c_str()),1));
    }
    trainingSet.push_back(s);
  }
}

int DecisionTree::ComputeNodeLabelId(std::list<Sample>& trainingSet)
{
 std::vector<int> counts = CountLabels(trainingSet);
 int index = -1;
 int maxCount = 0;
 for (int i = 0; i < counts.size();i++)
 {
   if (counts[i]>maxCount)
   {
      maxCount = counts[i];
      index = i;
   }
 }
 return index;
}

std::vector<std::pair<int,double> > DecisionTree::CalcInfoGainsBatch(std::list<Sample>&trainingSet)
{
  std::vector<std::pair<int,double> > results;
  int i = 0;
  for (std::map<int,int>::iterator it = allFeats.begin(); it != allFeats.end(); it++)
  {
    if (visitedFeats.count(it->first))
    {
      continue;
    }
    double ig = CalcInfoGain(trainingSet,it->first);
    std::cerr <<i<<"\t"<<ig<<std::endl;
    i++;
    results.push_back(std::make_pair(it->first,ig));
  }
  return results;
}

std::pair<int,double> DecisionTree:: GetBestFeatId(std::list<Sample>&trainingSet)
{
  std::vector<std::pair<int,double> > feats2Gains = CalcInfoGainsBatch(trainingSet);
  if (feats2Gains.size()==0)
  {
    return std::make_pair(-1,0);
  }
  std::sort(feats2Gains.begin(),feats2Gains.end(),IsMeLarger);
 return feats2Gains[0]; 
}


void DecisionTree::ConstructDecisionTree(std::list<Sample>&trainingSet,Node *root)
{
    double H0 = CalcEntropy(trainingSet);
    root->labelId = ComputeNodeLabelId(trainingSet);
    if (H0==0)
    {
        NodeCount++;
        if (NodeCount%1==0)
        {
          std::cout << NodeCount << std::endl;
        }
        return;
    }
    std::pair<int,double> infopair=GetBestFeatId(trainingSet);
    // no  attributes any more
    if (infopair.first==-1)
    {
      NodeCount++;
      if (NodeCount%1==0)
      {
        std::cout << NodeCount << std::endl;
      }
      return;
    }
    if (infopair.second<SPLIT_THRESHOLD)
    {
        NodeCount++;
        if (NodeCount%1==0)
        {
          std::cout << NodeCount << std::endl;
        }
        return;
    }
    root->featsId = infopair.first;
    visitedFeats.insert(std::make_pair(root->featsId,1));
    std::list<Sample> subSet1;
    std::list<Sample> subSet2;
    SplitTrainingSet(trainingSet,infopair.first,subSet1,subSet2);
    if (subSet1.size()!=0)
    {
      Node *left = new Node;
      root->left =left;
      ConstructDecisionTree(subSet1,left);
    }
    if (subSet2.size()!=0)
    {
      Node *right = new Node;
      root->right = right;
      ConstructDecisionTree(subSet2,right);
    }
    NodeCount++;
    if (NodeCount%1==0)
    {
      std::cout << NodeCount << std::endl;
    }
    return;
}

void DecisionTree::Train(std::list<Sample> &trainingSet)
{
  root = new Node;
  ConstructDecisionTree(trainingSet,root);
}
void DecisionTree::DeleteNode(Node*root)
{
  if (root->left != NULL)
  {
    DeleteNode(root->left);
  }

  if (root->right != NULL)
  {
    DeleteNode(root->right);
  }
  delete root;

  root = NULL;
}

void DecisionTree::clear()
{
  DeleteNode(root);
}
int DecisionTree::Predict(Sample &sample)
{
  return PredictByTraverse(sample,root);
}
int DecisionTree::PredictByTraverse(Sample&sample,Node *root)
{
   bool has_key = false;
   if (root->featsId==-1)
   {
     return root->labelId;
   }
   if (sample.FeatMap.count(root->featsId))
   {
      has_key = true;
   }
   if (has_key&& root->left!=NULL)
   {
      return PredictByTraverse(sample,root->left);
   }
   if (!has_key && root->right!=NULL)
   {
     return PredictByTraverse(sample,root->right);
   }
   return  root->labelId;
}

