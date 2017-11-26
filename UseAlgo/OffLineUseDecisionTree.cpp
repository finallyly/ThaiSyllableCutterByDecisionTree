/*===============================================================
*   Copyright (C) 2017 All rights reserved.
*   
*   FileName:OffLineUseDecisionTree.cpp
*   creator:yuliu1@microsoft.com
*   Time:11/25/2017
*   Description:
*   Notice: 
*   Updates:
*
================================================================*/
 
#include "DecisionTree.h"
#include<iostream>
#include<list>
#include<fstream>
// please add your code here!
// notice StringSplit2 is the same as StringSplit,  I just get lazy here
std::vector<std::string> StringSplit2(std::string sstr, const char* delim)
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
void TestLoading(const char *trainfilename,const char* testfilename,const char* ofilename,const char* featsfilename)
{
  DecisionTree dtree;
  std::list<Sample> trainingSet;
  std::cerr << "begin loading trainingSet" << std::endl;
  dtree.LoadTrainingSet(trainfilename,trainingSet);
  std::cerr << "finish loading trainingSet" << std::endl;
  std::cerr << "begin loading featureSet" << std::endl;
  dtree.LoadFeats(featsfilename);
  std::cerr << "finish loading featureSet" << std::endl;
  std::cerr << "begin constructing trees" << std::endl;
  dtree.Train(trainingSet);
  std::cerr << "end constructing trees" << std::endl;
  std::cerr <<"begin predicting"<<std::endl;
  int linecount = 0;
  std::ifstream fin(testfilename);
  std::ofstream fout(ofilename);
  std::string line="";
  while (std::getline(fin,line))
  {
    if (line == "")
    {
      fout<<std::endl;
      continue;
    }
    linecount++;
    if (linecount%1000==0)
    {
      std::cerr <<"linecount="<<linecount<<std::endl;
    }
    std::vector<std::string> lines = StringSplit2(line,"\t");
    Sample s;
    int len = int(lines.size());
    std::vector<std::string> cols = StringSplit2(lines[len-1],"#");
    s.labelId = atoi(lines[len-2].c_str());
    s.grammar = cols[1];
    s.text = cols[0];
    for (int i = 0; i <len-2; i++)
    {
      s.FeatMap.insert(std::make_pair(atoi(lines[i].c_str()),1));
    }
    int labelId = dtree.Predict(s);
    std::string label = dtree.GetLabelNameById(labelId);
    fout<<s.text<<"\t"<<s.grammar<<"\t"<<dtree.GetLabelNameById(s.labelId)<<"\t"<<label<<std::endl;
  }
  fin.close();
  fout.close();
}
int main(int argc, char **argv)
{
  if (argc <5)
  {
    std::cerr << "no enough params"<<std::endl;
    exit(1);
  }
  TestLoading(argv[1],argv[2],argv[3],argv[4]);
  return 0;
}
