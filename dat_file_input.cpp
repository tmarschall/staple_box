/*
 *  dat_file_input.cpp
 *  
 *
 *  Created by Theodore Marschall on 7/9/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "file_input.h"
#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib> //for exit, atoi etc

using namespace std;


//constructors
DatFileInput::DatFileInput(const char* szFile)
{
	m_szFileName = szFile;
	m_bHeader = 0;
	m_nColumns = datGetColumns();
	m_nFileLength = datGetRows();
	if (m_nFileLength * m_nColumns < 5000000)
	  m_nRows = m_nFileLength;
	else
	  m_nRows = 5000000 / m_nColumns;
	m_strArray = new string[m_nRows * m_nColumns];
	m_nArrayLine1 = 0;
	datFillArray(); 
}
DatFileInput::DatFileInput(const char* szFile, bool bHeader)
{
	m_szFileName = szFile;
	m_bHeader = bHeader;
	m_nColumns = datGetColumns();
	m_nFileLength = datGetRows();
	if (m_nFileLength * m_nColumns < 5000000)
	  m_nRows = m_nFileLength;
	else
	  m_nRows = 5000000 / m_nColumns;
	m_strArray = new string[m_nRows * m_nColumns];
	m_nArrayLine1 = 0;
	datFillArray();
}
DatFileInput::DatFileInput(const char* szFile, int nRows, int nColumns)
{
	m_szFileName = szFile;
	m_bHeader = 0;
	m_nColumns = nColumns;
	m_nFileLength = nRows;
	if (nRows * nColumns < 5000000)
	  m_nRows = nRows;
	else
	  m_nRows = 5000000 / nColumns;
	m_strArray = new string[m_nRows * m_nColumns];
	m_nArrayLine1 = 0;
	datFillArray(); 
}
DatFileInput::DatFileInput(const char* szFile, bool bHeader, int nRows, int nColumns)
{
	m_szFileName = szFile;
	m_bHeader = bHeader;
	m_nColumns = nColumns;
	m_nFileLength = nRows;
	if (nRows * nColumns < 5000000)
	  m_nRows = nRows;
	else
	  m_nRows = 5000000 / nColumns;
	m_strArray = new string[m_nRows * m_nColumns];
	m_nArrayLine1 = 0;
	datFillArray();
	//cout << "'" << szFile << "'  Loaded Successfully" << endl;
}

//find the number of rows by reading the file
int DatFileInput::datGetRows()
{
	ifstream inf(m_szFileName);
	if (!inf)
	{
		cerr << "Could not load, check file name and directory" << endl;
		exit(1);
	}
	
	int nRows = 0;
	while (inf)
	{
		nRows += 1;
		string strLine;
		getline(inf, strLine);
	}
	
	//while loop returns an extra line with no data so we must subtract 1
	return nRows - 1;
}
//find the number of columns by reading the file
//the number of columns will be based on the FIRST LINE OF THE FILE
int DatFileInput::datGetColumns()
{
	ifstream inf(m_szFileName);
	if (!inf)
	{
		cerr << "Could not load, check file name and directory" << endl;
		exit(1);
	}
	
	string strLine;
	getline(inf, strLine);
	
	int nColumns = 1;
	size_t spaceIndex = strLine.find_first_of(" ");
	while (spaceIndex != string::npos)
	{
		nColumns += 1;
		spaceIndex = strLine.find_first_of(" ", spaceIndex + 1);
	}
	
	return nColumns;
}

//fill a 1D array with strings from the file
void DatFileInput::datFillArray()
{
	ifstream inf(m_szFileName);
	
	if (!inf)
	{
	  cerr << "Could not load, check file name and directory" << endl;
	  exit(1);
	}

	for (int n = 0; n < m_nArrayLine1 + static_cast<int>(m_bHeader); n++)
	{
	  string strLine;
	  getline(inf, strLine);
	}
	
	for (int nRowIndex = 0; nRowIndex < m_nRows; nRowIndex++)
	{
	  if (m_nArrayLine1 + nRowIndex >= m_nFileLength)
	    break;

		int nColIndex = 0;
		string strLine;
		getline(inf, strLine);
		
		size_t leftSpace = -1;
		size_t rightSpace = strLine.find_first_of(" ");
		m_strArray[nRowIndex * m_nColumns + nColIndex] = strLine.substr(0, rightSpace);
		for (nColIndex = 1; nColIndex < m_nColumns; nColIndex++)
		{
			if (rightSpace == string::npos)  //should prevent seg fault if dat file has errors or varying number of columns
				break;
			leftSpace = rightSpace;
			rightSpace = strLine.find_first_of(" ", leftSpace + 1);
			m_strArray[nRowIndex * m_nColumns + nColIndex] = strLine.substr(leftSpace + 1, rightSpace - leftSpace - 1);
		}
	}
}

void DatFileInput::datDisplayArray()
{
	for (int nR = 0; nR < m_nRows; nR++)
	{
		for (int nC = 0; nC < m_nColumns; nC++)
		{
			cout << m_strArray[m_nColumns * nR + nC] << "\t";
		}
		cout << "\n";
	}
}

void DatFileInput::getColumn(int anColumn[], int nColumn)
{
  int nR = 0;
  while (nR < m_nFileLength)
    {
      //cout << "Get column, row: " << nR << endl;
      anColumn[nR] = getInt(nR, nColumn);
      nR++;
    }
}
//get column as long
void DatFileInput::getColumn(long alColumn[], int nColumn)
{
	int nR = 0;
	while (nR < m_nFileLength)
	{
		alColumn[nR] = getLong(nR, nColumn);
		nR++;
	}
}
//get column as double, float
void DatFileInput::getColumn(double adColumn[], int nColumn)
{
	int nR = 0;
	while (nR < m_nFileLength)
	{
		adColumn[nR] = getFloat(nR, nColumn);
		nR++;
	}
}
//get column as string
void DatFileInput::getColumn(string astrColumn[], int nColumn)
{
	int nR = 0;
	while (nR < m_nFileLength)
	{
		astrColumn[nR] = getString(nR, nColumn);
		nR++;
	}
}

int DatFileInput::getInt(int nRow, int nColumn)
{
  int nARow = nRow - m_nArrayLine1;
  if (nARow < 0 || nARow >= m_nRows)
  {
    m_nArrayLine1 = nRow;
    datFillArray();
    nARow = 0;
    //cout << "Getting row: " << nRow << endl;
  }
  return atoi(m_strArray[nARow * m_nColumns + nColumn].c_str());
}
long DatFileInput::getLong(int nRow, int nColumn)
{
    int nARow = nRow - m_nArrayLine1;
  if (nARow < 0 || nARow >= m_nRows)
  {
    m_nArrayLine1 = nRow;
    datFillArray();
    nARow = 0;
  }
  return atol(m_strArray[nARow * m_nColumns + nColumn].c_str());
}
double DatFileInput::getFloat(int nRow, int nColumn)
{
  int nARow = nRow - m_nArrayLine1;
  if (nARow < 0 || nARow >= m_nRows)
  {
    m_nArrayLine1 = nRow;
    datFillArray();
    nARow = 0;
    //cout << "Getting row: " << nRow << endl;
  }
  return atof(m_strArray[nARow * m_nColumns + nColumn].c_str());
}
string DatFileInput::getString(int nRow, int nColumn)
{
  int nARow = nRow - m_nArrayLine1;
  if (nARow < 0 || nARow >= m_nRows)
  {
    m_nArrayLine1 = nRow;
    datFillArray();
    nARow = 0;
  }
  return m_strArray[nARow * m_nColumns + nColumn];
}
