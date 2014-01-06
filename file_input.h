/*
 *  file_input.h
 *  
 *
 *  Created by Theodore Marschall on 7/9/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FILE_INPUT_H
#define FILE_INPUT_H

#include <string>
#include <fstream>
#include <vector>

using namespace std;

//dat files are assumed to contain columns of data with 
class DatFileInput
{
private:
	const char* m_szFileName;
	bool m_bHeader;
	int m_nFileLength;
	int m_nColumns;
	int m_nRows;  //the number of rows in the file includes the header row if the file has one
	string* m_strArray;
	int m_nArrayLine1;
	
	int datGetRows();
	int datGetColumns();
	void datFillArray();
public:
	//constructors
	DatFileInput(const char* szFile);
	DatFileInput(const char* szFile, bool bHeader);
	DatFileInput(const char* szFile, int nRows, int nColumns);
	DatFileInput(const char* szFile, bool bHeader, int nRows, int nColumns);
	//destructor
	~DatFileInput() { delete[] m_strArray; }
	
	void datDisplayArray();
	
	void getColumn(int anColumn[], int nColumn);
	void getColumn(long alColumn[], int nColumn);
	void getColumn(double adColumn[], int nColumn);
	void getColumn(string astrColumn[], int nColumn);
	
	int getInt(int nRow, int nColumn);
	long getLong(int nRow, int nColumn);
	double getFloat(int nRow, int nColumn);
	string getString(int nRow, int nColumn);
	
	int getColumns() { return m_nColumns; }
	int getRows() { return m_nFileLength; }
};

const int LBL_CHAR = -1;
const int LBL_SHORT = -2;
const int LBL_INT = -3;
const int LBL_LONG = -4;
const int LBL_FLOAT = -5;
const int LBL_DOUBLE = -6;

class BinFileInput
{
private:
  ifstream m_FileStream;
  int m_nPosition;
  int m_nLength;
  int m_nCharSize;
  int m_nShortSize;
  int m_nIntSize;
  int m_nFloatSize;
  int m_nLongSize;
  int m_nDoubleSize;
  char *m_szBuffer;

  vector<int> m_vnHeaderTypes;
  vector<int> m_vnRowTypes;

public:
  BinFileInput(const char* szInput);
  //BinFileInput(const char* szInput, vector<int> vnHeaderTypes, vector<int> nRowTypes);
  ~BinFileInput();

  void set_position(int nPosition);
  void skip_bytes(int nBytes);
  void jump_back_bytes(int nBytes);

  char getChar();
  short getShort();
  int getInt();
  float getFloat();
  long getLong();
  double getDouble();

  int getFileSize() { return m_nLength; }
};


#endif
