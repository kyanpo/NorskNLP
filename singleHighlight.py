

import glob   
import sys
import fileinput
import argparse
import os




#For commandline functionality
def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN to single Highlight.')
    parser.add_argument("folderPath", help = "Location of folder.")
    #parser.add_argument("imagePath", help ="Location of image or dump.")
    #parser.add_argument("readStart", help ="Byte position processing begins at.")
    #parser.add_argument("blockSize", help ="Assumed Size of Ext block")
    return parser.parse_args()


def main(folderPath):

	savepath = 'C:\\Documents\\Speech-to-Text\\raw_data\\'
	files = glob.glob(folderPath)

	for subdir, dirs, files in os.walk(folderPath):

		fcount = 0
		for filename in files:
			filepath = subdir + os.sep + filename

			f=open(filepath, "r",  encoding="utf-8")

			savePathFN = savepath + filename
			g=open(savePathFN, "w",  encoding="utf-8")


			highlightCount = 0
			#doubleBreak = False
			for line in f:


				if((line == "@highlight\n") and (highlightCount == 0)):
					#print("hit???")
					g.write(line)
					highlightCount = highlightCount + 1
				elif((line == "@highlight\n") and (highlightCount > 0)):
					highlightCount = highlightCount + 1
				#	doubleBreak = True
				else:
					g.write(line)

	



			fcount = fcount + 1

			#print(fcount)

			#print(filename)
			f.close
			g.close


	#for file in files:

    	#savePath = 'C:\\Documents\\Speech-to-Text\\raw_data'
    	#savePath.append()
    
    
    	#g=open(savePath,  'w')
    



if __name__ == '__main__':


    arguments = parse_arguments()

    #startTime = time.time()

    main(arguments.folderPath) 

    #endTime = time.time()

    #print(endTime - startTime)
