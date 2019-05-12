import glob
import xlwt

path = 'Results/*.out'

files = glob.glob(path)
print("Number of files: ", len(files), "\n")

fileNumber = 0

book = xlwt.Workbook(encoding="utf-8")

sheet1 = book.add_sheet("Sheet 1")

sheet1.write(0, 0, "Epoch")
sheet1.write(0, 1, "Max val_accuracy")
sheet1.write(0, 2, "Config file name")
sheet1.write(0, 3, "Result file name")

for name in files:
    print("File: ", name)
    
    resultFile = open(name, "r")
    
    maxVal_accuracy = 0
    maxAccuracyIndice = 0
    
    for line in resultFile:
        
        if line[:36] == "I am using configuration file number":
            configFile = line[37:]
            print("line =", line)
            print("configFile =", configFile)
        
        if line[:6] == "Epoch ":
            if line[7:8].isdigit() == True:
                epochNb = line[6:8]
            else:
                epochNb = line[6:7]
                
                valAccuracyStart = line.find("val_accuracy: ") + len("val_accuracy: ")
                
            curVal_accuracy = line[valAccuracyStart:valAccuracyStart+7]
            
            if float(curVal_accuracy) > float(maxVal_accuracy):
                maxVal_accuracy = curVal_accuracy
                maxAccuracyIndice = epochNb
                
    print("Max val accuracy =", maxVal_accuracy, "at epoch:", maxAccuracyIndice, "\n")
    fileNumber = fileNumber + 1
    
    sheet1.write(fileNumber, 0, maxAccuracyIndice)
    sheet1.write(fileNumber, 1, maxVal_accuracy)
    sheet1.write(fileNumber, 2, configFile)
    sheet1.write(fileNumber, 3, name)
    
    
book.save("Results.xls")