import glob
import xlwt

path = 'Results/*.out'
fileNumber = 0

files = glob.glob(path)
print("Number of files: ", len(files), "\n")

book = xlwt.Workbook(encoding="utf-8")

sheet1 = book.add_sheet("Sheet 1")

sheet1.write(0, 0, "At epoch")
sheet1.write(0, 1, "Max val_accuracy")
sheet1.write(0, 2, "Test_accuracy")

sheet1.write(0, 4, "Config file name")
sheet1.write(0, 5, "Result file name")

sheet1.write(0, 7, "Model")
sheet1.write(0, 8, "Optimizer")
sheet1.write(0, 9, "Loss function")
sheet1.write(0, 10, "Number of epochs")
sheet1.write(0, 11, "Batch size")
sheet1.write(0, 12, "LR")

parametersList = ["Model: ", "Optimizer: ", "Loss_function: ", "Number of epochs: ", "Batch_size: ", "Learning rate: "]
parametersUsedList = []

for name in files:
    
    resultFile = open(name, "r")
    
    maxVal_accuracy = 0
    maxAccuracyIndice = 0
    test_accuracy = 0
    
    for line in resultFile:

        for param in parametersList:
            if line[:len(param)] == param:
                parametersUsedList.append(line[len(param):-1])
        
        if line[:36] == "I am using configuration file number":
            configFile = line[37:-1]

        if line[:len("Test_accuracy: ")] == "Test_accuracy: ":
            test_accuracy = line[len("Test_accuracy: "):-1]

        if line[:6] == "Epoch ":
            if line[7:8].isdigit() == True:
                epochNb = line[6:8]
                
                valAccuracyStart = line.find("val_accuracy: ") + len("val_accuracy: ")
            else:
                epochNb = line[6:7]
                
                valAccuracyStart = line.find("val_accuracy: ") + len("val_accuracy: ")
                
            curVal_accuracy = line[valAccuracyStart:valAccuracyStart+7]
            
            if float(curVal_accuracy) > float(maxVal_accuracy):
                maxVal_accuracy = curVal_accuracy
                maxAccuracyIndice = epochNb
     
    print(name, "( configFile:", configFile, ") -> Max val accuracy =", maxVal_accuracy, "at epoch:", maxAccuracyIndice)           
    fileNumber = fileNumber + 1
    
    sheet1.write(fileNumber, 0, maxAccuracyIndice)
    sheet1.write(fileNumber, 1, maxVal_accuracy)
    sheet1.write(fileNumber, 2, test_accuracy)

    sheet1.write(fileNumber, 4, configFile)
    sheet1.write(fileNumber, 5, name)

    sheet1.write(fileNumber, 7, parametersUsedList[0])
    sheet1.write(fileNumber, 8, parametersUsedList[1])
    sheet1.write(fileNumber, 9, parametersUsedList[2])
    sheet1.write(fileNumber, 10, parametersUsedList[3])
    sheet1.write(fileNumber, 11, parametersUsedList[4])
    sheet1.write(fileNumber, 12, parametersUsedList[5])

book.save("Results.xls")