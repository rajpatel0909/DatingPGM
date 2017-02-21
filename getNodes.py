def getNodesFromCSV():
    print("hello")
    file = open("SpeedDating_discrete02.csv");
    print(type(file))
    for line in file.readlines():
        array = line.split(',')
        first_item = array[0]