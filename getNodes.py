def getNodesFromCSV():
    print("getting nodes from files")
    file = open("SpeedDating_discrete02.csv");
    line = file.readline()
    nodes = line.split(',')
    #print(type(nodes))
    return nodes