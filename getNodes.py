def getNodesFromCSV():
    print("getting nodes from files")
    file = open("newData.csv");
    line = file.readline()
    nodes = line.split(',')
    #print(type(nodes))
    #nodes.pop()
    nodes[len(nodes)-1] = nodes[len(nodes)-1].strip('\n')
    return nodes