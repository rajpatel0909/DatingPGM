import xlrd


def getLinksOfNodes():
    file = xlrd.open_workbook('newLinks.xlsx')
    sheet = file.sheet_by_index(1)
    links = []
    
    for i, row in enumerate(range(sheet.nrows)):
        temp = [0,0]
        for j, col in enumerate(range(sheet.ncols)):
            #temp[j] = sheet.cell_value(i,j)
            temp[j] = str(sheet.cell_value(i,j).encode('utf-8').decode('ascii', 'ignore'))
        if temp[0] != temp[1]:
            links.append(tuple(temp))
            
        
        
    return links

def getParents():
    file = xlrd.open_workbook('newLinks.xlsx')
    sheet = file.sheet_by_index(1)
    parents = {}
    childList = []
    for i, row in enumerate(range(sheet.nrows)):
        temp = [0,0]
        for j, col in enumerate(range(sheet.ncols)):
            #temp[j] = sheet.cell_value(i,j)
            temp[j] = str(sheet.cell_value(i,j).encode('utf-8').decode('ascii', 'ignore'))
        if temp[0] != temp[1]:
            if parents.has_key(temp[1]):
                tempList = parents.get(temp[1])
                tempList.append(temp[0])
                parents[temp[1]] = tempList
            else:
                parents
                childList = []
                childList.append(temp[0])
                parents[temp[1]] = childList
                
    #print parents        
    return parents